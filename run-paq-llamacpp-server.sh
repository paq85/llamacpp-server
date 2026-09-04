#!/usr/bin/env bash
set -euo pipefail

resolve_script_dir() {
  local source_path="${BASH_SOURCE[0]}"

  while [[ -L "$source_path" ]]; do
    local source_dir

    source_dir="$(cd -P "$(dirname "$source_path")" && pwd)"
    source_path="$(readlink "$source_path")"

    if [[ "$source_path" != /* ]]; then
      source_path="$source_dir/$source_path"
    fi
  done

  cd -P "$(dirname "$source_path")" && pwd
}

ROOT="${ROOT:-$(resolve_script_dir)}"
CUDA_ROOT="$ROOT/cuda-env"
BIN="$ROOT/llama.cpp/build/bin/llama-server"
BUILD_CACHE="$ROOT/llama.cpp/build/CMakeCache.txt"
DEFAULT_CLOUDFLARED_COMPOSE_FILE="$ROOT/cloudflared.compose.yaml"
TIMEOUT_PROXY_SCRIPT="$ROOT/cloudflare-timeout-proxy.py"
DEFAULT_MODEL="$ROOT/models/Qwen3.8-27B-UD-Q5_K_XL.gguf"
DEFAULT_MMPROJ="$ROOT/models/mmproj-qwen38-27b-F16.gguf"
LOCK_FILE="$ROOT/.run-paq-llamacpp-server.lock"
# DEFAULT_CHAT_TEMPLATE_FILE="$ROOT/chat_templates/chat_template.jinja" # this can slow down the decoding quite a lot!
DEFAULT_CHAT_TEMPLATE_FILE=0
DEFAULT_CHAT_TEMPLATE_KWARGS='{"preserve_reasoning":false,"reasoning_effort":"medium"}'
ENV_FILE="$ROOT/.env"

resolve_env_file_path() {
  local env_file_path="$1"

  if [[ "$env_file_path" = /* ]]; then
    printf '%s\n' "$env_file_path"
  else
    printf '%s/%s\n' "$ROOT" "$env_file_path"
  fi
}

source_env_file() {
  local env_file_path="$1"

  [[ -f "$env_file_path" ]] || return 0

  set -a
  # shellcheck disable=SC1090
  . "$env_file_path"
  set +a
}

declare -A INHERITED_ENV=()

if ! command -v flock >/dev/null 2>&1; then
  echo "Required command not found: flock" >&2
  exit 1
fi

exec {LOCK_FD}>"$LOCK_FILE"
if ! flock -n "$LOCK_FD"; then
  echo "Another $ROOT/run-paq-llamacpp-server.sh instance is already running." >&2
  echo "This workspace is normally managed by systemd via paq-llamacpp-server.service." >&2
  echo "Stop the active service before starting another local runner or benchmark." >&2
  exit 1
fi

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_falsey() {
  case "${1:-}" in
    0|false|FALSE|no|NO|off|OFF) return 0 ;;
    *) return 1 ;;
  esac
}

is_disabled() {
  case "${1:-}" in
    ''|none|NONE|off|OFF|0|false|FALSE|no|NO) return 0 ;;
    *) return 1 ;;
  esac
}

has_docker_compose() {
  command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1
}

pick_python_bin() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo "Python is required to run the Cloudflare timeout proxy." >&2
  return 127
}

docker_compose() {
  if has_docker_compose; then
    docker compose "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi

  echo "Docker Compose is required to manage the cloudflared connector." >&2
  return 127
}

cloudflared_is_running() {
  command -v docker >/dev/null 2>&1 || return 1
  docker ps \
    --filter "name=^/${CLOUDFLARED_CONTAINER_NAME}$" \
    --filter status=running \
    --format '{{.ID}}' \
    | grep -q .
}

should_start_cloudflared() {
  local tunnel_token="${CLOUDFLARED_TUNNEL_TOKEN:-}"

  if is_falsey "$CLOUDFLARED_ENABLED"; then
    return 1
  fi

  if is_truthy "$CLOUDFLARED_ENABLED"; then
    if [[ -z "$tunnel_token" && ! -s "$CLOUDFLARED_TOKEN_FILE" ]]; then
      echo "CLOUDFLARED_ENABLED is set, but CLOUDFLARED_TUNNEL_TOKEN is empty." >&2
      exit 1
    fi

    return 0
  fi

  [[ -n "$tunnel_token" || -s "$CLOUDFLARED_TOKEN_FILE" ]]
}

ensure_cloudflared_token_file() {
  local tunnel_token="${CLOUDFLARED_TUNNEL_TOKEN:-}"

  if [[ -z "$tunnel_token" && -s "$CLOUDFLARED_TOKEN_FILE" ]]; then
    return 0
  fi

  if [[ -z "$tunnel_token" ]]; then
    echo "CLOUDFLARED_TUNNEL_TOKEN is required to start the cloudflared connector." >&2
    exit 1
  fi

  mkdir -p "$CLOUDFLARED_STATE_DIR"
  chmod 700 "$CLOUDFLARED_STATE_DIR"

  umask 077
  printf '%s' "$tunnel_token" > "$CLOUDFLARED_TOKEN_FILE"
  chmod 600 "$CLOUDFLARED_TOKEN_FILE"

  unset CLOUDFLARED_TUNNEL_TOKEN
}

start_cloudflared() {
  if ! should_start_cloudflared; then
    return 0
  fi

  if [[ ! -f "$CLOUDFLARED_COMPOSE_FILE" ]]; then
    echo "Cloudflared compose file not found: $CLOUDFLARED_COMPOSE_FILE" >&2
    exit 1
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to start the cloudflared connector." >&2
    exit 1
  fi

  ensure_cloudflared_token_file

  export CLOUDFLARED_IMAGE
  export CLOUDFLARED_CONTAINER_NAME
  export CLOUDFLARED_PROTOCOL
  export CLOUDFLARED_LOGLEVEL
  export CLOUDFLARED_TOKEN_FILE
  export ROOT

  if cloudflared_is_running; then
    CLOUDFLARED_WAS_RUNNING=1
  fi

  echo "Ensuring cloudflared connector container is running..."
  docker_compose \
    --project-name "$CLOUDFLARED_PROJECT_NAME" \
    -f "$CLOUDFLARED_COMPOSE_FILE" \
    up -d --force-recreate cloudflared

  if [[ "$PUBLIC_SCHEME" == "https" ]]; then
    echo "Cloudflare published application should target https://localhost:$PUBLIC_BIND_PORT with No TLS Verify enabled."
  else
    echo "Cloudflare published application should target http://localhost:$PUBLIC_BIND_PORT."
  fi

  if [[ "$CLOUDFLARED_WAS_RUNNING" != "1" ]]; then
    CLOUDFLARED_STARTED_BY_SCRIPT=1
  fi
}

stop_cloudflared() {
  if [[ "$CLOUDFLARED_STARTED_BY_SCRIPT" != "1" ]]; then
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi

  if cloudflared_is_running; then
    echo "Stopping cloudflared connector container..."
    docker stop "$CLOUDFLARED_CONTAINER_NAME" >/dev/null || true
  fi
}

start_timeout_proxy() {
  if [[ "$TIMEOUT_PROXY_MODE" == "off" ]]; then
    return 0
  fi

  local proxy_args=(
    "$TIMEOUT_PROXY_SCRIPT"
    --listen-host "$PUBLIC_BIND_HOST"
    --listen-port "$PUBLIC_BIND_PORT"
    --target-scheme http
    --target-host "$SERVER_BIND_HOST"
    --target-port "$SERVER_BIND_PORT"
    --mode "$TIMEOUT_PROXY_MODE"
    --heartbeat-interval "$TIMEOUT_PROXY_HEARTBEAT_SECONDS"
  )

  if [[ -n "$API_KEY_FILE" ]]; then
    proxy_args+=(--api-key-file "$API_KEY_FILE")
  fi

  if [[ -n "$SSL_KEY_FILE" ]]; then
    proxy_args+=(--tls-cert-file "$SSL_CERT_FILE" --tls-key-file "$SSL_KEY_FILE")
  fi

  if [[ -n "$COST_INPUT_PRICE" ]]; then
    proxy_args+=(--cost-input-price "$COST_INPUT_PRICE")
  fi

  if [[ -n "$COST_CACHED_PRICE" ]]; then
    proxy_args+=(--cost-cached-price "$COST_CACHED_PRICE")
  fi

  if [[ -n "$COST_OUTPUT_PRICE" ]]; then
    proxy_args+=(--cost-output-price "$COST_OUTPUT_PRICE")
  fi

  if [[ -n "$COST_LOG_DIR" ]]; then
    proxy_args+=(--cost-log-dir "$COST_LOG_DIR")
  fi

  echo "Starting Cloudflare timeout proxy ($TIMEOUT_PROXY_MODE) on $PUBLIC_BIND_HOST:$PUBLIC_BIND_PORT -> http://$SERVER_BIND_HOST:$SERVER_BIND_PORT ..."
  "$TIMEOUT_PROXY_PYTHON_BIN" "${proxy_args[@]}" &
  PROXY_PID=$!
}

stop_timeout_proxy() {
  if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill -TERM "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
}

forward_child_signal() {
  local signal="$1"

  if [[ -n "${PROXY_PID:-}" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill -"$signal" "$PROXY_PID" 2>/dev/null || true
  fi

  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -"$signal" "$SERVER_PID" 2>/dev/null || true
  fi
}

warn_known_risky_settings() {
  local warned=0

  if (( PARALLEL > 1 )) || is_truthy "$KV_UNIFIED"; then
    echo "Warning: this stack has known correctness/stability issues with multi-slot or unified-KV serving." >&2
    warned=1
  fi

  if (( PARALLEL > 1 )); then
    echo "Warning: PARALLEL=$PARALLEL enables multi-slot serving." >&2
    echo "Observed on this PAQ_LLAMACPP_SERVER/Qwen multimodal stack: repeated or stale-looking responses and instability under longer prompts." >&2
    echo "Prefer PARALLEL=1 unless you are actively benchmarking parallel slots." >&2
  fi

  if is_truthy "$KV_UNIFIED"; then
    echo "Warning: KV_UNIFIED=$KV_UNIFIED enables a unified KV buffer shared across sequences." >&2
    echo "Observed on this stack to interact badly with multi-slot serving and cache reuse." >&2
    echo "Prefer KV_UNIFIED=0 for correctness-first serving." >&2
  fi

  if (( PARALLEL > 1 )) && is_truthy "$KV_UNIFIED"; then
    echo "Warning: PARALLEL>1 + KV_UNIFIED=1 is an especially risky combination here." >&2
    echo "If you see repeated responses or crashes, try: PARALLEL=1 KV_UNIFIED=0 PROMPT_CACHE=0 CTX_CHECKPOINTS=0 CACHE_RAM_MIB=0 CACHE_IDLE_SLOTS=0" >&2
  fi

  if (( warned )); then
    echo >&2
  fi
}

cleanup() {
  local exit_code=$?
  local mode="${1:-exit}"

  trap - EXIT INT TERM

  stop_timeout_proxy

  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi

  if [[ "$mode" != "restart" ]]; then
    stop_cloudflared
  fi

  exit "$exit_code"
}

capture_inherited_environment() {
  local name
  local value

  INHERITED_ENV=()

  while IFS='=' read -r name value; do
    [[ -n "$name" ]] || continue
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    INHERITED_ENV["$name"]="$value"
  done < <(env)
}

restore_inherited_environment() {
  local name

  for name in "${!INHERITED_ENV[@]}"; do
    printf -v "$name" '%s' "${INHERITED_ENV[$name]}"
    export "$name"
  done
}

load_env_file() {
  capture_inherited_environment

  source_env_file "$ENV_FILE"

  local profile_env_file=""
  if [[ -v INHERITED_ENV[PAQ_LLAMACPP_SERVER_ENV_FILE] ]]; then
    profile_env_file="${INHERITED_ENV[PAQ_LLAMACPP_SERVER_ENV_FILE]}"
  else
    profile_env_file="${PAQ_LLAMACPP_SERVER_ENV_FILE:-}"
  fi

  if ! is_disabled "$profile_env_file"; then
    profile_env_file="$(resolve_env_file_path "$profile_env_file")"

    if [[ ! -f "$profile_env_file" ]]; then
      echo "PAQ_LLAMACPP_SERVER_ENV_FILE not found: $profile_env_file" >&2
      exit 1
    fi

    source_env_file "$profile_env_file"
  fi

  restore_inherited_environment
}

load_env_file

MODEL="${MODEL:-$DEFAULT_MODEL}"
MMPROJ="${MMPROJ:-$DEFAULT_MMPROJ}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MODEL_ALIAS="${MODEL_ALIAS:-unsloth/Qwen3.8-27B-Q5}"
CTX_SIZE="${CTX_SIZE:-250000}"
PARALLEL="${PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-1536}"
UBATCH_SIZE="${UBATCH_SIZE:-384}"
THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-16}"
THREADS_HTTP="${THREADS_HTTP:-16}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-32}"
CHECKPOINT_MIN_STEP="${CHECKPOINT_MIN_STEP:-${CHECKPOINT_EVERY_N_TOKENS:-16000}}"
CACHE_RAM_MIB="${CACHE_RAM_MIB:-16000}"
CACHE_IDLE_SLOTS="${CACHE_IDLE_SLOTS:-0}"
TIMEOUT_PROXY_MODE="${CLOUDFLARE_TIMEOUT_PROXY_MODE:-off}"
TIMEOUT_PROXY_BACKEND_HOST="${CLOUDFLARE_TIMEOUT_PROXY_BACKEND_HOST:-127.0.0.1}"
TIMEOUT_PROXY_BACKEND_PORT="${CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT:-}"
TIMEOUT_PROXY_HEARTBEAT_SECONDS="${CLOUDFLARE_TIMEOUT_PROXY_HEARTBEAT_SECONDS:-15}"
COST_INPUT_PRICE="${COST_INPUT_PRICE:-}"
COST_CACHED_PRICE="${COST_CACHED_PRICE:-}"
COST_OUTPUT_PRICE="${COST_OUTPUT_PRICE:-}"
COST_LOG_DIR="${COST_LOG_DIR:-}"
KV_UNIFIED="${KV_UNIFIED:-0}"
KV_OFFLOAD="${KV_OFFLOAD:-1}"
PROMPT_CACHE="${PROMPT_CACHE:-1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
GPU_LAYERS="${GPU_LAYERS:-all}"
FIT="${FIT:-off}"
WARMUP="${WARMUP:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPEAT_PENALTY="${REPEAT_PENALTY:-${REPETITION_PENALTY:-1.0}}"
REASONING="${REASONING:-auto}"
# Thinking-token budget. Empty/-1 = unrestricted (default). N>0 caps the 。
# block so a tool call always has output room left (prevents reasoning from
# consuming the whole budget and truncating the tool call). Keeps reasoning ON.
REASONING_BUDGET="${REASONING_BUDGET:-}"
REASONING_BUDGET_MESSAGE="${REASONING_BUDGET_MESSAGE:-}"
CHAT_TEMPLATE_KWARGS="${LLAMA_CHAT_TEMPLATE_KWARGS:-$DEFAULT_CHAT_TEMPLATE_KWARGS}"
CHAT_TEMPLATE_FILE="${LLAMA_CHAT_TEMPLATE_FILE:-$DEFAULT_CHAT_TEMPLATE_FILE}"
ENABLE_MMPROJ="${ENABLE_MMPROJ:-1}"
MMPROJ_OFFLOAD="${MMPROJ_OFFLOAD:-1}"
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-1024}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-1}"
SPEC_DRAFT_N_MIN="${SPEC_DRAFT_N_MIN:-}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-}"
SPEC_DRAFT_NGL="${SPEC_DRAFT_NGL:-auto}"
SPEC_DRAFT_CACHE_TYPE_K="${SPEC_DRAFT_CACHE_TYPE_K:-}"
SPEC_DRAFT_CACHE_TYPE_V="${SPEC_DRAFT_CACHE_TYPE_V:-}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-}"
SPEC_DRAFT_BACKEND_SAMPLING="${SPEC_DRAFT_BACKEND_SAMPLING:-}"
API_KEY_FILE="${LLAMA_SERVER_API_KEY_FILE:-}"
SSL_KEY_FILE="${LLAMA_SERVER_SSL_KEY_FILE:-}"
SSL_CERT_FILE="${LLAMA_SERVER_SSL_CERT_FILE:-}"
CLOUDFLARED_COMPOSE_FILE="${CLOUDFLARED_COMPOSE_FILE:-$DEFAULT_CLOUDFLARED_COMPOSE_FILE}"
CLOUDFLARED_ENABLED="${CLOUDFLARED_ENABLED:-auto}"
CLOUDFLARED_IMAGE="${CLOUDFLARED_IMAGE:-cloudflare/cloudflared:2026.2.0}"
CLOUDFLARED_PROTOCOL="${CLOUDFLARED_PROTOCOL:-auto}"
CLOUDFLARED_LOGLEVEL="${CLOUDFLARED_LOGLEVEL:-info}"
CLOUDFLARED_PROJECT_NAME="${CLOUDFLARED_PROJECT_NAME:-paq-llamacpp-server}"
CLOUDFLARED_CONTAINER_NAME="${CLOUDFLARED_CONTAINER_NAME:-paq-llamacpp-server-cloudflared}"
CLOUDFLARED_STATE_DIR="${CLOUDFLARED_STATE_DIR:-$ROOT/.cloudflared}"
CLOUDFLARED_TOKEN_FILE="${CLOUDFLARED_TOKEN_FILE:-$CLOUDFLARED_STATE_DIR/tunnel-token}"
CLOUDFLARED_TUNNEL_TOKEN="${CLOUDFLARED_TUNNEL_TOKEN:-}"
CLOUDFLARED_STARTED_BY_SCRIPT=0
CLOUDFLARED_WAS_RUNNING=0
TIMEOUT_PROXY_PYTHON_BIN=
PROXY_PID=
SERVER_PID=
PUBLIC_BIND_HOST="$HOST"
PUBLIC_BIND_PORT="$PORT"
SERVER_BIND_HOST="$HOST"
SERVER_BIND_PORT="$PORT"
PUBLIC_SCHEME=http
SHUTDOWN_REQUESTED=0

for path in "$BIN" "$BUILD_CACHE" "$MODEL"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

if is_truthy "$ENABLE_MMPROJ" && [[ ! -f "$MMPROJ" ]]; then
  echo "Required multimodal projector file not found: $MMPROJ" >&2
  echo "Set ENABLE_MMPROJ=0 for text-only MTP, or set MMPROJ to a compatible mmproj file." >&2
  exit 1
fi

if [[ -n "$SPEC_DRAFT_MODEL" && ! -f "$SPEC_DRAFT_MODEL" ]]; then
  echo "Required speculative draft model file not found: $SPEC_DRAFT_MODEL" >&2
  exit 1
fi

if ! grep -q '^GGML_CUDA:BOOL=ON$' "$BUILD_CACHE"; then
  echo "llama.cpp build is not CUDA-enabled (expected GGML_CUDA:BOOL=ON in $BUILD_CACHE)" >&2
  exit 1
fi

CACHED_CUDA_COMPILER="$(sed -nE 's/^CMAKE_CUDA_COMPILER:(FILEPATH|STRING)=(.*)$/\2/p' "$BUILD_CACHE" | head -n 1)"
if [[ -z "$CACHED_CUDA_COMPILER" ]]; then
  echo "Warning: could not determine CMAKE_CUDA_COMPILER from $BUILD_CACHE; continuing with the existing build output." >&2
elif [[ "$CACHED_CUDA_COMPILER" != "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "Warning: llama.cpp build cache references nvcc at $CACHED_CUDA_COMPILER, not $CUDA_ROOT/bin/nvcc." >&2
  echo "The existing llama-server binary can still run, but re-run CMake if you want future rebuilds to use the local CUDA toolchain." >&2
fi

if [[ "$FLASH_ATTN" != "off" && "$CACHE_TYPE_K" != "$CACHE_TYPE_V" ]] && ! grep -q '^GGML_CUDA_FA_ALL_QUANTS:BOOL=ON$' "$BUILD_CACHE"; then
  echo "Mixed KV cache types (CACHE_TYPE_K=$CACHE_TYPE_K, CACHE_TYPE_V=$CACHE_TYPE_V) require GGML_CUDA_FA_ALL_QUANTS=ON for CUDA Flash Attention." >&2
  echo "Use matching cache types such as f16/f16, bf16/bf16, or q8_0/q8_0, or rebuild llama.cpp with GGML_CUDA_FA_ALL_QUANTS=ON." >&2
  exit 1
fi

if [[ "$FLASH_ATTN" != "off" && -n "$SPEC_DRAFT_CACHE_TYPE_K" && -n "$SPEC_DRAFT_CACHE_TYPE_V" && "$SPEC_DRAFT_CACHE_TYPE_K" != "$SPEC_DRAFT_CACHE_TYPE_V" ]] && ! grep -q '^GGML_CUDA_FA_ALL_QUANTS:BOOL=ON$' "$BUILD_CACHE"; then
  echo "Mixed speculative KV cache types (SPEC_DRAFT_CACHE_TYPE_K=$SPEC_DRAFT_CACHE_TYPE_K, SPEC_DRAFT_CACHE_TYPE_V=$SPEC_DRAFT_CACHE_TYPE_V) require GGML_CUDA_FA_ALL_QUANTS=ON for CUDA Flash Attention." >&2
  echo "Use matching speculative cache types such as f16/f16, bf16/bf16, or q8_0/q8_0, or rebuild llama.cpp with GGML_CUDA_FA_ALL_QUANTS=ON." >&2
  exit 1
fi

if [[ -n "$API_KEY_FILE" && ! -f "$API_KEY_FILE" ]]; then
  echo "Required API key file not found: $API_KEY_FILE" >&2
  exit 1
fi

if [[ -n "$SSL_KEY_FILE" || -n "$SSL_CERT_FILE" ]]; then
  if [[ -z "$SSL_KEY_FILE" || -z "$SSL_CERT_FILE" ]]; then
    echo "Both LLAMA_SERVER_SSL_KEY_FILE and LLAMA_SERVER_SSL_CERT_FILE must be set together" >&2
    exit 1
  fi

  for path in "$SSL_KEY_FILE" "$SSL_CERT_FILE"; do
    if [[ ! -f "$path" ]]; then
      echo "Required TLS file not found: $path" >&2
      exit 1
    fi
  done
fi

if ! is_disabled "$CHAT_TEMPLATE_FILE" && [[ ! -f "$CHAT_TEMPLATE_FILE" ]]; then
  echo "Required chat template file not found: $CHAT_TEMPLATE_FILE" >&2
  exit 1
fi

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || (( PARALLEL < 1 )); then
  echo "PARALLEL must be a positive integer (got: $PARALLEL)" >&2
  exit 1
fi

case "$TIMEOUT_PROXY_MODE" in
  off|stream|optimistic) ;;
  *)
    echo "CLOUDFLARE_TIMEOUT_PROXY_MODE must be one of: off, stream, optimistic (got: $TIMEOUT_PROXY_MODE)" >&2
    exit 1
    ;;
esac

if [[ "$KV_UNIFIED" != "auto" ]] && ! is_truthy "$KV_UNIFIED" && ! is_falsey "$KV_UNIFIED"; then
  echo "KV_UNIFIED must be 'auto' or a boolean-like value (got: $KV_UNIFIED)" >&2
  exit 1
fi

if ! is_truthy "$KV_OFFLOAD" && ! is_falsey "$KV_OFFLOAD"; then
  echo "KV_OFFLOAD must be a boolean-like value (got: $KV_OFFLOAD)" >&2
  exit 1
fi

if [[ "$TIMEOUT_PROXY_MODE" != "off" ]]; then
  if [[ ! -f "$TIMEOUT_PROXY_SCRIPT" ]]; then
    echo "Required timeout proxy script not found: $TIMEOUT_PROXY_SCRIPT" >&2
    exit 1
  fi

  if [[ -z "$TIMEOUT_PROXY_BACKEND_PORT" ]]; then
    if (( PORT >= 65535 )); then
      echo "CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT must be set when PORT is 65535" >&2
      exit 1
    fi

    TIMEOUT_PROXY_BACKEND_PORT=$((PORT + 1))
  fi

  if ! [[ "$TIMEOUT_PROXY_BACKEND_PORT" =~ ^[0-9]+$ ]] || (( TIMEOUT_PROXY_BACKEND_PORT < 1 || TIMEOUT_PROXY_BACKEND_PORT > 65535 )); then
    echo "CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT must be a valid TCP port (got: $TIMEOUT_PROXY_BACKEND_PORT)" >&2
    exit 1
  fi

  if [[ "$TIMEOUT_PROXY_BACKEND_PORT" == "$PORT" ]]; then
    echo "CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT must differ from PORT when the timeout proxy is enabled" >&2
    exit 1
  fi

  if ! TIMEOUT_PROXY_PYTHON_BIN="$(pick_python_bin)"; then
    exit 1
  fi

  SERVER_BIND_HOST="$TIMEOUT_PROXY_BACKEND_HOST"
  SERVER_BIND_PORT="$TIMEOUT_PROXY_BACKEND_PORT"
fi

if [[ -n "$SSL_KEY_FILE" ]]; then
  PUBLIC_SCHEME=https
fi

warn_known_risky_settings

export PATH="$CUDA_ROOT/bin:$PATH"

# WSL exposes the Windows NVIDIA driver through /usr/lib/wsl.  Keep this
# ordering in one shared helper so rebuild utilities and the launcher cannot
# accidentally mix the host libcuda with an older Ubuntu PTX JIT library.
# Native Linux hosts simply skip the WSL-specific directories.
# shellcheck source=scripts/wsl-cuda-env.sh
source "$ROOT/scripts/wsl-cuda-env.sh"
# Put the ACTIVE server's own lib directory first in LD_LIBRARY_PATH so the
# binary picks up its matching libllama/libggml (profiles may point BIN at a
# different llama.cpp build tree, e.g. the PrismML fork for Bonsai).
BIN_DIR="${BIN%/*}"
configure_wsl_cuda_runtime "$BIN_DIR" "$CUDA_ROOT"

ARGS=(
  -m "$MODEL"
  --alias "$MODEL_ALIAS"
  --host "$SERVER_BIND_HOST"
  --port "$SERVER_BIND_PORT"
  --ctx-size "$CTX_SIZE"
  --threads "$THREADS"
  --threads-batch "$THREADS_BATCH"
  --threads-http "$THREADS_HTTP"
  --poll 0
  --poll-batch 0
  --gpu-layers "$GPU_LAYERS"
  --split-mode none
  --main-gpu 0
  --fit "$FIT"
  --flash-attn "$FLASH_ATTN"
  --parallel "$PARALLEL"
  --batch-size "$BATCH_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --ctx-checkpoints "$CTX_CHECKPOINTS"
  --checkpoint-min-step "$CHECKPOINT_MIN_STEP"
  --cache-ram "$CACHE_RAM_MIB"
  --temp "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --presence-penalty "$PRESENCE_PENALTY"
  --repeat-penalty "$REPEAT_PENALTY"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --jinja
  --reasoning "$REASONING"
)

if [[ -n "$REASONING_BUDGET" ]]; then
  ARGS+=(--reasoning-budget "$REASONING_BUDGET")
fi

if [[ -n "$REASONING_BUDGET_MESSAGE" ]]; then
  ARGS+=(--reasoning-budget-message "$REASONING_BUDGET_MESSAGE")
fi

if is_truthy "$KV_UNIFIED"; then
  ARGS+=(--kv-unified)
elif is_falsey "$KV_UNIFIED"; then
  ARGS+=(--no-kv-unified)
fi

if is_truthy "$KV_OFFLOAD"; then
  ARGS+=(--kv-offload)
else
  ARGS+=(--no-kv-offload)
fi

if ! is_disabled "$CHAT_TEMPLATE_FILE"; then
  ARGS+=(--chat-template-file "$CHAT_TEMPLATE_FILE")
fi

if is_truthy "$ENABLE_MMPROJ"; then
  ARGS+=(--mmproj "$MMPROJ")

  if is_falsey "$MMPROJ_OFFLOAD"; then
    ARGS+=(--no-mmproj-offload)
  else
    ARGS+=(--mmproj-offload)
  fi

  if [[ -n "$IMAGE_MIN_TOKENS" ]]; then
    ARGS+=(--image-min-tokens "$IMAGE_MIN_TOKENS")
  fi
else
  ARGS+=(--no-mmproj)
fi

if ! is_disabled "$SPEC_TYPE"; then
  ARGS+=(--spec-type "$SPEC_TYPE")

  if [[ -n "$SPEC_DRAFT_N_MAX" ]]; then
    ARGS+=(--spec-draft-n-max "$SPEC_DRAFT_N_MAX")
  fi

  if [[ -n "$SPEC_DRAFT_N_MIN" ]]; then
    ARGS+=(--spec-draft-n-min "$SPEC_DRAFT_N_MIN")
  fi

  if [[ -n "$SPEC_DRAFT_P_MIN" ]]; then
    ARGS+=(--spec-draft-p-min "$SPEC_DRAFT_P_MIN")
  fi

  if [[ -n "$SPEC_DRAFT_NGL" ]]; then
    ARGS+=(--spec-draft-ngl "$SPEC_DRAFT_NGL")
  fi

  if [[ -n "$SPEC_DRAFT_CACHE_TYPE_K" ]]; then
    ARGS+=(--spec-draft-type-k "$SPEC_DRAFT_CACHE_TYPE_K")
  fi

  if [[ -n "$SPEC_DRAFT_CACHE_TYPE_V" ]]; then
    ARGS+=(--spec-draft-type-v "$SPEC_DRAFT_CACHE_TYPE_V")
  fi

  if [[ -n "$SPEC_DRAFT_MODEL" ]]; then
    ARGS+=(--spec-draft-model "$SPEC_DRAFT_MODEL")
  fi

  if [[ -n "$SPEC_DRAFT_BACKEND_SAMPLING" ]]; then
    if is_falsey "$SPEC_DRAFT_BACKEND_SAMPLING"; then
      ARGS+=(--no-spec-draft-backend-sampling)
    else
      ARGS+=(--spec-draft-backend-sampling)
    fi
  fi
fi

if is_falsey "$PROMPT_CACHE"; then
  ARGS+=(--no-cache-prompt)
else
  ARGS+=(--cache-prompt)
fi

if [[ -n "$CHAT_TEMPLATE_KWARGS" ]]; then
  ARGS+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

if is_falsey "$WARMUP"; then
  ARGS+=(--no-warmup)
else
  ARGS+=(--warmup)
fi

if is_truthy "$CACHE_IDLE_SLOTS"; then
  ARGS+=(--cache-idle-slots)
else
  ARGS+=(--no-cache-idle-slots)
fi

if [[ -n "$API_KEY_FILE" ]]; then
  ARGS+=(--api-key-file "$API_KEY_FILE")
fi

if [[ -n "$SSL_KEY_FILE" && "$TIMEOUT_PROXY_MODE" == "off" ]]; then
  ARGS+=(--ssl-key-file "$SSL_KEY_FILE" --ssl-cert-file "$SSL_CERT_FILE")
fi

start_server_and_attach() {
  "$BIN" "${ARGS[@]}" &
  SERVER_PID=$!

  start_timeout_proxy
}

attach_signal_traps() {
  trap 'cleanup exit' EXIT
  trap 'SHUTDOWN_REQUESTED=1; forward_child_signal INT' INT
  trap 'SHUTDOWN_REQUESTED=1; forward_child_signal TERM' TERM
}

# Start cloudflared before the first launch
start_cloudflared

attach_signal_traps
start_server_and_attach

# Wait for server (and optional proxy) to complete — no auto-restart
set +e
if [[ -n "${PROXY_PID:-}" ]]; then
  while true; do
    wait -n "$SERVER_PID" "$PROXY_PID" 2>/dev/null
    WAIT_EXIT=$?

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      SERVER_EXIT=$WAIT_EXIT
      break
    fi

    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      PROXY_EXIT=$WAIT_EXIT
      echo "Cloudflare timeout proxy exited unexpectedly with code $PROXY_EXIT" >&2
      # Restart proxy on its own
      stop_timeout_proxy
      start_timeout_proxy
    fi
  done
else
  wait "$SERVER_PID" 2>/dev/null
  SERVER_EXIT=$?
fi

set -e
if [[ "$SHUTDOWN_REQUESTED" == "1" ]]; then
  exit 0
fi

exit "$SERVER_EXIT"
