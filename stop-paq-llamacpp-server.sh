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
ENV_FILE="$ROOT/.env"
DEFAULT_PORT=8080
DEFAULT_CLOUDFLARED_CONTAINER_NAME="paq-llamacpp-server-cloudflared"
DEFAULT_SYSTEMD_UNIT_NAME="paq-llamacpp-server.service"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

systemd_unit_is_loaded() {
  local load_state=""

  command -v systemctl >/dev/null 2>&1 || return 1
  load_state="$(systemctl show -p LoadState --value "$SYSTEMD_UNIT_NAME" 2>/dev/null || true)"
  [[ -n "$load_state" && "$load_state" != "not-found" ]]
}

systemd_unit_is_active() {
  systemd_unit_is_loaded || return 1
  systemctl is-active --quiet "$SYSTEMD_UNIT_NAME"
}

stop_systemd_service_if_active() {
  if ! systemd_unit_is_active; then
    return 0
  fi

  if is_truthy "$ALLOW_SYSTEMD_MANUAL_KILL"; then
    echo "Detected active systemd unit $SYSTEMD_UNIT_NAME, but ALLOW_SYSTEMD_MANUAL_KILL is set; continuing with direct port-based shutdown."
    return 0
  fi

  echo "Detected active systemd unit $SYSTEMD_UNIT_NAME."

  if is_truthy "$DRY_RUN"; then
    echo "DRY_RUN is set; would stop it with: sudo systemctl stop $SYSTEMD_UNIT_NAME"
    exit 0
  fi

  if (( EUID == 0 )); then
    echo "Stopping systemd service via systemctl so it does not auto-restart immediately..."
    systemctl stop "$SYSTEMD_UNIT_NAME"
    echo "Stopped systemd service $SYSTEMD_UNIT_NAME."
    exit 0
  fi

  echo "This server is systemd-managed and will auto-restart if you only kill the port listener." >&2
  echo "Use: sudo systemctl stop $SYSTEMD_UNIT_NAME" >&2
  echo "Or rerun with ALLOW_SYSTEMD_MANUAL_KILL=1 to force a direct port-based stop." >&2
  exit 3
}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

PORT="${1:-${PORT:-$DEFAULT_PORT}}"
STOP_TIMEOUT="${STOP_TIMEOUT:-20}"
DRY_RUN="${DRY_RUN:-0}"
CLOUDFLARED_CONTAINER_NAME="${CLOUDFLARED_CONTAINER_NAME:-$DEFAULT_CLOUDFLARED_CONTAINER_NAME}"
SYSTEMD_UNIT_NAME="${SYSTEMD_UNIT_NAME:-$DEFAULT_SYSTEMD_UNIT_NAME}"
ALLOW_SYSTEMD_MANUAL_KILL="${ALLOW_SYSTEMD_MANUAL_KILL:-0}"
TIMEOUT_PROXY_MODE="${CLOUDFLARE_TIMEOUT_PROXY_MODE:-off}"
TIMEOUT_PROXY_BACKEND_PORT="${CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT:-}"

case "$PORT" in
  ''|*[!0-9]*)
    echo "Invalid port: $PORT" >&2
    exit 2
    ;;
esac

if (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 2
fi

case "$TIMEOUT_PROXY_MODE" in
  off|stream|optimistic) ;;
  *)
    echo "Invalid CLOUDFLARE_TIMEOUT_PROXY_MODE: $TIMEOUT_PROXY_MODE" >&2
    exit 2
    ;;
esac

stop_systemd_service_if_active

TARGET_PORTS=("$PORT")

if [[ "$TIMEOUT_PROXY_MODE" != "off" ]]; then
  if [[ -z "$TIMEOUT_PROXY_BACKEND_PORT" ]]; then
    if (( PORT >= 65535 )); then
      echo "CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT must be set when PORT is 65535" >&2
      exit 2
    fi

    TIMEOUT_PROXY_BACKEND_PORT=$((PORT + 1))
  fi

  case "$TIMEOUT_PROXY_BACKEND_PORT" in
    ''|*[!0-9]*)
      echo "Invalid backend port: $TIMEOUT_PROXY_BACKEND_PORT" >&2
      exit 2
      ;;
  esac

  if (( TIMEOUT_PROXY_BACKEND_PORT < 1 || TIMEOUT_PROXY_BACKEND_PORT > 65535 )); then
    echo "Invalid backend port: $TIMEOUT_PROXY_BACKEND_PORT" >&2
    exit 2
  fi

  if [[ "$TIMEOUT_PROXY_BACKEND_PORT" != "$PORT" ]]; then
    TARGET_PORTS+=("$TIMEOUT_PROXY_BACKEND_PORT")
  fi
fi

if ! command -v ss >/dev/null 2>&1; then
  echo "Required command not found: ss" >&2
  exit 1
fi

PIDS=()
for target_port in "${TARGET_PORTS[@]}"; do
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && PIDS+=("$pid")
  done < <(
    ss -H -ltnp "sport = :$target_port" 2>/dev/null \
      | sed -nE 's/.*pid=([0-9]+).*/\1/p'
  )
done

if (( ${#PIDS[@]} > 0 )); then
  mapfile -t PIDS < <(printf '%s\n' "${PIDS[@]}" | sort -n -u)
fi

if (( ${#PIDS[@]} == 0 )); then
  if (( ${#TARGET_PORTS[@]} == 1 )); then
    echo "No process is listening on port $PORT."
  else
    echo "No process is listening on ports: ${TARGET_PORTS[*]}."
  fi
else
  if (( ${#TARGET_PORTS[@]} == 1 )); then
    echo "Found process(es) listening on port $PORT: ${PIDS[*]}"
  else
    echo "Found process(es) listening on ports ${TARGET_PORTS[*]}: ${PIDS[*]}"
  fi
  ps -o pid,ppid,stat,etime,comm,args -p "$(IFS=,; echo "${PIDS[*]}")" || true

  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
    echo "DRY_RUN is set; not stopping the llama.cpp process(es)."
  else
    echo "Sending SIGTERM..."
    kill -TERM "${PIDS[@]}" 2>/dev/null || true

    deadline=$((SECONDS + STOP_TIMEOUT))
    while (( SECONDS < deadline )); do
      remaining=()
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          remaining+=("$pid")
        fi
      done

      if (( ${#remaining[@]} == 0 )); then
        echo "Stopped process(es) on port $PORT."
        break
      fi

      sleep 1
    done

    remaining=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        remaining+=("$pid")
      fi
    done

    if (( ${#remaining[@]} > 0 )); then
      echo "Process(es) still running after ${STOP_TIMEOUT}s: ${remaining[*]}" >&2
      echo "Sending SIGKILL..." >&2
      kill -KILL "${remaining[@]}" 2>/dev/null || true
    fi

    final=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        final+=("$pid")
      fi
    done

    if (( ${#final[@]} > 0 )); then
      echo "Failed to stop process(es): ${final[*]}" >&2
      exit 1
    fi
  fi
fi

if command -v docker >/dev/null 2>&1; then
  mapfile -t CLOUDFLARED_IDS < <(
    docker ps -aq \
      --filter "name=^/${CLOUDFLARED_CONTAINER_NAME}$" \
      --format '{{.ID}}'
  )

  if (( ${#CLOUDFLARED_IDS[@]} == 0 )); then
    echo "cloudflared connector container is not present."
    exit 0
  fi

  echo "Found cloudflared connector container: $CLOUDFLARED_CONTAINER_NAME"
  docker ps -a \
    --filter "name=^/${CLOUDFLARED_CONTAINER_NAME}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true

  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
    echo "DRY_RUN is set; not stopping the cloudflared connector container."
    exit 0
  fi

  mapfile -t CLOUDFLARED_RUNNING_IDS < <(
    docker ps \
      --filter "name=^/${CLOUDFLARED_CONTAINER_NAME}$" \
      --filter status=running \
      --format '{{.ID}}'
  )

  if (( ${#CLOUDFLARED_RUNNING_IDS[@]} == 0 )); then
    echo "cloudflared connector container is already stopped."
    exit 0
  fi

  echo "Stopping cloudflared connector container..."
  docker stop "$CLOUDFLARED_CONTAINER_NAME" >/dev/null
  echo "Stopped cloudflared connector container."
  exit 0
fi

echo "Docker is not installed; skipping cloudflared connector shutdown."
