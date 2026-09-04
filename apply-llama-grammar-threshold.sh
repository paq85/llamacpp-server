#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT/llama.cpp}"
BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$LLAMA_CPP_DIR/build}"
CUDA_ROOT="${CUDA_ROOT:-$ROOT/cuda-env}"
UPSTREAM_THRESHOLD=2000
THRESHOLD="${LLAMA_GRAMMAR_MAX_REPETITION_THRESHOLD:-100000}"
PATCH=0
REBUILD=0
CHECK_ONLY=0
BUILD_TARGET="${BUILD_TARGET:-llama-server}"
JOBS="${JOBS:-8}"

# Keep rebuild-time CUDA library ordering identical to run-paq-llamacpp-server.sh.  In WSL2,
# the Windows host supplies libcuda and its PTX JIT; an older Ubuntu NVIDIA
# package must not win the dynamic-loader search.
# shellcheck source=scripts/wsl-cuda-env.sh
source "$ROOT/scripts/wsl-cuda-env.sh"

usage() {
  cat <<'EOF'
Usage: apply-llama-grammar-threshold.sh [options]

Optionally patch the embedded llama.cpp grammar repetition guard for large
agent tool-call grammars, and/or rebuild llama-server.

The MAX_REPETITION_THRESHOLD patch is OPT-IN and is NOT applied by default.
Plain rebuilds use the upstream llama.cpp source as-is (threshold 2000).

Options:
  --patch         Raise MAX_REPETITION_THRESHOLD in llama-grammar.cpp (opt-in)
  --threshold N   Threshold value used with --patch (default: 100000; implies --patch)
  --rebuild       Rebuild the llama-server target after (optionally) patching
  --no-rebuild    Do not rebuild; this is the default
  --check         Report the current source threshold value and exit
  -h, --help      Show this help

Environment:
  LLAMA_CPP_DIR                         llama.cpp checkout path
  LLAMA_CPP_BUILD_DIR                   CMake build directory
  CUDA_ROOT                             local CUDA/toolchain root
  LLAMA_GRAMMAR_MAX_REPETITION_THRESHOLD threshold value used with --patch
  BUILD_TARGET                          CMake target to rebuild (default: llama-server)
  JOBS                                  parallel build jobs (default: 8)

Typical use after updating llama.cpp (stock rebuild, no patch):
  ./apply-llama-grammar-threshold.sh --rebuild

Only if tool-call grammars hit the repetition guard:
  ./apply-llama-grammar-threshold.sh --patch --rebuild
EOF
}

while (($#)); do
  case "$1" in
    --patch)
      PATCH=1
      ;;
    --threshold)
      shift
      if (($# == 0)); then
        echo "--threshold requires a value" >&2
        exit 2
      fi
      THRESHOLD="$1"
      PATCH=1
      ;;
    --threshold=*)
      THRESHOLD="${1#--threshold=}"
      PATCH=1
      ;;
    --rebuild)
      REBUILD=1
      ;;
    --no-rebuild)
      REBUILD=0
      ;;
    --check)
      CHECK_ONLY=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! "$THRESHOLD" =~ ^[0-9]+$ ]] || (( THRESHOLD < 1 )); then
  echo "Invalid threshold: $THRESHOLD" >&2
  exit 2
fi

GRAMMAR_FILE="$LLAMA_CPP_DIR/src/llama-grammar.cpp"
if [[ ! -f "$GRAMMAR_FILE" ]]; then
  echo "Required file not found: $GRAMMAR_FILE" >&2
  exit 1
fi

current_values="$(sed -nE 's/^#define[[:space:]]+MAX_REPETITION_THRESHOLD[[:space:]]+([0-9]+).*$/\1/p' "$GRAMMAR_FILE")"
value_count="$(printf '%s\n' "$current_values" | sed '/^$/d' | wc -l)"
if [[ "$value_count" != "1" ]]; then
  echo "Could not find exactly one MAX_REPETITION_THRESHOLD #define in $GRAMMAR_FILE" >&2
  echo "Manual review needed; upstream may have changed the grammar configuration." >&2
  exit 1
fi

current_value="$current_values"
if (( CHECK_ONLY )); then
  if [[ "$current_value" == "$UPSTREAM_THRESHOLD" ]]; then
    echo "OK: MAX_REPETITION_THRESHOLD is at the upstream default ($UPSTREAM_THRESHOLD); tree is unpatched"
  else
    echo "PATCHED: MAX_REPETITION_THRESHOLD is $current_value (upstream default is $UPSTREAM_THRESHOLD)"
  fi
  exit 0
fi

if (( PATCH )); then
  if [[ "$current_value" == "$THRESHOLD" ]]; then
    echo "MAX_REPETITION_THRESHOLD is already $THRESHOLD"
  else
    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT
    awk -v threshold="$THRESHOLD" '
      BEGIN { replaced = 0 }
      /^#define[[:space:]]+MAX_REPETITION_THRESHOLD[[:space:]]+[0-9]+/ {
        print "#define MAX_REPETITION_THRESHOLD " threshold
        replaced++
        next
      }
      { print }
      END {
        if (replaced != 1) {
          exit 42
        }
      }
    ' "$GRAMMAR_FILE" > "$tmp"
    cp "$tmp" "$GRAMMAR_FILE"
    rm -f "$tmp"
    trap - EXIT
    echo "Updated MAX_REPETITION_THRESHOLD: $current_value -> $THRESHOLD"
  fi
else
  echo "Keeping stock llama.cpp source: MAX_REPETITION_THRESHOLD stays at the upstream default $UPSTREAM_THRESHOLD"
fi

if (( ! REBUILD )); then
  if (( PATCH )); then
    echo "Patch applied. Rebuild later, or rerun with --rebuild."
  else
    echo "Nothing requested (no --patch, no --rebuild); exiting."
  fi
  exit 0
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo "Build cache not found: $BUILD_DIR/CMakeCache.txt" >&2
  echo "Configure llama.cpp before using --rebuild." >&2
  exit 1
fi

cmake_home="$(sed -nE 's/^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$/\1/p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)"
if [[ -n "$cmake_home" && "$cmake_home" != "$LLAMA_CPP_DIR" ]]; then
  echo "Build cache points at a different source tree:" >&2
  echo "  CMAKE_HOME_DIRECTORY=$cmake_home" >&2
  echo "  LLAMA_CPP_DIR=$LLAMA_CPP_DIR" >&2
  echo "Refusing to rebuild the wrong checkout." >&2
  exit 1
fi

add_existing_dir() {
  local -n arr_ref="$1"
  local dir="$2"
  if [[ -d "$dir" ]]; then
    arr_ref+=("$dir")
  fi
}

join_by_colon() {
  local IFS=:
  echo "$*"
}

gcc_lib_dir=""
if [[ -d "$CUDA_ROOT/lib/gcc/x86_64-conda-linux-gnu" ]]; then
  gcc_lib_dir="$(find "$CUDA_ROOT/lib/gcc/x86_64-conda-linux-gnu" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
fi

ld_paths=()
add_existing_dir ld_paths "$(find_wsl_driver_dir)"
add_existing_dir ld_paths "/usr/lib/wsl/lib"
add_existing_dir ld_paths "$BUILD_DIR/bin"
add_existing_dir ld_paths "$CUDA_ROOT/lib"
add_existing_dir ld_paths "$CUDA_ROOT/lib64"
if [[ -n "$gcc_lib_dir" ]]; then
  add_existing_dir ld_paths "$gcc_lib_dir"
fi
add_existing_dir ld_paths "/usr/lib/x86_64-linux-gnu"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  ld_paths+=("$LD_LIBRARY_PATH")
fi
export LD_LIBRARY_PATH="$(join_by_colon "${ld_paths[@]}")"

library_paths=()
add_existing_dir library_paths "$(find_wsl_driver_dir)"
add_existing_dir library_paths "/usr/lib/wsl/lib"
add_existing_dir library_paths "$CUDA_ROOT/lib"
add_existing_dir library_paths "$CUDA_ROOT/lib64"
if [[ -n "$gcc_lib_dir" ]]; then
  add_existing_dir library_paths "$gcc_lib_dir"
fi
add_existing_dir library_paths "$CUDA_ROOT/x86_64-conda-linux-gnu/sysroot/lib"
add_existing_dir library_paths "$CUDA_ROOT/x86_64-conda-linux-gnu/sysroot/usr/lib"
add_existing_dir library_paths "$CUDA_ROOT/targets/x86_64-linux/lib/stubs"
add_existing_dir library_paths "/usr/lib/x86_64-linux-gnu"
if [[ -n "${LIBRARY_PATH:-}" ]]; then
  library_paths+=("$LIBRARY_PATH")
fi
export LIBRARY_PATH="$(join_by_colon "${library_paths[@]}")"

cmake_bin="${CMAKE_BIN:-/usr/bin/cmake}"
if [[ ! -x "$cmake_bin" ]]; then
  cmake_bin="$(command -v cmake || true)"
fi
if [[ -z "$cmake_bin" ]]; then
  echo "Required command not found: cmake" >&2
  exit 1
fi

if (( PATCH )); then
  echo "Rebuilding $BUILD_TARGET in $BUILD_DIR with MAX_REPETITION_THRESHOLD=$THRESHOLD (patched)"
else
  echo "Rebuilding $BUILD_TARGET in $BUILD_DIR with stock llama.cpp (MAX_REPETITION_THRESHOLD=$UPSTREAM_THRESHOLD, unpatched)"
fi
"$cmake_bin" --build "$BUILD_DIR" --target "$BUILD_TARGET" -j "$JOBS"
echo "Rebuild complete. Restart any running llama-server process to load the new binary/libraries."