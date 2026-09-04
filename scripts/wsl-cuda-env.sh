#!/usr/bin/env bash
# Shared WSL2 CUDA runtime path setup.
# Source this file; it intentionally does not change shell options or execute work.

find_wsl_driver_dir() {
  local drivers_root="${WSL_DRIVERS_ROOT:-/usr/lib/wsl/drivers}"
  local candidate best=""

  [[ -d "$drivers_root" ]] || return 0

  while IFS= read -r candidate; do
    [[ -f "$candidate/libnvidia-ptxjitcompiler.so.1" ]] || continue

    if [[ -z "$best" || "$candidate/libnvidia-ptxjitcompiler.so.1" -nt "$best/libnvidia-ptxjitcompiler.so.1" ]]; then
      best="$candidate"
    fi
  done < <(find "$drivers_root" -maxdepth 1 -type d -name 'nv_dispi.inf_amd64_*' -print | sort)

  [[ -n "$best" ]] && printf '%s\n' "$best"
}

configure_wsl_cuda_runtime() {
  local project_bin="${1:-}"
  local cuda_root="${2:-}"
  local driver_dir
  local -a library_paths=()
  local -a executable_paths=()

  driver_dir="$(find_wsl_driver_dir)"
  WSL_DRIVER_DIR="$driver_dir"
  export WSL_DRIVER_DIR

  if [[ -n "$driver_dir" && -d "$driver_dir" ]]; then
    library_paths+=("$driver_dir")
    executable_paths+=("$driver_dir")
  fi

  if [[ -d /usr/lib/wsl/lib ]]; then
    library_paths+=(/usr/lib/wsl/lib)
    executable_paths+=(/usr/lib/wsl/lib)
  fi

  if [[ -n "$project_bin" && -d "$project_bin" ]]; then
    library_paths+=("$project_bin")
  fi

  if [[ -n "$cuda_root" ]]; then
    [[ -d "$cuda_root/lib" ]] && library_paths+=("$cuda_root/lib")
    [[ -d "$cuda_root/lib64" ]] && library_paths+=("$cuda_root/lib64")
    [[ -d "$cuda_root/bin" ]] && executable_paths+=("$cuda_root/bin")
  fi

  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    library_paths+=("$LD_LIBRARY_PATH")
  fi
  if [[ -n "${PATH:-}" ]]; then
    executable_paths+=("$PATH")
  fi

  if (( ${#library_paths[@]} )); then
    local IFS=:
    export LD_LIBRARY_PATH="${library_paths[*]}"
  fi
  if (( ${#executable_paths[@]} )); then
    local IFS=:
    export PATH="${executable_paths[*]}"
  fi
}
