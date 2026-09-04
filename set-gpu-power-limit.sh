#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
POWER_PERCENT="${POWER_PERCENT:-70}"
RETRIES="${RETRIES:-30}"
SLEEP_SECS="${SLEEP_SECS:-2}"
NVIDIA_SMI="${NVIDIA_SMI:-/usr/bin/nvidia-smi}"

log() {
  printf '[gpu-power-limit] %s\n' "$*"
}

if [[ ! -x "$NVIDIA_SMI" ]]; then
  log "nvidia-smi not found at $NVIDIA_SMI"
  exit 1
fi

if [[ ! "$POWER_PERCENT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  log "POWER_PERCENT must be numeric, got '$POWER_PERCENT'"
  exit 1
fi

for ((attempt = 1; attempt <= RETRIES; attempt++)); do
  if mapfile -t power_row < <("$NVIDIA_SMI" -i "$GPU_INDEX" --query-gpu=name,power.min_limit,power.max_limit --format=csv,noheader,nounits 2>/dev/null); then
    if [[ ${#power_row[@]} -gt 0 && -n "${power_row[0]// }" ]]; then
      IFS=',' read -r gpu_name min_limit max_limit <<<"${power_row[0]}"
      gpu_name="${gpu_name## }"
      gpu_name="${gpu_name%% }"
      min_limit="${min_limit// /}"
      max_limit="${max_limit// /}"

      target_limit="$(awk -v min="$min_limit" -v max="$max_limit" -v pct="$POWER_PERCENT" 'BEGIN {
        raw = max * pct / 100.0;
        rounded = int(raw + 0.5);
        if (rounded < min) rounded = min;
        if (rounded > max) rounded = max;
        print rounded;
      }')"

      current_limit="$("$NVIDIA_SMI" -i "$GPU_INDEX" --query-gpu=power.limit --format=csv,noheader,nounits | head -n1 | tr -d ' ')"

      if [[ "$current_limit" == "$target_limit" || "$current_limit" == "$target_limit.00" ]]; then
        log "$gpu_name already capped at ${current_limit} W"
        exit 0
      fi

      log "setting $gpu_name (GPU $GPU_INDEX) power limit to ${target_limit} W (${POWER_PERCENT}% of ${max_limit} W max, clamped to ${min_limit}-${max_limit} W)"
      "$NVIDIA_SMI" -i "$GPU_INDEX" -pl "$target_limit"
      exit 0
    fi
  fi

  log "GPU $GPU_INDEX not ready yet (attempt $attempt/$RETRIES); retrying in ${SLEEP_SECS}s"
  sleep "$SLEEP_SECS"
done

log "timed out waiting for GPU $GPU_INDEX to become ready"
exit 1
