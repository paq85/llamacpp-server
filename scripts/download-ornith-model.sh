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

ROOT="${ROOT:-$(cd "$(resolve_script_dir)/.." && pwd)}"
MODELS_DIR="$ROOT/models"
MODEL_FILE="Ornith-1.0-35B-UD-Q5_K_XL.gguf"
MODEL_PATH="$MODELS_DIR/$MODEL_FILE"
HF_REPO="unsloth/Ornith-1.0-35B-GGUF"
HF_URL="https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE"

echo "=== Ornith-1.0-35B Q5_K_XL model download ==="
echo "Model:  $MODEL_FILE"
echo "Target: $MODEL_PATH"
echo "Size:   ~26.5 GB"
echo

if [[ -f "$MODEL_PATH" ]]; then
  local_size="$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)"
  local_size_gb="$(awk "BEGIN { printf \"%.1f\", $local_size / 1073741824 }")"
  echo "Model already exists at $MODEL_PATH ($local_size_gb GB)."
  echo "Delete it first if you want to re-download."
  exit 0
fi

mkdir -p "$MODELS_DIR"

download_with_hf_cli() {
  echo "Downloading with huggingface-cli..."
  huggingface-cli download "$HF_REPO" "$MODEL_FILE" \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False
}

download_with_wget() {
  echo "Downloading with wget..."
  wget --continue --show-progress -O "$MODEL_PATH" "$HF_URL"
}

download_with_curl() {
  echo "Downloading with curl (no progress bar; this will take a while)..."
  curl -L -C - -o "$MODEL_PATH" "$HF_URL"
}

if command -v huggingface-cli >/dev/null 2>&1; then
  download_with_hf_cli || {
    echo "huggingface-cli failed; falling back to wget..."
    download_with_wget
  }
elif command -v wget >/dev/null 2>&1; then
  download_with_wget
elif command -v curl >/dev/null 2>&1; then
  download_with_curl
else
  echo "No download tool found. Install one of: huggingface-cli, wget, or curl." >&2
  exit 1
fi

if [[ -f "$MODEL_PATH" ]]; then
  final_size="$(stat -c%s "$MODEL_PATH")"
  final_size_gb="$(awk "BEGIN { printf \"%.1f\", $final_size / 1073741024 }")"
  echo
  echo "Download complete: $MODEL_PATH ($final_size_gb GB)"

  if (( final_size < 20000000000 )); then
    echo "WARNING: file size ($final_size bytes) is smaller than expected (~26.5 GB)." >&2
    echo "The download may be incomplete or corrupted." >&2
    exit 1
  fi

  echo
  echo "To use this model, switch with:"
  echo "  sudo bash $ROOT/scripts/switch-model.sh ornith35"
else
  echo "Download failed — model file not found at $MODEL_PATH" >&2
  exit 1
fi
