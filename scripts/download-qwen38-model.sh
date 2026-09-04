#!/usr/bin/env bash
set -euo pipefail

# download-qwen38-model.sh — download a Qwen3.8-27B (UD) GGUF + shared mmproj
# Source: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
#   q6: Qwen3.8-27B-UD-Q6_K_XL.gguf (25.3 GB) [default]
#   q4: Qwen3.8-27B-UD-Q4_K_XL.gguf (17.6 GB)
#   mmproj: mmproj-F16.gguf (928 MB) from same repo, renamed to follow the
#   local mmproj-<model>-F16.gguf convention.
#
# Usage:
#   bash scripts/download-qwen38-model.sh            # Q6_K_XL (default)
#   bash scripts/download-qwen38-model.sh q4         # Q4_K_XL

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
HF_REPO="unsloth/Qwen3.8-27B-GGUF"

QUANT="${1:-q6}"

case "$QUANT" in
  q4)
    MODEL_FILE="Qwen3.8-27B-UD-Q4_K_XL.gguf"
    EXPECTED_BYTES=17559178144
    EXPECTED_GB="17.6"
    SWITCH_NAME="qwen38-q4"
    ;;
  q6)
    MODEL_FILE="Qwen3.8-27B-UD-Q6_K_XL.gguf"
    EXPECTED_BYTES=25299061664
    EXPECTED_GB="25.3"
    SWITCH_NAME="qwen38"
    ;;
  *)
    echo "Unknown quant '$QUANT'. Valid: q4, q6 (default)." >&2
    exit 1
    ;;
esac

MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

MMPROJ_SOURCE="mmproj-F16.gguf"
MMPROJ_FILE="mmproj-qwen38-27b-F16.gguf"
MMPROJ_PATH="$MODELS_DIR/$MMPROJ_FILE"

echo "=== Qwen3.8-27B UD-${QUANT^^}_K_XL model download ==="
echo "Repo:   $HF_REPO"
echo "Model:  $MODEL_FILE (~$EXPECTED_GB GB)"
echo "mmproj: $MMPROJ_FILE (~928 MB)"
echo

if [[ ! -f "$MODEL_PATH" || ! -f "$MMPROJ_PATH" ]]; then
  mkdir -p "$MODELS_DIR"
fi

if [[ -f "$MODEL_PATH" ]]; then
  local_size="$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0)"
  local_size_gb="$(awk "BEGIN { printf \"%.1f\", $local_size / 1073741824 }")"
  echo "Model already exists at $MODEL_PATH ($local_size_gb GB)."
  echo "Delete it first if you want to re-download."
else
  download_with_wget() {
    echo "Downloading model with wget..."
    wget --continue --show-progress -O "$MODEL_PATH" \
      "https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE"
  }

  download_with_curl() {
    echo "Downloading model with curl (no progress bar; this will take a while)..."
    curl -L -C - -o "$MODEL_PATH" \
      "https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE"
  }

  if command -v wget >/dev/null 2>&1; then
    download_with_wget || {
      echo "wget failed; falling back to curl..."
      download_with_curl
    }
  elif command -v curl >/dev/null 2>&1; then
    download_with_curl
  else
    echo "No download tool found. Install wget or curl." >&2
    exit 1
  fi
fi

if [[ -f "$MODEL_PATH" ]]; then
  final_size="$(stat -c%s "$MODEL_PATH")"
  final_size_gb="$(awk "BEGIN { printf \"%.1f\", $final_size / 1073741824 }")"
  echo
  echo "Model download complete: $MODEL_PATH ($final_size_gb GB)"

  if (( final_size < EXPECTED_BYTES )); then
    echo "WARNING: file size ($final_size bytes) is smaller than expected (~$EXPECTED_GB GB)." >&2
    echo "The download may be incomplete or corrupted." >&2
    exit 1
  fi
else
  echo "Download failed — model file not found at $MODEL_PATH" >&2
  exit 1
fi

if [[ -f "$MMPROJ_PATH" ]]; then
  echo
  echo "mmproj already exists at $MMPROJ_PATH"
else
  echo
  echo "Downloading mmproj ($MMPROJ_SOURCE -> $MMPROJ_FILE)..."
  if command -v wget >/dev/null 2>&1; then
    wget --continue --show-progress -O "$MMPROJ_PATH" \
      "https://huggingface.co/$HF_REPO/resolve/main/$MMPROJ_SOURCE"
  else
    curl -L -C - -o "$MMPROJ_PATH" \
      "https://huggingface.co/$HF_REPO/resolve/main/$MMPROJ_SOURCE"
  fi
fi

if [[ -f "$MMPROJ_PATH" ]]; then
  mmproj_size="$(stat -c%s "$MMPROJ_PATH")"
  echo
  echo "mmproj download complete: $MMPROJ_PATH ($(awk "BEGIN { printf \"%.1f\", $mmproj_size / 1073741824 }") GB)"

  if (( mmproj_size < 800000000 )); then
    echo "WARNING: mmproj size ($mmproj_size bytes) is smaller than expected (~928 MB)." >&2
    exit 1
  fi
else
  echo "mmproj download failed — file not found at $MMPROJ_PATH" >&2
  exit 1
fi

echo
echo "To use this model, switch with:"
echo "  sudo bash $ROOT/scripts/switch-model.sh $SWITCH_NAME"
