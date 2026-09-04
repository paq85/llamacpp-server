#!/usr/bin/env bash
set -euo pipefail

# download-bonsai-model.sh — fetch Prism ML's Ternary-Bonsai-27B GGUF files
# into models/.
#
# Files downloaded:
#   Ternary-Bonsai-27B-Q2_g64.gguf        (7.59 GB)  group-64 pack — mainline CPU/Metal/Vulkan only
#   Ternary-Bonsai-27B-Q2_0.gguf         (7.17 GB)  fork g128 pack (CUDA serving)
#   Ternary-Bonsai-27B-dspark-Q4_1.gguf  (1.95 GB)  fork DSpark drafter
#   Ternary-Bonsai-27B-mmproj-BF16.gguf   (931 MB)   vision projector
#
# Source: https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf
# The g128 (*-Q2_0.gguf) and DSpark drafter (*-dspark-*.gguf) packs are
# PrismML-fork-only (they do not load on mainline llama.cpp); the g64 pack is
# the mainline-compatible variant (CPU/Metal/Vulkan, no CUDA ternary kernels
# in official master as of 2026-08-10).

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
HF_REPO="prism-ml/Ternary-Bonsai-27B-gguf"

declare -A FILES=(
  ["Ternary-Bonsai-27B-Q2_g64.gguf"]="7300000000"   # 7,585,330,240 bytes (7.06 GiB) — mainline CPU-only, kept for A/B
  ["Ternary-Bonsai-27B-Q2_0.gguf"]="7000000000"     # ~7.17 GB — fork g128 (CUDA serving)
  ["Ternary-Bonsai-27B-dspark-Q4_1.gguf"]="1800000000" # ~1.95 GB — fork DSpark drafter
  ["Ternary-Bonsai-27B-mmproj-BF16.gguf"]="900000000" # 931,145,760 bytes (888 MiB)
)

echo "=== Ternary-Bonsai-27B model download ==="
echo "Repo:   $HF_REPO"
echo "Target: $MODELS_DIR"
for f in "${!FILES[@]}"; do
  echo "  - $f"
done
echo

mkdir -p "$MODELS_DIR"

download_with_hf_cli() {
  local file="$1"
  echo "Downloading $file with hf..."
  if command -v hf >/dev/null 2>&1; then
    hf download "$HF_REPO" "$file" --local-dir "$MODELS_DIR"
  else
    echo "hf not available; falling back to wget..."
    return 1
  fi
}

download_with_wget() {
  local file="$1"
  echo "Downloading $file with wget..."
  wget --continue --show-progress -O "$MODELS_DIR/$file" "https://huggingface.co/$HF_REPO/resolve/main/$file"
}

download_with_curl() {
  local file="$1"
  echo "Downloading $file with curl (no progress bar; this will take a while)..."
  curl -L -C - -o "$MODELS_DIR/$file" "https://huggingface.co/$HF_REPO/resolve/main/$file"
}

download_file() {
  local file="$1"
  local target="$MODELS_DIR/$file"

  if [[ -f "$target" ]]; then
    local size
    size="$(stat -c%s "$target" 2>/dev/null || echo 0)"
    echo "$file already exists ($(awk "BEGIN { printf \"%.2f\", $size / 1073741824 }") GB). Skipping."
    return 0
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    download_with_hf_cli "$file" || {
      echo "hf failed for $file; falling back to wget..."
      download_with_wget "$file"
    }
  elif command -v wget >/dev/null 2>&1; then
    download_with_wget "$file"
  elif command -v curl >/dev/null 2>&1; then
    download_with_curl "$file"
  else
    echo "No download tool found. Install one of: huggingface-cli, wget, or curl." >&2
    exit 1
  fi

  if [[ ! -f "$target" ]]; then
    echo "Download failed — file not found at $target" >&2
    exit 1
  fi
}

ok=1
for file in "${!FILES[@]}"; do
  download_file "$file"
done

echo
echo "=== Verifying sizes ==="
for file in "${!FILES[@]}"; do
  target="$MODELS_DIR/$file"
  size="$(stat -c%s "$target")"
  min_size="${FILES[$file]}"
  size_gb="$(awk "BEGIN { printf \"%.2f\", $size / 1073741824 }")"
  if (( size < min_size )); then
    echo "WARNING: $file is $size_gb GB ($size bytes), smaller than expected. May be incomplete/corrupt." >&2
    ok=0
  else
    echo "OK: $file ($size_gb GB)"
  fi
done

if (( ok )); then
  echo
  echo "All downloads complete. To use this model, switch with:"
  echo "  sudo bash $ROOT/scripts/switch-model.sh bonsai27"
else
  exit 1
fi
