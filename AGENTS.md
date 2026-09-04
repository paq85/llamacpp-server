# AGENTS.md — llamacpp-server

## What this project is

A **wrapper repo** around an embedded `llama.cpp` checkout that serves a **Qwen 3.6 27B multimodal GGUF** model via `llama-server` on a single NVIDIA GPU (RTX 5090). Exposes OpenAI-compatible endpoints for AI coding agents (GitHub Copilot in VS Code + CLI). Includes a custom Python proxy for Cloudflare timeout mitigation, sampling normalization, and tool-call hardening.

For full project overview: [README.md](README.md)
For Qwen-specific operational hardening: [docs/QWEN38_RELIABILITY_GUIDE.md](docs/QWEN38_RELIABILITY_GUIDE.md)
For tool-call parsing investigation: [docs/FIXME_QWEN_TOOL_PARSING.md](docs/FIXME_QWEN_TOOL_PARSING.md)
For CUDA/toolchain upgrades: [docs/UPDATING_CUDA.md](docs/UPDATING_CUDA.md)

## Quick reference

### Build

```bash
bash scripts/provision-wsl2-ubuntu.sh              # full provisioning + build
bash scripts/provision-wsl2-ubuntu.sh --no-packages  # build only
```

See `scripts/provision-wsl2-ubuntu.sh` for CMake flags and prerequisites.

### Run

```bash
./run-paq-llamacpp-server.sh                                     # foreground (flock-locked)
sudo systemctl start paq-llamacpp-server                 # systemd (production)
bash scripts/servicectl.sh status                   # check service status
```

### Stop

```bash
./stop-paq-llamacpp-server.sh
sudo systemctl stop paq-llamacpp-server
```

### Test

```bash
python3 test-context-usage.py                       # validates usage/timings in responses
python3 scripts/toolcall-stress.py --iterations 40   # tool-call reliability stress test
python3 bench-direct.py                             # direct llama-server benchmark
```

All output to `benchmarks/`.

## Architecture

```
Client (VS Code Copilot / CLI)
          │
          ▼
cloudflare-timeout-proxy.py  (port 8080)
  - SSE keepalives for Cloudflare's 100s timeout
  - Sampling clamp (temp→0.6, top_p→0.95, top_k→20)
  - Tool-result nudge after tool-call turns
  - Optional request capture for debugging
          │
          ▼
llama-server (backend port 8081)
  llama.cpp/build/bin/llama-server
  model: Qwen3.8-27B-UD-Q5_K_XL.gguf
  mmproj: mmproj-qwen38-27b-F16.gguf
```

## Configuration

Two-layer env system managed by `run-paq-llamacpp-server.sh`:

1. **`.env`** — base defaults (not committed; see `.env.example`)
2. **`PAQ_LLAMACPP_SERVER_ENV_FILE`** — profile overlay (currently `dot.env.qwen38-27b-q5kxl-200k-32gb`)

The systemd service sets `PAQ_LLAMACPP_SERVER_ENV_FILE=dot.env.qwen38-27b-q5kxl-200k-32gb`. All env vars map to `llama-server` CLI flags in `run-paq-llamacpp-server.sh`.

Key configs:
- `PARALLEL=1` — **intentional**: single concurrent request to avoid KV cache conflicts
- `KV_UNIFIED=0` + `KV_OFFLOAD=1` — split KV cache across GPU + RAM
- `REASONING=auto` — `<think>` blocks can consume output budget; use `REASONING_BUDGET` if tool calls get truncated
- `SPEC_TYPE=draft-mtp` — speculative decoding for faster generation
- `MODEL_ALIAS=<descriptive>,PAQ_LLAMACPP_SERVER` — **every profile must include `,PAQ_LLAMACPP_SERVER` as a secondary alias** so the model is always reachable under the name `PAQ_LLAMACPP_SERVER` regardless of which profile is active

## Critical conventions & pitfalls

### WSL2 CUDA

- **Never install Linux NVIDIA drivers in WSL2** — the driver comes from the Windows host
- CUDA PTX JIT library ordering is handled by `scripts/wsl-cuda-env.sh` (sourced by `run-paq-llamacpp-server.sh`)
- `cuda-env` is a **symlink** to `/usr/local/cuda-*` (gitignored)
- **Upgrading the CUDA toolkit:** follow [`docs/UPDATING_CUDA.md`](docs/UPDATING_CUDA.md). Always `rm -rf llama.cpp/build` and pass `-DCUDAToolkit_ROOT=<new>` when bumping CUDA — in-place reconfigure keeps stale `CUDAToolkit_*` cache entries (mixed toolchain / `GGML_CUDA=OFF`). A freshly built binary segfaults on `--help` from a bare shell unless you `source scripts/wsl-cuda-env.sh` + `configure_wsl_cuda_runtime` first (benign).

### Models

- **Do NOT load GGUFs from Windows-mounted drives** (`/mnt/n/...`) — causes `munmap_chunk(): invalid pointer` crash
- Use local Linux copies in `models/`
- Model files and `api-keys.txt` are gitignored

### Script conventions

- All Bash scripts use `set -euo pipefail`
- `ROOT` is resolved from script location (no hardcoded paths)
- `run-paq-llamacpp-server.sh` is `flock`-locked — only one instance at a time
- Python scripts expect `api-keys.txt` in repo root for auth
- Use the **workspace Python environment**, not `/bin/python3`, for package visibility

### llama.cpp

- `llama.cpp/` is an **embedded upstream checkout** — not forked, not vendored
- Track changes in its own git repo; the wrapper repo gitignores it
- Rebuilds use the **stock upstream** llama.cpp source — no source patches by default. The MAX_REPETITION_THRESHOLD grammar patch is **opt-in** (`./apply-llama-grammar-threshold.sh --patch --rebuild`); only use it if tool-call grammars hit the repetition guard

### Proxy hardening

- The proxy **clamps** client sampling params (Copilot sends temp=1/top_p=1 → clamped to 0.6/0.95)
- `LLAMA_PROXY_NUDGE_AFTER_TOOL_RESULT=on` — appends a nudge after tool results to prevent the model from stopping prematurely
- See `docs/QWEN38_RELIABILITY_GUIDE.md` for all active tweak values

### Systemd

- Service name: `paq-llamacpp-server.service` (historical; doesn't match repo name)
- Install: `sudo bash install-systemd-service.sh`
- Service sets `PAQ_LLAMACPP_SERVER_ENV_FILE=dot.env.qwen38-27b-q5kxl-200k-32gb`
- GPU power limit service: `nvidia-power-limit.service` (optional, via `set-gpu-power-limit.sh`)

## Key files

| Path | Role |
|---|---|
| `run-paq-llamacpp-server.sh` | Main launcher (~830 lines) — env loading, arg building, process lifecycle |
| `cloudflare-timeout-proxy.py` | HTTP SSE proxy with sampling clamp, keepalives, tool nudging |
| `stop-paq-llamacpp-server.sh` | Graceful shutdown (systemd-aware) |
| `scripts/provision-wsl2-ubuntu.sh` | Full WSL2 provisioning: packages, CUDA setup, CMake build |
| `scripts/servicectl.sh` | Systemd service management wrapper |
| `scripts/wsl-cuda-env.sh` | WSL2 CUDA library path ordering (sourced, not executed) |
| `scripts/toolcall-stress.py` | Tool-call reliability stress/reproduction harness |
| `install-systemd-service.sh` | Renders systemd unit template with `@ROOT@` variables |
| `apply-llama-grammar-threshold.sh` | Opt-in `llama-grammar.cpp` threshold patch + rebuild helper (no patch by default) |
| `dot.env.qwen38-27b-q5kxl-200k-32gb` | Current active runtime profile |
| `paq-llamacpp-server-base-meta/` | Model metadata only (config.json, tokenizer_config.json, processor_config.json) |
