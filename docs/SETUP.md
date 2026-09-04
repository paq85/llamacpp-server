# Setup Guide

Step-by-step guide to getting `llamacpp-server` running on your machine, from bare WSL2 to a fully tunneled Cloudflare endpoint usable by VS Code GitHub Copilot.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Provision WSL2 + Build llama.cpp](#step-1-provision-wsl2--build-llamacpp)
3. [Step 2: Download Model Files](#step-2-download-model-files)
4. [Step 3: Configure `.env`](#step-3-configure-env)
5. [Step 4: Start the Server](#step-4-start-the-server)
6. [Step 5: Verify Locally](#step-5-verify-locally)
7. [Step 6: Set Up Cloudflare Tunnel](#step-6-set-up-cloudflare-tunnel)
8. [Step 7: Configure the Timeout Proxy](#step-7-configure-the-timeout-proxy)
9. [Step 8: Install as a Boot Service](#step-8-install-as-a-boot-service)
10. [Step 9: Connect VS Code Copilot](#step-9-connect-vs-code-copilot)
11. [Switching Model Profiles](#switching-model-profiles)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows 11 + WSL2 | Ubuntu 22.04/24.04 recommended |
| NVIDIA GPU | RTX 5090 (32 GB) or any 16 GB+ card |
| Windows NVIDIA driver | Latest Game Ready or Studio driver; WSL2 picks it up automatically |
| Docker Desktop | For the `cloudflared` tunnel connector container |
| Cloudflare account | Free tier works; you need a domain on Cloudflare |
| ~30 GB disk | Model files + llama.cpp build |

> **Do NOT install a Linux NVIDIA driver inside WSL2.** The Windows driver is shared via the WSL2 CUDA bridge.

---

## Step 1: Provision WSL2 + Build llama.cpp

Open a terminal **inside your Ubuntu WSL2 distribution** (not PowerShell).

```bash
# Clone or cd into the repo
cd ~/llamacpp-server

# Run the provisioning script (installs deps, builds llama.cpp with CUDA)
bash scripts/provision-wsl2-ubuntu.sh
```

This takes 10–30 minutes depending on your hardware. It will:

- Install `cmake`, `gcc`, `g++`, `python3`, `git`
- Verify the WSL2 CUDA bridge (`nvidia-smi` works)
- Create the `cuda-env` symlink to your CUDA toolkit
- Configure and build `llama-server` with CUDA + Flash Attention

**Verify the build:**

```bash
llama.cpp/build/bin/llama-server --version
```

You should see a version string and CUDA info. If you get `munmap_chunk(): invalid pointer`, rerun:

```bash
bash scripts/provision-wsl2-ubuntu.sh --no-build
```

> **Different GPU?** Pass `--arch` to override the compute architecture:
> ```bash
> bash scripts/provision-wsl2-ubuntu.sh --arch 86   # RTX 4090
> bash scripts/provision-wsl2-ubuntu.sh --arch 89   # RTX 5090 (default)
> ```

---

## Step 2: Download Model Files

Download from [Hugging Face — unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF):

| File | Size (approx) | For |
|------|--------------|-----|
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` | ~20 GB | 32 GB profile |
| `Qwen3.8-27B-UD-Q3_K_XL.gguf` | ~14 GB | 16 GB profile |
| `mmproj-qwen38-27b-F16.gguf` | ~1.5 GB | Vision (both profiles) |

Place them in the `models/` directory:

```bash
mkdir -p models
# Download via huggingface-cli, wget, or browser
# Example:
huggingface-cli download unsloth/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-UD-Q5_K_XL.gguf" "mmproj-qwen38-27b-F16.gguf" \
  --local-dir models/
```

> **Important:** Do NOT load GGUF files from Windows-mounted paths (`/mnt/c/...`). Copy them into the Linux filesystem (`~/llamacpp-server/models/`).

---

## Step 3: Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and verify these key values match your setup:

```env
# Model (32 GB profile shown; adjust for 16 GB)
MODEL=models/Qwen3.8-27B-UD-Q5_K_XL.gguf
MMPROJ=models/mmproj-qwen38-27b-F16.gguf
MODEL_ALIAS=Qwen3.8-27B,PAQ_LLAMACPP_SERVER

# Context and batching (32 GB profile)
CTX_SIZE=200000
BATCH_SIZE=2048
UBATCH_SIZE=512

# Cloudflare Tunnel (fill in Step 6)
CLOUDFLARED_TUNNEL_TOKEN=
CLOUDFLARED_ENABLED=auto

# Timeout proxy (fill in Step 7)
CLOUDFLARE_TIMEOUT_PROXY_MODE=stream
```

**For the 16 GB profile**, change:

```env
MODEL=models/Qwen3.8-27B-UD-Q3_K_XL.gguf
CTX_SIZE=100000
BATCH_SIZE=1024
UBATCH_SIZE=256
```

Or use the profile overlay file instead of editing `.env` directly:

```bash
PAQ_LLAMACPP_SERVER_ENV_FILE=dot.env.qwen38-27b-q3kxl-100k-16gb ./run-paq-llamacpp-server.sh
```

---

## Step 4: Start the Server

```bash
./run-paq-llamacpp-server.sh
```

Expected output (abbreviated):

```
[run-paq-llamacpp-server] Loading .env
[run-paq-llamacpp-server] Model: models/Qwen3.8-27B-UD-Q5_K_XL.gguf
[run-paq-llamacpp-server] Starting llama-server on 0.0.0.0:8081
[run-paq-llamacpp-server] Starting timeout proxy on 0.0.0.0:8080
[run-paq-llamacpp-server] Starting cloudflared connector
[run-paq-llamacpp-server] All services started
```

**Stop the server:**

```bash
./stop-paq-llamacpp-server.sh
```

---

## Step 5: Verify Locally

Test the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PAQ_LLAMACPP_SERVER",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }'
```

You should get a JSON response with a `choices` array. If you get a connection refused, check that the server is running and the port is correct.

---

## Step 6: Set Up Cloudflare Tunnel

This step exposes your local server to the internet through Cloudflare's edge network, giving you a stable HTTPS hostname.

### 6a. Create a tunnel in the Cloudflare dashboard

1. Go to [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) → **Networks** → **Tunnels**
2. Click **Create a tunnel**
3. Choose **Cloudflared** as the connector type
4. Name it (e.g., `llamacpp`)
5. Copy the **tunnel token** (a long alphanumeric string)

### 6b. Configure the public hostname

1. In the tunnel you just created, go to the **Public Hostname** tab
2. Click **Add a public hostname**
3. Set:
   - **Hostname**: `llm.yourdomain.com` (or whatever subdomain you want)
   - **Service**: depends on your setup:

| Environment | Service value |
|---|---|
| WSL2 + Docker Desktop | `http://host.docker.internal:8080` |
| Native Linux | `http://127.0.0.1:8080` |

4. Save

> **Why `host.docker.internal`?** On WSL2 with Docker Desktop, the `cloudflared` container runs in the Docker VM, not in your WSL2 distro. `host.docker.internal` resolves to the WSL2 host where your proxy actually listens.

### 6c. Put the token in `.env`

```env
CLOUDFLARED_TUNNEL_TOKEN=eyJhbGciOi...your-token-here...
CLOUDFLARED_ENABLED=auto
```

The `run-paq-llamacpp-server.sh` launcher will automatically start/stop the `cloudflared` Docker container when you start/stop the server.

### 6d. Verify the tunnel

After starting the server, test from outside:

```bash
curl https://llm.yourdomain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"PAQ_LLAMACPP_SERVER","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

If you get a `524` error on long prompts, proceed to Step 7.

---

## Step 7: Configure the Timeout Proxy

The timeout proxy solves Cloudflare's ~120-second read timeout. Without it, any request where the model takes >120s to produce the first byte will be killed by Cloudflare with a `524` error.

### What it does

- Sits between Cloudflare and `llama-server`
- Emits SSE keepalive pings every 15 seconds while waiting for the model's first byte
- Clamps sampling parameters (temperature, top_p, top_k) for stable tool-call JSON
- Optionally injects a nudge after tool-result turns

### Configuration

In `.env`:

```env
# Enable the proxy (stream mode for Copilot, which uses stream=true)
CLOUDFLARE_TIMEOUT_PROXY_MODE=stream

# Sampling clamps (recommended for Qwen 3.8 tool-calling)
LLAMA_PROXY_CLAMP_TEMPERATURE=0.6
LLAMA_PROXY_CLAMP_TOP_P=0.95
LLAMA_PROXY_SET_TOP_K=20
LLAMA_PROXY_MAX_TOKENS_CTX_PCT=50

# Optional: nudge after tool results (prevents premature stopping)
LLAMA_PROXY_NUDGE_AFTER_TOOL_RESULT=on
```

### Port layout

| Port | What listens | Who connects |
|------|-------------|-------------|
| `8080` | Timeout proxy (public) | Cloudflare tunnel, clients |
| `8081` | `llama-server` (private) | Timeout proxy only |

With `CLOUDFLARE_TIMEOUT_PROXY_MODE=off`, `llama-server` listens directly on `8080` and there is no proxy.

### Mode selection

| Mode | Use when |
|------|----------|
| `off` | Local-only use, or first-byte is always < 120s |
| `stream` | Client sends `"stream": true` (VS Code Copilot does this) |
| `optimistic` | Client needs non-stream JSON and first-byte can exceed 120s |

---

## Step 8: Install as a Boot Service

To have the server start automatically on boot and restart on crash:

```bash
sudo ./install-systemd-service.sh
```

This installs `paq-llamacpp-server.service` which:

- Runs `run-paq-llamacpp-server.sh` as your user
- Restarts on failure (`Restart=always`)
- Starts after network is available

**Manage the service:**

```bash
sudo systemctl status paq-llamacpp-server.service
sudo systemctl restart paq-llamacpp-server.service
sudo systemctl stop paq-llamacpp-server.service
sudo journalctl -u paq-llamacpp-server.service -f   # live logs
```

**Optional: GPU power cap** (reduces heat/fan noise):

```bash
sudo ./install-systemd-service.sh --unit nvidia-power-limit.service
```

---

## Step 9: Connect VS Code Copilot

### In VS Code

1. Open **Settings** → search for `chat` → find **Chat: Manage Language Models**
2. Add a new model provider:
   - **Base URL**: `https://llm.yourdomain.com/v1`
   - **API Key**: any non-empty string (or your key from `api-keys.txt`)
   - **Model name**: `PAQ_LLAMACPP_SERVER` (or whatever `MODEL_ALIAS` is set to)
3. Save and reload VS Code

### In Copilot CLI

```bash
export OPENAI_API_BASE=https://llm.yourdomain.com/v1
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=PAQ_LLAMACPP_SERVER
```

### Verify

Ask Copilot a simple question. If you get a response, you're done. If you get `invalid_api_key`, re-enter the key in VS Code's secure prompt (not plain text in settings).

---

## Switching Model Profiles

Two profiles are checked in:

| Profile | VRAM | Context | Command |
|---------|------|---------|---------|
| `qwen38-16gb` | ~16 GB | 100K | `sudo bash scripts/switch-model.sh qwen38-16gb` |
| `qwen38-32gb` | ~32 GB | 200K | `sudo bash scripts/switch-model.sh qwen38-32gb` |

The switch command writes a systemd drop-in override and restarts the service. It requires the service to be installed (Step 8).

For one-off runs without systemd:

```bash
PAQ_LLAMACPP_SERVER_ENV_FILE=dot.env.qwen38-27b-q3kxl-100k-16gb ./run-paq-llamacpp-server.sh
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `munmap_chunk(): invalid pointer` | Mixed WSL2 driver libs. Run `bash scripts/provision-wsl2-ubuntu.sh --no-build` |
| `524` from Cloudflare | Enable timeout proxy (Step 7) or reduce prompt size |
| `invalid_api_key` in Copilot | Re-enter key in VS Code secure prompt, not plain text |
| Server won't start, "model not found" | Check `MODEL` path in `.env` matches actual filename in `models/` |
| `GGML_CUDA:BOOL=OFF` in CMakeCache | Rebuild with CUDA: `bash scripts/provision-wsl2-ubuntu.sh` |
| Repeated/stale responses | Set `PARALLEL=1`, `KV_UNIFIED=0`, `PROMPT_CACHE=0` in `.env` |
| `cloudflared` container won't start | Check Docker Desktop is running; verify token in `.env` |
| Port already in use | Change `PORT` in `.env` or kill the conflicting process |

For deeper issues, see:

- [`QWEN38_RELIABILITY_GUIDE.md`](QWEN38_RELIABILITY_GUIDE.md) — Qwen-specific hardening
- [`UPDATING_CUDA.md`](UPDATING_CUDA.md) — CUDA toolkit upgrades
- [`FIXME_QWEN_TOOL_PARSING.md`](FIXME_QWEN_TOOL_PARSING.md) — tool-call parsing notes
- [`TUNING_REPORT.md`](TUNING_REPORT.md) — batch/ubatch benchmarks
