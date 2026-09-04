# Updating CUDA on this host (WSL2)

This is the single reference for upgrading the CUDA stack and rebuilding
`llama.cpp`. It consolidates the hard-won gotchas so a future CUDA bump takes
minutes instead of requiring a fresh investigation.

**Scope:** NVIDIA CUDA toolkit + driver on the **WSL2** host used by this repo
(`llamacpp-server`). It is NOT a generic llama.cpp tuning guide.

> The source of truth for the project is [`README.md`](README.md) and
> [`AGENTS.md`](AGENTS.md). This guide is referenced from both.

---

## Mental model

On WSL2 there are **two independent CUDA layers**:

| Layer | Where it lives | Who updates it |
|---|---|---|
| **GPU driver + user-mode libraries** | Windows host (the `wsl` kernel driver the Windows NVIDIA driver exposes into `/usr/lib/wsl/...`) | Update the **Windows** NVIDIA driver / `wsl --update` |
| **CUDA toolkit** (nvcc, `libcudart`, `libcublas`, headers) | Ubuntu filesystem (`/usr/local/cuda-*`) | `apt` install of `cuda-toolkit-<NN>-<MM>` in Ubuntu |

- The repo resolves the toolkit through the **`cuda-env`** symlink
  (`cuda-env -> /usr/local/cuda-<N>`), so builds and `run-paq-llamacpp-server.sh` never
  hardcode an absolute CUDA version path.
- The **driver must never** be installed as a Linux package inside WSL. It
  comes from Windows. Installing `nvidia-driver-*` / `libnvidia-*` in Ubuntu
  breaks the host/WSL CUDA bridge.
- `scripts/wsl-cuda-env.sh` sets the runtime `LD_LIBRARY_PATH` so the
  **host-matched WSL driver directory** (containing `libnvidia-ptxjitcompiler.so.1`)
  is first, ahead of `/usr/lib/wsl/lib`, the toolkit libs, and any inherited
  paths.

---

## Prerequisite: Windows host driver is current first

`nvidia-smi` in Ubuntu reports the **UMD (user-mode driver)** version that the
Windows host exposes. Check it before doing anything on the Ubuntu side:

```bash
nvidia-smi   # look for "CUDA UMD Version"
```

If the UMD version is older than the CUDA toolkit you want to use, update the
**Windows** NVIDIA driver (and `wsl --update`), then restart the WSL
distribution. Do NOT try to fix a driver mismatch by installing Linux driver
packages.

---

## Step-by-step upgrade

Run from the repo root.

### 1. Install (or upgrade) the CUDA toolkit in Ubuntu

```bash
sudo apt-get update
apt-cache policy cuda-toolkit-13-3   # confirm the version you want is available
sudo apt-get install -y cuda-toolkit-13-3
```

- Installs **side-by-side** under `/usr/local/cuda-13.3` (older versions stay).
- Updates `/etc/alternatives/cuda` / `cuda-13` to the newest toolkit.
- Does **not** touch the driver.

> **sudo needs a password** on this box. `scripts/provision-wsl2-ubuntu.sh`
> calls `sudo -v`, so it will hang if run non-interactively. Run the `apt`
> install yourself; the rest of the rebuild below needs **no sudo**.

### 2. Repoint the `cuda-env` symlink

The provision script does this automatically (`select_cuda_root` picks the
newest `/usr/local/cuda-*`). To do it manually:

```bash
rm -f cuda-env
ln -s /usr/local/cuda-13.3 cuda-env
cuda-env/bin/nvcc --version        # confirm, e.g. "release 13.3, V13.3.73"
```

### 3. Fetch the latest `llama.cpp`

`llama.cpp/` is an embedded **upstream** checkout (own git repo, ignored by the
root repo).

```bash
cd llama.cpp
git fetch origin
git pull --ff-only origin master
```

If the pull refuses because of local changes, commit/discard them first — the
repo ships **stock upstream source** by default (the `MAX_REPETITION_THRESHOLD`
grammar patch is opt-in via `apply-llama-grammar-threshold.sh --patch --rebuild`).

### 4. Clean rebuild

> **Important:** always `rm -rf build` when changing the CUDA version. Doing an
> in-place reconfigure keeps stale `CUDAToolkit_BIN_DIR` / `CUDAToolkit_*` cache
> entries pinned to the old toolkit, which causes a **mixed toolchain** and can
> silently leave `GGML_CUDA=OFF` (a CPU-only binary).

```bash
cd llama.cpp
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$PWD/../cuda-env/bin/nvcc" \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_COMPRESSION_MODE=size \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server --parallel "$(nproc)"
```

Notes:
- `CUDAToolkit_ROOT` pins the finder to the new toolkit (prevents it from
  matching `/usr/local/cuda` alternatives strictly by name).
- `120a-real` targets the RTX 5090 (`sm_120a`). For another GPU, pass
  `--arch` / `-DCMAKE_CUDA_ARCHITECTURES`.
- The CMake args mirror `scripts/provision-wsl2-ubuntu.sh` exactly, so the build
  stays deterministic with the provisioner.

### 5. Verify

```bash
cd llama.cpp

# The build must be CUDA-enabled and use the new toolkit
grep -E "^(GGML_CUDA|CMAKE_CUDA_COMPILER|CUDAToolkit_NVCC_EXECUTABLE):" build/CMakeCache.txt
# expect: GGML_CUDA:BOOL=ON, CUDAToolkit_NVCC_EXECUTABLE=/usr/local/cuda-13.3/bin/nvcc

# Smoke-test with the CORRECT runtime library ordering (see gotcha below)
cd ..
source scripts/wsl-cuda-env.sh
configure_wsl_cuda_runtime "$PWD/llama.cpp/build/bin" "$PWD/cuda-env"
./llama.cpp/build/bin/llama-server --help >/dev/null && echo "startup OK"
```

### 6. Optional: full GPU smoke test (use a real model)

```bash
cd ..
source scripts/wsl-cuda-env.sh
configure_wsl_cuda_runtime "$PWD/llama.cpp/build/bin" "$PWD/cuda-env"
./llama.cpp/build/bin/llama-server \
  -m models/Qwen3.5-9B-MTP-Q8_0.gguf \
  --host 127.0.0.1 --port 18099 -ngl 200 --no-webui &
sleep 15
curl -s http://127.0.0.1:18099/health      # expect {"status":"ok"}
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv   # confirm GPU memory in use
pkill -f "llama-server -m models/Qwen3.5-9B"
```

---

## Gotchas (these cost real time if ignored)

1. **Benign segfault from a bare shell.** A freshly built `llama-server` /
   `llama-cli` segfaults in `libnvidia-ptxjitcompiler.so.*` (signal 11) on
   `--help`/`--version` when run without the driver-dir-first `LD_LIBRARY_PATH`.
   This is **normal** — you must `source scripts/wsl-cuda-env.sh` **and** call
   `configure_wsl_cuda_runtime ...` first. Sourcing alone does nothing.
2. **Stale CMake cache after a CUDA bump.** In-place reconfigure keeps old
   `CUDAToolkit_*` entries → mixed toolkit / `GGML_CUDA=OFF`. Wipe `build/` and
   pass `-DCUDAToolkit_ROOT=<new>`.
3. **Do not install a Linux NVIDIA driver in WSL2.** The driver belongs to the
   Windows host.
4. **Do not load GGUFs from Windows-mounted drives** (`/mnt/n/...`) — causes
   `munmap_chunk(): invalid pointer` aborts. Use local copies in `models/`.
5. **Verify with `nvidia-smi` after a model load** (GPU memory should increase);
   a build can be CUDA-linked yet fail at runtime if the driver UMD is stale.

---

## What to do after the rebuild

- Restart the service so it picks up the new binary:
  ```bash
  sudo systemctl restart paq-llamacpp-server
  ```
- Confirm the service comes up cleanly:
  ```bash
  bash scripts/servicectl.sh status
  ```
- If the model fails to load, check `scripts/servicectl.sh tail` for CUDA /
  VTK / context errors before blaming the llama.cpp version.
