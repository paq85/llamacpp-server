Service management helpers

This folder contains a tiny helper to manage the `paq-llamacpp-server.service` unit.

scripts/servicectl.sh
----------------------
Usage: scripts/servicectl.sh <command> [service]

Quick examples:

  # start the service (clears failed state first)
  scripts/servicectl.sh start

  # enable and start now
  scripts/servicectl.sh start-now

  # view status
  scripts/servicectl.sh status

  # follow logs
  scripts/servicectl.sh tail

Notes:
- The script uses `sudo` under the hood when invoked by a non-root user.
- The default service is `paq-llamacpp-server.service`. You can pass a different
  systemd unit name as the second argument.

WSL2 Ubuntu provisioning
------------------------

For a fresh or repaired WSL2 checkout, run from the repository root:

  bash scripts/provision-wsl2-ubuntu.sh

The provisioning script installs ordinary Ubuntu build prerequisites,
verifies the Windows-provided WSL CUDA bridge, creates the ignored
`cuda-env -> /usr/local/cuda-*` symlink, configures/builds the CUDA
`llama-server`, and runs its version smoke test. It never installs a Linux
NVIDIA driver inside WSL. Use `--server-smoke-test` after the model is in
place to start a temporary text-only server and check `/health`.

`wsl-cuda-env.sh` is a sourced implementation helper used by the launcher and
the grammar rebuild utility. It places the host-matched WSL driver directory
before `/usr/lib/wsl/lib`, toolkit libraries, and inherited library paths so
the Windows `libcuda` is not paired with an older Ubuntu PTX JIT library.

For **upgrading the CUDA toolkit** (install the new toolkit, repoint
`cuda-env`, fetch latest `llama.cpp`, clean rebuild, and smoke test), see
[`UPDATING_CUDA.md`](../docs/UPDATING_CUDA.md) in the docs folder.
