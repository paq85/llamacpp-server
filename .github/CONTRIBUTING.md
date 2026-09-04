# Contributing to llamacpp-server

Thank you for your interest in contributing! This project is a wrapper around
`llama.cpp` that serves a local LLM via an OpenAI-compatible API, with a
Cloudflare tunnel + SSE keepalive proxy to eliminate VS Code GitHub Copilot
timeouts during long inference.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/llamacpp-server.git
   cd llamacpp-server
   ```
3. Set up the environment (see [docs/SETUP.md](docs/SETUP.md))

## Project Structure

```
llamacpp-server/
├── run-paq-llamacpp-server.sh              # Main launcher (env loading, arg building, process lifecycle)
├── cloudflare-timeout-proxy.py  # SSE proxy with keepalives, sampling clamp, tool nudging
├── stop-paq-llamacpp-server.sh             # Graceful shutdown
├── install-systemd-service.sh  # Systemd unit installation
├── scripts/                  # Provisioning, service management, CUDA env
├── docs/                     # Setup guide and other documentation
├── dot.env.*                 # Model profiles (one per GPU size)
└── .github/                  # Contribution guidelines, templates
```

## Making Changes

### Code Style

- **Bash**: `set -euo pipefail` in all scripts. Resolve `ROOT` from script
  location (no hardcoded paths).
- **Python**: Standard library only for the proxy (no external dependencies).
  Follow PEP 8.
- **Shell scripts**: Use `flock` for single-instance locking where needed.

### Testing

Before submitting a PR:

1. Verify your changes work locally:
   ```bash
   ./run-paq-llamacpp-server.sh          # Start the server
   python3 test-context-usage.py  # Validate usage/timings
   ```
2. If you modified the proxy, test with a real Copilot session or:
   ```bash
   python3 scripts/toolcall-stress.py --iterations 40
   ```
3. Ensure no regressions in existing functionality

### Commit Messages

Use clear, imperative commit messages:

```
Add support for model X
Fix KV cache offload race condition
Update CUDA toolkit to 12.4
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test thoroughly
4. Submit a PR with a clear description of what changed and why
5. Address review feedback

## What to Contribute

- Bug fixes (especially around proxy behavior, CUDA compatibility)
- New model profiles (add a `dot.env.*` file + update `scripts/switch-model.sh`)
- Documentation improvements
- Performance optimizations
- New features (proxy capabilities, monitoring, etc.)

## What NOT to Contribute

- Model files (GGUF weights) — use HuggingFace links instead
- API keys or credentials
- Changes to the embedded `llama.cpp/` checkout (upstream only)

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).
