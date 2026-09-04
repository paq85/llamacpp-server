# Qwen 3.8 Reliability Guide (`llamacpp-server`)

This document consolidates the **Qwen 3.8-specific reliability tweaks** implemented in this project so Copilot-style agent workloads (VS Code Copilot and Copilot CLI) run consistently.

> **Multi-profile note:** This project supports two Qwen 3.8 27B profiles via
> profile switching (16 GB VRAM and 32 GB VRAM variants).
> The proxy-side tweaks (§2–§5) apply to both profiles since they share
> the same `<tool_call>` XML format and `<think>` reasoning blocks.
> Server-side settings (§1, §6) are profile-specific — see each profile file for
> model-dependent values. Switch between profiles with:
> ```
> sudo bash scripts/switch-model.sh qwen38-16gb   # Qwen3.8-27B Q3_K_XL @ 100K (16 GB VRAM)
> sudo bash scripts/switch-model.sh qwen38-32gb   # Qwen3.8-27B Q5_K_XL @ 200K (32 GB VRAM)
> ```

> Scope: this is an operations/config guide for this repository (`run-paq-llamacpp-server.sh` + `.env` + `cloudflare-timeout-proxy.py`), not a generic llama.cpp tuning guide.

## WSL2/CUDA prerequisite

On WSL2, NVIDIA driver ownership belongs to the **Windows host**. Do not
install a separate Linux NVIDIA driver in Ubuntu. Before changing llama.cpp
versions or model settings, run the repository bootstrap from Ubuntu WSL2:

  bash scripts/provision-wsl2-ubuntu.sh

It verifies `/usr/lib/wsl/lib/libcuda.so.1`, `nvidia-smi`, and the
host-matched PTX JIT library under `/usr/lib/wsl/drivers/`, then rebuilds the
CUDA server through the stable `cuda-env` symlink. The launcher and grammar
rebuild helper use the same library ordering. If CUDA initialization aborts
with `munmap_chunk(): invalid pointer`, repair this layer first; changing the
llama.cpp commit is unlikely to help when host `libcuda` is mixed with an old
Ubuntu `libnvidia-ptxjitcompiler.so.1`.

To **upgrade the CUDA toolkit**, follow
[`UPDATING_CUDA.md`](UPDATING_CUDA.md) (install the new toolkit, repoint
`cuda-env`, fetch latest `llama.cpp`, clean rebuild, and smoke test).

For the standard `PORT=8080` configuration, clients use the public listener
on `8080`. When the timeout proxy is enabled, the backend defaults to `8081`;
`8082` is only used if `CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT=8082` is set
explicitly. With the proxy disabled, `llama-server` itself listens on `8080`.

## Active tweak values (proxy-side, shared across profiles)

These are the currently active proxy-side reliability values (identical in both profiles):

- `CLOUDFLARE_TIMEOUT_PROXY_MODE=stream`
- `CLOUDFLARE_TIMEOUT_PROXY_BACKEND_PORT` unset (runtime default: `PORT + 1`,
  which is `8081` for the standard profile)
- `CLOUDFLARE_TIMEOUT_PROXY_HEARTBEAT_SECONDS=15`
- `LLAMA_PROXY_STREAM_KEEPALIVE_MODE=reasoning`
- `LLAMA_PROXY_STREAM_KEEPALIVE_TEXT=.`
- `LLAMA_PROXY_CLAMP_TEMPERATURE=0.6`
- `LLAMA_PROXY_CLAMP_TOP_P=0.95`
- `LLAMA_PROXY_SET_TOP_K=20`
- `LLAMA_PROXY_MIN_MAX_TOKENS=` (disabled)
- `LLAMA_PROXY_NUDGE_AFTER_TOOL_RESULT=on`
- `LLAMA_PROXY_CAPTURE_ENABLED=off`
- `REASONING=auto`
- `REASONING_BUDGET=` (unrestricted)

### Profile-specific server-side values

| Setting | 16 GB profile | 32 GB profile |
|---------|---------------|---------------|
| `PARALLEL` | 1 | 1 |
| `KV_UNIFIED` | 0 | 0 |
| `KV_OFFLOAD` | 1 | 1 |
| `CACHE_TYPE_K` | q4_0 | q8_0 |
| `CACHE_TYPE_V` | q4_0 | q8_0 |
| `CTX_SIZE` | 100000 | 200000 |
| `BATCH_SIZE` | 1024 | 1536 |
| `UBATCH_SIZE` | 256 | 384 |
| `PROMPT_CACHE` | 1 | 1 |
| `CACHE_RAM_MIB` | 50000 | 50000 |
| `CTX_CHECKPOINTS` | 20 | 32 |
| `SPEC_TYPE` | draft-mtp | draft-mtp |
| `SPEC_DRAFT_N_MAX` | 1 | 3 |
| `SPEC_DRAFT_N_MIN` | 0 | 0 |
| `SPEC_DRAFT_P_MIN` | 0.0 | 0.0 |
| `SPEC_DRAFT_NGL` | auto | auto |
| `SPEC_DRAFT_CACHE_TYPE_K` | f16 | f16 |
| `SPEC_DRAFT_CACHE_TYPE_V` | f16 | f16 |
| `SPEC_DRAFT_BACKEND_SAMPLING` | 1 | 1 |

---

## 1) Baseline serving profile used here

The checked-in Qwen 3.8 profiles favor correctness + stable tool-calling over maximum concurrency:

- `PARALLEL=1`
- `KV_UNIFIED=0`
- `REASONING=auto`
- `SPEC_TYPE=draft-mtp` (+ explicit draft knobs)
- KV cache quantization per profile (16 GB: `q4_0`, 32 GB: `q8_0`)
- Context checkpointing and prompt cache enabled (`CTX_CHECKPOINTS=20/32`, `CACHE_RAM_MIB=50000`)
- Timeout proxy enabled in stream mode (`CLOUDFLARE_TIMEOUT_PROXY_MODE=stream`)

Why: multi-slot + unified-KV combinations have historically been less predictable on this stack, while single-slot serving gives the most deterministic behavior for long agent loops.

---

## 2) Proxy-side sampling normalization (critical)

Implemented in `cloudflare-timeout-proxy.py` under `RequestStabilizeConfig`.

### Problem addressed

Copilot clients often send high-variance sampling (`temperature=1`, `top_p=1`) which increases malformed or cut-off tool calls for Qwen 3.8 coding flows.

### Tweaks

- `LLAMA_PROXY_CLAMP_TEMPERATURE=0.6`
- `LLAMA_PROXY_CLAMP_TOP_P=0.95`
- `LLAMA_PROXY_SET_TOP_K=20`

Optional floor (disabled by default):

- `LLAMA_PROXY_MIN_MAX_TOKENS=`
  - Only raises an **existing** too-small `max_tokens` / `max_completion_tokens`
  - Never injects a cap when client omitted one

Output token ceiling (enabled by default at 50% of context):

- `LLAMA_PROXY_MAX_TOKENS_CTX_PCT=50`
  - Caps `max_tokens` / `max_completion_tokens` at this percentage of `CTX_SIZE`
  - Unlike the floor, **does** impose a limit when the client sends none
  - Prevents unbounded generation from exhausting the context window
  - Set to `0` or leave empty to disable

### Effect

Outgoing requests are stabilized to Qwen-friendly sampling without changing client integrations.

---

## 3) Tool-result continuation nudge (critical)

Implemented in `cloudflare-timeout-proxy.py` via `_append_tool_result_continuation_nudge`.

### Problems addressed

After a `role=tool` result, Qwen 3.8 could intermittently end a turn with:

1. **empty assistant stop** (no content, no tool call), or
2. **reasoning-only stop** (thinking text only, no visible content/tool call)

In both cases Copilot appears to stop mid-task because there is no actionable next tool call.

### Tweaks

- `LLAMA_PROXY_NUDGE_AFTER_TOOL_RESULT=on`
- `LLAMA_PROXY_TOOL_RESULT_NUDGE_TEXT` set to:

`Continue after the latest tool result. Do not stop in the reasoning/thinking channel. Before finishing, you must emit either a valid tool_call or visible assistant content. If any planned work remains, call the next appropriate tool now. Do not merely describe the next step in reasoning.`

The same exact string is also the proxy default fallback (`DEFAULT_TOOL_RESULT_CONTINUATION_NUDGE`) in `cloudflare-timeout-proxy.py`.

The current nudge explicitly forbids stopping in reasoning-only mode and requires either:

- a valid `tool_call`, or
- visible assistant content.

### Deployment detail fixed

The proxy now prefers `.env` values over inherited process env for this nudge text (`_env_text_with_envfile`, `_env_bool_with_envfile`), so **proxy-only restarts** pick up updated wording reliably.

---

## 4) Streaming keepalive behavior for Copilot

Implemented in `cloudflare-timeout-proxy.py`.

### Problem addressed

When the single backend slot is busy, strict clients can appear stalled if they receive no meaningful stream activity before first model bytes.

### Tweaks

- `LLAMA_PROXY_STREAM_KEEPALIVE_MODE=reasoning`
- `LLAMA_PROXY_STREAM_KEEPALIVE_TEXT=.`

Modes available:

- `comment`: SSE comment only
- `data`: empty OpenAI chunk
- `reasoning`: reasoning delta chunk (used here)

### Additional relay fix already in place

SSE relay uses `read1()` instead of blocking `read()` so final chunks / `[DONE]` are not held behind large buffer waits.

---

## 5) Reasoning budget support (optional safety rail)

Wired through `run-paq-llamacpp-server.sh`:

- `REASONING_BUDGET` → `--reasoning-budget`
- `REASONING_BUDGET_MESSAGE` → `--reasoning-budget-message`

Default in this repo is unrestricted (`REASONING_BUDGET=`). If tool calls ever truncate under extreme turns, start with a bounded value (for example `8192`) and re-validate.

---

## 6) MTP + cache choices used for stability/perf balance

Both profiles share the same MTP speculative-decoding settings:

- `SPEC_TYPE=draft-mtp`
- `SPEC_DRAFT_N_MAX=3`
- `SPEC_DRAFT_N_MIN=0`
- `SPEC_DRAFT_P_MIN=0.0`
- `SPEC_DRAFT_NGL=auto`
- `SPEC_DRAFT_CACHE_TYPE_K=f16`
- `SPEC_DRAFT_CACHE_TYPE_V=f16`
- `SPEC_DRAFT_BACKEND_SAMPLING=1`

Cache/runtime settings differ per profile (see the profile-specific table in the "Active tweak values" section above):

| Setting | 16 GB profile | 32 GB profile |
|---------|---------------|---------------|
| `CACHE_TYPE_K` | `q4_0` | `q8_0` |
| `CACHE_TYPE_V` | `q4_0` | `q8_0` |
| `KV_OFFLOAD` | `1` | `1` |
| `PROMPT_CACHE` | `1` | `1` |
| `CTX_CHECKPOINTS` | `20` | `32` |

This combination is the currently validated one in this project; treat changes as experiments and replay captured traffic after each major change.

---

## 7) Capture + replay workflow (recommended)

For real-client validation:

- `LLAMA_PROXY_CAPTURE_ENABLED=on` (temporarily while debugging)
- `LLAMA_PROXY_CAPTURE_DIR="$ROOT/captures"`

Use real captured request bodies to replay regressions and confirm fixes before declaring stability.

Current default in this workspace is `LLAMA_PROXY_CAPTURE_ENABLED=off`; keep it that way in normal operation because request bodies can be large and sensitive.

---

## 8) Reliability checklist for changes

When touching Qwen-serving parameters, validate in this order:

1. `python3 -m py_compile cloudflare-timeout-proxy.py`
2. health endpoint: `https://127.0.0.1:8080/health`
3. replay the latest failing capture multiple times
4. confirm no:
   - reasoning-only stops,
   - empty assistant stops,
   - malformed tool-call arguments,
   - missing `[DONE]` / stalled streams
5. check backend logs for `truncated = 0` on validated runs

---

## 9) What to avoid first

If reliability regresses, avoid jumping immediately to high-risk changes like:

- `PARALLEL>1`
- `KV_UNIFIED=1`
- aggressive sampling increases

Start by preserving the baseline above and adjusting only one variable at a time with capture replay.
