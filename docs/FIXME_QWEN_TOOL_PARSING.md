# FIXME: Qwen 3.8 Tool-Call / Reasoning Parsing Instability

Date: 2026-06-08  
Workspace: `llamacpp-server`  
Embedded `llama.cpp`: branch `master` (observed commit: `18ef86ecec723361362a332a79b4d913fd724d40`)

---

## Goal

Fix intermittent coding-agent failures (GitHub Copilot in VS Code, GitHub Copilot CLI stopping after tool calls) when serving `Qwen3.8-27B-UD-Q5_K_XL.gguf`, **while keeping reasoning enabled**.

> Status: **ROOT CAUSE REPRODUCED AND CONFIRMED (2026-06-08).** The original
> hypothesis (server-side PEG "Failed to parse" throw) was **disproven** as the
> primary cause by reproduction. The actual cause is **tool-call truncation**:
> with reasoning enabled, the `<think>` block consumes the output-token budget,
> so the tool call is cut off (`finish_reason="length"`) and the client receives
> **incomplete/invalid tool-call JSON** and stops. See "Reproduction" below.

> Update (real traffic captured): VS Code Copilot sends **`max_tokens=None`**
> (no output cap), **`temperature=1`, `top_p=1`** (overriding this model's
> recommended `0.6`/`0.95`), **~93 tools**, and conversations up to ~143 messages
> / ~79K tokens. So in normal use, truncation is rare (lots of context headroom),
> but the client's **temp=1/top_p=1 override** raises the odds of malformed/
> truncated tool calls. The fix therefore (a) **clamps sampling** back to the
> model's recommended values at the proxy and (b) adds an **optional reasoning
> budget** so thinking can't crowd out the tool call. Reasoning stays ON.

> Implementation status (applied, off-by-default except the clamp):
> - **Proxy sampling clamp** (`cloudflare-timeout-proxy.py`): env-gated
>   `LLAMA_PROXY_CLAMP_TEMPERATURE` / `LLAMA_PROXY_CLAMP_TOP_P` /
>   `LLAMA_PROXY_SET_TOP_K` / `LLAMA_PROXY_MIN_MAX_TOKENS`. **Enabled** in `.env`
>   at `0.6` / `0.95` / `20`. The `max_tokens` floor only **raises an existing
>   too-small** limit and never adds one when the client sent none.
> - **Reasoning budget** (`run-paq-llamacpp-server.sh`): `REASONING_BUDGET` →
>   `--reasoning-budget`, plus `REASONING_BUDGET_MESSAGE`. Present in `.env`,
>   empty (unrestricted) by default; set `8192` if truncation is ever observed.
> - **Request capture** (`LLAMA_PROXY_CAPTURE_DIR`) + harness `--replay` for
>   validating against real client payloads.
> - Validated: unit + end-to-end proxy tests pass; a real captured Copilot
>   payload and an 18-turn run at client `temp=1` both complete with **0
>   failures**, reasoning on every turn.
> - Pending (approved): optional Layer-3 `llama.cpp` final-parse fallback patch
>   (needs a rebuild — confirm before doing it); capture a Copilot CLI session.

> **WSL2 prerequisite (current operational guidance):** Before investigating
> tool parsing or downgrading llama.cpp, run
> `bash scripts/provision-wsl2-ubuntu.sh` from Ubuntu WSL2. The Windows host
> supplies the NVIDIA driver; do not install a Linux NVIDIA driver in WSL.
> The launcher now prefers the host-matched PTX JIT library under
> `/usr/lib/wsl/drivers/` over older Ubuntu packages. This avoids the separate
> CUDA initialization abort (`munmap_chunk(): invalid pointer`) that was
> diagnosed after the original tool-call work documented here.

---

## Update 2026-06-08 (afternoon): SECOND, distinct root cause found & fixed — proxy SSE buffering hang

The truncation/sampling work above addresses *malformed* tool calls. But a
live "Copilot died in the middle of the next task" incident turned out to be a
**different bug in the timeout proxy**, not in the model or llama.cpp parsing.

**Symptom:** Copilot hung mid-agentic-loop. The server stayed healthy
(`/health` 200 the whole time). The SIGTERM seen in the log was the **user
stopping the server later** — it was *not* the cause.

**Evidence (from `/tmp/paq-llamacpp-server-test.log`, original launch):**
- The last completion (`task 1124`) finished cleanly server-side:
  `release: task 1124 | stop processing: n_tokens = 9757, truncated = 0`,
  `reasoning-budget: deactivated (natural end)`, `all slots are idle`.
- Tasks 988 and 1053 each got a `POST /v1/chat/completions 200` access-log line
  (14:40:17, 14:40:20). **`task 1124` never did** — the next line is a
  `GET /health 200` ~67 s later, then the user's SIGTERM. The proxy access-log
  line is only emitted when the response stream **fully closes**, so the proxy
  was still inside that request, blocked, for 60 s+.

**Root cause:** `cloudflare-timeout-proxy.py` → `_relay_response_body` SSE
branch read the upstream with `response.read(self.config.read_chunk_size)` and
`--read-chunk-size` defaults to **65536**. `http.client.HTTPResponse.read(N)`
**blocks until it has N bytes OR upstream EOF**. A small final response (here 56
generated tokens) never fills 64 KiB, so the finish chunk + `data: [DONE]` are
held inside `read()` until llama.cpp's terminating 0-chunk/EOF. The proxy sends
`Connection: close` upstream and uses `upstream_timeout=3600`, so any delayed
EOF (keep-alive race, tunnel backpressure) pins `[DONE]` for up to an hour →
the streaming client hangs mid-turn. It is **intermittent** because it depends
on llama.cpp's EOF timing.

**Reproduced** by replaying the captured request:
- backend direct (`:8081`): **13 arrival bursts** (true streaming).
- proxy (`:8080`, before fix): **2 bursts** — an immediate `: keep-alive`, then
  the *entire* body delivered at once at the end (buffered/held).

**Fix:** use `response.read1(self.config.read_chunk_size)` in the SSE relay
(single underlying read, returns whatever is available, breaks promptly at EOF).
Each `data:` line — including `[DONE]` — is now forwarded the instant it
arrives, so the client gets its stream terminator before any final blocking
read.

**Validated (after fix):**
- proxy now streams in **10 incremental bursts** (matches backend's 7), with
  `finish_reason` and `[DONE]`, clean close.
- no-regression stress (VS Code client, `temp=1`, 6×4 turns): **24/24 turns with
  tool calls, 22 with reasoning, 0 failures**.
- proxy hot-restarted via the supervisor (kill proxy PID → `run-paq-llamacpp-server.sh`
  relaunches it) with **no model reload**.

**File:** `cloudflare-timeout-proxy.py`, `_relay_response_body` (sse branch).

> Note: the sibling non-SSE streaming loop in the same method still uses
> `response.read(...)`; it is not on the Copilot path (chat `stream=true` is
> always `sse=True`). Left unchanged to keep the fix surgical; could be switched
> to `read1` later for consistency.

---

## Current symptom

- Failures are intermittent:
  - sometimes stable for several minutes,
  - sometimes fail after the first tool call.
- Affects **GitHub Copilot in VS Code** (agent mode) and **GitHub Copilot CLI**
  (terminal agent; ships the GitHub MCP server + supports custom MCP servers).
- Disabling reasoning "fixes" it for some people — consistent with the confirmed
  cause (without `<think>`, the full output budget goes to the tool call). The
  user requires reasoning ON, so the fix must prevent truncation **with**
  reasoning enabled.

---

## Environment snapshot (relevant)

From `.env`:

- `REASONING=auto`
- `SPEC_TYPE=draft-mtp`
- `SPEC_DRAFT_N_MAX=2`
- `SPEC_DRAFT_BACKEND_SAMPLING=1`
- Timeout proxy enabled: `CLOUDFLARE_TIMEOUT_PROXY_MODE=stream`
- Custom template currently commented out:
  - `chat_templates/chat_template.jinja`

From launcher (`run-paq-llamacpp-server.sh`):

- Starts with `--jinja`
- Uses `--reasoning "$REASONING"`
- Optional `--chat-template-file` if configured

---

## Reproduction (CONFIRMED)

### How it was reproduced

- Server run **without** systemd (so no root needed):
  `CLOUDFLARED_ENABLED=off ./run-paq-llamacpp-server.sh > /tmp/paq-llamacpp-server-test.log 2>&1 &`
  - backend: `http://127.0.0.1:8081` (plain HTTP), proxy: `https://127.0.0.1:8080` (TLS).
- New harness [`scripts/toolcall-stress.py`](scripts/toolcall-stress.py) drives many
  streamed, multi-turn tool-calling conversations with **reasoning enabled** and a
  **Copilot-faithful** setup: ~80-tool catalog (VS Code agent tools + GitHub MCP
  tools + padding), a Copilot/Copilot-CLI-style system prompt, parallel tool calls,
  and large code-heavy arguments. It flags HTTP errors, SSE `error` events,
  `Failed to parse`, missing `finish_reason`, and **invalid tool-call argument JSON**,
  saving the raw request + raw SSE of each failure.

### Results

| Condition | Requests | Failures | Kind |
|---|---|---|---|
| Direct backend, `max_tokens=1400`, temp 0.7 | 434 | **0** | — |
| Proxy path, `max_tokens=2000`, temp 0.95 | ~45 | **0** | — |
| Proxy path, `max_tokens=220`, temp 0.9 | 119 | **17 (14%)** | `bad_tool_args` (all `finish_reason=length`) |

- The failure is **100% controlled by the output-token budget**: generous budget
  → 0 failures; tight budget → frequent failures. This is the A/B proof.
- **Zero** `Failed to parse input at pos` in the server logs across ~1000+ requests
  → the server's PEG "final parse throw" is **not** the cause.

### Captured evidence (a representative failure)

Final SSE chunk: `"finish_reason":"length"` after the model emitted a `<think>` block,
then an incomplete tool call. The streamed `tool_calls[].function.arguments` assembled to:

```
{"filePath":"cli/main.py","content":"#!/usr/bin/env python3\n\"\"\"...class RecordManager:\n    \"\"\
```

…an **unterminated JSON string**. llama.cpp's lenient parser returns this *partial*
tool call with `HTTP 200`; the **client** then fails to `JSON.parse` the arguments
and aborts the task. Artifacts: `benchmarks/toolcall-stress/*bad_tool_args*.{response.sse,summary.txt,request.json}`.

---

## Confirmed root cause

1. Qwen 3.8 reasoning emits a `<think>...</think>` block **before** the tool call.
2. The effective **output-token budget** = the client's `max_tokens` (for VS Code
   Copilot BYOK / Copilot CLI this is the configured *max output tokens*), capped by
   remaining context.
3. When `reasoning_tokens + tool_call_tokens > budget`, generation stops with
   `finish_reason="length"` **inside** the tool call.
4. The tool call's `arguments` is therefore **truncated → invalid JSON**.
5. llama.cpp does **not** error (lenient parse, HTTP 200). The **client** receives a
   broken tool call, cannot parse it, and **stops the task**.

Why it's intermittent: only turns where `reasoning + tool call` overflow the budget
fail — i.e. long reasoning and/or large tool arguments (big file create/edit). Short
exchanges complete fine, which is exactly the "works 10 min, then fails" pattern.

### Secondary / ruled-out

- **PEG "Failed to parse" throw** — real code path
  ([`common/chat.cpp`](llama.cpp/common/chat.cpp) final parse;
  [`tools/server/server-context.cpp`](llama.cpp/tools/server/server-context.cpp)
  streaming loop wraps it in try/catch and emits an SSE `error`), but **not observed**
  in reproduction. Keep as a lower-priority hardening target.
- **Backend-sampling vs grammar**: confirmed in `common/sampling.cpp` that this disables
  *backend sampling*, not the grammar — **not** a corruption vector.
- **Proxy SSE rewrite**: failures occur identically on the **direct backend**, so the
  proxy is **not** the cause (it fails safe to passthrough on non-JSON lines).
- **`chat_template.jinja` (unified froggeric template, v22.1)**: the template
  adds decode overhead.

---

## Client compatibility (VS Code Copilot + Copilot CLI)

Both clients hit the **same** server, so a server-side fix covers both. Key facts:

- **GitHub Copilot in VS Code (BYOK / agent mode)**: the model config exposes a
  *context size* / *max output tokens*. If max output tokens is small, reasoning +
  tool call overflow it → truncated tool call → "Copilot stopped". Agent mode also
  sends a **large tool catalog** and expects **complete, valid** tool-call JSON.
- **GitHub Copilot CLI** (`github/copilot-cli`): "same agentic harness as Copilot
  coding agent", ships the **GitHub MCP server** by default and supports custom MCP
  servers → even larger tool catalogs and large tool arguments, same truncation risk.

Validation gold standard: the proxy now supports **request capture** (set
`LLAMA_PROXY_CAPTURE_DIR`) to record the *exact* payloads each client sends, and the
harness supports **replay** (`--replay <dir>`). This lets us reproduce and verify the
fix against **authentic** VS Code Copilot and Copilot CLI traffic, not just synthetic
requests.

---

## Fix plan (keep reasoning ON; must work with VS Code Copilot + Copilot CLI)

The fix is **budget management**, layered so it is robust regardless of how each
client is configured. Layers 1–2 are the core; layer 3 is hardening.

### Layer 1 — Bound reasoning so it cannot consume the whole output budget (server-side)

- Add a **reasoning budget** so the `<think>` block is capped, always leaving room
  for the tool call. llama.cpp supports `--reasoning-budget N` (thinking-token cap;
  on exhaustion it closes `</think>` and proceeds to the answer/tool call).
- Wire a new `REASONING_BUDGET` env var through `run-paq-llamacpp-server.sh` → `--reasoning-budget`,
  default unset (current behavior) but set to a sane value (e.g. a few thousand
  tokens) in `.env`.
- Tradeoff to verify: `--reasoning-budget` disables *backend sampling*
  (`common/sampling.cpp`), which interacts with `SPEC_DRAFT_BACKEND_SAMPLING=1` (MTP).
  Measure the speculative-decoding throughput impact (reuse `bench-*.py`) and confirm
  MTP still functions; keep it only if the perf cost is acceptable.
- Keeps reasoning **enabled** — just bounded.

### Layer 2 — Ensure an adequate output-token budget (client config; both clients)

Even with bounded reasoning, large tool calls (full-file `create_file` /
`replace_string_in_file`) need room. Both clients control `max_tokens`, so document
exact settings:

- **VS Code Copilot (BYOK model config)**: raise the model's **max output tokens**
  (and context size) so `reasoning + largest tool call` fits comfortably
  (recommend ≥ 8192–16384 output tokens for agentic coding).
- **GitHub Copilot CLI (model config)**: same — configure a generous max output for
  the custom/local model.
- Optional server-side safety floor (proxy): when an inference request carries
  `tools` and either omits `max_tokens` or sets it very low, the proxy can raise it to
  a safe floor before forwarding to llama-server. Targeted (tools-only) and
  client-agnostic. Evaluate carefully (changes client semantics) before enabling.

### Layer 3 — Hardening (defense in depth)

- **Server PEG throw**: as a separate robustness improvement, make a failed **final**
  parse fall back to content-only instead of throwing (lower priority; not observed in
  reproduction). Requires an embedded `llama.cpp` patch/rebuild — only if it recurs.
- **Observability**: keep `LLAMA_PROXY_CAPTURE_DIR` capture available; optionally log a
  warning when `finish_reason=length` coincides with an open tool call, to monitor
  residual truncation in production.

---

## Validation criteria (must pass for BOTH clients)

1. **Synthetic harness**: [`scripts/toolcall-stress.py`](scripts/toolcall-stress.py)
   with `--client vscode` and `--client cli`, large catalog, reasoning ON, generous
   budget → **0** `bad_tool_args` / **0** parse failures across ≥ 200 tool-call turns.
2. **Real-traffic replay (gold standard)**: capture real payloads via
   `LLAMA_PROXY_CAPTURE_DIR`, then `toolcall-stress.py --replay <dir>` → 0 failures.
   Capture from **both** a VS Code Copilot agent session and a Copilot CLI session.
3. **Live sessions**: a sustained VS Code Copilot agent task **and** a Copilot CLI task
   each complete without a mid-task stop, with reasoning visible/working.
4. **Logs**: zero `Failed to parse input at pos`; zero `finish_reason=length` inside
   tool calls during the validated runs.
5. **No regression**: MTP speculative decoding still active; acceptable
   throughput/VRAM (reuse `bench-*.py`); `reasoning_content` still streams.

---

## How to capture real client traffic (for replay validation)

1. Start the server with capture enabled (no systemd):
   `CLOUDFLARED_ENABLED=off LLAMA_PROXY_CAPTURE_DIR=$PWD/captures ./run-paq-llamacpp-server.sh`
2. Point **VS Code Copilot** (BYOK) and **Copilot CLI** at the endpoint and run a real
   tool-using task in each. Each `/v1/chat/completions` request body is saved under
   `captures/` (auth headers stripped; dir is git-ignored).
3. Reproduce/validate against authentic payloads:
   `python3 scripts/toolcall-stress.py --replay captures --stop-on-first`

---

## Artifacts produced

- [`scripts/toolcall-stress.py`](scripts/toolcall-stress.py) — repro + replay harness
  (flags: `--url`, `--catalog-size`, `--client {vscode,cli}`, `--max-tokens`,
  `--replay`, `--stop-on-first`).
- [`cloudflare-timeout-proxy.py`](cloudflare-timeout-proxy.py) — added env-gated request
  capture (`LLAMA_PROXY_CAPTURE_DIR`); detection-only, does not alter requests.
- `benchmarks/toolcall-stress/` — saved failing request + raw SSE artifacts (git-ignored).
- `captures/` — real client request payloads when capture is enabled (git-ignored).

---

## Non-goals

- No model/template swap as the primary fix (reasoning stays ON; template is not the cause).
- No embedded `llama.cpp` rebuild unless Layer 3 server hardening becomes necessary.

---

## Notes

- Primary framing is now **output-budget truncation**, not parser/format mismatch.
- Preserve reasoning capability as a hard requirement throughout.
