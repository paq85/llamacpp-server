# Tuning Report — Qwen3.8-27B Q5_K_XL @ 200K on RTX 5090

- **Date:** 2026-09-04
- **Host:** RTX 5090 (32 GB / 32,607 MiB), WSL2, llama.cpp `0.3.0-dev` (build 10807)
- **Model:** `Qwen3.8-27B-UD-Q5_K_XL.gguf` (20 GB) + `mmproj-qwen38-27b-F16.gguf` (885 MB)
- **Context:** 200,000 tokens (kept)
- **Profile:** single concurrent user, everything fits entirely in VRAM
- **Scope:** agentic coding / code generation, proxy disabled (`CLOUDFLARE_TIMEOUT_PROXY_MODE=off`)

## Approach

Restart-based sweeps via `run-paq-llamacpp-server.sh` with `CLOUDFLARE_TIMEOUT_PROXY_MODE=off`
(no Cloudflare proxy). Each variant reloads the model, warms up, then measures:

- **Prefill** (prompt tok/s) and **decode** (tok/s) at a **51,200-token prompt**
  (≥50K per requirement) for both.
- **GPU memory** used/free via `nvidia-smi` to enforce the "everything fits in
  VRAM" constraint (configs that spill/approach 100% VRAM were excluded as
  obviously slower).

Model architecture (from `paq-llamacpp-server-base-meta/config.json`): hybrid Qwen3.5-style,
32 layers (8 full-attention, 24 linear-attention), 4 KV heads × head_dim 256,
so the KV cache is small enough that model + KV + compute sit in VRAM at 200K.

## 1) MTP (draft-mtp) n_max sweep — decode tok/s @ ~50K context

| n_max | decode tok/s | draft accept | GPU used/free |
| ---: | ---: | ---: | ---: |
| 1 | 89.4 | 85.5% | 29.8 / 2.4 GB |
| 2 | 100.4 | 72.9% | 29.9 / 2.3 GB |
| **3** | **113.5** (best, verified) | 70.1% | 30.1 / 2.1 GB |
| 4 | 109.3 (noisy: 122.6 on first pass) | 58.6% | 30.2 / 2.0 GB |
| 5 | 102.5 | 46.0% | 30.4 / 1.8 GB |
| 6 | 98.5 | 40.7% | 30.5 / 1.7 GB |

Two passes were run. First pass (2 runs): n3=118.4, n4=122.6, n2=100.4.
Confirmation pass (4 runs, `benchmarks/20260904-143457-agent.*`): **n2=106.4,
n3=113.5, n4=109.3**. With more samples n3 is the robust maximum — n4's earlier
peak was run-to-run variance. **Conclusion: `SPEC_DRAFT_N_MAX=3`** (~7% decode
gain over the previous n2), healthy ~70% acceptance, and 2.1 GB VRAM headroom.

Prefill at 50K was ~2,600 tok/s regardless of n_max (~19 s for 50K tokens).

## 2) KV cache type (at 200K, q8_0 baseline)

| Cache type | decode tok/s | GPU used/free |
| ---: | ---: | ---: |
| **q8_0 / q8_0** | **104.1** | 29.9 / 2.3 GB |
| f16 / f16 | 19.8 | 32.2 / 0.03 GB |

**Conclusion:** keep `q8_0`. f16 nearly doubles KV bytes → ~100% VRAM and a
5× decode collapse; excluded per the "must fit in VRAM" rule.

## 3) Vision config (MTP n2 fixed)

| Config | decode tok/s | GPU used/free |
| ---: | ---: | ---: |
| on + mmproj offloaded (GPU) | 101.5 | 29.9 / 2.3 GB |
| on + mmproj on CPU | 103.3 | 29.0 / 3.2 GB |
| off | 102.6 | 29.0 / 3.2 GB |

All within run-to-run noise (~±2%). `MMPROJ_OFFLOAD=0` (CPU) frees ~0.9 GB VRAM
and is marginally faster for text; it only costs image-encoding speed when the
agent actually sends screenshots. **Profile keeps vision on + offloaded (GPU)**
for multimodal agent correctness.

## 4) Context / VRAM fit

The tuned config uses **~30.1 GB / 32 GB** at 200K with MTP n3, leaving ~2.1 GB
headroom. No RAM-offload of the active KV cache is required; everything fits.

## Applied configuration (written to `.env` and `dot.env.qwen38-27b-q5kxl-200k-32gb`)

This is the **RTX 5090 (32 GB) tuned profile**. The only change from the previous
`.env` was **`SPEC_DRAFT_N_MAX: 2 → 3`**. The dedicated profile
`dot.env.qwen38-27b-q5kxl-200k-32gb` captures the full best set (model,
200K ctx, PARALLEL=1, q8_0 KV, b1536/u384, threads 8/16/4, MTP n3, vision
on+offloaded). Switch with `sudo bash scripts/switch-model.sh qwen38-32gb`.

## Reliability note (tool-call validation)

`scripts/toolcall-stress.py --iterations 40` was run twice:

- **With the old 1024-token cap** (`--max-tokens 1024`): **31/40 OK, 9
  `bad_tool_args`** — all `finish=length` truncations. A `create_file` whose
  `content` is a large file plus some reasoning overflowed the harness's small
  1024-token per-turn budget, cutting the JSON string off mid-argument.
- **With the cap removed** (`--max-tokens 0` = no cap; the harness now defaults
  to 30000 so outputs up to ~30K tokens complete): **40/40 OK, 0 failures**
  (215 turns, 200 with tool calls). The model wrote full file bodies up to
  ~8,000 tokens with valid JSON and `truncated = 0` on every task.

**Conclusion:** the tool-call issue was the **harness's token cap**, not the
model or MTP. With an adequate output budget, Qwen3.8-27B Q5 @ MTP n3 produces
valid, complete tool calls with **no reasoning budget required**
(`REASONING_BUDGET` is left unrestricted). `REASONING_BUDGET` remains an optional
safeguard for real client paths that send a very tight cap, but it was not needed
here.

The harness was updated in `scripts/toolcall-stress.py`: default `--max-tokens`
is now **30000** (allow up to 30K outputs), and `--max-tokens 0` omits the cap
entirely (server decides). The replay path default was raised to 30000 as well.

## Artifacts

- `benchmarks/20260904-142011-agent.json` / `.md` — 50K MTP + vision sweep
- `benchmarks/20260904-143457-agent.json` / `.md` — MTP n2/n3/n4 confirmation
- `benchmarks/20260904-140538-mtp-benchmark.*` — short-context MTP sweep
- `benchmarks/20260904-144523-toolcall-stress` — tool-call stress artifacts
