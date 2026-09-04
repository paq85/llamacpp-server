#!/usr/bin/env python3
"""Direct llama.cpp server benchmark — hits the API without any proxy layer.

Tests:
  1. Prompt throughput (prefill speed) at various prompt lengths
  2. Decode throughput (token generation speed) at various output lengths
  3. Concurrent request handling at various concurrency levels
  4. KV cache hit performance (repeated prompts)

Outputs JSON + Markdown to benchmarks/ directory.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "benchmarks"

# Direct llama.cpp server (no proxy)
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


# ── helpers ──────────────────────────────────────────────────────────────────

def mean_val(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev_val(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def post_json(base_url: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        join_url(base_url, path),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def get_json(base_url: str, path: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(join_url(base_url, path))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def generate_text(approx_tokens: int) -> str:
    """Generate a prompt that tokenizes to roughly approx_tokens."""
    chunk = (
        "The transformer architecture has become the dominant paradigm in modern "
        "natural language processing, replacing recurrent and convolutional "
        "networks for most sequence modeling tasks. At its core, the transformer "
        "relies on self-attention mechanisms that compute pairwise interactions "
        "between all positions in a sequence, enabling the model to capture long-"
        "range dependencies without the sequential computation bottleneck of RNNs. "
        "The multi-head attention mechanism allows the model to jointly attend to "
        "information from different representation subspaces at different positions. "
    )
    repeats = max(1, approx_tokens // 40)
    return " ".join(chunk for _ in range(repeats))


# ── test sections ────────────────────────────────────────────────────────────

@dataclass
class ServerInfo:
    model: str
    n_ctx: int
    n_params: str
    ftype: str
    fingerprint: str


def get_server_info(base_url: str) -> ServerInfo:
    models = get_json(base_url, "/models")
    data = models.get("data", [])
    meta = (data[0] or {}).get("meta", {}) if data else {}
    details = (data[0] or {}).get("details", {}) if data else {}
    return ServerInfo(
        model=str((data[0] or {}).get("id", "unknown")),
        n_ctx=int(meta.get("n_ctx", 0)),
        n_params=str(details.get("parameter_size", meta.get("n_params", "unknown"))),
        ftype=str(details.get("ftype", "unknown")),
        fingerprint="",
    )


@dataclass
class PromptThroughputResult:
    prompt_tokens: int
    max_output_tokens: int
    runs: int
    prompt_tps_mean: float
    prompt_tps_stdev: float
    prompt_ms_mean: float
    prompt_ms_stdev: float
    first_token_ms_mean: float
    first_token_ms_stdev: float
    wall_s_mean: float


def bench_prompt_throughput(
    base_url: str, model: str, prompt_lengths: list[int], runs: int, max_tokens: int
) -> list[PromptThroughputResult]:
    """Measure prefill speed at various prompt lengths."""
    results = []
    for ptokens in prompt_lengths:
        prompt = generate_text(ptokens)
        prompt_tps_list = []
        prompt_ms_list = []
        first_token_ms_list = []
        wall_s_list = []

        for run_idx in range(runs):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": f"Run {run_idx}: {prompt}"}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 42 + run_idx,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            t0 = time.perf_counter()
            resp = post_json(base_url, "/chat/completions", payload)
            wall_s = time.perf_counter() - t0

            timings = resp.get("timings", {})
            prompt_tps_list.append(float(timings.get("prompt_per_second", 0)))
            prompt_ms_list.append(float(timings.get("prompt_ms", 0)))
            # first token time ≈ prompt_ms + first predicted token
            wall_s_list.append(wall_s)
            # approximate first token latency as total wall time for small outputs
            first_token_ms_list.append(wall_s * 1000)

        actual_prompt_n = int(mean_val(prompt_ms_list) / 1000 * mean_val(prompt_tps_list)) if mean_val(prompt_ms_list) > 0 else 0

        results.append(PromptThroughputResult(
            prompt_tokens=actual_prompt_n or ptokens,
            max_output_tokens=max_tokens,
            runs=runs,
            prompt_tps_mean=round(mean_val(prompt_tps_list), 2),
            prompt_tps_stdev=round(stdev_val(prompt_tps_list), 2),
            prompt_ms_mean=round(mean_val(prompt_ms_list), 2),
            prompt_ms_stdev=round(stdev_val(prompt_ms_list), 2),
            first_token_ms_mean=round(mean_val(first_token_ms_list), 2),
            first_token_ms_stdev=round(stdev_val(first_token_ms_list), 2),
            wall_s_mean=round(mean_val(wall_s_list), 4),
        ))
    return results


@dataclass
class DecodeThroughputResult:
    prompt_tokens: int
    output_tokens: int
    runs: int
    decode_tps_mean: float
    decode_tps_stdev: float
    decode_ms_mean: float
    decode_ms_stdev: float
    wall_tps_mean: float


def bench_decode_throughput(
    base_url: str, model: str, output_lengths: list[int], runs: int, prompt_tokens: int
) -> list[DecodeThroughputResult]:
    """Measure decode speed at various output lengths with a fixed prompt."""
    prompt = generate_text(prompt_tokens)
    results = []

    for out_len in output_lengths:
        decode_tps_list = []
        decode_ms_list = []
        wall_tps_list = []
        actual_out = []

        for run_idx in range(runs):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": f"Run {run_idx}: {prompt}"}],
                "max_tokens": out_len,
                "temperature": 0.0,
                "seed": 42 + run_idx,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            t0 = time.perf_counter()
            resp = post_json(base_url, "/chat/completions", payload)
            wall_s = time.perf_counter() - t0

            timings = resp.get("timings", {})
            usage = resp.get("usage", {})
            comp_n = int(usage.get("completion_tokens", 0))
            decode_tps_list.append(float(timings.get("predicted_per_second", 0)))
            decode_ms_list.append(float(timings.get("predicted_ms", 0)))
            wall_tps_list.append(comp_n / wall_s if wall_s > 0 else 0)
            actual_out.append(comp_n)

        results.append(DecodeThroughputResult(
            prompt_tokens=prompt_tokens,
            output_tokens=int(mean_val(actual_out)),
            runs=runs,
            decode_tps_mean=round(mean_val(decode_tps_list), 2),
            decode_tps_stdev=round(stdev_val(decode_tps_list), 2),
            decode_ms_mean=round(mean_val(decode_ms_list), 2),
            decode_ms_stdev=round(stdev_val(decode_ms_list), 2),
            wall_tps_mean=round(mean_val(wall_tps_list), 2),
        ))
    return results


@dataclass
class ConcurrencyResult:
    concurrency: int
    num_requests: int
    wall_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    aggregate_prompt_tps: float
    aggregate_decode_tps: float
    aggregate_wall_tps: float
    per_request_decode_tps_mean: float
    per_request_decode_tps_stdev: float
    p50_wall_s: float
    p95_wall_s: float
    p99_wall_s: float


def bench_concurrency(
    base_url: str, model: str, concurrency_levels: list[int], runs_per_level: int,
    prompt_tokens: int, max_output_tokens: int
) -> list[ConcurrencyResult]:
    """Measure throughput under concurrent load."""
    prompt = generate_text(prompt_tokens)
    results = []

    for conc in concurrency_levels:
        all_decode_tps = []
        all_wall_s = []
        total_prompt_t = 0
        total_comp_t = 0

        def make_request(req_idx: int) -> dict:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": f"Conc={conc} Req={req_idx}: {prompt}"}],
                "max_tokens": max_output_tokens,
                "temperature": 0.0,
                "seed": 1000 + req_idx,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            t0 = time.perf_counter()
            resp = post_json(base_url, "/chat/completions", payload)
            wall = time.perf_counter() - t0
            timings = resp.get("timings", {})
            usage = resp.get("usage", {})
            return {
                "wall_s": wall,
                "prompt_tps": float(timings.get("prompt_per_second", 0)),
                "decode_tps": float(timings.get("predicted_per_second", 0)),
                "prompt_n": int(usage.get("prompt_tokens", 0)),
                "comp_n": int(usage.get("completion_tokens", 0)),
            }

        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [pool.submit(make_request, i) for i in range(runs_per_level)]
            for f in as_completed(futures):
                r = f.result()
                all_decode_tps.append(r["decode_tps"])
                all_wall_s.append(r["wall_s"])
                total_prompt_t += r["prompt_n"]
                total_comp_t += r["comp_n"]
        wall_total = time.perf_counter() - t_start

        sorted_wall = sorted(all_wall_s)
        n = len(sorted_wall)

        results.append(ConcurrencyResult(
            concurrency=conc,
            num_requests=runs_per_level,
            wall_s=round(wall_total, 3),
            total_prompt_tokens=total_prompt_t,
            total_completion_tokens=total_comp_t,
            aggregate_prompt_tps=round(total_prompt_t / wall_total if wall_total else 0, 2),
            aggregate_decode_tps=round(total_comp_t / wall_total if wall_total else 0, 2),
            aggregate_wall_tps=round((total_prompt_t + total_comp_t) / wall_total if wall_total else 0, 2),
            per_request_decode_tps_mean=round(mean_val(all_decode_tps), 2),
            per_request_decode_tps_stdev=round(stdev_val(all_decode_tps), 2),
            p50_wall_s=round(sorted_wall[n // 2], 3),
            p95_wall_s=round(sorted_wall[int(n * 0.95)], 3),
            p99_wall_s=round(sorted_wall[min(int(n * 0.99), n - 1)], 3),
        ))
    return results


@dataclass
class CacheResult:
    prompt_tokens: int
    cold_prompt_tps: float
    cold_prompt_ms: float
    cold_first_token_ms: float
    warm_prompt_tps: float
    warm_prompt_ms: float
    warm_first_token_ms: float
    speedup_tps: float
    speedup_ms: float


def bench_kv_cache(
    base_url: str, model: str, prompt_tokens: int, max_output_tokens: int
) -> CacheResult:
    """Compare cold vs warm (cached) prompt processing."""
    prompt = generate_text(prompt_tokens)

    # Cold run
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    cold_resp = post_json(base_url, "/chat/completions", payload)
    cold_wall = time.perf_counter() - t0
    cold_timings = cold_resp.get("timings", {})

    # Warm run (same prompt, should hit cache)
    t0 = time.perf_counter()
    warm_resp = post_json(base_url, "/chat/completions", payload)
    warm_wall = time.perf_counter() - t0
    warm_timings = warm_resp.get("timings", {})

    cold_tps = float(cold_timings.get("prompt_per_second", 0))
    warm_tps = float(warm_timings.get("prompt_per_second", 0))

    return CacheResult(
        prompt_tokens=prompt_tokens,
        cold_prompt_tps=round(cold_tps, 2),
        cold_prompt_ms=round(float(cold_timings.get("prompt_ms", 0)), 2),
        cold_first_token_ms=round(cold_wall * 1000, 2),
        warm_prompt_tps=round(warm_tps, 2),
        warm_prompt_ms=round(float(warm_timings.get("prompt_ms", 0)), 2),
        warm_first_token_ms=round(warm_wall * 1000, 2),
        speedup_tps=round(cold_tps / warm_tps if warm_tps else float("inf"), 2),
        speedup_ms=round(float(cold_timings.get("prompt_ms", 0)) / float(warm_timings.get("prompt_ms", 0)) if warm_timings.get("prompt_ms") else float("inf"), 2),
    )


# ── markdown report ──────────────────────────────────────────────────────────

def render_markdown(
    server: ServerInfo,
    prompt_results: list[PromptThroughputResult],
    decode_results: list[DecodeThroughputResult],
    conc_results: list[ConcurrencyResult],
    cache_result: CacheResult,
) -> str:
    lines = []
    lines.append("# Direct llama.cpp Server Benchmark")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Server**: `{DEFAULT_BASE_URL}` (direct, no proxy)")
    lines.append(f"**Model**: `{server.model}`")
    lines.append(f"**Parameters**: {server.n_params}")
    lines.append(f"**Quantization**: {server.ftype}")
    lines.append(f"**Context**: {server.n_ctx:,} tokens")
    lines.append("")

    # Prompt throughput
    lines.append("## 1. Prompt Throughput (Prefill Speed)")
    lines.append("")
    lines.append("| Prompt Tokens | Prompt TPS (mean ± σ) | Prompt MS (mean ± σ) | First Token MS (mean ± σ) |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for r in prompt_results:
        lines.append(
            f"| {r.prompt_tokens:,} | {r.prompt_tps_mean:,} ± {r.prompt_tps_stdev:,} "
            f"| {r.prompt_ms_mean:.1f} ± {r.prompt_ms_stdev:.1f} "
            f"| {r.first_token_ms_mean:.1f} ± {r.first_token_ms_stdev:.1f} |"
        )
    lines.append("")

    # Decode throughput
    lines.append("## 2. Decode Throughput (Generation Speed)")
    lines.append("")
    lines.append("| Output Tokens | Decode TPS (mean ± σ) | Decode MS (mean ± σ) | Wall TPS |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for r in decode_results:
        lines.append(
            f"| {r.output_tokens:,} | {r.decode_tps_mean:,} ± {r.decode_tps_stdev:,} "
            f"| {r.decode_ms_mean:.1f} ± {r.decode_ms_stdev:.1f} "
            f"| {r.wall_tps_mean:.2f} |"
        )
    lines.append("")

    # Concurrency
    lines.append("## 3. Concurrency")
    lines.append("")
    lines.append("| Concurrency | Requests | Wall (s) | Aggregate Decode TPS | Per-Req Decode TPS (mean ± σ) | P50 (s) | P95 (s) | P99 (s) |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in conc_results:
        lines.append(
            f"| {r.concurrency} | {r.num_requests} | {r.wall_s:.2f} "
            f"| {r.aggregate_decode_tps:,.1f} "
            f"| {r.per_request_decode_tps_mean:,.1f} ± {r.per_request_decode_tps_stdev:,.1f} "
            f"| {r.p50_wall_s:.2f} | {r.p95_wall_s:.2f} | {r.p99_wall_s:.2f} |"
        )
    lines.append("")

    # KV Cache
    lines.append("## 4. KV Cache (Cold vs Warm)")
    lines.append("")
    lines.append(f"**Prompt length**: {cache_result.prompt_tokens:,} tokens")
    lines.append("")
    lines.append("| Phase | Prompt TPS | Prompt MS | First Token MS |")
    lines.append("|:---|:---:|:---:|:---:|")
    lines.append(
        f"| Cold | {cache_result.cold_prompt_tps:,.1f} "
        f"| {cache_result.cold_prompt_ms:.1f} "
        f"| {cache_result.cold_first_token_ms:.1f} |"
    )
    lines.append(
        f"| Warm | {cache_result.warm_prompt_tps:,.1f} "
        f"| {cache_result.warm_prompt_ms:.1f} "
        f"| {cache_result.warm_first_token_ms:.1f} |"
    )
    lines.append("")
    lines.append(f"**Speedup (TPS)**: {cache_result.speedup_tps:.2f}x")
    lines.append(f"**Speedup (MS)**: {cache_result.speedup_ms:.2f}x")
    lines.append("")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp server directly")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--model", default="", help="Model name (auto-detected if empty)")
    parser.add_argument("--prompt-lengths", default="256,1024,4096,8192,16384",
                        help="Comma-separated prompt lengths for prefill test")
    parser.add_argument("--output-lengths", default="32,128,256,512,1024",
                        help="Comma-separated output lengths for decode test")
    parser.add_argument("--concurrency-levels", default="1,2,4,8",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--runs", type=int, default=3, help="Runs per measurement")
    parser.add_argument("--conc-runs", type=int, default=4, help="Requests per concurrency level")
    parser.add_argument("--cache-prompt-tokens", type=int, default=4096,
                        help="Prompt length for KV cache test")
    parser.add_argument("--max-output", type=int, default=64,
                        help="Max output tokens for prefill test")
    parser.add_argument("--decode-prompt-tokens", type=int, default=512,
                        help="Prompt length for decode throughput test")
    args = parser.parse_args()

    # Parse comma-separated lists
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]
    output_lengths = [int(x) for x in args.output_lengths.split(",")]
    conc_levels = [int(x) for x in args.concurrency_levels.split(",")]

    # Discover server
    print("=" * 60)
    print("Direct llama.cpp Server Benchmark")
    print("=" * 60)
    print(f"Server: {args.base_url}")
    server = get_server_info(args.base_url)
    model = args.model or server.model
    print(f"Model:  {server.model}")
    print(f"Params: {server.n_params}  |  Quant: {server.ftype}  |  Context: {server.n_ctx:,}")
    print()

    results: dict[str, Any] = {"server": asdict(server), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # 1. Prompt throughput
    print("[1/4] Prompt throughput (prefill speed)...")
    prompt_results = bench_prompt_throughput(args.base_url, model, prompt_lengths, args.runs, args.max_output)
    for r in prompt_results:
        print(f"  {r.prompt_tokens:>6,} tokens → {r.prompt_tps_mean:>8,.1f} TPS  ({r.prompt_ms_mean:.0f} ms prefill  |  {r.first_token_ms_mean:.0f} ms first token)")
    results["prompt_throughput"] = [asdict(r) for r in prompt_results]
    print()

    # 2. Decode throughput
    print("[2/4] Decode throughput (generation speed)...")
    decode_results = bench_decode_throughput(args.base_url, model, output_lengths, args.runs, args.decode_prompt_tokens)
    for r in decode_results:
        print(f"  {r.output_tokens:>5,} output → {r.decode_tps_mean:>8,.1f} TPS  ({r.decode_ms_mean:.0f} ms decode)")
    results["decode_throughput"] = [asdict(r) for r in decode_results]
    print()

    # 3. Concurrency
    print(f"[3/4] Concurrency ({conc_levels})...")
    conc_results = bench_concurrency(args.base_url, model, conc_levels, args.conc_runs, args.decode_prompt_tokens, 256)
    for r in conc_results:
        print(f"  conc={r.concurrency:>2}  →  {r.aggregate_decode_tps:>8,.1f} agg decode TPS  |  per-req: {r.per_request_decode_tps_mean:,.1f} ± {r.per_request_decode_tps_stdev:,.1f}  |  P50={r.p50_wall_s:.1f}s  P95={r.p95_wall_s:.1f}s")
    results["concurrency"] = [asdict(r) for r in conc_results]
    print()

    # 4. KV cache
    print(f"[4/4] KV cache (cold vs warm, {args.cache_prompt_tokens} tokens)...")
    cache_result = bench_kv_cache(args.base_url, model, args.cache_prompt_tokens, 64)
    print(f"  Cold: {cache_result.cold_prompt_tps:,.1f} TPS ({cache_result.cold_prompt_ms:.0f} ms)  |  First token: {cache_result.cold_first_token_ms:.0f} ms")
    print(f"  Warm: {cache_result.warm_prompt_tps:,.1f} TPS ({cache_result.warm_prompt_ms:.0f} ms)  |  First token: {cache_result.warm_first_token_ms:.0f} ms")
    print(f"  Speedup: {cache_result.speedup_tps:.1f}x (TPS)  /  {cache_result.speedup_ms:.1f}x (MS)")
    results["kv_cache"] = asdict(cache_result)

    # Write results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = RESULTS_DIR / f"{timestamp}-direct-benchmark.json"
    md_path = RESULTS_DIR / f"{timestamp}-direct-benchmark.md"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = render_markdown(server, prompt_results, decode_results, conc_results, cache_result)
    md_path.write_text(md, encoding="utf-8")

    print()
    print(f"Results → {json_path}")
    print(f"Report  → {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()