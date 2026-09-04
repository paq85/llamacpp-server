#!/usr/bin/env python3
"""Benchmark an already-running OpenAI-compatible llama.cpp server.

Unlike the restart-based benchmark harnesses in this repo, this script assumes
the server is already up and reachable. It can benchmark one or more
concurrency levels, reports per-client plus aggregate throughput, and writes
JSON + Markdown artifacts into `benchmarks/`.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "benchmarks"
DEFAULT_API_KEY_FILE = ROOT / "api-keys.txt"
DEFAULT_URL_CANDIDATES = (
    "https://127.0.0.1:8080/v1",
    "http://127.0.0.1:8080/v1",
    "https://127.0.0.1:8082/v1",
    "http://127.0.0.1:8082/v1",
)


@dataclass(frozen=True)
class RequestMetrics:
    concurrency: int
    batch: int
    client: int
    seed: int
    wall_s: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_ms: float
    completion_ms: float
    prompt_tps: float
    completion_tps: float
    wall_tps: float
    finish_reason: str
    content_preview: str
    reasoning_preview: str


@dataclass(frozen=True)
class BatchMetrics:
    concurrency: int
    batch: int
    wall_s: float
    aggregate_prompt_tokens: int
    aggregate_cached_tokens: int
    aggregate_completion_tokens: int
    aggregate_total_tokens: int
    aggregate_completion_wall_tps: float
    aggregate_total_wall_tps: float
    client_prompt_tps_mean: float
    client_completion_tps_mean: float
    client_wall_tps_mean: float
    requests: tuple[RequestMetrics, ...]


@dataclass(frozen=True)
class ConcurrencySummary:
    concurrency: int
    batches: int
    requests: int
    prompt_tokens_mean: float
    cached_tokens_mean: float
    completion_tokens_mean: float
    client_prompt_tps_mean: float
    client_prompt_tps_stdev: float
    client_completion_tps_mean: float
    client_completion_tps_stdev: float
    client_wall_tps_mean: float
    client_wall_tps_stdev: float
    aggregate_completion_wall_tps_mean: float
    aggregate_completion_wall_tps_stdev: float
    aggregate_total_wall_tps_mean: float
    aggregate_total_wall_tps_stdev: float
    batch_wall_s_mean: float


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def join_url(base_url: str, path: str) -> str:
    return normalize_base_url(base_url) + path


def tls_context(verify_tls: bool) -> ssl.SSLContext:
    return ssl.create_default_context() if verify_tls else ssl._create_unverified_context()


def generate_prompt(approx_tokens: int) -> str:
    chunk = (
        "You are benchmarking a running inference server. Continue with a dense "
        "technical discussion of GPU inference, CUDA kernels, KV cache layout, "
        "memory bandwidth, scheduler overhead, prompt processing, concurrent "
        "clients, and decode throughput. Keep the prose compact, specific, and "
        "information-dense. "
    )
    repeats = max(1, approx_tokens // 40)
    body = " ".join(chunk for _ in range(repeats))
    return (
        "Benchmark prompt. Write a technical continuation about LLM serving, "
        "throughput, concurrency, and memory behavior.\n\n"
        f"{body}\n"
    )


def read_text_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    return generate_prompt(args.approx_prompt_tokens)


def parse_concurrency_levels(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise SystemExit("--concurrency must contain at least one positive integer")

    levels: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            level = int(part)
        except ValueError as exc:
            raise SystemExit(f"Invalid concurrency value: {part!r}") from exc
        if level < 1:
            raise SystemExit(f"Concurrency must be at least 1 (got {level})")
        if level not in seen:
            seen.add(level)
            levels.append(level)
    return levels


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key

    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key

    api_key_file = Path(args.api_key_file)
    if api_key_file.is_file():
        lines = api_key_file.read_text(encoding="utf-8").splitlines()
        if lines:
            return lines[0].strip()

    return ""


def request_json(
    base_url: str,
    path: str,
    *,
    verify_tls: bool,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {
        "User-Agent": "bench-running/1.0 (llamacpp-benchmark)",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    method = "GET" if payload is None else "POST"
    req = urllib.request.Request(join_url(base_url, path), data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=tls_context(verify_tls), timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def autodetect_base_url(args: argparse.Namespace, api_key: str) -> str:
    if args.base_url:
        return normalize_base_url(args.base_url)

    env_base_url = os.environ.get("OPENAI_BASE_URL")
    if env_base_url:
        return normalize_base_url(env_base_url)

    errors: list[str] = []
    for candidate in DEFAULT_URL_CANDIDATES:
        try:
            request_json(candidate, "/models", verify_tls=args.verify_tls, api_key=api_key, timeout=15.0)
            return candidate
        except urllib.error.HTTPError as exc:
            errors.append(f"{candidate}: HTTP {exc.code}")
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            errors.append(f"{candidate}: {exc}")

    raise SystemExit(
        "Could not auto-detect a running server. Pass --base-url explicitly. Tried:\n- "
        + "\n- ".join(errors or DEFAULT_URL_CANDIDATES)
    )


def discover_model(base_url: str, *, verify_tls: bool, api_key: str, explicit_model: str) -> str:
    if explicit_model:
        return explicit_model

    models = request_json(base_url, "/models", verify_tls=verify_tls, api_key=api_key, timeout=30.0)
    data = models.get("data") or []
    if data and isinstance(data[0], dict) and data[0].get("id"):
        return str(data[0]["id"])

    legacy_models = models.get("models") or []
    if legacy_models and isinstance(legacy_models[0], dict):
        for key in ("model", "name"):
            value = legacy_models[0].get(key)
            if value:
                return str(value)

    raise SystemExit("Could not discover a model from /v1/models. Pass --model explicitly.")


def decorate_prompt(prompt: str, request_tag: str, reuse_prompt: bool) -> str:
    if reuse_prompt:
        return prompt
    return f"Request nonce: {request_tag}\n\n{prompt}"


def build_messages(prompt: str, image_url: str | None) -> list[dict[str, Any]]:
    if not image_url:
        return [{"role": "user", "content": prompt}]

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_payload(
    args: argparse.Namespace,
    *,
    model: str,
    prompt: str,
    seed: int,
    request_tag: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_messages(decorate_prompt(prompt, request_tag, args.reuse_prompt), args.image_url),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": seed,
        "stream": False,
    }
    if not args.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def run_request(
    args: argparse.Namespace,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    concurrency: int,
    batch: int,
    client: int,
    seed: int,
) -> RequestMetrics:
    request_tag = f"concurrency={concurrency};batch={batch};client={client};seed={seed}"
    payload = build_payload(args, model=model, prompt=prompt, seed=seed, request_tag=request_tag)
    start = time.perf_counter()
    response = request_json(
        base_url,
        "/chat/completions",
        verify_tls=args.verify_tls,
        api_key=api_key,
        payload=payload,
        timeout=args.timeout,
    )
    wall_s = time.perf_counter() - start

    timings = response.get("timings") or {}
    usage = response.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    choices = response.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}

    prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    cached_tokens = int(prompt_details.get("cached_tokens") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0.0)
    completion_ms = float(timings.get("predicted_ms") or 0.0)
    prompt_tps = float(timings.get("prompt_per_second") or (prompt_tokens / (prompt_ms / 1000.0) if prompt_ms else 0.0))
    completion_tps = float(
        timings.get("predicted_per_second")
        or (completion_tokens / (completion_ms / 1000.0) if completion_ms else 0.0)
    )
    wall_tps = completion_tokens / wall_s if wall_s > 0 else 0.0

    return RequestMetrics(
        concurrency=concurrency,
        batch=batch,
        client=client,
        seed=seed,
        wall_s=wall_s,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_ms=prompt_ms,
        completion_ms=completion_ms,
        prompt_tps=prompt_tps,
        completion_tps=completion_tps,
        wall_tps=wall_tps,
        finish_reason=str((choices[0] or {}).get("finish_reason") or ""),
        content_preview=str(message.get("content") or "")[:120],
        reasoning_preview=str(message.get("reasoning_content") or "")[:120],
    )


def run_batch(
    args: argparse.Namespace,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    concurrency: int,
    batch: int,
    seed_base: int,
) -> BatchMetrics:
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="bench-running") as executor:
        futures = [
            executor.submit(
                run_request,
                args,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                concurrency=concurrency,
                batch=batch,
                client=client,
                seed=seed_base + client - 1,
            )
            for client in range(1, concurrency + 1)
        ]
        requests = tuple(future.result() for future in futures)
    batch_wall_s = time.perf_counter() - batch_start

    aggregate_prompt_tokens = sum(request.prompt_tokens for request in requests)
    aggregate_cached_tokens = sum(request.cached_tokens for request in requests)
    aggregate_completion_tokens = sum(request.completion_tokens for request in requests)
    aggregate_total_tokens = sum(request.total_tokens for request in requests)

    return BatchMetrics(
        concurrency=concurrency,
        batch=batch,
        wall_s=batch_wall_s,
        aggregate_prompt_tokens=aggregate_prompt_tokens,
        aggregate_cached_tokens=aggregate_cached_tokens,
        aggregate_completion_tokens=aggregate_completion_tokens,
        aggregate_total_tokens=aggregate_total_tokens,
        aggregate_completion_wall_tps=(aggregate_completion_tokens / batch_wall_s if batch_wall_s > 0 else 0.0),
        aggregate_total_wall_tps=(aggregate_total_tokens / batch_wall_s if batch_wall_s > 0 else 0.0),
        client_prompt_tps_mean=mean([request.prompt_tps for request in requests]),
        client_completion_tps_mean=mean([request.completion_tps for request in requests]),
        client_wall_tps_mean=mean([request.wall_tps for request in requests]),
        requests=requests,
    )


def summarize_batches(concurrency: int, batches: list[BatchMetrics]) -> ConcurrencySummary:
    requests = [request for batch in batches for request in batch.requests]
    return ConcurrencySummary(
        concurrency=concurrency,
        batches=len(batches),
        requests=len(requests),
        prompt_tokens_mean=mean([float(request.prompt_tokens) for request in requests]),
        cached_tokens_mean=mean([float(request.cached_tokens) for request in requests]),
        completion_tokens_mean=mean([float(request.completion_tokens) for request in requests]),
        client_prompt_tps_mean=mean([request.prompt_tps for request in requests]),
        client_prompt_tps_stdev=stdev([request.prompt_tps for request in requests]),
        client_completion_tps_mean=mean([request.completion_tps for request in requests]),
        client_completion_tps_stdev=stdev([request.completion_tps for request in requests]),
        client_wall_tps_mean=mean([request.wall_tps for request in requests]),
        client_wall_tps_stdev=stdev([request.wall_tps for request in requests]),
        aggregate_completion_wall_tps_mean=mean([batch.aggregate_completion_wall_tps for batch in batches]),
        aggregate_completion_wall_tps_stdev=stdev([batch.aggregate_completion_wall_tps for batch in batches]),
        aggregate_total_wall_tps_mean=mean([batch.aggregate_total_wall_tps for batch in batches]),
        aggregate_total_wall_tps_stdev=stdev([batch.aggregate_total_wall_tps for batch in batches]),
        batch_wall_s_mean=mean([batch.wall_s for batch in batches]),
    )


def render_markdown(
    *,
    timestamp: str,
    base_url: str,
    model: str,
    args: argparse.Namespace,
    levels: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Running-server benchmark {timestamp}",
        "",
        f"- base_url: `{base_url}`",
        f"- model: `{model}`",
        f"- concurrency levels: `{args.concurrency}`",
        f"- measured batches per level: `{args.runs}`",
        f"- warmup batches per level: `{args.warmup_runs}`",
        f"- approximate prompt tokens target: `{args.approx_prompt_tokens}`",
        f"- max completion tokens: `{args.max_tokens}`",
        f"- thinking enabled: `{args.enable_thinking}`",
        f"- image_url provided: `{bool(args.image_url)}`",
        f"- exact prompt reuse: `{args.reuse_prompt}`",
        "",
        "## Concurrency summary",
        "",
        "| clients | aggregate completion tok/s | aggregate total tok/s | avg client completion tok/s | avg client prompt tok/s | avg client wall tok/s | avg cached tok | requests |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for level in levels:
        summary: ConcurrencySummary = level["summary"]
        lines.append(
            f"| {summary.concurrency}"
            f" | {summary.aggregate_completion_wall_tps_mean:.2f} ± {summary.aggregate_completion_wall_tps_stdev:.2f}"
            f" | {summary.aggregate_total_wall_tps_mean:.2f} ± {summary.aggregate_total_wall_tps_stdev:.2f}"
            f" | {summary.client_completion_tps_mean:.2f} ± {summary.client_completion_tps_stdev:.2f}"
            f" | {summary.client_prompt_tps_mean:.2f} ± {summary.client_prompt_tps_stdev:.2f}"
            f" | {summary.client_wall_tps_mean:.2f} ± {summary.client_wall_tps_stdev:.2f}"
            f" | {summary.cached_tokens_mean:.1f}"
            f" | {summary.requests} |"
        )

    for level in levels:
        summary = level["summary"]
        batches: list[BatchMetrics] = level["batches"]
        lines.extend(
            [
                "",
                f"## {summary.concurrency} concurrent client{'s' if summary.concurrency != 1 else ''}",
                "",
                "| batch | aggregate completion tok/s | aggregate total tok/s | batch wall s | avg client completion tok/s | avg client prompt tok/s | avg client wall tok/s | completion tok | cached tok |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for batch in batches:
            lines.append(
                f"| {batch.batch}"
                f" | {batch.aggregate_completion_wall_tps:.2f}"
                f" | {batch.aggregate_total_wall_tps:.2f}"
                f" | {batch.wall_s:.3f}"
                f" | {batch.client_completion_tps_mean:.2f}"
                f" | {batch.client_prompt_tps_mean:.2f}"
                f" | {batch.client_wall_tps_mean:.2f}"
                f" | {batch.aggregate_completion_tokens}"
                f" | {batch.aggregate_cached_tokens} |"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="",
        help="OpenAI-compatible base URL, for example https://127.0.0.1:8080/v1. Default: auto-detect common local URLs or use OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model ID to benchmark. Default: auto-discover first entry from /v1/models.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer token. Default: OPENAI_API_KEY, then api-keys.txt if present, else no auth header.",
    )
    parser.add_argument(
        "--api-key-file",
        default=str(DEFAULT_API_KEY_FILE),
        help="File used when --api-key and OPENAI_API_KEY are unset. Default: %(default)s",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2",
        help="Comma-separated concurrency levels to test. Default: %(default)s",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured concurrent batches per concurrency level. Default: %(default)s",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup concurrent batches per concurrency level. Default: %(default)s",
    )
    parser.add_argument(
        "--approx-prompt-tokens",
        type=int,
        default=4096,
        help="Approximate size of the generated text prompt when --prompt/--prompt-file are not provided. Default: %(default)s",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Completion tokens requested from the server. Default: %(default)s",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature. Default: %(default)s")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p. Default: %(default)s")
    parser.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Base seed; each request derives its own seed from this value. Default: %(default)s",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-request timeout in seconds. Default: %(default)s",
    )
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        help="Verify HTTPS certificates. Default is insecure/self-signed-friendly for local serving.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave Qwen thinking enabled. Default disables thinking for cleaner throughput measurements.",
    )
    parser.add_argument(
        "--reuse-prompt",
        action="store_true",
        help="Reuse the exact same prompt for every request. Default prepends a small nonce to each request to reduce prompt-cache contamination.",
    )
    parser.add_argument(
        "--image-url",
        default="",
        help="Optional image URL or data URL to benchmark the multimodal path.",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="", help="Use this exact text prompt instead of the generated benchmark prompt.")
    prompt_group.add_argument("--prompt-file", default="", help="Read the prompt from a UTF-8 text file instead of generating one.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")

    concurrency_levels = parse_concurrency_levels(args.concurrency)
    api_key = resolve_api_key(args)
    base_url = autodetect_base_url(args, api_key)
    model = discover_model(base_url, verify_tls=args.verify_tls, api_key=api_key, explicit_model=args.model)
    prompt = read_text_prompt(args)

    print(f"Benchmarking running server at {base_url} with model {model}")
    print(
        f"Prompt chars={len(prompt)} concurrency={concurrency_levels} "
        f"warmup={args.warmup_runs} measured={args.runs} max_tokens={args.max_tokens}"
    )

    levels: list[dict[str, Any]] = []
    for concurrency in concurrency_levels:
        print(f"\n== concurrency {concurrency} ==")
        for warmup_index in range(args.warmup_runs):
            run_batch(
                args,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                concurrency=concurrency,
                batch=-(warmup_index + 1),
                seed_base=args.seed + (concurrency * 1_000_000) + 900_000_000 + (warmup_index * 1_000),
            )

        batches = [
            run_batch(
                args,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                concurrency=concurrency,
                batch=batch_index,
                seed_base=args.seed + (concurrency * 1_000_000) + (batch_index * 1_000),
            )
            for batch_index in range(1, args.runs + 1)
        ]
        summary = summarize_batches(concurrency, batches)
        levels.append({"concurrency": concurrency, "summary": summary, "batches": batches})
        print(
            "summary: aggregate_completion_tps={:.2f} aggregate_total_tps={:.2f} "
            "client_completion_tps={:.2f} client_prompt_tps={:.2f} avg_cached_tokens={:.1f}".format(
                summary.aggregate_completion_wall_tps_mean,
                summary.aggregate_total_wall_tps_mean,
                summary.client_completion_tps_mean,
                summary.client_prompt_tps_mean,
                summary.cached_tokens_mean,
            )
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    markdown = render_markdown(timestamp=timestamp, base_url=base_url, model=model, args=args, levels=levels)
    output = {
        "timestamp": timestamp,
        "base_url": base_url,
        "model": model,
        "args": vars(args),
        "levels": [
            {
                "concurrency": level["concurrency"],
                "summary": asdict(level["summary"]),
                "batches": [asdict(batch) for batch in level["batches"]],
            }
            for level in levels
        ],
        "markdown": markdown,
    }

    json_path = RESULTS_DIR / f"{timestamp}-running-benchmark.json"
    md_path = RESULTS_DIR / f"{timestamp}-running-benchmark.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
