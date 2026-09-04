#!/usr/bin/env python3
"""Benchmark llama.cpp serving knobs (KV cache type, threads, flash-attn,
batch/ubatch, fit, cache-RAM) for single-user decode/prefill throughput on the
PAQ_LLAMACPP_SERVER/Qwen3.8-27B-Q5 host. Proxy is disabled per variant (CLOUDFLARE_TIMEOUT_PROXY_MODE=off).

For each variant we restart the server via run-paq-llamacpp-server.sh with the variant env
overrides, then issue a prompt to measure prompt-eval tok/s and a small prompt
with ignore_eos to measure decode tok/s.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run-paq-llamacpp-server.sh"
API_KEY_FILE = ROOT / "api-keys.txt"
RESULTS_DIR = ROOT / "benchmarks"
SYSTEMD_UNIT = "paq-llamacpp-server.service"


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]


def variants_default() -> list[Variant]:
    return [
        # baseline = current .env (q8_0/q8_0 KV, b1536/u384, threads 8/16/4, flash-attn on,
        # fit off, cache-ram 50000, MTP draft-mtp n_max=2, mmproj on)
        Variant("baseline", {}),
        # KV cache data type
        Variant("kv_f16", {"CACHE_TYPE_K": "f16", "CACHE_TYPE_V": "f16"}),
        Variant("kv_bf16", {"CACHE_TYPE_K": "bf16", "CACHE_TYPE_V": "bf16"}),
        # CPU threads
        Variant("threads_batch24", {"THREADS_BATCH": "24"}),
        Variant("threads_batch32", {"THREADS_BATCH": "32"}),
        Variant("threads16", {"THREADS": "16"}),
        # Flash attention mode
        Variant("fa_auto", {"FLASH_ATTN": "auto"}),
        # batch / ubatch
        Variant("b2048_u384", {"BATCH_SIZE": "2048", "UBATCH_SIZE": "384"}),
        Variant("b2048_u512", {"BATCH_SIZE": "2048", "UBATCH_SIZE": "512"}),
        # fit / KV offload RAM budget
        Variant("fit_on", {"FIT": "on"}),
        Variant("cache_ram_30000", {"CACHE_RAM_MIB": "30000"}),
        Variant("cache_ram_8000", {"CACHE_RAM_MIB": "8000"}),
    ]


def read_api_key() -> str:
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError) as exc:
        raise SystemExit(f"Could not read first API key from {API_KEY_FILE}: {exc}") from exc


def https_json(base_url: str, path: str, api_key: str, payload: dict[str, Any] | None = None, timeout: float = 900.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    method = "GET"
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def query_gpu() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def systemd_service_is_active(unit: str) -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def log_reader(proc: subprocess.Popen[str], out_path: Path, lines: "queue.Queue[str]") -> None:
    with out_path.open("w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            lines.put(line.rstrip("\n"))


def wait_for_server(proc: subprocess.Popen[str], lines: "queue.Queue[str]", timeout_s: float) -> list[str]:
    deadline = time.monotonic() + timeout_s
    startup: list[str] = []
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            line = lines.get(timeout=max(0.1, min(2.0, deadline - time.monotonic())))
        except queue.Empty:
            continue
        startup.append(line)
        if ("server is listening" in line) or ("listening on http" in line):
            return startup
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s")


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.kill(proc.pid, signal.SIGKILL)


def big_prompt(approx_tokens: int) -> str:
    chunk = (
        "GPU inference benchmarking paragraph. The CUDA scheduler dispatches kernels "
        "across streaming multiprocessors, evaluating KV cache reads, attention "
        "projections, and feed-forward matmuls. Memory bandwidth and tensor-core "
        "utilization determine the prompt processing throughput, while decoding is "
        "predominantly bandwidth-bound by KV cache reads at long contexts. "
    )
    repeats = max(1, approx_tokens // 60 + 1)
    return chunk * repeats


def run_pp(base_url: str, api_key: str, prompt: str, seed: int) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": False,
        "stream": False,
    }
    start = time.perf_counter()
    resp = https_json(base_url, "/completion", api_key, payload, timeout=900.0)
    wall = time.perf_counter() - start
    t = resp.get("timings", {}) or {}
    return {
        "wall_s": wall,
        "prompt_n": float(t.get("prompt_n") or 0),
        "prompt_ms": float(t.get("prompt_ms") or 0),
        "prompt_tps": float(t.get("prompt_per_second") or 0),
        "predicted_n": float(t.get("predicted_n") or 0),
        "predicted_tps": float(t.get("predicted_per_second") or 0),
    }


def run_tg(base_url: str, api_key: str, seed: int, n_predict: int) -> dict[str, Any]:
    payload = {
        "prompt": (
            "You are benchmarking decode tok/s. Continue with a dense technical "
            "paragraph about GPU inference, CUDA kernels, KV cache, memory "
            f"bandwidth, scheduler overhead, and speculative decoding. Seed: {seed}.\n\n"
        ),
        "n_predict": n_predict,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    start = time.perf_counter()
    resp = https_json(base_url, "/completion", api_key, payload, timeout=900.0)
    wall = time.perf_counter() - start
    t = resp.get("timings", {}) or {}
    draft_n = float(t.get("draft_n") or 0)
    draft_acc = float(t.get("draft_n_accepted") or 0)
    return {
        "wall_s": wall,
        "predicted_n": float(t.get("predicted_n") or 0),
        "predicted_tps": float(t.get("predicted_per_second") or 0),
        "draft_acceptance": draft_acc / draft_n if draft_n else 0.0,
    }


def run_variant(variant: Variant, args, api_key: str, timestamp: str) -> dict[str, Any]:
    port = args.port
    base_url = f"http://127.0.0.1:{port}"
    log_path = RESULTS_DIR / f"{timestamp}-tune-{variant.name}.log"
    env = os.environ.copy()
    env.update({
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "WARMUP": "0",
        "PROMPT_CACHE": "0",
        "CACHE_IDLE_SLOTS": "0",
        "CLOUDFLARED_ENABLED": "off",
        "CLOUDFLARE_TIMEOUT_PROXY_MODE": "off",
        "LLAMA_SERVER_SSL_KEY_FILE": "",
        "LLAMA_SERVER_SSL_CERT_FILE": "",
    })
    env.update(variant.env)

    lines: "queue.Queue[str]" = queue.Queue()
    proc = subprocess.Popen(
        [str(RUNNER)], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    reader = threading.Thread(target=log_reader, args=(proc, log_path, lines), daemon=True)
    reader.start()
    try:
        t0 = time.perf_counter()
        startup = wait_for_server(proc, lines, args.startup_timeout)
        startup_s = time.perf_counter() - t0
        https_json(base_url, "/health", api_key, timeout=60.0)

        prompt = big_prompt(args.pp_tokens)

        # Warmup (small): primes kernels.
        run_pp(base_url, api_key, big_prompt(256), seed=args.seed + 9000)

        pp_runs = [run_pp(base_url, api_key, prompt, seed=args.seed + 100 + i) for i in range(args.pp_runs)]
        tg_runs = [run_tg(base_url, api_key, seed=args.seed + 200 + i, n_predict=args.tg_tokens) for i in range(args.tg_runs)]

        pp_tps = [r["prompt_tps"] for r in pp_runs if r["prompt_tps"]]
        tg_tps = [r["predicted_tps"] for r in tg_runs if r["predicted_tps"]]
        return {
            "variant": variant.name,
            "env": variant.env,
            "startup_s": startup_s,
            "gpu_after": query_gpu(),
            "log_path": str(log_path),
            "pp_runs": pp_runs,
            "tg_runs": tg_runs,
            "summary": {
                "pp_tps_mean": statistics.fmean(pp_tps) if pp_tps else 0.0,
                "pp_tps_min": min(pp_tps) if pp_tps else 0.0,
                "pp_tokens": pp_runs[0]["prompt_n"] if pp_runs else 0,
                "tg_tps_mean": statistics.fmean(tg_tps) if tg_tps else 0.0,
                "tg_acceptance_mean": statistics.fmean([r["draft_acceptance"] for r in tg_runs if r["draft_acceptance"]]) if tg_runs else 0.0,
            },
            "startup_excerpt": startup[-15:],
        }
    finally:
        stop_server(proc)
        reader.join(timeout=5)


def render_markdown(results: list[dict[str, Any]]) -> str:
    rows = sorted(results, key=lambda r: r["summary"]["tg_tps_mean"], reverse=True)
    lines = [
        "| rank | variant | decode tok/s | prompt tok/s | draft accept | pp tokens | gpu mem used MiB |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, r in enumerate(rows, 1):
        s = r["summary"]
        used = (r.get("gpu_after") or "").split(",")[0].strip()
        lines.append(
            f"| {rank} | `{r['variant']}` | {s['tg_tps_mean']:.2f} | {s['pp_tps_mean']:.1f} | "
            f"{s['tg_acceptance_mean']:.2%} | {s['pp_tokens']:.0f} | {used} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18086)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--pp-tokens", type=int, default=16000)
    p.add_argument("--pp-runs", type=int, default=1)
    p.add_argument("--tg-tokens", type=int, default=1024)
    p.add_argument("--tg-runs", type=int, default=2)
    p.add_argument("--startup-timeout", type=float, default=300.0)
    p.add_argument("--variants", default="", help="Comma list of variant names; default all")
    args = p.parse_args()

    if systemd_service_is_active(SYSTEMD_UNIT):
        raise SystemExit(
            f"{SYSTEMD_UNIT} is active. Stop the systemd-managed server before running tuning benchmarks "
            "so run-paq-llamacpp-server.sh does not contend with the live instance."
        )

    api_key = read_api_key()
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    all_v = {v.name: v for v in variants_default()}
    if args.variants:
        names = [n.strip() for n in args.variants.split(",") if n.strip()]
        bad = [n for n in names if n not in all_v]
        if bad:
            raise SystemExit(f"Unknown variants: {bad}. Known: {list(all_v)}")
        chosen = [all_v[n] for n in names]
    else:
        chosen = list(all_v.values())

    print(f"Benchmarking {len(chosen)} variants on port {args.port}: pp={args.pp_tokens} tg={args.tg_tokens}x{args.tg_runs}")
    results: list[dict[str, Any]] = []
    for v in chosen:
        print(f"\n== {v.name} ==  env={v.env}")
        try:
            r = run_variant(v, args, api_key, timestamp)
        except (RuntimeError, TimeoutError) as exc:
            print(f"  FAILED: {exc}")
            results.append({"variant": v.name, "env": v.env, "error": str(exc), "summary": {}})
            continue
        results.append(r)
        s = r["summary"]
        print(
            "decode_tps={:.2f} prompt_tps={:.1f} accept={:.2%} pp_tokens={:.0f} gpu='{}'".format(
                s["tg_tps_mean"], s["pp_tps_mean"], s["tg_acceptance_mean"],
                s["pp_tokens"], r.get("gpu_after"),
            )
        )

    output = {
        "timestamp": timestamp,
        "port": args.port,
        "pp_tokens": args.pp_tokens,
        "tg_tokens": args.tg_tokens,
        "results": results,
    }
    json_path = RESULTS_DIR / f"{timestamp}-tune.json"
    json_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    md = render_markdown([r for r in results if not r.get("error")])
    md_path = RESULTS_DIR / f"{timestamp}-tune.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nResults → {json_path}")
    print(f"Report  → {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
