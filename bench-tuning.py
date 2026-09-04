#!/usr/bin/env python3
"""Benchmark llama.cpp batch/ubatch (and KV cache type) tuning for single-user
prompt-processing and decoding throughput on this PAQ_LLAMACPP_SERVER/Qwen3.8-27B host.

For each variant we restart the server via run-paq-llamacpp-server.sh with the variant env
overrides, then issue a large prompt to measure prompt-eval tok/s and a small
prompt with ignore_eos to measure decode tok/s.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import ssl
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
    base = lambda b, u: {"BATCH_SIZE": str(b), "UBATCH_SIZE": str(u)}
    return [
        Variant("b128_u32_baseline", base(128, 32)),
        Variant("b512_u128", base(512, 128)),
        Variant("b1024_u256", base(1024, 256)),
        Variant("b1536_u384", base(1536, 384)),
        Variant("b2048_u384", base(2048, 384)),
        Variant("b2048_u512", base(2048, 512)),
        Variant("b2048_u1024", base(2048, 1024)),
        Variant("b2048_u2048", base(2048, 2048)),
        Variant("b4096_u2048", base(4096, 2048)),
    ]


def read_api_key() -> str:
    return API_KEY_FILE.read_text(encoding="utf-8").splitlines()[0]


def https_json(base_url: str, path: str, api_key: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    method = "GET" if payload is None else "POST"
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
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
        proc.wait(timeout=30); return
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
    return (chunk * repeats)


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
    rows = sorted(results, key=lambda r: r["summary"]["pp_tps_mean"] + r["summary"]["tg_tps_mean"], reverse=True)
    lines = [
        "| rank | variant | prompt tok/s | decode tok/s | draft accept | pp tokens | gpu mem used MiB |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, r in enumerate(rows, 1):
        s = r["summary"]
        used = (r.get("gpu_after") or "").split(",")[0].strip()
        lines.append(
            f"| {rank} | `{r['variant']}` | {s['pp_tps_mean']:.1f} | {s['tg_tps_mean']:.2f} | "
            f"{s['tg_acceptance_mean']:.2%} | {s['pp_tokens']:.0f} | {used} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18086)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--pp-tokens", type=int, default=16000)
    p.add_argument("--pp-runs", type=int, default=1)
    p.add_argument("--tg-tokens", type=int, default=512)
    p.add_argument("--tg-runs", type=int, default=2)
    p.add_argument("--startup-timeout", type=float, default=240.0)
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
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"variant": v.name, "env": v.env, "error": str(exc),
                            "summary": {"pp_tps_mean": 0, "pp_tps_min": 0, "pp_tokens": 0,
                                        "tg_tps_mean": 0, "tg_acceptance_mean": 0}})
            continue
        s = r["summary"]
        print(f"  pp_tps={s['pp_tps_mean']:.1f} ({s['pp_tokens']:.0f} tok)  tg_tps={s['tg_tps_mean']:.2f}  acc={s['tg_acceptance_mean']:.2%}  gpu={r['gpu_after']}")
        results.append(r)

    md = render_markdown(results)
    out = {"timestamp": timestamp, "args": vars(args), "results": results, "markdown": md}
    json_path = RESULTS_DIR / f"{timestamp}-tuning.json"
    md_path = RESULTS_DIR / f"{timestamp}-tuning.md"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
