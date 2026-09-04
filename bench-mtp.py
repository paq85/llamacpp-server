#!/usr/bin/env python3
"""Benchmark llama.cpp draft-MTP variants through the local run-paq-llamacpp-server.sh launcher.

The harness intentionally restarts the server for each variant so the measured
process exactly reflects the launcher environment for that MTP setting.
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
DEFAULT_PORT = 18080
SYSTEMD_UNIT = "paq-llamacpp-server.service"


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]


def built_in_variants() -> list[Variant]:
    variants = [Variant("mtp_off", {"SPEC_TYPE": "off"})]
    variants.extend(
        Variant(f"mtp_n{n}", {"SPEC_TYPE": "draft-mtp", "SPEC_DRAFT_N_MAX": str(n), "SPEC_DRAFT_NGL": "auto"})
        for n in range(1, 7)
    )
    variants.extend(
        Variant(
            f"mtp_n{n}_backend_off",
            {
                "SPEC_TYPE": "draft-mtp",
                "SPEC_DRAFT_N_MAX": str(n),
                "SPEC_DRAFT_NGL": "auto",
                "SPEC_DRAFT_BACKEND_SAMPLING": "0",
            },
        )
        for n in (1, 2, 3)
    )
    variants.extend(
        Variant(
            f"mtp_n{n}_min{n_min}",
            {
                "SPEC_TYPE": "draft-mtp",
                "SPEC_DRAFT_N_MAX": str(n),
                "SPEC_DRAFT_N_MIN": str(n_min),
                "SPEC_DRAFT_NGL": "auto",
            },
        )
        for n, n_min in ((1, 1), (2, 1), (2, 2))
    )
    variants.extend(
        Variant(
            f"mtp_n{n}_p{int(p_min * 100):02d}",
            {
                "SPEC_TYPE": "draft-mtp",
                "SPEC_DRAFT_N_MAX": str(n),
                "SPEC_DRAFT_P_MIN": str(p_min),
                "SPEC_DRAFT_NGL": "auto",
            },
        )
        for n in (1, 2)
        for p_min in (0.05, 0.10, 0.20)
    )
    return variants


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def read_api_key() -> str:
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError) as exc:
        raise SystemExit(f"Could not read first API key from {API_KEY_FILE}: {exc}") from exc


def https_json(base_url: str, path: str, api_key: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    method = "GET"
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, context=ssl._create_unverified_context(), timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def query_gpu() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,power.draw",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
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
    startup_lines: list[str] = []
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            line = lines.get(timeout=max(0.1, min(2.0, deadline - time.monotonic())))
        except queue.Empty:
            continue
        startup_lines.append(line)
        if ("server is listening" in line) or ("listening on http" in line):
            return startup_lines
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s")


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.kill(proc.pid, signal.SIGKILL)


def completion_payload(run_index: int, n_predict: int, temperature: float, seed: int) -> dict[str, Any]:
    # Use a stable, non-chat completion to focus on decode speed. The nonce keeps
    # prompt cache from making prompt timings look unrealistically perfect, while
    # decode timings stay comparable across variants because the seed is fixed.
    nonce = f"run-{run_index}-seed-{seed}"
    prompt = (
        "You are benchmarking token generation speed. "
        "Continue with a dense technical paragraph about GPU inference, CUDA kernels, "
        "KV cache layouts, memory bandwidth, scheduler overhead, and speculative decoding. "
        f"Nonce: {nonce}.\n\nParagraph:\n"
    )
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "seed": seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }


def run_one_request(base_url: str, api_key: str, run_index: int, n_predict: int, temperature: float, seed: int) -> dict[str, Any]:
    payload = completion_payload(run_index, n_predict, temperature, seed)
    start = time.perf_counter()
    response = https_json(base_url, "/completion", api_key, payload, timeout=900.0)
    wall_s = time.perf_counter() - start
    timings = response.get("timings", {}) or {}
    draft_n = float(timings.get("draft_n") or 0)
    draft_accepted = float(timings.get("draft_n_accepted") or 0)
    predicted_n = float(timings.get("predicted_n") or response.get("tokens_predicted") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0)
    prompt_n = float(timings.get("prompt_n") or response.get("tokens_evaluated") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0)
    return {
        "run": run_index,
        "seed": seed,
        "wall_s": wall_s,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "predicted_tps": float(timings.get("predicted_per_second") or (predicted_n / (predicted_ms / 1000.0) if predicted_ms else 0)),
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "prompt_tps": float(timings.get("prompt_per_second") or (prompt_n / (prompt_ms / 1000.0) if prompt_ms else 0)),
        "draft_n": draft_n,
        "draft_accepted": draft_accepted,
        "draft_acceptance": draft_accepted / draft_n if draft_n else 0.0,
        "content_preview": (response.get("content") or "")[:80],
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_tps = [float(r["predicted_tps"]) for r in runs]
    wall_tps = [float(r["predicted_n"]) / float(r["wall_s"]) for r in runs if float(r["wall_s"]) > 0]
    acceptances = [float(r["draft_acceptance"]) for r in runs if float(r["draft_n"]) > 0]
    draft_tokens = [float(r["draft_n"]) for r in runs]
    accepted_tokens = [float(r["draft_accepted"]) for r in runs]
    return {
        "runs": len(runs),
        "predicted_tps_mean": mean(predicted_tps),
        "predicted_tps_stdev": stdev(predicted_tps),
        "predicted_tps_p50": pct(predicted_tps, 0.50),
        "wall_tps_mean": mean(wall_tps),
        "wall_tps_stdev": stdev(wall_tps),
        "draft_acceptance_mean": mean(acceptances),
        "draft_tokens_total": sum(draft_tokens),
        "draft_accepted_total": sum(accepted_tokens),
        "draft_acceptance_total": sum(accepted_tokens) / sum(draft_tokens) if sum(draft_tokens) else 0.0,
        "prompt_tps_mean": mean([float(r["prompt_tps"]) for r in runs]),
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    rows = sorted(results, key=lambda r: r["summary"]["predicted_tps_mean"], reverse=True)
    lines = [
        "| rank | variant | predicted tok/s | wall tok/s | draft acceptance | draft accepted / generated | runs |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, result in enumerate(rows, 1):
        s = result["summary"]
        lines.append(
            "| {rank} | `{name}` | {ptps:.2f} ± {pstdev:.2f} | {wtps:.2f} ± {wstdev:.2f} | {acc:.2%} | {accepted:.0f}/{draft:.0f} | {runs} |".format(
                rank=rank,
                name=result["variant"],
                ptps=s["predicted_tps_mean"],
                pstdev=s["predicted_tps_stdev"],
                wtps=s["wall_tps_mean"],
                wstdev=s["wall_tps_stdev"],
                acc=s["draft_acceptance_total"],
                accepted=s["draft_accepted_total"],
                draft=s["draft_tokens_total"],
                runs=s["runs"],
            )
        )
    return "\n".join(lines) + "\n"


def run_variant(variant: Variant, args: argparse.Namespace, api_key: str, timestamp: str) -> dict[str, Any]:
    port = args.port
    base_url = f"http://127.0.0.1:{port}"
    log_path = RESULTS_DIR / f"{timestamp}-{variant.name}.log"
    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "WARMUP": "0",
            "PROMPT_CACHE": "0",
            "CACHE_IDLE_SLOTS": "0",
            "CLOUDFLARED_ENABLED": "off",
            "CLOUDFLARE_TIMEOUT_PROXY_MODE": "off",
            "LLAMA_SERVER_SSL_KEY_FILE": "",
            "LLAMA_SERVER_SSL_CERT_FILE": "",
        }
    )
    if args.no_vision:
        env["ENABLE_MMPROJ"] = "0"
    env.update(variant.env)

    lines: "queue.Queue[str]" = queue.Queue()
    proc = subprocess.Popen(
        [str(RUNNER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    reader = threading.Thread(target=log_reader, args=(proc, log_path, lines), daemon=True)
    reader.start()

    startup_lines: list[str] = []
    try:
        startup_started = time.perf_counter()
        startup_lines = wait_for_server(proc, lines, args.startup_timeout)
        startup_s = time.perf_counter() - startup_started

        # Confirm HTTP readiness and capture advertised metadata.
        health = https_json(base_url, "/health", api_key, timeout=60.0)
        models = https_json(base_url, "/v1/models", api_key, timeout=60.0)
        meta = (models.get("data") or [{}])[0].get("meta", {})
        capabilities = ((models.get("models") or [{}])[0].get("capabilities") or [])

        for i in range(args.warmup_runs):
            run_one_request(base_url, api_key, -(i + 1), min(args.tokens, 96), args.temperature, args.seed + 10_000 + i)

        runs = [
            run_one_request(base_url, api_key, i + 1, args.tokens, args.temperature, args.seed + i)
            for i in range(args.runs)
        ]
        summary = summarize_runs(runs)
        return {
            "variant": variant.name,
            "env": variant.env,
            "startup_s": startup_s,
            "health": health,
            "n_ctx": meta.get("n_ctx"),
            "capabilities": capabilities,
            "gpu_after": query_gpu(),
            "startup_log_excerpt": startup_lines[-25:],
            "log_path": str(log_path),
            "runs": runs,
            "summary": summary,
        }
    finally:
        stop_server(proc)
        reader.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Measured requests per variant.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup requests per variant.")
    parser.add_argument("--tokens", type=int, default=512, help="Completion tokens per measured request.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature used for measured requests.")
    parser.add_argument("--seed", type=int, default=424242, help="Base seed for comparable requests.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local port used by benchmark servers.")
    parser.add_argument("--startup-timeout", type=float, default=180.0, help="Seconds to wait for server startup.")
    parser.add_argument("--variants", default="", help="Comma-separated variant names. Default: all built-ins.")
    parser.add_argument("--no-vision", action="store_true", help="Disable mmproj during benchmark. Default keeps production vision loaded.")
    args = parser.parse_args()

    if not RUNNER.exists():
        raise SystemExit(f"Missing launcher: {RUNNER}")

    if systemd_service_is_active(SYSTEMD_UNIT):
        raise SystemExit(
            f"{SYSTEMD_UNIT} is active. Stop the systemd-managed server before running MTP benchmarks "
            "so run-paq-llamacpp-server.sh does not contend with the live instance."
        )

    all_variants = {variant.name: variant for variant in built_in_variants()}
    if args.variants:
        names = [name.strip() for name in args.variants.split(",") if name.strip()]
        missing = [name for name in names if name not in all_variants]
        if missing:
            raise SystemExit(f"Unknown variants: {', '.join(missing)}. Known: {', '.join(all_variants)}")
        variants = [all_variants[name] for name in names]
    else:
        variants = list(all_variants.values())

    api_key = read_api_key()
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results: list[dict[str, Any]] = []

    print(f"Benchmarking {len(variants)} variants on port {args.port} with {args.runs}x{args.tokens} token requests each")
    for variant in variants:
        print(f"\n== {variant.name} ==")
        result = run_variant(variant, args, api_key, timestamp)
        results.append(result)
        s = result["summary"]
        print(
            "predicted_tps={:.2f} wall_tps={:.2f} acceptance={:.2%} n_ctx={} gpu='{}'".format(
                s["predicted_tps_mean"],
                s["wall_tps_mean"],
                s["draft_acceptance_total"],
                result["n_ctx"],
                result["gpu_after"],
            )
        )

    output = {
        "timestamp": timestamp,
        "args": vars(args),
        "results": results,
        "markdown": render_markdown(results),
    }
    json_path = RESULTS_DIR / f"{timestamp}-mtp-benchmark.json"
    md_path = RESULTS_DIR / f"{timestamp}-mtp-benchmark.md"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md_path.write_text(output["markdown"], encoding="utf-8")

    print("\n" + output["markdown"])
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
