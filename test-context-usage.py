#!/usr/bin/env python3
"""Test: Context usage reporting through the cloudflare-timeout-proxy.

Runs various request scenarios against the live proxy, captures raw
responses, and validates that usage / context data is present and correct.

This is a **detection-only** script — it does NOT fix the proxy.

Usage:
    python3 test-context-usage.py
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
import threading
import urllib.error
import urllib.request
import http.client
import csv
import socket
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "benchmarks"
API_KEY_FILE = ROOT / "api-keys.txt"

# URL candidates — base URLs (without /v1) for the proxy or direct backend
URL_CANDIDATES = [
    "https://127.0.0.1:8080",
    "http://127.0.0.1:8080",
    "https://127.0.0.1:8082",
    "http://127.0.0.1:8082",
]

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class UsageCheck:
    """Result of extracting usage from a single request."""
    scenario: str
    group: str
    endpoint: str
    stream: bool
    had_usage_key: bool
    had_timings_key: bool
    had_finish_reason: bool
    prompt_tokens: Optional[int]
    input_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cached_tokens: Optional[int]
    cache_read: Optional[int]
    result: str  # OK / MISSING_USAGE / PARTIAL / MISSING_TIMINGS / ERROR
    error: Optional[str] = None
    raw_final_chunk: Optional[str] = None
    raw_full_response: Optional[str] = None
    response_time_s: Optional[float] = None


@dataclass
class TestReport:
    """Aggregated test report."""
    timestamp: str
    proxy_url: str
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    ok_count: int = 0
    missing_count: int = 0
    partial_count: int = 0
    error_count: int = 0
    findings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_api_key() -> str:
    """Read the first non-empty, non-comment line from api-keys.txt."""
    with open(API_KEY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise SystemExit(f"No API key found in {API_KEY_FILE}")


def detect_proxy_url() -> str:
    """Detect the live proxy URL by trying candidates.

    Returns the base URL (without any path) so that test paths
    like ``/v1/chat/completions`` concatenate cleanly.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    for base_url in URL_CANDIDATES:
        try:
            models_url = f"{base_url.rstrip('/')}/v1/models"
            req = urllib.request.Request(models_url)
            resp = urllib.request.urlopen(req, timeout=5, context=ctx)
            if resp.status == 200:
                print(f"[OK] Proxy detected at {base_url}")
                return base_url
        except Exception:
            continue

    raise SystemExit("Could not detect a live proxy. Is the server running?")


def make_request(
    base_url: str,
    path: str,
    body: dict,
    api_key: str,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> http.client.HTTPResponse:
    """Make a raw HTTP request to the proxy."""
    url = base_url + path
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if url.startswith("https://"):
        host_port = url[8:].split("/", 1)[0]
        path_only = "/" + url[8:].split("/", 1)[1] if "/" in url[8:] else "/"
        conn = http.client.HTTPSConnection(
            host_port.split(":")[0],
            int(host_port.split(":")[1]) if ":" in host_port else 443,
            timeout=timeout,
            context=ctx,
        )
    else:
        host_port = url[7:].split("/", 1)[0]
        path_only = "/" + url[7:].split("/", 1)[1] if "/" in url[7:] else "/"
        conn = http.client.HTTPConnection(
            host_port.split(":")[0],
            int(host_port.split(":")[1]) if ":" in host_port else 80,
            timeout=timeout,
        )

    conn.request("POST", path_only, body=data, headers=headers)
    return conn.getresponse()


def read_sse_stream(response: http.client.HTTPResponse) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    """Read SSE stream and return (events, raw_lines).

    Uses line-oriented parsing so chunk boundaries cannot corrupt SSE events.
    Each event is (event_type, data_bytes, raw_line).
    """
    events: list[tuple[str, bytes, str]] = []
    raw_lines: list[str] = []

    while True:
        chunk = response.readline()
        if not chunk:
            break

        line = chunk.decode("utf-8", errors="replace")
        raw_lines.append(line.rstrip("\n"))
        stripped = line.rstrip("\r\n")

        if stripped.startswith("event:"):
            event_type = stripped[6:].strip()
            events.append((event_type, b"", stripped))
        elif stripped.startswith("data:"):
            data = stripped[5:]
            while data.startswith(" "):
                data = data[1:]
            events.append(("data", data.encode("utf-8"), stripped))
        elif stripped.startswith(":"):
            events.append(("comment", b"", stripped))

    return events, raw_lines


def try_parse_json(data_bytes: bytes) -> dict[str, Any] | None:
    """Try to parse bytes as JSON, return None on failure."""
    try:
        parsed = json.loads(data_bytes.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


def find_last_json_event(data_events: list[tuple[str, bytes, str]]) -> tuple[Optional[int], bytes, dict[str, Any]]:
    """Find the last SSE data event that is valid JSON (skipping [DONE] / non-JSON terminators).
    
    Returns (index, raw_bytes, parsed_dict) or (None, b'', {}) if none found.
    Searches backwards from the end so [DONE] is automatically skipped.
    """
    for idx in range(len(data_events) - 1, -1, -1):
        ev = data_events[idx]
        parsed = try_parse_json(ev[1])
        if parsed is not None:
            return (idx, ev[1], parsed)
    return (None, b"", {})


def fill_usage_from_timings(check: UsageCheck, timings: dict[str, Any]) -> None:
    """Fill token counts from timings without pretending a real usage object existed.

    This keeps timing-derived counts available for debugging while still
    allowing the test result to flag missing explicit usage as a failure.
    """
    tn = timings.get("prompt_n")
    cn = timings.get("cache_n") or 0
    pn = timings.get("predicted_n")

    if check.prompt_tokens is None and tn is not None:
        check.prompt_tokens = tn + cn
    if check.completion_tokens is None and pn is not None:
        check.completion_tokens = pn
    if check.cached_tokens is None:
        check.cached_tokens = cn

    if check.prompt_tokens is not None and check.completion_tokens is not None:
        check.total_tokens = check.prompt_tokens + check.completion_tokens


def extract_usage_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract all possible usage fields from a payload.
    
    Looks in multiple places: top-level, usage dict, message.usage, etc.
    """
    result: dict[str, Any] = {}

    # Check for usage at various levels
    usage_candidates = []
    for key in ("usage", "Usage", "USAGE"):
        if key in payload and isinstance(payload[key], dict):
            usage_candidates.append(payload[key])

    # For choices[].message.usage (only if the message actually has a usage field)
    choices = payload.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict) and "usage" in msg:
                    msg_usage = msg.get("usage")
                    if isinstance(msg_usage, dict):
                        usage_candidates.append(msg_usage)

    if not usage_candidates:
        result["had_usage_key"] = False
        return result

    result["had_usage_key"] = True
    usage = usage_candidates[0]

    # OpenAI-style keys
    result["prompt_tokens"] = _first_int(usage, "prompt_tokens", "promptTokens", "prompt_n")
    result["input_tokens"] = _first_int(usage, "input", "input_tokens", "inputTokens")
    result["completion_tokens"] = _first_int(usage, "completion_tokens", "completionTokens", "output", "output_tokens", "outputTokens", "predicted_n")
    result["total_tokens"] = _first_int(usage, "total_tokens", "totalTokens", "total")

    # Cache-related
    result["cached_tokens"] = _first_int(usage, "cached_tokens", "cache_read", "cacheRead", "cache_read_input_tokens")
    result["cache_read"] = result.get("cached_tokens")

    # Prompt tokens details
    ptd = usage.get("prompt_tokens_details", {})
    if isinstance(ptd, dict):
        result["ptd_cached"] = ptd.get("cached_tokens")
    itd = usage.get("input_tokens_details", {})
    if isinstance(itd, dict):
        result["itd_cached"] = itd.get("cached_tokens")

    return result


def extract_timings_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract timing info from a payload."""
    result: dict[str, Any] = {"had_timings": False}
    timings = payload.get("timings", {})
    if not isinstance(timings, dict):
        timings = {}

    if timings:
        result["had_timings"] = True
        result["prompt_n"] = timings.get("prompt_n")
        result["predicted_n"] = timings.get("predicted_n")
        result["cache_n"] = timings.get("cache_n")

    return result


def extract_finish_reason(payload: dict[str, Any]) -> Optional[str]:
    """Extract finish_reason from choices."""
    choices = payload.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                fr = choice.get("finish_reason")
                if fr is not None:
                    return fr
    return None


def _first_int(d: dict[str, Any], *keys: str):
    """Get the first non-None integer value for the given keys."""
    for key in keys:
        val = d.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def classify_usage_source(scenario: dict[str, Any]) -> str:
    """Classify whether a scenario got explicit usage, timings-only counts, or neither."""
    if scenario.get("had_usage_key"):
        return "explicit"
    if scenario.get("had_timings_key"):
        return "timings_only"
    return "none"


def build_usage_source_counts(scenarios: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize how usage information was sourced across scenarios."""
    counts = {"explicit": 0, "timings_only": 0, "none": 0}
    for scenario in scenarios:
        counts[classify_usage_source(scenario)] += 1
    return counts


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def get_long_prompt() -> str:
    """Generate a longer prompt (~200 tokens) for testing prefill."""
    return (
        "Please write a detailed explanation of how transformers work in "
        "modern large language models. Cover the following topics in detail: "
        "self-attention mechanism, multi-head attention, positional encoding, "
        "feed-forward networks, layer normalization, residual connections, "
        "and the decoder architecture. Include mathematical intuition where "
        "helpful. Explain how the attention mechanism allows the model to "
        "focus on different parts of the input sequence and why this is "
        "superior to traditional RNN-based approaches for long-range "
        "dependencies. Describe the computational complexity of the "
        "attention mechanism and how techniques like FlashAttention optimize "
        "this process. Discuss the trade-offs between model depth and width "
        "in terms of performance and computational cost."
    )


def get_inline_base64_image() -> str:
    """Return a tiny 1x1 white PNG as base64 for vision testing."""
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
        "HQME2u7h8gAAAABJRU5ErkJggg=="
    )


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------


def test_openai_stream_simple(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 1: Streaming OpenAI chat with simple prompt."""
    check = UsageCheck(
        scenario="openai_stream_simple",
        group="basic-openai",
        endpoint="/v1/chat/completions",
        stream=True,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [{"role": "user", "content": "Reply with a single short sentence."}],
        "max_tokens": 20,
        "stream": True,
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
            events, raw_lines = read_sse_stream(resp)
            elapsed = time.monotonic() - t0

        raw_response_text = "\n".join(raw_lines[:100])

        # Find the last data event
        data_events = [e for e in events if e[0] == "data"]
        if data_events:
            # Skip [DONE] terminator, find last real JSON event
            _, last_data, payload = find_last_json_event(data_events)

            if payload:
                check.had_finish_reason = extract_finish_reason(payload) is not None
                usage_info = extract_usage_from_payload(payload)
                check.had_usage_key = usage_info.get("had_usage_key", False)
                check.prompt_tokens = usage_info.get("prompt_tokens")
                check.input_tokens = usage_info.get("input_tokens")
                check.completion_tokens = usage_info.get("completion_tokens")
                check.total_tokens = usage_info.get("total_tokens")
                check.cached_tokens = usage_info.get("cached_tokens")
                check.cache_read = usage_info.get("cache_read")

                timings = extract_timings_from_payload(payload)
                check.had_timings_key = timings.get("had_timings", False)
                # If no usage but has timings, synthesize usage from timings
                if not check.had_usage_key and timings.get("had_timings"):
                    fill_usage_from_timings(check, timings)

            # Check all events for usage
            for idx, ev in enumerate(data_events):
                p = try_parse_json(ev[1])
                if p and "usage" in p:
                    if not check.had_usage_key:
                        report.findings.append(
                            f"[BUG?] {check.scenario}: usage on event #{idx}, not last JSON event"
                        )
                        check.had_usage_key = True
                        u = p.get("usage", {})
                        check.prompt_tokens = check.prompt_tokens or _first_int(u, "prompt_tokens", "promptTokens", "prompt_n")

            check.raw_final_chunk = last_data.decode("utf-8", errors="replace")[:3000] if last_data else "(no JSON events)"
        else:
            check.error = "No data events in SSE stream"

        check.response_time_s = round(elapsed, 3)
        check.raw_full_response = raw_response_text[:30000]

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    # Evaluate
    check.result = evaluate_usage(check)
    return check


def test_openai_nostream_simple(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 1: Non-streaming OpenAI chat with simple prompt."""
    check = UsageCheck(
        scenario="openai_nostream_simple",
        group="basic-openai",
        endpoint="/v1/chat/completions",
        stream=False,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [{"role": "user", "content": "Reply with a single short word."}],
        "max_tokens": 10,
        "stream": False,
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
            raw = resp.read()
            elapsed = time.monotonic() - t0

        payload = try_parse_json(raw)
        if payload:
            check.had_finish_reason = extract_finish_reason(payload) is not None
            usage_info = extract_usage_from_payload(payload)
            check.had_usage_key = usage_info.get("had_usage_key", False)
            check.prompt_tokens = usage_info.get("prompt_tokens")
            check.completion_tokens = usage_info.get("completion_tokens")
            check.total_tokens = usage_info.get("total_tokens")
            check.cached_tokens = usage_info.get("cached_tokens")

            timings = extract_timings_from_payload(payload)
            check.had_timings_key = timings.get("had_timings", False)

            check.raw_final_chunk = json.dumps(payload, indent=2)[:5000]
        else:
            check.error = "Response is not valid JSON"
            check.raw_final_chunk = raw.decode("utf-8", errors="replace")[:3000]

        check.response_time_s = round(elapsed, 3)
        check.raw_full_response = raw.decode("utf-8", errors="replace")[:30000]

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_openai_stream_json_mode(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 1: Streaming with JSON response format."""
    check = UsageCheck(
        scenario="openai_stream_json_mode",
        group="basic-openai",
        endpoint="/v1/chat/completions",
        stream=True,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [
            {"role": "system", "content": "Respond in JSON format."},
            {"role": "user", "content": "Return a JSON with a 'greeting' key."},
        ],
        "max_tokens": 30,
        "stream": True,
        "response_format": {"type": "json_object"},
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
            events, raw_lines = read_sse_stream(resp)
            elapsed = time.monotonic() - t0

        data_events = [e for e in events if e[0] == "data"]
        if data_events:
            # Skip [DONE] terminator, find last real JSON event
            _, last_data, payload = find_last_json_event(data_events)

            if payload:
                check.had_finish_reason = extract_finish_reason(payload) is not None
                usage_info = extract_usage_from_payload(payload)
                check.had_usage_key = usage_info.get("had_usage_key", False)
                check.prompt_tokens = usage_info.get("prompt_tokens")
                check.completion_tokens = usage_info.get("completion_tokens")
                check.total_tokens = usage_info.get("total_tokens")
                check.cached_tokens = usage_info.get("cached_tokens")

                timings = extract_timings_from_payload(payload)
                check.had_timings_key = timings.get("had_timings", False)
                # If no usage but has timings, synthesize usage from timings
                if not check.had_usage_key and timings.get("had_timings"):
                    fill_usage_from_timings(check, timings)

            # Check all events for usage
            for idx, ev in enumerate(data_events):
                p = try_parse_json(ev[1])
                if p and "usage" in p:
                    if not check.had_usage_key:
                        report.findings.append(
                            f"[BUG?] {check.scenario}: usage on event #{idx}, not last JSON event"
                        )
                        check.had_usage_key = True
                        u = p.get("usage", {})
                        check.prompt_tokens = check.prompt_tokens or _first_int(u, "prompt_tokens", "promptTokens", "prompt_n")

            check.raw_final_chunk = last_data.decode("utf-8", errors="replace")[:3000]
        else:
            check.error = "No data events"

        check.response_time_s = round(elapsed, 3)

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_openai_nostream_json_mode(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 1: Non-streaming with JSON response format."""
    check = UsageCheck(
        scenario="openai_nostream_json_mode",
        group="basic-openai",
        endpoint="/v1/chat/completions",
        stream=False,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [
            {"role": "system", "content": "Respond in JSON."},
            {"role": "user", "content": "Return {\"answer\": 42}"},
        ],
        "max_tokens": 30,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
            raw = resp.read()
            elapsed = time.monotonic() - t0

        payload = try_parse_json(raw)
        if payload:
            check.had_finish_reason = extract_finish_reason(payload) is not None
            usage_info = extract_usage_from_payload(payload)
            check.had_usage_key = usage_info.get("had_usage_key", False)
            check.prompt_tokens = usage_info.get("prompt_tokens")
            check.completion_tokens = usage_info.get("completion_tokens")
            check.total_tokens = usage_info.get("total_tokens")
            check.cached_tokens = usage_info.get("cached_tokens")

            timings = extract_timings_from_payload(payload)
            check.had_timings_key = timings.get("had_timings", False)

            check.raw_final_chunk = json.dumps(payload, indent=2)[:5000]
        else:
            check.error = "Not JSON"

        check.response_time_s = round(elapsed, 3)

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_tool_calling_stream(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 3: Tool calling request (streaming)."""
    check = UsageCheck(
        scenario="tool_call_stream",
        group="edge-cases",
        endpoint="/v1/chat/completions",
        stream=True,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
        "max_tokens": 50,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
            events, raw_lines = read_sse_stream(resp)
            elapsed = time.monotonic() - t0

        data_events = [e for e in events if e[0] == "data"]
        if data_events:
            # Skip [DONE] terminator, find last real JSON event
            _, last_data, payload = find_last_json_event(data_events)

            if payload:
                check.had_finish_reason = extract_finish_reason(payload) is not None
                usage_info = extract_usage_from_payload(payload)
                check.had_usage_key = usage_info.get("had_usage_key", False)
                check.prompt_tokens = usage_info.get("prompt_tokens")
                check.completion_tokens = usage_info.get("completion_tokens")
                check.total_tokens = usage_info.get("total_tokens")
                check.cached_tokens = usage_info.get("cached_tokens")

                timings = extract_timings_from_payload(payload)
                check.had_timings_key = timings.get("had_timings", False)
                # If no usage but has timings, synthesize usage from timings
                if not check.had_usage_key and timings.get("had_timings"):
                    fill_usage_from_timings(check, timings)

                # Check finish_reason — tool calls may use "tool_calls" instead of "stop"
                fr = extract_finish_reason(payload)
                check.had_finish_reason = fr is not None

                # Check all events for usage
                usage_events = []
                for idx, ev in enumerate(data_events):
                    p = try_parse_json(ev[1])
                    if p and "usage" in p:
                        usage_events.append(idx)

                if usage_events and not check.had_usage_key:
                    report.findings.append(
                        f"[BUG?] {check.scenario}: usage on events {usage_events}, not last"
                    )

            check.raw_final_chunk = last_data.decode("utf-8", errors="replace")[:3000]
        else:
            check.error = "No data events"

        check.response_time_s = round(elapsed, 3)

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_vision_nostream(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 3: Vision request with inline base64 image (non-streaming)."""
    check = UsageCheck(
        scenario="vision_nostream",
        group="edge-cases",
        endpoint="/v1/chat/completions",
        stream=False,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this tiny image in a word."},
                    {
                        "type": "image_url",
                        "image_url": {"url": get_inline_base64_image()},
                    },
                ],
            }
        ],
        "max_tokens": 20,
        "stream": False,
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key, timeout=180) as resp:
            raw = resp.read()
            elapsed = time.monotonic() - t0

        payload = try_parse_json(raw)
        if payload:
            check.had_finish_reason = extract_finish_reason(payload) is not None
            usage_info = extract_usage_from_payload(payload)
            check.had_usage_key = usage_info.get("had_usage_key", False)
            check.prompt_tokens = usage_info.get("prompt_tokens")
            check.completion_tokens = usage_info.get("completion_tokens")
            check.total_tokens = usage_info.get("total_tokens")
            check.cached_tokens = usage_info.get("cached_tokens")

            timings = extract_timings_from_payload(payload)
            check.had_timings_key = timings.get("had_timings", False)

            check.raw_final_chunk = json.dumps(payload, indent=2)[:5000]
        else:
            check.error = "Not JSON"
            check.raw_final_chunk = raw.decode("utf-8", errors="replace")[:3000]

        check.response_time_s = round(elapsed, 3)

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_short_prompt_stream(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 3: Minimal prompt "Hi" (streaming)."""
    check = UsageCheck(
        scenario="short_prompt_stream",
        group="edge-cases",
        endpoint="/v1/chat/completions",
        stream=True,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": True,
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
            events, raw_lines = read_sse_stream(resp)
            elapsed = time.monotonic() - t0

        data_events = [e for e in events if e[0] == "data"]
        if data_events:
            # Skip [DONE] terminator, find last real JSON event
            _, last_data, payload = find_last_json_event(data_events)

            if payload:
                check.had_finish_reason = extract_finish_reason(payload) is not None
                usage_info = extract_usage_from_payload(payload)
                check.had_usage_key = usage_info.get("had_usage_key", False)
                check.prompt_tokens = usage_info.get("prompt_tokens")
                check.completion_tokens = usage_info.get("completion_tokens")
                check.total_tokens = usage_info.get("total_tokens")
                check.cached_tokens = usage_info.get("cached_tokens")

                timings = extract_timings_from_payload(payload)
                check.had_timings_key = timings.get("had_timings", False)
                # If no usage but has timings, synthesize usage from timings
                if not check.had_usage_key and timings.get("had_timings"):
                    fill_usage_from_timings(check, timings)

            check.raw_final_chunk = last_data.decode("utf-8", errors="replace")[:3000]
        else:
            check.error = "No data events"

        check.response_time_s = round(elapsed, 3)

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_long_prompt_nostream(report: TestReport, base_url: str, api_key: str) -> UsageCheck:
    """Group 3: Longer prompt (non-streaming)."""
    check = UsageCheck(
        scenario="long_prompt_nostream",
        group="edge-cases",
        endpoint="/v1/chat/completions",
        stream=False,
        had_usage_key=False,
        had_timings_key=False,
        had_finish_reason=False,
        prompt_tokens=None,
        input_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cached_tokens=None,
        cache_read=None,
        result="",
    )

    body = {
        "model": "PAQ_LLAMACPP_SERVER",
        "messages": [{"role": "user", "content": get_long_prompt()}],
        "max_tokens": 50,
        "stream": False,
    }

    try:
        t0 = time.monotonic()
        with make_request(base_url, "/v1/chat/completions", body, api_key, timeout=180) as resp:
            raw = resp.read()
            elapsed = time.monotonic() - t0

        payload = try_parse_json(raw)
        if payload:
            check.had_finish_reason = extract_finish_reason(payload) is not None
            usage_info = extract_usage_from_payload(payload)
            check.had_usage_key = usage_info.get("had_usage_key", False)
            check.prompt_tokens = usage_info.get("prompt_tokens")
            check.completion_tokens = usage_info.get("completion_tokens")
            check.total_tokens = usage_info.get("total_tokens")
            check.cached_tokens = usage_info.get("cached_tokens")

            timings = extract_timings_from_payload(payload)
            check.had_timings_key = timings.get("had_timings", False)

            check.raw_final_chunk = json.dumps(payload, indent=2)[:5000]
        else:
            check.error = "Not JSON"

        check.response_time_s = round(elapsed, 3)

    except Exception as e:
        check.result = "ERROR"
        check.error = str(e)
        return check

    check.result = evaluate_usage(check)
    return check


def test_cache_cold_then_warm(report: TestReport, base_url: str, api_key: str) -> tuple[UsageCheck, UsageCheck]:
    """Group 2: Same prompt twice — first cold, second cached."""
    prompt = "Explain in one sentence what prompt caching is in LLM inference servers."

    def make_one(label: str, n: int) -> UsageCheck:
        is_stream = n == 1  # First one streaming, second non-streaming
        check = UsageCheck(
            scenario=f"cache_cold_warm_{label}",
            group="cache",
            endpoint="/v1/chat/completions",
            stream=is_stream,
            had_usage_key=False,
            had_timings_key=False,
            had_finish_reason=False,
            prompt_tokens=None,
            input_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cached_tokens=None,
            cache_read=None,
            result="",
        )

        body = {
            "model": "PAQ_LLAMACPP_SERVER",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "stream": is_stream,
        }

        try:
            t0 = time.monotonic()
            with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
                if is_stream:
                    events, raw_lines = read_sse_stream(resp)
                    data_events = [e for e in events if e[0] == "data"]
                    if data_events:
                        _, last_data, payload = find_last_json_event(data_events)
                        if payload:
                            check.had_finish_reason = extract_finish_reason(payload) is not None
                            ui = extract_usage_from_payload(payload)
                            check.had_usage_key = ui.get("had_usage_key", False)
                            check.prompt_tokens = ui.get("prompt_tokens")
                            check.completion_tokens = ui.get("completion_tokens")
                            check.total_tokens = ui.get("total_tokens")
                            check.cached_tokens = ui.get("cached_tokens")
                            ti = extract_timings_from_payload(payload)
                            check.had_timings_key = ti.get("had_timings", False)
                            if not check.had_usage_key and ti.get("had_timings"):
                                fill_usage_from_timings(check, ti)
                            check.raw_final_chunk = last_data.decode("utf-8", errors="replace")[:3000]
                    check.raw_full_response = "\n".join(raw_lines[:100])[:30000]
                else:
                    raw = resp.read()
                    payload = try_parse_json(raw)
                    if payload:
                        check.had_finish_reason = extract_finish_reason(payload) is not None
                        ui = extract_usage_from_payload(payload)
                        check.had_usage_key = ui.get("had_usage_key", False)
                        check.prompt_tokens = ui.get("prompt_tokens")
                        check.completion_tokens = ui.get("completion_tokens")
                        check.total_tokens = ui.get("total_tokens")
                        check.cached_tokens = ui.get("cached_tokens")
                        ti = extract_timings_from_payload(payload)
                        check.had_timings_key = ti.get("had_timings", False)
                        # If no usage but has timings, synthesize usage from timings
                        if not check.had_usage_key and ti.get("had_timings"):
                            fill_usage_from_timings(check, ti)
                        check.raw_full_response = raw.decode("utf-8", errors="replace")[:30000]

            check.response_time_s = round(time.monotonic() - t0, 3)
            check.result = evaluate_usage(check)
            return check

        except Exception as e:
            check.result = "ERROR"
            check.error = str(e)
            return check

    check1 = make_one("cold_first", 1)
    # Small delay to ensure cache is populated
    time.sleep(1)
    check2 = make_one("warm_second", 2)

    # Cross-validation
    if check1.cached_tokens is not None and check2.cached_tokens is not None:
        if check1.cached_tokens > 0 and check2.cached_tokens > 0:
            report.findings.append(
                f"[WARN] {check1.scenario}: Both requests show cached_tokens > 0. "
                f"First should be cold (cache_n=0) unless prefix was already cached."
            )
        elif check1.cached_tokens == 0 and check2.cached_tokens == 0:
            if check1.prompt_tokens is not None and check2.prompt_tokens is not None:
                report.findings.append(
                    f"[BUG?] Cache test: Both requests show cached_tokens=0 despite "
                    f"same prompt. Prompt caching may not be working, OR proxy is not "
                    f"reporting cache_n correctly. prompt_n: {check1.prompt_tokens} / {check2.prompt_tokens}"
                )
        elif check1.cached_tokens is not None and check1.cached_tokens == 0 and check2.cached_tokens is not None and check2.cached_tokens > 0:
            report.findings.append(
                f"[OK] Cache: Cold={check1.cached_tokens} cached, Warm={check2.cached_tokens} cached. "
                f"Cache behavior is correct."
            )

    return check1, check2


def test_concurrent_requests(report: TestReport, base_url: str, api_key: str) -> tuple[UsageCheck, UsageCheck]:
    """Group 3: Two concurrent requests with PARALLEL=1."""
    checks: list[UsageCheck] = []
    errors: list[str] = []

    def run_one(idx: int, stream: bool) -> UsageCheck:
        check = UsageCheck(
            scenario=f"concurrent_{idx}",
            group="concurrency",
            endpoint="/v1/chat/completions",
            stream=stream,
            had_usage_key=False,
            had_timings_key=False,
            had_finish_reason=False,
            prompt_tokens=None,
            input_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cached_tokens=None,
            cache_read=None,
            result="",
        )

        body = {
            "model": "PAQ_LLAMACPP_SERVER",
            "messages": [{"role": "user", "content": f"Request {idx}: say hello."}],
            "max_tokens": 10,
            "stream": stream,
        }

        try:
            t0 = time.monotonic()
            with make_request(base_url, "/v1/chat/completions", body, api_key, timeout=120) as resp:
                if stream:
                    events, _ = read_sse_stream(resp)
                    data_events = [e for e in events if e[0] == "data"]
                    if data_events:
                        _, _, payload = find_last_json_event(data_events)
                        if payload:
                            ui = extract_usage_from_payload(payload)
                            check.had_usage_key = ui.get("had_usage_key", False)
                            check.prompt_tokens = ui.get("prompt_tokens")
                            check.completion_tokens = ui.get("completion_tokens")
                            check.total_tokens = ui.get("total_tokens")
                            check.cached_tokens = ui.get("cached_tokens")
                            check.had_finish_reason = extract_finish_reason(payload) is not None
                            ti = extract_timings_from_payload(payload)
                            check.had_timings_key = ti.get("had_timings", False)
                            # If no usage but has timings, synthesize usage from timings
                            if not check.had_usage_key and ti.get("had_timings"):
                                fill_usage_from_timings(check, ti)
                        if data_events:
                            _, last_data, _ = find_last_json_event(data_events)
                            if last_data:
                                check.raw_final_chunk = last_data.decode("utf-8", errors="replace")[:3000]
                else:
                    raw = resp.read()
                    payload = try_parse_json(raw)
                    if payload:
                        ui = extract_usage_from_payload(payload)
                        check.had_usage_key = ui.get("had_usage_key", False)
                        check.prompt_tokens = ui.get("prompt_tokens")
                        check.completion_tokens = ui.get("completion_tokens")
                        check.total_tokens = ui.get("total_tokens")
                        check.cached_tokens = ui.get("cached_tokens")
                        check.had_finish_reason = extract_finish_reason(payload) is not None
                        ti = extract_timings_from_payload(payload)
                        check.had_timings_key = ti.get("had_timings", False)
                        if not check.had_usage_key and ti.get("had_timings"):
                            fill_usage_from_timings(check, ti)
                        check.raw_final_chunk = json.dumps(payload, indent=2)[:5000]
                    else:
                        check.error = "Response is not valid JSON"
                        check.raw_final_chunk = raw.decode("utf-8", errors="replace")[:3000]

            check.response_time_s = round(time.monotonic() - t0, 3)
            if not stream:
                # Preserve the full non-stream response for debugging flaky concurrency cases.
                check.raw_full_response = raw.decode("utf-8", errors="replace")[:30000]
            check.result = evaluate_usage(check)
        except Exception as e:
            check.result = "ERROR"
            check.error = str(e)

        return check

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(run_one, 1, stream=True)
        f2 = pool.submit(run_one, 2, stream=False)
        for f in as_completed([f1, f2]):
            try:
                checks.append(f.result())
            except Exception as e:
                errors.append(str(e))

    # Pad if needed
    while len(checks) < 2:
        checks.append(UsageCheck(
            scenario=f"concurrent_missing", group="concurrency",
            endpoint="", stream=False, had_usage_key=False,
            had_timings_key=False, had_finish_reason=False,
            prompt_tokens=None, input_tokens=None,
            completion_tokens=None, total_tokens=None,
            cached_tokens=None, cache_read=None,
            result="ERROR", error="Request did not complete"
        ))

    # Validate both got usage
    for c in checks:
        if c.had_usage_key and (c.prompt_tokens is None or c.prompt_tokens == 0):
            report.findings.append(
                f"[BUG?] {c.scenario}: had usage key but prompt_tokens is None/0"
            )

    return checks[0], checks[1]


def test_rapid_sequential(report: TestReport, base_url: str, api_key: str) -> list[UsageCheck]:
    """Group 3: 3 rapid sequential requests."""
    results: list[UsageCheck] = []

    for i in range(3):
        check = UsageCheck(
            scenario=f"rapid_seq_{i+1}",
            group="concurrency",
            endpoint="/v1/chat/completions",
            stream=(i % 2 == 0),  # alternate stream/non-stream
            had_usage_key=False,
            had_timings_key=False,
            had_finish_reason=False,
            prompt_tokens=None,
            input_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            cached_tokens=None,
            cache_read=None,
            result="",
        )

        body = {
            "model": "PAQ_LLAMACPP_SERVER",
            "messages": [{"role": "user", "content": f"Rapid {i+1}: ok"}],
            "max_tokens": 5,
            "stream": check.stream,
        }

        try:
            t0 = time.monotonic()
            with make_request(base_url, "/v1/chat/completions", body, api_key) as resp:
                if check.stream:
                    events, _ = read_sse_stream(resp)
                    data_events = [e for e in events if e[0] == "data"]
                    if data_events:
                        _, _, payload = find_last_json_event(data_events)
                        if payload:
                            ui = extract_usage_from_payload(payload)
                            check.had_usage_key = ui.get("had_usage_key", False)
                            check.prompt_tokens = ui.get("prompt_tokens")
                            check.completion_tokens = ui.get("completion_tokens")
                            check.total_tokens = ui.get("total_tokens")
                            check.cached_tokens = ui.get("cached_tokens")
                            check.had_finish_reason = extract_finish_reason(payload) is not None
                            ti = extract_timings_from_payload(payload)
                            check.had_timings_key = ti.get("had_timings", False)
                            # If no usage but has timings, synthesize usage from timings
                            if not check.had_usage_key and ti.get("had_timings"):
                                fill_usage_from_timings(check, ti)

                else:
                    raw = resp.read()
                    payload = try_parse_json(raw)
                    if payload:
                        ui = extract_usage_from_payload(payload)
                        check.had_usage_key = ui.get("had_usage_key", False)
                        check.prompt_tokens = ui.get("prompt_tokens")
                        check.completion_tokens = ui.get("completion_tokens")
                        check.total_tokens = ui.get("total_tokens")
                        check.cached_tokens = ui.get("cached_tokens")
                        check.had_finish_reason = extract_finish_reason(payload) is not None
                        check.had_timings_key = extract_timings_from_payload(payload).get("had_timings", False)

            check.response_time_s = round(time.monotonic() - t0, 3)
            check.result = evaluate_usage(check)
        except Exception as e:
            check.result = "ERROR"
            check.error = str(e)

        results.append(check)

    return results


# ---------------------------------------------------------------------------
# Usage evaluation
# ---------------------------------------------------------------------------


def evaluate_usage(check: UsageCheck) -> str:
    """Determine if usage data is complete, partial, or missing."""
    # Check if we have ANY token count
    any_token = (
        check.prompt_tokens or check.input_tokens or
        check.completion_tokens or check.total_tokens
    )

    if check.had_usage_key:
        if check.prompt_tokens or check.input_tokens or check.completion_tokens:
            if (
                (check.prompt_tokens or check.input_tokens) and
                check.completion_tokens and
                check.total_tokens
            ):
                return "OK"
            elif check.completion_tokens is not None:
                # Has completion tokens — at least partial
                missing = []
                if not check.prompt_tokens and not check.input_tokens:
                    missing.append("prompt/input")
                if not check.total_tokens:
                    missing.append("total")
                return f"PARTIAL (missing: {', '.join(missing)})"
            else:
                return "PARTIAL (only has usage key, no effective tokens)"
        else:
            return "PARTIAL (has usage key, no meaningful tokens)"
    elif check.had_timings_key:
        # Has timings but no usage — proxy might need to synthesize
        return "MISSING_USAGE (timings present, usage missing)"
    elif any_token:
        return "PARTIAL (has token data somehow)"
    else:
        return "MISSING_USAGE"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(report: TestReport) -> None:
    """Write JSON and Markdown reports to benchmarks/."""
    ts = report.timestamp.replace(":", "").replace(" ", "-")
    json_path = RESULTS_DIR / f"{ts}-context-usage-test.json"
    md_path = RESULTS_DIR / f"{ts}-context-usage-test.md"
    usage_source_counts = build_usage_source_counts(report.scenarios)

    # JSON report
    report_data = {
        "timestamp": report.timestamp,
        "proxy_url": report.proxy_url,
        "summary": {
            "ok": report.ok_count,
            "missing": report.missing_count,
            "partial": report.partial_count,
            "error": report.error_count,
            "total": len(report.scenarios),
            "usage_source_counts": usage_source_counts,
        },
        "findings": report.findings,
        "scenarios": report.scenarios,
    }

    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Report] JSON: {json_path}")

    # Markdown report
    lines = [
        f"# Context Usage Test Report",
        f"",
        f"**Timestamp:** {report.timestamp}",
        f"**Proxy:** {report.proxy_url}",
        f"",
        f"## Summary",
        f"",
        f"| Result | Count |",
        f"|--------|-------|",
        f"| OK | {report.ok_count} |",
        f"| MISSING | {report.missing_count} |",
        f"| PARTIAL | {report.partial_count} |",
        f"| ERROR | {report.error_count} |",
        f"",
        f"### Usage source breakdown",
        f"",
        f"| Source | Count |",
        f"|--------|-------|",
        f"| explicit | {usage_source_counts['explicit']} |",
        f"| timings_only | {usage_source_counts['timings_only']} |",
        f"| none | {usage_source_counts['none']} |",
        f"",
        f"## Findings",
        f"",
    ]

    if report.findings:
        for finding in report.findings:
            lines.append(f"- {finding}")
    else:
        lines.append("No issues detected. All scenarios reported usage correctly.")

    lines.extend([
        f"",
        f"## Scenario Details",
        f"",
        f"| Scenario | Endpoint | Stream | Usage | Timings | Source | Prompt Tok | Output Tok | Cached | Result |",
        f"|----------|----------|--------|-------|---------|--------|------------|------------|--------|--------|",
    ])

    for s in report.scenarios:
        sc = s.get("scenario", "?")
        ep = s.get("endpoint", "?")
        st = "Y" if s.get("stream") else "N"
        usage = "Y" if s.get("had_usage_key") else "N"
        timings = "Y" if s.get("had_timings_key") else "N"
        source = classify_usage_source(s)
        pt = s.get("prompt_tokens") or "-"
        ct = s.get("completion_tokens") or "-"
        cached = s.get("cached_tokens") or "-"
        result = s.get("result", "?")

        # Truncate result for table
        if len(result) > 40:
            result = result[:37] + "..."

        lines.append(f"| {sc} | {ep} | {st} | {usage} | {timings} | {source} | {pt} | {ct} | {cached} | {result} |")

    lines.extend([
        f"",
        f"## Raw Final Chunks (for scenarios with issues)",
        f"",
    ])

    for s in report.scenarios:
        if s.get("result", "").startswith(("MISSING", "PARTIAL", "ERROR")):
            lines.append(f"### {s.get('scenario', '?')}")
            lines.append(f"**Result:** {s.get('result', '?')}")
            if s.get("error"):
                lines.append(f"**Error:** {s['error']}")
            if s.get("raw_final_chunk"):
                raw = s["raw_final_chunk"]
                # Clean up for markdown — truncate very long content
                if len(raw) > 4000:
                    raw = raw[:4000] + "\n... [truncated]"
                lines.append(f"```\n{raw}\n```")
            lines.append(f"")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Report] Markdown: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Context Usage Bug Detection Test")
    print("=" * 70)

    # Setup
    api_key = read_api_key()
    print(f"[Setup] API key loaded from {API_KEY_FILE}")

    base_url = detect_proxy_url()
    print(f"[Setup] Using proxy at {base_url}")

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report = TestReport(timestamp=ts, proxy_url=base_url)

    all_checks: list[UsageCheck] = []

    # Group 1: Basic OpenAI
    print(f"\n{'='*70}")
    print("Group 1: Basic OpenAI API paths")
    print(f"{'='*70}")

    print("\n[1/4] Streaming chat (simple)...")
    c = test_openai_stream_simple(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens}, cached={c.cached_tokens})")

    print("\n[2/4] Non-streaming chat (simple)...")
    c = test_openai_nostream_simple(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens}, cached={c.cached_tokens})")

    print("\n[3/4] Streaming with JSON mode...")
    c = test_openai_stream_json_mode(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens})")

    print("\n[4/4] Non-streaming with JSON mode...")
    c = test_openai_nostream_json_mode(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens})")

    # Group 2: Edge cases (tool calls, vision, short/long prompts)
    print(f"\n{'='*70}")
    print("Group 2: Edge cases (tool calls, vision, short/long prompts)")
    print(f"{'='*70}")

    print("\n[1/5] Tool calling (streaming)...")
    c = test_tool_calling_stream(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens})")

    print("\n[2/5] Vision request (non-streaming)...")
    c = test_vision_nostream(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens})")

    print("\n[3/5] Short prompt 'Hi' (streaming)...")
    c = test_short_prompt_stream(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens})")

    print("\n[4/5] Long prompt (non-streaming)...")
    c = test_long_prompt_nostream(report, base_url, api_key)
    all_checks.append(c)
    print(f"       -> {c.result} (usage={c.had_usage_key}, prompt_toks={c.prompt_tokens})")

    # Group 2: Cache behavior (cold vs warm)
    print(f"\n{'='*70}")
    print("Group 2: Cache behavior (cold vs warm)")
    print(f"{'='*70}")

    print("\n[1/2] Same prompt — COLD (streaming)...")
    c1, c2 = test_cache_cold_then_warm(report, base_url, api_key)
    all_checks.extend([c1, c2])
    print(f"       Cold  -> {c1.result} (cached={c1.cached_tokens}, prompt={c1.prompt_tokens})")
    print(f"       Warm  -> {c2.result} (cached={c2.cached_tokens}, prompt={c2.prompt_tokens})")

    # Group 3: Concurrency (PARALLEL=1)
    print(f"\n{'='*70}")
    print("Group 3: Concurrency (PARALLEL=1)")
    print(f"{'='*70}")

    print("\n[1/2] Concurrent requests (stream + non-stream)...")
    cc1, cc2 = test_concurrent_requests(report, base_url, api_key)
    all_checks.extend([cc1, cc2])
    print(f"       #1 -> {cc1.result} (usage={cc1.had_usage_key})")
    print(f"       #2 -> {cc2.result} (usage={cc2.had_usage_key})")

    print("\n[2/2] Rapid sequential (3 requests)...")
    rapid = test_rapid_sequential(report, base_url, api_key)
    all_checks.extend(rapid)
    for i, rc in enumerate(rapid):
        print(f"       #{i+1} -> {rc.result} (stream={'Y' if rc.stream else 'N'}, usage={rc.had_usage_key})")

    # Final summary
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    for s in all_checks:
        result_upper = s.result.upper()
        if result_upper.startswith("MISSING"):
            report.missing_count += 1
        elif result_upper.startswith("PARTIAL"):
            report.partial_count += 1
        elif result_upper.startswith("ERROR"):
            report.error_count += 1
        else:
            report.ok_count += 1

        report.scenarios.append(asdict(s))

    total = len(all_checks)
    print(f"\nTotal scenarios: {total}")
    print(f"  OK:      {report.ok_count}")
    print(f"  MISSING: {report.missing_count}")
    print(f"  PARTIAL: {report.partial_count}")
    print(f"  ERROR:   {report.error_count}")
    usage_source_counts = build_usage_source_counts(report.scenarios)
    print(f"  Usage sources:")
    print(f"    explicit:     {usage_source_counts['explicit']}")
    print(f"    timings_only: {usage_source_counts['timings_only']}")
    print(f"    none:         {usage_source_counts['none']}")

    if report.findings:
        print(f"\n⚠ Findings ({len(report.findings)}):")
        for f in report.findings:
            print(f"  {f}")
    elif report.missing_count == 0 and report.partial_count == 0 and report.error_count == 0:
        print(f"\n✓ No issues detected.")

    if report.missing_count > 0 or report.partial_count > 0:
        print(f"\n🔍 BUG DETECTED: {report.missing_count} missing + {report.partial_count} partial usage reports.")
        print("   See raw_final_chunk in the JSON report for affected scenarios.")

    generate_report(report)

    # Exit code based on findings
    if report.missing_count > 0 or report.partial_count > 0 or report.error_count > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
