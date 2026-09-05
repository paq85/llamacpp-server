#!/usr/bin/env python3
"""Small front proxy for keeping Cloudflare requests alive during slow first-byte inference.

Modes:

* ``stream``: for streaming OpenAI requests, send an
    immediate SSE comment and periodic SSE keep-alive comments while waiting for the
    upstream response headers.
* ``optimistic``: for selected JSON inference endpoints, immediately return a chunked
  ``200`` response and emit whitespace heartbeats until the upstream response arrives.
  This keeps the body JSON-compatible, but upstream non-2xx status codes cannot be
  preserved once the optimistic response has started.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import http.client
import signal
import ssl
import sys
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, cast

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

OPENAI_INFERENCE_PATHS = {
    "/completion",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
}

INFERENCE_PATHS = OPENAI_INFERENCE_PATHS

DEFAULT_CORS_ALLOW_HEADERS = "Authorization, Content-Type, X-API-Key"
MODEL_LIST_PATHS = {"/models", "/v1/models"}

# llama.cpp resumable-stream session lookup endpoint. GitHub's BYOK
# configuration UI probes this path when connecting an OpenAI-compatible
# endpoint. Backend builds without the resumable-stream feature answer 404,
# which a strict client may read as "streaming unsupported"; the proxy answers
# with the llama.cpp "no live sessions" shape (200 []) instead.
STREAM_LOOKUP_PATHS = {"/v1/streams/lookup"}
MULTIMODAL_CAPABILITY_ALIASES = ("vision",)

_cost_dashboard = None
_dashboard_lock = threading.Lock()

def _read_model_alias_from_envfile() -> str | None:
    """Return MODEL_ALIAS from the repo .env file if present, else None."""
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if not os.path.exists(env_path):
            return None
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "MODEL_ALIAS":
                    return v.strip().strip('"\'')
    except Exception:
        return None
    return None


def _read_envfile_value(key: str) -> str | None:
    """Return KEY's value from the repo .env file if present, else None.

    Lets the proxy pick up a handful of toggles on a proxy-only restart (without
    a full launcher restart), since ``run-paq-llamacpp-server.sh`` only sources ``.env`` once
    at service start. Strips inline ``#`` comments and surrounding quotes.
    """
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if not os.path.exists(env_path):
            return None
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.split("#", 1)[0].strip().strip('"\'')
    except Exception:
        return None
    return None


def _append_unique_strings(values: list[Any], additions: Iterable[str]) -> bool:
    existing = {value for value in values if isinstance(value, str)}
    changed = False
    for addition in additions:
        if addition in existing:
            continue
        values.append(addition)
        existing.add(addition)
        changed = True
    return changed


def _add_model_capability_aliases(payload: object) -> bool:
    """Add local compatibility capability tags to llama.cpp model-list payloads.

    llama.cpp advertises the MTMD stack as ``multimodal``. Some clients look for
    narrower tags such as ``vision``. This proxy serves a vision-enabled stack, so
    expose that alias without patching the upstream server source.
    """
    if not isinstance(payload, dict):
        return False

    models = payload.get("models")
    if not isinstance(models, list):
        return False

    changed = False
    for model in models:
        if not isinstance(model, dict):
            continue
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, list):
            continue
        if "multimodal" not in capabilities:
            continue
        changed |= _append_unique_strings(capabilities, MULTIMODAL_CAPABILITY_ALIASES)
    return changed


def _add_data_entry_capabilities(payload: object) -> bool:
    """Mirror llama.cpp ``models[]`` capabilities onto OpenAI-standard ``data[]`` entries.

    llama.cpp's ``/v1/models`` only advertises capabilities in its own
    ``models[]`` format. OpenAI-shaped clients (including GitHub's BYOK model
    discovery) read the ``data[]`` entries instead, and a missing
    ``capabilities`` field makes a multimodal model look text-only — so the
    Vision checkbox in GitHub's custom-model configuration can be rejected or
    dropped. Copy the (already alias-normalized) capabilities list onto each
    matching ``data`` entry so discovery cannot miss the multimodal stack.

    Matching is by model id, with a fallback through the entry's ``aliases``
    list for clients that address the model by its secondary alias.
    """
    if not isinstance(payload, dict):
        return False

    models = payload.get("models")
    data = payload.get("data")
    if not isinstance(models, list) or not isinstance(data, list):
        return False

    # Build id -> capabilities lookup from the llama.cpp models[] section.
    by_id: dict[str, list[Any]] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("name") or model.get("model")
        capabilities = model.get("capabilities")
        if isinstance(model_id, str) and isinstance(capabilities, list):
            by_id[model_id] = capabilities

    changed = False
    for entry in data:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str):
            continue
        capabilities = by_id.get(entry_id)
        if capabilities is None:
            aliases = entry.get("aliases")
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias in by_id:
                        capabilities = by_id[alias]
                        break
        if capabilities is None or "multimodal" not in capabilities:
            continue
        existing = entry.get("capabilities")
        if isinstance(existing, list):
            if existing == capabilities:
                continue
            changed |= _append_unique_strings(existing, capabilities)
        else:
            entry["capabilities"] = list(capabilities)
            changed = True
    return changed


def rewrite_model_capabilities_payload(raw: bytes) -> bytes:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw

    changed = _add_model_capability_aliases(payload)
    changed |= _add_data_entry_capabilities(payload)

    if not changed:
        return raw

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _mirror_openai_reasoning_fields(payload: object) -> bool:
    """Mirror llama.cpp reasoning_content into fields OpenAI-compatible clients read.

    llama.cpp currently emits reasoning as ``reasoning_content`` for Chat
    Completions. VS Code Copilot's OpenAI-compatible BYOK parser looks for
    ``reasoning_text``, ``thinking``, or ``cot_summary`` on ``message``/``delta``
    objects. Keep the original field and add a compatibility alias when needed.
    """
    if not isinstance(payload, dict):
        return False

    changed = False
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            part = choice.get(key)
            if not isinstance(part, dict):
                continue
            reasoning = part.get("reasoning_content")
            if reasoning is None:
                continue
            if any(part.get(alias) is not None for alias in ("reasoning_text", "thinking", "cot_summary")):
                continue
            part["reasoning_text"] = reasoning
            changed = True
    return changed


def rewrite_openai_reasoning_payload(raw: bytes) -> bytes:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw

    if not _mirror_openai_reasoning_fields(payload):
        return raw

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _rewrite_payload_model_fields(payload: object, requested_model: str) -> bool:
    if not isinstance(payload, dict):
        return False

    changed = False

    model = payload.get("model")
    if isinstance(model, str) and model != requested_model:
        payload["model"] = requested_model
        changed = True

    message = payload.get("message")
    if isinstance(message, dict):
        message_model = message.get("model")
        if isinstance(message_model, str) and message_model != requested_model:
            message["model"] = requested_model
            changed = True

    return changed


def rewrite_requested_model_payload(raw: bytes, requested_model: str | None) -> bytes:
    if not requested_model:
        return raw

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw

    if not _rewrite_payload_model_fields(payload, requested_model):
        return raw

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Inference request stabilization (sampling clamp + max_tokens floor)
#
# Real coding-agent clients (VS Code Copilot, GitHub Copilot CLI) override the
# server's recommended sampling and may omit an output-token limit. For the
# Qwen3.8 tool-calling stack this contributes to malformed / truncated tool
# calls (the model thinks first, then runs out of room, and the client receives
# invalid tool-call JSON and stops). These helpers optionally normalize the
# OUTGOING request before it reaches llama-server. All of this is env-gated and
# off unless explicitly configured, so default behavior is unchanged.
# ---------------------------------------------------------------------------

def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return None


def _env_bool_with_envfile(name: str) -> bool | None:
    raw = (_read_envfile_value(name) or "").strip().lower()
    if not raw:
        return _env_bool(name)
    if raw in {"1", "true", "yes", "on", "y"}:
        return True
    if raw in {"0", "false", "no", "off", "n"}:
        return False
    return None


def _env_text_with_envfile(name: str, default: str = "") -> str:
    raw = _read_envfile_value(name) or ""
    if raw:
        return raw
    raw = os.environ.get(name, "")
    return raw or default


DEFAULT_TOOL_RESULT_CONTINUATION_NUDGE = (
    "Continue after the latest tool result. Do not stop in the reasoning/thinking "
    "channel. Before finishing, you must emit either a valid tool_call or visible "
    "assistant content. If any planned work remains, call the next appropriate "
    "tool now. Do not merely describe the next step in reasoning."
)


@dataclass(frozen=True)
class RequestStabilizeConfig:
    """Env-driven, opt-in normalization of outgoing inference requests."""
    clamp_temperature: float | None = None  # cap temperature at this value
    clamp_top_p: float | None = None        # cap top_p at this value
    set_top_k: int | None = None            # set/cap top_k to this value
    min_max_tokens: int | None = None       # raise an existing too-small max_tokens
    max_tokens_ctx_pct: float | None = None # cap max_tokens at this % of ctx_size
    ctx_size: int | None = None             # context size for percentage-based ceiling
    nudge_after_tool_result: bool = False   # append hidden continuation nudge after tool results
    tool_result_nudge_text: str = DEFAULT_TOOL_RESULT_CONTINUATION_NUDGE

    @classmethod
    def from_env(cls) -> "RequestStabilizeConfig":
        return cls(
            clamp_temperature=_env_float("LLAMA_PROXY_CLAMP_TEMPERATURE"),
            clamp_top_p=_env_float("LLAMA_PROXY_CLAMP_TOP_P"),
            set_top_k=_env_int("LLAMA_PROXY_SET_TOP_K"),
            min_max_tokens=_env_int("LLAMA_PROXY_MIN_MAX_TOKENS"),
            max_tokens_ctx_pct=_env_float("LLAMA_PROXY_MAX_TOKENS_CTX_PCT"),
            ctx_size=_env_int("CTX_SIZE"),
            nudge_after_tool_result=bool(_env_bool_with_envfile("LLAMA_PROXY_NUDGE_AFTER_TOOL_RESULT")),
            tool_result_nudge_text=_env_text_with_envfile(
                "LLAMA_PROXY_TOOL_RESULT_NUDGE_TEXT",
                DEFAULT_TOOL_RESULT_CONTINUATION_NUDGE,
            ),
        )

    @property
    def active(self) -> bool:
        return any(v is not None for v in (
            self.clamp_temperature, self.clamp_top_p, self.set_top_k,
            self.min_max_tokens, self.max_tokens_ctx_pct)) \
            or self.nudge_after_tool_result


def _append_tool_result_continuation_nudge(payload: dict[str, Any], text: str) -> bool:
    """Append a hidden user nudge when the client asks right after a tool result.

    Real VS Code Copilot captures with Qwen3.8 show an intermittent failure mode:
    after a successful tool result, the model sometimes immediately emits EOS
    without a visible action. This can be a truly empty assistant message, or a
    reasoning-only response with ``finish_reason=stop`` and no content/tool call.
    Copilot hides reasoning and has no tool to execute, so the agent appears to
    stop mid-task. Adding a tiny continuation user message mirrors the context
    refresh Copilot sends when it later recovers, and avoids altering the tool
    output itself.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "tool":
        return False

    nudge = text.strip() or DEFAULT_TOOL_RESULT_CONTINUATION_NUDGE
    messages.append({"role": "user", "content": nudge})
    return True


def _stabilize_inference_payload(payload: dict[str, Any], cfg: RequestStabilizeConfig) -> bool:
    changed = False

    if cfg.clamp_temperature is not None:
        temp = payload.get("temperature")
        if isinstance(temp, (int, float)) and float(temp) > cfg.clamp_temperature:
            payload["temperature"] = cfg.clamp_temperature
            changed = True

    if cfg.clamp_top_p is not None:
        top_p = payload.get("top_p")
        if isinstance(top_p, (int, float)) and float(top_p) > cfg.clamp_top_p:
            payload["top_p"] = cfg.clamp_top_p
            changed = True

    if cfg.set_top_k is not None:
        top_k = payload.get("top_k")
        # llama.cpp accepts top_k; set it when missing or larger than the target.
        if not isinstance(top_k, int) or top_k <= 0 or top_k > cfg.set_top_k:
            payload["top_k"] = cfg.set_top_k
            changed = True

    if cfg.min_max_tokens is not None:
        # IMPORTANT: only RAISE an existing, too-small limit. Never ADD a limit
        # when the client sent none (None == unlimited); adding one would
        # introduce truncation rather than prevent it.
        for key in ("max_tokens", "max_completion_tokens"):
            val = payload.get(key)
            if isinstance(val, int) and 0 < val < cfg.min_max_tokens:
                payload[key] = cfg.min_max_tokens
                changed = True

    if cfg.max_tokens_ctx_pct is not None and cfg.ctx_size is not None and cfg.ctx_size > 0:
        # CEILING: cap max_tokens at pct% of context size. Unlike the floor,
        # this DOES impose a limit when the client sends none — unbounded
        # generation can exhaust the context window and cause truncation.
        max_allowed = int(cfg.ctx_size * cfg.max_tokens_ctx_pct / 100.0)
        if max_allowed < 1:
            max_allowed = 1
        for key in ("max_tokens", "max_completion_tokens"):
            val = payload.get(key)
            if isinstance(val, int) and val > max_allowed:
                payload[key] = max_allowed
                changed = True
            elif not isinstance(val, int) or val <= 0:
                # Client sent no limit (or sent 0/unlimited) — impose the ceiling.
                payload[key] = max_allowed
                changed = True

    if cfg.nudge_after_tool_result:
        changed |= _append_tool_result_continuation_nudge(payload, cfg.tool_result_nudge_text)

    return changed


def stabilize_inference_request_body(raw: bytes, cfg: RequestStabilizeConfig) -> bytes:
    if not cfg.active or not raw:
        return raw
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw
    if not isinstance(payload, dict):
        return raw
    if not _stabilize_inference_payload(payload, cfg):
        return raw
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Usage cost injection
# ---------------------------------------------------------------------------

def _usage_lookup(usage: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = usage
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _usage_first_value(usage: dict[str, Any], *paths: tuple[str, ...]) -> Any | None:
    for path in paths:
        value = _usage_lookup(usage, path)
        if value is not None:
            return value
    return None


def _usage_first_int(usage: dict[str, Any], *paths: tuple[str, ...]) -> int | None:
    value = _usage_first_value(usage, *paths)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _set_usage_value(container: dict[str, Any], key: str, value: Any) -> bool:
    if key not in container or container[key] is None:
        container[key] = value
        return True
    return False


def _normalize_usage_aliases(usage: dict[str, Any]) -> bool:
    changed = False

    prompt = _usage_first_int(
        usage,
        ("input",),
        ("inputTokens",),
        ("input_tokens",),
        ("promptTokens",),
        ("prompt_tokens",),
        ("prompt_n",),
        ("timings", "prompt_n"),
    )
    completion = _usage_first_int(
        usage,
        ("output",),
        ("outputTokens",),
        ("output_tokens",),
        ("completionTokens",),
        ("completion_tokens",),
        ("predicted_n",),
        ("timings", "predicted_n"),
    )
    total = _usage_first_int(usage, ("total",), ("totalTokens",), ("total_tokens",))
    cached = _usage_first_int(
        usage,
        ("cacheRead",),
        ("cache_read",),
        ("cache_read_input_tokens",),
        ("cached_tokens",),
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("timings", "cache_n"),
    )
    cache_write = _usage_first_int(
        usage,
        ("cacheWrite",),
        ("cache_write",),
        ("cache_creation_input_tokens",),
    )

    if prompt is not None:
        for key in ("input", "inputTokens", "input_tokens", "promptTokens", "prompt_tokens", "prompt_n"):
            existing = usage.get(key)
            # Overwrite if missing, None, or has a different non-zero value
            if existing is None or not isinstance(existing, (int, float)) or int(existing) == 0 or int(existing) != prompt:
                usage[key] = prompt
                changed = True
        timings = usage.get("timings")
        if not isinstance(timings, dict):
            timings = {}
            usage["timings"] = timings
            changed = True
        if timings.get("prompt_n") != prompt:
            timings["prompt_n"] = prompt
            changed = True

    if completion is not None:
        for key in ("output", "outputTokens", "output_tokens", "completionTokens", "completion_tokens", "predicted_n"):
            existing = usage.get(key)
            if existing is None or not isinstance(existing, (int, float)) or int(existing) == 0 or int(existing) != completion:
                usage[key] = completion
                changed = True
        timings = usage.get("timings")
        if not isinstance(timings, dict):
            timings = {}
            usage["timings"] = timings
            changed = True
        if timings.get("predicted_n") != completion:
            timings["predicted_n"] = completion
            changed = True

    expected_total = None
    if prompt is not None and completion is not None:
        expected_total = prompt + completion
        if expected_total != total:
            total = expected_total
    if total is not None:
        for key in ("total", "totalTokens", "total_tokens"):
            existing = usage.get(key)
            if existing is None or not isinstance(existing, (int, float)) or int(existing) == 0 or int(existing) != total:
                usage[key] = total
                changed = True

    if cached is not None:
        for key in ("cacheRead", "cache_read", "cache_read_input_tokens", "cached_tokens"):
            existing = usage.get(key)
            if existing is None or not isinstance(existing, (int, float)) or int(existing) == 0 or int(existing) != cached:
                usage[key] = cached
                changed = True

        prompt_details = usage.get("prompt_tokens_details")
        if not isinstance(prompt_details, dict):
            prompt_details = {}
            usage["prompt_tokens_details"] = prompt_details
            changed = True
        input_details = usage.get("input_tokens_details")
        if not isinstance(input_details, dict):
            input_details = {}
            usage["input_tokens_details"] = input_details
            changed = True

        # Always ensure cached_tokens is set on both detail dicts
        if prompt_details.get("cached_tokens") != cached:
            prompt_details["cached_tokens"] = cached
            changed = True
        if input_details.get("cached_tokens") != cached:
            input_details["cached_tokens"] = cached
            changed = True

    # Ensure input_tokens_details exists even when cached is None (some clients
    # check for this key's presence before reading cached_tokens).
    else:
        input_details = usage.get("input_tokens_details")
        if input_details is None:
            usage["input_tokens_details"] = {}
            changed = True

    return changed


def _extract_usage_cost(usage: dict[str, Any], config: ProxyConfig) -> dict[str, object] | None:
    """Compute per-bucket cost from a usage object using server-side prices.

    Handles OpenAI-style, llama.cpp, and OpenClaw-friendly
    field names so the proxy can normalize upstream usage while staying
    compatible with existing clients.
    """
    prompt = _usage_first_int(
        usage,
        ("input",),
        ("inputTokens",),
        ("input_tokens",),
        ("promptTokens",),
        ("prompt_tokens",),
        ("prompt_n",),
        ("timings", "prompt_n"),
    ) or 0
    completion = _usage_first_int(
        usage,
        ("output",),
        ("outputTokens",),
        ("output_tokens",),
        ("completionTokens",),
        ("completion_tokens",),
        ("predicted_n",),
        ("timings", "predicted_n"),
    ) or 0

    # Cached / cache-read tokens.
    cached = _usage_first_int(
        usage,
        ("cacheRead",),
        ("cache_read",),
        ("cache_read_input_tokens",),
        ("cached_tokens",),
        ("prompt_tokens_details", "cached_tokens"),
        ("input_tokens_details", "cached_tokens"),
        ("timings", "cache_n"),
    ) or 0

    # Cache-write tokens (not reported by OpenAI, so we
    # assume 0 unless present).
    cache_write = _usage_first_int(
        usage,
        ("cacheWrite",),
        ("cache_write",),
        ("cache_creation_input_tokens",),
    ) or 0

    # Non-cached (fresh) prompt tokens.
    # fresh = total prompt - cached - cache_write
    fresh = max(0, prompt - cached - cache_write)

    _1m = 1_000_000.0
    cfg = config
    input_cost = round((fresh * cfg.cost_input_price + cache_write * cfg.cost_input_price) / _1m, 8) if (cfg.cost_input_price or (fresh + cache_write)) else 0.0
    cached_cost = round(cached * cfg.cost_cached_price / _1m, 8) if cfg.cost_cached_price else 0.0
    output_cost = round(completion * cfg.cost_output_price / _1m, 8) if cfg.cost_output_price else 0.0

    total_cost = round(input_cost + cached_cost + output_cost, 8)

    # Only return a non-trivial cost dict if there's something meaningful.
    if total_cost == 0.0 and not (cfg.cost_input_price or cfg.cost_cached_price or cfg.cost_output_price):
        return None

    return {
        "input_cost": _safe_fmt(input_cost),
        "cached_cost": _safe_fmt(cached_cost),
        "output_cost": _safe_fmt(output_cost),
        "total_cost": _safe_fmt(total_cost),
    }


def _safe_fmt(value: float) -> float:
    """Round to 8 decimal places to avoid floating-point noise."""
    return round(value, 8)


def _synthesize_usage_from_timings(payload: dict[str, Any]) -> dict[str, Any] | None:
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        return None

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None

    has_final_choice = any(isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)
    if not has_final_choice:
        return None

    prompt_n = _usage_first_int(payload, ("timings", "prompt_n")) or 0
    cached = _usage_first_int(payload, ("timings", "cache_n")) or 0
    completion = _usage_first_int(payload, ("timings", "predicted_n")) or 0

    # `timings.prompt_n` is the number of *newly processed* tokens (cache misses),
    # while `timings.cache_n` is the number of cache hits.  Clients such as
    # VS Code Copilot use `prompt_tokens`/`input_tokens` as context-length
    # indicators, so we must report the *total* input token count.
    total_prompt = prompt_n + cached

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": completion,
        "total_tokens": total_prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def _inject_usage_cost_into_payload(payload: object, config: ProxyConfig | None = None) -> object:
    """Normalize usage aliases, inject cost fields, and add a top-level ``cost``.

    Matches the OpenAI-compatible convention that OpenClaw and similar clients
    already read from ``usage``.  A top-level ``cost`` dict is added as a
    convenience for clients that don't traverse ``usage``.
    """
    if not isinstance(payload, dict):
        return payload

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return payload

    _normalize_usage_aliases(usage)
    if config is None:
        return payload

    cost = _extract_usage_cost(usage, config)
    if cost is None:
        return payload

    # Merge cost into usage (as usage.cost — compatible with OpenClaw's transcript shape).
    usage["cost"] = cost
    payload["cost"] = cost
    return payload


def rewrite_usage_cost_payload(raw: bytes, config: ProxyConfig | None = None, *, synthesize_from_timings: bool = False) -> bytes:
    """Decode JSON, normalize usage aliases, inject cost, re-encode.

    Also logs the request to the cost CSV if a CostLogger is active.

    Returns *raw* unchanged on errors.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return raw

    if synthesize_from_timings and isinstance(payload, dict) and not isinstance(payload.get("usage"), dict):
        synthesized = _synthesize_usage_from_timings(payload)
        if synthesized is not None:
            payload["usage"] = synthesized

    rewritten = _inject_usage_cost_into_payload(payload, config)

    # Log to CSV logger if available and config has pricing set.
    if isinstance(payload, dict) and config is not None and (config.cost_input_price or config.cost_cached_price or config.cost_output_price):
        _log_usage_from_payload(payload, config)

    return json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _log_usage_from_payload(payload: dict[str, Any], config: ProxyConfig) -> None:
    """Extract token counts and costs from a finalized payload and log to CSV."""
    logger = _get_cost_logger()
    if logger is None:
        return

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return

    # Tokens (already normalized by _normalize_usage_aliases).
    input_tokens = _usage_first_int(usage, ("input",), ("input_tokens",), ("prompt_tokens",), ("prompt_n",), ("timings", "prompt_n")) or 0
    output_tokens = _usage_first_int(usage, ("output",), ("output_tokens",), ("completion_tokens",), ("predicted_n",), ("timings", "predicted_n")) or 0
    cached_tokens = _usage_first_int(usage, ("cacheRead",), ("cache_read",), ("cached_tokens",), ("timings", "cache_n")) or 0

    # Costs — re-compute from the same formula as _extract_usage_cost to match exactly.
    cost = payload.get("cost")
    if isinstance(cost, dict):
        input_cost = float(cost.get("input_cost", 0)) if config.cost_input_price else 0.0
        cached_cost = float(cost.get("cached_cost", 0)) if config.cost_cached_price else 0.0
        output_cost = float(cost.get("output_cost", 0)) if config.cost_output_price else 0.0
        total_cost = float(cost.get("total_cost", 0))
    else:
        input_cost = 0.0
        cached_cost = 0.0
        output_cost = 0.0
        total_cost = 0.0

    model = payload.get("model") if isinstance(payload.get("model"), str) else None

    try:
        logger.log_request(
            model=model,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            cached_cost=cached_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            status=200,
        )
    except Exception:
        # Never let logging failures interrupt the response path.
        try:
            sys.stderr.write(f"cost logger error (suppressed): {sys.exc_info()[1]}\n")
        except Exception:
            pass


def rewrite_json_sse_line(
    line: bytes,
    *,
    rewrite_openai_reasoning: bool = False,
    requested_model: str | None = None,
    cost_config: ProxyConfig | None = None,
) -> bytes:
    stripped = line.rstrip(b"\r\n")
    newline = line[len(stripped):]

    if not stripped.startswith(b"data:"):
        return line

    prefix = b"data:"
    data = stripped[len(prefix):]
    leading_space = b""
    while data.startswith(b" "):
        leading_space += b" "
        data = data[1:]

    if data.strip() == b"[DONE]":
        return line

    rewritten = data
    if rewrite_openai_reasoning:
        rewritten = rewrite_openai_reasoning_payload(rewritten)
    if requested_model:
        rewritten = rewrite_requested_model_payload(rewritten, requested_model)

    # Normalize usage aliases and inject cost. Final streamed chunks may only
    # carry timings, so synthesize a usage object from timings when needed.
    if b'"usage"' in rewritten or (b'"timings"' in rewritten and b'"choices"' in rewritten and b'"finish_reason"' in rewritten):
        rewritten = rewrite_usage_cost_payload(rewritten, cost_config, synthesize_from_timings=True)

    if rewritten is data or rewritten == data:
        return line
    return prefix + leading_space + rewritten + newline


def rewrite_openai_reasoning_sse_line(line: bytes) -> bytes:
    return rewrite_json_sse_line(line, rewrite_openai_reasoning=True)


def get_api_family(path: str) -> str:
    if path in OPENAI_INFERENCE_PATHS:
        return "openai"
    return "strict"


def _stream_keepalive_mode_from_env() -> str:
    """How to keep a *waiting* streaming client alive: ``comment`` or ``data``.

    While the single backend slot is busy with another request, the proxy emits
    periodic keep-alives so the connection (and Cloudflare tunnel) stays open.

    * ``comment`` (default): an SSE comment ``: keep-alive``. It keeps the socket
      alive but is NOT surfaced as an event by every SSE client, so it may fail
      to reset a client's time-to-first-token / inactivity timer (observed with
      GitHub Copilot stalling while queued behind a second window).
    * ``data``: an OpenAI-shaped ``chat.completion.chunk`` with an EMPTY delta —
      a real ``data:`` event that resets client content timers while adding
      nothing to the assembled message.
    """
    mode = os.environ.get("LLAMA_PROXY_STREAM_KEEPALIVE_MODE", "").strip().lower()
    if not mode:
        # Fall back to the .env file so a proxy-only restart applies it without
        # a full (model-reloading) launcher restart.
        mode = (_read_envfile_value("LLAMA_PROXY_STREAM_KEEPALIVE_MODE") or "").strip().lower()
    return mode if mode in {"comment", "data", "reasoning"} else "comment"


def _stream_keepalive_text_from_env() -> str:
    """Reasoning text emitted per keep-alive in ``reasoning`` mode (default '.')."""
    txt = os.environ.get("LLAMA_PROXY_STREAM_KEEPALIVE_TEXT", "")
    if not txt:
        txt = _read_envfile_value("LLAMA_PROXY_STREAM_KEEPALIVE_TEXT") or ""
    return txt or "."


def _build_stream_keepalive_chunk(model: str | None) -> bytes:
    """Empty-delta OpenAI streaming chunk used as a content-timer-safe keep-alive.

    The empty ``delta`` means the chunk contributes no content/role/tool data to
    the response the client assembles; it exists only to be a real SSE event.
    """
    payload = {
        "id": "chatcmpl-proxy-keepalive",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model or os.environ.get("MODEL_ALIAS", "") or "model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }
    return b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


def _build_stream_keepalive_reasoning_chunk(model: str | None, text: str) -> bytes:
    """Reasoning-delta keep-alive: a real ``reasoning_content`` streaming chunk.

    Unlike an empty delta, this carries actual streamed content, which is the
    strongest signal to reset a strict client's time-to-first-token timer while
    a request waits for the single busy backend slot. The text lands in the
    model's *reasoning* (thinking) channel — matching the backend's own
    ``reasoning_content`` field — so it is rendered as collapsible thinking and
    never pollutes the final answer text or tool-call arguments.
    """
    payload = {
        "id": "chatcmpl-proxy-keepalive",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model or os.environ.get("MODEL_ALIAS", "") or "model",
        "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None}],
    }
    return b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


@dataclass(frozen=True)
class ProxyConfig:
    listen_host: str
    listen_port: int
    target_scheme: str
    target_host: str
    target_port: int
    mode: str
    heartbeat_interval: float
    upstream_timeout: float
    read_chunk_size: int
    frontend_tls_cert_file: str = ""
    frontend_tls_key_file: str = ""
    api_keys: frozenset[str] = field(default_factory=lambda: frozenset())
    # USD per 1M tokens — injected into usage object for client cost tracking.
    cost_input_price: float = 0.0
    cost_cached_price: float = 0.0
    cost_output_price: float = 0.0
    cost_log_dir: str = ""
    # Opt-in outgoing-request normalization (sampling clamp + max_tokens floor).
    stabilize: "RequestStabilizeConfig" = field(default_factory=lambda: RequestStabilizeConfig())
    # Keep-alive style for waiting streaming clients: "comment", "data", or "reasoning".
    stream_keepalive_mode: str = "comment"
    # Reasoning text emitted per keep-alive when mode == "reasoning".
    stream_keepalive_text: str = "."


@dataclass
class UpstreamResult:
    connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
    response: http.client.HTTPResponse | None = None
    error: Exception | None = None


class DownstreamWriteError(Exception):
    """Raised when the client-facing socket closes while we are still proxying."""


# ---------------------------------------------------------------------------
# Cost logging (per-request CSV + daily summary)
# ---------------------------------------------------------------------------

REQUESTS_HEADER = [
    "timestamp",
    "model",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost",
    "cached_cost",
    "output_cost",
    "total_cost",
    "status",
]

DAILY_HEADER = [
    "date",
    "requests",
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost",
    "cached_cost",
    "output_cost",
    "total_cost",
]


@dataclass
class _DailyTotals:
    requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    cached_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _to_row(self, day: date) -> list[str | int | float]:
        return [
            str(day),
            self.requests,
            self.input_tokens,
            self.cached_tokens,
            self.output_tokens,
            self.total_tokens,
            f"{self.input_cost:.8f}",
            f"{self.cached_cost:.8f}",
            f"{self.output_cost:.8f}",
            f"{self.total_cost:.8f}",
        ]


class CostLogger:
    """Thread-safe CSV cost logger with per-request and daily summary."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._requests_path = os.path.join(log_dir, "requests.csv")
        self._daily_path = os.path.join(log_dir, "daily-summary.csv")
        self._requests_file = open(self._requests_path, "a", newline="", encoding="utf-8")
        self._daily_file = open(self._daily_path, "a", newline="", encoding="utf-8")
        self._requests_writer = csv.writer(self._requests_file)
        self._daily_writer = csv.writer(self._daily_file)

        # Ensure CSV headers exist.
        self._ensure_header(self._requests_file, self._requests_writer, REQUESTS_HEADER)
        self._ensure_header(self._daily_file, self._daily_writer, DAILY_HEADER)

        # Recover today's totals from the authoritative per-request log.
        self._daily_totals = self._load_existing_daily_totals(date.today())

        self._current_date = date.today()

    @staticmethod
    def _ensure_header(fh, writer, header: list[str]) -> None:
        fh.seek(0, 2)  # seek to end
        if fh.tell() == 0:
            writer.writerow(header)
            fh.flush()

    # -- startup re-accumulation ---------------------------------------------

    def _load_existing_daily_totals(self, today: date) -> _DailyTotals:
        """Read today's rows from the per-request log and sum them."""
        if not os.path.exists(self._requests_path):
            return _DailyTotals()

        totals = _DailyTotals()
        today_prefix = str(today)  # "2026-06-06" — timestamps start with this
        with open(self._requests_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 11:
                    continue
                if not row[0].startswith(today_prefix):
                    continue
                # columns: timestamp, model, input_tokens, cached_tokens,
                #          output_tokens, total_tokens, input_cost,
                #          cached_cost, output_cost, total_cost, status
                totals.requests += 1
                totals.input_tokens += int(row[2])
                totals.cached_tokens += int(row[3])
                totals.output_tokens += int(row[4])
                totals.input_cost += float(row[6])
                totals.cached_cost += float(row[7])
                totals.output_cost += float(row[8])
                totals.total_cost += float(row[9])
        return totals

    # -- public API ----------------------------------------------------------

    def log_request(
        self,
        model: str | None,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        input_cost: float,
        cached_cost: float,
        output_cost: float,
        total_cost: float,
        status: int = 200,
    ) -> None:
        """Record a single completed request and accumulate daily totals."""
        now = datetime.now()
        today = now.date()

        with self._lock:
            # Midnight rollover: flush previous day.
            if today != self._current_date:
                self._flush_daily_unlocked(self._current_date)
                self._current_date = today

            # Append per-request row.
            self._requests_writer.writerow([
                now.strftime("%Y-%m-%dT%H:%M:%S%z"),
                model or "",
                input_tokens,
                cached_tokens,
                output_tokens,
                input_tokens + output_tokens,
                f"{input_cost:.8f}",
                f"{cached_cost:.8f}",
                f"{output_cost:.8f}",
                f"{total_cost:.8f}",
                status,
            ])
            self._requests_file.flush()
            os.fsync(self._requests_file.fileno())

            # Accumulate daily totals.
            self._daily_totals.requests += 1
            self._daily_totals.input_tokens += input_tokens
            self._daily_totals.cached_tokens += cached_tokens
            self._daily_totals.output_tokens += output_tokens
            self._daily_totals.input_cost += input_cost
            self._daily_totals.cached_cost += cached_cost
            self._daily_totals.output_cost += output_cost
            self._daily_totals.total_cost += total_cost

    def flush(self) -> None:
        """Write any remaining daily totals and close files."""
        with self._lock:
            self.flush_unlocked()

    def flush_unlocked(self) -> None:
        """Flush daily totals and close files (caller must hold the lock)."""
        if self._daily_totals.requests > 0:
            self._flush_daily_unlocked(self._current_date)
        self._requests_file.close()
        self._daily_file.close()

    # -- internals -----------------------------------------------------------

    def _flush_daily_unlocked(self, day: date) -> None:
        day_str = str(day)
        # Read all existing rows, drop any for this date, then write the
        # consolidated row on top.
        existing_rows = []
        with open(self._daily_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header_row = next(reader, None)  # skip header
            for row in reader:
                if row and row[0] != day_str:
                    existing_rows.append(row)

        with open(self._daily_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if header_row:
                writer.writerow(header_row)
            for row in existing_rows:
                writer.writerow(row)
            writer.writerow(self._daily_totals._to_row(day))
            fh.flush()
            os.fsync(fh.fileno())

        self._daily_totals = _DailyTotals()

    def __del__(self) -> None:
        # Safety: try to close if normal shutdown was missed.
        try:
            self.flush()
        except Exception:
            pass


# Global logger instance (set at startup if cost_log_dir is configured).
_cost_logger: CostLogger | None = None


def _set_cost_logger(logger: CostLogger | None) -> None:
    global _cost_logger
    _cost_logger = logger


def _get_cost_logger() -> CostLogger | None:
    return _cost_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-scheme", choices=("http", "https"), default="http")
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--mode", choices=("stream", "optimistic"), required=True)
    parser.add_argument("--heartbeat-interval", type=float, default=15.0)
    parser.add_argument("--upstream-timeout", type=float, default=3600.0)
    parser.add_argument("--read-chunk-size", type=int, default=64 * 1024)
    parser.add_argument("--tls-cert-file", default="")
    parser.add_argument("--tls-key-file", default="")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--cost-input-price", type=float, default=0.0)
    parser.add_argument("--cost-cached-price", type=float, default=0.0)
    parser.add_argument("--cost-output-price", type=float, default=0.0)
    parser.add_argument("--cost-log-dir", default="", help="Directory for cost CSV logs (requests.csv + daily-summary.csv)")
    return parser.parse_args()


def read_api_keys(path: str) -> frozenset[str]:
    if not path:
        return frozenset()

    with open(path, "r", encoding="utf-8") as handle:
        keys = {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }
    return frozenset(keys)


def make_config(args: argparse.Namespace) -> ProxyConfig:
    if args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be greater than zero")
    if args.upstream_timeout <= 0:
        raise SystemExit("--upstream-timeout must be greater than zero")
    if args.read_chunk_size <= 0:
        raise SystemExit("--read-chunk-size must be greater than zero")

    if bool(args.tls_cert_file) != bool(args.tls_key_file):
        raise SystemExit("--tls-cert-file and --tls-key-file must be set together")

    return ProxyConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        target_scheme=args.target_scheme,
        target_host=args.target_host,
        target_port=args.target_port,
        mode=args.mode,
        heartbeat_interval=args.heartbeat_interval,
        upstream_timeout=args.upstream_timeout,
        read_chunk_size=args.read_chunk_size,
        frontend_tls_cert_file=args.tls_cert_file,
        frontend_tls_key_file=args.tls_key_file,
        api_keys=read_api_keys(args.api_key_file),
        cost_input_price=args.cost_input_price,
        cost_cached_price=args.cost_cached_price,
        cost_output_price=args.cost_output_price,
        cost_log_dir=args.cost_log_dir,
        stabilize=RequestStabilizeConfig.from_env(),
        stream_keepalive_mode=_stream_keepalive_mode_from_env(),
        stream_keepalive_text=_stream_keepalive_text_from_env(),
    )


class ProxyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], config: ProxyConfig):
        super().__init__(server_address, handler_class)
        self.config = config


class TimeoutProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "cf-timeout-proxy/0.1"

    def do_GET(self) -> None:
        self._handle_request()

    def do_HEAD(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_OPTIONS(self) -> None:
        allow_headers = self.headers.get("Access-Control-Request-Headers", "").strip() or DEFAULT_CORS_ALLOW_HEADERS
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", allow_headers)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[attr-defined]

    def _handle_request(self) -> None:
        self._write_lock = threading.Lock()
        # Serve costs dashboard before proxying.
        if self.command == "GET" and self.path.split("?", 1)[0] == "/dashboard":
            self._serve_dashboard()
            return
        # Serve cost calculator (static HTML, no auth).
        if self.command == "GET" and self.path.split("?", 1)[0] == "/cost-calculator":
            self._serve_cost_calculator()
            return
        body = self._read_body()

        # Optional: capture raw inference request bodies for offline replay.
        # Enable with LLAMA_PROXY_CAPTURE_ENABLED=on and LLAMA_PROXY_CAPTURE_DIR.
        # Used to record the exact
        # payloads sent by real clients (VS Code Copilot, Copilot CLI) so the
        # tool-call/reasoning parsing issue can be reproduced and validated
        # against authentic traffic. Detection-only; does not alter the request.
        if body:
            self._maybe_capture_request(body)

        # Optional: stabilize the OUTGOING inference request (sampling clamp +
        # max_tokens floor). Off unless configured via env. Runs AFTER capture
        # so the capture records the original client payload. This counters the
        # measured client sampling override (temp=1/top_p=1) for the Qwen3.8
        # tool-calling stack while keeping reasoning enabled.
        if body and self.config.stabilize.active and self.command == "POST":
            if self.path.split("?", 1)[0] in INFERENCE_PATHS:
                body = stabilize_inference_request_body(body, self.config.stabilize)

        if body is not None:
            request_meta = self._inspect_request(body)
        else:
            request_meta = {"mode": "strict", "wants_stream": False, "api_family": "strict", "requested_model": None}

        path = self.path.split("?", 1)[0]

        # Skip auth for non-inference paths (static assets, web UI, etc.)
        # These are proxied through to the backend without authentication.
        if path not in INFERENCE_PATHS and path not in MODEL_LIST_PATHS and path not in STREAM_LOOKUP_PATHS:
            self._proxy_strict(
                body,
                rewrite_openai_reasoning=False,
                rewrite_model_capabilities=False,
                requested_model=None,
                sse=False,
            )
            return

        if request_meta["mode"] != "strict" and not self._is_authorized():
            self._write_json_response(401, {"error": {"message": "invalid_api_key", "type": "auth_error"}})
            return

        if request_meta["mode"] == "strict":
            if self.command == "POST" and path in STREAM_LOOKUP_PATHS:
                self._proxy_stream_lookup(body)
                return
            self._proxy_strict(
                body,
                rewrite_openai_reasoning=request_meta["api_family"] == "openai",
                rewrite_model_capabilities=self.command == "GET" and path in MODEL_LIST_PATHS,
                requested_model=cast(str | None, request_meta["requested_model"]),
                sse=bool(request_meta["wants_stream"]),
            )
            return

        self._proxy_optimistic(
            body,
            wants_stream=bool(request_meta["wants_stream"]),
            api_family=str(request_meta["api_family"]),
            requested_model=cast(str | None, request_meta["requested_model"]),
        )

    def _serve_dashboard(self) -> None:
        """Serve the costs dashboard at GET /dashboard."""
        global _cost_dashboard

        cfg = self.config
        if not cfg.cost_log_dir:
            body = b'{"error": "cost logging not configured"}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True
            return

        with _dashboard_lock:
            # Import or reload the dashboard module so changes on disk take effect
            import importlib
            if _cost_dashboard is None:
                import cost_dashboard as _mod
                _cost_dashboard = _mod
            else:
                try:
                    # Replace the module object with the reloaded module so the
                    # latest edits on disk are used without restarting the process.
                    reloaded = importlib.reload(_cost_dashboard)
                    _cost_dashboard = reloaded
                except Exception:
                    # If reload fails, keep using the existing module
                    pass

        model_name_val = _read_model_alias_from_envfile() or os.environ.get("MODEL_ALIAS", "PAQ_LLAMACPP_SERVER")

        data = _cost_dashboard.collect_dashboard_data(
            requests_path=os.path.join(cfg.cost_log_dir, "requests.csv"),
            daily_path=os.path.join(cfg.cost_log_dir, "daily-summary.csv"),
            cost_input_price=cfg.cost_input_price,
            cost_cached_price=cfg.cost_cached_price,
            cost_output_price=cfg.cost_output_price,
            model_name=model_name_val,
        )

        # Ensure the returned data carries the authoritative model name the
        # proxy computed (avoid stale module or env-cache mismatches).
        try:
            if isinstance(data, dict):
                data["model_name"] = model_name_val
        except Exception:
            pass

        if data is None:
            body = b'{"error": "no cost data available"}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True
            return

        html_bytes = _cost_dashboard.render_dashboard_html(data)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(html_bytes)
        self.wfile.flush()
        self.close_connection = True

    def _serve_cost_calculator(self) -> None:
        """Serve the cost calculator at GET /cost-calculator."""
        html_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cost_calculator.html",
        )
        if not os.path.exists(html_path):
            body = b'{"error": "cost calculator not found"}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True
            return
        with open(html_path, "rb") as fh:
            html_bytes = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(html_bytes)
        self.wfile.flush()
        self.close_connection = True

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length", "")
        if not raw_length:
            return b""
        try:
            length = int(raw_length)
        except ValueError:
            return b""
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _maybe_capture_request(self, body: bytes) -> None:
        capture_enabled = _env_bool("LLAMA_PROXY_CAPTURE_ENABLED")
        if capture_enabled is False:
            return

        capture_dir = os.environ.get("LLAMA_PROXY_CAPTURE_DIR", "").strip()
        if not capture_dir:
            return
        if self.command != "POST":
            return
        path = self.path.split("?", 1)[0]
        if path not in INFERENCE_PATHS:
            return
        try:
            os.makedirs(capture_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            safe_path = path.strip("/").replace("/", "_") or "root"
            ua = self.headers.get("User-Agent", "")
            client = "client"
            ua_low = ua.lower()
            if "copilot" in ua_low or "github" in ua_low:
                client = "copilot"
            elif "code" in ua_low or "vscode" in ua_low:
                client = "vscode"
            base = os.path.join(capture_dir, f"{stamp}-{client}-{safe_path}")
            with open(base + ".body.json", "wb") as fh:
                fh.write(body)
            meta = {
                "timestamp": stamp,
                "path": path,
                "method": self.command,
                "headers": {k: v for k, v in self.headers.items()
                            if k.lower() not in {"authorization", "x-api-key", "cookie"}},
            }
            with open(base + ".meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        except Exception as exc:  # capture must never break proxying
            self.log_message("request capture failed: %s", exc)

    def _inspect_request(self, body: bytes) -> dict[str, object]:
        if self.command != "POST":
            return {"mode": "strict", "wants_stream": False, "api_family": "strict", "requested_model": None}

        path = self.path.split("?", 1)[0]
        api_family = get_api_family(path)
        if api_family == "strict":
            return {"mode": "strict", "wants_stream": False, "api_family": "strict", "requested_model": None}

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            return {"mode": "strict", "wants_stream": False, "api_family": api_family, "requested_model": None}

        try:
            payload = cast(dict[str, Any], json.loads(body.decode("utf-8")) if body else {})
        except Exception:
            return {"mode": "strict", "wants_stream": False, "api_family": api_family, "requested_model": None}

        wants_stream = bool(payload.get("stream"))
        requested_model = payload.get("model") if isinstance(payload.get("model"), str) else None
        if self.config.mode == "stream" and wants_stream:
            return {"mode": "stream", "wants_stream": True, "api_family": api_family, "requested_model": requested_model}
        if self.config.mode == "optimistic":
            return {"mode": "optimistic", "wants_stream": wants_stream, "api_family": api_family, "requested_model": requested_model}
        return {"mode": "strict", "wants_stream": wants_stream, "api_family": api_family, "requested_model": requested_model}

    def _is_authorized(self) -> bool:
        if not self.config.api_keys:
            return True

        token = ""
        auth_header = self.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        elif self.headers.get("X-API-Key"):
            token = self.headers.get("X-API-Key", "").strip()

        return bool(token) and token in self.config.api_keys

    def _make_upstream_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self.config.target_scheme == "https":
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return http.client.HTTPSConnection(
                self.config.target_host,
                self.config.target_port,
                timeout=self.config.upstream_timeout,
                context=context,
            )
        return http.client.HTTPConnection(
            self.config.target_host,
            self.config.target_port,
            timeout=self.config.upstream_timeout,
        )

    def _forward_headers(self, body: bytes) -> dict[str, str]:
        forwarded: dict[str, str] = {}
        client_accept_encoding = ""
        for name, value in self.headers.items():
            lower = name.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in {"host", "content-length"}:
                continue
            if lower == "accept-encoding":
                client_accept_encoding = value
                continue
            forwarded[name] = value

        host_header = self.config.target_host
        default_port = 443 if self.config.target_scheme == "https" else 80
        if self.config.target_port != default_port:
            host_header = f"{host_header}:{self.config.target_port}"

        forwarded["Host"] = host_header
        forwarded["Accept-Encoding"] = client_accept_encoding if client_accept_encoding else "gzip"
        forwarded["Connection"] = "close"
        forwarded["X-Forwarded-For"] = self.client_address[0]
        forwarded["X-Forwarded-Proto"] = "https" if self.config.frontend_tls_cert_file else "http"
        if self.headers.get("Host"):
            forwarded["X-Forwarded-Host"] = self.headers["Host"]
        if body:
            forwarded["Content-Length"] = str(len(body))
        return forwarded

    def _request_upstream(self, body: bytes) -> UpstreamResult:
        result = UpstreamResult()
        try:
            connection = self._make_upstream_connection()
            connection.request(
                self.command,
                self.path,
                body=body if self.command in {"POST", "PUT", "PATCH"} else None,
                headers=self._forward_headers(body),
            )
            result.connection = connection
            result.response = connection.getresponse()
            return result
        except Exception as exc:
            if result.connection is not None:
                try:
                    result.connection.close()
                except Exception:
                    pass
            result.error = exc
            return result

    def _proxy_strict(
        self,
        body: bytes,
        *,
        rewrite_openai_reasoning: bool = False,
        rewrite_model_capabilities: bool = False,
        requested_model: str | None = None,
        sse: bool = False,
    ) -> None:
        upstream = self._request_upstream(body)
        if upstream.error is not None or upstream.response is None:
            self._write_json_response(
                502,
                {"error": {"message": f"upstream connection failed: {upstream.error}", "type": "proxy_error"}},
            )
            return

        response = upstream.response
        self.send_response(response.status, response.reason)
        self._relay_response_headers(response.getheaders(), chunked=self.command != "HEAD")
        self.end_headers()

        try:
            if self.command != "HEAD":
                try:
                    self._relay_response_body(
                        response,
                        chunked=True,
                        rewrite_openai_reasoning=rewrite_openai_reasoning,
                        rewrite_model_capabilities=rewrite_model_capabilities,
                        requested_model=requested_model,
                        sse=sse,
                        cost_config=self.config,
                    )
                    self._finish_chunked()
                except DownstreamWriteError as exc:
                    self.log_message("downstream closed during strict proxy for %s: %s", self.path, exc)
        finally:
            if upstream.connection is not None:
                upstream.connection.close()
            self.close_connection = True

    def _proxy_stream_lookup(self, body: bytes) -> None:
        """Handle POST /v1/streams/lookup (llama.cpp resumable-stream session lookup).

        GitHub's BYOK configuration UI probes this endpoint when connecting an
        OpenAI-compatible model, and builds of llama-server without the
        resumable-stream feature (e.g. the PrismML fork) answer 404. A strict
        client can read that as "streaming unsupported", so instead of
        relaying the 404 we answer exactly like a llama.cpp server with no
        live sessions: ``200`` with an empty JSON array. If the backend does
        implement the endpoint (mainline builds), its answer is relayed
        unchanged so genuine session lookups keep working.
        """
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as exc:
                self._write_json_response(
                    400,
                    {"error": {"message": f"invalid body: {exc}", "type": "invalid_request_error"}},
                )
                return
            if not isinstance(payload, dict):
                self._write_json_response(
                    400,
                    {"error": {"message": "invalid body: expected a JSON object", "type": "invalid_request_error"}},
                )
                return
            conversation_ids = payload.get("conversation_ids")
            if conversation_ids is not None and not isinstance(conversation_ids, list):
                self._write_json_response(
                    400,
                    {"error": {"message": "invalid body: conversation_ids must be an array", "type": "invalid_request_error"}},
                )
                return

        upstream = self._request_upstream(body)
        if upstream.error is None and upstream.response is not None and upstream.response.status != 404:
            response = upstream.response
            self.send_response(response.status, response.reason)
            self._relay_response_headers(response.getheaders(), chunked=self.command != "HEAD")
            self.end_headers()
            try:
                if self.command != "HEAD":
                    try:
                        self._relay_response_body(response, chunked=True)
                        self._finish_chunked()
                    except DownstreamWriteError as exc:
                        self.log_message("downstream closed during streams/lookup relay for %s: %s", self.path, exc)
            finally:
                if upstream.connection is not None:
                    upstream.connection.close()
                self.close_connection = True
            return

        if upstream.connection is not None:
            try:
                upstream.connection.close()
            except Exception:
                pass

        # Backend lacks the resumable-stream feature (404): report no sessions.
        raw = b"[]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)
            self.wfile.flush()
        self.close_connection = True

    def _proxy_optimistic(self, body: bytes, wants_stream: bool, api_family: str, requested_model: str | None = None) -> None:
        ready = threading.Event()
        upstream = UpstreamResult()

        def worker() -> None:
            result = self._request_upstream(body)
            upstream.connection = result.connection
            upstream.response = result.response
            upstream.error = result.error
            ready.set()

        threading.Thread(target=worker, daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8" if wants_stream else "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            self._write_chunk(self._stream_keepalive_payload(requested_model) if wants_stream else b" \n")

            while not ready.wait(timeout=self.config.heartbeat_interval):
                if wants_stream:
                    self._write_chunk(self._stream_keepalive_payload(requested_model))
                else:
                    self._write_chunk(b" \n")

            if upstream.error is not None or upstream.response is None:
                self._write_optimistic_error(upstream.error, wants_stream=wants_stream)
                self._finish_chunked()
                return

            response = upstream.response
            if response.status >= 400:
                error_payload = response.read()
                self._write_optimistic_upstream_error(
                    error_payload,
                    response.status,
                    wants_stream=wants_stream,
                )
                self._finish_chunked()
                return

            self._relay_response_body(
                response,
                chunked=True,
                rewrite_openai_reasoning=api_family == "openai",
                requested_model=requested_model,
                sse=wants_stream,
                cost_config=self.config,
            )
            self._finish_chunked()
        except DownstreamWriteError as exc:
            self.log_message("downstream closed during optimistic proxy for %s: %s", self.path, exc)
        finally:
            if upstream.connection is not None:
                upstream.connection.close()
            self.close_connection = True

    def _stream_keepalive_payload(self, model: str | None) -> bytes:
        """Bytes for one streaming keep-alive, per configured keep-alive mode.

        ``reasoning`` mode emits a real ``reasoning_content`` delta (strongest
        reset for a client's time-to-first-token timer; shown as thinking, never
        pollutes the answer); ``data`` mode emits an empty-delta OpenAI chunk;
        ``comment`` mode emits the legacy ``: keep-alive`` SSE comment.
        """
        if self.config.stream_keepalive_mode == "reasoning":
            return _build_stream_keepalive_reasoning_chunk(model, self.config.stream_keepalive_text)
        if self.config.stream_keepalive_mode == "data":
            return _build_stream_keepalive_chunk(model)
        return b": keep-alive\n\n"

    def _relay_response_headers(self, headers: Iterable[tuple[str, str]], chunked: bool) -> None:
        for name, value in headers:
            lower = name.lower()
            if lower in HOP_BY_HOP_HEADERS or lower == "content-length":
                continue
            self.send_header(name, value)
        self.send_header("Connection", "close")
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")

    def _relay_response_body(
        self,
        response: http.client.HTTPResponse,
        chunked: bool,
        *,
        rewrite_openai_reasoning: bool = False,
        rewrite_model_capabilities: bool = False,
        requested_model: str | None = None,
        sse: bool = False,
        cost_config: ProxyConfig | None = None,
    ) -> None:
        usage_config = cost_config or self.config
        content_type = response.getheader("Content-Type", "") or ""
        is_json = "json" in content_type.lower()

        if (rewrite_openai_reasoning or rewrite_model_capabilities or requested_model or is_json) and not sse:
            raw = response.read()
            if raw:
                if rewrite_openai_reasoning:
                    raw = rewrite_openai_reasoning_payload(raw)
                if requested_model:
                    raw = rewrite_requested_model_payload(raw, requested_model)
                if rewrite_model_capabilities:
                    raw = rewrite_model_capabilities_payload(raw)
                if is_json:
                    # Non-streaming chat responses should be normalized the same way
                    # as SSE final chunks.  If the backend only returns timings on a
                    # final response, synthesize the usage object here so clients see
                    # a stable OpenAI-style payload.
                    raw = rewrite_usage_cost_payload(raw, usage_config, synthesize_from_timings=True)
                if chunked:
                    self._write_chunk(raw)
                else:
                    self.wfile.write(raw)
                    self.wfile.flush()
            return

        if sse:
            pending = b""
            while True:
                # Use read1() (single underlying read, returns whatever is
                # available) instead of read(), which blocks until it has
                # *read_chunk_size* bytes OR the upstream hits EOF. With SSE the
                # final bytes (the finish_reason chunk + "data: [DONE]") rarely
                # fill the 64 KiB buffer, so read() would hold them until the
                # upstream closes; if llama.cpp delays its terminating chunk/EOF
                # (keep-alive race, tunnel backpressure) those bytes — including
                # [DONE] — are pinned for up to --upstream-timeout (3600s),
                # stalling streaming clients like Copilot mid-turn. read1()
                # forwards each piece immediately and breaks promptly at EOF.
                chunk = response.read1(self.config.read_chunk_size)
                if not chunk:
                    break
                pending += chunk
                while True:
                    newline_index = pending.find(b"\n")
                    if newline_index < 0:
                        break
                    line = pending[:newline_index + 1]
                    pending = pending[newline_index + 1:]
                    line = rewrite_json_sse_line(
                        line,
                        rewrite_openai_reasoning=rewrite_openai_reasoning,
                        requested_model=requested_model,
                        cost_config=usage_config,
                    )
                    if chunked:
                        self._write_chunk(line)
                    else:
                        self.wfile.write(line)
                        self.wfile.flush()
            if pending:
                pending = rewrite_json_sse_line(
                    pending,
                    rewrite_openai_reasoning=rewrite_openai_reasoning,
                    requested_model=requested_model,
                    cost_config=usage_config,
                )
                if chunked:
                    self._write_chunk(pending)
                else:
                    self.wfile.write(pending)
                    self.wfile.flush()
            return

        while True:
            chunk = response.read(self.config.read_chunk_size)
            if not chunk:
                break
            if chunked:
                self._write_chunk(chunk)
            else:
                self.wfile.write(chunk)
                self.wfile.flush()

    def _write_json_response(self, status: int, payload: dict[str, object]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)
            self.wfile.flush()
        self.close_connection = True

    def _write_optimistic_error(self, error: Exception | None, wants_stream: bool) -> None:
        error_message = f"upstream connection failed: {error}"
        message = json.dumps(
            {
                "error": {
                    "message": error_message,
                    "type": "proxy_error",
                }
            }
        ).encode("utf-8")
        if wants_stream:
            self._write_chunk(b"data: " + message + b"\n\n")
            self._write_chunk(b"data: [DONE]\n\n")
            return
        self._write_chunk(message)

    def _write_optimistic_upstream_error(self, raw: bytes, upstream_status: int, wants_stream: bool) -> None:
        payload = raw.strip() or json.dumps(
            {
                "error": {
                    "message": f"upstream returned HTTP {upstream_status}",
                    "type": "upstream_error",
                    "upstream_status": upstream_status,
                }
            }
        ).encode("utf-8")

        if wants_stream:
            self._write_chunk(b"data: " + payload + b"\n\n")
            self._write_chunk(b"data: [DONE]\n\n")
            return

        self._write_chunk(payload)

    def _write_chunk(self, data: bytes) -> None:
        if self.command == "HEAD" or not data:
            return
        lock = getattr(self, "_write_lock", None)
        if lock is None:
            try:
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError) as exc:
                raise DownstreamWriteError(str(exc)) from exc
            return

        with lock:
            try:
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError) as exc:
                raise DownstreamWriteError(str(exc)) from exc

    def _finish_chunked(self) -> None:
        if self.command == "HEAD":
            return
        lock = getattr(self, "_write_lock", None)
        if lock is None:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError) as exc:
                raise DownstreamWriteError(str(exc)) from exc
            return

        with lock:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError) as exc:
                raise DownstreamWriteError(str(exc)) from exc


def serve(config: ProxyConfig) -> int:
    # Initialize cost logger if configured.
    cost_logger: CostLogger | None = None
    if config.cost_log_dir:
        cost_logger = CostLogger(config.cost_log_dir)
        _set_cost_logger(cost_logger)
        print(
            f"Cost logging enabled: {config.cost_log_dir}/",
            flush=True,
        )

    server = ProxyHTTPServer((config.listen_host, config.listen_port), TimeoutProxyHandler, config)

    if config.frontend_tls_cert_file:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(config.frontend_tls_cert_file, config.frontend_tls_key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)

    stop_event = threading.Event()

    def shutdown_handler(signum: int, _frame: object) -> None:
        if stop_event.is_set():
            return
        stop_event.set()
        print(f"Received signal {signum}; shutting down Cloudflare timeout proxy", file=sys.stderr, flush=True)
        if cost_logger is not None:
            cost_logger.flush()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    scheme = "https" if config.frontend_tls_cert_file else "http"
    print(
        f"Cloudflare timeout proxy listening on {scheme}://{config.listen_host}:{config.listen_port} "
        f"-> {config.target_scheme}://{config.target_host}:{config.target_port} mode={config.mode}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        if cost_logger is not None:
            cost_logger.flush()
        server.server_close()
    return 0


def main() -> int:
    args = parse_args()
    config = make_config(args)
    return serve(config)


if __name__ == "__main__":
    raise SystemExit(main())
