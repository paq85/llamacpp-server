#!/usr/bin/env python3
"""Tool-call / reasoning stress + reproduction harness for the Qwen3.8 stack.

Goal: reproduce the intermittent tool-call / reasoning parsing failures that
make coding agents (e.g. VS Code Copilot) stop mid-task.

This is a **detection-only** harness — it does NOT change the server or apply
any fix. It drives many streamed, multi-turn tool-calling conversations with
reasoning enabled, reconstructs the streamed tool calls, and flags any of:

  * HTTP status != 200
  * an SSE ``event: error`` / a data payload carrying a top-level ``error``
  * the substring "Failed to parse" anywhere in the stream (the llama.cpp
    final-parse throw surfaces here)
  * a stream that ends without a final ``finish_reason``
  * a streamed ``tool_call`` whose accumulated ``arguments`` is not valid JSON

On any failure, the raw request and the raw SSE bytes are written to the output
directory so the exact offending exchange can be inspected and correlated with
the server log (grep for "Failed to parse input at pos" and "Chat format:").

Usage:
    python3 scripts/toolcall-stress.py                 # auto-detect endpoint
    python3 scripts/toolcall-stress.py --url https://127.0.0.1:8082
    python3 scripts/toolcall-stress.py --iterations 40 --max-turns 8
    python3 scripts/toolcall-stress.py --temperature 0.8   # provoke more drift
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
API_KEY_FILE = ROOT / "api-keys.txt"
DEFAULT_OUT_DIR = ROOT / "benchmarks" / "toolcall-stress"

URL_CANDIDATES = [
    "https://127.0.0.1:8082",  # direct backend first (isolates the proxy)
    "http://127.0.0.1:8082",
    "https://127.0.0.1:8080",  # proxy
    "http://127.0.0.1:8080",
]

MODEL_NAME = "PAQ_LLAMACPP_SERVER"


# ---------------------------------------------------------------------------
# HTTP / SSE plumbing (mirrors test-context-usage.py so behavior matches)
# ---------------------------------------------------------------------------

def read_api_key() -> Optional[str]:
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except FileNotFoundError:
        return None
    return None


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _connect(base_url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    if base_url.startswith("https://"):
        rest = base_url[len("https://"):]
        host_port = rest.split("/", 1)[0]
        host = host_port.split(":")[0]
        port = int(host_port.split(":")[1]) if ":" in host_port else 443
        return http.client.HTTPSConnection(host, port, timeout=timeout, context=_ssl_ctx()), host
    rest = base_url[len("http://"):]
    host_port = rest.split("/", 1)[0]
    host = host_port.split(":")[0]
    port = int(host_port.split(":")[1]) if ":" in host_port else 80
    return http.client.HTTPConnection(host, port, timeout=timeout), host


def detect_url(explicit: Optional[str]) -> str:
    candidates = [explicit] if explicit else URL_CANDIDATES
    api_key = read_api_key()
    for base_url in candidates:
        if not base_url:
            continue
        try:
            conn, _ = _connect(base_url, timeout=5)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            conn.request("GET", "/v1/models", headers=headers)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                print(f"[ok] endpoint detected: {base_url}")
                return base_url
        except Exception:
            continue
    raise SystemExit("Could not reach a live endpoint. Is the server running?")


# ---------------------------------------------------------------------------
# Coding-agent tool catalog (shaped like VS Code Copilot agent mode + GitHub
# Copilot CLI, which ships the GitHub MCP server and supports custom MCP
# servers). Real clients send a LARGE catalog (dozens of tools, tens of KB of
# JSON). That size + breadth is part of what stresses the model into producing
# tool-call/reasoning output that trips the llama.cpp PEG parser, so we emulate
# it faithfully here instead of using a tiny tool set.
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, properties: dict[str, Any],
        required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


# Core VS Code Copilot agent-mode tools (names mirror the real surface).
_CORE_TOOLS: list[dict[str, Any]] = [
    _fn("read_file", "Read the contents of a file. Provide a line range to read a section.",
        {"filePath": {"type": "string", "description": "Absolute path to the file."},
         "startLine": {"type": "integer"}, "endLine": {"type": "integer"}},
        ["filePath", "startLine", "endLine"]),
    _fn("list_dir", "List the contents of a directory.",
        {"path": {"type": "string"}}, ["path"]),
    _fn("file_search", "Search for files by glob pattern.",
        {"query": {"type": "string"}, "maxResults": {"type": "integer"}}, ["query"]),
    _fn("grep_search", "Do a fast text/regex search in the workspace.",
        {"query": {"type": "string"}, "isRegexp": {"type": "boolean"},
         "includePattern": {"type": "string"}, "maxResults": {"type": "integer"}},
        ["query", "isRegexp"]),
    _fn("semantic_search", "Natural-language search for relevant code in the workspace.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("create_file", "Create a new file with the given content.",
        {"filePath": {"type": "string"}, "content": {"type": "string", "description": "Full file content."}},
        ["filePath", "content"]),
    _fn("replace_string_in_file", "Replace an exact literal string in a file with a new string. "
        "Include 3-5 lines of context before and after to uniquely identify the location.",
        {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}},
        ["filePath", "oldString", "newString"]),
    _fn("insert_edit_into_file", "Insert/replace a region of a file with new code.",
        {"filePath": {"type": "string"}, "code": {"type": "string"}, "explanation": {"type": "string"}},
        ["filePath", "code"]),
    _fn("run_in_terminal", "Run a shell command in the workspace terminal.",
        {"command": {"type": "string"}, "explanation": {"type": "string"},
         "isBackground": {"type": "boolean"}}, ["command"]),
    _fn("get_terminal_output", "Get output from a terminal execution by id.",
        {"id": {"type": "string"}}, ["id"]),
    _fn("get_errors", "Get compile/lint errors for files.",
        {"filePaths": {"type": "array", "items": {"type": "string"}}}, []),
    _fn("run_tests", "Run tests in the workspace.",
        {"files": {"type": "array", "items": {"type": "string"}},
         "testNames": {"type": "array", "items": {"type": "string"}}}, []),
    _fn("test_search", "Find test files for a given source file.",
        {"filePaths": {"type": "array", "items": {"type": "string"}}}, ["filePaths"]),
    _fn("list_code_usages", "Find references/definitions/implementations of a symbol.",
        {"symbolName": {"type": "string"}, "filePaths": {"type": "array", "items": {"type": "string"}}},
        ["symbolName"]),
    _fn("fetch_webpage", "Fetch and summarize a web page.",
        {"urls": {"type": "array", "items": {"type": "string"}}, "query": {"type": "string"}},
        ["urls", "query"]),
]

# GitHub MCP server tools (shipped by Copilot CLI by default).
_GITHUB_MCP_TOOLS: list[dict[str, Any]] = [
    _fn("github_create_issue", "Create a new GitHub issue.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "title": {"type": "string"},
         "body": {"type": "string"}, "labels": {"type": "array", "items": {"type": "string"}},
         "assignees": {"type": "array", "items": {"type": "string"}}},
        ["owner", "repo", "title"]),
    _fn("github_list_issues", "List issues in a repository.",
        {"owner": {"type": "string"}, "repo": {"type": "string"},
         "state": {"type": "string", "enum": ["open", "closed", "all"]},
         "labels": {"type": "array", "items": {"type": "string"}}}, ["owner", "repo"]),
    _fn("github_get_pull_request", "Get a pull request by number.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "pullNumber": {"type": "integer"}},
        ["owner", "repo", "pullNumber"]),
    _fn("github_create_pull_request", "Open a pull request.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "title": {"type": "string"},
         "head": {"type": "string"}, "base": {"type": "string"}, "body": {"type": "string"},
         "draft": {"type": "boolean"}}, ["owner", "repo", "title", "head", "base"]),
    _fn("github_merge_pull_request", "Merge a pull request.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "pullNumber": {"type": "integer"},
         "mergeMethod": {"type": "string", "enum": ["merge", "squash", "rebase"]}},
        ["owner", "repo", "pullNumber"]),
    _fn("github_search_code", "Search code across GitHub.",
        {"query": {"type": "string"}, "perPage": {"type": "integer"}}, ["query"]),
    _fn("github_get_file_contents", "Get file contents from a GitHub repo.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "path": {"type": "string"},
         "ref": {"type": "string"}}, ["owner", "repo", "path"]),
    _fn("github_create_or_update_file", "Create or update a file in a GitHub repo.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "path": {"type": "string"},
         "content": {"type": "string"}, "message": {"type": "string"}, "branch": {"type": "string"}},
        ["owner", "repo", "path", "content", "message", "branch"]),
    _fn("github_list_commits", "List commits on a branch.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "sha": {"type": "string"}},
        ["owner", "repo"]),
    _fn("github_get_issue", "Get a single issue by number.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "issueNumber": {"type": "integer"}},
        ["owner", "repo", "issueNumber"]),
    _fn("github_add_issue_comment", "Comment on an issue or PR.",
        {"owner": {"type": "string"}, "repo": {"type": "string"}, "issueNumber": {"type": "integer"},
         "body": {"type": "string"}}, ["owner", "repo", "issueNumber", "body"]),
]


def _padding_tools(n: int) -> list[dict[str, Any]]:
    """Generate extra plausible MCP-style tools to reach a realistic catalog size."""
    verbs = ["analyze", "format", "lint", "build", "deploy", "migrate", "render",
             "validate", "summarize", "transform", "index", "diff", "patch", "scan"]
    nouns = ["config", "schema", "dependency", "container", "database", "endpoint",
             "manifest", "snapshot", "artifact", "workspace", "document", "module"]
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n:
        verb = verbs[i % len(verbs)]
        noun = nouns[(i // len(verbs)) % len(nouns)]
        out.append(_fn(
            f"{verb}_{noun}",
            f"{verb.capitalize()} the given {noun} and return a structured result.",
            {"target": {"type": "string"},
             "options": {"type": "object",
                         "properties": {"recursive": {"type": "boolean"},
                                        "format": {"type": "string", "enum": ["json", "yaml", "text"]}}},
             "tags": {"type": "array", "items": {"type": "string"}}},
            ["target"]))
        i += 1
    return out


def build_tools(catalog_size: int = 60) -> list[dict[str, Any]]:
    tools = list(_CORE_TOOLS) + list(_GITHUB_MCP_TOOLS)
    if catalog_size > len(tools):
        tools += _padding_tools(catalog_size - len(tools))
    return tools[:catalog_size] if catalog_size > 0 else tools


# Tasks chosen to force MULTI-STEP tool use AND code-heavy arguments
# (lots of quotes / braces / newlines), which is exactly what stresses the
# tool-call JSON parsing while reasoning is interleaved.
TASK_PROMPTS = [
    "Create a Python module utils/strings.py with a function slugify(text) that "
    "lowercases, strips, and replaces non-alphanumerics with hyphens. Then read it "
    "back to verify, and run the tests with pytest. Use the tools step by step.",

    "Refactor the function process(data) in src/pipeline.py: first read the file, "
    "then replace the body to add input validation and error handling, then run the "
    "linter. Do it one tool call at a time.",

    "Add a TypeScript Express route GET /api/health that returns {status:'ok'} in "
    "src/routes/health.ts. Create the file, then grep for where routes are registered, "
    "then run the build.",

    "Investigate why the test suite is failing: list the tests directory, read the "
    "failing test file tests/test_auth.py, search for the symbol authenticate, then "
    "propose and apply a fix with replace_string_in_file.",

    "Write a bash script scripts/backup.sh that tars /var/data into a timestamped "
    "archive and uploads it. Create the file with full content, make it executable via "
    "run_in_terminal, and read it back to confirm.",

    "Create a JSON config config/app.json with nested keys (server.host, server.port, "
    "features.flags as an array of strings with special characters like \"a/b\", "
    "\"c\\\"d\"), then read it back and validate it parses.",

    # Parallel-call + large-argument tasks (closest to real Copilot agent bursts).
    "In a single step, read THREE files in parallel: src/a.py, src/b.py and src/c.py "
    "(issue all three read_file calls at once). Then summarize them.",

    "Create a complete Python CLI app in one file cli/main.py: include argparse with "
    "5 subcommands, type hints, docstrings, a class with 4 methods, error handling, and "
    "a main guard. Put the ENTIRE source (60+ lines, with quotes, regex strings like "
    "r\"\\d+\\.\\d+\", f-strings, and a dict literal) into the create_file content argument. "
    "Then read it back.",

    "Generate a React component src/Form.tsx with TypeScript: include useState hooks, an "
    "onSubmit handler, JSX with nested elements and className strings, an interface with 6 "
    "fields, and inline styles using object literals. Put the full component (with backticks, "
    "quotes and braces) into a single create_file call, then run the build and the tests in "
    "parallel.",

    "Apply three edits at once with three parallel replace_string_in_file calls to src/app.py: "
    "rename a function, add an import, and change a constant. Then run the linter and tests in "
    "parallel.",
]


# ---------------------------------------------------------------------------
# Streaming tool-call reconstruction
# ---------------------------------------------------------------------------

@dataclass
class StreamedToolCall:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class TurnResult:
    ok: bool
    failure_kind: str = ""
    detail: str = ""
    finish_reason: Optional[str] = None
    reasoning_chars: int = 0
    content: str = ""
    tool_calls: list[StreamedToolCall] = field(default_factory=list)
    raw_sse: str = ""
    http_status: int = 0


def _accumulate_tool_call_delta(slots: dict[int, StreamedToolCall], tc: dict[str, Any]) -> None:
    idx = tc.get("index", 0)
    if not isinstance(idx, int):
        idx = 0
    slot = slots.setdefault(idx, StreamedToolCall(index=idx))
    if tc.get("id"):
        slot.id = tc["id"]
    fn = tc.get("function")
    if isinstance(fn, dict):
        if fn.get("name"):
            slot.name = fn["name"]
        args = fn.get("arguments")
        if isinstance(args, str):
            slot.arguments += args


def run_turn(base_url: str, api_key: Optional[str], messages: list[dict[str, Any]],
             tools: list[dict[str, Any]], temperature: float, max_tokens: int,
             timeout: float, raw_body_override: dict[str, Any] | None = None) -> TurnResult:
    if raw_body_override is not None:
        body = dict(raw_body_override)
        body["stream"] = True
        body.setdefault("model", MODEL_NAME)
    else:
        body = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens:  # 0 / None => no output cap (server decides, up to 30K+)
            body["max_tokens"] = max_tokens
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    conn, _ = _connect(base_url, timeout=timeout)
    path = base_url.split("://", 1)[1]
    path = "/" + path.split("/", 1)[1] if "/" in path else "/"
    path = "/v1/chat/completions"

    try:
        conn.request("POST", path, body=data, headers=headers)
        resp = conn.getresponse()
    except Exception as exc:
        conn.close()
        return TurnResult(ok=False, failure_kind="http_exception", detail=str(exc))

    status = resp.status
    raw_lines: list[str] = []
    tool_slots: dict[int, StreamedToolCall] = {}
    content_parts: list[str] = []
    reasoning_chars = 0
    finish_reason: Optional[str] = None
    saw_error = False
    error_detail = ""

    try:
        while True:
            chunk = resp.readline()
            if not chunk:
                break
            line = chunk.decode("utf-8", errors="replace")
            raw_lines.append(line.rstrip("\n"))
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue

            if stripped.startswith("event:"):
                ev = stripped[len("event:"):].strip()
                if ev == "error":
                    saw_error = True
                    error_detail = error_detail or "sse event: error"
                continue

            if stripped.startswith(":"):
                continue  # heartbeat comment

            if not stripped.startswith("data:"):
                continue

            payload = stripped[len("data:"):]
            while payload.startswith(" "):
                payload = payload[1:]
            if payload.strip() == "[DONE]":
                continue

            if "Failed to parse" in payload:
                saw_error = True
                error_detail = payload[:600]

            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if isinstance(obj, dict) and isinstance(obj.get("error"), (dict, str)):
                saw_error = True
                error_detail = json.dumps(obj["error"])[:600]
                continue

            choices = obj.get("choices") if isinstance(obj, dict) else None
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    c = delta.get("content")
                    if isinstance(c, str):
                        content_parts.append(c)
                    for rkey in ("reasoning_content", "reasoning_text", "thinking"):
                        rc = delta.get(rkey)
                        if isinstance(rc, str):
                            reasoning_chars += len(rc)
                            break
                    tcs = delta.get("tool_calls")
                    if isinstance(tcs, list):
                        for tc in tcs:
                            if isinstance(tc, dict):
                                _accumulate_tool_call_delta(tool_slots, tc)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    except Exception as exc:
        error_detail = error_detail or f"stream read error: {exc}"
        saw_error = True
    finally:
        conn.close()

    raw_sse = "\n".join(raw_lines)
    tool_calls = [tool_slots[k] for k in sorted(tool_slots)]

    # Classify the outcome.
    if status != 200:
        return TurnResult(ok=False, failure_kind="http_status", detail=f"HTTP {status}: {raw_sse[:400]}",
                          finish_reason=finish_reason, reasoning_chars=reasoning_chars,
                          content="".join(content_parts), tool_calls=tool_calls, raw_sse=raw_sse,
                          http_status=status)
    if saw_error:
        kind = "parse_error" if "Failed to parse" in error_detail else "sse_error"
        return TurnResult(ok=False, failure_kind=kind, detail=error_detail,
                          finish_reason=finish_reason, reasoning_chars=reasoning_chars,
                          content="".join(content_parts), tool_calls=tool_calls, raw_sse=raw_sse,
                          http_status=status)
    if finish_reason is None:
        return TurnResult(ok=False, failure_kind="no_finish_reason",
                          detail="stream ended without a finish_reason",
                          reasoning_chars=reasoning_chars, content="".join(content_parts),
                          tool_calls=tool_calls, raw_sse=raw_sse, http_status=status)

    # Validate tool-call argument JSON.
    for tc in tool_calls:
        if tc.arguments.strip() == "":
            continue
        try:
            json.loads(tc.arguments)
        except json.JSONDecodeError as exc:
            return TurnResult(ok=False, failure_kind="bad_tool_args",
                              detail=f"tool '{tc.name}' args not valid JSON: {exc}; raw={tc.arguments[:400]!r}",
                              finish_reason=finish_reason, reasoning_chars=reasoning_chars,
                              content="".join(content_parts), tool_calls=tool_calls, raw_sse=raw_sse,
                              http_status=status)

    return TurnResult(ok=True, finish_reason=finish_reason, reasoning_chars=reasoning_chars,
                      content="".join(content_parts), tool_calls=tool_calls, raw_sse=raw_sse,
                      http_status=status)


def synth_tool_result(name: str, arguments: str) -> str:
    """Return a plausible fake tool result to keep the conversation going."""
    if name == "read_file":
        return "def process(data):\n    return [x * 2 for x in data]\n"
    if name == "list_dir":
        return "src/\ntests/\nREADME.md\npackage.json\n"
    if name == "grep_search":
        return "src/app.py:42: authenticate(user, token)\n"
    if name == "create_file":
        return "File created successfully."
    if name == "replace_string_in_file":
        return "Edit applied. 1 occurrence replaced."
    if name == "run_in_terminal":
        return "exit code 0\nAll checks passed.\n"
    return "ok"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    iterations: int = 0
    turns: int = 0
    failures: int = 0
    turns_with_reasoning: int = 0
    turns_with_tool_calls: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)


# A large, Copilot-style system prompt. VS Code Copilot agent mode and Copilot
# CLI both send long system prompts with detailed tool-use rules. The size and
# the explicit "think then call tools" instructions are part of what shapes the
# model's interleaved reasoning + tool-call output.
COPILOT_SYSTEM_PROMPT = (
    "You are GitHub Copilot, an AI programming assistant working as an autonomous "
    "coding agent inside the user's editor/terminal. You have access to a set of "
    "tools to inspect and modify the user's workspace and to interact with GitHub.\n\n"
    "<instructions>\n"
    "Keep going until the user's task is completely resolved before yielding. Use the "
    "tools to gather context and make changes rather than guessing. Prefer reading "
    "files before editing them. When you make code edits, ensure the result is correct "
    "and idiomatic.\n"
    "Think step by step. Before each tool call, briefly reason about what you are doing "
    "and why. You may call multiple independent tools in parallel when there are no "
    "dependencies between them; otherwise call them one at a time and wait for results.\n"
    "When the task is fully complete, stop calling tools and give a short final summary.\n"
    "</instructions>\n\n"
    "<toolUseInstructions>\n"
    "Always provide fully-specified arguments to tools. For file edits, include enough "
    "surrounding context to locate the edit uniquely. Never fabricate file contents you "
    "have not read. Do not call a tool you do not need.\n"
    "</toolUseInstructions>\n"
)


def build_system_prompt(client: str) -> str:
    if client == "cli":
        return (
            COPILOT_SYSTEM_PROMPT
            + "\nYou are running in GitHub Copilot CLI, a terminal-native agent. The GitHub "
            "MCP server is available by default, plus any custom MCP servers the user has "
            "configured. Preview actions before executing destructive commands.\n"
        )
    return (
        COPILOT_SYSTEM_PROMPT
        + "\nYou are running in VS Code agent mode. Respect the user's workspace settings "
        "and use the editor tools to apply edits.\n"
    )


# ---------------------------------------------------------------------------
# Replay mode (gold standard: replay REAL captured client request bodies)
# ---------------------------------------------------------------------------

def _iter_capture_files(replay_path: Path) -> list[Path]:
    if replay_path.is_dir():
        return sorted(replay_path.glob("*.body.json")) or sorted(
            p for p in replay_path.glob("*.json") if not p.name.endswith(".meta.json"))
    return [replay_path]


def run_replay(base_url: str, api_key: Optional[str], replay_path: Path,
               out_dir: Path, run_id: str, timeout: float, stop_on_first: bool) -> int:
    files = _iter_capture_files(replay_path)
    if not files:
        raise SystemExit(f"No capture files found at {replay_path}")
    print(f"[replay] {len(files)} captured request(s) from {replay_path}")
    stats = Stats()
    for idx, f in enumerate(files):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [skip] {f.name}: cannot parse ({exc})")
            continue
        if not isinstance(payload, dict) or "messages" not in payload:
            print(f"  [skip] {f.name}: not a chat payload")
            continue
        payload["stream"] = True  # force streaming so we see the SSE failure path
        messages = payload.get("messages", [])
        tools = payload.get("tools", []) or []
        temperature = float(payload.get("temperature", 0.7) or 0.7)
        max_tokens = int(payload.get("max_tokens", 30000) or 30000)
        stats.iterations += 1
        res = run_turn(base_url, api_key, messages, tools, temperature=temperature,
                       max_tokens=max_tokens, timeout=timeout, raw_body_override=payload)
        stats.turns += 1
        if res.reasoning_chars > 0:
            stats.turns_with_reasoning += 1
        if res.tool_calls:
            stats.turns_with_tool_calls += 1
        if res.ok:
            print(f"  [ok]   {f.name} finish={res.finish_reason} "
                  f"reasoning={res.reasoning_chars}c tools={len(res.tool_calls)}")
            continue
        stats.failures += 1
        stats.by_kind[res.failure_kind] = stats.by_kind.get(res.failure_kind, 0) + 1
        base = out_dir / f"{run_id}-replay-fail-{stats.failures:03d}-{res.failure_kind}-{f.stem}"
        with open(f"{base}.response.sse", "w", encoding="utf-8") as fh:
            fh.write(res.raw_sse)
        with open(f"{base}.summary.txt", "w", encoding="utf-8") as fh:
            fh.write(f"source: {f}\nfailure_kind: {res.failure_kind}\n"
                     f"http_status: {res.http_status}\nfinish_reason: {res.finish_reason}\n"
                     f"detail:\n{res.detail}\n")
        print(f"  [FAIL] {f.name} kind={res.failure_kind} :: {res.detail[:200]}")
        print(f"         saved: {base}.*")
        if stop_on_first:
            break
    _print_summary(stats)
    return 2 if (stats.failures and stop_on_first) else (1 if stats.failures else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.8 tool-call/reasoning stress harness")
    parser.add_argument("--url", default=None, help="Base URL (default: auto-detect 8082 then 8080)")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=30000,
                        help="Output cap per turn (0 = no cap / server decides; default allows up to 30K tokens).")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--catalog-size", type=int, default=60,
                        help="Number of tools to advertise (real Copilot sends dozens).")
    parser.add_argument("--client", choices=["vscode", "cli"], default="vscode",
                        help="Which client to emulate in the system prompt.")
    parser.add_argument("--replay", default=None,
                        help="Replay captured real request bodies (file or dir) instead of "
                             "synthetic tasks. Use with LLAMA_PROXY_CAPTURE_DIR captures.")
    parser.add_argument("--stop-on-first", action="store_true",
                        help="Stop as soon as the first failure is reproduced.")
    args = parser.parse_args()

    base_url = detect_url(args.url)
    api_key = read_api_key()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.replay:
        return run_replay(base_url, api_key, Path(args.replay), out_dir, run_id,
                          args.timeout, args.stop_on_first)

    tools = build_tools(args.catalog_size)
    system_msg = {"role": "system", "content": build_system_prompt(args.client)}

    stats = Stats()
    print(f"[start] url={base_url} iterations={args.iterations} max_turns={args.max_turns} "
          f"temp={args.temperature} run_id={run_id}")

    for i in range(args.iterations):
        prompt = TASK_PROMPTS[i % len(TASK_PROMPTS)]
        messages: list[dict[str, Any]] = [system_msg, {"role": "user", "content": prompt}]
        stats.iterations += 1

        for turn in range(args.max_turns):
            res = run_turn(base_url, api_key, messages, tools,
                           temperature=args.temperature, max_tokens=args.max_tokens,
                           timeout=args.timeout)
            stats.turns += 1
            if res.reasoning_chars > 0:
                stats.turns_with_reasoning += 1
            if res.tool_calls:
                stats.turns_with_tool_calls += 1

            if not res.ok:
                stats.failures += 1
                stats.by_kind[res.failure_kind] = stats.by_kind.get(res.failure_kind, 0) + 1
                stamp = datetime.now().strftime("%H%M%S")
                base = out_dir / f"{run_id}-fail-{stats.failures:03d}-{res.failure_kind}-{stamp}"
                with open(f"{base}.request.json", "w", encoding="utf-8") as fh:
                    json.dump({"url": base_url, "iteration": i, "turn": turn,
                               "temperature": args.temperature, "messages": messages,
                               "tools": tools}, fh, indent=2)
                with open(f"{base}.response.sse", "w", encoding="utf-8") as fh:
                    fh.write(res.raw_sse)
                with open(f"{base}.summary.txt", "w", encoding="utf-8") as fh:
                    fh.write(f"failure_kind: {res.failure_kind}\n"
                             f"http_status: {res.http_status}\n"
                             f"finish_reason: {res.finish_reason}\n"
                             f"reasoning_chars: {res.reasoning_chars}\n"
                             f"detail:\n{res.detail}\n")
                print(f"  [FAIL] iter={i} turn={turn} kind={res.failure_kind} "
                      f"finish={res.finish_reason} reasoning={res.reasoning_chars} :: {res.detail[:200]}")
                print(f"         saved: {base}.*")
                if args.stop_on_first:
                    _print_summary(stats)
                    print(f"[stopped on first failure] artifacts in {out_dir}")
                    return 2
                break  # abandon this conversation, move to next iteration

            # Turn OK. If no tool calls, the task is done.
            if not res.tool_calls:
                print(f"  [ok]   iter={i} done after {turn+1} turns "
                      f"(reasoning={res.reasoning_chars}c, content={len(res.content)}c)")
                break

            # Append assistant tool-call message + synthetic tool results, continue.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": res.content or None}
            assistant_msg["tool_calls"] = [
                {"id": tc.id or f"call_{turn}_{tc.index}", "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments or "{}"}}
                for tc in res.tool_calls
            ]
            messages.append(assistant_msg)
            for tc in res.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id or f"call_{turn}_{tc.index}",
                    "content": synth_tool_result(tc.name, tc.arguments),
                })
        else:
            print(f"  [ok]   iter={i} reached max_turns={args.max_turns}")

    _print_summary(stats)
    print(f"[done] artifacts (failures only) in {out_dir}")
    return 1 if stats.failures else 0


def _print_summary(stats: Stats) -> None:
    print("\n==================== SUMMARY ====================")
    print(f"iterations           : {stats.iterations}")
    print(f"turns (requests)     : {stats.turns}")
    print(f"turns w/ reasoning   : {stats.turns_with_reasoning}")
    print(f"turns w/ tool_calls  : {stats.turns_with_tool_calls}")
    print(f"FAILURES             : {stats.failures}")
    if stats.by_kind:
        for kind, n in sorted(stats.by_kind.items(), key=lambda kv: -kv[1]):
            print(f"   - {kind}: {n}")
    print("================================================\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
