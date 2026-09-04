# GitHub BYOK Vision Endpoint Contract

Provider conformance specification for **GitHub organization/enterprise BYOK** using an **OpenAI-compatible Chat Completions endpoint**.

> Scope: GitHub-managed organization/enterprise BYOK with endpoint type **Chat completions API**.
> Not in scope: local VS Code BYOK, the OpenAI Responses API, or the Anthropic Messages API, PDF vision, or image generation. The requirement is image understanding that returns text/tool calls.

## 1. Connectivity and authentication

The endpoint must:

- Be reachable from GitHub's servers, not merely from the developer's machine.
- Use publicly trusted HTTPS.
- Accept `Authorization: Bearer <configured API key>`.
- Accept `Content-Type: application/json`.
- Not require browser CORS; communication is server-to-server.
- Return standard HTTP status codes and OpenAI-shaped errors.

## 2. Required endpoints

### Runtime endpoint

- `POST /v1/chat/completions`, or the exact Chat Completions URL configured in GitHub.

### Model discovery

When GitHub's **Fetch new models** feature is used:

- `GET /v1/models`
- Return `object: "list"` and `data` entries containing at least `id`, `object: "model"`, `created`, and `owned_by`.
- The returned `id` must exactly match the model identifier accepted by Chat Completions.

> Model discovery does **not** advertise vision. Vision metadata is controlled separately by GitHub's model configuration.

### Capability fields in the model list (observed behavior, 2026-08)

llama.cpp's `/v1/models` advertises capabilities (`["completion","multimodal","vision"]`) only in its own `models[]` entries. The OpenAI-standard `data[]` entries carry **no** `capabilities` field by default, and OpenAI-shaped discovery clients read `data[]` — so a multimodal model can look text-only.

The Cloudflare proxy rewrites `/v1/models` on the way out (`rewrite_model_capabilities_payload`):

- Adds the `"vision"` alias to every `models[]` entry whose capabilities already include `"multimodal"`.
- **Mirrors** the normalized capabilities onto each matching `data[]` entry (matched by `id`, with an `aliases` fallback), so both sections advertise `["completion","multimodal","vision"]`. Text-only models are left untouched and the rewrite is idempotent.

Both fields are now present on the live endpoint and through the public tunnel.

## 2a. GitHub BYOK UI probe requests

When a GitHub org admin opens the custom-model configuration for an OpenAI-compatible endpoint, GitHub's servers probe the origin with this exact sequence (captured in the service journal, 2026-08-10):

| # | Request | Healthy response |
|---|---|---|
| 1 | `GET /props` | `200` — llama.cpp server global properties |
| 2 | `GET /props?autoload=false` | `200` |
| 3 | `POST /v1/streams/lookup` | `200` with a JSON array (`[]` when no sessions) |
| 4 | `GET /v1/models` | `200` — model list |
| 5 | `GET /props?model=<id>&autoload=false` | `200` — per-model props |

Two of these are llama.cpp-native endpoints rather than OpenAI endpoints:

- **`GET /props`** returns the server properties and, critically, `modalities: {"vision": true, "audio": false}` for a multimodal stack. This is the strongest signal GitHub can use to detect vision on a llama.cpp endpoint — more specific than `/v1/models` capabilities.
- **`POST /v1/streams/lookup`** is llama.cpp's resumable-stream session lookup (body `{"conversation_ids": [...]}`, response is an array of `ApiStreamSession` entries). Builds without the resumable-stream feature (e.g. the PrismML fork) answer `404`, which a strict client can read as "streaming unsupported".

The Cloudflare proxy handles the 404: for `POST /v1/streams/lookup` it validates the body, relays the backend's answer when the endpoint is implemented (non-404), and otherwise answers exactly like a llama.cpp server with no live sessions — `200` + `[]` (see `_proxy_stream_lookup`).

> All five probes return `200` on the live endpoint, locally and through the tunnel.

## 3. Multimodal request format

The endpoint must accept:

- `model`
- `messages`
- `stream`
- `max_tokens` or `max_completion_tokens`
- `temperature`, `top_p`, and other ordinary optional parameters
- `tools`, `tool_choice`, and optionally `parallel_tool_calls`

A user message containing an image arrives as a content array containing:

- Text part: `type: "text"` with `text`
- Image part: `type: "image_url"` with `image_url.url`
- Optional image detail: `auto`, `low`, or `high`

The image URL can be:

- An HTTPS URL, potentially signed or redirected.
- A data URL such as `data:image/png;base64,<encoded bytes>`.

The endpoint must decode or download the image and send the actual pixels to the multimodal model. It must not flatten the content array into text or silently discard `image_url`.

It should support:

- PNG and JPEG as a minimum.
- WebP and non-animated GIF.
- BMP by transcoding it when possible, because VS Code can attach BMP images.
- Standard padded or unpadded Base64; accepting URL-safe Base64 is recommended for client compatibility.
- Multiple images and images retained in conversation history.

## 4. Tool-calling compatibility

For Copilot Agent mode, the endpoint must also support:

- OpenAI function tools with JSON Schema parameters.
- `tool_choice`: `none`, `auto`, `required`, or a named function.
- Assistant messages containing `tool_calls`.
- Tool-result messages containing `role: "tool"` and `tool_call_id`.
- Image input and tool definitions in the same request.
- Streamed tool calls using stable IDs and indexed argument fragments.

> Tool calling and streaming are not intrinsically required for vision, but they are required for the model to work correctly in Copilot Agent mode.

## 5. Streaming response

When `stream: true`, return:

- HTTP `200`.
- `Content-Type: text/event-stream`.
- UTF-8 SSE events beginning with `data:`.
- OpenAI `chat.completion.chunk` objects.
- Stable completion ID and model name across chunks.
- Text in `choices[0].delta.content`.
- Terminal `finish_reason`, normally `stop`, `length`, or `tool_calls`.
- Final `data: [DONE]`.

If tool calls are generated, stream them through `delta.tool_calls` with:

- Stable tool-call `id`
- `type: "function"`
- Function `name`
- Incrementally streamed JSON-string `arguments`
- Final `finish_reason: "tool_calls"`

A usage-only chunk with empty `choices` is optional but should be supported when `stream_options.include_usage` is requested.

## 6. Error and operational requirements

The endpoint should:

- Return errors before starting SSE, using appropriate statuses such as `400`, `401`, `404`, `413`, `429`, or `5xx`.
- Return an OpenAI-shaped `error` object.
- Never return HTTP `200` containing a provider-specific error document.
- Permit sufficient request size for Base64 expansion; at least 8–10 MiB is recommended.
- Cancel inference when the caller disconnects.
- Return `Retry-After` for rate limiting.
- Avoid logging API keys or image data.
- Log only safe diagnostics such as request ID, MIME type, decoded size, dimensions, and selected model.

## 7. Separate GitHub requirement

Even a perfectly compliant endpoint will not receive images unless GitHub advertises the model as vision-capable.

GitHub's Copilot model catalog must contain `capabilities.supports.vision: true`. The standard provider `/v1/models` response cannot set this. It comes from the **Vision** checkbox in GitHub's custom-model configuration.

> Known issue: when GitHub's catalog omits that field, VS Code removes the image before calling the provider. The provider contract becomes testable only after the GitHub metadata issue is resolved.

### How the catalog entry is built (observed 2026-08)

- The catalog entry for a llama.cpp BYOK model is a flat object shaped like:
  ```json
  {
    "id": "<org>/<key-name>/<model-id>",
    "name": "Qwen 3.8",
    "object": "model",
    "vendor": "Experimental",
    "custom_model": { "provider": "openaicompatible", "owner_name": "<org>", "key_name": "<key-name>" },
    "capabilities": {
      "family": "custom",
      "limits": { "max_context_window_tokens": 100000, "max_output_tokens": 30000, "max_prompt_tokens": 69997 },
      "supports": { "streaming": true, "tool_calls": true },
      "tokenizer": "cl100k_base",
      "type": "chat"
    }
  }
  ```
- `capabilities.supports` is the **only** vision signal VS Code reads for org BYOK models: `this.supportsVision = !!e.capabilities.supports.vision` in the bundled Copilot extension. There is **no client-side merge** from provider discovery for org custom models.
- VS Code refreshes the catalog from GitHub's CAPI (`api.githubcopilot.com`) roughly every 10 minutes while active, and caches it in debug-log snapshots (`.../GitHub.copilot-chat/debug-logs/*/models.json`). Changes can take up to that refresh to appear; a full VS Code reload forces it sooner.
- **Checkbox persistence gotcha:** `streaming` and `tool_calls` were observed persisting as `true`, but `vision` never appeared in any snapshot — even after the admin said the Vision box was ticked. The checkbox mechanism works, so a missing `vision` key means either the tick did not save/propagate or the tick was applied to a different model entry.

### Provider-side mitigation (implemented 2026-08-10)

The proxy now gives GitHub's discovery every redundant signal it can:

- `/v1/models`: `capabilities: ["completion","multimodal","vision"]` on **both** `models[]` and `data[]`.
- `/props`: `modalities.vision: true` (from the backend, no rewrite needed).
- `/v1/streams/lookup`: returns `200 []` instead of the fork's 404.

These do **not** set GitHub's catalog `supports.vision` — that still requires the Vision checkbox. But they remove every "discovery thought the model was text-only" failure mode, and match what OpenAI-compatible and llama.cpp-native fetchers each look for.

### Admin-side checklist

1. Confirm the tick is on the model with the **exact key name** and **model id** that the endpoint serves. Orgs often have several endpoint keys; ticking the wrong entry changes nothing.
2. Toggle Vision **off → on → Save**, or delete the model entry and re-run **Fetch new models**, then tick Vision again. Re-fetching/renaming a model can reset capability flags.
3. Reload VS Code (or wait ≤10 min for the catalog refresh) and check the model picker for the Vision badge.
4. `vendor: "Experimental"` (GitHub's openaicompatible BYOK) may have incomplete vision-flag plumbing — a known upstream limitation to keep in mind if steps 1–3 fail.

## 7a. Diagnosing vision-support metadata

To see exactly what VS Code believes about a BYOK model (read-only):

```bash
# The catalog snapshot VS Code fetched from GitHub's CAPI lives in per-workspace
# debug logs. Find the newest one:
find ~/.vscode-server/data/User/workspaceStorage/*/GitHub.copilot-chat/debug-logs \
  -name models.json -printf '%T@ %p\n' | sort -rn | head -1
```

```python
# Check the entry's capabilities.supports for the model:
import json, sys
d = json.load(open(sys.argv[1]))
MODEL_ID = "<your-model-id>"  # e.g. "PAQ_LLAMACPP_SERVER" or whatever id your endpoint serves
def walk(o):
    if isinstance(o, dict):
        if MODEL_ID in str(o.get("id", "")):
            print(o.get("id"), "->", json.dumps((o.get("capabilities") or {}).get("supports", {})))
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(d)
```

Interpretation:

- `supports: {streaming: true, tool_calls: true}` with **no** `vision` key → GitHub's CAPI is not sending vision; the provider is not the problem. Re-check the admin checkbox.
- `supports: {..., vision: true}` → the catalog is correct; if VS Code still strips images, look at the provider side (ingress logs for `image_url`) or client cache (reload window).
- No matching model entry at all → the endpoint key or model wasn't added/saved on GitHub's side.

The extension's decision point can be confirmed in the bundled Copilot extension bundle:

```js
this.supportsVision = !!e.capabilities.supports.vision
```

(`.../server/extensions/copilot/dist/extension.js` — org BYOK models get `capabilities` only from the GitHub catalog; there is no client-side vision merge.)

## 8. Acceptance tests

1. Text-only streamed request succeeds.
2. One small PNG/JPEG data URL produces a description grounded in the image.
3. An HTTPS image URL produces the same result.
4. Image plus tools works with streaming enabled.
5. A complete tool-call/tool-result round trip succeeds.
6. GitHub's model catalog shows `supports.vision: true`.
7. VS Code displays the Vision badge and provider ingress logs confirm receipt of `image_url`.
8. The GitHub BYOK UI probe sequence returns `200` for every request:
   - `GET /props`, `GET /props?autoload=false`, `POST /v1/streams/lookup` (expect `200 []`), `GET /v1/models`, `GET /props?model=<id>&autoload=false`
9. `/v1/models` advertises `["completion","multimodal","vision"]` in **both** `models[]` and `data[]` for the multimodal model, and `/props` reports `modalities.vision: true`.

## References

- [Bring your own key for GitHub Copilot](https://docs.github.com/en/copilot/concepts/models/bring-your-own-key)
- [Enabling custom models for GitHub Copilot in your organization](https://docs.github.com/en/copilot/how-tos/administer-copilot/manage-for-organization/enable-custom-models)
- [OpenAI Chat Completions API reference](https://developers.openai.com/docs/api-reference/chat/create)
- [OpenAI Models API reference](https://developers.openai.com/docs/api-reference/models/list)
- [OpenAI Images and vision guide](https://developers.openai.com/api/docs/guides/images)
