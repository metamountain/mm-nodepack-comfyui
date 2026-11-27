# LLM Multitool MM (`MMLLMMultitool`)

One node for vision and text LLM calls against **llama.cpp**, **NInfer**, **Ollama**, **Unsloth Studio**, and any **OpenAI-compatible** server.

## Modes (`mode`)

- **`auto`** *(default)* — Uses `vision` when an `image` is connected, otherwise `text`.
- **`vision`** — Sends `image` + prompt to a vision-capable model.
- **`text`** — Text-only prompt.

When multiple images are passed in a batch, **`image_mode`** switches between **`single`** (first frame only) and **`batch`** (all frames).

## Model & Server Options

- **`local_model = server (api_url)`** — Talks to whatever server is running at `api_url` (default `http://127.0.0.1:8080`, Ollama `http://127.0.0.1:11434`). Set `model_id = auto` to use the server's currently loaded model.
- **`local_model = <GGUF>`** — Automatically starts a managed `llama-server` on port `8081` (override via `LLM_MULTITOOL_PORT`) for any GGUF found in your HF cache or ComfyUI's `LLM` / `text_encoders` / `clip` folders. If an `mmproj` file sits beside it, vision works automatically.
- **`backend`** — API dialect: `llama.cpp` (also covers NInfer), `ollama`, `openai`, or `unsloth`.
- **NInfer failsafes:** Automatically clamps `top_k` to NInfer's `[0, 20]` range and reports a clear hint if `vision` is called against a text-only NInfer profile.

## Prompts & Security

- **`system_prompt`** and **`user_prompt`** boxes grow automatically with their text.
- Optional **`text`** input is inserted at `{text}` in `user_prompt` (or appended below).
- **`api_key_env`** names an environment variable (e.g. `OPENAI_API_KEY`) read at run time — keys are **never** saved inside workflow JSON.
- **`extra_body`** accepts optional JSON merged into the request payload. Reserved fields may **not** be overridden: `model`, `messages`, `stream`, the max-token limits (`max_tokens` / `num_predict` / …) and any url / auth / token / header-like key — also nested in backend options like Ollama's `options`.

## Shared Server: LLM Profiles

When ComfyUI is **shared** (same switch as the file policy: non-loopback `--listen`, or `MM_NODEPACK_SHARED_SERVER=1`), every LLM request — a queued node **and** the `unload model` route, which has no browser request to authenticate — must match an **operator-owned profile** before any credential is read, any discovery happens, any connection is opened, or the managed `llama-server` is started or stopped.

Profiles live in one JSON file pointed to by the environment variable **`MM_NODEPACK_LLM_PROFILES`** (set in the ComfyUI start script):

```json
{
  "profiles": [
    {
      "id": "ninfer",
      "backends": ["llama.cpp"],
      "urls": ["http://127.0.0.1:8080"],
      "api_key_env": "NINFER_API_KEY",
      "allow_unload": true,
      "allow_local_unload": false,
      "allow_local_model": true,
      "local_models": ["Qwen3.8-27B"],
      "max_response_bytes": 2097152
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `backends` | API dialects this profile covers (`llama.cpp`, `ollama`, `openai`, `unsloth`). |
| `urls` | Allowed base URLs (compared normalized, `http://host:port`). |
| `api_key_env` | Environment variable holding the key; `""` = no key required. |
| `allow_unload` | Remote unload: the `unload model` route and `unload_after` against these URLs. |
| `allow_local_unload` | The route may stop the managed local `llama-server`. |
| `allow_local_model` | A local GGUF `llama-server` may be started. |
| `local_models` | Optional: only these scanned model labels may be selected (empty/omitted = any). |
| `max_response_bytes` | Response size limit (default 8 MiB) — enforced while streaming, before JSON decoding. |

A workflow is accepted when **backend + normalized `api_url` + `api_key_env`** match one profile exactly (local runs match on the label). Any unknown combination is rejected with a **redacted** error — it never echoes the key, the env-var value, or a URL with credentials. Requests with a key never follow redirects. On a **private loopback server** none of this applies: existing workflows and the `unload model` button keep working unchanged (a non-local request on a private server is still profile-bound, like the image routes).

## VRAM Management

- **`unload model` button** on the node immediately stops the managed local server and unloads the active model on `api_url` (llama.cpp router / Ollama).
- **`unload_after = True`** unloads automatically after every run (recommended when sharing the GPU with ComfyUI rendering).

## Outputs

| Output | Type | Description |
|---|---|---|
| `response` | `STRING` | Model reply (or `ERROR: …` when `on_error = return text`). |
| `thinking` | `STRING` | Chain-of-thought content when `enable_thinking = True`. |
| `active_model` | `STRING` | Model ID that answered (also shown directly on the node). |
| `active_mode` | `STRING` | Resolved operating mode (`vision` or `text`). |
