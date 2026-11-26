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
- **`extra_body`** accepts optional JSON merged into the request payload.

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
