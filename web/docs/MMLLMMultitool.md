# LLM Multitool MM (`MMLLMMultitool`)

Ask a local or remote LLM for a text answer, or describe and analyse images with a vision model. Works with llama.cpp, NInfer, Ollama, Unsloth Studio and any OpenAI-compatible server. The node can also start its own local server for a GGUF model.

## How to use

1. Pick where the model comes from: `local_model` = **server (api_url)** talks to the server at `api_url`, or pick a GGUF name to start your own local server (an `mmproj` file next to it enables vision).
2. Set `backend` to the server's API (`llama.cpp`, `ollama`, `openai`, `unsloth`).
3. Write your `system_prompt` and `user_prompt`; `mode` = `auto` switches to vision when you connect an image.
4. Adjust sampling (`temperature`, `top_p`, `top_k`, `min_p`) and `max_tokens` to taste.
5. Queue the workflow: the reply comes out on `response`, reasoning on `thinking`, and the node shows which model answered.
6. Free the GPU again with the **unload model** button, or set `unload_after` to unload after every run.

## Inputs / Outputs

- **Inputs:** `local_model`, `api_url`, `backend`, `model_id`, `api_key_env`, `mode`, `system_prompt`, `user_prompt`, sampling (`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`), `max_tokens`, `seed`, `enable_thinking`, `reasoning_effort`, `clean_output`, `image_mode`, `unload_after`, `keep_alive`, `on_error`, `timeout_s`, `max_retries`, `extra_body`, `image`, `text`
- **Outputs:** `response`, `thinking`, `active_model`, `active_mode`

## Tips

- `model_id` = `auto` uses whatever model the server currently has loaded.
- Sharing the GPU with rendering? Keep `unload_after` on, or use the **unload model** button between runs.
- Own local server: the node estimates what the model needs (weights, projector, KV cache for the context), unloads only as many ComfyUI models as that takes, and llama.cpp puts as many layers on the GPU as then fit; the rest run on the CPU. `LLM_MULTITOOL_DEVICE` = `gpu` or `cpu` forces one side (default `auto`).
- `clean_output` strips think blocks, Markdown and quotes — useful when the reply goes straight into a render prompt.
- `api_key_env` names an environment variable on the server; the key itself is never stored in the workflow.
- `extra_body` passes extra JSON options, but reserved fields (model, limits, stream, auth) cannot be overridden.

## Shared server

On a shared server, every LLM target you choose must be approved by the operator: they list the allowed servers in a profile file (the `MM_NODEPACK_LLM_PROFILES` variable) — per server: the API, the URL, the key variable, and the permissions (remote unload, starting a local model, which models, response size limit). Any combination that is not on that list is rejected before a request is made, and error messages never echo keys or full URLs. On a private (local) server none of this applies.
