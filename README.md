<p align="left">
  <img src="assets/logo.svg" width="48" height="48" alt="MM Nodepack Logo" />
</p>

# MM Nodepack

**Four minimal, unified ComfyUI V3 nodes that turn ComfyUI into a complete image workbench — load & batch real files, compare & save edits next to the source, upscale & crop with a built-in model, and call vision/text LLMs — in a single install.**

All nodes live in the **`MM Nodepack`** category and share one consistent dark UI, progressive disclosure, and built-in in-app help (`?` button on every node).

![send to loader](assets/send_to_loader.gif)

*Flux.2 Klein edit loop: "add a cactus" → send → "make it blue" → send → "pink and white striped" → send.*

---

## Nodes & Documentation

Click any node for the full reference (the exact same pages also appear inside ComfyUI's `?` node help sidebar):

| Node | Display Name | What It Does | Help Page |
|---|---|---|---|
| `MMLoadImage` | **Load Image MM** | Single image (upload or real path) + folder batching (`incremental`, `random`, `image list`, `index`) with OS folder picker | [MMLoadImage.md](web/docs/MMLoadImage.md) |
| `MMPreviewImage` | **Preview Image MM** | Live preview, built-in compare (`hold` / `slide` / `off`), wireless source sync, non-overwriting save (`_1`, `_2`, …), `save as…`, `send to loader`, `auto_save` / `auto_send` | [MMPreviewImage.md](web/docs/MMPreviewImage.md) |
| `MMUpscaleCrop` | **Upscale Crop MM** | 6 sizing modes (up & downscale), built-in upscale model widget (OOM-safe tiling, automatic alpha preservation), 9-position + random crop with draggable live preview, failsafe `resize_condition`, image + latent | [MMUpscaleCrop.md](web/docs/MMUpscaleCrop.md) |
| `MMLLMMultitool` | **LLM Multitool MM** | Vision & text LLM calls (`auto` / `vision` / `text`) for llama.cpp, NInfer, Ollama, Unsloth Studio, and OpenAI-compatible APIs, optional local GGUF server, VRAM unload button | [MMLLMMultitool.md](web/docs/MMLLMMultitool.md) |

Also includes a compact **Free GPU** toolbar tile showing ComfyUI's reserved VRAM and unloading models on click.

---

## Install

Install via **ComfyUI Manager** / **Comfy Registry** (search **MM Nodepack**), or clone directly:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/metamountain/mm-nodepack-comfyui
```

No extra Python dependencies required. (`LLM Multitool MM` only needs a running LLM server or a local GGUF + `llama-server` on `PATH` / `LLAMA_SERVER_BIN`.)

---

## Notes

- Pasted images (`Ctrl+V`) have no original folder on disk, so **save** opens the save dialog to ask where to store them.
- When ComfyUI is accessed remotely (`--listen`), server-side saving is restricted to ComfyUI's `output` directory.
- `LLM Multitool MM` never stores API keys in workflows: `api_key_env` names an environment variable read at run time.

## License

MIT
