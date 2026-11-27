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
| `MMLoadImage` | **Load Image MM** | Single image (1-click OS file dialog, drag-drop/paste, or real path) + folder batching (`incremental`, `random`, natural-order `image list`, `index`) with OS folder picker | [MMLoadImage.md](web/docs/MMLoadImage.md) |
| `MMPreviewSave` | **Preview & Save MM** | Live preview, built-in compare (`hold` / `slide` / `off`), wireless source sync, non-overwriting save (`_1`, `_2`, …, PNGs always embed workflow metadata), `save as…`, `send to loader`, safe `auto_save` / `auto_send` | [MMPreviewSave.md](web/docs/MMPreviewSave.md) |
| `MMUpscaleCrop` | **Upscale Crop MM** | Image-only: 6 sizing modes (`scale_factor` default, up & downscale, remembered last setting), built-in upscale model (`lanczos` default, GPU → tile-halving → CPU auto-fallback, RAM-OOM guard, alpha preservation), 9-position + random crop with draggable live preview, failsafe `resize_condition` | [MMUpscaleCrop.md](web/docs/MMUpscaleCrop.md) |
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

## Shared server file policy

When ComfyUI is shared (reachable by other machines or used by several people), file access is restricted so that a request — or a queued node — cannot read or write arbitrary files on the ComfyUI computer:

- **Reads** (image load, folder listing, previews, upload-origin lookup) are limited to ComfyUI's `input/` and `output/` folders.
- **Writes** (save, save-as, auto-save) are limited to the `output/` folder. Files are created exclusively: never overwriting, never taking over a pre-created name or a link/hard link planted on it; a failed save leaves no partial file.
- Client-facing errors are bounded and never disclose file names outside these folders.

The policy switches on automatically when ComfyUI starts with a non-loopback `--listen` address (`127.0.0.1`, `::1`, `localhost` count as loopback; in a comma-separated list any non-loopback entry switches it on). It can be overridden with the environment variable `MM_NODEPACK_SHARED_SERVER`: `1/true/yes/on` forces the policy on, `0/false/no/off` forces it off (an explicit setting always wins).

On a private (loopback-only) server, every **non-local** request is still restricted in the same way; trusted local requests keep the unrestricted behaviour, including OS-dialog files outside the ComfyUI tree and saves next to the source.

## Notes

- Pasted images (`Ctrl+V`) have no original folder on disk, so **save** opens the save dialog to ask where to store them.
- When ComfyUI is accessed remotely (`--listen`), server-side saving is restricted to ComfyUI's `output` directory.
- `LLM Multitool MM` never stores API keys in workflows: `api_key_env` names an environment variable read at run time.

## License

MIT
