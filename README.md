# MM Nodepack

**Four nodes that turn ComfyUI into a complete image workbench: load and preview real files, upscale and crop with a built-in model, and call vision / text LLMs — in a single install.**

## Image editing

- **One loader for single images & folder batches.** Replaces both *Load Image* and *Load Image Batch* in a single node. Switch between **load image** and **choose folder** with one click — the active button lights up and shows only the controls you need.
- **Folder batch (`incremental`, `random`, and `image list`).** Step through any folder in natural filename order (`1` = first image, wraps around after the last), pick randomly using the index as seed, or connect the **`image list`** output to run an entire folder in one click (every image keeps its own resolution).
- **Real paths & wireless sync.** Type, paste, or pick any file or folder on your machine. **Preview Image MM** automatically detects its **Load Image MM** source wirelessly — no extra link required.
- **Built-in compare.** Compare original vs. result right inside the preview: **hold** (default), **slide**, or **off**.
- **Save next to the source.** `house.png` → `house_1.png`, `house_2.png`, … — never overwrites, even across folder batch runs. **save as…** opens your native OS dialog directly in the target folder.
- **Iterate in one click.** **send to loader** (or **auto send**) feeds the creation straight back into **Load Image MM** so the next run edits the new image.
- **Free GPU tile.** A compact toolbar button shows how much VRAM ComfyUI currently reserves and frees it on click.

## Upscale & crop

- **Five ways to ask for a size.** Percent, factor, shortest side, longest side or megapixels all keep the aspect ratio — and every one of them also works downwards, so the node doubles as a downscaler. `custom` takes a width x height box and gives you exactly that resolution (cover + crop).
- **The upscale model is built in.** No loader node: the model picker reads your `models/upscale_models` folder directly, tiling is automatic and halves itself when VRAM runs short. The other methods are plain kernels (lanczos, bicubic, …).
- **Crop only cuts, never rescales.** Classic formats (4:5, 1:1, 16:9, DIN, …) with an orientation that can follow the frame, plus position, offsets and a crop window you can drag right in the live preview panel.
- **Image and latent, framed identically.** Both inputs are optional and both are handled; the crop is applied in normalized coordinates so a mixed graph comes out framed the same.

## LLM

- **One node for vision and text LLMs.** llama.cpp (incl. NInfer), Ollama, Unsloth Studio and any OpenAI-compatible server. `auto` mode goes to vision when an image is connected and to text otherwise.
- **Local models, no setup.** `local_model` starts its own llama-server (port 8081, overridable via `LLM_MULTITOOL_PORT`) for any GGUF found in your HF cache or ComfyUI's `LLM` / `text_encoders` / `clip` folders — with an `mmproj` next to it the model can see images, without one it runs text only.
- **Prompt boxes that grow with their text**, an optional `text` input, sampling, chain-of-thought on a separate `thinking` output, clean-output mode, retries with exponential back-off.
- **VRAM on demand.** The **unload model** button frees VRAM at any time (own llama-server plus the model at `api_url` — llama.cpp router, Ollama); `unload_after` does the same after every run, which is the right choice when sharing the GPU with rendering.

![send to loader](assets/send_to_loader.gif)

*Flux.2 Klein edit: "add a cactus" → send → "make it blue" → send → "pink and white striped" → send.*

---

## Nodes

### Load Image MM

Two mode buttons sit at the top — the active one is bright, the other dimmed (click either at any time to switch):

- **load image** *(single image mode)*
  - Pick an image via the browser file picker (uploads to `ComfyUI/input`; if an identical original exists in `ComfyUI/output`, `path` resolves to that original).
  - Or type/paste any full file path into `path` (`path` always takes priority over the upload list).
- **choose folder** *(folder batch mode)*
  - Opens your native OS folder picker and loads images directly from that folder in natural order (`img_2.png` before `img_10.png`).
  - **incremental / random** toggle:
    - **incremental** steps through the folder using a **1-based `index`** (`1` = first image; numbers past the last image wrap cleanly back to `1`).
    - **random** uses `index` as a deterministic seed to pick an image from the folder.
  - Works seamlessly with ComfyUI's `control_after_generate` (`increment` / `randomize` / `fixed`) and Queue Batch Count.
  - Changing the folder, mode, or index immediately updates the node's thumbnail preview.
- **Outputs:** `image`, `mask`, `width`, `height`, `path`, `image list` (ComfyUI `IMAGE` list of all images in the folder — each image keeps its original resolution; zero overhead when unconnected), `index` (1-based index of the loaded image in the folder, or `1` in single mode).

### Preview Image MM

Replaces **Preview Image**, **Save Image**, and an **Image Comparer** in a single node:

- **Compare modes** under the image: **compare hold** (press & hold to view the original — default), **compare slide** (drag/hover divider), or **compare off**. Saved per workflow.
- **save** — writes the selected result next to the source file as `<name>_1`, `<name>_2`, … without overwriting.
- **save as…** — opens the native OS save dialog pre-filled in the target folder (uses the browser save picker when accessed remotely).
- **send to loader** — saves as `_#` and loads the result into **Load Image MM** (automatically switches the loader to single-image mode so the next run edits it).
- **Auto switches:**
  - `auto_save` saves every generated image automatically (`suffix`, `save_as`, `format`: same as source / png / jpg / webp, `quality`, `embed_workflow`).
  - `auto_send` sends every result straight back to **Load Image MM** (reuses the auto-saved file when `auto_save` is enabled).
  - Manual buttons already handled by an active auto switch are dimmed automatically.
- **Wireless source detection:** `source = auto` automatically finds the upstream (or only) **Load Image MM** in the workflow. With multiple loaders, enter its node number (e.g. `#12`).

### Upscale Crop MM

Upscale and crop in one node:

- **UPSCALE** sets the size: `target_mode` picks how you ask for it — `scale_percent`, `scale_factor`, `shortest_side`, `longest_side`, `megapixels` (all keep the aspect ratio) or `custom` (a `target_width` x `target_height` box the node covers and crops to, so the resolution you type is the resolution you get). Every mode works downwards as well.
- **THE MODEL IS BUILT IN.** `method = upscaler model` picks the file straight from `models/upscale_models` (UltraSharp, Remacri, Siax, RealESRGAN rank first if installed) — no loader node to wire up. The rest are plain kernels (`lanczos`, `bicubic`, `bilinear`, `nearest`, `area`). Tiling is automatic and halves itself if VRAM runs short; a downscale target skips the model and uses lanczos.
- **CROP** is off by default and only ever cuts, never rescales. Pick a format (`4:5`, `1:1`, `16:9`, `DIN`, …, or `custom` for a `target_width` x `target_height` box), an orientation (`auto` follows the frame — right for mixed batches), a position with optional offsets, or place the window by dragging it in the preview panel.
- **IMAGE and LATENT are both optional and both handled**; the crop is applied in normalized coordinates so both come out framed identically.
- **EXACT SIZE BY DEFAULT.** With `snap` off you get exactly the size you asked for; with it on the result is rounded to `multiple_of` (8 suits SD, SDXL and Flux).
- **THE PREVIEW TELLS YOU WHAT WILL HAPPEN.** A live panel shows the size, the factor and the crop window, computed by the same maths the node runs; a resize the `resize_condition` failsafe would skip is shown as skipped, not promised.

### LLM Multitool MM

One node for vision and text LLM calls:

- **Modes:** `auto` (vision when an image is connected, otherwise text), `vision` (image + text prompt → text reply) and `text` (text only).
- **Model:** `local_model` starts its own llama-server (port 8081, env `LLM_MULTITOOL_PORT`) for a local GGUF found in the HF cache or ComfyUI's `LLM` / `text_encoders` / `clip` folders — with an `mmproj` next to it it can see images, without one it is listed as `<name> (text)` and runs text only. `server (api_url)` talks to whatever the server at `api_url` serves; `backend` picks the API dialect (`llama.cpp`, `ollama`, `openai`, `unsloth`) and `model_id = auto` uses whatever that server currently has loaded.
- **Prompts:** `system_prompt` + `user_prompt` boxes that grow with their text, an optional `text` input (inserted at `{text}`, or appended below), and `extra_body` for any server-specific JSON override.
- **Outputs:** `response` (the reply, or an `ERROR: …` line when `on_error = return text`), `thinking` (chain-of-thought when `enable_thinking` is on), `active_model` and `active_mode`.
- **The node shows its active model** in a read-only line — no extra wire needed to see what answered.
- **VRAM:** the **unload model** button stops the own llama-server and frees the model at `api_url` (llama.cpp router, Ollama); `unload_after` does the same after every run. Recommended when sharing the GPU with rendering.

---

## Install

One install for all four nodes — via **ComfyUI Manager** / **Comfy Registry** by searching **MM Nodepack**, or clone manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/metamountain/mm-nodepack-comfyui
```

No extra Python dependencies required. (The LLM node only needs a running LLM server or a local GGUF + `llama-server` on `PATH` / `LLAMA_SERVER_BIN`.)

---

## Notes

- Images pasted with `Ctrl+V` have no original folder on disk, so **save** opens the save dialog to ask where to store them.
- When ComfyUI is accessed from another computer (`--listen`), server-side saving is restricted to the ComfyUI `output` folder.
- `LLM Multitool MM` never stores API keys in the workflow: `api_key_env` names an environment variable, and the key is read from the environment at run time.

## License

MIT
