<p align="left">
  <img src="assets/logo.svg" width="48" height="48" alt="MM Nodepack Logo" />
</p>

# MM Nodepack

Four small ComfyUI nodes that turn ComfyUI into a real image workbench: load and batch real files from disk, preview and save your edits right next to the source image, upscale and crop with a built-in model, and call local or remote LLMs. One install, one consistent UI, and built-in help on every node (the `?` button).

![send to loader](assets/send_to_loader.gif)

*Flux.2 Klein edit loop: "add a cactus" → send → "make it blue" → send → "pink and white striped" → send.*

All nodes live in the **MM Nodepack** category. Each one explains itself in-app — the `?` button on a node opens the same page as the [docs below](#nodes).

## Nodes

### Load Image MM

One loader for a single image or a folder batch — OS file dialog, drag & drop, paste or a plain file path.

- Single image (upload, dialog, drag & drop, paste or full path) or a folder as a batch
- Step through a folder incrementally (`index`) or at random; `image list` gives the whole folder
- Saves and other nodes find it automatically as the source image

Full reference: [MMLoadImage.md](web/docs/MMLoadImage.md)

### Preview & Save MM

See the results, compare them with the source (hold or slider) and save the one you keep.

- Saves next to the source image as `name_1`, `name_2`, … and never overwrites
- `save as…` and **send to loader** (saves the result and feeds it back for the next run)
- Optional `auto save` / `auto send` switches

Full reference: [MMPreviewSave.md](web/docs/MMPreviewSave.md)

### Upscale Crop MM

Resize an image — percentage, factor, side, megapixels or an exact box — and crop it to a format like 4:5, 1:1 or 16:9.

- Works down as well as up; an upscaler model from your `models/upscale_models` folder can replace the plain resampler
- Classic formats or a custom box, with position, offsets and a draggable crop window in the live preview
- `info` output tells you exactly what ran

Full reference: [MMUpscaleCrop.md](web/docs/MMUpscaleCrop.md)

### LLM Multitool MM

Ask a local or remote LLM for a text answer, or describe and analyse images with a vision model.

- llama.cpp, NInfer, Ollama, Unsloth Studio or any OpenAI-compatible server
- Can start its own local server for a GGUF model found on your machine
- `unload model` button and `unload_after` keep the GPU free for rendering

Full reference: [MMLLMMultitool.md](web/docs/MMLLMMultitool.md)

Plus a small **free GPU** tile in the top action bar: it shows how much GPU memory ComfyUI itself holds and unloads all models on click.

## Install

Shortest ways, either works:

1. **ComfyUI Manager** — search for **MM Nodepack** and install.
2. **git** — clone into `ComfyUI/custom_nodes` and restart ComfyUI:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/metamountain/mm-nodepack-comfyui
   ```

No extra Python dependencies. If you want `LLM Multitool MM` to start its own local server, put `llama-server` on your `PATH` (or point `LLAMA_SERVER_BIN` at it); otherwise any running LLM server works.

## Shared server

Running ComfyUI for other people? The pack locks down file access: it reads only from the `input/` and `output/` folders, writes only to `output/`, and never overwrites a file. LLM targets must be approved by you as the operator. Two environment variables in the ComfyUI start script control this: `MM_NODEPACK_SHARED_SERVER` (on/off, otherwise it switches on automatically for a non-loopback `--listen` address) and `MM_NODEPACK_LLM_PROFILES` (the list of allowed LLM servers). Details: [Load Image MM](web/docs/MMLoadImage.md#shared-server) and [LLM Multitool MM](web/docs/MMLLMMultitool.md#shared-server).

## License

MIT
