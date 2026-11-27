<p align="left">
  <img src="assets/logo.svg" width="48" height="48" alt="MM Nodepack Logo" />
</p>

# MM Nodepack

The basics ComfyUI is missing. Load from anywhere, save next to the source, send the result back, run again.

![send to loader](assets/send_to_loader.gif)

*Edit loop: "add a cactus" → send → "make it blue" → send → "pink and white striped" → send.*

## Nodes

- **Load Image MM**: any file or a whole folder, with its real path.
- **Preview & Save MM**: compare with the source, save next to it, **send** it back to the loader.
- **Upscale Crop MM**: resize and crop to any format.
- **LLM Multitool MM**: prompts and captions from any local or remote LLM.
- **Free GPU**: one click in the top bar unloads all models.

## What it's for

Edit loops, folder captioning, frame-by-frame movie work. Fewer nodes, fewer steps - simpler life.

## Install

ComfyUI Manager → search **MM Nodepack**, or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/metamountain/mm-nodepack-comfyui
```

No extra dependencies.

## License

MIT
