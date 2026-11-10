# MM Nodepack

Two small nodes for editing images in ComfyUI **where they live**: load an image with its real
file path, compare the result with the original, and save it back next to the source
(`house.png` → `house_1.png`, `house_2.png`, …) – or send it straight back into the loader and keep editing.

![send to loader](assets/send_to_loader.gif)

*Flux.2 Klein edit: "add a cactus" → send → "make it blue" → send → "pink and white striped" → send.*

## Nodes

### Load Image MM
- **choose file** opens your normal system file dialog (Windows, macOS, Linux with zenity/kdialog).
  The real path shows in the node at once, before you run anything. No copy in `ComfyUI/input`.
- `path` – full file path; wins over the upload list. You can also paste a path.
- Without a desktop (server, container) or from another computer, **choose file** is the normal upload.
- Outputs: `sync` (one link to Preview Image MM: path, original, mask, size, format), `image`, `mask`,
  `width`, `height`, `path`.

### Preview Image MM
- Shows the result with a compare widget under the image: **compare slide** (move the mouse),
  **compare hold** (press to see the original), **compare off**.
- **save** – next free `_#` name next to the source, never overwrites.
- **save as…** – your system "save as" dialog, starting in the source folder.
- **send to loader** – saves as `_#` and puts it into the Load Image MM, so the next run edits the new image.
- `auto_save` on: every run saves all images. `suffix` (default `_#`, `#` = next free number),
  `save_as` (name or full path), `format` (same as source / png / jpg / webp), `quality`, `embed_workflow`.
- Without a `sync` link, `source = auto` uses the Load Image MM upstream.

## Install

ComfyUI Manager / Comfy Registry: search **MM Nodepack**. Or:

```
cd ComfyUI/custom_nodes
git clone https://github.com/metamountain/mm-nodepack-comfyui
```

No extra Python packages.

## Notes

- Images pasted with Ctrl+V have no home folder: **save** then asks where to save.
- Opened from another computer (`--listen`), saving is limited to the ComfyUI output folder.

## License

MIT
