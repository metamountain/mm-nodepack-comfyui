# Preview Image MM (`MMPreviewImage`)

Replaces **Preview Image**, **Save Image**, and an **Image Comparer** in one node.

## Built-In Compare

Toggle the compare mode directly below the preview (saved per workflow):
- **`compare hold`** *(default)* — Press and hold on the preview to view the original from **Load Image MM**.
- **`compare slide`** — Move/drag the divider across the image to compare original vs. result.
- **`compare off`** — Plain preview.

Right-click the preview for quick **Open / Copy** actions on both the creation and the original.

## Saving & Iteration Buttons

- **`save`** — Writes the selected image next to the source file as `<name>_1`, `<name>_2`, … without ever overwriting existing files (including across folder batch runs).
- **`save as…`** — Opens the native OS save dialog pre-filled in the target folder (uses the browser save picker when accessed remotely).
- **`send to loader`** — Saves as `_#` and loads the result straight back into **Load Image MM** (automatically switches the loader to single-image mode so the next run edits the new image).

## Auto Switches

| Widget | Description |
|---|---|
| `auto_save` | Automatically saves every generated image on run. |
| `auto_send` | Automatically feeds every generated image back into **Load Image MM** (reuses the auto-saved file when `auto_save` is on). |
| `suffix` | Filename suffix pattern (default `_#`, where `#` increments to the next free number). |
| `save_as` | Optional custom filename or relative/absolute target path. |
| `format` | `same as source`, `png`, `jpg`, or `webp`. |
| `quality` | JPEG / WebP quality (`1–100`). |
| `embed_workflow` | Embeds ComfyUI prompt/workflow metadata in PNG outputs. |
| `source` | `auto` finds the upstream (or only) **Load Image MM** wirelessly. With multiple loaders, enter its node number (e.g. `#12`). |

Manual buttons already handled by an active auto switch are dimmed automatically.
