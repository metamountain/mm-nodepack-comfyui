# Preview & Save MM (`MMPreviewSave`)

Replaces **Preview Image**, **Save Image**, and an **Image Comparer** in one node.

## Built-In Compare

Toggle the compare mode directly below the preview (saved per workflow):
- **`compare hold`** *(default)* — Press and hold on the preview to view the original from **Load Image MM**.
- **`compare slide`** — Move/drag the divider across the image to compare original vs. result.
- **`compare off`** — Plain preview.

Right-click the preview for quick **Open / Copy** actions on both the creation and the original.

## Saving & Iteration Buttons

- **`save`** — Writes the selected image next to the source file as `<name>_1`, `<name>_2`, … without ever overwriting existing files (including across folder batch runs). With no loader in the graph, saves to `ComfyUI/output/preview_#.png`.
- **`save as…`** — Opens the native OS save dialog pre-filled in the target folder (remembers its own last save folder separately from Load Image MM; uses the browser save picker remotely, or a filename prompt over plain HTTP).
- **`send to loader`** — Saves as `_#` and loads the result straight back into **Load Image MM** (automatically switches the loader to single mode so the next run edits the new image).

The preview panel matches ComfyUI's standard `LoadImage` / `PreviewImage` / **Load Image MM** minimum preview height (`220 px`) across all aspect ratios — staying compact on portrait and landscape images alike without shrinking user-set heights. PNG outputs always embed ComfyUI workflow metadata (prompt + `extra_pnginfo`) — no toggle needed.

## Auto Switches

| Widget | Description |
|---|---|
| `auto_save` | Automatically saves every generated image on run. |
| `auto_send` | Automatically feeds every generated image back into **Load Image MM** (reuses the auto-saved file when `auto_save` is on). |
| `suffix` | Filename suffix pattern (default `_#`, where `#` increments to the next free number). |
| `save_as` | Optional custom filename or relative/absolute target path. |
| `format` | `same as source`, `png`, `jpg`, or `webp`. |
| `quality` | JPEG / WebP quality (`1–100`). |
| `source` | `auto` finds the upstream (or only) **Load Image MM** wirelessly. With multiple loaders, enter its node number (e.g. `#12`) or let the node ask you to pick one when you turn `auto_send` on. |

Manual buttons already handled by an active auto switch are dimmed automatically.

### Safe auto-send

`auto_send` never overwrites an ambiguous loader silently:

- **several loaders** — a toast tells you to pick one in `source` (a small picker menu offers them when you enable the switch);
- **loader in folder mode** — auto-send is skipped with a toast; use `send to loader`, which switches the loader to single mode first.
