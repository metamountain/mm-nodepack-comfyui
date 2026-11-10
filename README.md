# comfyui-mm-nodes

Load an image **with its path**, edit it, and save the result **next to the source** –
`haus.png` → `haus_1.png`, `haus_2.png`, … – or anywhere via "save as". A preview node
shows the variants with a compare slider (original vs. result) and saves single images
per button, without a new run.

Category: `metamountain/image`.

## Nodes

| Node | What it does |
|---|---|
| **Load Image (Path)** | Loads an image by upload (ComfyUI/input) or full file path (`path` wins). Outputs `source` (sync connection: path, original image, mask, size, format), `image`, `mask`, `path` (resolved absolute path). |
| **Preview Image MM (auto_save)** | Saves every image of the batch next to the source: `stem + suffix + ext`. `suffix` default `_#` (`#` = next free number: `haus_1`, `haus_2`, …). `save_as` (empty = next to source) accepts a name or a full path. Format: same as source / png / jpg / webp. Compares the result with the original (slide on the image). |
| **Preview Image MM** | Previews the variants (slide compare with the original) and offers `💾 save _#` / `💾 save as…` buttons for the selected variant (B1, B2, … bar). Saving goes through the server, no new run needed. |

## Naming rules

- `save_as` empty → `<source folder>/<stem><suffix><ext>`, e.g. `haus` + `_#` → `haus_1.png`.
- `#` in the suffix or in `save_as` → smallest free number (≥ 1).
- No `#` and the file already exists → `_2`, `_3`, … before the extension. **Never overwrites.**
- `save_as` with a full path is used as-is (relative names resolve against the source folder);
  a missing extension gets the target format appended.
- No `source` connection → the output directory, stem `preview`.

## Example

```
Load Image (Path) [path: D:\photos\haus.png]
  └─ source ──▶ Preview Image MM (auto_save) [suffix: _#]   → D:\photos\haus_1.png
  └─ image ──▶ (your edit nodes) ──▶ images
```

## Routes

- `POST /mm-nodes/target` `{path?, image?, suffix, save_as, format}` → `{ok, target}` (compute only).
- `POST /mm-nodes/save` `{filename, subfolder, type, source, suffix, save_as, format, quality}` →
  `{ok, path}` – re-saves a preview image (temp/output) next to the source.

## Tests

```
D:\ComfyUI_windows_portable\python_embeded\python.exe tests\test_mm_nodes.py
```
