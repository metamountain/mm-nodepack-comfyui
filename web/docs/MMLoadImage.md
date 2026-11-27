# Load Image MM (`MMLoadImage`)

Unified image loader replacing both **Load Image** and **Load Image Batch** in a single node.

## Mode Switcher

A visible **`mode`** selector sits at the top of the node (`single` / `folder`). Only the controls for the active mode are shown, and the thumbnail follows ComfyUI's standard `LoadImage` preview convention (`220 px` minimum preview height, matching **Preview & Save MM**; mode switches never crush the thumbnail).

### 1. `single` (single image)
- **`choose file`** (right below `image`) — one click opens the native OS file dialog on the ComfyUI computer
  (real server paths, including folders outside the ComfyUI tree via temp-cached previews; remembers its own last open folder separately from `save as…` and `choose folder`).
  A browser on another machine falls back to browser upload (files land in `ComfyUI/input`;
  if an identical original exists in `ComfyUI/output`, `path` resolves back to that original).
- Drag-and-drop, `Ctrl+V` paste, picking from `image`, or typing/pasting any full file path into **`path`** (`path` always takes priority over `image`) all refresh the thumbnail immediately.
- Cancelling a dialog changes nothing.

### 2. `folder` (folder batch)
- **`choose folder`** (right below `folder`) — one click opens the native OS folder picker (with its own remembered folder) and loads images directly
  from that folder in natural filename order (`img_2.png` before `img_10.png`).
  Cancelling a dialog changes nothing.
- **`batch_mode` (`incremental` / `random`):**
  - **`incremental`:** Steps through the folder using the 1-based **`index`** (`1` = first image; indices past the end wrap cleanly back to `1`).
  - **`random`:** Uses **`index`** as a deterministic seed to pick an image from the folder.
- Works with ComfyUI's `control_after_generate` (`increment` / `randomize` / `fixed`) and Queue Batch Count.
  In single mode the index control is parked on `fixed` and restored (with your previous value) when you switch back.
- Changing the folder, mode, or index updates the thumbnail preview immediately.

## Outputs

| Output | Type | Description |
|---|---|---|
| `image` | `IMAGE` | Loaded image tensor (`[N, H, W, 3]`, EXIF-transposed). |
| `mask` | `MASK` | Inverted alpha mask (`1 - alpha`, or zero mask if no alpha). |
| `width` | `INT` | Image width in pixels. |
| `height` | `INT` | Image height in pixels. |
| `path` | `STRING` | Resolved absolute file path on disk. |
| `image list` | `IMAGE` *(list)* | All images in the folder as a ComfyUI list, **always in natural filename order** (also in `random` mode). Zero-based access is `image_list[index - 1]` (the `index` output, 1-based) — it always reaches exactly the same file as the direct load. Every image keeps its own resolution; zero decode overhead when unconnected. |
| `index` | `INT` | 1-based index of the loaded image in the folder (`1` in single mode). |

## Wireless Sync

**Preview & Save MM** detects **Load Image MM** automatically (`source = auto`) — no extra path wire required.
