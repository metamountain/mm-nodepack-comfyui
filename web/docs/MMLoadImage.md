# Load Image MM (`MMLoadImage`)

Unified image loader replacing both **Load Image** and **Load Image Batch** in a single node.

## Mode Switcher

Two mode buttons sit at the top of the node — the active one is bright, the other dimmed. Click either button at any time to switch modes:

### 1. `load image` (Single Image Mode)
- Pick a file via the browser picker (uploads to `ComfyUI/input`; if an identical original exists in `ComfyUI/output`, `path` resolves back to that original).
- Or type / paste any full file path into **`path`** (`path` always takes priority over the upload combo).

### 2. `choose folder` (Folder Batch Mode)
- Opens the native OS folder picker and loads images directly from that folder in natural filename order (`img_2.png` before `img_10.png`).
- **`batch_mode` (`incremental` / `random`):**
  - **`incremental`:** Steps through the folder using the 1-based **`index`** (`1` = first image; indices past the end wrap cleanly back to `1`).
  - **`random`:** Uses **`index`** as a deterministic seed to pick an image from the folder.
- Works with ComfyUI's `control_after_generate` (`increment` / `randomize` / `fixed`) and Queue Batch Count.
- Changing the folder, mode, or index updates the thumbnail preview immediately.

## Outputs

| Output | Type | Description |
|---|---|---|
| `image` | `IMAGE` | Loaded image tensor (`[N, H, W, 3]`, EXIF-transposed). |
| `mask` | `MASK` | Inverted alpha mask (`1 - alpha`, or zero mask if no alpha). |
| `width` | `INT` | Image width in pixels. |
| `height` | `INT` | Image height in pixels. |
| `path` | `STRING` | Resolved absolute file path on disk. |
| `image list` | `IMAGE` *(list)* | All images in the folder as a ComfyUI list (every image keeps its own resolution; zero decode overhead when unconnected). |
| `index` | `INT` | 1-based index of the loaded image in the folder (`1` in single mode). |

## Wireless Sync

**Preview Image MM** detects **Load Image MM** automatically (`source = auto`) — no extra path wire required.
