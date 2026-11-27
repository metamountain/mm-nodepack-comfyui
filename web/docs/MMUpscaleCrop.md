# Upscale Crop MM (`MMUpscaleCrop`)

Two stages in one node, **image in / image out** — no latent involved.
**UPSCALE** sets the size (always keeps the aspect ratio, except the `custom`
box), then optional **CROP** cuts a format out of the result — it only ever
cuts, never rescales, so nothing is resampled twice.

Only the widgets that currently apply are shown: `target_mode` selects the
size field, `method` decides whether the model picker exists, and `crop`
decides whether the second block is there at all. A hidden widget keeps its
value, so nothing is lost when you switch modes.

## Inputs & Outputs

- **Input:** `image` (the image to resize and optionally crop).
- **Outputs:** `image`, `width`, `height` (final px), and `info` — a one-line
  summary of what actually ran: source → final size, the target mode, the
  method used, the snap state, the crop applied, and any skipped resize or
  fallback.

## Target Modes (`target_mode`)

| Mode | Behavior |
|---|---|
| `scale_factor` *(default)* | Multiply dimensions by `scale_factor` (0.05–8.0, default `2.0`). Values `< 1.0` downscale. |
| `scale_percent` | Scale by percentage (1–800%). |
| `shortest_side` | Pin `min(width, height)` in pixels. |
| `longest_side` | Pin `max(width, height)` in pixels. |
| `megapixels` | Match total pixel count (0.05–64 MP). |
| `custom` | Cover `target_width × target_height` and crop to it — the resolution you type is the resolution that comes out. The crop is mandatory here and the crop toggle / shape picker fold away; the framing controls (position, offsets) still apply. |

All six work downwards as well as up, so this node is also your downscaler. Newly placed nodes silently remember your last-chosen `target_mode` and sizing/custom `W × H` in `localStorage`.

## Resampling (`method`)

`lanczos` is the default — the sharpest plain kernel. Other plain kernels:
`bicubic`, `bilinear`, `nearest`, and `area` (the best choice for heavy
downscaling).

### Built-in upscale model

Switch `method` to **`upscaler model`** and the `upscale_model` picker
appears. It reads directly from `models/upscale_models` (no separate loader
node). Common general-purpose picks (UltraSharp, Remacri, Siax, RealESRGAN)
rank first automatically and become the default when installed.

- **One pass, then trim:** the model runs once at its native scale, then a
  single resize lands the exact target — a 4× model hitting a 2× target does
  not leave a needlessly huge intermediate.
- **OOM-safe tiling + automatic GPU → CPU fallback:** tiled inference (tile `512`, overlap `32`) halves tiles (`512 → 256 → 128`) on GPU OOM and automatically falls back to CPU (`CPU fallback (GPU VRAM full)` in `info`) if VRAM is still full. Batches stream per image so peak memory stays low.
- **Live RAM-OOM guard:** estimates output allocation against live free system RAM — amber warning above 35% free RAM (or side `> 8192 px`), red warning + pre-allocation block above 85% free RAM so ComfyUI never hard-crashes from system OOM.
- **Automatic alpha preservation:** 4-channel RGBA images split RGB for the
  model, resize the alpha bilinearly, and rejoin — no UI switches.
- **Fallbacks (named in `info`):** target smaller than the source → `lanczos`
  (no model on a downscale); no model selected or installed → `lanczos`;
  the model throws → `lanczos`.

## Exact Size vs Snapping (`snap`, `multiple_of`)

- **Snap off (default):** you get exactly the size you asked for.
- **Snap on:** the result is rounded to a multiple of `multiple_of`
  (`8` suits SD, SDXL and Flux VAEs; `64` for some tiling and video
  workflows). ComfyUI's VAE Encode trims a few pixels itself for non-multiple
  sizes, so snapping is optional.

Old workflows that saved `multiple_of` as a string (`"8"`, `"off"`) are
repaired automatically on load.

## Failsafe Conditional Resize (`resize_condition`)

Available on the absolute target modes (`shortest_side`, `longest_side`,
`megapixels`, `custom`); the two relative modes always resize, so the widget
is hidden for them.

- **`always`** *(default)* — always resize to the target.
- **`downscale_only`** — leave the image untouched (resize and model skipped
  together) if the target is not smaller than the source.
- **`upscale_only`** — leave the image untouched if the target is not larger
  than the source.

In `custom` mode a box the source cannot fill is clamped to the source, so
the crop keeps the source intact instead of upscaling to the box.

## Crop (`crop` + the crop widgets)

Off by default — off means the node is a pure upscaler. On, it cuts one of:

- **`crop_ratio`** — a classic format: `4:5`, `1:1`, `4:3`, `3:2`, `DIN`,
  `16:10`, `16:9`, `2:1`, `21:9`. Always the largest window of that shape
  that fits.
- **`crop_ratio = custom`** — the dedicated `crop_width × crop_height` box
  (defaults 1024×1024). This is independent of the `target_width/height`
  box; when left empty it falls back to that box.

### Orientation (`crop_orientation`)

`auto` *(default)* follows the source frame: a portrait shot gets a portrait
crop, a landscape one a landscape crop — the right choice for a mixed batch.
`portrait` / `landscape` force it. Ignored for `1:1` (there is no "way
round") and for `custom` (the box is the shape). A square source has no
orientation to take, so `auto` leaves the format as written (4:5 stays
portrait, 16:9 stays landscape).

### Position, offsets & seed

- **`crop_position`** — `center`, `top`, `bottom`, `left`, `right`,
  `top-left`, `top-right`, `bottom-left`, `bottom-right`, or `random`.
- **`crop_offset_x` / `crop_offset_y`** — nudge the window in px from its
  anchor (positive = right / down), clamped so it can never leave the image.
  Hidden and ignored when `crop_position = random`. Changing the position
  interactively resets both to zero (a new anchor makes old offsets
  meaningless).
- **`crop_seed`** — only for `random` placement: the same seed always gives
  the same window, so a framing you liked is reproducible. Its
  `control_after_generate` companion (`fixed` / `increment` / `randomize`)
  is parked on `fixed` while the widget is hidden, so a saved non-fixed mode
  never fires on runs that do not use it.

## Live Preview

One cached thumbnail of the last run with the discarded area dimmed and the
crop window drawn over it — the backend returns the numbers computed by the
same functions the node runs, so the frame is exactly what the graph will
cut (the image is never re-rendered per keystroke).

- **Drag** the highlighted window to place the crop — the drag writes back
  into `crop_offset_x` / `crop_offset_y`, so the widgets remain the single
  source of truth and a dragged crop is reproducible from the saved graph.
  Dragging a `random` window takes manual control (it becomes a `center`
  anchor at the drop point).
- **Double-click** the window to reset the offsets to the anchor.
- The status line reads e.g. `1200×1600 → 2176×2176 | lanczos | /64 | 1:1
  center → 2176×2176 | drag to place` (source → upscaled size, method,
  snap, crop, notes). Before the graph has run once it says so.

## Migration

Workflows saved with the old positional layout (image + latent, shared
target/crop box, string `multiple_of`) are remapped by widget name on load:
the phantom image value is cleared, `crop_width`/`crop_height` start clean
and inherit the old shared box when the crop shape was `custom`.
