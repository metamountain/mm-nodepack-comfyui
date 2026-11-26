# Upscale Crop MM (`MMUpscaleCrop`)

Two stages in one node — **UPSCALE** (always keeps aspect ratio, except `custom` box) followed by optional **CROP** (only ever cuts, never resamples twice). Handles both `image` and `samples` (latent) in normalized coordinates.

## Target Modes (`target_mode`)

| Mode | Behavior |
|---|---|
| `scale_factor` | Multiply dimensions by `scale_factor` (`0.05–8.0`). Values `< 1.0` downscale. |
| `scale_percent` | Scale by percentage (`1–800%`). |
| `shortest_side` | Pin `min(width, height)` in pixels. |
| `longest_side` | Pin `max(width, height)` in pixels. |
| `megapixels` | Match total pixel count (`0.05–64 MP`). |
| `custom` | Cover `target_width × target_height` and crop cleanly to that exact resolution. |

## Built-In Upscale Model & Alpha Preservation

- **`method = upscaler model`** reads directly from `models/upscale_models` (no separate loader node). Common workhorses (UltraSharp, Remacri, Siax, RealESRGAN) rank first automatically.
- **OOM-safe tiling:** Runs tiled inference (`512`, overlap `32`) and halves the tile size down to `128` on out-of-memory errors.
- **Automatic alpha preservation:** 4-channel RGBA images automatically split RGB for the model, resize alpha bilinearly, and rejoin — zero UI switches needed.
- **Smart downscale skip:** When the target is smaller than the source, the upscale model is skipped automatically in favor of `lanczos`.

## Failsafe Conditional Resize (`resize_condition`)

Available on absolute target modes (`shortest_side`, `longest_side`, `megapixels`, `custom`):
- **`always`** *(default)* — Always resize to target.
- **`downscale_only`** — Leave the image untouched (skip resize & model) if the target is not smaller than the source.
- **`upscale_only`** — Leave the image untouched (skip resize & model) if the target is not larger than the source.

## Crop & Live Preview

- **Formats:** `4:5`, `1:1`, `4:3`, `3:2`, `DIN`, `16:10`, `16:9`, `2:1`, `21:9`, or `custom`.
- **Orientation:** `auto` (follows each image's portrait/landscape framing), `portrait`, or `landscape`.
- **9 Positions + Random:** `center`, `top`, `bottom`, `left`, `right`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, and seeded `random`.
- **Draggable Preview:** Drag the highlighted crop window directly in the live preview panel to adjust `crop_offset_x` / `crop_offset_y` (double-click to reset).

## Outputs

`image`, `latent`, `width`, `height`, `info` (one-line summary of sizes, method, crop, and any skipped resize or fallback).
