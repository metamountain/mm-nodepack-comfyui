# Upscale Crop MM (`MMUpscaleCrop`)

Resize an image and, if you want, crop it to a format — in one node, image in, image out. Ask for the size as a percentage, a factor, a side, a megapixel count or an exact width × height box; all sizes work downwards as well as up, and an upscaler model can replace the plain resampler.

## How to use

1. Connect your image.
2. Pick `target_mode` and set the size — e.g. `shortest_side` = 1024, or `custom` = 1920 × 1080 for that exact output size.
3. Choose how to resample: `lanczos` (default) or `upscaler model` with a model from your `models/upscale_models` folder.
4. Turn on `crop` to cut a format (4:5, 1:1, 16:9, …) or your own box out of the result; `crop_orientation`, `crop_position` and the offsets decide what survives.
5. Switch `snap` on when you need the result as a multiple of a number (8 suits SD, SDXL and Flux).
6. Queue the workflow. The preview shows the crop window (drag it to place it), and the `info` output tells you exactly what ran.

## Inputs / Outputs

- **Inputs:** `image`, `target_mode` + the matching size fields, `resize_condition`, `method`, `upscale_model`, `snap`, `multiple_of`, `crop` + the crop fields (`crop_ratio`, `crop_width/height`, `crop_orientation`, `crop_position`, `crop_offset_x/y`, `crop_seed`)
- **Outputs:** `image`, `width`, `height` (final px), `info` (what actually ran)

## Tips

- Only the widgets that currently apply are shown; a hidden widget keeps its value, so switching `target_mode` loses nothing.
- A crop box bigger than the image shrinks proportionally to the largest window of that shape that fits.
- The upscaler model runs one pass at its native scale, then a single resize lands the target; tiling halves itself if GPU memory runs short and falls back to CPU.
- RGBA (alpha) images keep their transparency automatically.
- `downscale_only` / `upscale_only` in `resize_condition` leave the image untouched when the target would not really change the size.
