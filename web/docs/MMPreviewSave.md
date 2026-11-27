# Preview & Save MM (`MMPreviewSave`)

See the results, compare them with the source image and save the one you keep. One node replaces preview, compare and save: hold or slide over the preview to compare the creation with the original from Load Image MM. Saves never overwrite an existing file.

## How to use

1. Connect your images; the node finds Load Image MM automatically as the source.
2. Queue the workflow; the preview shows the results.
3. Switch the compare mode below the preview: hold, slider or off.
4. Select the creation you keep, then use **save** (next to the source, next free number), **save as…** (choose folder and name) or **send to loader** (saves and loads the result into Load Image MM for the next run).
5. Or turn on `auto save` to save every result after each run, and `auto send` to feed them back into the loader automatically.

## Inputs / Outputs

- **Inputs:** `images`, `auto save`, `auto send`, `source`, `suffix`, `save as`, `format`, `quality`
- **Output:** `saved` — one path per saved file (auto save)

## Tips

- Saves next to the source are numbered `name_1`, `name_2`, … so existing files are never touched; with no loader the files go to the `output` folder instead.
- `auto send` is skipped (with a toast) when several loaders exist or the loader is in folder mode — use **send to loader**, which switches the loader to single mode first. With `auto save` on, the auto-saved file is the one that gets sent.
- PNG saves always embed the workflow metadata (prompt and settings).
- Right-click the preview for quick open / copy actions on both the creation and the original.
- Buttons that an active auto switch already handles are dimmed.
