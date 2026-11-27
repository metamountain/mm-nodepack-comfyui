# MM Nodepack — project guide

ComfyUI custom node pack (category **"MM Nodepack"**, all node IDs start with `MM`):
load an image with its path, preview/compare/save creations next to the source,
upscale-crop, and call vision/text LLMs. Built on the **ComfyUI V3 node API**
(`comfy_api.latest`) and the new frontend. Version 0.2.x.

Before modifying V3 node code or frontend code, look the API up in the
`comfyui-sdk` MCP tools (`search_comfy` / `get_comfy_file` / `get_comfy_node`) —
the V3 schema and the new frontend internals are moving targets and local
knowledge goes stale.

## Repository structure

- `__init__.py` — V3 `ComfyExtension` entry point (`comfy_entrypoint`), `WEB_DIRECTORY = "./web"`.
  Collects `NODE_LIST` + `register_routes()` from all `py/` modules. Skips all
  of this under `pytest`.
- `py/`
  - `common.py` — shared building blocks: `IMAGE_EXTS`, `_clean` (strip + remove
    matching quotes), `_FMT_EXT`, `_pick_free` (never-overwrite numbering),
    `LRUCache`, lazy `get_folder_paths()`, `is_local_request`, `json_error`,
    image load/save core (`load_image_file`, `save_tensor`, `png_text_chunks`,
    `view_entry`, `preview_view_entry`).
  - `paths.py` — origin lookup for uploads, unified target resolution
    (`next_target`, `resolve_targets`, `resolve_save_target`), folder batch
    picking (`resolve_folder_image`, `list_folder_images`), last-dialog folder
    memory: `load_last_dir(mode)` / `save_last_dir(path, mode)` keep one
    `{"open", "save", "folder"}` dict per dialog mode in
    `mm_nodepack_last_dir.json` (a legacy `{"dir": ...}` file is read with a
    fallback and migrated on the next save).
  - `dialog.py` — OS file/folder dialogs on the ComfyUI computer: Windows runs
    the dialog in a child process (crash isolation), macOS via `osascript`,
    Linux via zenity/kdialog.
  - `image_io.py` — `MMLoadImage` + `MMPreviewSave` and the `/mm/image/*` routes.
  - `upscale_crop.py` — `MMUpscaleCrop` and `/mm/upscale/preview`.
  - `llm_multitool.py` — `MMLLMMultitool` and `/mm/llm/unload`.
- `web/`
  - `mm_nodepack.js` — Load Image MM / Preview & Save MM behavior: visible mode
    combo (single/folder), 1-click OS dialogs ("choose file" / "choose folder"
    buttons, browser upload only as a remote-browser fallback), thumbnail
    refresh, `sizeLoaderNode` (never crushes the loader height), save buttons,
    smart `auto_send` (multi-loader picker + folder-mode safeguard), loader
    resolution.
  - `mm_upscale_crop.js` / `mm_llm_multitool.js` / `mm_free_gpu.js` — the other
    nodes + the action-bar GPU button.
  - `mm_logo.js` — the ONE place that attaches the titlebar logo to every `MM*`
    node (via `attachMMLogo` from `lib/mm_ui.mjs`); other web files must not
    call `attachMMLogo` themselves.
  - `web/docs/*.md` — in-app node documentation pages.
  - `lib/mm_ui.mjs` — shared frontend helpers: `MIN_PREVIEW_HEIGHT` (`220`, matching ComfyUI core `LoadImage` / `PreviewImage`), `MM_THEME`, `viewUrl/viewBlob/loadImage` (the single `/view` URL builder), `toast`, `debounce`, `setWidgetObjVisible` (works for schema widgets AND runtime buttons; `setWidgetVisible` finds by name and delegates to it), `setWidgetDim`, `fitNodeSize`, `attachMMLogo`.
  - `lib/comparer.mjs` — the compare slider widget for Preview & Save MM
    (rgthree-derived, MIT). `.mjs` so ComfyUI does not load it as an extension.
- `tests/` — `test_mm_nodepack.py`, `test_upscale_crop.py`, `test_llm_multitool.py`
  plus `conftest_helpers.py` (`FolderPathsStub` fakes `folder_paths` with real
  directories). Unit tests must run without a ComfyUI runtime (pure helpers)
  or skip cleanly when `comfy_api` is missing. `live_debug_comfyui.py` is the
  headless E2E suite (own ComfyUI CPU instance + Playwright, run standalone,
  not part of `pytest`).
- `assets/logo.svg`, `example_workflows/`, `README.md`.

## Node contracts

- **`MMLoadImage` ("Load Image MM")** — all 6 inputs are in `required`, ordered
  `mode` (`single`/`folder`), `image` (upload combo from `ComfyUI/input`,
  options `_input_images() or [""]`), `path` (full file path, wins over
  `image`), `folder`, `batch_mode` (incremental/random), `index` (1-based /
  seed, `control_after_generate`); outputs: `image`, `mask`, `width`,
  `height`, `path` (resolved absolute path), `image_list` (`is_output_list`,
  **always natural filename order** — also in `random` mode, so
  `image_list[index]` always reaches the file the direct load produced),
  `index`. `validate_inputs` checks the folder only in folder mode; in single
  mode `path` wins over `image` (missing `path` file → error; empty `path` +
  empty/unknown `image` → error). `hidden: unique_id, prompt` feed the
  wireless source lookup and `_LAST_RESOLVED` (subgraph-prefixed IDs like
  `12:5` fall back to the bare ID).
- **`MMPreviewSave` ("Preview & Save MM")** — input `images` (IMAGE);
  `auto_save` (every run saves all), `auto_send` (saves `#` into the loader
  after each run — but never into an ambiguous one: several loaders → skip +
  toast "pick one in source", loader in folder mode → skip + toast),
  `source` (`auto` or `#nodeId`), `suffix` (`#` = next free number), `save_as`
  (auto save only), `format`, `quality`. No `embed_workflow` widget: PNG
  outputs **always** carry the embedded ComfyUI workflow metadata
  (`png_metadata(prompt, extra_pnginfo)`). Output `saved_paths`. UI sends
  `b_images`/`a_images` (never `images`, so the core preview does not paint
  over the comparer), `source`, `saved`, `source_id`, `note`. Saves never
  overwrite; with no loader the target falls back to `ComfyUI/output/
  preview_#.png` (or `save_as`).
- **`MMUpscaleCrop` ("Upscale Crop MM")** - image-only upscale (percent/
  factor/shortest/longest/megapixels/custom box) + exact crop: input `image`,
  outputs `image`, `width`, `height`, `info`. `method` defaults to `lanczos`
  (switch to `upscaler model` to pick from `models/upscale_models`); `snap`
  rounds to `multiple_of`; optional `crop` cuts `crop_ratio` (`custom` uses
  the dedicated `crop_width`/`crop_height` box) with `crop_orientation`
  (`auto` follows the frame), `crop_position` + offsets, `crop_seed` for
  `random`, and a live preview panel (`/mm/upscale/preview`) where the crop
  window is draggable. Only the widgets that currently apply are shown;
  legacy positional workflows are remapped by name on load.
- **`MMLLMMultitool` ("LLM Multitool MM")** — vision/text LLM calls against
  llama.cpp, Ollama, Unsloth Studio, OpenAI-compatible servers; optional own
  `llama-server` (env `LLM_MULTITOOL_PORT`, default 8081).

## HTTP routes (all answer `{"ok": ...}` JSON)

| Route | Payload (key fields) | Result |
|---|---|---|
| `POST /mm/image/target` | loader body (`mode/path/image/folder/batch_mode/index`) + `suffix`, `save_as`, `format` | next free save target for the loader's current source |
| `POST /mm/image/dialog` | `mode` (`open`/`save`/`folder`), `path` (start hint), `name` (suggested) | OS dialog on the ComfyUI computer; `{"ok", "path", "view", "count"?}`; `{"ok": false, "local": false}` for remote browsers |
| `POST /mm/image/resolve` | `image` (upload name) | absolute path + origin of an upload in `output/` (+ `candidates`) |
| `POST /mm/image/preview` | loader body | live `/view` entry of the loader's current file (+ `path`) |
| `POST /mm/image/save` | preview entry (`type/filename/subfolder`) + `source?`, `suffix`, `save_as`, `format`, `quality` | saves next to the source, or to `save_as` / `output/preview_#.png` when the graph has no loader |
| `POST /mm/upscale/preview` | `node_id` + node settings | live size/crop numbers for the preview panel |
| `POST /mm/llm/unload` | `api_url`, `backend`, `api_key_env` | stops the own local server / frees the API model |

## Non-negotiable invariants (V3 API & frontend)

- **`web/lib/*.mjs` import depth**: files in `web/lib/` must import
  `../../../scripts/app.js` and `../../../scripts/api.js` (the frontend serves
  `/scripts/app.js`; a `../../scripts/...` depth 404s and kills the whole pack,
  since every MM web file imports `lib/mm_ui.mjs`). Files in `web/` use
  `../../scripts/...`.
- **DOM widgets** (`addDOMWidget`): set `serialize = false` AND
  `options = { ..., serialize: false }`, and lock the width with
  `Object.defineProperty(w, "width", { get: () => node.size[0], set() {},
  configurable: true })` so the panel follows the node width.
- **`setWidgetObjVisible` / `setWidgetVisible`** (`lib/mm_ui.mjs`):
  `setWidgetVisible(node, name, visible)` finds the widget by name and
  delegates to `setWidgetObjVisible(node, widget, visible)` — the single
  implementation, which works for schema widgets AND runtime-created buttons
  (the loader's "choose file" / "choose folder" buttons hide with it).
  **Property order is critical in ComfyUI frontend v1.58+!** Hiding must set
  `w.type = "hidden"` **BEFORE** `w.hidden = true` (plus
  `computeSize = () => [0, -4]` and `w.y = -100000`), and showing must
  restore `w.type = w._mmType` **BEFORE** `w.hidden = false`. Setting
  `w.hidden = true` while `type` is still normal latches
  `_rawOptions.hidden = true`, and restoring `w.type` after `w.hidden = false`
  re-hides the widget permanently! Also restore/delete `computeSize` and clamp
  negative `w.y` back to `0`. A widget whose matching input socket is linked
  stays visible.
- **`fitNodeSize`**: must return early for `!node`, and when `!node.graph`
  (calls during `nodeCreated`) queue ONE `requestAnimationFrame` retry guarded
  by `node._mmFitQueued` (latest `shrinkHeight` wins) so the resize happens
  once the node is attached.
- **Loader sizing (`sizeLoaderNode` in `mm_nodepack.js`)**: the Load Image MM
  node must NEVER use `fitNodeSize({ shrinkHeight: true })` — it uses its own
  `sizeLoaderNode(forImage)` with the same `!node.graph` rAF retry (guards
  `_mmSizeQueued` / `_mmSizeForImage`): delegates image sizing to ComfyUI's
  native `setSizeForImage()` (matching core `LoadImage`'s `220 px`
  `MIN_PREVIEW_HEIGHT`, shared with `MMPreviewSave`'s `ComparerWidget`) and
  never shrinks height, so mode switches and user resizing never crush the
  thumbnail.
  Called from every thumbnail `onload` and `loader._mmEnsureSize()` (used by
  "send to loader").
- **Schema stability**: never shift, add, or remove schema inputs/outputs or
  widgets without a workflow migration — `widgets_values` restore is positional.
  Frontend-only buttons/widgets need `serialize: false` + `options.serialize:
  false` (graphToPrompt and the configure loop both check `options.serialize`).
- **`indexOf !== -1` before `splice`**: always guard
  `const i = node.widgets.indexOf(w); if (i !== -1) node.widgets.splice(i, 1);`
  (removal loops and reorder splices).
- **Node class check**: runtime-created nodes can lack `comfyClass` (only
  `.type` is set) — always use `n.comfyClass || n.type`
  (`nodeClass()` in `mm_nodepack.js`).
- **`py/dialog.py` Windows vtable**: the `IFileOpenDialog` vtable indices
  (`Release=2, Show=3, SetOptions=9, GetOptions=10, SetFolder=12, SetTitle=17,
  GetResult=20, IShellItem::GetDisplayName=5`) are verified against the Windows
  SDK `ShObjIdl_core.h` (`IUnknown` 0..2, `IModalWindow` 3, `IFileDialog`
  4..26). Do not "correct" them.
- **`comdlg32` `lpstrFile`**: when `name` is empty the buffer must stay `""` —
  only join `start_dir` into `first` when `first` is non-empty. A bare
  `"D:\\dir\\"` makes `GetOpenFileNameW`/`GetSaveFileNameW` fail with
  `FNERR_INVALIDFILENAME`.
- **`/mm/image/save` without a loader**: `source` may be empty (txt2img
  preview) — resolve via `resolve_source_like`, else `resolve_save_target("",
  ...)` (absolute `save_as` as-is, else `output/preview_#.png`). Never fall
  back to a hardcoded `output/preview.png` (404 on every save).
- **`/mm/image/dialog` thumbnails** use `preview_view_entry` (files outside
  `input/`/`output/` get a cached temp-folder thumbnail), not `view_entry`.
- **Route registration** logs through
  `logging.getLogger(__name__).warning(...)` (never bare `print`).
- Unit tests: `folder_paths` is imported lazily (`get_folder_paths()`);
  anything that needs ComfyUI's runtime must guard against its absence.

## Verification (run all before calling a change done)

```powershell
# 1. syntax check all 7 JS/MJS files
node --input-type=module --check web/mm_nodepack.js
node --input-type=module --check web/mm_logo.js
node --input-type=module --check web/mm_upscale_crop.js
node --input-type=module --check web/mm_llm_multitool.js
node --input-type=module --check web/mm_free_gpu.js
node --input-type=module --check web/lib/mm_ui.mjs
node --input-type=module --check web/lib/comparer.mjs

# 2. full test suite (CPU only)
$env:CUDA_VISIBLE_DEVICES='-1'
D:\ComfyUI_windows_portable\python_embeded\python.exe -B -m pytest -p no:cacheprovider tests/ -q

# 3. ComfyUI entrypoint & schema import smoke check (repo root contains a dash,
# so load __init__.py as a package via importlib instead of a plain import):
D:\ComfyUI_windows_portable\python_embeded\python.exe -B -c "import sys, asyncio, importlib.machinery, importlib.util; repo = r'D:\Coding\Comfyui\mm-nodepack-comfyui'; sys.path.insert(0, r'D:\ComfyUI_windows_portable\ComfyUI'); loader = importlib.machinery.SourceFileLoader('mmnodepack', repo + r'\__init__.py'); spec = importlib.util.spec_from_loader('mmnodepack', loader); spec.submodule_search_locations = [repo]; mod = importlib.util.module_from_spec(spec); sys.modules['mmnodepack'] = mod; loader.exec_module(mod); ext = asyncio.run(mod.comfy_entrypoint()); nodes = asyncio.run(ext.get_node_list()); print('nodes:', len(nodes)); [n.define_schema() for n in nodes]; print('schemas OK')"

# 4. Live headless CPU ComfyUI + Playwright E2E debug suite (zero GPU VRAM, tests real routes/prompts/UI):
D:\Coding\MCP\rags\.venv\Scripts\python.exe tests/live_debug_comfyui.py
```

Run all commands from the repo root so `tests/` and the repo paths resolve. Code, comments, and this file stay **English only**; task
reports go to `tmp/NN-bericht.md` (German allowed there).
