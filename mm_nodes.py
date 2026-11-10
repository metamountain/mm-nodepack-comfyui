"""comfyui-mm-nodes: load an image together with its path, edit it, and save the
result next to the source (haus_1.png, haus_2.png, ... or "save as").

V3 node API (comfy_api.latest). The pure helpers (next_target, resolve_source,
load_image_file, save_tensor, png_metadata, make_source) are unit-testable without
ComfyUI: folder_paths is only imported inside functions.
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, PngImagePlugin

from comfy_api.latest import io, ui

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CATEGORY = "metamountain/image"

# The sync connection: one link carries everything the saver needs (NACHTRAG B):
# {"path": abs. path, "image": IMAGE tensor (original), "mask": MASK,
#  "width": int, "height": int, "format": source extension, "name": stem}
SOURCE = io.Custom("MM_SYNC")

_FMT_EXT = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}


def _clean(s: str) -> str:
    """strip + remove surrounding quotes."""
    s = (s or "").strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1]
    return s.strip()


def _ext_for(source: str, fmt: str) -> str:
    """Target extension: 'same as source' keeps the source ext (lower, .jpeg stays .jpeg)
    if it is an image ext, else .png; png|jpg|webp map to .png/.jpg/.webp."""
    if fmt == "same as source":
        ext = os.path.splitext(source)[1].lower()
        return ext if ext in IMAGE_EXTS else ".png"
    return _FMT_EXT.get(fmt, ".png")


def _pick_free(base: str, ext: str, taken=()) -> str:
    """base = path WITHOUT extension, may contain '#'.

    '#' -> smallest n >= 1 so the file does not exist yet (haus_1, haus_2, ...).
    No '#' and the file exists -> _2, _3, ... before the extension. Never overwrites.
    `taken` = paths already reserved in this batch (not on disk yet).
    """
    taken = {os.path.abspath(t) for t in taken}

    def free(cand):
        return not (os.path.exists(cand) or cand in taken)

    if "#" in base:
        n = 1
        while True:
            cand = base.replace("#", str(n)) + ext
            if free(cand):
                return cand
            n += 1
    cand = base + ext
    if free(cand):
        return cand
    n = 2
    while True:
        cand = f"{base}_{n}{ext}"
        if free(cand):
            return cand
        n += 1


def save_dir(src: str) -> str:
    """Folder for saves of `src`: next to the file – except uploads in ComfyUI/input
    (only a copy of the user's file): those go to ComfyUI/output (same subfolder)."""
    d = os.path.dirname(os.path.abspath(src))
    try:
        import folder_paths
        inp = os.path.abspath(folder_paths.get_input_directory())
        if os.path.commonpath([inp, os.path.abspath(src)]) == inp:
            return os.path.normpath(os.path.join(folder_paths.get_output_directory(), os.path.relpath(d, inp)))
    except Exception:  # noqa: BLE001 – no ComfyUI (unit tests) or other drive
        pass
    return d


def next_target(source: str, suffix: str = "_#", save_as: str = "", fmt: str = "same as source") -> str:
    """Absolute target path for the next saved image. Never overwrites.

    save_as empty: <source folder>/<stem><suffix><ext>; '#' in the suffix = smallest
    free number (haus_1.png, then haus_2.png); no '#' and the file exists -> _2, _3 ...
    save_as set: quotes removed; absolute path used as-is, else relative to the source
    folder; without an image extension the target ext is appended; '#' like above;
    if the file exists -> _2, _3 ... (never overwrite).
    """
    src = _clean(source)
    if not src:
        raise ValueError("next_target: empty source path")
    src = os.path.abspath(src)
    if not os.path.isfile(src):
        raise ValueError(f"next_target: source not found: {src}")
    src_dir = save_dir(src)
    stem = os.path.splitext(os.path.basename(src))[0]
    ext = _ext_for(src, fmt)
    sa = _clean(save_as)
    if sa:
        base = os.path.abspath(sa) if os.path.isabs(sa) else os.path.join(src_dir, sa)
        root, bext = os.path.splitext(base)
        # an image extension in save_as wins; without one the target ext is appended
        return os.path.abspath(_pick_free(root, bext.lower() if bext.lower() in IMAGE_EXTS else ext))
    return os.path.abspath(_pick_free(os.path.join(src_dir, stem + (suffix or "")), ext))


def resolve_source(image: str, path: str) -> str:
    """`path` (stripped, quotes removed) wins; else the upload combo via folder_paths.
    The file must exist, else ValueError with the path."""
    p = _clean(path)
    if p:
        p = os.path.abspath(p)
    else:
        import folder_paths
        p = folder_paths.get_annotated_filepath(image)
    if not os.path.isfile(p):
        raise ValueError(f"source file not found: {p}")
    return p


def load_image_file(p: str):
    """PIL branch of core LoadImage (ComfyUI/nodes.py:1775-1806): exif_transpose, RGB,
    alpha -> mask (1 - alpha), multiple frames (animated webp).
    Returns (IMAGE [N,H,W,C] float32 0..1, MASK [N,H,W] float32 0..1)."""
    img = Image.open(p)
    output_images = []
    output_masks = []
    w, h = None, None
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        if len(output_images) == 0:
            w, h = image.size
        if image.size[0] != w or image.size[1] != h:
            continue
        arr = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
        if "A" in i.getbands():
            mask = 1.0 - torch.from_numpy(np.array(i.getchannel("A")).astype(np.float32) / 255.0)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)
        output_images.append(arr)
        output_masks.append(mask.unsqueeze(0))
    if not output_images:
        raise ValueError(f"no image frames found in: {p}")
    return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)


def make_source(path: str) -> dict:
    """Build the MM_SYNC dict from a file (NACHTRAG B/D)."""
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise ValueError(f"source file not found: {p}")
    image, mask = load_image_file(p)
    h, w = image.shape[1], image.shape[2]
    return {
        "path": p,
        "image": image,
        "mask": mask,
        "width": int(w),
        "height": int(h),
        "format": os.path.splitext(p)[1].lower() or ".png",
        "name": os.path.splitext(os.path.basename(p))[0],
    }


def save_tensor(img, target: str, quality: int = 95, pnginfo=None) -> str:
    """Save one image (H,W,C float 0..1) to `target` (png/jpg/webp by extension).
    The target folder is created if missing (save_as with a new folder)."""
    target = os.path.abspath(target)
    ext = os.path.splitext(target)[1].lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"unsupported target extension: {ext!r}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if ext == ".png":
        pil.save(target, pnginfo=pnginfo, compress_level=4)
    elif ext in (".jpg", ".jpeg"):
        pil.convert("RGB").save(target, quality=int(quality))
    else:  # .webp
        pil.save(target, quality=int(quality))
    return target


def png_metadata(prompt, extra_pnginfo) -> PngImagePlugin.PngInfo:
    """Like core SaveImage.save_images (ComfyUI/nodes.py:1709-1716)."""
    metadata = PngImagePlugin.PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for x in extra_pnginfo:
            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
    return metadata


def _input_images() -> list:
    """Upload-combo options: the files in ComfyUI/input (like core LoadImage,
    nodes.py:1750-1752)."""
    try:
        import folder_paths
        d = folder_paths.get_input_directory()
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        return sorted(folder_paths.filter_files_content_types(files, ["image"]))
    except Exception:
        return []


def _default_source() -> dict:
    """Fallback source (no connection): the output dir, stem 'preview', .png."""
    import folder_paths
    return {
        "path": os.path.join(folder_paths.get_output_directory(), "preview.png"),
        "name": "preview",
        "format": ".png",
    }


def _resolve_source_path(source) -> str:
    """The source path from a MM_SYNC dict, else the preview fallback."""
    if source is not None and source.get("path"):
        return str(source["path"])
    return _default_source()["path"]


LOADER_ID = "MMLoadImage"


def find_loader(prompt, my_id, select="auto"):
    """Wireless source (no sync link): -> (loader prompt id or None, note).
    select '#668' / '668' = that Load Image MM (also inside subgraphs: '12:668');
    'auto' = the one Load Image MM upstream of our images input, else the only one in the workflow."""
    if not isinstance(prompt, dict):
        return None, "no prompt"
    loaders = [k for k, v in prompt.items() if isinstance(v, dict) and v.get("class_type") == LOADER_ID]
    sel = str(select or "").strip().lstrip("#").strip()
    if sel and sel.lower() != "auto":
        hit = [k for k in loaders if str(k) == sel or str(k).endswith(":" + sel)]
        return (hit[0], "") if len(hit) == 1 else (None, f"source #{sel}: no Load Image MM with that number")
    found, seen = [], set()
    me = prompt.get(str(my_id)) or {}
    stack = [v for v in [(me.get("inputs") or {}).get("images")] if isinstance(v, list)]
    while stack:
        nid = str(stack.pop()[0])
        if nid in seen:
            continue
        seen.add(nid)
        node = prompt.get(nid) or {}
        if node.get("class_type") == LOADER_ID:
            found.append(nid)
            continue
        stack.extend(v for v in (node.get("inputs") or {}).values() if isinstance(v, list) and len(v) == 2)
    if len(found) == 1:
        return found[0], ""
    if not found and len(loaders) == 1:
        return loaders[0], ""
    if len(found) > 1 or len(loaders) > 1:
        ids = " ".join("#" + str(k).split(":")[-1] for k in (found or loaders))
        return None, f"several loaders ({ids}) – set source"
    return None, ""


def source_from_prompt(prompt, loader_id):
    """MM_SYNC dict of a loader read from the prompt (its image/path widgets)."""
    inp = (prompt.get(str(loader_id)) or {}).get("inputs") or {}
    image, path = inp.get("image", ""), inp.get("path", "")
    if not isinstance(image, str) or not isinstance(path, str):
        return None  # widget driven by a link: unknown before it runs
    return make_source(resolve_source(image, path))


def _targets(images, source, suffix, save_as, fmt) -> list:
    """One target per image in the batch – running numbers (haus_1, haus_2, ...),
    never overwriting. Same rules as next_target, but the batch reserves its own
    numbers (the files do not exist on disk while they are computed)."""
    if source is not None and source.get("path"):
        src = os.path.abspath(str(source["path"]))
        src_dir = save_dir(src)
        stem = str(source.get("name") or os.path.splitext(os.path.basename(src))[0])
        ext = _ext_for(src, fmt)
        sa = _clean(save_as)
        if sa:
            base = os.path.abspath(sa) if os.path.isabs(sa) else os.path.join(src_dir, sa)
            root, bext = os.path.splitext(base)
            if bext.lower() in IMAGE_EXTS:
                ext = bext.lower()
            base = root
        else:
            base = os.path.join(src_dir, stem + (suffix or ""))
    else:
        # no source: output dir; save_as = name/subfolder like SaveImage's filename_prefix, else "preview"
        import folder_paths
        out_dir = os.path.abspath(folder_paths.get_output_directory())
        stem = _clean(save_as).replace("\\", "/").strip("/") or "preview"
        base = os.path.abspath(os.path.join(out_dir, stem + (suffix or "")))
        if os.path.commonpath([out_dir, base]) != out_dir:  # no escape from output
            base = os.path.join(out_dir, "preview" + (suffix or ""))
        ext = ".png" if fmt == "same as source" else _FMT_EXT.get(fmt, ".png")
    taken = []
    out = []
    for _ in images:
        t = os.path.abspath(_pick_free(base, ext, taken))
        taken.append(t)
        out.append(t)
    return out


def _save_batch(images, targets, fmt, quality, embed_workflow, prompt, extra_pnginfo) -> list:
    """Save every image of the batch to its target. Returns the written paths."""
    paths = []
    for i, img in enumerate(images):
        target = targets[i]
        pnginfo = png_metadata(prompt, extra_pnginfo) if (embed_workflow and os.path.splitext(target)[1].lower() == ".png") else None
        save_tensor(img, target, quality=quality, pnginfo=pnginfo)
        paths.append(target)
    return paths


def _preview_entries(tensors, cls) -> list:
    """Temp-folder preview entries [{filename, subfolder, type}, ...] (rgthree
    image_comparer.py:30-45 pattern: PreviewImage(...).as_dict()['images'])."""
    return ui.PreviewImage(tensors, cls=cls).as_dict()["images"]


# ---------------------------------------------------------------------------
# Node 1: Load Image MM
# ---------------------------------------------------------------------------

ALIASES = ["mm", "metamountain", "mm nodes"]


def _size_text(p: str, w: int, h: int) -> str:
    try:
        mb = os.path.getsize(p) / 1e6
        size = f"{mb:.1f} MB" if mb >= 1 else f"{mb * 1000:.0f} KB"
    except OSError:
        size = "?"
    return f"{w}×{h} · {os.path.splitext(p)[1].lstrip('.').lower()} · {size}"


class MMLoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMLoadImage",
            display_name="Load Image MM",
            category=CATEGORY,
            description="Load an image by upload or full file path. 'sync' carries path, original image, mask, "
                        "size and format to Preview Image MM, which saves next to the source.",
            search_aliases=ALIASES + ["mm load", "load image mm", "load image path"],
            inputs=[
                io.Combo.Input("image", options=_input_images(), upload=io.UploadType.image,
                               tooltip="upload from ComfyUI/input (used when 'path' is empty)"),
                io.String.Input("path", default="", optional=True,
                                tooltip="full file path – wins over the upload; Preview Image MM saves next to this file"),
            ],
            outputs=[
                SOURCE.Output("sync", display_name="sync",
                              tooltip="one link to Preview Image MM: path + original image + mask + size + format"),
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("path", display_name="path", tooltip="resolved absolute path"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, image, path="") -> str:
        """Reload when the file changes (path + mtime)."""
        p = _clean(path)
        if not p:
            try:
                import folder_paths
                p = folder_paths.get_annotated_filepath(image)
            except Exception:
                p = image
        try:
            return f"{p}|{os.path.getmtime(p)}"
        except OSError:
            return str(p)

    @classmethod
    def validate_inputs(cls, image, path=""):
        p = _clean(path)
        if p and not os.path.isfile(os.path.abspath(p)):
            return f"file not found: {os.path.abspath(p)}"
        return True

    @classmethod
    def execute(cls, image, path=""):
        p = resolve_source(image, path)
        img, mask = load_image_file(p)
        src = make_source(p)
        h, w = int(img.shape[1]), int(img.shape[2])
        info = {"info": [_size_text(p, w, h)], "path": [p]}
        if _clean(path):  # path mode: the upload widget shows nothing -> preview the file
            info["images"] = _preview_entries(img[:1], cls)
        return io.NodeOutput(src, img, mask, w, h, p, ui=info)


# ---------------------------------------------------------------------------
# Node 2: Preview Image MM (preview + compare + save; auto save = save every run)
# ---------------------------------------------------------------------------

FORMATS = ["same as source", "png", "jpg", "webp"]


class MMPreviewImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMPreviewImage",
            display_name="Preview Image MM",
            category=CATEGORY,
            is_output_node=True,
            description="Preview the creations, compare them with the original (slider) and save the selected one "
                        "next to the source per button (name_1, name_2 … or save as) – or every run with auto_save. "
                        "Uploads (ComfyUI/input) are saved to ComfyUI/output.",
            search_aliases=ALIASES + ["mm preview", "mm save", "preview image mm", "save image mm", "compare"],
            inputs=[
                io.Image.Input("images"),
                SOURCE.Input("sync", optional=True,
                             tooltip="from Load Image MM: original for the compare, target folder, name and format"),
                io.Boolean.Input("auto_save", default=False, label_on="auto save", label_off="save per button",
                                 tooltip="on: every run saves all creations; off: preview only, save with the buttons"),
                io.Boolean.Input("compare", default=True, label_on="compare", label_off="creation only",
                                 tooltip="slider compare with the original"),
                io.String.Input("source", default="auto",
                                tooltip="without a sync link: 'auto' = the Load Image MM upstream (or the only one); "
                                        "with several loaders type its number, e.g. #12"),
                io.String.Input("suffix", default="_#",
                                tooltip="appended to the source name; '#' = next free number (name_1, name_2 …)"),
                io.String.Input("save_as", default="",
                                tooltip="auto save only: empty = next to the source with the suffix; a name or a full path"),
                io.Combo.Input("format", options=FORMATS, default="same as source"),
                io.Int.Input("quality", default=95, min=1, max=100, tooltip="jpg/webp quality"),
                io.Boolean.Input("embed_workflow", default=True, tooltip="embed prompt + workflow into the PNG (png only)"),
            ],
            outputs=[
                io.String.Output("saved_paths", display_name="saved", tooltip="one path per line (auto save)"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images, sync=None, auto_save=False, compare=True, source="auto", suffix="_#", save_as="",
                format="same as source", quality=95, embed_workflow=True):
        if images is None or len(images) == 0:
            raise ValueError("no images to preview")
        note, src_id = "", ""
        if sync is None:  # wireless: find the loader in the prompt
            lid, note = find_loader(cls.hidden.prompt, cls.hidden.unique_id, source)
            if lid is not None:
                try:
                    sync = source_from_prompt(cls.hidden.prompt, lid)
                    src_id = str(lid).split(":")[-1]
                except ValueError as e:
                    note = str(e)
        paths = []
        if auto_save:
            targets = _targets(images, sync, suffix, save_as, format)
            paths = _save_batch(images, targets, format, quality, embed_workflow,
                                cls.hidden.prompt, cls.hidden.extra_pnginfo)
        entries = _preview_entries(images, cls)
        has_orig = compare and sync is not None and sync.get("image") is not None and len(sync["image"])
        # no "images" key: the core preview would draw over the comparer (rgthree uses a_/b_images only)
        ui_out = {
            "b_images": entries,
            "a_images": _preview_entries(sync["image"][:1], cls) if has_orig else [],
            "source": [_resolve_source_path(sync) if sync is not None else ""],
            "saved": paths,
            "source_id": [src_id],
            "note": [note],
        }
        return io.NodeOutput("\n".join(paths), ui=ui_out)


NODE_LIST = [MMLoadImage, MMPreviewImage]


# ---------------------------------------------------------------------------
# Routes (pattern: film_nodes.py:90-107 – try/except, no server in unit tests)
# ---------------------------------------------------------------------------

def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.post("/mm-nodes/target")
        async def _target(request):
            data = await request.json()
            try:
                p = _clean(str(data.get("path", "")))
                image = str(data.get("image", ""))
                if not p:
                    if not image:
                        return web.json_response({"ok": False, "error": "no source (path or image)"}, status=400)
                    import folder_paths
                    p = os.path.abspath(folder_paths.get_annotated_filepath(image))
                else:
                    p = os.path.abspath(p)
                if not os.path.isfile(p):
                    return web.json_response({"ok": False, "error": f"file not found: {p}"}, status=404)
                target = next_target(p, str(data.get("suffix", "_#")), str(data.get("save_as", "")), str(data.get("format", "same as source")))
                return web.json_response({"ok": True, "target": target})
            except Exception as e:  # noqa: BLE001
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @PromptServer.instance.routes.post("/mm-nodes/save")
        async def _save(request):
            data = await request.json()
            try:
                t = str(data.get("type", ""))
                if t not in ("temp", "output"):
                    return web.json_response({"ok": False, "error": "type must be 'temp' or 'output'"}, status=400)
                import folder_paths
                base_dir = folder_paths.get_temp_directory() if t == "temp" else folder_paths.get_output_directory()
                sub = str(data.get("subfolder", ""))
                fname = str(data.get("filename", ""))
                if not fname or os.path.isabs(fname):
                    return web.json_response({"ok": False, "error": "bad filename"}, status=400)
                src_file = os.path.abspath(os.path.join(base_dir, sub, fname))
                # the preview file must live inside its own folder (no path escape)
                if os.path.commonpath([os.path.abspath(base_dir), src_file]) != os.path.abspath(base_dir):
                    return web.json_response({"ok": False, "error": "file outside the allowed folder"}, status=403)
                if not os.path.isfile(src_file):
                    return web.json_response({"ok": False, "error": f"preview file not found: {src_file}"}, status=404)
                fmt = str(data.get("format", "same as source"))
                if fmt not in ("same as source",) and fmt not in _FMT_EXT:
                    return web.json_response({"ok": False, "error": f"bad format: {fmt}"}, status=400)
                src = _clean(str(data.get("source", "")))
                if not src and (_clean(str(data.get("path", ""))) or str(data.get("image", ""))):
                    # the run had no source: the loader the user picked (its path / upload name)
                    src = resolve_source(str(data.get("image", "")), str(data.get("path", "")))
                if not src:
                    src = os.path.join(folder_paths.get_output_directory(), "preview.png")
                else:
                    src = os.path.abspath(src)
                if not os.path.isfile(src):
                    return web.json_response({"ok": False, "error": f"source not found: {src}"}, status=404)
                target = next_target(src, str(data.get("suffix", "_#")), str(data.get("save_as", "")), fmt)
                if os.path.splitext(target)[1].lower() not in IMAGE_EXTS:
                    return web.json_response({"ok": False, "error": f"target extension not allowed: {os.path.splitext(target)[1]}"}, status=400)
                with Image.open(src_file) as im:
                    # keep the preview's PNG text chunks (prompt/workflow) when saving as PNG
                    texts = {k: v for k, v in (im.info or {}).items() if isinstance(v, str)}
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    quality = int(data.get("quality", 95))
                    ext = os.path.splitext(target)[1].lower()
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    if ext == ".png":
                        from PIL.PngImagePlugin import PngInfo
                        info = PngInfo()
                        for k, v in texts.items():
                            info.add_text(k, v)
                        im.save(target, compress_level=4, pnginfo=info)
                    elif ext in (".jpg", ".jpeg"):
                        im.save(target, quality=quality)
                    else:
                        im.save(target, quality=quality)
                return web.json_response({"ok": True, "path": target})
            except Exception as e:  # noqa: BLE001
                return web.json_response({"ok": False, "error": str(e)}, status=400)
    except Exception as e:  # noqa: BLE001 - without a server (unit tests) no route
        print(f"[metamountain.MMNodes] routes not registered: {e}")
