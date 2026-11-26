"""Shared building blocks of the MM Nodepack: constants, path cleaning, the LRU
cache, folder_paths access, route error helpers, and the image load/save core
(mirrors the behaviour of core LoadImage / SaveImage / PreviewImage).

Importable without ComfyUI: folder_paths is only imported lazily inside functions.
"""
import json
import os
import re
import stat
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, PngImagePlugin

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CATEGORY = "MM Nodepack"

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


class LRUCache:
    """Dict with a hard entry cap: the least recently used entry is evicted first,
    so the cache can never grow unbounded (used for _ORIGIN_CACHE)."""

    def __init__(self, maxsize: int = 64):
        self.maxsize = int(maxsize)
        self._data = OrderedDict()

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __delitem__(self, key):
        del self._data[key]

    def get(self, key, default=None):
        if key in self._data:
            return self[key]
        return default

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def clear(self):
        self._data.clear()


def get_folder_paths():
    """ComfyUI's folder_paths module (only available inside a ComfyUI runtime).
    Lazy import so this package also imports in a plain test environment."""
    import folder_paths
    return folder_paths


def is_local_request(remote: str) -> bool:
    """True when the browser runs on the ComfyUI computer (else an OS file dialog
    would open on the wrong machine)."""
    import socket
    if remote in ("127.0.0.1", "::1"):
        return True
    try:
        return remote in {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None)}
    except OSError:
        return False


def json_error(msg: str, status: int = 400):
    """Unified route error response: {"ok": False, "error": msg}."""
    from aiohttp import web
    return web.json_response({"ok": False, "error": msg}, status=status)


def png_text_chunks(p: str) -> dict:
    """String-valued info entries of the file (the preview's PNG text chunks, e.g. the
    embedded prompt/workflow). {} for unreadable files or formats without text chunks."""
    try:
        with Image.open(p) as im:
            return {k: v for k, v in (im.info or {}).items() if isinstance(v, str)}
    except Exception:  # noqa: BLE001 – unreadable file: no metadata to carry over
        return {}


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


def _natural_key(name: str) -> list:
    """Sort key so img_2.png comes before img_10.png."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", name.lower())]


def list_folder_images(folder: str) -> list:
    """Return all direct image files in `folder` in natural filename order."""
    p = _clean(folder)
    if not p:
        raise ValueError("folder path is empty")
    p = os.path.abspath(p)
    if not os.path.isdir(p):
        raise ValueError(f"folder not found: {p}")
    files = []
    with os.scandir(p) as it:
        for entry in it:
            try:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS:
                    files.append(os.path.abspath(entry.path))
            except OSError:
                pass
    files.sort(key=lambda f: _natural_key(os.path.basename(f)))
    return files


def view_entry(p: str):
    """/view entry for a file inside ComfyUI's output or input folder (for the thumbnail), else None."""
    try:
        fp = get_folder_paths()
    except Exception:  # noqa: BLE001
        return None
    for kind, base in (("output", fp.get_output_directory()), ("input", fp.get_input_directory())):
        base = os.path.abspath(base)
        try:
            if os.path.commonpath([base, os.path.abspath(p)]) == base:
                sub = os.path.relpath(os.path.dirname(os.path.abspath(p)), base)
                return {"filename": os.path.basename(p), "subfolder": "" if sub == "." else sub, "type": kind}
        except ValueError:  # other drive
            pass
    return None


def preview_view_entry(p: str):
    """/view entry for any existing file: inside ComfyUI's output/input folder it is served
    directly (view_entry); anywhere else a cached 1024 px RGB JPEG thumbnail in the temp
    folder (the /view endpoint cannot serve files outside input/output)."""
    entry = view_entry(p)
    if entry is not None:
        return entry
    try:
        fp = get_folder_paths()
        st = os.stat(p)
    except Exception:  # noqa: BLE001 – missing file or no folder_paths (unit tests)
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    try:
        import hashlib
        h = hashlib.md5(f"{os.path.abspath(p)}|{st.st_mtime}|{st.st_size}".encode("utf-8")).hexdigest()[:16]
        fname = f"mm_prev_{h}.jpg"
        cache = os.path.join(fp.get_temp_directory(), fname)
        if not os.path.exists(cache):
            os.makedirs(fp.get_temp_directory(), exist_ok=True)
            tmp = f"{cache}.{os.getpid()}.tmp"
            try:
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im)
                    im = im.convert("RGB")
                    im.thumbnail((1024, 1024))
                    im.save(tmp, "JPEG", quality=88)
                os.replace(tmp, cache)  # atomic: the served file is never half-written
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:  # noqa: BLE001
                    pass
                return None
        return {"filename": fname, "subfolder": "", "type": "temp"}
    except OSError:
        return None
