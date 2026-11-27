"""Shared building blocks of the MM Nodepack: constants, path cleaning, the LRU
cache, folder_paths access, route error helpers, the file access policy, and
the image load/save core (mirrors the behaviour of core LoadImage / SaveImage /
PreviewImage).

File access policy (shared server): queued operations and every non-local
request may read (decode/list/metadata) only from the approved ComfyUI
input/output roots and write only under the output root. Trusted loopback
requests on a private server keep the unrestricted behaviour (see access_policy
for the shared_server() switch and image_io._restricted for the route level).

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

try:
    # server deployment policy, provided by the access_policy module
    # (shared_server() -> bool: the server is reachable by other machines /
    # used by several principals -> the file access policy applies)
    from .access_policy import shared_server
except ImportError:  # module not in place yet: private loopback behaviour
    def shared_server() -> bool:
        return False


class AccessDeniedError(Exception):
    """File access refused by the policy. The message is bounded on purpose:
    it never names the offending path (no external filename disclosure)."""


def policy_active() -> bool:
    """Server-level policy switch: True when the ComfyUI server is shared,
    i.e. queued (node) operations must stay inside the allowed roots.
    Route handlers additionally restrict every non-local request (see
    image_io._restricted), even on a private server."""
    try:
        return bool(shared_server())
    except Exception:  # noqa: BLE001 – a broken policy module must not break the pack
        return False


def _root_paths(*names) -> list:
    """Existing, real paths of the ComfyUI folders `names`
    (like 'input', 'output' of folder_paths)."""
    try:
        fp = get_folder_paths()
    except Exception:  # noqa: BLE001 – no ComfyUI runtime (unit tests)
        return []
    roots = []
    for n in names:
        try:
            p = os.path.realpath(getattr(fp, n)())
        except Exception:  # noqa: BLE001 – folder not configured
            continue
        if os.path.isdir(p):
            roots.append(p)
    return roots


def read_roots() -> list:
    """Real paths of the approved read roots (ComfyUI input + output)."""
    return _root_paths("get_input_directory", "get_output_directory")


def write_roots() -> list:
    """Real paths of the approved write roots (ComfyUI output). The preview
    thumbnail cache in the temp folder is a /view mechanism, not user content."""
    return _root_paths("get_output_directory")


def _is_within(path: str, root: str) -> bool:
    """True when the real path of `path` is inside (or is) the real path of
    `root`. Both sides are normalized with normcase(realpath(...)): symlinks
    are resolved (a link that leaves the folder escapes it) and on
    case-insensitive file systems (Windows) the comparison is case-insensitive
    (a missing write target keeps the caller's case, which realpath preserves)."""
    t = os.path.normcase(os.path.realpath(path))
    r = os.path.normcase(os.path.realpath(root))
    return t == r or t.startswith(r.rstrip(os.sep) + os.sep)


def _under_roots(p: str, roots) -> bool:
    """True when `p` is inside one of `roots` (see _is_within)."""
    return any(_is_within(p, r) for r in roots)


def authorize(path: str, kind: str = "read", forced: bool = False) -> None:
    """Policy gate: `kind` = 'read' (input/output roots) or 'write' (output
    root). Enforced when the server policy is active or `forced` (a route
    handler forces it for every non-local request, even on a private server).
    Raises AccessDeniedError with a bounded message that never names the path."""
    if not (policy_active() or forced):
        return
    if kind == "write":
        if not _under_roots(path, write_roots()):
            raise AccessDeniedError("target outside the allowed output folder")
    elif not _under_roots(path, read_roots()):
        raise AccessDeniedError("path outside the allowed folders")


def open_verified(path: str):
    """Open an existing file for reading and verify that the opened file is
    still the one that was inspected (device + inode match between stat and
    fstat): a swap or symlink replacement in between is detected. The caller
    closes the handle."""
    st = os.stat(path)  # follows links: what we decode is the resolved target
    fd = os.open(path, os.O_RDONLY)
    try:
        fst = os.fstat(fd)
        if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
            raise OSError("file changed between check and open")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def create_exclusive(path: str, roots=None):
    """Create `path` exclusively (O_EXCL): an existing or pre-created file,
    a hard link planted on the name, or a swap of the destination folder is
    all rejected (the caller retries with the next free number). `roots` =
    approved write roots re-verified on the real parent right before the
    open (catches a parent swapped for a link in between)."""
    if roots:  # empty/None = no root verification (private mode)
        parent = os.path.dirname(os.path.abspath(path))
        if not any(os.path.realpath(parent) == r for r in roots):
            raise AccessDeniedError("target outside the allowed output folder")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.fstat(fd)
        return os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        raise


def _next_free_conflict(cand: str, ext: str) -> str:
    """The next name after a file that appeared between the free-number pick
    and the exclusive create: `name_1` -> `name_2`, `name` -> `name_2`."""
    base = cand[: -len(ext)]
    m = re.match(r"^(.*?)(?:_(\d+))?$", base)
    n = int(m.group(2)) if m.group(2) else 1
    return f"{m.group(1)}_{n + 1}{ext}"


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
    The file is opened with an identity check (open_verified): a swap or
    symlink replacement between the check and the open is detected.
    Returns (IMAGE [N,H,W,C] float32 0..1, MASK [N,H,W] float32 0..1)."""
    with open_verified(p) as f:
        img = Image.open(f)
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


def save_tensor(img, target: str, quality: int = 95, pnginfo=None, roots=None, retries: int = 8) -> str:
    """Save one image (H,W,C float 0..1) to `target` (png/jpg/webp by extension).
    The target folder is created if missing (save_as with a new folder).

    The file is created exclusively (O_EXCL, never overwrites and never takes
    over a pre-created name or a link planted on it). If another process
    claimed the name in between the free-number pick and the open, the next
    free number is retried (`retries` times); the returned path is the file
    that was actually written, so the reported path is always accurate.
    `roots` = approved write roots re-verified on the parent before each open."""
    target = os.path.abspath(target)
    ext = os.path.splitext(target)[1].lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"unsupported target extension: {ext!r}")
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    cand = target
    last_err = None
    for _ in range(max(1, retries)):
        os.makedirs(os.path.dirname(cand), exist_ok=True)
        try:
            f = create_exclusive(cand, roots=roots)
        except FileExistsError as e:  # claimed in the race window: next free number
            last_err = e
            cand = _next_free_conflict(cand, ext)
            continue
        try:
            if ext == ".png":
                pil.save(f, format="PNG", pnginfo=pnginfo, compress_level=4)
            elif ext in (".jpg", ".jpeg"):
                pil.convert("RGB").save(f, format="JPEG", quality=int(quality))
            else:  # .webp
                pil.save(f, format="WEBP", quality=int(quality))
            f.close()
        except BaseException:
            # close BEFORE the unlink: on Windows an open handle cannot be unlinked
            try:
                f.close()
            except OSError:
                pass
            try:
                os.unlink(cand)  # no partial file exposed on a failed save
            except OSError:
                pass
            raise
        return cand
    raise FileExistsError(f"could not create a free file for: {target}")


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


def list_folder_images(folder: str, forced: bool = False) -> list:
    """Return all direct image files in `folder` in natural filename order.
    When the server policy is active (shared server) or `forced` (a non-local
    request on a private server) the folder must live in the approved read
    roots (ComfyUI input/output) - checked before the scan."""
    p = _clean(folder)
    if not p:
        raise ValueError("folder path is empty")
    p = os.path.abspath(p)
    # authorize BEFORE the existence check: a restricted request must not be
    # able to probe folder existence outside the approved roots
    authorize(p, "read", forced=forced)
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
    """/view entry for a file inside ComfyUI's output or input folder (for the thumbnail), else None.
    Containment is checked on the real paths: a symlink that leaves the folder is not served
    from it (it falls back to the temp thumbnail branch or to nothing)."""
    try:
        fp = get_folder_paths()
    except Exception:  # noqa: BLE001
        return None
    p = os.path.abspath(p)
    t = os.path.realpath(p)
    for kind, base in (("output", fp.get_output_directory()), ("input", fp.get_input_directory())):
        base = os.path.realpath(base)
        if _is_within(t, base):
            # subfolder from the REAL file path relative to the real base: behind a
            # junction/symlinked root this stays inside the tree (no '..\..' subfolders)
            sub = os.path.relpath(os.path.dirname(t), base)
            # filename from the real path too: a symlinked file must match its real subfolder
            return {"filename": os.path.basename(t), "subfolder": "" if sub == "." else sub, "type": kind}
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
                with open_verified(p) as f, Image.open(f) as im:
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
