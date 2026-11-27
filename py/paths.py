"""File-system plumbing of the MM Nodepack: origin lookup for uploads, the unified
target resolution (single file + batch), folder batch picking, and the per-mode
last-dialog folder memory (stored in ComfyUI's temp dir, never in the repository)."""
import json
import os
import random

from .common import (
    IMAGE_EXTS,
    _FMT_EXT,
    LRUCache,
    _clean,
    _ext_for,
    _pick_free,
    get_folder_paths,
    list_folder_images,
)

_ORIGIN_CACHE = LRUCache(64)  # (upload path, mtime, size) -> identical originals in the output tree


def _md5(p: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files_of_size(root: str, size: int) -> list:
    """All files below `root` with exactly `size` bytes (scandir: sizes come with the listing)."""
    hits, stack = [], [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.stat(follow_symlinks=False).st_size == size:
                            hits.append(e.path)
                    except OSError:
                        pass
        except OSError:
            pass
    return hits


def find_origin_files(src: str, out_dir: str) -> list:
    """An upload is a copy: find its original(s) in the output tree. Same name + same content
    first, else any file with the same content (uploads renamed to 'image (3).png').
    Several hits = identical copies in different folders (the browser never sends the real
    location, so the caller has to ask). Pasted bitmaps have no original -> []. Cached."""
    try:
        st = os.stat(src)
    except OSError:
        return []
    key = (os.path.abspath(src), st.st_mtime, st.st_size)
    if key in _ORIGIN_CACHE:
        return _ORIGIN_CACHE[key]
    same = _files_of_size(out_dir, st.st_size)
    named = [p for p in same if os.path.basename(p) == os.path.basename(src)]
    res = []
    if same:
        want = _md5(src)
        res = [p for p in named if _md5(p) == want] or [p for p in same if _md5(p) == want]
    res.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    _ORIGIN_CACHE[key] = res
    return res


def find_origin_file(src: str, out_dir: str):
    """The original of an upload when it is unambiguous (exactly one copy), else None."""
    f = find_origin_files(src, out_dir)
    return f[0] if len(f) == 1 else None


def find_origin(src: str, out_dir: str):
    """Folder of the original of an upload (see find_origin_file) or None."""
    f = find_origin_file(src, out_dir)
    return os.path.dirname(f) if f else None


def save_dir(src: str) -> str:
    """Folder for saves of `src`: next to the file – except uploads in ComfyUI/input
    (only a copy of the user's file): those go next to the original in ComfyUI/output when it
    is found there (same name + content), else to ComfyUI/output (same subfolder)."""
    d = os.path.dirname(os.path.abspath(src))
    try:
        fp = get_folder_paths()
        inp = os.path.abspath(fp.get_input_directory())
        if os.path.commonpath([inp, os.path.abspath(src)]) == inp:
            out = fp.get_output_directory()
            return find_origin(src, out) or os.path.normpath(os.path.join(out, os.path.relpath(d, inp)))
    except Exception:  # noqa: BLE001 – no ComfyUI (unit tests) or other drive
        pass
    return d


def resolve_source(image: str, path: str) -> str:
    """`path` (stripped, quotes removed) wins; else the upload combo via folder_paths.
    The file must exist, else ValueError with the path."""
    p = _clean(path)
    if p:
        p = os.path.abspath(p)
    else:
        p = get_folder_paths().get_annotated_filepath(image)
    if not os.path.isfile(p):
        raise ValueError(f"source file not found: {p}")
    return p


def resolve_folder_image(folder: str, batch_mode: bool = False, index: int = 1) -> tuple:
    """Pick one image from `folder`: incremental (1-based index, 1 = first image,
    wraps after the last) or random (seeded by index).
    Returns (abs_path, position_0_based, total_count)."""
    files = list_folder_images(folder)
    if not files:
        raise ValueError(f"no images found in folder: {os.path.abspath(_clean(folder))}")
    count = len(files)
    if batch_mode:
        pos = random.Random(int(index)).randrange(count)
    else:
        pos = (max(1, int(index)) - 1) % count
    return files[pos], pos, count


def _target_base(source, suffix: str, save_as: str, fmt: str):
    """Shared core of next_target() (str source) and resolve_targets() (source dict or None):
    returns (base, ext) – base = target path without extension, may contain '#'.

    With a source: save_as set -> quotes removed, absolute path used as-is, else relative to
    the source folder, an image extension in save_as wins (without one the target ext is
    appended); save_as empty -> <source folder>/<stem><suffix> (the dict source's 'name' wins
    over the file stem). Without any source: the output dir, save_as like SaveImage's
    filename_prefix (no escape from the output dir), else stem 'preview'.
    """
    if source is None or (isinstance(source, dict) and not source.get("path")):
        out_dir = os.path.abspath(get_folder_paths().get_output_directory())
        stem = _clean(save_as).replace("\\", "/").strip("/") or "preview"
        base = os.path.abspath(os.path.join(out_dir, stem + (suffix or "")))
        try:
            escapes = os.path.commonpath([out_dir, base]) != out_dir
        except ValueError:  # different drive: the output dir is definitely not the base
            escapes = True
        if escapes:  # no escape from the output dir
            base = os.path.join(out_dir, "preview" + (suffix or ""))
        ext = ".png" if fmt == "same as source" else _FMT_EXT.get(fmt, ".png")
        return base, ext
    if isinstance(source, dict):
        src = os.path.abspath(str(source["path"]))
        stem = str(source.get("name") or os.path.splitext(os.path.basename(src))[0])
    else:
        src = os.path.abspath(str(source))
        stem = os.path.splitext(os.path.basename(src))[0]
    src_dir = save_dir(src)
    ext = _ext_for(src, fmt)
    sa = _clean(save_as)
    if sa:
        base = os.path.abspath(sa) if os.path.isabs(sa) else os.path.join(src_dir, sa)
        root, bext = os.path.splitext(base)
        # an image extension in save_as wins; without one the target ext is appended
        if bext.lower() in IMAGE_EXTS:
            ext = bext.lower()
        base = root
    else:
        base = os.path.join(src_dir, stem + (suffix or ""))
    return base, ext


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
    base, ext = _target_base(src, suffix, save_as, fmt)
    return os.path.abspath(_pick_free(base, ext))


def resolve_save_target(source, suffix, save_as, fmt):
    """Next free save target, also for saves with NO source (e.g. a txt2img preview).
    With a source: same rules as next_target() (ValueError when the source is missing).
    Without a source: an absolute save_as is used as-is (an image extension in it wins,
    a foreign extension is kept and the target ext appended), else the output dir –
    save_as as a prefix (no escape from it) or stem 'preview'. Never overwrites."""
    src = _clean(source)
    if src:
        src = os.path.abspath(src)
        if not os.path.isfile(src):
            raise ValueError(f"source not found: {src}")
        base, ext = _target_base(src, suffix, save_as, fmt)
    else:
        sa = _clean(save_as)
        if sa and os.path.isabs(sa):
            root, bext = os.path.splitext(os.path.abspath(sa))
            if bext.lower() in IMAGE_EXTS:
                ext = bext.lower()
                base = root
            else:
                ext = ".png" if fmt == "same as source" else _FMT_EXT.get(fmt, ".png")
                base = os.path.abspath(sa)
        else:  # empty or relative save_as -> the output dir (like _target_base(None, ...))
            base, ext = _target_base(None, suffix, save_as, fmt)
    return os.path.abspath(_pick_free(base, ext))


def resolve_targets(images, source, suffix, save_as, fmt) -> list:
    """One target per image in the batch – running numbers (haus_1, haus_2, ...),
    never overwriting. Same rules as next_target, but the batch reserves its own
    numbers (the files do not exist on disk while they are computed).
    `source` = make_source() dict or None (no source: the output dir)."""
    base, ext = _target_base(source, suffix, save_as, fmt)
    taken, out = [], []
    for _ in images:
        t = os.path.abspath(_pick_free(base, ext, taken))
        taken.append(t)
        out.append(t)
    return out


class NoSourceError(ValueError):
    """Nothing in the payload identifies a source (no folder, path, or image)."""


def resolve_source_like(data: dict) -> str:
    """Absolute source path from a route payload: folder mode wins (the image the folder
    batch currently points at), then 'path' (stripped, quotes removed), then the upload
    combo via folder_paths. Raises NoSourceError when nothing is given and ValueError
    from resolve_folder_image when the folder has no supported images."""
    if data.get("mode") == "folder" and _clean(str(data.get("folder", ""))):
        p, _, _ = resolve_folder_image(str(data.get("folder", "")), bool(data.get("batch_mode", False)), int(data.get("index", 1)))
        return p
    p = _clean(str(data.get("path", "")))
    if p:
        return os.path.abspath(p)
    image = str(data.get("image", ""))
    if not image:
        raise NoSourceError("no source (path or image)")
    return os.path.abspath(get_folder_paths().get_annotated_filepath(image))


def last_dir_file() -> str:
    """Path of the last-dialog folder memory: in ComfyUI's temp dir (outside the
    repository) when available, else the OS temp dir."""
    try:
        d = get_folder_paths().get_temp_directory()
    except Exception:  # noqa: BLE001 – no ComfyUI runtime (unit tests)
        import tempfile
        d = tempfile.gettempdir()
    return os.path.join(d, "mm_nodepack_last_dir.json")


def load_last_dir(mode: str = "open") -> str:
    """Persisted last dialog folder for `mode` ('' = none). A legacy file that
    holds one shared 'dir' entry answers with that dir for every mode."""
    try:
        with open(last_dir_file(), "r") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(data, dict):
        return ""
    if "dir" in data:  # legacy file: one shared dir for all modes
        val = data.get("dir", "")
        return val if isinstance(val, str) else ""
    value = data.get(mode, "")
    return value if isinstance(value, str) else ""


def save_last_dir(path: str, mode: str = "open") -> None:
    """Persist the last dialog folder per mode (survives reboot). A legacy
    single-'dir' file is migrated: its dir seeds every mode exactly once."""
    try:
        p = last_dir_file()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = None
        try:
            with open(p, "r") as f:
                stored = json.load(f)
            if isinstance(stored, dict) and "dir" in stored:
                data = {m: stored.get("dir", "") for m in ("open", "save", "folder")}
            elif isinstance(stored, dict):
                data = stored
        except Exception:  # noqa: BLE001 – unreadable/missing file: start fresh
            data = None
        data = data if data is not None else {}
        data[mode] = path
        with open(p, "w") as f:
            json.dump(data, f)
    except Exception:  # noqa: BLE001 – a failed write must never break the dialog
        pass
