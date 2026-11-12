"""mm-nodepack-comfyui: load an image together with its path, edit it, and save the
result next to the source (haus_1.png, haus_2.png, ... or "save as").

V3 node API (comfy_api.latest). The pure helpers (next_target, resolve_source,
load_image_file, save_tensor, png_metadata, make_source) are unit-testable without
ComfyUI: folder_paths is only imported inside functions.
"""
import json
import asyncio
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence, PngImagePlugin

from comfy_api.latest import io, ui

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
CATEGORY = "MM Nodepack"

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


_ORIGIN_CACHE = {}  # (upload path, mtime, size) -> list of identical originals in output


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
        import folder_paths
        inp = os.path.abspath(folder_paths.get_input_directory())
        if os.path.commonpath([inp, os.path.abspath(src)]) == inp:
            out = folder_paths.get_output_directory()
            return find_origin(src, out) or os.path.normpath(os.path.join(out, os.path.relpath(d, inp)))
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

ALIASES = ["mm", "metamountain", "mm nodepack"]


# --- native "open file" dialog of the OS (opens on the ComfyUI computer) -------------------
def is_local_request(remote: str) -> bool:
    """True when the browser runs on the ComfyUI computer (else the dialog would open elsewhere)."""
    import socket
    if remote in ("127.0.0.1", "::1"):
        return True
    try:
        return remote in {ai[4][0] for ai in socket.getaddrinfo(socket.gethostname(), None)}
    except OSError:
        return False


def open_folder(p: str) -> str:
    """Show p in the OS file manager: a file is selected in its folder, a folder is opened.
    Returns the folder that was opened."""
    import subprocess
    import sys
    p = os.path.abspath(_clean(p))
    is_file = os.path.isfile(p)
    folder = os.path.dirname(p) if is_file or not os.path.isdir(p) else p
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"folder not found: {folder}")
    if os.name == "nt":
        if is_file:
            subprocess.Popen(["explorer", "/select,", p])
        else:
            os.startfile(folder)  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
    return folder


def dialog_available() -> bool:
    """An OS file dialog can open here (Linux: a desktop session and zenity/kdialog; a headless
    server or container has none -> the browser upload / save picker is used instead)."""
    import shutil
    import sys
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and bool(
        shutil.which("zenity") or shutil.which("kdialog"))


def _mm_last_dir() -> str:
    """Persist last file-dialog folder (survives reboot)."""
    import json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_dir.json")
    try:
        with open(p, "r") as f: return json.load(f).get("dir", "")
    except Exception: return ""

def _mm_save_dir(d: str) -> None:
    """Save last file-dialog folder."""
    import json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_dir.json")
    try:
        with open(p, "w") as f: json.dump({"dir": d}, f)
    except Exception: pass


def file_dialog(mode: str, start_dir: str, name: str = "") -> str:
    """The OS file dialog, mode 'open' or 'save'. Returns the chosen path, '' = cancelled."""
    import shutil
    import subprocess
    import sys
    import time
    save = mode == "save"
    # Restore last folder if no explicit start_dir
    if not start_dir:
        start_dir = _mm_last_dir()
    t0 = time.time()
    env = dict(os.environ, MM_START=start_dir, MM_NAME=name)
    if os.name == "nt":
        result = _win_file_dialog_tk(start_dir, name, save)
    else:
        result = _posix_file_dialog(mode, start_dir, name, env)
    elapsed = time.time() - t0
    print(f"[MM Nodepack] file_dialog ({mode}): {elapsed:.2f}s → {result or 'cancelled'}")
    # Save the folder of the chosen file
    if result:
        _mm_save_dir(os.path.dirname(result))
    return result


def _posix_file_dialog(mode: str, start_dir: str, name: str, env: dict) -> str:
    """macOS/Linux file dialog via subprocess."""
    import shutil
    import subprocess
    import sys
    save = mode == "save"
    if sys.platform == "darwin":
        loc = 'default location (POSIX file (system attribute "MM_START"))'
        cmd = ["osascript", "-e", f'POSIX path of (choose file name default name (system attribute "MM_NAME") {loc})' if save
               else f'POSIX path of (choose file of type {{"public.image"}} {loc})']
    elif shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", f"--filename={os.path.join(start_dir, name)}"] + (["--save"] if save else [])
    elif shutil.which("kdialog"):
        cmd = ["kdialog", "--getsavefilename" if save else "--getopenfilename", os.path.join(start_dir, name)]
    else:
        raise RuntimeError("no file dialog available (install zenity or kdialog)")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
                       env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
    return lines[-1] if lines else ""


def _win_file_dialog_tk(start_dir: str, name: str, save: bool) -> str:
    """Windows file dialog via COM IFileOpenDialog — instant, in-process,
    guaranteed foreground (owner = GetForegroundWindow), any drive, full path."""
    import ctypes
    import uuid
    from ctypes import wintypes

    ole32 = ctypes.OleDLL("Ole32")
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    user32 = ctypes.WinDLL("User32", use_last_error=True)

    # --- COM init ---
    ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED (STA — required for UI dialogs)

    # --- GUID helper ---
    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]

    def _g(s):
        return _GUID.from_buffer_copy(uuid.UUID(s).bytes_le)

    CLSID_FileOpenDialog = _g("DC1C5A9C-9B69-11d3-AF79-00C04F79C160")
    IID_IFileOpenDialog = _g("d57c7288-d4ad-4768-be02-9d969532d960")
    IID_IShellItem = _g("43826d1e-e718-42ee-bc55-a1e261c37bfe")

    # --- Vtable function prototypes ---
    # IUnknown: 0=QueryInterface, 1=AddRef, 2=Release
    # IFileDialog: 3=Show, 4=SetOptions, 5=GetOptions, 6=SetDefaultFolder,
    #   7=GetDefaultFolder, 8=SetFolder, 9=GetFolder, 10=GetFile,
    #   11=SetFileName, 12=GetFileName
    # IFileOpenDialog: 13=GetResults, 14=GetSelectedItems
    PFN_Show = ctypes.WINFUNCTYPE(wintypes.HRESULT, wintypes.HWND)
    PFN_SetOptions = ctypes.WINFUNCTYPE(wintypes.HRESULT, wintypes.DWORD)
    PFN_GetFile = ctypes.WINFUNCTYPE(wintypes.HRESULT, ctypes.POINTER(ctypes.c_void_p))
    PFN_SetFileName = ctypes.WINFUNCTYPE(wintypes.HRESULT, wintypes.LPCWSTR)
    PFN_GetFileName = ctypes.WINFUNCTYPE(wintypes.HRESULT, ctypes.POINTER(wintypes.LPWSTR))
    PFN_Release = ctypes.WINFUNCTYPE(wintypes.ULONG)
    # IShellItem::GetDisplayName (index 2): (SIGDN, LPWSTR*)
    PFN_GetDisplayName = ctypes.WINFUNCTYPE(wintypes.HRESULT, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR))

    # --- Create IFileOpenDialog ---
    pDialog = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(ctypes.byref(CLSID_FileOpenDialog), None, 1,
                                ctypes.byref(IID_IFileOpenDialog), ctypes.byref(pDialog))
    if hr != 0:
        ole32.CoUninitialize()
        raise RuntimeError(f"CoCreateInstance failed: 0x{hr & 0xFFFFFFFF:08x}")

    # Get vtable: pDialog → vtable (array of function pointers)
    vtbl = ctypes.cast(pDialog, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))

    def vcall(idx, proto):
        return proto.from_address(ctypes.addressof(vtbl.contents) + idx * ctypes.sizeof(ctypes.c_void_p))

    # --- Set options ---
    FOS_OVERWRITEPROMPT = 0x2
    FOS_PATHMUSTEXIST = 0x800
    FOS_FILEMUSTEXIST = 0x1000
    opts = FOS_PATHMUSTEXIST | (FOS_OVERWRITEPROMPT if save else FOS_FILEMUSTEXIST)
    vcall(4, PFN_SetOptions)(opts)

    # No initial folder — let Windows use its default (remembers last location itself)

    # --- Set filename (for save: initial filename) ---
    if name and save:
        vcall(11, PFN_SetFileName)(name)

    # --- Set file type filter ---
    # Use SetFileTypes via IFileDialog::SetFileTypes (not in our vtable indices above)
    # Skip for now — default is "All Files"

    # --- Show dialog with a temporary topmost window as owner (guaranteed foreground) ---
    kernel32 = ctypes.WinDLL("kernel32")
    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    _wp = WNDPROC(lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l))
    _wp_ref = _wp  # keep alive

    class _WC(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT),
                    ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                    ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE),
                    ("hbrBackground", wintypes.HBRUSH), ("lpszMenuName", wintypes.LPCWSTR),
                    ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]
    wc = _WC()
    wc.cbSize = ctypes.sizeof(_WC)
    wc.lpfnWndProc = _wp_ref
    wc.hInstance = kernel32.GetModuleHandleW(None)
    wc.lpszClassName = "MM_TmpTopmost"
    user32.RegisterClassExW(ctypes.byref(wc))

    owner = user32.CreateWindowExW(
        0x00000088, "MM_TmpTopmost", "", 0, 0, 0, 1, 1, None, None, wc.hInstance, None)
    user32.SetForegroundWindow(owner)

    hr = vcall(3, PFN_Show)(owner)
    user32.DestroyWindow(owner)

    result = ""
    if hr == 0:  # S_OK
        # GetFile(IShellItem**) → IShellItem::GetDisplayName(SIGDN_FILESYSPATH)
        pShellItem = ctypes.c_void_p()
        PFN_GetFile2 = ctypes.WINFUNCTYPE(wintypes.HRESULT, ctypes.POINTER(ctypes.c_void_p))
        hr2 = vcall(10, PFN_GetFile2)(ctypes.byref(pShellItem))
        if hr2 == 0 and pShellItem:
            # IShellItem vtable: 0=BindToHandler, 1=GetParent, 2=GetDisplayName
            si_vtbl = ctypes.cast(pShellItem, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
            si_GetDisplayName = PFN_GetDisplayName.from_address(
                ctypes.addressof(si_vtbl.contents) + 2 * ctypes.sizeof(ctypes.c_void_p))
            pszPath = wintypes.LPWSTR()
            SIGDN_FILESYSPATH = 0x80058000
            hr3 = si_GetDisplayName(SIGDN_FILESYSPATH, ctypes.byref(pszPath))
            if hr3 == 0 and pszPath:
                result = pszPath.value or ""
                ole32.CoTaskMemFree(pszPath)
            # Release IShellItem
            si_Release = PFN_Release.from_address(
                ctypes.addressof(si_vtbl.contents) + 2 * ctypes.sizeof(ctypes.c_void_p))
            si_Release()

    # --- Cleanup: Release dialog ---
    vcall(2, PFN_Release)()
    ole32.CoUninitialize()
    return result





def view_entry(p: str):
    """/view entry for a file inside ComfyUI's output or input folder (for the thumbnail), else None."""
    try:
        import folder_paths
    except Exception:  # noqa: BLE001
        return None
    for kind, base in (("output", folder_paths.get_output_directory()), ("input", folder_paths.get_input_directory())):
        base = os.path.abspath(base)
        try:
            if os.path.commonpath([base, os.path.abspath(p)]) == base:
                sub = os.path.relpath(os.path.dirname(os.path.abspath(p)), base)
                return {"filename": os.path.basename(p), "subfolder": "" if sub == "." else sub, "type": kind}
        except ValueError:  # other drive
            pass
    return None


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
            return (f"path points to a missing file (moved or deleted?): {os.path.abspath(p)} – "
                    "pick or upload an image (that sets a new path) or clear the path field")
        return True

    @classmethod
    def execute(cls, image, path=""):
        p = resolve_source(image, path)
        img, mask = load_image_file(p)
        src = make_source(p)
        h, w = int(img.shape[1]), int(img.shape[2])
        info = {"path": [p]}   # size: the frontend prints it under the thumbnail
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
                io.Boolean.Input("auto_save", default=False, label_on="auto save", label_off="no autosave",
                                 tooltip="on: every run saves all creations; off: preview only, save with the buttons"),
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
                # last on purpose: widgets_values of saved workflows are positional
                io.Boolean.Input("auto_send", default=False, label_on="auto send", label_off="no autosend",
                                 tooltip="on: after every run the result is saved as _# and put into the Load Image MM, "
                                         "so the next run edits it (with auto save on, the auto-saved file is used)"),
            ],
            outputs=[
                io.String.Output("saved_paths", display_name="saved", tooltip="one path per line (auto save)"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images, sync=None, auto_save=False, source="auto", suffix="_#", save_as="",
                format="same as source", quality=95, embed_workflow=True, auto_send=False):
        # auto_send is done by the browser after the run (same path as the "send to loader" button)
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
        # always send the original: the compare toggle under the image is display-only, so turning it on later works without a re-run
        has_orig = sync is not None and sync.get("image") is not None and len(sync["image"])
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

        @PromptServer.instance.routes.post("/mm-nodepack/target")
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

        @PromptServer.instance.routes.post("/mm-nodepack/dialog")
        async def _dialog(request):
            # the OS open/save dialog on the ComfyUI computer -> real paths, no browser sandbox
            if not is_local_request(request.remote) or not dialog_available():
                return web.json_response({"ok": False, "local": False, "error": "no file dialog on the ComfyUI computer"}, status=403)
            try:
                data = await request.json()
                import folder_paths
                mode = "save" if data.get("mode") == "save" else "open"
                cur = _clean(str(data.get("path", "")))
                start = (cur if os.path.isdir(cur) else os.path.dirname(cur)) if cur else ""
                if not start or not os.path.isdir(start):
                    start = folder_paths.get_output_directory()
                p = await asyncio.to_thread(file_dialog, mode, os.path.abspath(start), os.path.basename(str(data.get("name", ""))))
                return web.json_response({"ok": True, "path": p, "view": view_entry(p) if p and mode == "open" else None})
            except Exception as e:  # noqa: BLE001
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @PromptServer.instance.routes.post("/mm-nodepack/open-folder")
        async def _open_folder(request):
            # the file manager opens on the ComfyUI computer -> only for a browser on that computer
            if not is_local_request(request.remote):
                return web.json_response({"ok": False, "error": "open folder only works in a browser on the ComfyUI computer"}, status=403)
            data = await request.json()
            try:
                return web.json_response({"ok": True, "folder": open_folder(str(data.get("path", "")))})
            except Exception as e:  # noqa: BLE001
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @PromptServer.instance.routes.post("/mm-nodepack/resolve")
        async def _resolve(request):
            # picker/upload name ("x.png", "sub/x.png [input]") -> absolute path for the loader's path field
            data = await request.json()
            try:
                import folder_paths
                p = os.path.abspath(folder_paths.get_annotated_filepath(str(data.get("image", ""))))
                if not os.path.isfile(p):
                    return web.json_response({"ok": False, "error": f"file not found: {p}"}, status=404)
                inp = os.path.abspath(folder_paths.get_input_directory())
                if os.path.commonpath([inp, p]) == inp:
                    # an upload is only a copy: show where the original lives ("" = pasted, unknown)
                    o = await asyncio.to_thread(find_origin_files, p, folder_paths.get_output_directory())
                    return web.json_response({"ok": True, "path": o[0] if len(o) == 1 else "", "file": p,
                                              "candidates": o if len(o) > 1 else []})
                return web.json_response({"ok": True, "path": p, "file": p})
            except Exception as e:  # noqa: BLE001
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @PromptServer.instance.routes.post("/mm-nodepack/preview")
        async def _preview(request):
            # live preview of the loader's current file (compare-hold without re-run).
            # same resolution as the backend: path wins, else the upload combo.
            data = await request.json()
            try:
                p = resolve_source(str(data.get("image", "")), str(data.get("path", "")))
                entry = view_entry(p)
                if entry is None:
                    return web.json_response({"ok": False, "error": "file not in output/input folder"}, status=404)
                return web.json_response({"ok": True, "view": entry, "path": p})
            except Exception as e:  # noqa: BLE001
                return web.json_response({"ok": False, "error": str(e)}, status=400)

        @PromptServer.instance.routes.post("/mm-nodepack/save")
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
                out_dir = os.path.abspath(folder_paths.get_output_directory())
                if not is_local_request(request.remote) and os.path.commonpath([out_dir, os.path.abspath(target)]) != out_dir:
                    # browser on another computer (--listen): writes only inside the output folder
                    return web.json_response({"ok": False, "error": "from another computer, saving is limited to the output folder"}, status=403)
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
        print(f"[metamountain.MMNodepack] routes not registered: {e}")
