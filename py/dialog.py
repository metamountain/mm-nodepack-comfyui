"""OS file dialogs (open/save/folder) on the ComfyUI computer. Windows runs the dialog
in a child process (a crash there cannot take ComfyUI down), macOS via osascript,
Linux via zenity/kdialog. The last-used folder is remembered per dialog mode in
paths.last_dir_file()."""
import json
import os

from .paths import load_last_dir, save_last_dir


def dialog_available() -> bool:
    """An OS file dialog can open here (Linux: a desktop session and zenity/kdialog; a headless
    server or container has none -> the browser upload / save picker is used instead)."""
    import shutil
    import sys
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and bool(
        shutil.which("zenity") or shutil.which("kdialog"))


def file_dialog(mode: str, start_dir: str, name: str = "") -> str:
    """The OS file/folder dialog, mode 'open', 'save', or 'folder'. Returns the chosen path, '' = cancelled."""
    import time
    if mode not in ("open", "save", "folder"):
        mode = "open"
    # Restore last folder of this dialog mode if no explicit start_dir
    if not start_dir:
        start_dir = load_last_dir(mode)
    t0 = time.time()
    env = dict(os.environ, MM_START=start_dir, MM_NAME=name)
    if os.name == "nt":
        result = _win_file_dialog(start_dir, name, mode)
    else:
        result = _posix_file_dialog(mode, start_dir, name, env)
    elapsed = time.time() - t0
    print(f"[MM Nodepack] file_dialog ({mode}): {elapsed:.2f}s -> {result or 'cancelled'}")
    # Persist the chosen folder (or parent folder of the chosen file) per mode
    if result:
        save_last_dir(result if mode == "folder" else os.path.dirname(result), mode)
    return result


def _posix_file_dialog(mode: str, start_dir: str, name: str, env: dict) -> str:
    """macOS/Linux file or folder dialog via subprocess."""
    import shutil
    import subprocess
    import sys
    save = mode == "save"
    folder = mode == "folder"
    if sys.platform == "darwin":
        loc = 'default location (POSIX file (system attribute "MM_START"))' if start_dir else ""
        if folder:
            cmd = ["osascript", "-e", f'POSIX path of (choose folder {loc})']
        elif save:
            cmd = ["osascript", "-e", f'POSIX path of (choose file name default name (system attribute "MM_NAME") {loc})']
        else:
            cmd = ["osascript", "-e", f'POSIX path of (choose file of type {{"public.image"}} {loc})']
    elif shutil.which("zenity"):
        if folder:
            target = (start_dir.rstrip("/") + "/") if start_dir else ""
            cmd = ["zenity", "--file-selection", "--directory"] + ([f"--filename={target}"] if target else [])
        else:
            cmd = ["zenity", "--file-selection", f"--filename={os.path.join(start_dir, name)}"] + (["--save"] if save else [])
    elif shutil.which("kdialog"):
        if folder:
            cmd = ["kdialog", "--getexistingdirectory", start_dir or "."]
        else:
            cmd = ["kdialog", "--getsavefilename" if save else "--getopenfilename", os.path.join(start_dir, name)]
    else:
        raise RuntimeError("no file dialog available (install zenity or kdialog)")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
                       env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = [x.strip() for x in (r.stdout or "").splitlines() if x.strip()]
    return lines[-1].rstrip("/") if (lines and folder and lines[-1] != "/") else (lines[-1] if lines else "")


# Windows open/save/folder dialog in a child process (a crash there cannot take ComfyUI down).
# argv: mode ("open"|"save"|"folder"), start_dir, name. Prints the chosen path as JSON, "" = cancelled.
_WIN_DIALOG_SCRIPT = r'''
import ctypes, json, os, sys
from ctypes import wintypes
mode, start_dir, name = sys.argv[1], sys.argv[2], sys.argv[3]
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32")
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

# ComfyUI runs in the background and Windows refuses focus to background processes:
# invisible topmost owner window + ALT tap brings the dialog to the front
owner = user32.CreateWindowExW(0x88, "STATIC", "MM Nodepack", 0x80000000, 0, 0, 0, 0,
                               None, None, kernel32.GetModuleHandleW(None), None)
user32.keybd_event(0x12, 0, 0, 0)
user32.keybd_event(0x12, 0, 2, 0)
user32.SetForegroundWindow(owner)

chosen = ""

if mode == "folder":
    ole32 = ctypes.OleDLL("ole32")
    shell32 = ctypes.OleDLL("shell32")

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    def _guid(s):
        g = GUID()
        ole32.CLSIDFromString(ctypes.c_wchar_p(s), ctypes.byref(g))
        return g

    def _vt(ptr, idx, restype, *argtypes):
        vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[idx])

    ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
    dlg = ctypes.c_void_p()
    item = ctypes.c_void_p()
    start_item = ctypes.c_void_p()
    try:
        clsid = _guid("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")   # CLSID_FileOpenDialog
        iid_dlg = _guid("{D57C7288-D4AD-4768-BE02-9D969532D960}") # IID_IFileOpenDialog
        iid_item = _guid("{43826D1E-E718-42EE-BC55-A1E261C37BFE}") # IID_IShellItem
        ole32.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid_dlg), ctypes.byref(dlg))
        # GetOptions (10) / SetOptions (9): FOS_PICKFOLDERS (0x20) | FOS_FORCEFILESYSTEM (0x40) | FOS_PATHMUSTEXIST (0x800)
        opts = wintypes.DWORD()
        _vt(dlg, 10, ctypes.c_long, ctypes.POINTER(wintypes.DWORD))(dlg, ctypes.byref(opts))
        _vt(dlg, 9, ctypes.c_long, wintypes.DWORD)(dlg, opts.value | 0x20 | 0x40 | 0x800)
        _vt(dlg, 17, ctypes.c_long, wintypes.LPCWSTR)(dlg, "Choose Image Folder")
        if start_dir and os.path.isdir(start_dir):
            try:
                shell32.SHCreateItemFromParsingName(ctypes.c_wchar_p(start_dir), None, ctypes.byref(iid_item), ctypes.byref(start_item))
                if start_item.value:
                    _vt(dlg, 12, ctypes.c_long, ctypes.c_void_p)(dlg, start_item)  # SetFolder
            except OSError:
                pass
        hr = _vt(dlg, 3, ctypes.c_long, wintypes.HWND)(dlg, owner)  # Show
        if hr == 0:
            _vt(dlg, 20, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))(dlg, ctypes.byref(item))  # GetResult
            if item.value:
                psz = ctypes.c_wchar_p()
                _vt(item, 5, ctypes.c_long, wintypes.DWORD, ctypes.POINTER(ctypes.c_wchar_p))(item, 0x80058000, ctypes.byref(psz))
                if psz.value:
                    chosen = psz.value
                    ole32.CoTaskMemFree(psz)
    finally:
        if start_item.value:
            _vt(start_item, 2, wintypes.ULONG)(start_item)
        if item.value:
            _vt(item, 2, wintypes.ULONG)(item)
        if dlg.value:
            _vt(dlg, 2, wintypes.ULONG)(dlg)
        ole32.CoUninitialize()
else:
    save = mode == "save"
    comdlg32 = ctypes.WinDLL("comdlg32")

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND),
                    ("hInstance", wintypes.HINSTANCE), ("lpstrFilter", wintypes.LPCWSTR),
                    ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
                    ("nFilterIndex", wintypes.DWORD), ("lpstrFile", wintypes.LPWSTR),
                    ("nMaxFile", wintypes.DWORD), ("lpstrFileTitle", wintypes.LPWSTR),
                    ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
                    ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD),
                    ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
                    ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
                    ("lpfnHook", wintypes.LPVOID), ("lpTemplateName", wintypes.LPCWSTR),
                    ("pvReserved", wintypes.LPVOID), ("dwReserved", wintypes.DWORD),
                    ("FlagsEx", wintypes.DWORD)]

    # a full path in the name field is the only way to beat the dialog's remembered folder;
    # an empty name must stay empty (a bare "D:\\dir\\" would fail with FNERR_INVALIDFILENAME)
    first = os.path.basename(name)
    if first and start_dir and os.path.isdir(start_dir):
        first = os.path.join(start_dir, first)
    buf = ctypes.create_unicode_buffer(first, 32768)
    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.hwndOwner = owner
    ofn.lpstrFilter = "Images\0*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff\0All files\0*.*\0\0"
    ofn.nFilterIndex = 1
    ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
    ofn.nMaxFile = len(buf)
    ofn.lpstrInitialDir = start_dir or None
    # OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST | (OFN_OVERWRITEPROMPT or OFN_FILEMUSTEXIST)
    ofn.Flags = 0x80000 | 0x8 | 0x800 | (0x2 if save else 0x1000)
    if save:
        ofn.lpstrDefExt = "png"
    ok = (comdlg32.GetSaveFileNameW if save else comdlg32.GetOpenFileNameW)(ctypes.byref(ofn))
    if ok:
        chosen = buf.value

user32.DestroyWindow(owner)
# shell extensions inside the dialog may print to stdout too: the result line is marked
sys.stdout.write("\nMMPATH:" + json.dumps(chosen) + "\n")
'''


def _win_file_dialog(start_dir: str, name: str, mode: str) -> str:
    """Windows open/save/folder dialog, in front, starting in start_dir."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-I", "-c", _WIN_DIALOG_SCRIPT, mode,
                        start_dir or "", name or ""],
                       capture_output=True, text=True, encoding="utf-8", timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"file dialog failed: {(r.stderr or '').strip()[-300:]}")
    lines = [x[7:] for x in (r.stdout or "").splitlines() if x.startswith("MMPATH:")]
    if not lines:
        raise RuntimeError("file dialog gave no answer")
    return json.loads(lines[-1])
