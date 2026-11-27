"""Unit tests for the MM Nodepack (pure helpers; no ComfyUI server needed).

Run:
  D:\\ComfyUI_windows_portable\\python_embeded\\python.exe -m pytest tests/test_mm_nodepack.py -q
  (fallback: D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe -m pytest tests/test_mm_nodepack.py -q)

Tests that depend on the V3 node API (comfy_api) are skipped when the interpreter
does not provide it (e.g. a plain venv without ComfyUI's site-packages).
"""
import json
import os
import sys
import threading
import types

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# some interpreters have the legacy `py` library in site-packages (or a stale import
# of it in sys.modules): make sure it cannot shadow the `py/` package of this pack
sys.modules.pop("py", None)
sys.path.insert(0, REPO_ROOT)
# in --import-mode=importlib pytest does not put the tests/ directory on
# sys.path, so make conftest_helpers importable explicitly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image, PngImagePlugin  # noqa: E402

from py import common as py_common  # noqa: E402
from py import paths as py_paths  # noqa: E402
from py.access_policy import shared_server  # noqa: E402
from py.common import (  # noqa: E402
    IMAGE_EXTS,
    LRUCache,
    AccessDeniedError,
    authorize,
    create_exclusive,
    list_folder_images,
    load_image_file,
    open_verified,
    preview_view_entry,
    save_tensor,
    view_entry,
)
from py.paths import (  # noqa: E402
    NoSourceError,
    find_origin,
    find_origin_files,
    last_dir_file,
    load_last_dir,
    next_target,
    pick_position,
    resolve_folder_image,
    resolve_save_target,
    resolve_source_like,
    resolve_targets,
    save_dir,
    save_last_dir,
)
from conftest_helpers import FolderPathsStub, make_folder_paths_stub  # noqa: E402

try:
    from py import image_io  # noqa: E402
    COMFY_AVAILABLE = True
except Exception:  # noqa: BLE001 – plain venv without ComfyUI's site-packages
    image_io = None
    COMFY_AVAILABLE = False

needs_comfy = pytest.mark.skipif(
    not COMFY_AVAILABLE, reason="comfy_api not available in this interpreter")


def _png(path, size=(8, 5), color=(200, 30, 30)):
    Image.new("RGB", size, color).save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# next_target / resolve_targets (the unified target resolution)
# ---------------------------------------------------------------------------

def test_next_target_default_and_free_number(tmp_path):
    src = _png(tmp_path / "haus.png")
    t1 = next_target(src)
    assert t1 == os.path.join(str(tmp_path), "haus_1.png"), f"next_target(src) -> {os.path.basename(t1)}"
    open(os.path.join(str(tmp_path), "haus_1.png"), "wb").close()
    t2 = next_target(src)
    assert t2 == os.path.join(str(tmp_path), "haus_2.png"), f"next free number -> {os.path.basename(t2)}"


def test_next_target_suffix_without_hash(tmp_path):
    src = _png(tmp_path / "haus.png")
    t3 = next_target(src, suffix="_edit")
    assert t3 == os.path.join(str(tmp_path), "haus_edit.png"), f"suffix _edit -> {os.path.basename(t3)}"
    open(t3, "wb").close()
    t4 = next_target(src, suffix="_edit")
    assert t4 == os.path.join(str(tmp_path), "haus_edit_2.png"), f"existing _edit -> {os.path.basename(t4)}"


def test_next_target_save_as(tmp_path):
    src = _png(tmp_path / "haus.png")
    assert next_target(src, save_as="neu") == os.path.join(str(tmp_path), "neu.png"), "save_as neu"
    assert next_target(src, save_as=os.path.join("sub", "x.jpg")) == os.path.join(str(tmp_path), "sub", "x.jpg"), "save_as sub/x.jpg"
    absa = os.path.join(str(tmp_path), "abs.png")
    assert next_target(src, save_as=absa) == absa, "absolute save_as used as-is"
    open(absa, "wb").close()
    assert next_target(src, save_as=absa) == os.path.join(str(tmp_path), "abs_2.png"), "existing absolute save_as"


def test_next_target_format(tmp_path):
    src = _png(tmp_path / "haus.png")
    t9 = next_target(src, fmt="jpg")
    assert t9.endswith(".jpg"), f"fmt jpg -> {os.path.basename(t9)}"
    webp = _png(tmp_path / "image.webp", size=(4, 4))
    t10 = next_target(webp)
    assert t10.endswith(".webp"), f"same as source webp -> {os.path.basename(t10)}"


def test_next_target_errors(tmp_path):
    with pytest.raises(ValueError):
        next_target("")
    with pytest.raises(ValueError):
        next_target(str(tmp_path / "missing.png"))


def test_unified_target_resolver_parity(tmp_path):
    # single (str API) and batch (dict API) must agree on the same (base, ext)
    src = _png(tmp_path / "haus.png")
    source = {"path": os.path.abspath(src), "name": "haus"}
    assert resolve_targets([object()], source, "_#", "", "same as source")[0] == next_target(src)
    assert resolve_targets([object()], source, "", "neu", "jpg")[0] == next_target(src, suffix="", save_as="neu", fmt="jpg")


def test_unified_target_resolver_batch_numbers(tmp_path):
    src = _png(tmp_path / "haus.png")
    source = {"path": os.path.abspath(src), "name": "haus"}
    targets = resolve_targets([0, 0, 0], source, "_#", "", "same as source")
    assert targets == [os.path.join(str(tmp_path), n) for n in ("haus_1.png", "haus_2.png", "haus_3.png")], \
        "batch reserves its own running numbers"
    for t in targets:  # none of them exists yet, never overwrites what comes next
        open(t, "wb").close()
    assert next_target(src) == os.path.join(str(tmp_path), "haus_4.png"), "written batch numbers are skipped"


def test_resolve_targets_no_source(tmp_path):
    d = str(tmp_path)
    outp = os.path.join(d, "output")
    os.makedirs(os.path.join(outp, "flux"))
    with FolderPathsStub(make_folder_paths_stub(os.path.join(d, "input"), outp, os.path.join(d, "temp"))):
        imgs = [torch.zeros(4, 4, 3)] * 2
        assert resolve_targets(imgs, None, "_#", "flux/image", "same as source") == \
            [os.path.join(outp, "flux", "image_1.png"), os.path.join(outp, "flux", "image_2.png")], \
            "no source: save_as as prefix -> output/flux/image_1,_2"
        assert resolve_targets(imgs[:1], None, "_#", "", "png") == [os.path.join(outp, "preview_1.png")], \
            "no source, no save_as -> output/preview_1"
        assert resolve_targets(imgs[:1], None, "_#", "../../x", "png") == [os.path.join(outp, "preview_1.png")], \
            "no source: no escape from output"


def test_resolve_source_like(tmp_path):
    d = str(tmp_path)
    inp = os.path.join(d, "input")
    batch = os.path.join(d, "batch")
    os.makedirs(os.path.join(inp, "sub"))
    os.makedirs(batch)
    _png(os.path.join(inp, "x.png"), (4, 4))
    _png(os.path.join(batch, "a.png"), (4, 4))
    _png(os.path.join(batch, "b.png"), (4, 4))
    own = _png(os.path.join(d, "own.png"), (4, 4))
    with FolderPathsStub(make_folder_paths_stub(inp, os.path.join(d, "output"), os.path.join(d, "temp"))):
        assert resolve_source_like({"path": own}) == os.path.abspath(own), "'path' wins"
        assert resolve_source_like({"mode": "folder", "folder": batch, "index": 2}) == os.path.abspath(os.path.join(batch, "b.png")), \
            "folder mode picks by index"
        assert resolve_source_like({"image": "x.png"}) == os.path.abspath(os.path.join(inp, "x.png")), \
            "upload combo via folder_paths"
        with pytest.raises(NoSourceError):
            resolve_source_like({})
        assert isinstance(NoSourceError("x"), ValueError)


def test_resolve_save_target_no_source(tmp_path):
    d = str(tmp_path)
    outp = os.path.join(d, "output")
    os.makedirs(outp)
    with FolderPathsStub(make_folder_paths_stub(os.path.join(d, "input"), outp, os.path.join(d, "temp"))):
        # no source, no save_as -> output/preview_1.png
        assert resolve_save_target("", "_#", "", "same as source") == os.path.join(outp, "preview_1.png")
        open(os.path.join(outp, "preview_1.png"), "wb").close()
        assert resolve_save_target("", "_#", "", "same as source") == os.path.join(outp, "preview_2.png"), \
            "no source: never overwrites preview_1"
        # no source, relative save_as -> a prefix inside the output dir (no escape)
        assert resolve_save_target("", "_#", "flux/image", "png") == os.path.join(outp, "flux", "image_1.png")
        # no source, absolute save_as with an image extension -> used as-is
        absa = os.path.join(d, "abs.png")
        assert resolve_save_target("", "", absa, "same as source") == absa
        open(absa, "wb").close()
        assert resolve_save_target("", "", absa, "same as source") == os.path.join(d, "abs_2.png"), \
            "absolute save_as never overwrites"
        # absolute save_as with a foreign extension keeps it and appends the target ext
        absx = os.path.join(d, "photo.txt")
        assert resolve_save_target("", "", absx, "same as source") == os.path.join(d, "photo.txt.png")
        assert resolve_save_target("", "", absx, "jpg") == os.path.join(d, "photo.txt.jpg")


def test_resolve_save_target_with_source(tmp_path):
    src = _png(tmp_path / "haus.png")
    assert resolve_save_target(src, "_#", "", "same as source") == next_target(src), \
        "with a source: identical to next_target"
    assert resolve_save_target(src, "_#", "neu", "jpg") == next_target(src, suffix="_#", save_as="neu", fmt="jpg")
    with pytest.raises(ValueError):
        resolve_save_target(str(tmp_path / "missing.png"), "_#", "", "same as source")


# ---------------------------------------------------------------------------
# upload origin / save_dir (fake folder_paths)
# ---------------------------------------------------------------------------

def test_save_dir_upload_and_originals(tmp_path):
    d = str(tmp_path)
    inp, outp = os.path.join(d, "input"), os.path.join(d, "output")
    os.makedirs(os.path.join(inp, "pasted"))
    os.makedirs(os.path.join(outp, "Moods"))
    os.makedirs(os.path.join(outp, "Other"))
    up = _png(os.path.join(inp, "pasted", "up.png"), (4, 4))
    own = _png(os.path.join(d, "haus.png"))
    with FolderPathsStub(make_folder_paths_stub(inp, outp, os.path.join(d, "temp"))):
        assert save_dir(up) == os.path.join(outp, "pasted"), "save_dir: upload -> output/pasted"
        assert next_target(up) == os.path.join(outp, "pasted", "up_1.png"), "next_target: upload -> output"
        assert save_dir(own) == os.path.abspath(d), "save_dir: own folder stays"
        # upload whose original lies in the output tree -> save next to the original (same content wins)
        _png(os.path.join(inp, "orig.png"), (4, 4), (1, 2, 3))
        _png(os.path.join(outp, "Moods", "orig.png"), (4, 4), (1, 2, 3))
        _png(os.path.join(outp, "Other", "orig.png"), (4, 4), (9, 9, 9))
        assert save_dir(os.path.join(inp, "orig.png")) == os.path.join(outp, "Moods"), \
            "save_dir: upload -> folder of the original (content match)"


def test_find_origin_uses_lru_cache(tmp_path):
    d = str(tmp_path)
    inp, outp = os.path.join(d, "input"), os.path.join(d, "output")
    os.makedirs(inp)
    os.makedirs(os.path.join(outp, "Moods"))
    os.makedirs(os.path.join(outp, "Other"))
    up = _png(os.path.join(inp, "orig.png"), (4, 4), (1, 2, 3))
    _png(os.path.join(outp, "Moods", "orig.png"), (4, 4), (1, 2, 3))
    _png(os.path.join(outp, "Other", "orig.png"), (4, 4), (9, 9, 9))
    with FolderPathsStub(make_folder_paths_stub(inp, outp, os.path.join(d, "temp"))):
        hits = find_origin_files(up, outp)
        assert hits == [os.path.join(outp, "Moods", "orig.png")], "only the identical copy matches"
        assert find_origin(up, outp) == os.path.join(outp, "Moods"), "find_origin -> folder of the original"
        again = find_origin_files(up, outp)  # second call hits the LRU cache
        assert again == hits and again is hits, "cached result returned unchanged"


# ---------------------------------------------------------------------------
# image load/save core
# ---------------------------------------------------------------------------

def test_save_tensor(tmp_path):
    arr = torch.rand(5, 8, 3)
    png = str(tmp_path / "out.png")
    save_tensor(arr, png)
    with Image.open(png) as im:
        assert im.size == (8, 5) and im.mode == "RGB", "save_tensor png readable, size (8,5)"
    jpg = str(tmp_path / "out.jpg")
    save_tensor(arr, jpg, quality=90)
    with Image.open(jpg) as im:
        assert im.size == (8, 5), "save_tensor jpg readable, size (8,5)"
    deep = str(tmp_path / "a" / "b" / "c.png")
    save_tensor(arr, deep)
    assert os.path.isfile(deep), "save_tensor creates missing folders"


def test_load_image_file(tmp_path):
    src = _png(tmp_path / "haus.png")
    img, mask = load_image_file(src)
    assert tuple(img.shape) == (1, 5, 8, 3) and img.dtype == torch.float32, "load_image_file shape/dtype"
    assert mask.shape[0] == 1, "load_image_file mask"


def test_image_exts():
    assert IMAGE_EXTS == (".png", ".jpg", ".jpeg", ".webp"), "IMAGE_EXTS"


# ---------------------------------------------------------------------------
# preview /view entries (fake folder_paths with a temp dir)
# ---------------------------------------------------------------------------

def test_preview_view_entry(tmp_path):
    d = str(tmp_path)
    inp2, outp2, tmp2 = os.path.join(d, "input2"), os.path.join(d, "output2"), os.path.join(d, "temp2")
    os.makedirs(os.path.join(outp2, "sub"))
    os.makedirs(inp2)
    os.makedirs(os.path.join(d, "external"))
    with FolderPathsStub(make_folder_paths_stub(inp2, outp2, tmp2)):
        out_img = _png(os.path.join(outp2, "sub", "shot.png"), (2000, 1000), (10, 20, 30))
        assert preview_view_entry(out_img) == {"filename": "shot.png", "subfolder": "sub", "type": "output"}, \
            "preview_view_entry: file in output -> view entry (output)"
        in_img = _png(os.path.join(inp2, "pick.png"), (2000, 1000), (10, 20, 30))
        assert preview_view_entry(in_img) == {"filename": "pick.png", "subfolder": "", "type": "input"}, \
            "preview_view_entry: file in input -> view entry (input)"
        ext_img = _png(os.path.join(d, "external", "big.png"), (2000, 1000), (10, 20, 30))
        e_ext = preview_view_entry(ext_img)
        assert e_ext is not None and e_ext["type"] == "temp" and e_ext["subfolder"] == "", \
            "preview_view_entry: external file -> temp entry"
        assert e_ext["filename"].startswith("mm_prev_") and e_ext["filename"].endswith(".jpg"), \
            f"preview_view_entry: cache name {e_ext['filename']}"
        cache = os.path.join(tmp2, e_ext["filename"])
        assert os.path.isfile(cache), "preview_view_entry: cached thumbnail created in temp"
        with Image.open(cache) as im:
            assert im.size == (1024, 512) and im.mode == "RGB", "preview_view_entry: thumbnail max side 1024 px, RGB"
        e_ext2 = preview_view_entry(ext_img)
        assert e_ext2 == e_ext, "preview_view_entry: second call reuses the cached filename"
        assert os.path.isfile(cache), "preview_view_entry: cached file still present after reuse"


# ---------------------------------------------------------------------------
# LRUCache eviction
# ---------------------------------------------------------------------------

def test_lru_cache_eviction_cap():
    c = LRUCache(3)
    for k in "abcd":
        c[k] = k
    assert len(c) == 3, "cache never exceeds maxsize"
    assert "a" not in c and "d" in c, "least recently used entry evicted"
    c["b"]  # touch: LRU order is now c, d, b
    c["e"] = 1
    assert "c" not in c and "b" in c and "d" in c, "access refreshes recency"
    assert len(c) == 3
    c.clear()
    assert len(c) == 0


def test_origin_cache_is_capped_lru():
    assert isinstance(py_paths._ORIGIN_CACHE, LRUCache), "_ORIGIN_CACHE is an LRUCache"
    assert py_paths._ORIGIN_CACHE.maxsize == 64, "_ORIGIN_CACHE is capped"


# ---------------------------------------------------------------------------
# last_dir memory: written outside the repository
# ---------------------------------------------------------------------------

def test_last_dir_written_outside_repo(tmp_path, monkeypatch):
    d = str(tmp_path)
    temp_dir = os.path.join(d, "temp")
    os.makedirs(temp_dir)
    with FolderPathsStub(make_folder_paths_stub(os.path.join(d, "in"), os.path.join(d, "out"), temp_dir)):
        save_last_dir(temp_dir)
        assert load_last_dir() == temp_dir, "last dir round-trip"
        assert last_dir_file() == os.path.join(temp_dir, "mm_nodepack_last_dir.json"), \
            "last_dir file lives in the temp dir (ComfyUI temp dir when available)"
        assert os.path.isfile(last_dir_file()), "last_dir file created in the temp dir"
    # without ComfyUI (no folder_paths installed) the OS temp dir is used:
    import tempfile
    monkeypatch.setattr(py_paths, "get_folder_paths", lambda: (_ for _ in ()).throw(ImportError("no comfy")))
    outside = last_dir_file()
    assert outside == os.path.join(os.path.abspath(tempfile.gettempdir()), "mm_nodepack_last_dir.json"), \
        "fallback last_dir file lives in the OS temp dir"
    assert not outside.startswith(REPO_ROOT), "last_dir file is never inside the repository"
    assert not os.path.exists(os.path.join(REPO_ROOT, "last_dir.json")), \
        "no leftover last_dir.json in the repository root"


def test_last_dir_per_mode_and_legacy_migration(tmp_path):
    d = str(tmp_path)
    temp_dir = os.path.join(d, "temp")
    os.makedirs(temp_dir)
    legacy = os.path.join(temp_dir, "mm_nodepack_last_dir.json")
    with FolderPathsStub(make_folder_paths_stub(os.path.join(d, "in"), os.path.join(d, "out"), temp_dir)):
        # per-mode round-trip: the three modes are kept separately
        save_last_dir(temp_dir, "open")
        save_last_dir(os.path.join(d, "save_dir"), "save")
        save_last_dir(os.path.join(d, "folder_dir"), "folder")
        assert load_last_dir("open") == temp_dir, "open mode keeps its own dir"
        assert load_last_dir("save") == os.path.join(d, "save_dir"), "save mode keeps its own dir"
        assert load_last_dir("folder") == os.path.join(d, "folder_dir"), "folder mode keeps its own dir"
        # a legacy file (one shared 'dir' entry) answers for every mode
        with open(legacy, "w") as f:
            json.dump({"dir": os.path.join(d, "old")}, f)
        assert load_last_dir("open") == os.path.join(d, "old"), "legacy 'dir' answers for open"
        assert load_last_dir("folder") == os.path.join(d, "old"), "legacy 'dir' answers for folder"
        # saving migrates: the legacy dir seeds every mode exactly once, 'dir' disappears
        save_last_dir(os.path.join(d, "new_save"), "save")
        with open(legacy) as f:
            migrated = json.load(f)
        assert "dir" not in migrated, "legacy 'dir' key is dropped on migration"
        assert migrated == {"open": os.path.join(d, "old"),
                            "save": os.path.join(d, "new_save"),
                            "folder": os.path.join(d, "old")}, \
            f"migration seeds all modes from the legacy dir: {migrated}"
        assert load_last_dir("open") == os.path.join(d, "old"), "migrated open value readable"
        assert load_last_dir("folder") == os.path.join(d, "old"), "migrated folder value readable"


# ---------------------------------------------------------------------------
# folder batch helpers
# ---------------------------------------------------------------------------

def _make_batch_dir(base):
    batch = os.path.join(str(base), "batch")
    os.makedirs(os.path.join(batch, "nested"))
    for fname in ("img_10.png", "img_2.png", "img_1.jpg"):
        _png(os.path.join(batch, fname), (4, 4))
    _png(os.path.join(batch, "nested", "ignored.png"), (4, 4))
    open(os.path.join(batch, "notes.txt"), "w").close()
    return batch


def test_list_folder_images(tmp_path):
    batch = _make_batch_dir(tmp_path)
    listed = list_folder_images(batch)
    assert [os.path.basename(x) for x in listed] == ["img_1.jpg", "img_2.png", "img_10.png"], \
        "list_folder_images: natural sort, filters non-images and subdirs"


def test_resolve_folder_image(tmp_path):
    batch = _make_batch_dir(tmp_path)
    p1, pos1, cnt1 = resolve_folder_image(batch, batch_mode=False, index=1)
    p2, pos2, _ = resolve_folder_image(batch, batch_mode=False, index=2)
    p3, pos3, _ = resolve_folder_image(batch, batch_mode=False, index=3)
    p4, pos4, cnt4 = resolve_folder_image(batch, batch_mode=False, index=4)
    p0, pos0, _ = resolve_folder_image(batch, batch_mode=False, index=0)
    assert os.path.basename(p1) == "img_1.jpg" and pos1 == 0 and cnt1 == 3, "incremental 1-based: index=1 -> first image"
    assert os.path.basename(p2) == "img_2.png" and pos2 == 1, "incremental 1-based: index=2 -> second image"
    assert os.path.basename(p3) == "img_10.png" and pos3 == 2, "incremental 1-based: index=3 -> third image"
    assert os.path.basename(p4) == "img_1.jpg" and pos4 == 0 and cnt4 == 3, "incremental: index > count wraps to first"
    assert os.path.basename(p0) == "img_1.jpg" and pos0 == 0, "index=0 falls back to first image"
    r1, _, _ = resolve_folder_image(batch, batch_mode=True, index=999999)
    r2, _, _ = resolve_folder_image(batch, batch_mode=True, index=999999)
    assert r1 == r2 and r1 in list_folder_images(batch), "random: large seed (> count) deterministic, valid image"
    empty = os.path.join(str(tmp_path), "empty_dir")
    os.makedirs(empty)
    with pytest.raises(ValueError):
        resolve_folder_image(empty, batch_mode=False, index=0)


# ---------------------------------------------------------------------------
# wireless source lookup + V3 schemas (needs comfy_api)
# ---------------------------------------------------------------------------

@needs_comfy
def test_find_loader(tmp_path):
    src = _png(tmp_path / "haus.png")
    L = {"class_type": "MMLoadImage", "inputs": {"image": "x.png", "path": src}}
    pr = {"5": L, "6": {"class_type": "ImageInvert", "inputs": {"image": ["5", 0]}},
          "7": {"class_type": "MMPreviewSave", "inputs": {"images": ["6", 0]}}}
    assert image_io.find_loader(pr, "7", "auto") == ("5", ""), "auto: loader upstream"
    pr2 = dict(pr, **{"9": dict(L), "8": {"class_type": "MMPreviewSave", "inputs": {"images": ["6", 0]}}})
    assert image_io.find_loader(pr2, "8", "auto")[0] == "5", "auto: upstream wins over a second loader"
    pr3 = dict(pr2, **{"10": {"class_type": "EmptyImage", "inputs": {}},
                       "11": {"class_type": "MMPreviewSave", "inputs": {"images": ["10", 0]}}})
    lid, note = image_io.find_loader(pr3, "11", "auto")
    assert lid is None and "#5" in note and "#9" in note, f"auto: ambiguous -> note '{note}'"
    assert image_io.find_loader(pr3, "11", "#9") == ("9", ""), "select #9"
    assert image_io.find_loader({"3:9": L, "4": {"class_type": "X", "inputs": {}}}, "4", "9")[0] == "3:9", \
        "select inside subgraph"
    assert image_io.find_loader(pr3, "11", "#42")[0] is None, "select unknown number -> None"
    one = {"5": L, "10": {"class_type": "EmptyImage", "inputs": {}},
           "11": {"class_type": "MMPreviewSave", "inputs": {"images": ["10", 0]}}}
    assert image_io.find_loader(one, "11", "auto")[0] == "5", "auto: only loader in the workflow"


@needs_comfy
def test_source_from_prompt(tmp_path):
    src = _png(tmp_path / "haus.png")
    pr = {"5": {"class_type": "MMLoadImage", "inputs": {"image": "x.png", "path": src}}}
    assert image_io.source_from_prompt(pr, "5")["path"] == os.path.abspath(src), "source_from_prompt path"
    linked = {"5": {"class_type": "MMLoadImage", "inputs": {"image": "x.png", "path": ["2", 0]}}}
    assert image_io.source_from_prompt(linked, "5") is None, "source_from_prompt: linked path -> None"
    batch = _make_batch_dir(tmp_path)
    folder = {"5": {"class_type": "MMLoadImage", "inputs": {
        "image": "", "path": "", "mode": "folder", "folder": batch, "batch_mode": False, "index": 1}}}
    assert os.path.basename(image_io.source_from_prompt(folder, "5")["path"]) == "img_1.jpg", \
        "source_from_prompt: folder mode index=1 -> first image"
    folder["5"]["inputs"]["index"] = 2
    assert os.path.basename(image_io.source_from_prompt(folder, "5")["path"]) == "img_2.png", \
        "source_from_prompt: folder mode index=2 -> second image"


@needs_comfy
def test_schemas():
    loader_schema = image_io.MMLoadImage.define_schema()
    loader_inputs = [i.id for i in loader_schema.inputs]
    assert loader_inputs == ["mode", "image", "path", "folder", "batch_mode", "index"], \
        f"loader inputs: all six in required, in this order: {loader_inputs}"
    assert all(i.optional is False for i in loader_schema.inputs), "loader inputs are all required (no optional)"
    loader_outputs = [o.id for o in loader_schema.outputs]
    assert loader_outputs == ["image", "mask", "width", "height", "path", "image_list", "index"], \
        f"loader outputs match expected order without 'sync': {loader_outputs}"
    assert loader_schema.outputs[5].is_output_list is True, "loader output 'image_list' has is_output_list=True"
    assert loader_schema.display_name == "Load Image MM" and loader_schema.category == "MM Nodepack"
    assert image_io._output_slot_linked({"5": {}, "8": {"inputs": {"images": ["5", 5]}}}, "5", 5) is True, \
        "_output_slot_linked detects linked image_list slot"
    assert image_io._output_slot_linked({"5": {}, "8": {"inputs": {"images": ["5", 0]}}}, "5", 5) is False, \
        "_output_slot_linked returns False when image_list is unconnected"
    assert image_io._output_slot_linked({"3:42": {}, "8": {"inputs": {"images": ["42", 5]}}}, "3:42", 5) is True, \
        "_output_slot_linked: subgraph key matched by the plain node number"
    assert image_io._output_slot_linked({"42": {}, "8": {"inputs": {"images": ["3:42", 5]}}}, "42", 5) is True, \
        "_output_slot_linked: plain key matched by the subgraph-prefixed link"
    assert image_io._output_slot_linked({"3:42": {}, "8": {"inputs": {"images": ["42", 0]}}}, "3:42", 5) is False, \
        "_output_slot_linked: wrong slot stays False"
    preview = image_io.MMPreviewSave.define_schema()
    preview_inputs = [i.id for i in preview.inputs]
    # exact set: no legacy 'sync' and no workflow-embedding toggle (PNG always embeds)
    assert preview_inputs == ["images", "auto_save", "auto_send", "source", "suffix", "save_as", "format", "quality"], \
        f"preview schema must expose exactly these 8 inputs in this order: {preview_inputs}"
    assert preview.display_name == "Preview & Save MM" and preview.category == "MM Nodepack"
    assert preview.node_id == "MMPreviewSave", "preview node_id is MMPreviewSave"


@needs_comfy
def test_node_list():
    assert image_io.NODE_LIST == [image_io.MMLoadImage, image_io.MMPreviewSave], "NODE_LIST exports both nodes"


# ---------------------------------------------------------------------------
# MMLoadImage.execute: image_list stays in natural order (random mode included)
# ---------------------------------------------------------------------------

@needs_comfy
def test_loader_image_list_natural_order_random_mode(tmp_path, monkeypatch):
    """image_list is always the natural file order, even in random mode, so
    image_list[index - 1] is exactly the loaded image in every mode."""
    batch = os.path.join(str(tmp_path), "batch")
    os.makedirs(batch)
    for fname, color in (("img_1.png", (255, 0, 0)), ("img_2.png", (0, 255, 0)), ("img_10.png", (0, 0, 255))):
        _png(os.path.join(batch, fname), (4, 4), color)
    # slot 5 (image_list) linked -> execute returns the whole list
    linked = {"5": {"class_type": "MMLoadImage", "inputs": {}},
              "8": {"class_type": "X", "inputs": {"images": ["5", 5]}}}
    monkeypatch.setattr(image_io.MMLoadImage, "hidden",
                        types.SimpleNamespace(unique_id="5", prompt=linked))
    monkeypatch.setattr(image_io, "_preview_entries", lambda tensors, cls: [])
    try:
        out = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                           batch_mode=True, index=42)
        img, mask, w, h, path, image_list, idx = out.args
        files = list_folder_images(batch)
        assert [os.path.basename(f) for f in files] == ["img_1.png", "img_2.png", "img_10.png"], "natural sort"
        assert path in files, "random pick is a file of the folder"
        pos = files.index(path)
        assert len(image_list) == len(files), "image_list holds every image of the folder"
        assert image_list[0][0, 0, 0, 0] == pytest.approx(1.0), "image_list[0] is img_1 (red)"
        assert image_list[1][0, 0, 0, 1] == pytest.approx(1.0), "image_list[1] is img_2 (green)"
        assert image_list[2][0, 0, 0, 2] == pytest.approx(1.0), "image_list[2] is img_10 (blue)"
        assert image_list[pos] is img, "image_list[index - 1] is exactly the loaded image"
        assert idx == pos + 1, "the index output matches the natural position of the loaded image"
        # deterministic seed: the same index always picks the same file
        again = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                             batch_mode=True, index=42)
        assert again.args[4] == path, "random: same seed -> same file"
        # unlinked slot: the list stays the single loaded image
        unlinked = {"5": {"class_type": "MMLoadImage", "inputs": {}},
                    "8": {"class_type": "X", "inputs": {"images": ["5", 0]}}}
        monkeypatch.setattr(image_io.MMLoadImage, "hidden",
                            types.SimpleNamespace(unique_id="5", prompt=unlinked))
        out2 = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                            batch_mode=False, index=2)
        img2, *_rest2 = out2.args
        assert len(out2.args[5]) == 1, "unlinked image_list stays the single loaded image"
        assert out2.args[4] == os.path.join(batch, "img_2.png"), "incremental index=2 -> second image"
        assert out2.args[5][0] is img2, "the single-element list is the loaded image"
    finally:
        image_io._LAST_RESOLVED.pop("5", None)


# ---------------------------------------------------------------------------
# source_from_prompt: subgraph-prefixed loader IDs (_LAST_RESOLVED fallback)
# ---------------------------------------------------------------------------

@needs_comfy
def test_source_from_prompt_subgraph_id_fallback(tmp_path):
    src = os.path.abspath(str(_png(tmp_path / "haus.png")))
    # folder input wired to another node: no plain folder string to resolve from
    prompt = {"3:7": {"class_type": "MMLoadImage",
                      "inputs": {"mode": "folder", "folder": ["1", 0]}}}
    try:
        # direct hit: execute stored under the full subgraph prompt key
        image_io._LAST_RESOLVED["3:7"] = src
        s = image_io.source_from_prompt(prompt, "3:7")
        assert s is not None and s["path"] == src, "direct _LAST_RESOLVED hit by subgraph key"
        # fallback: stored under the plain node number, looked up by the subgraph key
        del image_io._LAST_RESOLVED["3:7"]
        image_io._LAST_RESOLVED["7"] = src
        s = image_io.source_from_prompt(prompt, "3:7")
        assert s is not None and s["path"] == src, "fallback via the plain node number"
    finally:
        image_io._LAST_RESOLVED.pop("3:7", None)
        image_io._LAST_RESOLVED.pop("7", None)


# ---------------------------------------------------------------------------
# MMPreviewSave: _save_batch always embeds the PNG workflow metadata
# ---------------------------------------------------------------------------

@needs_comfy
def test_save_batch_png_always_embeds_metadata(tmp_path):
    src = _png(tmp_path / "haus.png")
    img, _mask = load_image_file(src)
    one = img[0]
    target = str(tmp_path / "saved.png")
    image_io._save_batch([one], [target], "png", 95,
                         {"1": {"class_type": "X", "inputs": {}}}, {"extra": 1})
    with Image.open(target) as im:
        assert "prompt" in im.info, "PNG save embeds the prompt metadata"
        assert "extra" in im.info, "PNG save embeds the extra_pnginfo keys"
    target2 = str(tmp_path / "saved2.png")
    image_io._save_batch([one], [target2], "png", 95, None, None)
    with Image.open(target2) as im:
        assert "prompt" not in im.info, "no metadata only when prompt and extra_pnginfo are both None"
    jpg = str(tmp_path / "saved.jpg")
    image_io._save_batch([one], [jpg], "jpg", 95, {"1": {}}, None)
    with Image.open(jpg) as im:
        assert "prompt" not in im.info, "jpg never carries PNG text chunks"


@needs_comfy
def test_make_source(tmp_path):
    src = _png(tmp_path / "haus.png")
    s = image_io.make_source(src)
    assert s["path"] == os.path.abspath(src), "make_source path"
    assert s["width"] == 8 and s["height"] == 5, "make_source width/height"
    assert s["format"] == ".png", "make_source format"
    assert s["name"] == "haus", "make_source name"
    assert tuple(s["image"].shape) == (1, 5, 8, 3), "make_source image tensor"
    assert "mask" in s and "path" in s and "image" in s, "make_source keys"


# ---------------------------------------------------------------------------
# MMLoadImage.validate_inputs (mode-dependent validation, needs comfy_api)
# ---------------------------------------------------------------------------

@needs_comfy
def test_validate_inputs_modes(tmp_path):
    batch = _make_batch_dir(tmp_path)
    src = _png(tmp_path / "haus.png")
    # folder mode: only the folder is validated, image is ignored (empty or set)
    assert image_io.MMLoadImage.validate_inputs(image="", mode="folder", folder=batch) is True
    assert image_io.MMLoadImage.validate_inputs(image="whatever.png", mode="folder", folder=batch) is True
    assert isinstance(image_io.MMLoadImage.validate_inputs(mode="folder", folder=""), str), \
        "folder mode with an empty folder -> error string"
    # single mode: a valid path wins over an empty image
    assert image_io.MMLoadImage.validate_inputs(image="", path=src, mode="single") is True
    # single mode: path set but missing -> error
    assert isinstance(image_io.MMLoadImage.validate_inputs(image="", path=str(tmp_path / "missing.png"), mode="single"), str), \
        "single mode: missing path file -> error string"
    # single mode: empty path and no image -> clear error
    assert isinstance(image_io.MMLoadImage.validate_inputs(image="", path="", mode="single"), str), \
        "single mode with neither path nor image -> error string"
    # single mode: empty path, upload name that does not exist -> error
    assert isinstance(image_io.MMLoadImage.validate_inputs(image="mm-test-no-such-file-12345.png", path="", mode="single"), str), \
        "single mode: unknown upload name -> error string"


# ---------------------------------------------------------------------------
# Windows dialog script: the comdlg32 lpstrFile must stay empty for an empty name
# (a bare "D:\\dir\\" makes GetOpenFileNameW/GetSaveFileNameW fail with
# FNERR_INVALIDFILENAME). The logic is embedded in the child-process script,
# so verify the guard in its source.
# ---------------------------------------------------------------------------

def test_win_dialog_script_empty_name_guard():
    from py import dialog
    lines = [l.strip() for l in dialog._WIN_DIALOG_SCRIPT.splitlines()]
    i = lines.index("first = os.path.basename(name)")
    assert lines[i + 1] == "if first and start_dir and os.path.isdir(start_dir):", \
        f"lpstrFile start_dir join must be guarded by a non-empty name: {lines[i + 1]}"
    assert lines[i + 2] == "first = os.path.join(start_dir, first)", "guard must protect the join itself"
    assert "COM" not in "".join(lines[i:i + 3]), "the name logic must not touch the IFileDialog vtable"


def test_save_tensor_accepts_single_batch_4d_tensor(tmp_path):
    """Regression: /mm/image/save loads a [1,H,W,C] tensor; save_tensor must handle both 3D and single-batch 4D."""
    src = _png(tmp_path / "in.png")
    img, _ = load_image_file(src)
    assert img.ndim == 4 and img.shape[0] == 1
    out3 = save_tensor(img[0], str(tmp_path / "out3.png"))
    out4 = save_tensor(img[:1], str(tmp_path / "out4.png"))
    assert os.path.isfile(out3) and os.path.isfile(out4)


# ---------------------------------------------------------------------------
# access_policy: shared-server switch (env override, --listen, missing comfy)
# ---------------------------------------------------------------------------

def _fake_cli_args(monkeypatch, listen):
    """Install a fake comfy.cli_args module with a fixed --listen value."""
    comfy_mod = types.ModuleType("comfy")
    cli_mod = types.ModuleType("comfy.cli_args")
    cli_mod.args = types.SimpleNamespace(listen=listen)
    comfy_mod.cli_args = cli_mod
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli_mod)


def test_access_policy_env_values(monkeypatch):
    monkeypatch.delenv("MM_NODEPACK_SHARED_SERVER", raising=False)
    for v in ("1", "true", "yes", "on", "TRUE", "Yes", " ON "):
        monkeypatch.setenv("MM_NODEPACK_SHARED_SERVER", v)
        assert shared_server() is True, f"env {v!r} must enable the policy"
    for v in ("0", "false", "no", "off", "FALSE", "No", " off "):
        monkeypatch.setenv("MM_NODEPACK_SHARED_SERVER", v)
        assert shared_server() is False, f"env {v!r} must disable the policy"


def test_access_policy_env_overrides_listen(monkeypatch):
    _fake_cli_args(monkeypatch, "0.0.0.0")  # non-loopback listen
    monkeypatch.setenv("MM_NODEPACK_SHARED_SERVER", "0")
    assert shared_server() is False, "an explicit env 'off' wins over a non-loopback listen"
    _fake_cli_args(monkeypatch, "127.0.0.1")  # loopback listen
    monkeypatch.setenv("MM_NODEPACK_SHARED_SERVER", "1")
    assert shared_server() is True, "an explicit env 'on' wins over a loopback listen"


def test_access_policy_listen(monkeypatch):
    monkeypatch.delenv("MM_NODEPACK_SHARED_SERVER", raising=False)
    _fake_cli_args(monkeypatch, "127.0.0.1")
    assert shared_server() is False, "loopback listen -> private"
    _fake_cli_args(monkeypatch, "::1")
    assert shared_server() is False, "IPv6 loopback -> private"
    _fake_cli_args(monkeypatch, "localhost")
    assert shared_server() is False, "localhost -> private"
    _fake_cli_args(monkeypatch, "0.0.0.0")
    assert shared_server() is True, "non-loopback listen -> shared"
    _fake_cli_args(monkeypatch, "127.0.0.1,0.0.0.0")
    assert shared_server() is True, "any non-loopback entry of the list counts"
    _fake_cli_args(monkeypatch, "127.0.0.1,::1")
    assert shared_server() is False, "an all-loopback list stays private"


def test_access_policy_missing_comfy(monkeypatch):
    monkeypatch.delenv("MM_NODEPACK_SHARED_SERVER", raising=False)
    # a comfy module without the cli_args submodule: the lazy import must fail
    # back to the private behaviour instead of raising
    monkeypatch.setitem(sys.modules, "comfy", types.ModuleType("comfy"))
    assert shared_server() is False, "unreadable ComfyUI runtime -> private"
    # cli_args present but without a listen attribute (and with an empty one)
    _fake_cli_args(monkeypatch, "")
    assert shared_server() is False, "empty --listen stays private"


# ---------------------------------------------------------------------------
# authorize: the policy gate (roots, links, case, bounded errors)
# ---------------------------------------------------------------------------

def _policy_on(monkeypatch):
    monkeypatch.setattr(py_common, "policy_active", lambda: True)


def test_authorize_read_write_roots(tmp_path, monkeypatch):
    d = str(tmp_path)
    inp, outp = os.path.join(d, "input"), os.path.join(d, "output")
    os.makedirs(os.path.join(inp, "nested", "deep"))
    os.makedirs(os.path.join(outp, "nested"))
    _policy_on(monkeypatch)
    with FolderPathsStub(make_folder_paths_stub(inp, outp, os.path.join(d, "temp"))):
        authorize(os.path.join(inp, "nested", "deep", "x.png"), "read")   # nested under input
        authorize(os.path.join(outp, "nested", "y.png"), "read")         # nested under output
        authorize(os.path.join(outp, "nested", "z.png"), "write")        # write under output
        with pytest.raises(AccessDeniedError):
            authorize(os.path.dirname(inp), "read")            # the resolved parent of a root
        with pytest.raises(AccessDeniedError):
            authorize(os.path.join(d, "outside.png"), "read")  # outside every read root
        with pytest.raises(AccessDeniedError):
            authorize(os.path.join(d, "outside.png"), "write")  # write outside the output root


def test_authorize_error_message_never_names_the_path(tmp_path, monkeypatch):
    d = str(tmp_path)
    inp, outp = os.path.join(d, "input"), os.path.join(d, "output")
    os.makedirs(inp)
    os.makedirs(outp)
    secret = os.path.join(d, "SECRET_DIR", "top-secret-file.png")
    _policy_on(monkeypatch)
    with FolderPathsStub(make_folder_paths_stub(inp, outp, os.path.join(d, "temp"))):
        try:
            authorize(secret, "read")
            pytest.fail("authorize must refuse paths outside the read roots")
        except AccessDeniedError as e:
            msg = str(e)
        assert "SECRET_DIR" not in msg and "top-secret-file" not in msg, f"error message leaks the path: {msg!r}"
        assert "secret" not in msg.lower(), f"error message leaks the path: {msg!r}"
        assert msg == "path outside the allowed folders"
        try:
            authorize(secret, "write")
            pytest.fail("authorize must refuse writes outside the output root")
        except AccessDeniedError as e:
            assert "secret" not in str(e).lower() and str(e) == "target outside the allowed output folder"


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive containment is a Windows property")
def test_authorize_case_insensitive_paths(tmp_path, monkeypatch):
    d = str(tmp_path)
    inp, outp = os.path.join(d, "input"), os.path.join(d, "output")
    os.makedirs(os.path.join(inp, "dir"))
    os.makedirs(outp)
    _policy_on(monkeypatch)
    with FolderPathsStub(make_folder_paths_stub(inp, outp, os.path.join(d, "temp"))):
        # a case-different spelling of the same location is the same path on
        # case-insensitive file systems (a missing write target keeps the caller's case)
        authorize(os.path.join(inp, "DIR", "FILE.png"), "read")
        authorize(os.path.join(outp, "NESTED", "Z.png"), "write")


def _make_dir_link(path, target):
    """Create a directory symlink/junction `path` -> `target`; skip the test when
    the platform/privileges do not allow it."""
    try:
        os.symlink(target, path, target_is_directory=True)
    except OSError:
        if os.name == "nt":
            import subprocess
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(path), str(target)],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isdir(path):
                pytest.skip(f"symlink/junction cannot be created here: {r.stderr.strip()[:120]}")
        else:
            pytest.skip("directory symlinks cannot be created here")


def test_authorize_symlink_escaping_root_denied(tmp_path, monkeypatch):
    d = str(tmp_path)
    inp, outp = os.path.join(d, "input"), os.path.join(d, "output")
    outside = os.path.join(d, "outside")
    os.makedirs(os.path.join(inp, "dir"))
    os.makedirs(outp)
    os.makedirs(outside)
    _png(os.path.join(outside, "secret.png"), (2, 2))
    _png(os.path.join(inp, "x.png"), (2, 2))
    escape = os.path.join(inp, "dir", "escape")
    _make_dir_link(escape, outside)   # skips when links are impossible
    evil = os.path.join(outp, "evil")
    _make_dir_link(evil, outside)
    _policy_on(monkeypatch)
    with FolderPathsStub(make_folder_paths_stub(inp, outp, os.path.join(d, "temp"))):
        with pytest.raises(AccessDeniedError):
            authorize(os.path.join(escape, "secret.png"), "read")   # read link leaving the root
        with pytest.raises(AccessDeniedError):
            authorize(os.path.join(evil, "secret.png"), "write")    # write link leaving the root
        # a link that stays inside the root is allowed
        inner = os.path.join(inp, "inner")
        _make_dir_link(inner, inp)
        authorize(os.path.join(inner, "x.png"), "read")


# ---------------------------------------------------------------------------
# route handlers: restricted (non-local) requests are checked before any disk
# work; trusted local requests keep the unrestricted behaviour
# ---------------------------------------------------------------------------

class _CallSpy:
    """Monkeypatch target that records every call: a call fails the test."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError(f"{self.name} must not be called in a rejected request")


def _route_dirs(tmp_path, monkeypatch):
    """Real ComfyUI-like folders with a PRIVATE server (policy inactive): the
    restricted=True cases exercise the forced (non-local request) gates."""
    d = str(tmp_path)
    inp, outp, temp = os.path.join(d, "input"), os.path.join(d, "output"), os.path.join(d, "temp")
    os.makedirs(os.path.join(inp, "sub"))
    os.makedirs(outp)
    os.makedirs(temp)
    monkeypatch.setattr(py_common, "policy_active", lambda: False)
    return d, inp, outp, temp


def test_route_restricted_rejected_never_touches_disk(tmp_path, monkeypatch):
    d, inp, outp, temp = _route_dirs(tmp_path, monkeypatch)
    outside = _png(os.path.join(d, "outside.png"), (4, 4))
    outside_dir = os.path.join(d, "outside_dir")
    os.makedirs(outside_dir)
    evil = _png(os.path.join(d, "evil.png"), (4, 4))
    fp = make_folder_paths_stub(inp, outp, temp)
    fp.get_annotated_filepath = (lambda name: evil if name == "evil.png" else os.path.join(inp, name))
    # the scanner, the decoder and the writer must never run for a rejected request
    spy_scandir = _CallSpy("os.scandir")
    spy_open = _CallSpy("Image.open")
    spy_save = _CallSpy("save_tensor")
    monkeypatch.setattr(py_common.os, "scandir", spy_scandir)
    monkeypatch.setattr(Image, "open", spy_open)
    monkeypatch.setattr(image_io, "save_tensor", spy_save)
    with FolderPathsStub(fp):
        _png(os.path.join(temp, "prev.png"), (4, 4))  # a valid preview file for the save case
        st, p = image_io.handle_preview({"path": outside}, restricted=True)
        assert st == 403 and p["ok"] is False, f"restricted preview outside the roots -> 403: {st} {p}"
        st, p = image_io.handle_preview({"mode": "folder", "folder": outside_dir}, restricted=True)
        assert st == 403 and p["ok"] is False, f"restricted folder outside the roots -> 403: {st} {p}"
        st, p = image_io.handle_target({"path": outside}, restricted=True)
        assert st == 403 and p["ok"] is False, f"restricted target outside the roots -> 403: {st} {p}"
        st, p = image_io.handle_save({"type": "temp", "filename": "prev.png", "source": outside}, restricted=True)
        assert st == 403 and p["ok"] is False, f"restricted save from an outside source -> 403: {st} {p}"
        st, p = image_io.handle_resolve({"image": "evil.png"}, restricted=True)
        assert st == 403 and p["ok"] is False, f"restricted resolve outside the roots -> 403: {st} {p}"
        for spy in (spy_scandir, spy_open, spy_save):
            assert not spy.calls, f"{spy.name} was called in a rejected request: {spy.calls}"
        # bounded error text: no external path in the reply
        assert "outside.png" not in str(p)


def test_route_restricted_allowed_flows(tmp_path, monkeypatch):
    d, inp, outp, temp = _route_dirs(tmp_path, monkeypatch)
    _png(os.path.join(inp, "up.png"), (4, 4), (9, 9, 9))   # remote upload in input
    _png(os.path.join(temp, "prev.png"), (4, 4), (1, 2, 3))
    with FolderPathsStub(make_folder_paths_stub(inp, outp, temp)):
        # remote upload in input -> preview OK (restricted)
        st, p = image_io.handle_preview({"image": "up.png"}, restricted=True)
        assert st == 200 and p["ok"] is True
        assert p["view"] == {"filename": "up.png", "subfolder": "", "type": "input"}
        # source-less save (restricted) lands under the output root
        st, p = image_io.handle_save({"type": "temp", "filename": "prev.png", "format": "png"}, restricted=True)
        assert st == 200 and p["ok"] is True, f"source-less restricted save -> 200: {st} {p}"
        assert os.path.isfile(p["path"])
        assert os.path.commonpath([os.path.realpath(outp), os.path.realpath(p["path"])]) == os.path.realpath(outp), \
            "the restricted save must land inside the output root"
        # trusted local (loopback) requests keep the unrestricted behaviour:
        # an arbitrary path outside the ComfyUI tree still works
        own = _png(os.path.join(d, "own.png"), (4, 4))
        st, p = image_io.handle_target({"path": own}, restricted=False)
        assert st == 200 and p["ok"] is True, f"trusted local target next to an outside source -> 200: {st} {p}"
        assert os.path.dirname(p["target"]) == os.path.abspath(d)


# ---------------------------------------------------------------------------
# view_entry behind a junction/symlinked output root
# ---------------------------------------------------------------------------

def test_view_entry_symlinked_output_root(tmp_path):
    d = str(tmp_path)
    outp = os.path.join(d, "output")
    os.makedirs(os.path.join(outp, "sub"))
    _png(os.path.join(outp, "sub", "a.png"), (4, 4))
    _png(os.path.join(outp, "a2.png"), (4, 4))
    linked = os.path.join(d, "linked_out")
    _make_dir_link(linked, outp)   # skips when links are impossible
    with FolderPathsStub(make_folder_paths_stub(os.path.join(d, "input"), outp, os.path.join(d, "temp"))):
        e = view_entry(os.path.join(linked, "sub", "a.png"))
        assert e == {"filename": "a.png", "subfolder": "sub", "type": "output"}, \
            f"behind a linked output root the subfolder stays inside the tree (no '..'): {e}"
        e2 = view_entry(os.path.join(linked, "a2.png"))
        assert e2 == {"filename": "a2.png", "subfolder": "", "type": "output"}, \
            f"linked root file: {e2}"


# ---------------------------------------------------------------------------
# save_tensor: exclusive create (concurrent saves), metadata, no partial files
# ---------------------------------------------------------------------------

def test_save_tensor_concurrent_saves_same_target(tmp_path):
    arr = torch.rand(5, 8, 3)
    target = str(tmp_path / "race.png")
    results, errors = [None, None], []

    def worker(i):
        try:
            results[i] = save_tensor(arr, target)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"both concurrent saves to the same target must succeed: {errors}"
    assert results[0] is not None and results[1] is not None
    assert os.path.abspath(results[0]) != os.path.abspath(results[1]), \
        "the race loser retries with the next free number -> distinct files"
    for p in results:
        assert os.path.isfile(p), "both saves must exist on disk"
        with Image.open(p) as im:
            assert im.size == (8, 5), "both files are complete, readable images"


def test_save_tensor_png_metadata_roundtrip(tmp_path):
    arr = torch.rand(5, 8, 3)
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", '{"1": {"class_type": "X"}}')
    info.add_text("extra", '{"workflow": {}}')
    p = save_tensor(arr, str(tmp_path / "meta.png"), pnginfo=info)
    with Image.open(p) as im:
        assert im.info.get("prompt") == '{"1": {"class_type": "X"}}', "prompt text chunk round-trips"
        assert im.info.get("extra") == '{"workflow": {}}', "extra text chunks round-trip"


def test_save_tensor_failed_save_leaves_no_partial_file(tmp_path, monkeypatch):
    arr = torch.rand(5, 8, 3)
    target = str(tmp_path / "fail.png")
    real_fromarray = Image.fromarray

    def broken_fromarray(a, *args, **kwargs):
        im = real_fromarray(a, *args, **kwargs)

        def broken_save(*a2, **k2):
            raise OSError("simulated write failure")

        im.save = broken_save
        return im

    monkeypatch.setattr(Image, "fromarray", staticmethod(broken_fromarray))
    with pytest.raises(OSError):
        save_tensor(arr, target)
    assert not os.path.exists(target), "a failed save must leave no partial file"
    assert not [f for f in os.listdir(str(tmp_path)) if f.lower().startswith("fail")], \
        "no partial or conflict file may remain after a failed save"


def test_save_tensor_write_root_guard(tmp_path):
    arr = torch.rand(5, 8, 3)
    outp = os.path.join(str(tmp_path), "output")
    os.makedirs(outp)
    p = save_tensor(arr, os.path.join(outp, "ok.png"), roots=[outp])   # inside the root: fine
    assert os.path.isfile(p)
    with pytest.raises(AccessDeniedError):
        save_tensor(arr, os.path.join(str(tmp_path), "elsewhere.png"), roots=[outp])
    assert not os.path.exists(os.path.join(str(tmp_path), "elsewhere.png")), \
        "the exclusive create refuses a parent outside the write roots (no file)"


# ---------------------------------------------------------------------------
# MMLoadImage.execute folder mode: one listing per run (the snapshot), seeded
# selection and incremental wraparound, lazy decoding of the image list
# ---------------------------------------------------------------------------

@needs_comfy
def test_loader_folder_snapshot_single_listing(tmp_path, monkeypatch):
    """Exactly ONE listing per run: the picked file, the 1-based index and the
    (linked) image list all come from the same listing, so a file appearing
    between scans can never desynchronize them."""
    batch = os.path.join(str(tmp_path), "batch")
    os.makedirs(batch)
    for fname in ("a.png", "b.png", "c.png"):
        _png(os.path.join(batch, fname), (4, 4))
    scans = []

    def fake_list(folder, forced=False):
        scans.append(folder)
        if len(scans) == 1:
            # the first listing only sees b and c (a appears only later)
            return [os.path.join(batch, "b.png"), os.path.join(batch, "c.png")]
        return [os.path.join(batch, "a.png"), os.path.join(batch, "b.png"), os.path.join(batch, "c.png")]

    monkeypatch.setattr(image_io, "list_folder_images", fake_list)
    linked = {"5": {"class_type": "MMLoadImage", "inputs": {}},
              "8": {"class_type": "X", "inputs": {"images": ["5", 5]}}}
    monkeypatch.setattr(image_io.MMLoadImage, "hidden",
                        types.SimpleNamespace(unique_id="5", prompt=linked))
    monkeypatch.setattr(image_io, "_preview_entries", lambda tensors, cls: [])
    decodes = []
    real_load = image_io.load_image_file

    def load_spy(p):
        decodes.append(p)
        return real_load(p)

    monkeypatch.setattr(image_io, "load_image_file", load_spy)
    try:
        out = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                           batch_mode=False, index=2)
        img, _mask, _w, _h, path, image_list, idx = out.args
        assert len(scans) == 1 and scans[0] == os.path.abspath(batch), \
            f"folder mode must list exactly once (the snapshot): {len(scans)} scans"
        # the selection, the index and the list all refer to the FIRST listing [b, c]:
        # index 2 is the second element of THAT list (c), no matter what a later
        # listing ([a, b, c]) would say
        assert path == os.path.join(batch, "c.png"), "selection comes from the first listing (index 2 in [b, c] -> c)"
        assert idx == 2, "the 1-based index comes from the same listing"
        assert len(image_list) == 2, "the linked image list is the first listing itself"
        assert image_list[idx - 1] is img, "the selected tensor is reused at the reported position"
        other = image_list[0] if idx == 2 else image_list[1]
        assert other is not img, "the other listed file is a different tensor"
        assert len(decodes) == 2 and decodes[0] == path, \
            "the selected file is decoded once, the others are decoded for the list"
        # the next run takes a fresh listing (a full snapshot again)
        out2 = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                            batch_mode=False, index=1)
        assert out2.args[4] == os.path.join(batch, "a.png") and out2.args[6] == 1, \
            "the second run uses its own listing"
    finally:
        image_io._LAST_RESOLVED.pop("5", None)


@needs_comfy
def test_loader_folder_seeded_and_wraparound(tmp_path, monkeypatch):
    batch = os.path.join(str(tmp_path), "batch")
    os.makedirs(batch)
    for fname in ("a.png", "b.png", "c.png"):
        _png(os.path.join(batch, fname), (4, 4))
    unlinked = {"5": {"class_type": "MMLoadImage", "inputs": {}},
                "8": {"class_type": "X", "inputs": {"images": ["5", 0]}}}
    monkeypatch.setattr(image_io.MMLoadImage, "hidden",
                        types.SimpleNamespace(unique_id="5", prompt=unlinked))
    monkeypatch.setattr(image_io, "_preview_entries", lambda tensors, cls: [])
    try:
        # incremental: an index past the end wraps back to the first image
        out = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                           batch_mode=False, index=7)
        assert out.args[4] == os.path.join(batch, "a.png") and out.args[6] == 1, \
            "incremental index past the end wraps to the first image"
        # random: the seed makes the pick deterministic
        o1 = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                          batch_mode=True, index=42)
        o2 = image_io.MMLoadImage.execute(image="", path="", mode="folder", folder=batch,
                                          batch_mode=True, index=42)
        assert o1.args[4] == o2.args[4], "random: same seed -> same file"
        assert 1 <= o2.args[6] <= 3, "the reported index is valid for the folder"
        # unlinked list: only the picked file is decoded
        assert len(o2.args[5]) == 1 and o2.args[5][0] is o2.args[0], \
            "unlinked image_list stays the single loaded image"
    finally:
        image_io._LAST_RESOLVED.pop("5", None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
