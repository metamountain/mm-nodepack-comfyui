"""Unit tests for the MM Nodepack (pure helpers; no ComfyUI server needed).

Run:
  D:\\ComfyUI_windows_portable\\python_embeded\\python.exe -m pytest tests/test_mm_nodepack.py -q
  (fallback: D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe -m pytest tests/test_mm_nodepack.py -q)

Tests that depend on the V3 node API (comfy_api) are skipped when the interpreter
does not provide it (e.g. a plain venv without ComfyUI's site-packages).
"""
import os
import sys

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

from PIL import Image  # noqa: E402

from py import paths as py_paths  # noqa: E402
from py.common import (  # noqa: E402
    IMAGE_EXTS,
    LRUCache,
    list_folder_images,
    load_image_file,
    preview_view_entry,
    save_tensor,
)
from py.paths import (  # noqa: E402
    NoSourceError,
    find_origin,
    find_origin_files,
    last_dir_file,
    load_last_dir,
    next_target,
    resolve_folder_image,
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
          "7": {"class_type": "MMPreviewImage", "inputs": {"images": ["6", 0]}}}
    assert image_io.find_loader(pr, "7", "auto") == ("5", ""), "auto: loader upstream"
    pr2 = dict(pr, **{"9": dict(L), "8": {"class_type": "MMPreviewImage", "inputs": {"images": ["6", 0]}}})
    assert image_io.find_loader(pr2, "8", "auto")[0] == "5", "auto: upstream wins over a second loader"
    pr3 = dict(pr2, **{"10": {"class_type": "EmptyImage", "inputs": {}},
                       "11": {"class_type": "MMPreviewImage", "inputs": {"images": ["10", 0]}}})
    lid, note = image_io.find_loader(pr3, "11", "auto")
    assert lid is None and "#5" in note and "#9" in note, f"auto: ambiguous -> note '{note}'"
    assert image_io.find_loader(pr3, "11", "#9") == ("9", ""), "select #9"
    assert image_io.find_loader({"3:9": L, "4": {"class_type": "X", "inputs": {}}}, "4", "9")[0] == "3:9", \
        "select inside subgraph"
    assert image_io.find_loader(pr3, "11", "#42")[0] is None, "select unknown number -> None"
    one = {"5": L, "10": {"class_type": "EmptyImage", "inputs": {}},
           "11": {"class_type": "MMPreviewImage", "inputs": {"images": ["10", 0]}}}
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
    loader_outputs = [o.id for o in loader_schema.outputs]
    assert loader_outputs == ["image", "mask", "width", "height", "path", "image_list", "index"], \
        f"loader outputs match expected order without 'sync': {loader_outputs}"
    assert loader_schema.outputs[5].is_output_list is True, "loader output 'image_list' has is_output_list=True"
    assert loader_schema.display_name == "Load Image MM" and loader_schema.category == "MM Nodepack"
    assert image_io._output_slot_linked({"5": {}, "8": {"inputs": {"images": ["5", 5]}}}, "5", 5) is True, \
        "_output_slot_linked detects linked image_list slot"
    assert image_io._output_slot_linked({"5": {}, "8": {"inputs": {"images": ["5", 0]}}}, "5", 5) is False, \
        "_output_slot_linked returns False when image_list is unconnected"
    preview = image_io.MMPreviewImage.define_schema()
    preview_inputs = [i.id for i in preview.inputs]
    assert "sync" not in preview_inputs, f"preview inputs have no 'sync': {preview_inputs}"
    assert preview.display_name == "Preview Image MM" and preview.category == "MM Nodepack"


@needs_comfy
def test_node_list():
    assert image_io.NODE_LIST == [image_io.MMLoadImage, image_io.MMPreviewImage], "NODE_LIST exports both nodes"


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
