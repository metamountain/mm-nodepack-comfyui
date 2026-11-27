"""Unit tests for the Upscale Crop MM node (image-only V3 node).

Run:
  $env:CUDA_VISIBLE_DEVICES='-1'; D:\\ComfyUI_windows_portable\\python_embeded\\python.exe -B -m pytest -p no:cacheprovider tests/ -q

The node's logic is pure tensor maths, so the tests drive execute() directly
with synthetic images. No ComfyUI server and no real upscale model are
needed: the model path is exercised with a fake model that records what it
is handed, and the two-slot model cache is tested on its own.

Tests that depend on the V3 node API (comfy_api) are skipped when the
interpreter does not provide it (e.g. a plain venv without ComfyUI's
site-packages).
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# some interpreters have the legacy `py` library in site-packages (or a stale import
# of it in sys.modules): make sure it cannot shadow the `py/` package of this pack
sys.modules.pop("py", None)
sys.path.insert(0, REPO_ROOT)
# in --import-mode=importlib pytest does not put the tests/ directory on
# sys.path, so make conftest_helpers importable explicitly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest_helpers import make_folder_paths_stub  # noqa: E402

try:
    from py import upscale_crop  # noqa: E402
    COMFY_AVAILABLE = True
except Exception:  # noqa: BLE001 – plain venv without ComfyUI's site-packages
    upscale_crop = None
    COMFY_AVAILABLE = False

needs_comfy = pytest.mark.skipif(
    not COMFY_AVAILABLE, reason="comfy_api not available in this interpreter")

# The node class, looked up rather than spelled, so a rename cannot break
# the suite.
Node = upscale_crop.NODE_LIST[0] if COMFY_AVAILABLE else None

SRC_W, SRC_H = 1024, 1536


@pytest.fixture(autouse=True)
def _clear_node_state():
    """The preview cache and the model cache are module-level; one test's
    leftovers must not leak into the next."""
    if upscale_crop is not None:
        upscale_crop._LAST_IMAGE.clear()
        upscale_crop._MODEL_CACHE.clear()
    yield


# ---------------------------------------------------------------- fixtures

def image(w, h, b=1):
    """A B,H,W,C image with structure, so a resize has something to lose."""
    ys = torch.linspace(0, 1, h).view(1, h, 1, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w, 1)
    base = (ys * xs).repeat(b, 1, 1, 3)
    return (base + torch.rand(b, h, w, 3) * 0.1).clamp(0, 1)


def gradient(w, h, b=1):
    """A B,H,W,C image whose channels encode position: R rises left to
    right, G rises top to bottom. The mean colour of any cut-out window
    identifies where that window sat, so anchor direction is checkable."""
    xs = torch.linspace(0.0, 1.0, w).view(1, 1, w, 1).expand(1, h, w, 1)
    ys = torch.linspace(0.0, 1.0, h).view(1, h, 1, 1).expand(1, h, w, 1)
    zz = torch.zeros(1, h, w, 1)
    base = torch.cat([xs, ys, zz], dim=-1)
    return base.repeat(b, 1, 1, 1).contiguous()


def rgba(w, h, b=1, alpha=0.5):
    """A B,H,W,C RGBA image: the suite's structured RGB plus a constant
    alpha, so preservation can be checked against a known value."""
    base = image(w, h, b)
    a = torch.full((b, h, w, 1), float(alpha))
    return torch.cat([base, a], dim=-1)


def run(**kw):
    """One execute() call. The V3 node's execute is a classmethod, so the
    reference suite's instance-method call maps to a direct class call."""
    return Node.execute(**kw)


class FakeCU:
    """comfy.utils without a running ComfyUI: one single tile, so the tiling
    loop of _model_pass degenerates to one pass."""
    @staticmethod
    def get_tiled_scale_steps(w, h, tile_x=512, tile_y=512, overlap=32):
        return 1

    @staticmethod
    def ProgressBar(steps):
        class _pbar:
            def update(self, n=1):
                pass

            def close(self):
                pass
        return _pbar()

    @staticmethod
    def tiled_scale(x, fn, tile_x=512, tile_y=512, overlap=32,
                    upscale_amount=1, pbar=None):
        return fn(x)


class FakeModel:
    """A 3-channel 2x model that records its input.

    Mocking the loader (not _model_pass) is deliberate: the alpha split lives
    inside _model_pass, so the fake model must be what the real pass feeds.
    It returns pure RGB on purpose: a pixel model that received alpha would
    mangle it, which is exactly what the node must not let happen.

    state_dict() and the .model attribute satisfy
    comfy.model_management.module_size, which the pack's _model_pass
    consults whenever that import succeeds (the reference suite ran in a
    plain interpreter where it was absent).
    """
    scale = 2.0

    def __init__(self, received):
        self.received = received
        self.model = self

    def state_dict(self):
        return {}

    def __call__(self, x):
        self.received["tensor"] = x
        return F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2),
                             mode="bilinear", align_corners=False)

    def to(self, device):
        return self

    def eval(self):
        return self


def use_fake_model(received, monkeypatch):
    """Point the loader at a FakeModel and pretend comfy.utils is available."""
    fake = FakeModel(received)
    monkeypatch.setattr(upscale_crop, "_load_upscale_model", lambda name: fake)
    monkeypatch.setattr(upscale_crop, "_cu", FakeCU())


# ---------------------------------------------------------------- schema

@needs_comfy
def test_schema_id_name_category_and_aliases():
    schema = Node.define_schema()
    assert schema.node_id == "MM" + "UpscaleCrop"
    assert schema.display_name == "Upscale Crop MM"
    assert schema.category == upscale_crop.CATEGORY
    assert schema.search_aliases == ["mm", "metamountain", "mm nodepack",
                                     "upscale mm", "crop mm"]


@needs_comfy
def test_schema_input_and_output_contracts():
    schema = Node.define_schema()
    assert [i.id for i in schema.inputs] == [
        "image", "target_mode", "scale_factor", "scale_percent", "shortest_side",
        "longest_side", "megapixels", "target_width", "target_height",
        "resize_condition", "method", "upscale_model", "snap", "multiple_of",
        "crop", "crop_ratio", "crop_width", "crop_height", "crop_orientation",
        "crop_position", "crop_offset_x", "crop_offset_y", "crop_seed"]
    assert [o.id for o in schema.outputs] == ["image", "width", "height",
                                              "info"]


# ---------------------------------------------------------------- sizing
#
# The size maths, pinned. Source is 1024x1536 throughout -- a portrait
# render, which is the case the previews and defaults were designed against.

def _snap_out(n, m):
    """The same snap the node applies to output sizes (mirror of
    upscale_crop._snap): round to the nearest multiple, never below one."""
    n = max(1.0, float(n))
    if m < 1.0:
        return max(1, int(round(n)))
    m = max(1, int(round(m)))
    return max(m, int(round(n / m)) * m)


def expected_size(sw, sh, mode, snap, mult, **vals):
    """The (w, h) execute() must produce, straight from the spec --
    independent of the node's own helpers, so it cannot drift with them."""
    if mode == "scale_percent":
        k = max(0.01, vals["scale_percent"] / 100.0)
    elif mode == "scale_factor":
        k = max(0.01, vals["scale_factor"])
    elif mode == "shortest_side":
        k = vals["shortest_side"] / max(1, min(sw, sh))
    elif mode == "longest_side":
        k = vals["longest_side"] / max(1, max(sw, sh))
    elif mode == "megapixels":
        k = (vals["megapixels"] * 1e6 / max(1, sw * sh)) ** 0.5
    elif mode == "custom":
        # the crop is not optional in a box mode: cover the (snapped) box,
        # then cut to it -- the final size IS the box, whatever the source
        bw, bh = vals["target_width"], vals["target_height"]
        if snap and mult >= 1:
            bw, bh = _snap_out(bw, mult), _snap_out(bh, mult)
        return bw, bh
    else:
        raise ValueError(mode)
    return _snap_out(sw * k, mult if snap else 0.0), _snap_out(sh * k, mult if snap else 0.0)


@needs_comfy
def _out(**kw):
    kw.setdefault("image", image(SRC_W, SRC_H))
    kw.setdefault("method", "lanczos")     # never touch a model in tests
    r = run(**kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[3]


@needs_comfy
def test_percent_and_factor_agree():
    a = _out(target_mode="scale_percent", scale_percent=200.0)[:2]
    b = _out(target_mode="scale_factor", scale_factor=2.0)[:2]
    assert a == b == (2048, 3072), (a, b)


@needs_comfy
def test_shortest_side_pins_the_short_edge():
    w, h, _ = _out(target_mode="shortest_side", shortest_side=1536)
    assert min(w, h) == 1536, (w, h)
    assert (w, h) == (1536, 2304), (w, h)


@needs_comfy
def test_longest_side_pins_the_long_edge():
    w, h, _ = _out(target_mode="longest_side", longest_side=2048, multiple_of=1)
    assert max(w, h) == 2048, (w, h)
    assert (w, h) == (1365, 2048), (w, h)


@needs_comfy
def test_megapixels_hits_the_pixel_count():
    w, h, _ = _out(target_mode="megapixels", megapixels=2.0, multiple_of=1)
    mp = w * h / 1e6
    assert 1.94 < mp < 2.06, mp


@needs_comfy
def test_aspect_is_preserved_by_every_mode():
    src = SRC_W / SRC_H
    for kw in [dict(target_mode="scale_percent", scale_percent=137.0),
               dict(target_mode="scale_factor", scale_factor=1.37),
               dict(target_mode="shortest_side", shortest_side=999),
               dict(target_mode="longest_side", longest_side=999),
               dict(target_mode="megapixels", megapixels=1.3)]:
        w, h, _ = _out(multiple_of=1, **kw)
        assert abs(w / h - src) < 0.01, (kw, w, h)


@needs_comfy
def test_every_mode_downscales_too():
    for kw in [dict(target_mode="scale_percent", scale_percent=50.0),
               dict(target_mode="scale_factor", scale_factor=0.5),
               dict(target_mode="shortest_side", shortest_side=512),
               dict(target_mode="longest_side", longest_side=768),
               dict(target_mode="megapixels", megapixels=0.4)]:
        w, h, _ = _out(**kw)
        assert w < SRC_W and h < SRC_H, (kw, w, h)


@needs_comfy
def test_multiple_of_is_always_satisfied():
    for m in [2, 4, 8, 16, 32, 64]:
        for mode, kw in [("scale_factor", dict(scale_factor=1.37)),
                         ("shortest_side", dict(shortest_side=999)),
                         ("megapixels", dict(megapixels=1.3))]:
            w, h, _ = _out(target_mode=mode, multiple_of=m, snap=True, **kw)
            assert w % int(m) == 0 and h % int(m) == 0, (m, mode, w, h)


@needs_comfy
def test_multiple_of_off_gives_the_exact_number():
    w, h, _ = _out(target_mode="shortest_side", shortest_side=999, multiple_of=1)
    assert min(w, h) == 999, (w, h)

# ---------------------------------------------------------------- matrix
#
# The exhaustive sweep: every shape family of source (square, landscape,
# portrait, ultrawide, tiny) times every target mode and a spread of values,
# with snap off and on (8 and 64). The expected size is computed from the
# spec, not from the node's helpers.

RESOLUTIONS = [
    (512, 512), (1024, 1024),            # square
    (1920, 1080), (3840, 2160), (777, 533),   # landscape
    (1080, 1920), (512, 768), (533, 777),     # portrait
    (2560, 1080),                        # ultrawide
    (64, 96),                            # tiny
]

MODE_VALUES = [
    ("scale_percent", dict(scale_percent=v)) for v in (50.0, 150.0, 200.0)
]
MODE_VALUES += [
    ("scale_factor", dict(scale_factor=v)) for v in (0.5, 1.5, 2.0)
]
MODE_VALUES += [
    ("shortest_side", dict(shortest_side=v)) for v in (512, 1536)
]
MODE_VALUES += [
    ("longest_side", dict(longest_side=v)) for v in (512, 2048)
]
MODE_VALUES += [
    ("megapixels", dict(megapixels=v)) for v in (0.5, 2.0)
]
MODE_VALUES += [
    ("custom", dict(target_width=w, target_height=h))
    for w, h in [(1920, 1080), (1080, 1920), (1024, 1024), (512, 768), (768, 512)]
]
SNAP_CASES = [(False, 1), (True, 8), (True, 64)]


@needs_comfy
@pytest.mark.parametrize("sw, sh", RESOLUTIONS)
@pytest.mark.parametrize("mode, kws", MODE_VALUES)
@pytest.mark.parametrize("snap, mult", SNAP_CASES)
def test_matrix_output_size_is_exact(sw, sh, mode, kws, snap, mult):
    kw = dict(kws, target_mode=mode, snap=snap, multiple_of=mult, crop=False)
    # nearest exercises the exact same sizing/snap/crop geometry as lanczos at 30x speed
    r = run(image=image(sw, sh), method="nearest", **kw)
    w, h = int(r[0].shape[2]), int(r[0].shape[1])
    assert r[0].shape == (1, h, w, 3), r[0].shape
    vals = {k: v for k, v in kw.items()
            if k not in ("target_mode", "snap", "multiple_of", "crop")}
    want = expected_size(sw, sh, mode, snap, mult, **vals)
    assert (w, h) == want, (sw, sh, mode, kws, snap, mult, w, h, want)
    if snap and mult >= 1:
        assert w % mult == 0 and h % mult == 0, (sw, sh, mode, mult, w, h)


@needs_comfy
@pytest.mark.parametrize("sw, sh", RESOLUTIONS)
@pytest.mark.parametrize("snap, mult", [(False, 1), (True, 8), (True, 64)])
@pytest.mark.parametrize("ratio", ["1:1", "4:5", "16:9"])
def test_matrix_crop_keeps_size_and_multiples(sw, sh, snap, mult, ratio):
    r = run(image=image(sw, sh), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0,
            crop=True, crop_ratio=ratio, crop_orientation="auto",
            snap=snap, multiple_of=mult)
    w, h = int(r[0].shape[2]), int(r[0].shape[1])
    # the snap can push the upscaled frame itself past 2x (2160 -> 2176 at
    # mult 64), so the frame -- not the source -- is the upper bound
    fw = _snap_out(2 * sw, mult if (snap and mult >= 1) else 0.0)
    fh = _snap_out(2 * sh, mult if (snap and mult >= 1) else 0.0)
    assert 0 < w <= fw and 0 < h <= fh, (sw, sh, ratio, snap, w, h, fw, fh)
    if snap and mult >= 1:
        if w >= mult:
            assert w % mult == 0, (sw, sh, ratio, mult, w, h)
        if h >= mult:
            assert h % mult == 0, (sw, sh, ratio, mult, w, h)


@needs_comfy
@pytest.mark.parametrize("sw, sh", [(512, 512), (1920, 1080), (1080, 1920),
                                    (2560, 1080), (64, 96)])
@pytest.mark.parametrize("snap, mult", [(False, 1), (True, 8), (True, 64)])
def test_matrix_custom_crop_box_is_cut_out_of_the_result(sw, sh, snap, mult):
    # a dedicated crop box smaller than the upscaled result on every shape
    cw, ch = min(2 * sw, 1024), min(2 * sh, 768)
    r = run(image=image(sw, sh), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0,
            crop=True, crop_ratio="custom", crop_width=cw, crop_height=ch,
            crop_orientation="auto", snap=snap, multiple_of=mult)
    w, h = int(r[0].shape[2]), int(r[0].shape[1])
    fw = _snap_out(2 * sw, mult if (snap and mult >= 1) else 0.0)
    fh = _snap_out(2 * sh, mult if (snap and mult >= 1) else 0.0)
    assert w <= fw and h <= fh, (sw, sh, snap, w, h, fw, fh)
    if snap and mult >= 1:
        if w >= mult:
            assert w % mult == 0, (sw, sh, mult, w, h)
        if h >= mult:
            assert h % mult == 0, (sw, sh, mult, w, h)


# ---------------------------------------------------------------- crop
#
# The crop stage: formats, anchors, offset clamping. The crop window is a
# dedicated pair, crop_width x crop_height, for crop_ratio 'custom'.

@needs_comfy
def _crop(**kw):
    kw.setdefault("image", image(SRC_W, SRC_H))
    kw.setdefault("method", "lanczos")
    kw.setdefault("target_mode", "scale_factor")
    kw.setdefault("scale_factor", 2.0)
    kw.setdefault("crop", True)
    r = run(**kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[3]


@needs_comfy
def test_crop_off_leaves_the_upscale_alone():
    r = run(image=image(SRC_W, SRC_H), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0, crop=False)
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (2048, 3072)
    assert "crop off" in r[3]


@needs_comfy
def test_default_45_portrait_on_a_portrait_source():
    w, h, _ = _crop(crop_ratio="4:5", crop_orientation="portrait")
    assert (w, h) == (2048, 2560), (w, h)
    assert abs(w / h - 4 / 5) < 0.01


@needs_comfy
def test_orientation_flips_the_format():
    p = _crop(crop_ratio="4:5", crop_orientation="portrait")[:2]
    l = _crop(crop_ratio="4:5", crop_orientation="landscape")[:2]
    assert p[0] / p[1] < 1 < l[0] / l[1], (p, l)


@needs_comfy
def test_every_format_fits_inside_and_keeps_its_shape():
    """Whichever way the table writes a format, orientation normalises it:
    landscape is always the wide reading, portrait always the tall one."""
    want = {"4:5": 4 / 5, "1:1": 1.0, "4:3": 4 / 3, "3:2": 3 / 2,
            "DIN": 2 ** 0.5, "16:10": 1.6, "16:9": 16 / 9, "2:1": 2.0,
            "21:9": 64 / 27}
    for name, raw in want.items():
        wide, tall = max(raw, 1 / raw), min(raw, 1 / raw)
        for orient, ar in (("landscape", wide), ("portrait", tall)):
            w, h, _ = _crop(crop_ratio=name, crop_orientation=orient,
                            multiple_of=1)
            assert w <= 2048 and h <= 3072, (name, orient, w, h)
            assert abs(w / h - ar) < 0.02, (name, orient, w / h, ar)


@needs_comfy
def test_custom_uses_the_given_pixels():
    w, h, _ = _crop(crop_ratio="custom", crop_width=1200, crop_height=900,
                    multiple_of=1)
    assert (w, h) == (1200, 900), (w, h)


@needs_comfy
def test_custom_pixels_still_obey_multiple_of():
    """900 is not a multiple of 8, so it must come back snapped, not exact."""
    w, h, _ = _crop(crop_ratio="custom", crop_width=1200, crop_height=900,
                    multiple_of=8, snap=True)
    assert (w, h) == (1200, 896), (w, h)


@needs_comfy
def test_custom_missing_box_falls_back_to_the_target_box():
    """crop_ratio 'custom' without crop_width/crop_height takes the target
    box instead of guessing -- the UI shows the dedicated fields, but a
    workflow can still omit them."""
    w, h, _ = _crop(crop_ratio="custom", target_width=1200, target_height=900,
                    crop_width=None, crop_height=None, multiple_of=1)
    assert (w, h) == (1200, 900), (w, h)


@needs_comfy
def test_custom_larger_than_the_image_is_clamped():
    w, h, _ = _crop(crop_ratio="custom", crop_width=9000, crop_height=9000)
    assert w <= 2048 and h <= 3072, (w, h)


@needs_comfy
def test_offset_cannot_push_the_window_off_the_image():
    """A silly offset must still yield the same size, just shifted as far
    as the image allows -- never a short crop or a crash."""
    base = _crop(crop_ratio="16:9", crop_orientation="landscape")[:2]
    for off in (-8192, -500, 0, 500, 8192):
        for pos in ("center", "top", "bottom", "left", "right"):
            w, h, _ = _crop(crop_ratio="16:9", crop_orientation="landscape",
                            crop_position=pos, crop_offset_y=off)
            assert (w, h) == base, (pos, off, w, h, base)


@needs_comfy
def test_all_anchors_run_and_keep_the_size():
    base = None
    for pos in ("center", "top", "bottom", "left", "right"):
        w, h, _ = _crop(crop_ratio="1:1", crop_position=pos)
        if base is None:
            base = (w, h)
        assert (w, h) == base, (pos, w, h, base)


@needs_comfy
def test_crop_output_respects_multiple_of():
    for m in [8, 16, 64]:
        w, h, _ = _crop(crop_ratio="DIN", crop_orientation="portrait",
                        multiple_of=int(m), snap=True)
        assert w % int(m) == 0 and h % int(m) == 0, (m, w, h)


# ---------------------------------------------------------------- exact box
#
# target_mode = custom, the box mode, and the crop controls around it.
# It is the one mode that cannot preserve the aspect ratio: it scales until
# the target_width x target_height box is covered, then crops to it, so the
# resolution you type is the resolution you get. The crop is not optional
# here. That is NOT the same as crop_ratio = custom, which never rescales.

@needs_comfy
def _box(w, h, src=(1024, 1536), **kw):
    kw.setdefault("method", "lanczos")
    r = run(image=image(*src), target_mode="custom",
            target_width=w, target_height=h, **kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[3]


@needs_comfy
def test_you_get_exactly_the_box():
    for w, h in [(1920, 1080), (1080, 1920), (1000, 1000), (2048, 512)]:
        got = _box(w, h)[:2]
        assert got == (w, h), ((w, h), got)


@needs_comfy
def test_the_box_is_hit_from_any_source_shape():
    for src in [(1024, 1536), (1536, 1024), (512, 512), (3000, 400)]:
        got = _box(1920, 1080, src=src)[:2]
        assert got == (1920, 1080), (src, got)


@needs_comfy
def test_it_covers_rather_than_fits():
    """Covering means no bars. A fit would have left one axis short, and this
    node never pads, so every pixel of the box must come from the image."""
    w, h, _ = _box(1920, 1080, src=(1024, 1536))
    assert (w, h) == (1920, 1080)


@needs_comfy
def test_it_downscales_when_the_box_is_smaller():
    got = _box(256, 256, src=(2048, 2048))[:2]
    assert got == (256, 256), got


@needs_comfy
def test_the_box_obeys_multiple_of():
    """900 sits exactly halfway between 896 and 904, and Python rounds a tie
    to even -- 112.5 -> 112 -> 896. Worth pinning so a future refactor to
    round-half-up is noticed rather than silently shifting everyone's crops."""
    w, h, _ = _box(1000, 900, multiple_of=8, snap=True)
    assert w % 8 == 0 and h % 8 == 0, (w, h)
    assert (w, h) == (1000, 896), (w, h)

    # away from a tie there is nothing to argue about
    assert _box(1000, 902, multiple_of=8, snap=True)[:2] == (1000, 904)
    assert _box(1000, 899, multiple_of=8, snap=True)[:2] == (1000, 896)


@needs_comfy
def test_snap_off_ignores_multiple_of():
    assert _box(1366, 769, multiple_of=8)[:2] == (1366, 769)


@needs_comfy
def test_multiple_of_off_gives_the_literal_box():
    assert _box(1001, 903, multiple_of=1)[:2] == (1001, 903)


@needs_comfy
def test_position_and_offsets_still_frame_it():
    """The box fixes the size; these still decide which part survives."""
    for pos in ["center", "top", "bottom", "left", "right"]:
        for ox, oy in [(0, 0), (200, 0), (0, 200), (-9999, 9999)]:
            got = _box(1920, 1080, crop_position=pos,
                       crop_offset_x=ox, crop_offset_y=oy)[:2]
            assert got == (1920, 1080), (pos, ox, oy, got)


@needs_comfy
def test_it_overrides_the_manual_crop_rather_than_stacking():
    """Two crops fighting over one result would be nobody's idea of clear."""
    got = _box(1920, 1080, crop=True, crop_ratio="1:1")[:2]
    assert got == (1920, 1080), got


@needs_comfy
def test_it_is_not_the_same_as_custom_crop():
    """target_mode custom scales the image to cover the box, so it lands on
    those exact dimensions. crop_ratio custom never rescales -- it cuts the
    largest window of the shape you asked for out of what is already there.
    Both keep the shape; only the target mode guarantees the size."""
    assert _box(3000, 3000, src=(1024, 1536))[:2] == (3000, 3000)

    r = run(image=image(1024, 1536), method="lanczos",
            target_mode="scale_factor", scale_factor=1.0, crop=True,
            crop_ratio="custom", crop_width=3000, crop_height=3000,
            multiple_of=1)
    got = (int(r[0].shape[2]), int(r[0].shape[1]))
    assert got == (1024, 1024), got           # the biggest square that fits

# ---- the two crop controls that changed ----------------------------------

@needs_comfy
def test_offsets_move_on_their_own_axis():
    """One offset could only ever move along whichever axis the anchor left
    free -- with center that meant no horizontal nudge at all."""
    c = upscale_crop._crop_rect(2000, 1000, "1:1", "auto", 0, 0, "center", 0, 0)
    x = upscale_crop._crop_rect(2000, 1000, "1:1", "auto", 0, 0, "center", 300, 0)
    y = upscale_crop._crop_rect(2000, 1000, "1:1", "auto", 0, 0, "center", 0, 300)
    assert x[0] > c[0] and x[1] == c[1], (c, x)
    assert y[1] == c[1], "a 1:1 window in a 2:1 frame has no vertical slack"

    c2 = upscale_crop._crop_rect(1000, 2000, "1:1", "auto", 0, 0, "center", 0, 0)
    y2 = upscale_crop._crop_rect(1000, 2000, "1:1", "auto", 0, 0, "center", 0, 300)
    assert y2[1] > c2[1] and y2[0] == c2[0], (c2, y2)


@needs_comfy
def test_offsets_are_clamped_independently():
    for ox, oy in [(-99999, 0), (99999, 0), (0, -99999), (0, 99999)]:
        x0, y0, x1, y1 = upscale_crop._crop_rect(2000, 2000, "16:9", "landscape",
                                                 0, 0, "center", ox, oy)
        assert 0 <= x0 < x1 <= 1.0001, (ox, oy, x0, x1)
        assert 0 <= y0 < y1 <= 1.0001, (ox, oy, y0, y1)


@needs_comfy
def test_auto_orientation_follows_the_source():
    """A fixed orientation cuts a thin strip out of half a mixed batch."""
    tall = upscale_crop._crop_rect(1000, 1500, "4:5", "auto", 0, 0, "center", 0, 0)
    wide = upscale_crop._crop_rect(1500, 1000, "4:5", "auto", 0, 0, "center", 0, 0)
    tall_ar = ((tall[2] - tall[0]) * 1000) / ((tall[3] - tall[1]) * 1500)
    wide_ar = ((wide[2] - wide[0]) * 1500) / ((wide[3] - wide[1]) * 1000)
    assert tall_ar < 1 < wide_ar, (tall_ar, wide_ar)


@needs_comfy
def test_auto_keeps_the_natural_orientation_on_a_square():
    """A square source has no 'other way round' to follow: 4:5 stays portrait
    and 16:9 stays landscape, so a square frame in a mixed batch does not
    flip half the formats."""
    r = upscale_crop._crop_rect(1024, 1024, "4:5", "auto", 0, 0, "center", 0, 0)
    ar = ((r[2] - r[0]) * 1024) / ((r[3] - r[1]) * 1024)
    assert ar < 1, ("4:5 on a square must stay portrait", ar)

    r = upscale_crop._crop_rect(1024, 1024, "16:9", "auto", 0, 0, "center", 0, 0)
    ar = ((r[2] - r[0]) * 1024) / ((r[3] - r[1]) * 1024)
    assert ar > 1, ("16:9 on a square must stay landscape", ar)


@needs_comfy
def test_auto_beats_a_fixed_orientation_on_a_mixed_batch():
    """The whole point: auto keeps more of every frame than either fixed
    setting can, because half the batch is the wrong way round for it."""
    def area(w, h, orient):
        r = upscale_crop._crop_rect(w, h, "4:5", orient, 0, 0, "center", 0, 0)
        return (r[2] - r[0]) * (r[3] - r[1])

    batch = [(1024, 1536), (1536, 1024)]
    auto = sum(area(w, h, "auto") for w, h in batch)
    for fixed in ("portrait", "landscape"):
        assert auto > sum(area(w, h, fixed) for w, h in batch), fixed


# ---- custom box mode, and the random anchor ------------------------------

@needs_comfy
def test_custom_always_crops_to_the_box():
    """Typing a resolution and not getting it back would be no use, so the
    crop is not optional here -- the crop toggle is ignored and hidden."""
    for crop in (True, False):
        r = run(image=image(1024, 1536), method="lanczos",
                target_mode="custom", target_width=1920, target_height=1080,
                multiple_of=1, crop=crop)
        got = (int(r[0].shape[2]), int(r[0].shape[1]))
        assert got == (1920, 1080), (crop, got)


@needs_comfy
def test_a_box_mode_ignores_crop_ratio():
    """In a box mode the numbers you typed ARE the output size. Letting
    crop_ratio pick a different shape here meant asking for 832x1024 and
    getting 832x832 back. The box wins; crop_ratio is hidden in these modes
    precisely so it cannot argue with it."""
    for mode in ("custom",):
        for ratio in ("1:1", "16:9", "4:5", "custom"):
            r = run(image=image(1024, 1536), method="lanczos",
                    target_mode=mode, target_width=1920, target_height=1080,
                    multiple_of=1, crop=True, crop_ratio=ratio)
            got = (int(r[0].shape[2]), int(r[0].shape[1]))
            assert got == (1920, 1080), (mode, ratio, got)


@needs_comfy
def test_custom_obeys_multiple_of():
    for m in [8, 64]:
        r = run(image=image(1024, 1536), method="lanczos",
                target_mode="custom", target_width=1000, target_height=1000,
                multiple_of=int(m), snap=True)
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        assert w % int(m) == 0 and h % int(m) == 0, (m, w, h)


@needs_comfy
def test_random_is_reproducible_from_its_seed():
    """A free-running random would make the preview a lie and a result you
    liked impossible to get back."""
    def once(seed):
        r = upscale_crop._crop_rect(2000, 2000, "16:9", "landscape", 0, 0,
                                    "random", 0, 0, seed)
        return tuple(round(v, 6) for v in r)
    assert once(42) == once(42)
    assert once(42) != once(43)


@needs_comfy
def test_random_stays_inside_the_image():
    for seed in range(25):
        x0, y0, x1, y1 = upscale_crop._crop_rect(1900, 1200, "4:5", "portrait",
                                                 0, 0, "random", 0, 0, seed)
        assert 0 <= x0 < x1 <= 1.0001, (seed, x0, x1)
        assert 0 <= y0 < y1 <= 1.0001, (seed, y0, y1)


@needs_comfy
def test_random_actually_moves_around():
    seen = {tuple(round(v, 3) for v in
                  upscale_crop._crop_rect(2000, 2000, "16:9", "landscape",
                                          0, 0, "random", 0, 0, s)[:2])
            for s in range(30)}
    assert len(seen) > 20, f"only {len(seen)} distinct placements in 30 seeds"


@needs_comfy
def test_random_keeps_the_size_it_was_asked_for():
    base = None
    for seed in range(10):
        r = run(image=image(1024, 1536), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_ratio="16:9", crop_orientation="landscape",
                crop_position="random", crop_seed=seed)
        size = (int(r[0].shape[2]), int(r[0].shape[1]))
        base = base or size
        assert size == base, (seed, size, base)


@needs_comfy
def test_a_custom_box_too_big_keeps_its_shape():
    """It shrinks proportionally to the largest window that fits, the way the
    named formats do -- clamping each axis on its own would hand back some
    other shape entirely."""
    for cw, ch, want_ar in [(3000, 3000, 1.0), (1920, 1080, 16 / 9),
                            (1000, 4000, 0.25)]:
        r = run(image=image(1024, 1536), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_ratio="custom", crop_width=cw, crop_height=ch,
                multiple_of=1)
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        assert w <= 2048 and h <= 3072, (cw, ch, w, h)
        assert abs(w / h - want_ar) < 0.02, (cw, ch, w / h, want_ar)


@needs_comfy
def test_a_custom_box_that_fits_is_left_alone():
    """Only an oversized box is scaled down; one that fits is taken as given."""
    r = run(image=image(1024, 1536), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0, crop=True,
            crop_ratio="custom", crop_width=800, crop_height=600,
            multiple_of=1)
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (800, 600)


@needs_comfy
def test_every_crop_takes_the_largest_window_that_fits():
    """The promise across the whole crop stage: named formats and custom boxes
    alike give you as much of the image as their shape allows."""
    for kw in [dict(crop_ratio="1:1"), dict(crop_ratio="16:9"),
               dict(crop_ratio="custom", crop_width=3000, crop_height=3000),
               dict(crop_ratio="custom", crop_width=6000, crop_height=3000)]:
        r = run(image=image(1024, 1536), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_orientation="landscape", multiple_of=1, **kw)
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        # touching at least one edge is what "largest that fits" means
        assert w == 2048 or h == 3072, (kw, w, h)

# ---------------------------------------------------------------- spatial
#
# The anchors, checked with a marker image: R encodes the x coordinate and G
# the y coordinate, so the mean colour of a cut-out window says where it sat.
# The custom box 1024x768 in the default 2048x3072 result leaves slack on
# BOTH axes, which no named ratio can offer (they always fill one).

@needs_comfy
def _spatial(pos, off_x=0, off_y=0, mult=1, snap=False):
    """A gradient upscaled 2x with a 1024x768 custom crop at the anchor."""
    r = run(image=gradient(SRC_W, SRC_H), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0,
            crop=True, crop_ratio="custom", crop_width=1024, crop_height=768,
            crop_orientation="auto", crop_position=pos,
            crop_offset_x=off_x, crop_offset_y=off_y,
            snap=snap, multiple_of=mult)
    return r[0][0]  # H,W,3 float 0..1


def _mean(t, ch):
    return float(t[..., ch].mean().item())


@needs_comfy
def test_anchors_sit_where_the_gradient_says():
    left, center, right = (_mean(_spatial(p), 0)
                           for p in ("left", "center", "right"))
    assert left < center < right, (left, center, right)

    top, bottom = (_mean(_spatial(p), 1)
                   for p in ("top", "bottom"))
    middle = _mean(_spatial("center"), 1)
    assert top < middle < bottom, (top, middle, bottom)


@needs_comfy
def test_corner_anchors_hit_the_exact_corner_pixels():
    """The four corners have no slack on their edges: the window's corner
    pixel is the frame's corner pixel, so the markers read 0 or 1 there."""
    tl = _spatial("top-left")
    assert tl[0, 0, 0] < 0.06 and tl[0, 0, 1] < 0.06, (tl[0, 0, 0], tl[0, 0, 1])

    tr = _spatial("top-right")
    assert tr[0, -1, 0] > 0.94 and tr[0, -1, 1] < 0.06, (tr[0, -1, 0], tr[0, -1, 1])

    bl = _spatial("bottom-left")
    assert bl[-1, 0, 0] < 0.06 and bl[-1, 0, 1] > 0.94, (bl[-1, 0, 0], bl[-1, 0, 1])

    br = _spatial("bottom-right")
    assert br[-1, -1, 0] > 0.94 and br[-1, -1, 1] > 0.94, (br[-1, -1, 0], br[-1, -1, 1])


@needs_comfy
def test_snap_still_touches_the_exact_edges():
    """The old drift: snapping the crop AFTER the anchor was placed pushed
    right/bottom/corner placements off the edge. Use NON-multiple source and
    crop dimensions (1000x1000 with 500x350 snapped to mult=64 -> 512x320)
    so rounding actually changes box size in BOTH directions (up & down) and
    every anchor must still land on the exact frame edge."""
    src = gradient(1000, 1000)

    def cut(pos):
        r = run(image=src, method="nearest",
                target_mode="scale_factor", scale_factor=1.0,
                crop=True, crop_ratio="custom",
                crop_width=500, crop_height=350,
                crop_position=pos, snap=True, multiple_of=64)
        out = r[0][0]
        assert out.shape[1] % 64 == 0 and out.shape[0] % 64 == 0, out.shape
        return out

    for pos in ("left", "top-left", "bottom-left"):
        t = cut(pos)
        assert t[0, 0, 0] < 0.02, (pos, "left edge", float(t[0, 0, 0]))
    for pos in ("right", "top-right", "bottom-right"):
        t = cut(pos)
        assert t[0, -1, 0] > 0.98, (pos, "right edge", float(t[0, -1, 0]))
    for pos in ("top", "top-left", "top-right"):
        t = cut(pos)
        assert t[0, 0, 1] < 0.02, (pos, "top edge", float(t[0, 0, 1]))
    for pos in ("bottom", "bottom-left", "bottom-right"):
        t = cut(pos)
        assert t[-1, 0, 1] > 0.98, (pos, "bottom edge", float(t[-1, 0, 1]))


@needs_comfy
def test_snap_with_an_ugly_box_stays_inside_and_on_the_multiple():
    """A box that is not a multiple of the snap value comes back as the
    nearest multiple that still fits the frame -- never a window that is off
    the multiple AND touches an edge at the same time."""
    r = run(image=image(SRC_W, SRC_H), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0, crop=True,
            crop_ratio="custom", crop_width=1000, crop_height=700,
            snap=True, multiple_of=8)
    w, h = int(r[0].shape[2]), int(r[0].shape[1])
    assert w % 8 == 0 and h % 8 == 0, (w, h)
    assert (w, h) == (1000, 704), (w, h)      # 700 -> 704, the nearest fit


@needs_comfy
def test_all_ratios_all_orientations_on_every_shape():
    """9 named formats x 3 orientations on portrait, landscape and square
    sources: the window always fits, and the orientation lands where asked
    (auto following the frame, a square keeping the format's natural way)."""
    sources = [(1080, 1920), (1920, 1080), (1024, 1024)]
    ratios = ["4:5", "1:1", "4:3", "3:2", "DIN", "16:10", "16:9", "2:1", "21:9"]
    for sw, sh in sources:
        for ratio in ratios:
            for orient in ("auto", "portrait", "landscape"):
                r = run(image=image(sw, sh), method="lanczos",
                        target_mode="scale_factor", scale_factor=1.0,
                        crop=True, crop_ratio=ratio, crop_orientation=orient)
                w, h = int(r[0].shape[2]), int(r[0].shape[1])
                assert 0 < w <= sw and 0 < h <= sh, (sw, sh, ratio, orient, w, h)
                if ratio == "1:1":
                    assert w == h, (sw, sh, ratio, orient, w, h)
                    continue
                if orient == "portrait":
                    assert w < h, (sw, sh, ratio, orient, w, h)
                elif orient == "landscape":
                    assert w > h, (sw, sh, ratio, orient, w, h)
                else:  # auto: follow the frame; a square keeps the natural one
                    if sw < sh:
                        assert w < h, (sw, sh, ratio, w, h)
                    elif sw > sh:
                        assert w > h, (sw, sh, ratio, w, h)
                    else:
                        if ratio == "DIN":
                            assert w > h, (ratio, w, h)
                            continue
                        a, b = ratio.split(":")
                        ar = float(a) / float(b)
                        assert (w > h) == (ar > 1.0), (ratio, w, h)


@needs_comfy
def test_custom_crop_box_across_all_anchors_and_offsets():
    """The dedicated box at every anchor, with and without offsets: the size
    never changes and the window never leaves the frame."""
    base = None
    for pos in ("center", "top", "bottom", "left", "right",
                "top-left", "top-right", "bottom-left", "bottom-right"):
        for ox, oy in [(0, 0), (100, -50), (-9999, 9999)]:
            r = run(image=image(SRC_W, SRC_H), method="lanczos",
                    target_mode="scale_factor", scale_factor=2.0, crop=True,
                    crop_ratio="custom", crop_width=1000, crop_height=700,
                    crop_position=pos, crop_offset_x=ox, crop_offset_y=oy,
                    snap=True, multiple_of=8)
            size = (int(r[0].shape[2]), int(r[0].shape[1]))
            base = base or size
            assert size == base, (pos, ox, oy, size, base)


# ---------------------------------------------------------------- alpha
#
# Automatic alpha preservation: a 4-channel image goes through the model as
# RGB (the alpha is split off, resized bilinearly onto the model output and
# joined back), and non-model kernels resize all four channels.

@needs_comfy
def test_rgba_model_receives_rgb_and_the_output_keeps_alpha(monkeypatch):
    received = {}
    use_fake_model(received, monkeypatch)
    r = run(image=rgba(SRC_W, SRC_H), method="upscaler model",
            upscale_model="fake.pth", target_mode="scale_factor",
            scale_factor=2.0, crop=False)
    got = received["tensor"]
    # the model input is B,C,H,W: three channels, never the alpha
    assert got.shape == (1, 3, SRC_H, SRC_W), got.shape
    assert r[0].shape[-1] == 4, r[0].shape
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (2048, 3072), r[0].shape
    # the model output is pure RGB, so the joined alpha must still be the
    # source's constant across the whole frame
    alpha = r[0][0, :, :, 3]
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-4)


@needs_comfy
def test_rgba_alpha_survives_the_trim_and_the_crop(monkeypatch):
    """A target that is not the model's native scale trims the model output
    afterwards; the alpha must still be 4-channel and preserved all the way
    through. The window is anchored top-left, so the surviving part is known."""
    received = {}
    use_fake_model(received, monkeypatch)
    r = run(image=rgba(SRC_W, SRC_H), method="upscaler model",
            upscale_model="fake.pth", target_mode="scale_factor",
            scale_factor=1.5, crop=True, crop_ratio="1:1",
            crop_position="top-left", multiple_of=1)
    assert r[0].shape[-1] == 4, r[0].shape
    w, h = int(r[0].shape[2]), int(r[0].shape[1])
    assert w == h, (w, h)                             # the 1:1 window
    # the trim onto the target goes through PIL, so the alpha is 8-bit
    # quantised (128/255 = 0.502) -- tolerance covers that, not more
    alpha = r[0][0, :, :, 3]
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=5e-3)


@needs_comfy
def test_rgba_without_a_model_resizes_all_four_channels():
    r = run(image=rgba(256, 384), method="lanczos", scale_factor=2.0)
    assert r[0].shape[-1] == 4, r[0].shape
    # lanczos detours through PIL, so the alpha is 8-bit quantised
    alpha = r[0][0, :, :, 3]
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=5e-3)


@needs_comfy
def test_rgb_stays_three_channels():
    r = run(image=image(SRC_W, SRC_H), method="lanczos", scale_factor=2.0)
    assert r[0].shape[-1] == 3, r[0].shape


@needs_comfy
def test_rgba_preview_thumbnail_encodes():
    """The cached thumbnail must survive a 4-channel source, or the preview
    panel would die the first time someone connected an RGBA image."""
    upscale_crop._remember("alpha", rgba(SRC_W, SRC_H))
    c = upscale_crop._LAST_IMAGE["alpha"]
    assert c["png"], "expected a base64 RGBA thumbnail"
    assert (c["w"], c["h"]) == (SRC_W, SRC_H), (c["w"], c["h"])
    p = upscale_crop._preview({"target_mode": "scale_factor", "scale_factor": 2.0,
                               "method": "lanczos"}, c)
    assert p["ok"] and p["up"] == [2048, 3072], p


# ---------------------------------------------------------------- corners

@needs_comfy
def test_corner_anchors_land_on_their_exact_corner():
    # a 1:1 window in a landscape frame: slack on the x axis only
    tl = upscale_crop._crop_rect(2000, 1200, "1:1", "auto", 0, 0, "top-left", 0, 0)
    tr = upscale_crop._crop_rect(2000, 1200, "1:1", "auto", 0, 0, "top-right", 0, 0)
    assert tl[0] == 0.0 and tl[1] == 0.0, tl
    assert tr[0] == 0.4 and tr[1] == 0.0, tr          # x0 = slack_x = 800/2000

    # the same window in a portrait frame: slack on the y axis only
    bl = upscale_crop._crop_rect(1200, 2000, "1:1", "auto", 0, 0, "bottom-left", 0, 0)
    br = upscale_crop._crop_rect(1200, 2000, "1:1", "auto", 0, 0, "bottom-right", 0, 0)
    assert bl[1] == 0.4 and bl[0] == 0.0, bl          # y0 = slack_y = 800/2000
    assert br[1] == 0.4 and br[0] == 0.0, br


@needs_comfy
def test_corners_run_end_to_end_and_keep_the_size():
    base = None
    for pos in ("top-left", "top-right", "bottom-left", "bottom-right"):
        r = run(image=image(SRC_W, SRC_H), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_ratio="4:5", crop_orientation="landscape",
                crop_position=pos, multiple_of=1)
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        base = base or (w, h)
        assert (w, h) == base, (pos, w, h, base)


@needs_comfy
def test_corner_offsets_nudge_from_the_corner_and_clamp():
    # an 800x600 window in a 2000x1200 frame leaves slack on both axes,
    # which a named format (always filling one axis) cannot offer
    r = upscale_crop._crop_rect(2000, 1200, "custom", "auto", 800, 600,
                                "top-left", 300, 200)
    assert abs(r[0] - 300 / 2000) < 1e-9 and abs(r[1] - 200 / 1200) < 1e-9, r
    # a silly offset still clamps to the slack and never leaves the frame
    for pos in ("top-left", "top-right", "bottom-left", "bottom-right"):
        r = upscale_crop._crop_rect(2000, 1200, "custom", "auto", 800, 600,
                                    pos, -99999, 99999)
        assert 0 <= r[0] < r[2] <= 1.0001, (pos, r)
        assert 0 <= r[1] < r[3] <= 1.0001, (pos, r)


# ---------------------------------------------------------------- resize_condition

@needs_comfy
def _rc(**kw):
    kw.setdefault("image", image(SRC_W, SRC_H))
    kw.setdefault("method", "lanczos")
    r = run(**kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[3]


@needs_comfy
def test_downscale_only_skips_a_target_that_would_not_shrink():
    w, h, info = _rc(target_mode="shortest_side", shortest_side=2048,
                     resize_condition="downscale_only")
    assert (w, h) == (SRC_W, SRC_H), (w, h)
    assert "skipped resize (downscale_only)" in info, info

    w, h, _ = _rc(target_mode="longest_side", longest_side=4096,
                  resize_condition="downscale_only")
    assert (w, h) == (SRC_W, SRC_H), (w, h)

    # the source already has 1.57MP, so a 2MP budget would not shrink it
    w, h, info = _rc(target_mode="megapixels", megapixels=2.0, multiple_of=1,
                     resize_condition="downscale_only")
    assert (w, h) == (SRC_W, SRC_H), (w, h)
    assert "skipped resize (downscale_only)" in info, info


@needs_comfy
def test_downscale_only_still_resizes_a_real_downscale():
    w, h, info = _rc(target_mode="shortest_side", shortest_side=512,
                     resize_condition="downscale_only")
    assert (w, h) == (512, 768), (w, h)
    assert "skipped" not in info, info

    w, h, _ = _rc(target_mode="megapixels", megapixels=0.4, multiple_of=1,
                  resize_condition="downscale_only")
    assert w * h < SRC_W * SRC_H, (w, h)


@needs_comfy
def test_upscale_only_skips_a_target_that_would_not_enlarge():
    w, h, info = _rc(target_mode="shortest_side", shortest_side=512,
                     resize_condition="upscale_only")
    assert (w, h) == (SRC_W, SRC_H), (w, h)
    assert "skipped resize (upscale_only)" in info, info

    # 0.4MP is below the source's 1.57MP, so it would shrink -> skip
    w, h, info = _rc(target_mode="megapixels", megapixels=0.4, multiple_of=1,
                     resize_condition="upscale_only")
    assert (w, h) == (SRC_W, SRC_H), (w, h)
    assert "skipped resize (upscale_only)" in info, info

    # a real upscale still runs
    w, h, _ = _rc(target_mode="shortest_side", shortest_side=2048,
                  resize_condition="upscale_only")
    assert (w, h) == (2048, 3072), (w, h)


@needs_comfy
def test_resize_condition_is_ignored_by_the_relative_modes():
    """scale_percent and scale_factor are relative to the source, so the
    gate has nothing to gate -- even a downscale must run."""
    for mode, kw in [("scale_percent", dict(scale_percent=50.0)),
                     ("scale_factor", dict(scale_factor=0.5))]:
        for cond in ("downscale_only", "upscale_only"):
            w, h, info = _rc(target_mode=mode, resize_condition=cond, **kw)
            assert (w, h) == (512, 768), (mode, cond, w, h)
            assert "skipped" not in info, (mode, cond, info)


@needs_comfy
def test_invalid_and_none_conditions_fall_back_to_always():
    for bad in ("turbo", "", None, 7, "DOWNSCALE_ONLY"):
        w, h, info = _rc(target_mode="shortest_side", shortest_side=2048,
                         resize_condition=bad)
        assert (w, h) == (2048, 3072), (bad, w, h)
        assert "skipped" not in info, (bad, info)


@needs_comfy
def test_condition_skip_keeps_the_model_off_too(monkeypatch):
    """Skipping the resize must skip the model with it -- otherwise the
    failsafe would still burn VRAM to upscale what it then throws away."""
    received = {}
    use_fake_model(received, monkeypatch)
    r = run(image=image(SRC_W, SRC_H), method="upscaler model",
            upscale_model="fake.pth", target_mode="shortest_side",
            shortest_side=4096, resize_condition="downscale_only")
    assert "tensor" not in received, \
        "the model must not run when the resize is skipped"
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (SRC_W, SRC_H)
    assert "skipped resize (downscale_only)" in r[3]

    # control: the same target with 'always' must run the model
    r = run(image=image(SRC_W, SRC_H), method="upscaler model",
            upscale_model="fake.pth", target_mode="shortest_side",
            shortest_side=4096, resize_condition="always")
    assert "tensor" in received, "control: the model runs when nothing is skipped"
    assert (int(r[0].shape[2]), int(r[0].shape[1])) != (SRC_W, SRC_H)


@needs_comfy
def test_custom_downscale_only_keeps_a_small_source_intact():
    """The box is bigger than the source on both axes: no upscale, the box is
    clamped to the source, and the source leaves the node untouched -- never
    a 0px crop and never a silent upscale to the box."""
    w, h, info = _rc(target_mode="custom", target_width=1920, target_height=1080,
                     image=image(512, 384), resize_condition="downscale_only")
    assert (w, h) == (512, 384), (w, h)
    assert "skipped resize (downscale_only)" in info, info


@needs_comfy
def test_custom_upscale_only_cuts_the_box_from_the_untouched_source():
    """The box is smaller than the source: no downscale, but the box window
    is still cut out of the untouched source."""
    w, h, info = _rc(target_mode="custom", target_width=512, target_height=384,
                     image=image(1024, 1536), resize_condition="upscale_only")
    assert (w, h) == (512, 384), (w, h)
    assert "skipped resize (upscale_only)" in info, info


@needs_comfy
def test_custom_always_still_covers_and_crops():
    """The failsafe must not weaken the default: with 'always' a small source
    is still covered up to the box, exactly as before 1.1."""
    w, h, info = _rc(target_mode="custom", target_width=1920, target_height=1080,
                     image=image(512, 384), resize_condition="always")
    assert (w, h) == (1920, 1080), (w, h)
    assert "skipped" not in info, info


@needs_comfy
def test_preview_agrees_with_the_node_on_skipped_resizes():
    upscale_crop._remember("rc", image(SRC_W, SRC_H))
    c = upscale_crop._LAST_IMAGE["rc"]
    for data in [
        {"target_mode": "shortest_side", "shortest_side": 2048,
         "resize_condition": "downscale_only"},
        {"target_mode": "longest_side", "longest_side": 4096,
         "resize_condition": "downscale_only"},
        {"target_mode": "megapixels", "megapixels": 2.0,
         "resize_condition": "downscale_only"},
        {"target_mode": "shortest_side", "shortest_side": 512,
         "resize_condition": "upscale_only"},
        {"target_mode": "scale_factor", "scale_factor": 0.5,
         "resize_condition": "downscale_only"},
    ]:
        d = dict(data, method="lanczos", multiple_of=1)
        p = upscale_crop._preview(d, c)
        r = run(image=image(SRC_W, SRC_H), **d)
        assert p["up"] == [int(r[0].shape[2]), int(r[0].shape[1])], (data, p["up"])
        assert (p["skipped"] is None) == ("skipped resize" not in r[3]), (data, p)


@needs_comfy
def test_preview_agrees_on_the_clamped_custom_box():
    """The preview must show the source size and the clamped box -- not the
    box it will never produce -- so dragging and reading it stay honest."""
    upscale_crop._remember("rc2", image(512, 384))
    c = upscale_crop._LAST_IMAGE["rc2"]
    d = {"target_mode": "custom", "target_width": 1920, "target_height": 1080,
         "method": "lanczos", "resize_condition": "downscale_only"}
    p = upscale_crop._preview(d, c)
    r = run(image=image(512, 384), **d)
    assert p["up"] == [512, 384], p["up"]
    assert p["skipped"] == "downscale_only", p
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (512, 384)
    assert p["crop"] == [512, 384], (p["crop"], r[3])

# ---------------------------------------------------------------- model cache
#
# The pack keeps two models resident (LRUCache) instead of the reference's
# single slot, so switching between two favourite models while dragging a
# slider does not reload either of them.

@needs_comfy
def test_model_cache_is_a_two_slot_lru():
    cache = upscale_crop._MODEL_CACHE
    assert isinstance(cache, upscale_crop.LRUCache)
    assert cache.maxsize == 2
    cache["a.pth"] = object()
    cache["b.pth"] = object()
    cache["c.pth"] = object()
    assert len(cache) == 2
    assert "a.pth" not in cache
    assert set(cache) == {"b.pth", "c.pth"}


@needs_comfy
def test_load_upscale_model_serves_the_cache_without_reloading():
    """A cached name must never touch the loader (and never spandrel)."""
    sentinel = object()
    upscale_crop._MODEL_CACHE["cached.pth"] = sentinel
    assert upscale_crop._load_upscale_model("cached.pth") is sentinel


@needs_comfy
def test_three_model_switches_reload_only_the_oldest():
    """Simulate the loader's cache contract on the real cache: after a/b/c
    the node keeps b and c, so a round trip a->b->c->a reloads exactly a."""
    loaded = []

    def fake_loader(name):
        if name in upscale_crop._MODEL_CACHE:
            return upscale_crop._MODEL_CACHE[name]
        loaded.append(name)
        upscale_crop._MODEL_CACHE[name] = object()
        return upscale_crop._MODEL_CACHE[name]

    for name in ("a.pth", "b.pth", "c.pth", "a.pth"):
        fake_loader(name)
    assert loaded == ["a.pth", "b.pth", "c.pth", "a.pth"], loaded


# ---------------------------------------------------------------- model options

@needs_comfy
def test_model_options_uses_the_placeholder_outside_a_runtime(monkeypatch, tmp_path):
    """Without ComfyUI's folder_paths (or with an empty folder) the widget
    shows a placeholder, which the node reads as 'no model selected'."""
    def _no_folder_paths():
        raise ImportError("folder_paths is not available outside a ComfyUI runtime")

    with monkeypatch.context() as no_runtime:
        no_runtime.setattr(upscale_crop, "get_folder_paths", _no_folder_paths)
        assert upscale_crop._model_options() == ["(no models in models/upscale_models)"]

    # the same placeholder stands in when the folder simply is empty
    stub = make_folder_paths_stub(str(tmp_path / "in"), str(tmp_path / "out"),
                                  str(tmp_path / "temp"))
    stub.get_filename_list = lambda kind: []
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    assert upscale_crop._model_options() == ["(no models in models/upscale_models)"]


@needs_comfy
def test_model_options_lists_the_folder_ranked(monkeypatch, tmp_path):
    stub = make_folder_paths_stub(str(tmp_path / "in"), str(tmp_path / "out"),
                                  str(tmp_path / "temp"))
    stub.get_filename_list = lambda kind: [
        "zeta.pth", "realesrgan-x4plus.pth", "foo.pth", "4x-ultrasharp.pth"]
    monkeypatch.setitem(sys.modules, "folder_paths", stub)
    options = upscale_crop._model_options()
    assert set(options) == {"zeta.pth", "realesrgan-x4plus.pth", "foo.pth",
                            "4x-ultrasharp.pth"}
    # the workhorses rank before plain models, whatever the folder order is
    preferred = [o for o in options
                 if any(p in o.lower() for p in upscale_crop.PREFERRED_MODELS)]
    rest = [o for o in options if o not in preferred]
    assert preferred and rest
    assert max(options.index(p) for p in preferred) < min(options.index(o) for o in rest)


# ---------------------------------------------------------------- io
#
# Input/output contracts: what happens without an image, what the outputs
# carry, and that batches survive.

@needs_comfy
def test_no_image_is_a_clear_error():
    """The image input is required and the node has no placeholder to fall
    back on: an unconnected input must fail with a message that says so."""
    try:
        run(method="lanczos")
    except ValueError as exc:
        assert "connect an image" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("expected a ValueError naming the missing input")


@needs_comfy
def test_outputs_are_image_width_height_and_info():
    r = run(image=image(512, 512), method="lanczos", scale_factor=2.0)
    assert isinstance(r[0], torch.Tensor), "output 0 must be the image"
    assert (r[1], r[2]) == (int(r[0].shape[2]), int(r[0].shape[1])), r[1:3]
    assert isinstance(r[3], str) and r[3], r[3]


@needs_comfy
def test_batches_survive():
    r = run(image=image(256, 384, b=3), method="lanczos", scale_factor=2.0)
    assert r[0].shape[0] == 3, r[0].shape


@needs_comfy
def test_model_method_falls_back_when_nothing_is_selected():
    r = run(image=image(256, 384), method="upscaler model", scale_factor=2.0)
    assert tuple(r[0].shape)[1:3] == (768, 512), r[0].shape
    assert "lanczos" in r[3], r[3]


@needs_comfy
def test_model_is_skipped_on_a_downscale():
    r = run(image=image(1024, 1536), method="upscaler model", scale_factor=0.5,
            upscale_model="whatever.pth")
    assert tuple(r[0].shape)[1:3] == (768, 512), r[0].shape
    assert "downscale" in r[3], r[3]


@needs_comfy
def test_all_kernels_run():
    for k in ["lanczos", "bicubic", "bilinear", "nearest", "area"]:
        r = run(image=image(256, 384), method=k, scale_factor=1.5)
        assert r[0].shape[1] > 0, k


@needs_comfy
def test_none_values_from_the_frontend_do_not_crash():
    """ComfyUI can send an untouched widget as an explicit null."""
    r = run(image=image(256, 256), method="lanczos",
            target_mode=None, multiple_of=None, scale_factor=None,
            crop=None, crop_ratio=None, crop_orientation=None,
            crop_width=None, crop_height=None,
            target_width=None, target_height=None,
            crop_position=None, crop_offset_x=None, crop_offset_y=None,
            crop_seed=None)
    assert r[0].shape[0] == 1


# ---------------------------------------------------------------- preview
#
# The live preview must agree with the node. It exists to be trusted while
# dragging a slider, so the numbers it reports have to be the ones the graph
# will actually produce -- which is only true because both go through the
# same functions. These tests pin that.

@needs_comfy
def _cached():
    """What _remember() would have stashed after one run."""
    upscale_crop._remember("t", image(SRC_W, SRC_H))
    return upscale_crop._LAST_IMAGE["t"]


@needs_comfy
def test_remember_keeps_the_true_dimensions_not_the_thumbnail_ones():
    c = _cached()
    assert (c["w"], c["h"]) == (SRC_W, SRC_H), (c["w"], c["h"])
    assert c["png"], "expected a base64 thumbnail"


@needs_comfy
def test_cache_does_not_grow_without_bound():
    upscale_crop._LAST_IMAGE.clear()
    for i in range(40):
        upscale_crop._remember(f"n{i}", image(64, 64))
    assert len(upscale_crop._LAST_IMAGE) <= 32, len(upscale_crop._LAST_IMAGE)


@needs_comfy
def test_preview_size_matches_what_the_node_produces():
    c = _cached()
    for data in [
        {"target_mode": "scale_factor", "scale_factor": 2.0},
        {"target_mode": "scale_percent", "scale_percent": 137.0},
        {"target_mode": "shortest_side", "shortest_side": 999},
        {"target_mode": "longest_side", "longest_side": 2048},
        {"target_mode": "megapixels", "megapixels": 1.3},
    ]:
        for mult in [1, 8, 64]:
            d = dict(data, method="lanczos", multiple_of=mult, snap=True)
            p = upscale_crop._preview(d, c)
            r = run(image=image(SRC_W, SRC_H), **d)
            assert p["up"] == [int(r[0].shape[2]), int(r[0].shape[1])], \
                (d, mult, p["up"], r[0].shape)


@needs_comfy
def test_preview_crop_size_matches_what_the_node_produces():
    c = _cached()
    for ratio in ["4:5", "1:1", "16:9", "DIN", "custom"]:
        for pos in ["center", "top", "left"]:
            d = {"target_mode": "scale_factor", "scale_factor": 2.0,
                 "method": "lanczos", "multiple_of": 8, "snap": True, "crop": True,
                 "crop_ratio": ratio, "crop_orientation": "portrait",
                 "crop_width": 1200, "crop_height": 900,
                 "crop_position": pos, "crop_offset_x": 0, "crop_offset_y": 0}
            p = upscale_crop._preview(d, c)
            r = run(image=image(SRC_W, SRC_H), **d)
            assert p["crop"] == [int(r[0].shape[2]), int(r[0].shape[1])], \
                (ratio, pos, p["crop"], r[0].shape)


@needs_comfy
def test_preview_custom_crop_falls_back_to_the_target_box():
    """crop_width/crop_height absent from the payload: the node takes the
    target box, and the preview must say exactly that."""
    c = _cached()
    d = {"target_mode": "scale_factor", "scale_factor": 2.0,
         "method": "lanczos", "crop": True, "crop_ratio": "custom",
         "target_width": 1200, "target_height": 900}
    p = upscale_crop._preview(d, c)
    r = run(image=image(SRC_W, SRC_H), **d)
    assert p["crop"] == [int(r[0].shape[2]), int(r[0].shape[1])], (p["crop"], r[3])
    assert p["box"] == [1200, 900], p["box"]


@needs_comfy
def test_preview_rect_is_normalised_and_inside_the_frame():
    c = _cached()
    for pos in ["center", "top", "bottom", "left", "right"]:
        for off in [-4000, 0, 4000]:
            d = {"target_mode": "scale_factor", "scale_factor": 2.0,
                 "method": "lanczos", "multiple_of": 8, "snap": True, "crop": True,
                 "crop_ratio": "16:9", "crop_orientation": "landscape",
                 "crop_position": pos, "crop_offset_y": off}
            x0, y0, x1, y1 = upscale_crop._preview(d, c)["rect"]
            assert 0 <= x0 < x1 <= 1.0001, (pos, off, x0, x1)
            assert 0 <= y0 < y1 <= 1.0001, (pos, off, y0, y1)


@needs_comfy
def test_preview_frame_cuts_the_same_pixels_as_the_node():
    """The panel's promise: the window it draws is the window execute()
    cuts. Re-cut the resized image with the preview's rect and it must equal
    the node's output pixel for pixel, at every anchor, seed and snap."""
    c = _cached()
    img = image(SRC_W, SRC_H)     # one image for both sides of the comparison
    for ratio in ["4:5", "16:9", "1:1", "DIN", "custom", "21:9"]:
        for pos in ["center", "top", "bottom-left", "random"]:
            for snap, mult in [(False, 1), (True, 8), (True, 64)]:
                d = {"target_mode": "scale_factor", "scale_factor": 2.0,
                     "method": "lanczos", "multiple_of": mult, "snap": snap,
                     "crop": True, "crop_ratio": ratio,
                     "crop_orientation": "auto", "crop_position": pos,
                     "crop_seed": 7, "crop_width": 900, "crop_height": 700}
                p = upscale_crop._preview(d, c)
                r = run(image=img, **d)
                assert p["crop"] == [int(r[0].shape[2]), int(r[0].shape[1])], \
                    (ratio, pos, snap, p["crop"], r[0].shape)
                assert p["rect"] is not None
                work = upscale_crop._resize_image(img, int(p["up"][0]),
                                                  int(p["up"][1]), "lanczos")
                cut = upscale_crop._apply_rect(
                    work, tuple(p["rect"]), float(mult) if snap else 0.0)
                torch.testing.assert_close(cut, r[0])


@needs_comfy
def test_preview_reports_no_rect_when_crop_is_off():
    p = upscale_crop._preview({"target_mode": "scale_factor", "scale_factor": 2.0},
                              _cached())
    assert p["rect"] is None and p["crop"] is None


@needs_comfy
def test_preview_survives_rubbish_from_the_browser():
    """Widget values arrive as JSON, so anything can turn up -- nulls from
    untouched optionals, strings where numbers belong, unknown combo values."""
    c = _cached()
    for bad in [
        {"target_mode": None, "scale_factor": None, "multiple_of": None},
        {"target_mode": "nonsense", "method": 7, "multiple_of": "x"},
        {"target_mode": "scale_factor", "scale_factor": "two"},
        {"crop": True, "crop_ratio": "nope", "crop_offset_x": None, "crop_offset_y": None},
        {},
    ]:
        p = upscale_crop._preview(bad, c)
        assert p["ok"] and p["up"][0] > 0 and p["up"][1] > 0, (bad, p)


@needs_comfy
def test_random_ignores_fixed_offsets():
    """Random placement must ignore crop_offset_x/y so it is never biased into a corner."""
    src = gradient(512, 512)
    r1 = run(image=src, method="nearest", target_mode="scale_factor", scale_factor=1.0,
             crop=True, crop_ratio="custom", crop_width=256, crop_height=256,
             crop_position="random", crop_seed=42, crop_offset_x=0, crop_offset_y=0)
    r2 = run(image=src, method="nearest", target_mode="scale_factor", scale_factor=1.0,
             crop=True, crop_ratio="custom", crop_width=256, crop_height=256,
             crop_position="random", crop_seed=42, crop_offset_x=9999, crop_offset_y=-9999)
    torch.testing.assert_close(r1[0], r2[0])


@needs_comfy
def test_ram_oom_guard_and_warnings(monkeypatch):
    """RAM budget must warn above 35% of free RAM or >8192px, and block above 85% before allocating."""
    # Hard OOM block: simulate only 50 MB free RAM when requesting a ~60 MB peak resize
    monkeypatch.setattr(upscale_crop, "_available_ram_bytes", lambda: 50 * 1024 * 1024)
    with pytest.raises(ValueError, match="OOM risk"):
        run(image=image(1024, 1024), method="nearest", target_mode="scale_factor", scale_factor=1.0)
    p_oom = upscale_crop._preview(
        {"target_mode": "scale_factor", "scale_factor": 1.0, "method": "nearest"},
        {"png": None, "w": 1024, "h": 1024, "b": 1, "c": 3},
    )
    assert p_oom["oom"] is True and p_oom["warning"] and "OOM risk" in p_oom["warning"]

    # Soft RAM warning: simulate 120 MB free RAM for ~60 MB peak (50% > 35%, < 85%)
    monkeypatch.setattr(upscale_crop, "_available_ram_bytes", lambda: 120 * 1024 * 1024)
    r_warn = run(image=image(1024, 1024), method="nearest", target_mode="scale_factor", scale_factor=1.0)
    assert "WARNING: high RAM" in r_warn[3], r_warn[3]
    p_warn = upscale_crop._preview(
        {"target_mode": "scale_factor", "scale_factor": 1.0, "method": "nearest"},
        {"png": None, "w": 1024, "h": 1024, "b": 1, "c": 3},
    )
    assert p_warn["oom"] is False and p_warn["warning"] and "high RAM" in p_warn["warning"]

    # Large side (>8192 px) warning even with huge free RAM
    monkeypatch.setattr(upscale_crop, "_available_ram_bytes", lambda: 256 * 1024 ** 3)
    p_side = upscale_crop._preview(
        {"target_mode": "longest_side", "longest_side": 9000, "method": "lanczos"},
        {"png": None, "w": 1024, "h": 512, "b": 1, "c": 3},
    )
    assert p_side["oom"] is False and p_side["warning"] and "large output" in p_side["warning"]


@needs_comfy
def test_model_pass_falls_back_to_cpu_on_gpu_oom(monkeypatch):
    """When GPU OOMs even at tile=128, _model_pass must automatically fall back to CPU."""
    class _OomOnGpuModel(FakeModel):
        def __init__(self):
            super().__init__({})
            self.dev = "cpu"
            self.devices_tried = []

        def to(self, dev):
            self.dev = str(dev)
            return self

        def __call__(self, x):
            self.devices_tried.append(self.dev)
            if self.dev.startswith("cuda"):
                raise RuntimeError("CUDA out of memory")
            return super().__call__(x)

    fake_model = _OomOnGpuModel()

    class _FakeCudaDevice:
        type = "cuda"
        def __str__(self):
            return "cuda:0"

    class _FakeMM:
        OOM_EXCEPTION = RuntimeError
        @staticmethod
        def module_size(_m):
            return 0
        @staticmethod
        def free_memory(_need, _dev):
            pass
        @staticmethod
        def soft_empty_cache():
            pass

    monkeypatch.setattr(upscale_crop, "_load_upscale_model", lambda _n: fake_model)
    monkeypatch.setattr(upscale_crop, "_device", lambda: _FakeCudaDevice())
    monkeypatch.setattr(upscale_crop, "_cu", FakeCU)
    monkeypatch.setattr(upscale_crop, "_mm", _FakeMM)
    # Bypass Tensor.to(cuda) on CPU-only test environment by patching movedim/to via cpu passthrough
    orig_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, dev, *a, **kw: self if getattr(dev, "type", str(dev)) == "cuda" else orig_to(self, dev, *a, **kw))

    r = run(image=image(64, 64), method="upscaler model",
            upscale_model="fake.pth", target_mode="scale_factor", scale_factor=2.0)
    assert (r[1], r[2]) == (128, 128)
    assert any(d.startswith("cuda") for d in fake_model.devices_tried), "must try GPU first"
    assert "cpu" in fake_model.devices_tried, "must fall back to CPU"
    assert "CPU fallback (GPU VRAM full)" in r[3], r[3]



