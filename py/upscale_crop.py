"""Upscale Crop MM (V3 node API, comfy_api.latest) and its HTTP route.

MMUpscaleCrop ("Upscale Crop MM"): upscale and crop in one node, ported 1:1
from the standalone UpscaleCropUniversal node (v1.1.0):

  UPSCALE  sets the size and always keeps the aspect ratio. Percent, factor,
           shortest side, longest side or megapixels – five ways of asking
           for the same thing. Every mode works downwards as well. 'custom'
           takes a width x height box and gives exactly that (cover + crop).

  CROP     takes an exact format out of the result (off by default). It only
           ever cuts, never rescales, so nothing is resampled twice.

The upscale model is picked by a widget (read straight from
models/upscale_models, no loader node). The default method is 'lanczos', so
the model picker stays hidden until method is switched to 'upscaler model'.

Routes (answering {"ok": ...} JSON):
  /mm/upscale/preview – live size/crop numbers for the preview panel,
                        computed by the same functions the node runs
"""
import base64
import ctypes
import logging
import os
import random
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from comfy_api.latest import io

from .common import (
    CATEGORY,
    LRUCache,
    get_folder_paths,
    json_error,
)

# ComfyUI's own helpers. Guarded so the module still imports outside a running
# ComfyUI – tests run without ComfyUI and the model path falls back to kernels.
try:
    import comfy.model_management as _mm
    import comfy.utils as _cu
except Exception:
    _mm = None
    _cu = None


def _device():
    if _mm is not None:
        return _mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- sizing

TARGET_MODES = ["scale_factor", "scale_percent", "shortest_side",
                "longest_side", "megapixels", "custom"]

# 'custom' takes a target_width x target_height box and gives exactly that:
# it scales until the box is covered, then crops to it. The crop is not
# optional here, because the whole point of typing a resolution is getting
# that resolution out.
BOX_MODES = ("custom",)


def _as_mult(v):
    """multiple_of as a plain float, however it arrives.

    It used to be a combo of fixed strings, so a workflow saved back then
    restores "8" or "off" into what is now a number field. Both still mean
    what they always did rather than blowing up on the conversion.
    """
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("off", "none", ""):
            return 1.0
        try:
            v = float(v)
        except ValueError:
            return 1.0
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if v <= 0 else v


# Interpolation vocabulary follows ComfyUI core and the JPS crop nodes, so a
# name means the same thing here as it does everywhere else in a graph.
# 'lanczos' is first: it is the default and the safest general-purpose kernel.
METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "area", "upscaler model"]

# Classic formats, each written the way people actually say it – 4:5 is
# portrait, 16:9 is landscape – and crop_orientation normalises whichever
# one you pick, so the table needs no mirrored duplicates.
RATIOS = {
    "4:5":   (4.0, 5.0),
    "1:1":   (1.0, 1.0),
    "4:3":   (4.0, 3.0),
    "3:2":   (3.0, 2.0),
    "DIN":   (2.0 ** 0.5, 1.0),
    "16:10": (16.0, 10.0),
    "16:9":  (16.0, 9.0),
    "2:1":   (2.0, 1.0),
    "21:9":  (64.0, 27.0),
}
RATIO_NAMES = ["custom"] + list(RATIOS.keys())

CROP_POSITIONS = ["center", "top", "bottom", "left", "right",
                  "top-left", "top-right", "bottom-left", "bottom-right", "random"]
ORIENTATIONS = ["auto", "portrait", "landscape"]

# Sensible first pick for the model widget. ComfyUI shows whatever the folder
# holds, in folder order, which means the default is whatever happens to sort
# first – often a model you never wanted. These are the general-purpose
# workhorses most people reach for, so if one is installed it goes to the top
# of the list and becomes the default.
PREFERRED_MODELS = [
    "4x-ultrasharpv2", "4x_ultrasharp_v2", "ultrasharpv2", "ultrasharp v2",
    "4x-ultrasharp", "ultrasharp",
    "4x_foolhardy_remacri", "remacri",
    "4x_nmkd-siax", "siax",
    "realesrgan_x4plus", "realesrgan",
]


def _rank_models(names):
    """Preferred models first, everything else after, both alphabetical.

    Matching is loose on purpose – these files get renamed constantly and a
    user's copy of UltraSharp might be '4x-UltraSharpV2.pth' or
    '4x_UltraSharp_v2.safetensors'.
    """
    def key(n):
        low = n.lower()
        for i, pref in enumerate(PREFERRED_MODELS):
            if pref in low:
                return (0, i, low)
        return (1, 0, low)
    return sorted(names, key=key)


def _snap(n, mult):
    """Round to the nearest multiple, never below one multiple.

    Used for output sizes, which may legally end up larger than the source.
    Anything under 1 means no snapping at all – there is no such thing as
    rounding a pixel count to a multiple of a half.
    """
    n = max(1.0, float(n))
    if mult < 1.0:
        return max(1, int(round(n)))
    m = max(1, int(round(mult)))
    return max(m, int(round(n / m)) * m)


def _snap_inside(n, limit, mult):
    """Round n to the nearest multiple of mult, never above limit.

    A crop box has to stay inside the image, so unlike _snap the result is
    always a multiple of mult AND no bigger than limit. When the image is too
    small for a single multiple the whole limit is given back instead of an
    empty crop, and mult under 1 means plain rounding with the same clamp.
    """
    limit_i = max(1, int(limit))
    if mult < 1.0:
        return max(1, min(int(round(n)), limit_i))
    m = max(1, int(round(mult)))
    max_fit = (limit_i // m) * m
    if max_fit < m:
        return limit_i
    snapped = max(m, int(round(n / m)) * m)
    return min(snapped, max_fit)


def _scale_for(w, h, mode, percent, factor, shortest, longest, megapixels,
               box_w=0, box_h=0):
    """The factor a mode asks for, given a tensor's own size.

    Every mode returns a plain multiplier, so the caller applies it the same
    way regardless of which one the user picked – and a multiplier below 1.0
    is a downscale, which is deliberately not clamped away.
    """
    if mode == "scale_percent":
        return max(0.01, percent / 100.0)
    if mode == "scale_factor":
        return max(0.01, factor)
    if mode == "shortest_side":
        return shortest / max(1, min(w, h))
    if mode == "longest_side":
        return longest / max(1, max(w, h))
    if mode == "megapixels":
        return (megapixels * 1e6 / max(1, w * h)) ** 0.5
    if mode in BOX_MODES:
        # Cover: scale until the box is filled on both axes, then the crop
        # that always follows cuts it to size. Fitting inside instead would
        # leave bars, and this node never pads.
        return max(box_w / max(1, w), box_h / max(1, h))
    return 1.0


def _target(w, h, k, mult):
    return _snap(w * k, mult), _snap(h * k, mult)


# ---------------------------------------------------------------- RAM guard

_WARN_SIDE_PX = 8192
_WARN_RAM_FRAC = 0.35
_HARD_RAM_FRAC = 0.85


def _available_ram_bytes():
    """Live available physical RAM in bytes, or None if unavailable."""
    if os.name == "nt":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = _MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
        except Exception:  # noqa: BLE001
            pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def _check_ram_budget(batch, src_w, src_h, up_w, up_h, channels, method, k,
                      avail_bytes=None):
    """Estimate peak RAM for the resize and compare against live free RAM.

    Returns (est_bytes, avail_bytes, warning_text_or_None, is_oom_hard_limit).
    Peak factor 5x accounts for float32 output tensor, intermediate resample/PIL
    buffers, and ComfyUI output cache; a 4x neural model pass also holds an
    intermediate 4x RGB tensor before trimming to the target size.
    """
    b = max(1, int(batch))
    c = max(3, int(channels))
    out_bytes = b * int(up_w) * int(up_h) * c * 4
    model_bytes = (b * int(src_w) * 4 * int(src_h) * 4 * 3 * 4) if (
        method == "upscaler model" and k > 1.0) else 0
    est_bytes = max(out_bytes * 5, model_bytes * 3)
    avail = _available_ram_bytes() if avail_bytes is None else avail_bytes
    est_gb = est_bytes / (1024 ** 3)
    avail_gb = (avail / (1024 ** 3)) if avail else None
    hard_limit = bool(avail is not None and avail > 0 and est_bytes > _HARD_RAM_FRAC * avail)
    warn_ram = bool(avail is not None and avail > 0 and est_bytes > _WARN_RAM_FRAC * avail)
    warn_side = max(int(up_w), int(up_h)) > _WARN_SIDE_PX
    warning = None
    if hard_limit:
        warning = (f"OOM risk: ~{est_gb:.1f} GB RAM needed "
                   f"({avail_gb:.1f} GB free) for {up_w}x{up_h}")
    elif warn_ram:
        warning = (f"high RAM: ~{est_gb:.1f} GB est. "
                   f"({avail_gb:.1f} GB free) for {up_w}x{up_h}")
    elif warn_side:
        warning = (f"large output: {up_w}x{up_h} "
                   f"(~{est_bytes / (1024 ** 2):.0f} MB est.)")
    return est_bytes, avail, warning, hard_limit


# ---------------------------------------------------------------- failsafe
#
# An absolute target is a number the user typed, and sometimes that number
# would not really change the image – asking for "longest side 1024" on a
# 1024px frame, or a megapixel budget below what the source already has. The
# resize condition decides what to do with such a target: always resizes,
# downscale_only only ever shrinks, upscale_only only ever grows. "always"
# is the default and the safe fall-back for anything unknown.
#
# Only the absolute modes take effect: scale_percent and scale_factor are
# relative to the source, so their target is whatever the source says it is
# and there is nothing to gate.

RESIZE_CONDITIONS = ["always", "downscale_only", "upscale_only"]
ABSOLUTE_MODES = ("shortest_side", "longest_side", "megapixels", "custom")


def _normalize_condition(value):
    """Whatever arrives from the widget or a saved workflow, back to one of
    the three values the skip rules understand. Nothing else can fall
    through and change behaviour."""
    return value if value in RESIZE_CONDITIONS else "always"


def _should_skip_resize(condition, mode, tw, th, sw, sh):
    """Whether the target leaves the source untouched, per the condition.

    downscale_only skips a target that would not shrink the image,
    upscale_only one that would not enlarge it. In megapixels mode the
    comparison is on pixel count, because the target keeps the source's
    shape only up to rounding and a count says what actually happens.
    """
    if condition == "always" or mode not in ABSOLUTE_MODES:
        return False
    if mode == "megapixels":
        if condition == "downscale_only":
            return tw * th >= sw * sh
        return tw * th <= sw * sh
    if condition == "downscale_only":
        return tw >= sw and th >= sh
    return tw <= sw and th <= sh


# ---------------------------------------------------------------- crop rect

def _crop_rect(w, h, ratio, orientation, cw, ch, position, off_x, off_y,
               seed=0, mult=0.0):
    """The surviving region as normalised 0..1 (x0, y0, x1, y1).

    The box is snapped (when mult >= 1) BEFORE the slack is computed, and the
    anchor is an integer pixel count. execute() and _preview() therefore cut
    the identical pixels – snapping the crop after the fact used to push
    bottom/right/centre placements off the edge by up to half a multiple.
    """
    if ratio == "custom":
        box_w, box_h = max(1.0, float(cw)), max(1.0, float(ch))
        # A box bigger than the image shrinks PROPORTIONALLY to the largest
        # one that fits, keeping the shape you asked for. Clamping each axis
        # on its own instead would hand back a window of some other shape
        # entirely – ask for a square from a portrait frame and you would get
        # the whole portrait frame. Every named format already takes the
        # largest window of its shape that fits; custom now does the same.
        fit = min(w / box_w, h / box_h, 1.0)
        box_w *= fit
        box_h *= fit
    else:
        rw, rh = RATIOS[ratio]
        ar = rw / rh
        if ratio != "1:1":
            # Normalise to the requested orientation, whichever way the table
            # happens to write this format. 'auto' follows the source, which
            # is what a mixed batch of portrait and landscape frames needs –
            # a fixed orientation would cut a thin strip out of half of them.
            # A square source is the exception: there is no "other way round"
            # to follow, so the format keeps the table's own form – 4:5 stays
            # portrait and 16:9 stays landscape.
            if orientation == "auto" and w == h:
                want_wide = None
            elif orientation == "auto":
                want_wide = w > h
            else:
                want_wide = orientation == "landscape"
            if want_wide is not None and want_wide != (ar >= 1.0):
                ar = 1.0 / ar
        # largest window of that shape that fits inside
        if (w / max(1.0, float(h))) > ar:
            box_h = float(h)
            box_w = box_h * ar
        else:
            box_w = float(w)
            box_h = box_w / ar

    # Snap and clamp to the frame BEFORE the slack, so the anchor is placed
    # against the box that actually leaves the node.
    box_w = float(_snap_inside(box_w, w, mult))
    box_h = float(_snap_inside(box_h, h, mult))

    slack_x = float(w) - box_w
    slack_y = float(h) - box_h

    # The anchor sets where the window starts; the two offsets then move it on
    # their own axis. Separate x and y because one offset could only ever move
    # along whichever axis the anchor left free – with 'center' that meant no
    # horizontal nudge at all, which is exactly when you most want one.
    if position == "left":
        x0, y0 = 0.0, slack_y / 2.0
    elif position == "right":
        x0, y0 = slack_x, slack_y / 2.0
    elif position == "top":
        x0, y0 = slack_x / 2.0, 0.0
    elif position == "bottom":
        x0, y0 = slack_x / 2.0, slack_y
    elif position == "top-left":
        x0, y0 = 0.0, 0.0
    elif position == "top-right":
        x0, y0 = slack_x, 0.0
    elif position == "bottom-left":
        x0, y0 = 0.0, slack_y
    elif position == "bottom-right":
        x0, y0 = slack_x, slack_y
    elif position == "random":
        # Seeded, not free-running: a fixed seed has to give the same window
        # every time or the preview would show one crop and the run produce
        # another, and re-queueing a graph would never reproduce a result you
        # liked. Turn crop_seed's control_after_generate to 'randomize' for a
        # fresh window per run.
        rng = random.Random(int(seed))
        x0 = rng.uniform(0.0, slack_x)
        y0 = rng.uniform(0.0, slack_y)
    else:  # center
        x0, y0 = slack_x / 2.0, slack_y / 2.0

    # positive x moves right, positive y moves down; clamped so the window
    # can never leave the image, and rounded to whole pixels so both the
    # preview and the crop cut the same pixels. Random placement ignores fixed
    # pixel offsets so the distribution is never biased into a corner.
    if position == "random":
        off_x, off_y = 0.0, 0.0
    x0 = float(int(round(min(max(0.0, x0 + off_x), slack_x))))
    y0 = float(int(round(min(max(0.0, y0 + off_y), slack_y))))

    return (x0 / w, y0 / h, (x0 + box_w) / w, (y0 + box_h) / h)


def _apply_rect(t, rect, mult, channels_last=True):
    """Cut a normalised rect out of a B,H,W,C or B,C,H,W tensor.

    The rect comes from _crop_rect, where the box is already snapped to the
    same multiple and the anchor is an integer pixel count, so snapping here
    with _snap_inside reproduces exactly those pixels – the crop the node
    cuts is the one the preview framed, with no drift.
    """
    h = t.shape[1] if channels_last else t.shape[2]
    w = t.shape[2] if channels_last else t.shape[3]

    x0 = int(round(rect[0] * w))
    y0 = int(round(rect[1] * h))
    cw = int(_snap_inside((rect[2] - rect[0]) * w, w, mult))
    ch = int(_snap_inside((rect[3] - rect[1]) * h, h, mult))

    # defensive: keep the window inside even if the rect came from elsewhere
    x0 = max(0, min(x0, w - cw))
    y0 = max(0, min(y0, h - ch))

    if channels_last:
        return t[:, y0:y0 + ch, x0:x0 + cw, :]
    return t[:, :, y0:y0 + ch, x0:x0 + cw]


# ---------------------------------------------------------------- resampling

# torch has no lanczos, so the image path detours through PIL for that one
# kernel. Everything else stays in torch on the GPU.
_TORCH_MODES = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "nearest": "nearest-exact",
    "area": "area",
}


def _resize_image(img, tw, th, kernel):
    """B,H,W,C float 0..1 -> resized the same way."""
    if img.shape[1] == th and img.shape[2] == tw:
        return img

    if kernel == "lanczos":
        out = []
        for i in range(img.shape[0]):
            a = (img[i].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            p = Image.fromarray(a).resize((tw, th), Image.LANCZOS)
            out.append(torch.from_numpy(np.asarray(p).astype(np.float32) / 255.0))
        return torch.stack(out, 0).to(img.device)

    mode = _TORCH_MODES.get(kernel, "bicubic")
    x = img.movedim(-1, 1)
    kw = {} if mode in ("nearest-exact", "area") else {"align_corners": False}
    x = F.interpolate(x, size=(th, tw), mode=mode, **kw)
    return x.movedim(1, -1).clamp(0, 1)


# ---------------------------------------------------------------- the model

# Two slots, not one: dragging the model widget between two files would
# otherwise reload one of them on every other drag, and a 20 s model load in
# the middle of tuning sliders feels like a freeze.
_MODEL_CACHE = LRUCache(2)


def _load_upscale_model(name):
    """Load and keep up to two models, so dragging a slider does not reload them."""
    model = _MODEL_CACHE.get(name)
    if model is not None:
        return model

    from spandrel import ModelLoader, ImageModelDescriptor

    path = get_folder_paths().get_full_path_or_raise("upscale_models", name)
    sd = _cu.load_torch_file(path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = _cu.state_dict_prefix_replace(sd, {"module.": ""})
    model = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(f"{name} is not a single-image upscale model.")

    _MODEL_CACHE[name] = model
    return model


def _model_pass(img, name, notes=None):
    """One pass at the model's native scale, tiled, shrinking tiles on OOM.

    Processes batches one image at a time to keep VRAM flat, prefers GPU with
    automatic tile halving (512 -> 256 -> 128), and falls back to tiled CPU
    execution if GPU VRAM is completely full instead of crashing or losing
    neural upscaler quality.
    """
    # A 4-channel image carries an alpha channel the pixel model can neither
    # see nor keep. It is split off, the RGB goes through the model, and the
    # alpha is resized bilinearly onto the model's exact output size before
    # being joined back – so the resize and crop that follow work on a clean
    # RGBA tensor, and the 3-channel path is untouched.
    alpha = None
    if img.shape[-1] == 4:
        alpha = img[..., 3:4]
        img = img[..., :3]

    model = _load_upscale_model(name)
    device = _device()
    cpu = torch.device("cpu")
    oom_exc = getattr(_mm, "OOM_EXCEPTION", RuntimeError) if _mm else RuntimeError

    def _run_on(dev, start_tile):
        model.to(dev)
        tile, overlap = start_tile, 32
        while True:
            try:
                frames = []
                for idx in range(img.shape[0]):
                    x = img[idx:idx + 1].movedim(-1, -3).to(dev)
                    steps = _cu.get_tiled_scale_steps(
                        x.shape[3], x.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = _cu.ProgressBar(steps)
                    y = _cu.tiled_scale(x, lambda a: model(a), tile_x=tile, tile_y=tile,
                                        overlap=overlap, upscale_amount=model.scale, pbar=pbar)
                    frames.append(y.to("cpu"))
                return torch.cat(frames, dim=0)
            except oom_exc:
                tile //= 2
                if tile < 128:
                    raise

    if _mm is not None and device.type != "cpu":
        single_frame = img[:1]
        need = _mm.module_size(model.model)
        need += (512 * 512 * 3) * single_frame.element_size() * max(model.scale, 1.0) * 384.0
        need += single_frame.nelement() * single_frame.element_size()
        _mm.free_memory(need, device)

    try:
        out = _run_on(device, 512)
    except oom_exc:
        if device.type == "cpu":
            model.to("cpu")
            raise
        # GPU VRAM full even at 128px tiles: clear cache and finish on CPU
        model.to("cpu")
        if _mm is not None and hasattr(_mm, "soft_empty_cache"):
            _mm.soft_empty_cache()
        out = _run_on(cpu, 512)
        if notes is not None:
            notes.append("CPU fallback (GPU VRAM full)")

    model.to("cpu")
    out = out.movedim(-3, -1).clamp(0, 1)

    if alpha is not None:
        # Bilinear keeps the edge soft the way the RGB trim does; clamping
        # guards against interpolation ringing past 0..1 at hard edges.
        a = alpha.movedim(-1, -3).float().to(out.device)
        if a.shape[-2] != out.shape[1] or a.shape[-1] != out.shape[2]:
            a = F.interpolate(a, size=(out.shape[1], out.shape[2]),
                              mode="bilinear", align_corners=False,
                              antialias=True)
        a = a.clamp(0, 1).movedim(-3, -1)
        out = torch.cat([out, a], dim=-1)

    return out, float(model.scale)


# ---------------------------------------------------------------- node


def _model_options():
    """Model widget options: everything in models/upscale_models, the
    general-purpose workhorses first (see _rank_models). Outside a ComfyUI
    runtime (unit tests) or with an empty folder the placeholder stands in,
    which the node treats as 'no model selected' and falls back to lanczos."""
    try:
        names = list(get_folder_paths().get_filename_list("upscale_models") or [])
    except Exception:  # noqa: BLE001 – no ComfyUI runtime (unit tests)
        names = []
    models = _rank_models(names)
    if not models:
        models = ["(no models in models/upscale_models)"]
    return models


class MMUpscaleCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        ctrl_rand = getattr(getattr(io, "ControlAfterGenerate", None), "randomize", True)
        model_opts = _model_options()
        return io.Schema(
            node_id="MMUpscaleCrop",
            display_name="Upscale Crop MM",
            category=CATEGORY,
            description=(
                "Upscale and crop in one node. UPSCALE sets the size -- ask for it as a "
                "percent, a factor, a shortest or longest side, or a megapixel count, "
                "whichever is natural for what you are doing. Those five keep the aspect "
                "ratio. 'custom' takes a width x height box instead and gives you exactly "
                "that: it covers the box and crops to it, so the resolution you type is "
                "the resolution that comes out. Every mode works downwards too, so this "
                "is also your downscaler.\n\n"
                "CROP is off by default and only ever cuts, never rescales, so nothing is "
                "resampled twice. Switch it on and pick a classic format -- 4:5, 1:1, 3:2, "
                "DIN, 16:9 and the rest (or crop_width x crop_height for an exact shape) "
                "-- with crop_orientation deciding which way round it sits ('auto' follows "
                "the frame, which is what a mixed batch of portrait and landscape shots "
                "needs), and crop_position plus crop_offset_x / crop_offset_y deciding "
                "what survives.\n\n"
                "THE MODEL IS BUILT IN. method defaults to 'lanczos'; switch it to "
                "'upscaler model' and upscale_model picks the file straight from your "
                "models/upscale_models folder -- no separate loader node to wire up. One "
                "pass at the model's native scale, then a single resize onto the target, "
                "so a 4x model hitting a 2x target does not leave you with a needlessly "
                "huge intermediate. Tiling is automatic and halves itself if VRAM runs "
                "short.\n\n"
                "EXACT SIZE BY DEFAULT. With snap off you get exactly the size you asked "
                "for. Switch snap on to round the result to multiple_of (8 suits SD, "
                "SDXL and Flux).\n\n"
                "IF NOTHING SEEMS TO HAPPEN. Only the widgets that currently apply are "
                "shown, so switching target_mode swaps the size field and switching "
                "method to 'upscaler model' reveals the model picker -- a value you set "
                "earlier is not gone, just not in use. The info output always names the "
                "mode and method that actually ran."),
            search_aliases=["mm", "metamountain", "mm nodepack", "upscale mm", "crop mm"],
            inputs=[
                io.Image.Input("image",
                               tooltip="The image to resize and optionally crop."),
                io.Combo.Input("target_mode", options=list(TARGET_MODES), default="scale_factor",
                               tooltip="How you want to ask for the size. The first five keep the aspect ratio. 'custom' takes a target_width x target_height box and gives you exactly that -- it covers the box and crops to it, so the resolution you type is the resolution you get. All of them work downwards as well as up."),
                io.Float.Input("scale_factor", default=2.0, min=0.05, max=8.0, step=0.05,
                               tooltip="Used by target_mode scale_factor. The same thing as percent, expressed the other way."),
                io.Float.Input("scale_percent", default=200.0, min=1.0, max=800.0, step=5.0,
                               tooltip="Used by target_mode scale_percent. 200 doubles, 50 halves."),
                io.Int.Input("shortest_side", default=1536, min=16, max=16384, step=8,
                             tooltip="Used by target_mode shortest_side. Pins min(width, height); the other side follows the aspect ratio."),
                io.Int.Input("longest_side", default=2048, min=16, max=16384, step=8,
                             tooltip="Used by target_mode longest_side. Pins max(width, height)."),
                io.Float.Input("megapixels", default=2.0, min=0.05, max=64.0, step=0.05,
                               tooltip="Used by target_mode megapixels. Pins the total pixel count; the shape is unchanged."),
                io.Int.Input("target_width", default=1920, min=16, max=16384, step=8,
                             tooltip="Width of the box. target_mode 'custom' gives you exactly this."),
                io.Int.Input("target_height", default=1080, min=16, max=16384, step=8,
                             tooltip="Height of the box. target_mode 'custom' gives you exactly this."),
                io.Combo.Input("resize_condition", options=list(RESIZE_CONDITIONS), default="always",
                               tooltip="Failsafe switch for the absolute target modes (shortest_side, longest_side, megapixels, custom): downscale_only leaves the image untouched when the target is not smaller, upscale_only when it is not larger -- the resize and the upscale model are skipped together and the image leaves the node exactly as it came in. In custom mode a box bigger than the source keeps the source intact instead of upscaling to it. scale_percent and scale_factor are relative and always resize, so the widget is hidden for them. Anything unknown falls back to 'always'."),
                io.Combo.Input("method", options=list(METHODS), default="lanczos",
                               tooltip="How to resample. Default is lanczos, the sharpest plain kernel. 'upscaler model' uses the upscale_model below; area is the best choice for heavy downscaling."),
                io.Combo.Input("upscale_model", options=model_opts, default=model_opts[0],
                               tooltip="Used when method is 'upscaler model'. Read straight from models/upscale_models -- no loader node needed. UltraSharp, Remacri, Siax and RealESRGAN are listed first if you have them, since they are the usual general-purpose picks."),
                io.Boolean.Input("snap", default=False, label_on="snap to multiple_of", label_off="exact size",
                                 tooltip="Off: you get exactly the size you asked for. On: the result is rounded to a multiple of multiple_of (e.g. 8 for SD, SDXL and Flux VAEs). ComfyUI's VAE Encode crops a few pixels itself when a size is not a multiple, so this is optional."),
                io.Int.Input("multiple_of", default=8, min=1, max=512, step=1,
                             tooltip="Used when snap is on: the result is rounded to a multiple of this. 8 suits SD, SDXL and Flux; 64 for some tiling and video workflows."),
                io.Boolean.Input("crop", default=False,
                                 tooltip="Off = pure upscaler. On = cut a format out of the result afterwards. Always on for target_mode 'custom', which has to crop to reach the box you typed."),
                io.Combo.Input("crop_ratio", options=list(RATIO_NAMES), default="4:5",
                               tooltip="The format to cut. Always takes the largest window of that shape that fits. 'custom' uses the crop_width x crop_height box below. Ignored by target_mode 'custom' -- that already cuts to its box, so the size you typed is the size you get."),
                io.Int.Input("crop_width", default=1024, min=16, max=16384, step=8,
                             tooltip="Width of the crop window for crop_ratio 'custom'."),
                io.Int.Input("crop_height", default=1024, min=16, max=16384, step=8,
                             tooltip="Height of the crop window for crop_ratio 'custom'."),
                io.Combo.Input("crop_orientation", options=list(ORIENTATIONS), default="auto",
                               tooltip="Which way round the format sits. 'auto' follows the source, so a portrait frame gets a portrait crop and a landscape one gets landscape -- the right choice for a mixed batch. Ignored for 1:1 and for custom."),
                io.Combo.Input("crop_position", options=list(CROP_POSITIONS), default="center",
                               tooltip="Where the window sits before the offsets nudge it. 'random' places it anywhere that fits, driven by crop_seed -- for varying the framing across a batch or a dataset."),
                io.Int.Input("crop_offset_x", default=0, min=-8192, max=8192, step=8,
                             tooltip="Nudge the window sideways in px. Positive moves right. Clamped so it can never leave the image."),
                io.Int.Input("crop_offset_y", default=0, min=-8192, max=8192, step=8,
                             tooltip="Nudge the window up or down in px. Positive moves down. Clamped so it can never leave the image. You can also just drag the window in the preview."),
                io.Int.Input("crop_seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=ctrl_rand,
                             tooltip="Only used when crop_position is 'random'. The same seed always gives the same window, so a framing you liked can be reproduced; set control_after_generate to randomize for a new one each run."),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The resized (and optionally cropped) image."),
                io.Int.Output("width", tooltip="Final width in px."),
                io.Int.Output("height", tooltip="Final height in px."),
                io.String.Output("info", tooltip="What actually happened: sizes, the method used, the crop applied, and any fallback."),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, upscale_model=None, **kwargs):
        # The model list differs on every machine, so a workflow saved where a
        # model was installed would fail combo validation on a machine without
        # it. execute() falls back to lanczos instead, which is the behaviour
        # the user actually wants, so the name is not validated.
        return True

    @classmethod
    def execute(cls, image=None, target_mode="scale_factor", scale_factor=2.0,
                scale_percent=200.0, shortest_side=1536, longest_side=2048,
                megapixels=2.0, target_width=1920, target_height=1080,
                resize_condition="always", method="lanczos",
                upscale_model=None, snap=False, multiple_of=8,
                crop=False, crop_ratio="4:5", crop_width=None, crop_height=None,
                crop_orientation="auto", crop_position="center",
                crop_offset_x=0, crop_offset_y=0, crop_seed=0):

        if image is None:
            raise ValueError(
                "Upscale Crop MM: connect an image – there is nothing to resize.")

        # ComfyUI's frontend can send an untouched widget as an explicit None
        # instead of omitting it, which otherwise blows up in the float()/int()
        # conversion before this ever runs. crop_width and crop_height are the
        # exception: None there means "fall back to the target box".
        if target_mode is None: target_mode = "scale_factor"
        if scale_factor is None: scale_factor = 2.0
        if scale_percent is None: scale_percent = 200.0
        if shortest_side is None: shortest_side = 1536
        if longest_side is None: longest_side = 2048
        if megapixels is None: megapixels = 2.0
        if target_width is None: target_width = 1920
        if target_height is None: target_height = 1080
        if resize_condition is None: resize_condition = "always"
        if method is None: method = "lanczos"
        if snap is None: snap = False
        if multiple_of is None: multiple_of = 8
        if crop is None: crop = False
        if crop_ratio is None: crop_ratio = "4:5"
        if crop_orientation is None: crop_orientation = "auto"
        if crop_position is None: crop_position = "center"
        if crop_offset_x is None: crop_offset_x = 0
        if crop_offset_y is None: crop_offset_y = 0
        if crop_seed is None: crop_seed = 0

        hidden = getattr(cls, "hidden", None)
        uid = getattr(hidden, "unique_id", None) if hidden is not None else None
        if uid is not None:
            _remember(str(uid), image)

        mult = _as_mult(multiple_of) if snap else 0.0     # 0 = no snapping
        # the failsafe rule, normalised once; relative target modes treat it
        # as "always" inside _should_skip_resize
        condition = _normalize_condition(resize_condition)
        notes = []

        # A box mode scales to cover target_width x target_height and then
        # crops to that exact box -- the numbers you typed are the output size,
        # so the crop is not optional here and it overrides any manual crop
        # rather than stacking a second one on top.
        box = None
        exact_box = None
        if target_mode in BOX_MODES:
            box = (_snap(target_width, mult), _snap(target_height, mult))
            crop_ratio = "custom"
            exact_box = box
            crop = True

        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        k = _scale_for(src_w, src_h, target_mode, scale_percent, scale_factor,
                       shortest_side, longest_side, megapixels, *(box or (0, 0)))
        if exact_box:
            # cover the box exactly, no snapping in between -- the crop right
            # after lands on the already-snapped box anyway
            tw = max(exact_box[0], int(round(src_w * k)))
            th = max(exact_box[1], int(round(src_h * k)))
        else:
            tw, th = _target(src_w, src_h, k, mult)

        # The failsafe rule: an absolute target that would not really change
        # the size leaves the source untouched -- no resize and no model pass.
        # In custom mode a box the source cannot fill is clamped to the source
        # so the crop keeps it intact instead of upscaling to the box; a box
        # the source can fill is simply cut out of the untouched source.
        skipped = (condition != "always"
                   and _should_skip_resize(condition, target_mode,
                                           tw, th, src_w, src_h))
        if skipped:
            tw, th = src_w, src_h
            if target_mode in BOX_MODES:
                exact_box = (min(box[0], src_w), min(box[1], src_h))
            notes.append(f"skipped resize ({condition})")

        if not skipped:
            _, _, ram_warn, ram_oom = _check_ram_budget(
                int(image.shape[0]), src_w, src_h, tw, th,
                int(image.shape[-1]), method, k)
            if ram_oom:
                raise ValueError(
                    f"Upscale Crop MM: {ram_warn} – reduce target size to prevent system OOM.")
            if ram_warn:
                notes.append(f"WARNING: {ram_warn}")

        kernel = method
        work = image
        if not skipped and method == "upscaler model":
            if k <= 1.0:
                kernel = "lanczos"
                notes.append("model skipped, target is a downscale")
            elif _cu is None:
                kernel = "lanczos"
                notes.append("comfy.utils unavailable, fell back to lanczos")
            elif not upscale_model or upscale_model.startswith("(no models"):
                kernel = "lanczos"
                notes.append("no upscale model selected, fell back to lanczos")
            else:
                try:
                    work, native = _model_pass(image, upscale_model, notes=notes)
                    kernel = "lanczos"      # the trim onto the exact target
                    notes.append(f"{upscale_model} x{native:g}")
                except Exception as exc:
                    work = image
                    kernel = "lanczos"
                    notes.append(f"model failed ({type(exc).__name__}), fell back to lanczos")

        out_image = _resize_image(work, tw, th, kernel)
        img_dims = (tw, th)

        # ---- crop, in normalised coordinates -------------------------------
        crop_note = "crop off"
        if crop:
            ref_w, ref_h = img_dims
            if exact_box:
                # a box mode hands over its already-snapped box
                box_w, box_h = exact_box
            elif crop_ratio == "custom":
                # a dedicated crop box, with the target box as fallback
                box_w = crop_width if crop_width is not None else target_width
                box_h = crop_height if crop_height is not None else target_height
            else:
                box_w, box_h = 0, 0
            rect = _crop_rect(ref_w, ref_h, crop_ratio, crop_orientation,
                              box_w, box_h, crop_position,
                              crop_offset_x, crop_offset_y, crop_seed, mult=mult)
            out_image = _apply_rect(out_image, rect, mult, channels_last=True)
            img_dims = (int(out_image.shape[2]), int(out_image.shape[1]))
            shape = crop_ratio if crop_ratio == "custom" else (
                crop_ratio + ("" if crop_ratio == "1:1" else " " + crop_orientation))
            crop_note = f"crop {shape} {crop_position}"
            if crop_position == "random":
                crop_note += f"({crop_seed})"
            elif crop_offset_x or crop_offset_y:
                crop_note += f" {crop_offset_x:+d}{crop_offset_y:+d}"

        w = int(out_image.shape[2])
        h = int(out_image.shape[1])

        parts = [f"image {src_w}x{src_h} -> {w}x{h}"]
        if target_mode in BOX_MODES:
            parts.append(f"{target_mode} {box[0]}x{box[1]}")
        else:
            parts.append(target_mode)
        parts.append(method)
        parts.append("no snap" if mult < 1 else f"/{int(round(mult))}")
        parts.append(crop_note)
        parts.extend(notes)

        return io.NodeOutput(out_image, w, h, " | ".join(parts))


NODE_LIST = [MMUpscaleCrop]


# ---------------------------------------------------------------- live preview
#
# The node's preview needs an image to draw, and the only place one exists is
# inside execute(). So each run stashes a small copy keyed by the node's id,
# and the route works from that. Before the graph has ever run there is
# nothing cached -- the route says so rather than inventing something.
#
# The preview deliberately does NOT re-render the image. An upscale looks
# identical at thumbnail size, so re-running the resize would cost real time
# to show nothing new. What actually changes with the sliders are the numbers
# and the crop rectangle, so the route returns those -- computed by the very
# same functions execute() uses, which is what keeps the preview honest -- and
# the browser draws the frame over the cached thumbnail.

_LAST_IMAGE = {}
_CACHE_EDGE = 320


def _png_b64(arr):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _remember(node_id, image):
    """Keep a small copy of the last input for the preview."""
    try:
        x = image[:1].movedim(-1, 1).float()
        h, w = int(x.shape[-2]), int(x.shape[-1])
        edge = max(h, w)
        if edge > _CACHE_EDGE:
            k = _CACHE_EDGE / edge
            x = F.interpolate(x, size=(max(1, round(h * k)), max(1, round(w * k))),
                              mode="area")
        a = (x[0].movedim(0, -1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        _LAST_IMAGE[str(node_id)] = {
            "png": _png_b64(a), "w": w, "h": h,
            "b": int(image.shape[0]), "c": int(image.shape[-1]),
        }
        if len(_LAST_IMAGE) > 32:               # do not grow without bound
            _LAST_IMAGE.pop(next(iter(_LAST_IMAGE)))
    except Exception:
        pass


def _preview(data, cached):
    """Everything the panel shows, from the same maths the node runs."""
    src_w, src_h = cached["w"], cached["h"]

    def num(key, default):
        try:
            v = data.get(key)
            return default if v is None else float(v)
        except (TypeError, ValueError):
            return default

    def txt(key, default, valid):
        v = data.get(key)
        return v if isinstance(v, str) and v in valid else default

    mode = txt("target_mode", "scale_factor", TARGET_MODES)
    method = txt("method", "lanczos", METHODS)
    mult = _as_mult(data.get("multiple_of", 8)) if data.get("snap") is True else 0.0

    box = None
    exact_box = None
    if mode in BOX_MODES:
        box = (_snap(num("target_width", 1920), mult),
               _snap(num("target_height", 1080), mult))
        exact_box = box

    k = _scale_for(src_w, src_h, mode, num("scale_percent", 200.0),
                   num("scale_factor", 2.0), num("shortest_side", 1536),
                   num("longest_side", 2048), num("megapixels", 2.0),
                   *(box or (0, 0)))
    if exact_box:
        up_w = max(exact_box[0], int(round(src_w * k)))
        up_h = max(exact_box[1], int(round(src_h * k)))
    else:
        up_w, up_h = _target(src_w, src_h, k, mult)

    # The very same failsafe the node applies, so a target that will be
    # skipped shows the source size -- and in custom mode the clamped box --
    # instead of promising a resize that never happens.
    condition = txt("resize_condition", "always", RESIZE_CONDITIONS)
    skipped = None
    if condition != "always" and _should_skip_resize(condition, mode,
                                                     up_w, up_h, src_w, src_h):
        up_w, up_h = src_w, src_h
        if mode in BOX_MODES:
            box = (min(box[0], src_w), min(box[1], src_h))
            exact_box = box
        skipped = condition

    est_bytes, avail_bytes, ram_warn, ram_oom = (
        (0, _available_ram_bytes(), None, False) if skipped else
        _check_ram_budget(cached.get("b", 1), src_w, src_h, up_w, up_h,
                          cached.get("c", 3), method, k)
    )

    out = {
        "ok": True,
        "png": cached["png"],
        "src": [src_w, src_h],
        "up": [up_w, up_h],
        "factor": round(k, 3),
        "method": method,
        "multiple_of": mult,
        "skipped": skipped,
        "warning": ram_warn,
        "oom": ram_oom,
        "est_ram_mb": round(est_bytes / (1024 ** 2)),
        "free_ram_mb": round(avail_bytes / (1024 ** 2)) if avail_bytes else None,
        "rect": None,
        "crop": None,
    }

    if bool(data.get("crop")) or exact_box:
        # a box mode always cuts to its box, whatever crop_ratio says
        ratio = "custom" if exact_box else txt("crop_ratio", "4:5", RATIO_NAMES)
        orient = txt("crop_orientation", "auto", ORIENTATIONS)
        pos = txt("crop_position", "center", CROP_POSITIONS)
        if exact_box:
            box_w, box_h = exact_box
        elif ratio == "custom":
            box_w = num("crop_width", 1024)
            box_h = num("crop_height", 1024)
            if data.get("crop_width") is None:
                box_w = num("target_width", 1920)
            if data.get("crop_height") is None:
                box_h = num("target_height", 1080)
        else:
            box_w, box_h = 0, 0
        rect = _crop_rect(up_w, up_h, ratio, orient, box_w, box_h,
                          pos, num("crop_offset_x", 0), num("crop_offset_y", 0),
                          int(num("crop_seed", 0)), mult=mult)
        cw = int(_snap_inside((rect[2] - rect[0]) * up_w, up_w, mult))
        ch = int(_snap_inside((rect[3] - rect[1]) * up_h, up_h, mult))
        out["rect"] = [round(v, 5) for v in rect]
        out["crop"] = [cw, ch]
        out["ratio"] = ratio
        out["box"] = [box_w, box_h]
        out["orientation"] = orient
        out["position"] = pos

    return out


# ---------------------------------------------------------------------------
# Route (pattern: image_io.py register_routes – try/except, no server in unit tests)
# ---------------------------------------------------------------------------

def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.post("/mm/upscale/preview")
        async def _upscale_preview(request):
            # live size/crop numbers for the preview panel (see _preview)
            try:
                data = await request.json()
            except Exception:
                data = {}
            cached = _LAST_IMAGE.get(str(data.get("node_id", "")))
            if cached is None:
                # a normal state, not an error: 200 with ok:false, so the panel
                # can show the hint instead of a failure
                return web.json_response(
                    {"ok": False, "error": "Nothing cached yet – run the graph once, then adjust."})
            try:
                return web.json_response(_preview(data, cached))
            except Exception as exc:  # noqa: BLE001
                return json_error(f"{type(exc).__name__}: {exc}")
    except Exception as e:  # noqa: BLE001 - without a server (unit tests) no route
        logging.getLogger(__name__).warning("[MM Nodepack] routes not registered (%s): %s", type(e).__name__, e)
