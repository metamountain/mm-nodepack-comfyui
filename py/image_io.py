"""The MM Nodepack nodes (V3 node API, comfy_api.latest) and their HTTP routes.

MMLoadImage ("Load Image MM"): load a single image (upload or full path) or batch
from a folder. MMPreviewSave ("Preview & Save MM"): preview, compare, save.
Routes (/mm/image/*, all answering {"ok": ...} JSON):
  /mm/image/target  – next free save target for the loader's current source
  /mm/image/dialog  – OS open/save/folder dialog on the ComfyUI computer
  /mm/image/resolve – picker/upload name -> absolute path (+ origin of an upload)
  /mm/image/preview – live /view entry of the loader's current file
  /mm/image/save    – save a preview image next to its source (never overwrites)
"""
import asyncio
import logging
import os

from PIL import PngImagePlugin

from comfy_api.latest import io, ui

from .common import (
    CATEGORY,
    IMAGE_EXTS,
    _FMT_EXT,
    AccessDeniedError,
    _clean,
    _is_within,
    authorize,
    get_folder_paths,
    is_local_request,
    json_error,
    list_folder_images,
    load_image_file,
    policy_active,
    png_metadata,
    png_text_chunks,
    preview_view_entry,
    save_tensor,
    write_roots,
)
from .dialog import dialog_available, file_dialog
from .paths import (
    NoSourceError,
    find_origin_files,
    load_last_dir,
    next_target,
    pick_position,
    resolve_folder_image,
    resolve_save_target,
    resolve_source,
    resolve_source_like,
    resolve_targets,
)

LOADER_ID = "MMLoadImage"
ALIASES = ["mm", "metamountain", "mm nodepack"]
FORMATS = ["same as source", "png", "jpg", "webp"]


def _input_images() -> list:
    """Upload-combo options: the files in ComfyUI/input (like core LoadImage,
    nodes.py:1750-1752)."""
    try:
        fp = get_folder_paths()
        d = fp.get_input_directory()
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        return sorted(fp.filter_files_content_types(files, ["image"]))
    except Exception:  # noqa: BLE001 – no ComfyUI runtime (unit tests)
        return []


def _default_source() -> dict:
    """Fallback source (no connection): the output dir, stem 'preview', .png."""
    return {
        "path": os.path.join(get_folder_paths().get_output_directory(), "preview.png"),
        "name": "preview",
        "format": ".png",
    }


def _resolve_source_path(source) -> str:
    """The source path from a source dict, else the preview fallback."""
    if source is not None and source.get("path"):
        return str(source["path"])
    return _default_source()["path"]


def make_source(path: str) -> dict:
    """Build the source dict from a file: path, original image, mask, width,
    height, format and name (the preview reads it wirelessly from the prompt)."""
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


_LAST_RESOLVED = {}  # str(loader unique_id) -> last resolved absolute image path


def _output_slot_linked(prompt, node_id, slot_index: int) -> bool:
    """True when any node in `prompt` connects to output `slot_index` of `node_id`.
    Subgraph-safe: prompt keys look like '3:42' while a link may store the plain
    node number (or vice versa) – both directions are matched."""
    if not isinstance(prompt, dict) or node_id is None:
        return False
    want = str(node_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        for v in (node.get("inputs") or {}).values():
            if not (isinstance(v, list) and len(v) == 2):
                continue
            src = str(v[0])
            if (src == want or src.endswith(":" + want) or want.endswith(":" + src)) and v[1] == slot_index:
                return True
    return False


def find_loader(prompt, my_id, select="auto"):
    """Wireless source: -> (loader prompt id or None, note).
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
    """Source dict of a loader read from the prompt (single or folder mode)."""
    inp = (prompt.get(str(loader_id)) or {}).get("inputs") or {}
    mode = inp.get("mode", "single")
    if mode == "folder":
        folder = inp.get("folder", "")
        batch_mode = inp.get("batch_mode", False)
        index = inp.get("index", 1)
        if isinstance(folder, str) and isinstance(index, int) and not isinstance(index, bool):
            p, _, _ = resolve_folder_image(folder, bool(batch_mode), int(index))
            return make_source(p)
        # subgraph-safe: the loader may be known by its full prompt key ('3:7') or
        # its plain node number ('7') – the execute side wrote either form
        last = _LAST_RESOLVED.get(str(loader_id)) or _LAST_RESOLVED.get(str(loader_id).split(":")[-1])
        if last and os.path.isfile(last):
            return make_source(last)
        return None
    image, path = inp.get("image", ""), inp.get("path", "")
    if not isinstance(image, str) or not isinstance(path, str):
        return None  # widget driven by a link: unknown before it runs
    return make_source(resolve_source(image, path))


def _save_batch(images, targets, fmt, quality, prompt, extra_pnginfo) -> list:
    """Save every image of the batch to its target. Returns the written paths
    (the actually written file when a race moved the save to the next number).
    PNG saves ALWAYS embed the prompt + workflow metadata (like core SaveImage).
    On a shared server each file is created exclusively under the output root."""
    roots = write_roots() if policy_active() else None
    paths = []
    for i, img in enumerate(images):
        target = targets[i]
        pnginfo = None
        if os.path.splitext(target)[1].lower() == ".png" and (prompt is not None or extra_pnginfo is not None):
            pnginfo = png_metadata(prompt, extra_pnginfo)
        paths.append(save_tensor(img, target, quality=quality, pnginfo=pnginfo, roots=roots))
    return paths


def _preview_entries(tensors, cls) -> list:
    """Temp-folder preview entries [{filename, subfolder, type}, ...] (rgthree
    image_comparer.py:30-45 pattern: PreviewImage(...).as_dict()['images'])."""
    return ui.PreviewImage(tensors, cls=cls).as_dict()["images"]


# ---------------------------------------------------------------------------
# Node 1: Load Image MM
# ---------------------------------------------------------------------------

class MMLoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        ctrl_inc = getattr(getattr(io, "ControlAfterGenerate", None), "increment", True)
        return io.Schema(
            node_id="MMLoadImage",
            display_name="Load Image MM",
            category=CATEGORY,
            description="Load one image, or a folder batch, incrementally or at random; give a file, an "
                        "upload or a folder path. Preview & Save MM finds this node automatically as the source.",
            search_aliases=ALIASES + ["mm load", "load image mm", "load image path", "load image folder", "batch"],
            inputs=[
                io.Combo.Input("mode", options=["single", "folder"], default="single",
                               tooltip="single = one image, folder = batch"),
                io.Combo.Input("image", options=_input_images() or [""], upload=io.UploadType.image,
                               tooltip="Pick an image you uploaded to ComfyUI (used when 'path' is empty)"),
                io.String.Input("path", default="",
                                tooltip="Full file path on the server. Wins over the upload; saves go next to this file"),
                io.String.Input("folder", default="",
                                tooltip="Full path of a folder containing the images to load"),
                io.Boolean.Input("batch_mode", default=False, label_on="random", label_off="incremental",
                                 tooltip="incremental = step through in file order, random = index picks the image as a seed"),
                io.Int.Input("index", default=1, min=1, max=0xFFFFFFFFFFFFFFFF, control_after_generate=ctrl_inc,
                             tooltip="Which image of the folder: 1 = first, wraps after the last. Random mode uses it as the seed"),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("path", display_name="path", tooltip="Full path of the loaded file"),
                io.Image.Output("image_list", display_name="image list", is_output_list=True,
                                tooltip="Every image of the folder in file order; one image in single mode"),
                io.Int.Output("index", display_name="index",
                              tooltip="Position of the loaded image in the folder (always 1 in single mode)"),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def fingerprint_inputs(cls, image="", path="", mode="single", folder="", batch_mode=False, index=1) -> str:
        """Reload when the selected file changes (path + mtime)."""
        if mode == "folder":
            try:
                p, pos, count = resolve_folder_image(folder, bool(batch_mode), int(index))
                return f"{p}|{os.path.getmtime(p)}|{pos}/{count}"
            except Exception:  # noqa: BLE001
                return f"folder:{folder}|{batch_mode}|{index}"
        p = _clean(path)
        if not p:
            try:
                p = get_folder_paths().get_annotated_filepath(image)
            except Exception:  # noqa: BLE001
                p = image
        try:
            return f"{p}|{os.path.getmtime(p)}"
        except OSError:
            return str(p)

    @classmethod
    def validate_inputs(cls, image="", path="", mode="single", folder="", batch_mode=False, index=1):
        if mode == "folder":
            # folder mode: only the folder is validated (image/path are ignored)
            f = _clean(folder)
            if not f:
                return "folder is empty – click 'choose folder' or enter a folder path"
            abs_f = os.path.abspath(f)
            try:  # shared server: access check before any existence check or scan
                authorize(abs_f, "read")
                if not os.path.isdir(abs_f):
                    return f"folder not found: {abs_f}"
                if not list_folder_images(abs_f):
                    return f"no supported images found in folder: {abs_f}"
            except AccessDeniedError:  # shared server: folder outside the approved roots
                return "folder outside the allowed folders (shared server)"
            return True
        # single mode: 'path' wins over the upload combo
        p = _clean(str(path))
        if p:
            abs_p = os.path.abspath(p)
            try:  # shared server: access check before the existence check (no-op otherwise)
                authorize(abs_p, "read")
            except AccessDeniedError:
                return "path outside the allowed folders (shared server)"
            if not os.path.isfile(abs_p):
                return (f"path points to a missing file (moved or deleted?): {abs_p} – "
                        "pick or upload an image (that sets a new path) or clear the path field")
            return True
        if isinstance(image, (list, tuple)):  # wired to an output: resolved at run time
            return True
        img = _clean(str(image))
        if not img:
            return "no image selected: use 'choose file' or enter a file path"
        try:
            fp = get_folder_paths().get_annotated_filepath(img)
        except Exception:  # noqa: BLE001 – no ComfyUI runtime (unit tests)
            return True
        if not os.path.isfile(os.path.abspath(fp)):
            return f"image not found: {os.path.abspath(fp)} – use 'choose file' or enter a file path"
        return True

    @classmethod
    def execute(cls, image="", path="", mode="single", folder="", batch_mode=False, index=1):
        hidden = getattr(cls, "hidden", None)
        uid = getattr(hidden, "unique_id", None)
        prompt = getattr(hidden, "prompt", None)
        if mode == "folder":
            # exactly ONE listing per run: the selected file, the reported 1-based
            # index and the linked image list all come from the same sorted list,
            # so a file appearing between scans can never desynchronize them
            files = list_folder_images(folder)
            if not files:
                raise ValueError(f"no images found in folder: {os.path.abspath(folder)}")
            pos = pick_position(files, bool(batch_mode), int(index))
            p = files[pos]
            out_index = pos + 1
        else:
            p = resolve_source(image, path)
            out_index = 1
        if uid is not None:
            _LAST_RESOLVED[str(uid)] = p
        try:
            img, mask = load_image_file(p)
        except OSError as e:
            # the file disappeared or was replaced after the listing: fail the
            # node (no inconsistent outputs) with a bounded message
            raise ValueError(f"could not read the selected image ({type(e).__name__})") from e
        h, w = int(img.shape[1]), int(img.shape[2])
        if mode == "folder" and _output_slot_linked(prompt, uid, 5):
            # always the natural sort order (even in random mode): image_list[index - 1]
            # must stay the loaded image in every mode; the selected tensor is reused
            # (kept lazy when unlinked: only the picked file is decoded)
            try:
                image_list = [img if f == p else load_image_file(f)[0] for f in files]
            except OSError as e:
                raise ValueError(f"a folder image disappeared during load ({type(e).__name__})") from e
        else:
            image_list = [img]
        info = {"path": [p]}   # size: the frontend prints it under the thumbnail
        if mode == "folder" or _clean(path):  # folder/path mode: preview the resolved file
            info["images"] = _preview_entries(img[:1], cls)
        return io.NodeOutput(img, mask, w, h, p, image_list, out_index, ui=info)


# ---------------------------------------------------------------------------
# Node 2: Preview & Save MM (preview + compare + save; auto save = save every run)
# ---------------------------------------------------------------------------

class MMPreviewSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMPreviewSave",
            display_name="Preview & Save MM",
            category=CATEGORY,
            is_output_node=True,
            description="See the results, compare them with the source image and save the one you keep. "
                        "Saves go next to the source image (or to the output folder for uploads), "
                        "never overwriting an existing file.",
            search_aliases=ALIASES + ["mm preview", "mm save", "preview save mm", "preview image mm", "save image mm", "compare"],
            inputs=[
                io.Image.Input("images"),
                io.Boolean.Input("auto_save", default=False, label_on="auto save", label_off="no autosave",
                                 tooltip="on = every run saves all results; off = preview only, save with the buttons"),
                io.Boolean.Input("auto_send", default=False, label_on="auto send", label_off="no autosend",
                                 tooltip="After every run, saves the result and sends it to Load Image MM. "
                                         "Next run edits it. "
                                         "Skipped (toast) with several loaders or folder mode."),
                io.String.Input("source", default="auto",
                                tooltip="Shown only when several Load Image MM nodes could be the source. "
                                        "auto = upstream or the only loader; #id = pin one, e.g. #12"),
                io.String.Input("suffix", default="_#",
                                tooltip="Added to the source name. # = next free number (name_1, name_2 …)"),
                io.String.Input("save_as", default="",
                                tooltip="Auto save only: empty = next to the source with the suffix, or a name or full path"),
                io.Combo.Input("format", options=FORMATS, default="same as source"),
                io.Int.Input("quality", default=95, min=1, max=100, tooltip="Quality for JPEG and WebP saves"),
            ],
            outputs=[
                io.String.Output("saved_paths", display_name="saved", tooltip="The saved file paths, one per line (auto save)"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images, auto_save=False, source="auto", suffix="_#", save_as="",
                format="same as source", quality=95, auto_send=False):
        # auto_send is done by the browser after the run (same path as the "send to loader" button)
        if images is None or len(images) == 0:
            raise ValueError("no images to preview")
        # wireless: always resolve the source in the prompt
        note, src_id = "", ""
        sync = None
        lid, note = find_loader(cls.hidden.prompt, cls.hidden.unique_id, source)
        if lid is not None:
            try:
                sync = source_from_prompt(cls.hidden.prompt, lid)
                src_id = str(lid).split(":")[-1]
            except ValueError as e:
                note = str(e)
        paths = []
        if auto_save:
            targets = resolve_targets(images, sync, suffix, save_as, format)
            paths = _save_batch(images, targets, format, quality,
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


NODE_LIST = [MMLoadImage, MMPreviewSave]


# ---------------------------------------------------------------------------
# Routes (pattern: film_nodes.py:90-107 – try/except, no server in unit tests)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Route handlers (pure functions: testable without a server).
# Each returns (status, payload) with payload = {"ok": ...}. The aiohttp
# handlers below are thin wrappers: parse the JSON body, decide the route
# policy (restricted = shared server OR non-local request) and send the reply.
# ---------------------------------------------------------------------------

def _restricted(request) -> bool:
    """Route-level policy: enforced for every request when the server is
    shared, and additionally for every non-local request on a private server
    (trusted loopback keeps the unrestricted behaviour: local dialog files,
    paths outside the ComfyUI tree, saves next to the source)."""
    return policy_active() or not is_local_request(request.remote)


def _err_text(e: Exception, restricted: bool, default: str = "invalid request") -> str:
    """Client-facing error text. Restricted contexts get bounded messages:
    NoSourceError/AccessDeniedError are bounded by construction, anything else
    must not leak a path outside the approved roots."""
    if isinstance(e, (NoSourceError, AccessDeniedError)):
        return str(e)
    return str(e) if not restricted else default


def handle_target(data: dict, restricted: bool = False):
    """/mm/image/target: next free save target for the loader's current source.
    The policy gates run with `forced=restricted`: the source is read-authorized
    and the target write-authorized before any file probe, origin scan or
    number pick."""
    try:
        p = resolve_source_like(data, forced=restricted)
        if not os.path.isfile(p):
            return 404, {"ok": False, "error": "file not found"}
        target = next_target(p, str(data.get("suffix", "_#")), str(data.get("save_as", "")),
                             str(data.get("format", "same as source")), forced=restricted)
        return 200, {"ok": True, "target": target}
    except Exception as e:  # noqa: BLE001
        if isinstance(e, (NoSourceError, AccessDeniedError)):
            return (403 if isinstance(e, AccessDeniedError) else 400), {"ok": False, "error": str(e)}
        return 400, {"ok": False, "error": _err_text(e, restricted)}


def handle_resolve(data: dict, restricted: bool = False):
    """/mm/image/resolve: picker/upload name -> absolute path. An upload in
    ComfyUI/input is flagged (payload 'upload'): the origin scan over the output
    tree is done by the caller (the async wrapper runs it in a thread, tests
    call find_origin_files directly)."""
    try:
        fp = get_folder_paths()
        p = os.path.abspath(fp.get_annotated_filepath(str(data.get("image", ""))))
        authorize(p, "read", forced=restricted)
        if not os.path.isfile(p):
            return 404, {"ok": False, "error": "file not found"}
        inp = os.path.abspath(fp.get_input_directory())
        if os.path.commonpath([inp, p]) == inp:
            return 200, {"ok": True, "file": p, "upload": True}
        return 200, {"ok": True, "path": p, "file": p}
    except Exception as e:  # noqa: BLE001
        if isinstance(e, AccessDeniedError):
            return 403, {"ok": False, "error": str(e)}
        return 400, {"ok": False, "error": _err_text(e, restricted)}


def handle_preview(data: dict, restricted: bool = False):
    """/mm/image/preview: live /view entry of the loader's current file.
    The source is read-authorized (forced=restricted) before the existence
    check and the thumbnail/decode work."""
    try:
        p = resolve_source_like(data, forced=restricted)
        if not os.path.isfile(p):
            return 404, {"ok": False, "error": "file not found"}
        entry = preview_view_entry(p)
        if entry is None:
            return 404, {"ok": False, "error": "file not previewable (outside the allowed folders)" if restricted
                                else "file not previewable (outside output/input or unreadable)"}
        return 200, {"ok": True, "view": entry, "path": p}
    except Exception as e:  # noqa: BLE001
        if isinstance(e, AccessDeniedError):
            return 403, {"ok": False, "error": str(e)}
        return 400, {"ok": False, "error": _err_text(e, restricted)}


def handle_save(data: dict, restricted: bool = False):
    """/mm/image/save: save a preview image next to its source (never overwrites)."""
    try:
        t = str(data.get("type", ""))
        if t not in ("temp", "output"):
            return 400, {"ok": False, "error": "type must be 'temp' or 'output'"}
        fp = get_folder_paths()
        base_dir = fp.get_temp_directory() if t == "temp" else fp.get_output_directory()
        sub = str(data.get("subfolder", ""))
        fname = str(data.get("filename", ""))
        if not fname or os.path.isabs(fname):
            return 400, {"ok": False, "error": "bad filename"}
        src_file = os.path.abspath(os.path.join(base_dir, sub, fname))
        # the preview file must live inside its own folder (real paths: a link
        # that leaves the folder is rejected, not followed)
        if not _is_within(src_file, base_dir):
            return 403, {"ok": False, "error": "file outside the allowed folder"}
        if not os.path.isfile(src_file):
            return 404, {"ok": False, "error": "preview file not found"}
        fmt = str(data.get("format", "same as source"))
        if fmt != "same as source" and fmt not in _FMT_EXT:
            return 400, {"ok": False, "error": "bad format"}
        src = _clean(str(data.get("source", "")))
        if not src:
            try:
                # forced here as well: a folder-mode payload must not scan an
                # outside folder before the source is policy-checked
                src = resolve_source_like(data, forced=restricted)
            except NoSourceError:
                src = ""
        if src:
            # a source the loader does not know is a 404, not a silent fallback;
            # the payload's source field is authorized before any probe
            src = os.path.abspath(src)
            authorize(src, "read", forced=restricted)
            if not os.path.isfile(src):
                return 404, {"ok": False, "error": "source not found"}
            target = next_target(src, str(data.get("suffix", "_#")), str(data.get("save_as", "")), fmt,
                                 forced=restricted)
        else:
            # no loader in the graph (e.g. txt2img preview): save_as as-is
            # (absolute) or the output dir (preview_#.png) – never overwrites
            target = resolve_save_target("", str(data.get("suffix", "_#")), str(data.get("save_as", "")), fmt,
                                         forced=restricted)
        if os.path.splitext(target)[1].lower() not in IMAGE_EXTS:
            return 400, {"ok": False, "error": "target extension not allowed"}
        # same load/save core as the nodes (no divergent PIL path); the preview's PNG
        # text chunks (prompt/workflow) are carried over for .png targets
        img, _mask = load_image_file(src_file)
        pnginfo = None
        if os.path.splitext(target)[1].lower() == ".png":
            pnginfo = PngImagePlugin.PngInfo()
            for k, v in png_text_chunks(src_file).items():
                pnginfo.add_text(k, v)
        # restricted writes are created exclusively under the approved output root
        written = save_tensor(img[0], target, quality=int(data.get("quality", 95)),
                              pnginfo=pnginfo, roots=write_roots() if restricted else None)
        return 200, {"ok": True, "path": written}
    except Exception as e:  # noqa: BLE001
        if isinstance(e, AccessDeniedError):
            return 403, {"ok": False, "error": str(e)}
        return 400, {"ok": False, "error": _err_text(e, restricted)}


def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.post("/mm/image/target")
        async def _target(request):
            data = await request.json()
            status, payload = handle_target(data, _restricted(request))
            return web.json_response(payload, status=status)

        @PromptServer.instance.routes.post("/mm/image/dialog")
        async def _dialog(request):
            # the OS open/save/folder dialog on the ComfyUI computer -> real paths, no browser sandbox
            if not is_local_request(request.remote) or not dialog_available():
                return web.json_response({"ok": False, "local": False, "error": "no file dialog on the ComfyUI computer"}, status=403)
            try:
                data = await request.json()
                fp = get_folder_paths()
                raw_mode = data.get("mode")
                mode = raw_mode if raw_mode in ("open", "save", "folder") else "open"
                cur = _clean(str(data.get("path", "")))
                start = (cur if os.path.isdir(cur) else os.path.dirname(cur)) if cur else ""
                if not start or not os.path.isdir(start):
                    start = load_last_dir(mode)
                if not start or not os.path.isdir(start):
                    start = fp.get_output_directory()
                p = await asyncio.to_thread(file_dialog, mode, os.path.abspath(start), os.path.basename(str(data.get("name", ""))))
                if mode == "folder":
                    count, first_view = 0, None
                    if p and os.path.isdir(p):
                        try:
                            files = list_folder_images(p)
                            count = len(files)
                            # preview_view_entry: folders outside input/ still get a /view thumbnail
                            first_view = preview_view_entry(files[0]) if files else None
                        except Exception:  # noqa: BLE001
                            pass
                    return web.json_response({"ok": True, "path": p, "count": count, "view": first_view})
                return web.json_response({"ok": True, "path": p,
                                          "view": preview_view_entry(p) if p and mode == "open" else None})
            except Exception as e:  # noqa: BLE001
                return json_error(str(e))

        @PromptServer.instance.routes.post("/mm/image/resolve")
        async def _resolve(request):
            # picker/upload name ("x.png", "sub/x.png [input]") -> absolute path for the loader's path field
            data = await request.json()
            restricted = _restricted(request)
            status, payload = handle_resolve(data, restricted)
            if status != 200 or not payload.get("upload"):
                return web.json_response(payload, status=status)
            # an upload is a copy: find the original in the output tree.
            # the scan walks the whole tree: keep the event loop free of disk I/O
            # (both the upload and the scanned tree are authorized with forced=restricted)
            o = await asyncio.to_thread(find_origin_files, payload["file"],
                                        get_folder_paths().get_output_directory(), restricted)
            return web.json_response({"ok": True, "path": o[0] if len(o) == 1 else "",
                                      "file": payload["file"], "candidates": o if len(o) > 1 else []})

        @PromptServer.instance.routes.post("/mm/image/preview")
        async def _preview(request):
            # live preview of the loader's current file (compare-hold without re-run).
            data = await request.json()
            status, payload = handle_preview(data, _restricted(request))
            return web.json_response(payload, status=status)

        @PromptServer.instance.routes.post("/mm/image/save")
        async def _save(request):
            data = await request.json()
            status, payload = handle_save(data, _restricted(request))
            return web.json_response(payload, status=status)
    except Exception as e:  # noqa: BLE001 - without a server (unit tests) no route
        logging.getLogger(__name__).warning("[MM Nodepack] routes not registered (%s): %s", type(e).__name__, e)
