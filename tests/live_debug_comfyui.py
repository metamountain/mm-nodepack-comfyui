"""Live ComfyUI + Playwright end-to-end debug test suite for MM Nodepack.

Runs against either:
  1. A self-contained headless CPU-only ComfyUI server (default, port 8199,
     CUDA_VISIBLE_DEVICES=-1, zero GPU VRAM used, always loads the current
     working-tree Python + JS without touching an active :8188 instance), OR
  2. An existing ComfyUI server via --url http://127.0.0.1:8188.

Usage:
  D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe tests/live_debug_comfyui.py
  D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe tests/live_debug_comfyui.py --url http://127.0.0.1:8188
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

from _proc_guard import kill_with_parent


COMFY_PYTHON = r"D:\ComfyUI_windows_portable\python_embeded\python.exe"
COMFY_MAIN = r"D:\ComfyUI_windows_portable\ComfyUI\main.py"


def http_json(url, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def wait_server(base_url, timeout_s=90):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            code, _ = http_json(f"{base_url}/system_stats", timeout=2)
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def queue_and_wait(base_url, prompt, timeout_s=45):
    code, res = http_json(f"{base_url}/prompt", {"prompt": prompt, "client_id": "mm-live-debug"})
    assert code == 200, f"POST /prompt failed ({code}): {res}"
    assert not res.get("node_errors"), f"Node validation errors: {res['node_errors']}"
    pid = res["prompt_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _, hist = http_json(f"{base_url}/history/{pid}", timeout=5)
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise AssertionError(f"Prompt {pid} failed: {json.dumps(status, indent=2)}")
            return entry
        time.sleep(0.25)
    raise TimeoutError(f"Prompt {pid} timed out after {timeout_s}s")


def make_gradient_png(path, w, h):
    """Spatial gradient PNG: R=X (0..255), G=Y (0..255), B=128."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        g = int(round(255 * y / max(1, h - 1)))
        for x in range(w):
            r = int(round(255 * x / max(1, w - 1)))
            px[x, y] = (r, g, 128)
    img.save(path, "PNG")
    return path


def fetch_view_image(base_url, entry):
    q = urllib.parse.urlencode({
        "filename": entry["filename"],
        "subfolder": entry.get("subfolder", ""),
        "type": entry.get("type", "temp"),
    })
    with urllib.request.urlopen(f"{base_url}/view?{q}", timeout=10) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def run_api_tests(base_url, tmpdir):
    print("[1/4] Checking /object_info schemas...")
    for nid in ("MMLoadImage", "MMPreviewSave", "MMUpscaleCrop", "MMLLMMultitool"):
        code, info = http_json(f"{base_url}/object_info/{nid}")
        assert code == 200 and nid in info, f"{nid} missing from /object_info: {code}"

    _, up_info = http_json(f"{base_url}/object_info/MMUpscaleCrop")
    up = up_info["MMUpscaleCrop"]
    req_keys = list(up["input"]["required"].keys())
    opt_keys = list(up["input"].get("optional", {}).keys())
    assert "samples" not in req_keys and "samples" not in opt_keys, "latent input still present!"
    assert "LATENT" not in up["output"], f"latent output still present: {up['output']}"
    assert up["output"] == ["IMAGE", "INT", "INT", "STRING"], f"unexpected outputs: {up['output']}"
    assert req_keys[:3] == ["image", "target_mode", "scale_factor"], f"bad input order: {req_keys[:6]}"
    assert up["input"]["required"]["method"][1]["default"] == "lanczos", "method default must be lanczos"
    assert up["input"]["required"]["target_mode"][1]["default"] == "scale_factor"
    print("  OK: all 4 MM schemas verified (MMUpscaleCrop image-only, lanczos default, exact widget order)")

    # Create a test folder with 3 distinct gradient images
    src_dir = os.path.join(tmpdir, "src_folder")
    os.makedirs(src_dir, exist_ok=True)
    p1 = make_gradient_png(os.path.join(src_dir, "img_1.png"), 400, 300)
    p2 = make_gradient_png(os.path.join(src_dir, "img_2.png"), 300, 400)
    p3 = make_gradient_png(os.path.join(src_dir, "img_3.png"), 320, 320)

    print("[2/4] Testing MMLoadImage + MMPreviewSave live prompt & HTTP routes...")
    # Test MMLoadImage (path mode) -> MMUpscaleCrop -> MMPreviewSave
    prompt = {
        "1": {
            "class_type": "MMLoadImage",
            "inputs": {
                "image": "",
                "path": p1,
                "mode": "single",
                "folder": "",
                "batch_mode": False,
                "index": 1,
            },
        },
        "2": {
            "class_type": "MMUpscaleCrop",
            "inputs": {
                "image": ["1", 0],
                "target_mode": "scale_factor",
                "scale_factor": 2.0,
                "scale_percent": 200.0,
                "shortest_side": 512,
                "longest_side": 512,
                "megapixels": 0.5,
                "target_width": 640,
                "target_height": 480,
                "resize_condition": "always",
                "method": "lanczos",
                "upscale_model": up["input"]["required"]["upscale_model"][0][0],
                "snap": False,
                "multiple_of": 8,
                "crop": False,
                "crop_ratio": "4:5",
                "crop_width": 256,
                "crop_height": 256,
                "crop_orientation": "auto",
                "crop_position": "center",
                "crop_offset_x": 0,
                "crop_offset_y": 0,
                "crop_seed": 0,
            },
        },
        "3": {
            "class_type": "MMPreviewSave",
            "inputs": {
                "images": ["2", 0],
                "auto_save": False,
                "auto_send": False,
                "source": "auto",
                "suffix": "_#",
                "save_as": "",
                "format": "same as source",
                "quality": 95,
            },
        },
    }
    h = queue_and_wait(base_url, prompt)
    out3 = h["outputs"]["3"]
    assert out3["b_images"], "MMPreviewSave produced no b_images"
    assert out3["a_images"], "MMPreviewSave wireless source detection missed original a_images"
    assert out3["source"][0] == os.path.abspath(p1), f"wrong source: {out3['source']}"
    b_img = fetch_view_image(base_url, out3["b_images"][0])
    assert b_img.size == (800, 600), f"expected 800x600 (2x 400x300), got {b_img.size}"

    # Live HTTP route checks: /mm/image/preview, /mm/image/target, /mm/image/save (with AND without loader!)
    code, prev = http_json(f"{base_url}/mm/image/preview", {"mode": "single", "path": p1, "image": "", "folder": "", "batch_mode": False, "index": 1})
    assert code == 200 and prev["ok"] and prev["view"]["type"] == "temp", f"external preview failed: {prev}"

    code, tgt = http_json(f"{base_url}/mm/image/target", {"mode": "single", "path": p1, "suffix": "_#", "save_as": "", "format": "same as source"})
    assert code == 200 and tgt["ok"] and tgt["target"].endswith("img_1_1.png"), f"target failed: {tgt}"

    # Save WITH source (saves next to p1 as img_1_1.png)
    entry = out3["b_images"][0]
    code, sv = http_json(f"{base_url}/mm/image/save", {
        "filename": entry["filename"], "subfolder": entry.get("subfolder", ""), "type": entry.get("type", "temp"),
        "source": p1, "suffix": "_#", "save_as": "", "format": "same as source", "quality": 95,
    })
    assert code == 200 and sv["ok"] and os.path.isfile(sv["path"]), f"save with source failed: {sv}"

    # Save WITHOUT source (txt2img case: explicit absolute save_as AND default output/preview_#.png)
    abs_out = os.path.join(tmpdir, "txt2img_saved.png")
    code, sv2 = http_json(f"{base_url}/mm/image/save", {
        "filename": entry["filename"], "subfolder": entry.get("subfolder", ""), "type": entry.get("type", "temp"),
        "source": "", "suffix": "_#", "save_as": abs_out, "format": "png", "quality": 95,
    })
    assert code == 200 and sv2["ok"] and os.path.isfile(sv2["path"]), f"loader-less save_as failed: {sv2}"

    # Folder mode test on MMLoadImage (separate clean folder so img_1_1.png saved above does not shift indices)
    batch_dir = os.path.join(tmpdir, "batch_folder")
    os.makedirs(batch_dir, exist_ok=True)
    b1 = make_gradient_png(os.path.join(batch_dir, "frame_1.png"), 200, 150)
    b2 = make_gradient_png(os.path.join(batch_dir, "frame_2.png"), 150, 200)
    folder_prompt = {
        "10": {
            "class_type": "MMLoadImage",
            "inputs": {"image": "", "path": "", "mode": "folder", "folder": batch_dir, "batch_mode": False, "index": 2},
        },
        "11": {
            "class_type": "MMPreviewSave",
            "inputs": {
                "images": ["10", 0], "auto_save": False, "auto_send": False,
                "source": "auto", "suffix": "_#", "save_as": "",
                "format": "same as source", "quality": 95,
            },
        },
    }
    hf = queue_and_wait(base_url, folder_prompt)
    assert hf["outputs"]["10"]["path"][0] == os.path.abspath(b2), f"folder index 2 wrong: {hf['outputs']['10']}"
    print("  OK: MMLoadImage (single/folder) + MMPreviewSave (wireless compare + save with/without loader) passed")

    print("[3/4] Testing MMUpscaleCrop live across all target modes, crop directions & /mm/upscale/preview parity...")
    base_up = dict(prompt["2"]["inputs"])
    cases = [
        ("scale_percent 50%", {"target_mode": "scale_percent", "scale_percent": 50.0}, (200, 150)),
        ("shortest_side 300", {"target_mode": "shortest_side", "shortest_side": 300}, (400, 300)),
        ("longest_side 600", {"target_mode": "longest_side", "longest_side": 600}, (600, 450)),
        ("megapixels 0.48", {"target_mode": "megapixels", "megapixels": 0.48}, (800, 600)),
        ("custom 300x500 top-left", {"target_mode": "custom", "target_width": 300, "target_height": 500, "crop_position": "top-left"}, (300, 500)),
        ("custom 300x500 bottom-right", {"target_mode": "custom", "target_width": 300, "target_height": 500, "crop_position": "bottom-right"}, (300, 500)),
        ("scale 1x + crop 4:5 portrait on square", {"target_mode": "scale_factor", "scale_factor": 1.0, "crop": True, "crop_ratio": "4:5", "crop_orientation": "auto"}, (256, 320)),
        ("snap=64 + custom crop bottom-right", {"target_mode": "scale_factor", "scale_factor": 1.0, "crop": True, "crop_ratio": "custom", "crop_width": 150, "crop_height": 150, "crop_position": "bottom-right", "snap": True, "multiple_of": 64}, (128, 128)),
    ]
    for label, overrides, want in cases:
        src_p = p3 if "square" in label else p1
        inp = dict(base_up, **overrides)
        p = {
            "1": {"class_type": "MMLoadImage", "inputs": {"image": "", "path": src_p, "mode": "single", "folder": "", "batch_mode": False, "index": 1}},
            "2": {"class_type": "MMUpscaleCrop", "inputs": inp},
            "3": {"class_type": "MMPreviewSave", "inputs": prompt["3"]["inputs"]},
        }
        res = queue_and_wait(base_url, p)
        im = fetch_view_image(base_url, res["outputs"]["3"]["b_images"][0])
        assert im.size == want, f"{label}: expected {want}, got {im.size}"
        # Verify live /mm/upscale/preview matches actual executed dimensions
        prev_body = {k: v for k, v in inp.items() if k != "image"}
        prev_body["node_id"] = "2"
        code, pr = http_json(f"{base_url}/mm/upscale/preview", prev_body)
        assert code == 200 and pr["ok"], f"{label}: preview failed: {pr}"
        got_prev = tuple(pr["crop"] if pr["crop"] else pr["up"])
        assert got_prev == want, f"{label}: preview {got_prev} != actual {want}"
        if "bottom-right" in label:
            # Bottom-right corner pixel must come from the right/bottom gradient edge
            r_px, g_px, _ = im.getpixel((im.size[0] - 1, im.size[1] - 1))
            assert r_px >= 240 and g_px >= 240, f"{label}: corner pixel not bottom-right: {(r_px, g_px)}"
    print(f"  OK: {len(cases)} live MMUpscaleCrop modes & crop-direction parity cases passed")


def run_playwright_tests(base_url):
    print("[4/4] Running live Chromium Playwright frontend UI tests...")
    from playwright.sync_api import sync_playwright

    mm_console_errors = []
    failed_404 = []
    in_mm_phase = [False]

    def on_console(m):
        if m.type != "error":
            return
        txt = m.text or ""
        low = txt.lower()
        if in_mm_phase[0] or any(k in low for k in ("mm-nodepack", "mm_nodepack", "mm_upscale", "mm_llm", "mm_free_gpu", "mm_logo", "mm_ui", "comparer.mjs")):
            mm_console_errors.append(txt)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", on_console)
        page.on("pageerror", lambda e: mm_console_errors.append(str(e)) if in_mm_phase[0] or "mm" in str(e).lower() else None)
        page.on("response", lambda r: failed_404.append((r.status, r.url)) if r.status >= 400 and ("mm-nodepack" in r.url.lower() or "/mm/" in r.url.lower() or "scripts/" in r.url) else None)

        page.goto(base_url, wait_until="networkidle", timeout=45000)
        page.wait_for_function("() => window.app && window.app.graph && window.LiteGraph && window.LiteGraph.registered_node_types['MMUpscaleCrop']", timeout=30000)
        in_mm_phase[0] = True

        # Verify all 4 MM nodes register and instantiate cleanly, check fold-outs, shrink, and localStorage memory
        report = page.evaluate("""async () => {
            const app = window.app;
            const LG = window.LiteGraph;
            localStorage.removeItem('mm_upscale_crop_last_v2');
            // Seed poisoned v1/v2 values (e.g. longest_side: 8) -> must be rejected and fall back to 2048
            localStorage.setItem('mm_upscale_crop_last_v1', JSON.stringify({ longest_side: 8 }));
            localStorage.setItem('mm_upscale_crop_last_v2', JSON.stringify({ longest_side: 8, shortest_side: 4 }));

            const created = {};
            for (const cls of ['MMLoadImage', 'MMPreviewSave', 'MMUpscaleCrop', 'MMLLMMultitool']) {
                const n = LG.createNode(cls);
                app.graph.add(n);
                created[cls] = { id: n.id, wCount: n.widgets?.length || 0, size: [...n.size] };
            }

            const up = app.graph._nodes.find(n => (n.comfyClass || n.type) === 'MMUpscaleCrop');
            const visNames = (n) => (n.widgets || []).filter(w => !w.hidden && w.type !== 'hidden').map(w => w.name);
            const setW = (n, name, val) => {
                const w = n.widgets.find(x => x.name === name);
                if (!w) throw new Error('missing widget ' + name);
                w.value = val;
                w.callback?.(val);
            };
            const tick = () => new Promise(r => setTimeout(r, 40));

            await tick();
            const stateDefault = { vis: visNames(up), h: up.size[1] };

            // Switch to custom -> target_width/target_height/resize_condition/crop_position/offsets visible, crop hidden
            setW(up, 'target_mode', 'custom');
            setW(up, 'target_width', 1344);
            setW(up, 'target_height', 768);
            await tick();
            const stateCustom = { vis: visNames(up), h: up.size[1] };

            // Switch crop_position to random -> offsets hidden, crop_seed + control_after_generate visible
            setW(up, 'crop_position', 'random');
            await tick();
            const stateRandom = { vis: visNames(up), h: up.size[1] };

            // Switch back to scale_factor, crop off -> node must shrink back!
            setW(up, 'target_mode', 'scale_factor');
            setW(up, 'crop', false);
            await tick();
            const stateBack = { vis: visNames(up), h: up.size[1] };

            // Turn crop on with custom ratio -> crop_width & crop_height right under crop_ratio
            setW(up, 'crop', true);
            setW(up, 'crop_ratio', 'custom');
            await tick();
            const stateCropCustom = { vis: visNames(up), h: up.size[1] };

            // Create a SECOND MMUpscaleCrop -> must remember last custom size (1344x768) & target_mode!
            setW(up, 'target_mode', 'custom');
            await tick();
            const up2 = LG.createNode('MMUpscaleCrop');
            app.graph.add(up2);
            await tick();
            const wv = (n, name) => n.widgets.find(x => x.name === name)?.value;
            const remembered = {
                target_mode: wv(up2, 'target_mode'),
                target_width: wv(up2, 'target_width'),
                target_height: wv(up2, 'target_height'),
                longest_side: wv(up2, 'longest_side'),
                shortest_side: wv(up2, 'shortest_side'),
            };

            // Check DOM widget serialize=false on MMUpscaleCrop
            const ser = up.serialize();
            const hasPreviewLeak = (ser.widgets_values || []).length !== up.widgets.filter(w => w.serialize !== false).length;

            // Check MMLoadImage legacy mode repair, button uniqueness & mode switching
            const ld = app.graph._nodes.find(n => (n.comfyClass || n.type) === 'MMLoadImage');
            const wVis = (w) => !!w && !w.hidden && w.type !== 'hidden';
            const btnLabel = (w) => String(w?.label || w?.name || '');
            // match buttons by their original type (_mmType survives hiding: a hidden
            // button reads type === 'hidden', which would break a type === 'button' filter)
            const isBtn = (w) => !!w && (w._mmType || w.type) === 'button';
            const modeW = ld.widgets?.find(w => w.name === 'mode');
            const upBtns = (ld.widgets || []).filter(w => isBtn(w) && btnLabel(w) === 'choose file');
            const folderBtns = (ld.widgets || []).filter(w => isBtn(w) && btnLabel(w) === 'choose folder');
            const upBtn = upBtns[0];
            const folderBtn = folderBtns[0];
            const imgW = ld.widgets?.find(w => w.name === 'image');
            const folderW = ld.widgets?.find(w => w.name === 'folder');
            const ldWidgetCounts = {
                total: (ld.widgets || []).length,
                unique: new Set(ld.widgets || []).size,
                chooseFile: upBtns.length,
                chooseFolder: folderBtns.length,
                chooseFileRightAfterImage: ld.widgets.indexOf(upBtn) === ld.widgets.indexOf(imgW) + 1,
                chooseFolderRightAfterFolder: ld.widgets.indexOf(folderBtn) === ld.widgets.indexOf(folderW) + 1,
            };
            const ldState = () => ({ mode: wVis(modeW), up: wVis(upBtn), folderBtn: wVis(folderBtn), h: ld.size[1] });

            const ldDefaultVis = visNames(ld);
            const ldDefaultState = ldState();
            ld._mmSetMode('folder');
            await tick();
            const ldFolderVis = visNames(ld);
            const ldFolderState = ldState();
            ld._mmSetMode('single');
            await tick();
            const ldSingleVis = visNames(ld);
            const ldSingleState = ldState();

            // Measure REAL 512x512 image preview minimum height across core LoadImage,
            // MMLoadImage, and MMPreviewSave: all three must match (220 px image preview min).
            const makeTestImg = () => new Promise(resolve => {
                const c = document.createElement('canvas');
                c.width = 512; c.height = 512;
                const im = new Image();
                im.onload = () => resolve(im);
                im.src = c.toDataURL();
            });
            const testImg = await makeTestImg();

            const coreLd = LG.createNode('LoadImage');
            app.graph.add(coreLd);
            await tick();
            coreLd.imgs = [testImg];
            coreLd.imageIndex = 0;
            coreLd.setSizeForImage?.();
            app.canvas.setDirty(true, true);
            await tick(120);
            const corePreviewMin = coreLd.widgets?.find(w => w.name === '$$canvas-image-preview')?.computeLayoutSize?.(coreLd)?.minHeight ?? null;

            ld.imgs = [testImg];
            ld.imageIndex = 0;
            ld._mmEnsureSize?.();
            app.canvas.setDirty(true, true);
            await tick(120);
            const ldPreviewMin = ld.widgets?.find(w => w.name === '$$canvas-image-preview')?.computeLayoutSize?.(ld)?.minHeight ?? null;
            const ldAfterMin = ld.computeSize()[1];
            const hWithImage = ld.size[1];
            ld._mmSetMode('folder');
            await tick();
            const hAfterSwitch = ld.size[1];
            ld._mmSetMode('single');
            await tick();
            const hBackSingle = ld.size[1];
            const minFinal = ld.computeSize?.() || [0, 0];

            // MMPreviewSave: new display name, no embed_workflow, initial size >= min,
            // and exact same 220 px image preview minHeight (+16 px header) as LoadImage / MMLoadImage.
            const pv = app.graph._nodes.find(n => (n.comfyClass || n.type) === 'MMPreviewSave');
            await tick();
            const pvInitSize = pv.size[1];
            const pvInitMin = pv.computeSize()[1];
            const cmp = pv.widgets?.find(w => w.name === 'comparer');
            cmp.value = { a: [], b: [{ filename: 'x.png', subfolder: '', type: 'temp' }] };
            cmp.imgB = testImg;
            pv.arrange?.();
            const pvImgMinHeight = cmp.computeLayoutSize().minHeight - 16; // minus 16 px label header
            const previewDef = {
                title: pv?.title,
                display_name: pv?.constructor?.nodeData?.display_name || pv?.constructor?.title || pv?.title,
                hasEmbedWorkflow: pv?.widgets?.some(w => w.name === 'embed_workflow') || false,
                initSize: pvInitSize,
                initMin: pvInitMin,
                imgMinHeight: pvImgMinHeight,
            };
            const previewMinParity = {
                coreLoadMinHeight: corePreviewMin,
                mmLoadMinHeight: ldPreviewMin,
                mmSaveMinHeight: pvImgMinHeight,
            };

            // MMPreviewSave source picker: hidden while auto is unambiguous, shown
            // when several loaders could be the source, when a #id is pinned, or
            // while the source socket is linked (then the string widget, not the combo).
            const srcState = (n) => {
                const vis = (w) => !!w && !w.hidden && w.type !== 'hidden';
                // the combo may be deduped to name "source#1": match by type + name prefix;
                // the schema string widget may be typed "text" in the V3 frontend, so
                // match by name and exclude the combo by type
                const combo = (n.widgets || []).find(w => (w._mmType || w.type) === 'combo' && /^source/.test(String(w.name || '')));
                const str = (n.widgets || []).find(w => w.name === 'source' && (w._mmType || w.type) !== 'combo');
                return {
                    combo: vis(combo), str: vis(str),
                    strType: str ? String(str._mmType || str.type) : null,
                    value: str ? String(str.value) : null,
                    comboValue: combo ? String(combo.value) : null,
                    label: combo ? String(combo.label) : null,
                    options: combo ? [...(combo.options?.values || [])] : [],
                    checkAt: n._mmSourceCheckAt ?? null,
                    drawWrapped: n.onDrawForeground ? String(n.onDrawForeground).includes('mmSourceCheckAt') : null,
                };
            };
            const srcWait = (ms) => new Promise(r => setTimeout(r, ms));
            // headless pages may run no idle canvas draw frames: force one so the
            // throttled onDrawForeground check (500 ms) is guaranteed to run
            const srcNudge = () => { app.canvas.draw?.(true, true); };
            const srcInitial = srcState(pv);
            // link the loader to the images input: auto resolves via that link,
            // but the SOURCE socket itself is not linked -> both stay hidden
            ld.connect(0, pv, 0);
            srcNudge();
            await srcWait(700);
            const srcLinked = srcState(pv);
            // disconnect and add a second loader: auto is ambiguous -> combo visible
            const srcInp = pv.inputs.find(i => i.name === 'images');
            if (srcInp && srcInp.link != null) {
                if (app.graph.disconnectLink) app.graph.disconnectLink(srcInp.link);
                else srcInp.link = null;
            }
            const l2 = LG.createNode('MMLoadImage');
            app.graph.add(l2);
            srcNudge();
            await srcWait(700);
            const srcMulti = srcState(pv);
            // pin the first loader: the pinned value keeps the combo visible
            const srcStrW = pv.widgets.find(w => w.name === 'source' && (w._mmType || w.type) !== 'combo');
            if (srcStrW) {
                srcStrW.value = '#' + ld.id;
                srcStrW.callback?.(srcStrW.value);
            }
            await srcWait(700);
            const srcPinned = srcState(pv);
            const srcIds = { ld: ld.id, l2: l2.id };

            return {
                created, stateDefault, stateCustom, stateRandom, stateBack,
                stateCropCustom, remembered, hasPreviewLeak,
                ldWidgetCounts, ldDefaultVis, ldFolderVis, ldSingleVis,
                ldDefaultState, ldFolderState, ldSingleState,
                hWithImage, ldAfterMin, hAfterSwitch, hBackSingle, minFinal, previewDef, previewMinParity,
                srcInitial, srcLinked, srcMulti, srcPinned, srcIds,
                pvInputs: (pv.inputs || []).map(i => i.name),
            };
        }""")
        browser.close()

    assert not failed_404, f"HTTP 4xx/5xx errors on MM/scripts assets: {failed_404}"
    assert not mm_console_errors, f"MM Nodepack browser console/page errors: {mm_console_errors}"

    # Verify MMUpscaleCrop exact fold-out order & rules
    dvis = report["stateDefault"]["vis"]
    assert dvis[:5] == ["target_mode", "scale_factor", "method", "snap", "crop"], f"bad default foldout: {dvis}"
    assert "upscale_model" not in dvis, "upscale_model should be hidden for lanczos default"
    assert "multiple_of" not in dvis, "multiple_of should be hidden when snap=false"
    assert "crop_position" not in dvis and "crop_offset_x" not in dvis, "crop controls must be hidden when crop=false"

    cvis = report["stateCustom"]["vis"]
    assert cvis[:6] == ["target_mode", "target_width", "target_height", "resize_condition", "method", "snap"], f"bad custom foldout: {cvis}"
    assert "crop" not in cvis and "crop_ratio" not in cvis, f"crop toggle must hide in custom mode: {cvis}"
    assert "crop_position" in cvis and "crop_offset_x" in cvis and "crop_offset_y" in cvis, f"framing missing in custom: {cvis}"

    rvis = report["stateRandom"]["vis"]
    assert "crop_offset_x" not in rvis and "crop_offset_y" not in rvis, f"offsets must hide in random mode: {rvis}"
    assert "crop_seed" in rvis, f"crop_seed missing in random mode: {rvis}"

    assert report["stateBack"]["h"] < report["stateCustom"]["h"], f"node did not shrink back: {report['stateBack']['h']} vs {report['stateCustom']['h']}"

    ccvis = report["stateCropCustom"]["vis"]
    ci = ccvis.index("crop_ratio")
    assert ccvis[ci:ci + 3] == ["crop_ratio", "crop_width", "crop_height"], f"crop_width/height not directly under crop_ratio: {ccvis}"

    assert report["remembered"] == {
        "target_mode": "custom", "target_width": 1344, "target_height": 768,
        "longest_side": 2048, "shortest_side": 1024,
    }, f"localStorage memory / poisoned-value guard failed: {report['remembered']}"
    assert not report["hasPreviewLeak"], "DOM widget leaked into widgets_values!"

    # Verify MMLoadImage has zero duplicate widgets, exact button slots, and clean mode fold-outs
    wc = report["ldWidgetCounts"]
    assert wc["unique"] == wc["total"], f"duplicate widget object in MMLoadImage.widgets: {wc}"
    assert wc["chooseFile"] == 1 and wc["chooseFolder"] == 1, f"expected exactly 1 choose file + 1 choose folder button: {wc}"
    assert wc["chooseFileRightAfterImage"], f"choose file must sit directly below image: {wc}"
    assert wc["chooseFolderRightAfterFolder"], f"choose folder must sit directly below folder: {wc}"

    assert "path" in report["ldDefaultVis"] and "folder" not in report["ldDefaultVis"], report["ldDefaultVis"]
    assert "batch_mode" not in report["ldDefaultVis"] and "index" not in report["ldDefaultVis"], report["ldDefaultVis"]
    for wname in ("mode", "folder", "batch_mode", "index", "control_after_generate"):
        assert wname in report["ldFolderVis"], f"batch widget '{wname}' missing in folder mode: {report['ldFolderVis']}"
    assert "path" not in report["ldFolderVis"] and "image" not in report["ldFolderVis"], report["ldFolderVis"]
    assert "path" in report["ldSingleVis"] and "folder" not in report["ldSingleVis"], report["ldSingleVis"]

    # MMLoadImage: the mode switch is always visible; "choose file" only in single mode,
    # "choose folder" only in folder mode
    for st_name, want_mode in (("ldDefaultState", "single"), ("ldFolderState", "folder"), ("ldSingleState", "single")):
        st = report[st_name]
        assert st["mode"] is True, f"mode switch must stay visible in {want_mode} mode: {st}"
        if want_mode == "single":
            assert st["up"] is True and st["folderBtn"] is False, f"single mode buttons wrong: {st}"
        else:
            assert st["up"] is False and st["folderBtn"] is True, f"folder mode buttons wrong: {st}"

    # Exact 1:1 parity with core LoadImage: core LoadImage, MMLoadImage, and MMPreviewSave
    # all enforce the exact same 220 px image preview minHeight!
    par = report["previewMinParity"]
    assert par["coreLoadMinHeight"] == 220, f"unexpected core LoadImage preview minHeight: {par}"
    assert par["mmLoadMinHeight"] == par["coreLoadMinHeight"], f"MMLoadImage preview minHeight != core LoadImage: {par}"
    assert par["mmSaveMinHeight"] == par["coreLoadMinHeight"], f"MMPreviewSave preview minHeight != core LoadImage: {par}"
    assert report["hWithImage"] >= report["ldAfterMin"], f"thumbnail crush: with-image height {report['hWithImage']} < min {report['ldAfterMin']}"
    assert report["hAfterSwitch"] >= report["hWithImage"], f"mode switch shrank the node: {report['hAfterSwitch']} < {report['hWithImage']}"
    assert report["hBackSingle"] >= report["hAfterSwitch"], f"mode switch shrank the node: {report['hBackSingle']} < {report['hAfterSwitch']}"
    assert report["hBackSingle"] >= report["minFinal"][1], f"height below widget minimum: {report['hBackSingle']} < {report['minFinal'][1]}"

    # MMPreviewSave: renamed, no embed_workflow, and initial size never clipped below min
    assert report["previewDef"]["title"] == "Preview & Save MM", report["previewDef"]
    assert report["previewDef"]["display_name"] == "Preview & Save MM", report["previewDef"]
    assert report["previewDef"]["hasEmbedWorkflow"] is False, "embed_workflow widget must be removed from MMPreviewSave"
    assert report["previewDef"]["initSize"] >= report["previewDef"]["initMin"], f"MMPreviewSave clipped on create: {report['previewDef']}"

    # MMPreviewSave source picker: hidden while auto is unambiguous (one loader,
    # linked or not), shown when several loaders could be the source, and a
    # pinned #id keeps it visible. The combo keeps the "source" label.
    ld_id = report["srcIds"]["ld"]
    l2_id = report["srcIds"]["l2"]
    src = report["srcInitial"]
    assert src["combo"] is False and src["str"] is False, f"source picker must be hidden with one loader and auto: {src}"
    print(f"  MMPreviewSave inputs: {report['pvInputs']}")
    src = report["srcLinked"]
    assert src["combo"] is False, f"combo must stay hidden while linked auto is unambiguous: {src}"
    # the images link feeds auto resolution; only a linked SOURCE socket would
    # show the string widget (setWidgetObjVisible keeps it visible then)
    assert src["str"] is False, f"string widget stays hidden while only the images socket is linked: {src}"
    src = report["srcMulti"]
    assert src["combo"] is True, f"combo must be visible with several candidate loaders: {src}"
    assert src["str"] is False, f"string widget must be hidden again once the socket is unlinked: {src}"
    assert any(o.startswith(f"#{ld_id} ") for o in src["options"]), f"loader #1 missing from options: {src['options']}"
    assert any(o.startswith(f"#{l2_id} ") for o in src["options"]), f"loader #2 missing from options: {src['options']}"
    assert src["label"] == "source", f"combo label must stay 'source' (got {src['label']!r})"
    src = report["srcPinned"]
    assert src["combo"] is True, f"pinned source must keep the combo visible: {src}"
    assert src["value"] == f"#{ld_id}", f"pinned value not stored in the schema string widget: {src}"
    expected = next((o for o in src["options"] if o.startswith(f"#{ld_id} ")), None)
    assert expected is not None, f"pinned loader label missing from options: {src['options']}"
    assert src["comboValue"] == expected, f"combo must show the pinned loader label: {src}"

    print("  OK: Playwright Chromium checks passed (0 console errors, 0 404s, all foldouts, shrink, mode buttons, no thumbnail crush, localStorage memory, no leaks, source picker rule)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="Existing ComfyUI URL (default: spawn headless CPU ComfyUI on port 8199)")
    ap.add_argument("--port", type=int, default=8199)
    args = ap.parse_args()

    proc = None
    log_f = None
    with tempfile.TemporaryDirectory(prefix="mm_live_", ignore_cleanup_errors=True) as tmpdir:
        try:
            if args.url:
                base_url = args.url.rstrip("/")
                print(f"Targeting existing ComfyUI at {base_url}")
                assert wait_server(base_url, timeout_s=10), f"Server not reachable at {base_url}"
            else:
                base_url = f"http://127.0.0.1:{args.port}"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
                log_path = os.path.join(tmpdir, "comfy_headless.log")
                print(f"Starting headless CPU ComfyUI on {base_url} (output={tmpdir})...")
                log_f = open(log_path, "w", encoding="utf-8")
                proc = subprocess.Popen(
                    [
                        COMFY_PYTHON, COMFY_MAIN,
                        "--cpu", "--listen", "127.0.0.1", "--port", str(args.port),
                        "--output-dir", os.path.join(tmpdir, "output"),
                        "--temp-directory", os.path.join(tmpdir, "temp"),
                        "--disable-auto-launch",
                    ],
                    stdout=log_f, stderr=subprocess.STDOUT, env=env,
                )
                kill_with_parent(proc)  # no orphaned server if this script is killed hard
                if not wait_server(base_url, timeout_s=90):
                    log_f.close()
                    log_f = None
                    tail = open(log_path, encoding="utf-8", errors="replace").read()[-2000:]
                    raise RuntimeError(f"Headless ComfyUI failed to start:\n{tail}")
                print("Headless ComfyUI ready.")

            run_api_tests(base_url, tmpdir)
            run_playwright_tests(base_url)
            print("\nALL LIVE COMFYUI + PLAYWRIGHT CHECKS PASSED!")
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if log_f is not None:
                log_f.close()


if __name__ == "__main__":
    main()
