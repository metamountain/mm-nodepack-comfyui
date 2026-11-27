"""Live tooltip-position probe for MM Nodepack (standalone, not part of pytest).

Places every MM node near the canvas center / right edge / bottom edge of a
1440x900 headless Chromium viewport, hovers each widget and each custom
button, and measures the ComfyUI tooltip (div.node-tooltip) with numbers:

  - tooltip bounding rect fully inside the viewport (x/y/w/h reported)
  - overlap with the hovered element (px and % of the element area)
  - distance between hovered element and tooltip (must be <= 40 px)
  - tooltip disappears again after the mouse leaves
  - multi-line tooltips (LLM status / unload) get the same checks, which
    exposes viewport clipping at the right / bottom edges

The free-GPU action-bar button uses a native (OS-rendered) title tooltip
that headless Chromium cannot measure; the probe asserts its existence and
prints its title / aria-label values instead.

Usage:
  D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe tests/live_tooltip_probe.py
  D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe tests/live_tooltip_probe.py --url http://127.0.0.1:8188
  D:\\Coding\\MCP\\rags\\.venv\\Scripts\\python.exe tests/live_tooltip_probe.py --nodes MMLLMMultitool --positions center bottom

Notes:
  - Default: spawns a headless CPU ComfyUI on port 8200 (the debug suite uses
    8199) with the same flags and CUDA_VISIBLE_DEVICES=-1.
  - The probe queues one small test workflow (gradient -> upscale -> preview)
    so the runtime save buttons of Preview & Save MM exist before hovering.
    With --url it runs that workflow on the given server.
  - Hover points that lie inside the canvas element but are covered by a DOM
    overlay (top bar / tab strip, or third-party UI such as a custom-node
    monitor panel) are reported as off-canvas with the blocking element: the
    mouse event never targets the canvas there, so no tooltip can appear.
  - Frontend 1.58 draws customtext widgets as DOM <textarea> overlays
    (class comfy-multiline-input) that intercept pointer events over most of
    the widget's hit-test region, and the rendered textarea position does not
    match the widget's logical row. For those widgets the probe scans the
    hit-test region (row body, then the region down to the next widget) and
    hovers the first point where the canvas element is topmost; points only
    reachable through a textarea cannot produce a tooltip at all and are
    reported as blocked.
"""
import argparse
import io
import json
import os
import re
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
VIEWPORT = {"width": 1440, "height": 900}
POSITIONS = ["center", "right", "bottom"]
NODES = ["MMLoadImage", "MMPreviewSave", "MMUpscaleCrop", "MMLLMMultitool"]
FREE_GPU_TIP = "Unload all models and free the cache (GPU and RAM). Takes effect after the running job."


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


def queue_and_wait(base_url, prompt, timeout_s=60):
    code, res = http_json(f"{base_url}/prompt", {"prompt": prompt, "client_id": "mm-tooltip-probe"})
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
                raise AssertionError(f"Prompt {pid} failed: {json.dumps(status)}")
            return entry
        time.sleep(0.25)
    raise TimeoutError(f"Prompt {pid} timed out after {timeout_s}s")


def make_gradient_png(path, w, h):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        g = int(round(255 * y / max(1, h - 1)))
        for x in range(w):
            r = int(round(255 * x / max(1, w - 1)))
            px[x, y] = (r, g, 128)
    img.save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# page-side helpers (run inside Chromium via page.evaluate)
# ---------------------------------------------------------------------------

JS_READY = """() => {
    const app = window.app;
    return !!(app && app.graph && window.LiteGraph
        && NODES.map(c => window.LiteGraph.registered_node_types[c]).every(Boolean));
}"""

JS_READ = """(cls) => {
    const app = window.app;
    const LG = window.LiteGraph;
    const n = app.graph._nodes.find(x => (x.comfyClass || x.type) === cls && x._mmProbe);
    if (!n) return null;
    const rect = app.canvas.canvas.getBoundingClientRect();
    const w = n.size ? n.size[0] : (n.computeSize() || [300, 100])[0];
    const h = n.size ? n.size[1] : (n.computeSize() || [300, 100])[1];
    const title_mode = (n.title_mode != null ? n.title_mode
        : ((LG.Node && LG.Node.title_mode) || LG.LGraphNode?.title_mode || ""));
    const widgets = (n.widgets || []).map((wd, i) => {
        // the same geometry the frontend's getWidgetOnPos() hit-test uses:
        // widget.last_y (layout position in the node) + margin insets
        const visible = n.isWidgetVisible ? n.isWidgetVisible(wd) : !wd.hidden;
        const m = wd.margin != null ? wd.margin : 6;
        const h = wd.computedHeight || wd.h || 24;
        return {
            i: i, name: wd.name, type: wd.type, label: String(wd.label ?? wd.name ?? ""),
            x: n.pos[0] + m, y: n.pos[1] + (wd.last_y != null ? wd.last_y : 0),
            w: (wd.width || (n.size ? n.size[0] : 300)) - 2 * m, h: h,
            tooltip: wd.tooltip != null ? String(wd.tooltip) : "",
            schema_tip: (n.constructor?.nodeData?.inputs?.[wd.name]?.tooltip != null
                ? String(n.constructor.nodeData.inputs[wd.name].tooltip) : null),
            hidden: !visible,
        };
    });
    const cpos = [0, 0];  // canvas offset is pinned to [0, 0] by JS_POSITION
    return { id: n.id, w: w, h: h, pos: [n.pos[0], n.pos[1]], cpos: cpos,
             rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
             title_mode: title_mode,
             description: (n.constructor?.nodeData?.description || n.description || ""),
             widgets: widgets };
}"""

JS_CREATE = """(cls) => {
    const app = window.app;
    const LG = window.LiteGraph;
    for (const n of [...app.graph._nodes]) {
        if (n._mmProbe) app.graph.remove(n);
    }
    const n = LG.createNode(cls);
    n._mmProbe = true;
    n.pos = [0, 0];
    app.graph.add(n);
    if (app.canvas.ds) { app.canvas.ds.offset = [0, 0]; app.canvas.ds.scale = 1; }
    app.canvas.setDirty(true, true);
    return { ok: true };
}"""

JS_POSITION = """(args) => {
    // The canvas pan/offset is left at [0, 0]: the node is placed in graph
    // coordinates directly. (Driving the pan from the outside does not reach
    // the render transform in headless, which leaves stale hit-test areas.)
    // Positions are computed relative to the canvas element rect, not the
    // window: in this frontend the canvas does not fill the viewport (a top
    // bar sits above it, the queue/footer strip below it) and a hover point
    // outside the canvas element never targets the canvas, so no tooltip can
    // appear there. Nodes taller than the canvas overflow it; the affected
    // widgets are reported as off-canvas (not hoverable) with numbers.
    const app = window.app;
    const n = app.graph._nodes.find(x => (x.comfyClass || x.type) === args.cls && x._mmProbe);
    if (!n) return { ok: false };
    if (app.canvas.ds) app.canvas.ds.offset = [0, 0];
    const rect = app.canvas.canvas.getBoundingClientRect();
    const w = n.size ? n.size[0] : 300, h = n.size ? n.size[1] : 100;
    const gx = x => x - rect.x, gy = y => y - rect.y;
    if (args.position === "center") n.pos = [gx((rect.width - w) / 2), gy((rect.height - h) / 2)];
    else if (args.position === "right") n.pos = [gx(rect.width - w - 10), gy((rect.height - h) / 2)];
    else n.pos = [gx((rect.width - w) / 2), gy(rect.height - h - 10)];
    app.graph.change();
    app.canvas.setDirty(true, true);
    return { ok: true };
}"""

JS_TOP_EL = """(p) => {
    const t = document.elementFromPoint(p[0], p[1]);
    if (!t) return null;
    if (t === window.app.canvas.canvas) return "CANVAS";
    return t.tagName + (t.id ? "#" + t.id : "") + " " + String(t.className).slice(0, 70);
}"""

JS_WIDGET_AT = """(p) => {
    const c = window.app.canvas;
    const w = c.getWidgetAtCursor ? c.getWidgetAtCursor() : null;
    return { name: w ? w.name : null, at: p };
}"""

JS_TOOLTIP = """() => {
    const el = document.querySelector(".node-tooltip");
    if (!el) return null;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") return null;
    const r = el.getBoundingClientRect();
    const text = el.textContent || "";
    if (!text.trim()) return null;
    return { x: r.x, y: r.y, w: r.width, h: r.height, lines: text.split("\\n").length, text: text };
}"""

JS_AFTER_LITE = """(p) => {
    const app = window.app;
    const cv = app.canvas;
    const el = document.querySelector(".node-tooltip");
    const cs = el ? getComputedStyle(el) : null;
    let top_el = null;
    if (p) {
        const t = document.elementFromPoint(p[0], p[1]);
        if (t) {
            top_el = t === cv.canvas ? "CANVAS"
                : t.tagName + (t.id ? "#" + t.id : "") + " " + String(t.className).slice(0, 70);
        }
    }
    return {
        canvas_mouse: cv.mouse ? [...cv.mouse] : null,
        graph_mouse: cv.graph_mouse ? [...cv.graph_mouse] : null,
        node_over: cv.node_over ? cv.node_over.id : null,
        tooltip_display: cs ? cs.display : null,
        tooltip_text: el ? (el.textContent || "").slice(0, 80) : null,
        top_element: top_el,
    };
}"""

JS_ACTIONBAR = """(tip) => {
    const sel = `[data-testid="action-bar-buttons"] button[aria-label="${tip}"]`;
    const btn = document.querySelector(sel);
    if (!btn) {
        const all = [...document.querySelectorAll('[data-testid="action-bar-buttons"] button')]
            .map(b => ({ label: b.getAttribute("aria-label"), title: b.title }));
        return { found: false, buttons: all };
    }
    return { found: true, title: btn.title, aria: btn.getAttribute("aria-label"),
             rect: (() => { const r = btn.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; })() };
}"""

JS_STABLE = """(cls) => {
    const app = window.app;
    const n = app.graph._nodes.find(x => (x.comfyClass || x.type) === cls && x._mmProbe);
    if (!n) return null;
    const s = n.size ? n.size.join(",") : "";
    const ws = (n.widgets || []).map(w => w.computedHeight || 0).join(",");
    return s + "|" + ws;
}"""

JS_CLEANUP = """() => {
    const app = window.app;
    for (const n of [...app.graph._nodes]) {
        if (n._mmProbe) app.graph.remove(n);
    }
    return true;
}"""


def rect_overlap(a, b):
    """a/b = (x, y, w, h) in viewport coords -> overlap area in px^2."""
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2]); y1 = min(a[1] + a[3], b[1] + b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def rect_gap(a, b):
    """Minimum distance between two rects (0 when overlapping)."""
    dx = max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]), 0)
    dy = max(b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]), 0)
    return round((dx * dx + dy * dy) ** 0.5, 1)


def widget_hover_candidates(cr, cpos, wd, next_wd):
    """Screen-space hover points for a widget, in preference order.

    Regular widgets: a single point just below the widget row (works for all
    canvas-drawn widgets).

    customtext widgets: frontend 1.58 hit-tests the region from the widget's
    baseline to the next widget's baseline (empirically verified: hovering a
    point in that region shows this widget's tooltip), but draws the input as
    a DOM <textarea> (class comfy-multiline-input) that covers most of that
    region and intercepts pointer events. The row body above the baseline
    belongs to the PREVIOUS widget's hit region, so only points in the region
    below the baseline are candidates; the caller must pick the first point
    where the canvas element is topmost (JS_TOP_EL == "CANVAS").
    """
    cx = cr["x"] + wd["x"] - cpos[0] + min(8, max(2, (wd["w"] or 24) // 2))
    bottom = cr["y"] + wd["y"] - cpos[1]
    if wd["type"] != "customtext":
        return [(cx, bottom + min(9, max(4, (wd["h"] or 24) // 2)))]
    next_bottom = (cr["y"] + next_wd["y"] - cpos[1]) if next_wd else bottom + 40
    # start 2 px below the baseline: the exact baseline boundary can still be
    # assigned to the previous widget's region by the frontend hit test
    return [(cx, bottom + dy) for dy in range(2, max(3, int(next_bottom - bottom)), 4)]


def first_canvas_point(page, cr, pts):
    """First (x, y) of pts inside the canvas rect where the canvas is topmost,
    else (None, None)."""
    for (px, py) in pts:
        if not (cr["x"] <= px < cr["x"] + cr["w"] and cr["y"] <= py < cr["y"] + cr["h"]):
            continue
        if page.evaluate(JS_TOP_EL, [px, py]) == "CANVAS":
            return px, py
    return None, None


def run_probe(base_url, nodes, positions, out_json):
    from playwright.sync_api import sync_playwright

    records = []

    def record(cls, position, target, trect, widget_rect, extra=None):
        rec = {
            "node": cls, "position": position, "target": target,
            "tooltip_rect": None, "inside_viewport": None,
            "overlap_px": None, "overlap_pct": None, "gap_px": None,
            "appears": False, "hides_on_leave": None,
        }
        if trect:
            rec["tooltip_rect"] = [round(trect["x"], 1), round(trect["y"], 1),
                                   round(trect["w"], 1), round(trect["h"], 1), trect.get("lines", 1)]
            rec["inside_viewport"] = (0 <= trect["x"] and trect["x"] + trect["w"] <= VIEWPORT["width"]
                                      and 0 <= trect["y"] and trect["y"] + trect["h"] <= VIEWPORT["height"])
            if widget_rect:
                ov = rect_overlap((trect["x"], trect["y"], trect["w"], trect["h"]), widget_rect)
                rec["overlap_px"] = round(ov, 1)
                rec["overlap_pct"] = round(100.0 * ov / max(1.0, widget_rect[2] * widget_rect[3]), 1)
                rec["gap_px"] = rect_gap((trect["x"], trect["y"], trect["w"], trect["h"]), widget_rect)
        if extra:
            rec.update(extra)
        records.append(rec)
        return rec

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(base_url, wait_until="networkidle", timeout=45000)
        page.wait_for_function(
            f"{JS_READY.replace('NODES', json.dumps(nodes))}", timeout=30000)
        page.wait_for_timeout(500)

        # The frontend arms its tooltip timer with the *settings store* value of
        # 'LiteGraph.Node.TooltipDelay', which can differ from the LiteGraph class
        # constant (e.g. a user setting of 1500 ms). Both are read below and the
        # real end-to-end delay is calibrated on the first node placement.
        const_delay = page.evaluate(
            "() => window.LiteGraph?.Node?.TooltipDelay ?? window.LiteGraph?.LGraphNode?.TooltipDelay ?? 600")
        cfg_delay = page.evaluate("""() => {
            try {
                const v = window.app?.settings?.getSettingValue?.('LiteGraph.Node.TooltipDelay');
                if (typeof v === 'number' && isFinite(v)) return v;
            } catch (e) {}
            return null;
        }""")
        print(f"TooltipDelay: class constant = {const_delay} ms, settings store = {cfg_delay} ms")
        tooltip_delay = None

        first_rect = None
        for cls in nodes:
            for position in positions:
                # create the node, wait until its layout is stable (widget lists
                # and async option loads finish), only then place it at the edge
                page.evaluate(JS_CREATE, cls)
                stable = ""
                for _ in range(60):
                    s = page.evaluate(JS_STABLE, cls)
                    if s and s == stable:
                        break
                    stable = s or ""
                    page.wait_for_timeout(150)
                page.evaluate(JS_POSITION, {"cls": cls, "position": position})
                # let the render loop pick up the move (updates the hit-test areas)
                page.wait_for_timeout(400)
                info = page.evaluate(JS_READ, cls)
                if not info:
                    print(f"  !! {cls} {position}: node not found")
                    continue
                print(f"\n=== {cls} @ {position}  (node {info['w']}x{info['h']}, title_mode={info['title_mode']!r})")
                cpos = info.get("cpos") or [0, 0]
                if first_rect is None:
                    first_rect = {k: round(v, 1) for k, v in info["rect"].items()}

                if tooltip_delay is None:
                    # calibrate the real end-to-end delay: test-hover the first
                    # widget point where the canvas element is topmost (DOM
                    # overlays such as the tab strip or the customtext
                    # textareas would otherwise swallow the hover) and poll
                    # until that widget's tooltip appears
                    cr0 = info["rect"]
                    vis = [w for w in info["widgets"] if not w["hidden"]]
                    cal = (None, None)
                    for i, wd0 in enumerate(vis):
                        nxt = vis[i + 1] if i + 1 < len(vis) else None
                        cal = first_canvas_point(page, cr0,
                                                 widget_hover_candidates(cr0, cpos, wd0, nxt))
                        if cal[0] is not None:
                            break
                    if cal[0] is None:
                        tooltip_delay = (cfg_delay or const_delay) + 400
                        print(f"  calibration: no canvas-hoverable widget point; assuming {tooltip_delay} ms")
                    else:
                        cx0, cy0 = cal
                        page.mouse.move(cx0, cy0)
                        t0 = time.monotonic()
                        appeared = None
                        for _ in range(40):  # up to 4 s
                            page.wait_for_timeout(100)
                            if page.evaluate(JS_TOOLTIP):
                                appeared = (time.monotonic() - t0) * 1000
                                break
                        page.mouse.move(5, 5)
                        page.wait_for_timeout(400)
                        if appeared is not None:
                            tooltip_delay = int(max(appeared, cfg_delay or 0, const_delay or 0)) + 400
                            print(f"  calibrated tooltip delay: {tooltip_delay} ms "
                                  f"(appearance {appeared:.0f} ms at ({cx0:.0f},{cy0:.0f}))")
                        else:
                            tooltip_delay = (cfg_delay or const_delay) + 400
                            print(f"  calibration: no tooltip within 4 s; assuming {tooltip_delay} ms")

                # description tooltip: the frontend only shows it when the mouse
                # is ABOVE the node top edge (graph_mouse[1] < node.pos[1]).
                # DOM overlays (tab strip, third-party panels) often cover the
                # strip closest to the top bar, so try several offsets.
                cr = info["rect"]
                np = info["pos"]
                node_top_y = cr["y"] + np[1] - cpos[1]
                cx = cr["x"] + np[0] + info["w"] / 2 - cpos[0]
                title_rect = (cr["x"] + np[0] - cpos[0] + 4, node_top_y, info["w"] - 8, 20)
                desc_pts = [(cx, node_top_y - off) for off in (12, 24, 36, 48, 64, 80)]
                cy = node_top_y - 12
                dx, dy = first_canvas_point(page, cr, desc_pts)
                if dx is None:
                    # no offset reached the canvas: record the first in-rect
                    # point and what covers it
                    for (px, py) in desc_pts:
                        if cr["x"] <= px < cr["x"] + cr["w"] and cr["y"] <= py < cr["y"] + cr["h"]:
                            rec = record(cls, position, "description", None, title_rect,
                                         {"expected_text_len": len(info["description"])})
                            rec["off_canvas"] = True
                            rec["blocked_by"] = page.evaluate(JS_TOP_EL, [px, py])
                            rec["hover_xy"] = [round(px, 1), round(py, 1)]
                            print(f"  description: BLOCKED by DOM overlay at ({px:.0f},{py:.0f}): {rec['blocked_by']}")
                            break
                    else:
                        rec = record(cls, position, "description", None, title_rect,
                                     {"expected_text_len": len(info["description"])})
                        rec["off_canvas"] = True
                        rec["hover_xy"] = [round(cx, 1), round(cy, 1)]
                        print(f"  description: OFF-CANVAS hover point ({cx:.0f},{cy:.0f}) - not hoverable")
                else:
                    cx, cy = dx, dy
                    page.mouse.move(cx, cy)
                    page.wait_for_timeout(tooltip_delay + 350)
                    trect = page.evaluate(JS_TOOLTIP)
                    rec = record(cls, position, "description", trect, title_rect,
                                 {"expected_text_len": len(info["description"])})
                    if trect:
                        rec["text_ok"] = trect["text"] == info["description"]
                        print(f"  description: rect={rec['tooltip_rect']} inside={rec['inside_viewport']} "
                              f"overlap={rec['overlap_pct']}% gap={rec['gap_px']}px")
                    else:
                        rec["diag"] = page.evaluate(JS_AFTER_LITE, [cx, cy])
                        print(f"  description: NO tooltip appeared (diag={rec['diag']})")
                    page.mouse.move(5, 5)
                    page.wait_for_timeout(350)
                    still = page.evaluate(JS_TOOLTIP)
                    rec["hides_on_leave"] = still is None
                    if still:
                        print(f"  description: STILL VISIBLE after mouse left: {still['text'][:60]!r}")

                for wi, wd in enumerate(info["widgets"]):
                    if wd["hidden"]:
                        continue
                    nxt = info["widgets"][wi + 1] if wi + 1 < len(info["widgets"]) else None
                    cands = widget_hover_candidates(cr, cpos, wd, nxt)
                    cx, cy = cands[0]
                    widget_rect = (cr["x"] + wd["x"] - cpos[0], cr["y"] + wd["y"] - cpos[1],
                                   wd["w"], wd["h"] or 24)
                    # runtime widget tooltip wins, otherwise the schema tooltip
                    want = wd["tooltip"] if wd["tooltip"] else (wd.get("schema_tip") or "")
                    label = f"{wd['name']} ({wd['type']})"
                    if wd["type"] == "button":
                        label = f"button {wd['label']!r}"
                    sel_x, sel_y = first_canvas_point(page, cr, cands)
                    if sel_x is None:
                        # no candidate reached the canvas: record the preferred
                        # point and what covers it (off-canvas top bar / queue
                        # strip, tab strip, customtext textarea overlay, ...)
                        rec = record(cls, position, label, None, widget_rect)
                        rec["off_canvas"] = True
                        rec["hover_xy"] = [round(cx, 1), round(cy, 1)]
                        rec["expected_text_len"] = len(want)
                        in_rect = [p for p in cands
                                   if cr["x"] <= p[0] < cr["x"] + cr["w"]
                                   and cr["y"] <= p[1] < cr["y"] + cr["h"]]
                        if in_rect:
                            rec["blocked_by"] = page.evaluate(JS_TOP_EL, [in_rect[0][0], in_rect[0][1]])
                            print(f"  {label:<38} BLOCKED by DOM overlay near ({cx:.0f},{cy:.0f}): {rec['blocked_by']}")
                        else:
                            print(f"  {label:<38} OFF-CANVAS hover point ({cx:.0f},{cy:.0f}) - not hoverable")
                        continue
                    page.mouse.move(sel_x, sel_y)
                    page.wait_for_timeout(tooltip_delay + 350)
                    got = page.evaluate(JS_WIDGET_AT, [sel_x, sel_y])
                    trect = page.evaluate(JS_TOOLTIP)
                    if want:
                        if trect and trect["text"] == want:
                            rec = record(cls, position, label, trect, widget_rect)
                            rec["hit_widget"] = (got["name"] == wd["name"])
                            rec["expected_text_len"] = len(want)
                            mark = "" if rec["inside_viewport"] else "  <== OFFSCREEN"
                            print(f"  {label:<38} rect={rec['tooltip_rect']} inside={rec['inside_viewport']} "
                                  f"overlap={rec['overlap_pct']}% gap={rec['gap_px']}px{mark}")
                        else:
                            rec = record(cls, position, label, None, widget_rect)
                            rec["hit_widget"] = (got["name"] == wd["name"])
                            rec["expected_text_len"] = len(want)
                            rec["got"] = (trect or {}).get("text", "")[:80]
                            rec["diag"] = page.evaluate(JS_AFTER_LITE, [sel_x, sel_y])
                            print(f"  {label:<38} MISSING (expected {len(want)} chars)"
                                  + (f", got {rec['got']!r}" if rec["got"] else "")
                                  + f" diag={rec['diag']}")
                    else:
                        if trect is None:
                            rec = record(cls, position, label + " (no tooltip)", None, widget_rect)
                            rec["hit_widget"] = (got["name"] == wd["name"])
                        else:
                            rec = record(cls, position, label + " (unexpected tooltip)", trect, widget_rect)
                            rec["got"] = trect["text"][:80]
                            print(f"  {label:<38} UNEXPECTED tooltip {trect['text'][:60]!r}")
                    page.mouse.move(5, 5)
                    page.wait_for_timeout(300)
                    still = page.evaluate(JS_TOOLTIP)
                    if still:
                        rec = records[-1]
                        rec["hides_on_leave"] = False
                        print(f"  {label:<38} STILL VISIBLE after mouse left: {still['text'][:60]!r}")
                    else:
                        records[-1]["hides_on_leave"] = True

        # free-GPU action bar button: native title -> existence + values only
        ab = page.evaluate(JS_ACTIONBAR, FREE_GPU_TIP)
        print(f"\nfree-GPU action bar button: {ab}")
        records.append({"node": "freeGPU (action bar)", "position": "n/a",
                        "target": "title/aria-label", "found": ab.get("found"),
                        "title": ab.get("title"), "aria": ab.get("aria"),
                        "rect": ab.get("rect"),
                        "note": "native OS tooltip - not measurable in headless Chromium"})

        page.evaluate(JS_CLEANUP)
        browser.close()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"viewport": VIEWPORT, "tooltip_delay_ms": tooltip_delay,
                  "tooltip_delay_class_ms": const_delay, "tooltip_delay_settings_ms": cfg_delay,
                  "canvas_rect": first_rect, "records": records},
                  f, indent=2, ensure_ascii=False)
    print(f"\nMeasurements written to {out_json} ({len(records)} records)")

    # summary
    off = [r for r in records if r.get("inside_viewport") is False]
    offc = [r for r in records if r.get("off_canvas")]
    cov = [r for r in records if (r.get("overlap_pct") or 0) > 50]
    gap = [r for r in records if r.get("gap_px") is not None and r["gap_px"] > 40]
    miss = [r for r in records if r.get("expected_text_len", 0) > 0
            and r.get("tooltip_rect") is None and not r.get("off_canvas")]
    appear = [r for r in records if r.get("tooltip_rect")]
    hide = [r for r in records if r.get("hides_on_leave") is False]
    print("\n=== SUMMARY ===")
    print(f"tooltips appeared:         {len(appear)} / {len(records)}")
    print(f"offscreen tooltips:        {len(off)}  {[ (r['node'], r['position'], r['target']) for r in off ]}")
    print(f"off-canvas hover points:   {len(offc)}  "
          f"{[ (r['node'], r['position'], r['target'], r.get('blocked_by')) for r in offc ]}")
    print(f"covers hovered element:    {len(cov)}  {[ (r['node'], r['position'], r['target'], r['overlap_pct']) for r in cov ]}")
    print(f"distance > 40 px:          {len(gap)}")
    print(f"expected tooltip missing:  {len(miss)}  {[ (r['node'], r['position'], r['target']) for r in miss ]}")
    print(f"stayed visible on leave:   {len(hide)}")
    if not (off or cov or gap or miss or hide):
        print("ALL TOOLTIP POSITION CHECKS OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="", help="Existing ComfyUI URL (default: spawn headless CPU ComfyUI on port 8200)")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--nodes", default=",".join(NODES), help="Comma list of node IDs to probe")
    ap.add_argument("--positions", default=",".join(POSITIONS), help="Space of canvas positions: center, right, bottom")
    args = ap.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]
    repo_tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
    os.makedirs(repo_tmp, exist_ok=True)
    out_json = os.path.join(repo_tmp, "18-tooltip-measurements.json")

    proc = None
    log_f = None
    with tempfile.TemporaryDirectory(prefix="mm_ttip_", ignore_cleanup_errors=True) as tmpdir:
        try:
            if args.url:
                base_url = args.url.rstrip("/")
                print(f"Targeting existing ComfyUI at {base_url}")
                assert wait_server(base_url, timeout_s=10), f"Server not reachable at {base_url}"
            else:
                base_url = f"http://127.0.0.1:{args.port}"
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
                log_path = os.path.join(tmpdir, "comfy_headless.log")
                print(f"Starting headless CPU ComfyUI on {base_url} ...")
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

            # one small run so Preview & Save MM gets its runtime save buttons
            src = make_gradient_png(os.path.join(tmpdir, "probe_src.png"), 300, 200)
            code, up_info = http_json(f"{base_url}/object_info/MMUpscaleCrop")
            model = up_info["MMUpscaleCrop"]["input"]["required"]["upscale_model"][0][0]
            prompt = {
                "1": {"class_type": "MMLoadImage",
                      "inputs": {"image": "", "path": src, "mode": "single", "folder": "",
                                 "batch_mode": False, "index": 1}},
                "2": {"class_type": "MMUpscaleCrop",
                      "inputs": {"image": ["1", 0], "target_mode": "scale_factor",
                                 "scale_factor": 1.0, "scale_percent": 100.0, "shortest_side": 512,
                                 "longest_side": 512, "megapixels": 1.0, "target_width": 512,
                                 "target_height": 512, "resize_condition": "always",
                                 "method": "lanczos", "upscale_model": model, "snap": False,
                                 "multiple_of": 8, "crop": False, "crop_ratio": "1:1",
                                 "crop_width": 256, "crop_height": 256, "crop_orientation": "auto",
                                 "crop_position": "center", "crop_offset_x": 0, "crop_offset_y": 0,
                                 "crop_seed": 0}},
                "3": {"class_type": "MMPreviewSave",
                      "inputs": {"images": ["2", 0], "auto_save": False, "auto_send": False,
                                 "source": "auto", "suffix": "_#", "save_as": "",
                                 "format": "same as source", "quality": 95}},
            }
            queue_and_wait(base_url, prompt)
            print("Test workflow executed (runtime buttons now exist on Preview & Save MM).")

            run_probe(base_url, nodes, positions, out_json)
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
