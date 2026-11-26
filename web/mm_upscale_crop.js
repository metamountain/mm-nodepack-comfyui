import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import { MM_THEME, setWidgetVisible, fitNodeSize, debounce } from "./lib/mm_ui.js";

// Upscale Crop MM -- widget visibility and the live preview panel.
//
// The node has two stages and many widgets, but only a handful matter at any
// one time: target_mode decides which size field is live, method decides
// whether the model picker is, and crop decides whether the second block
// exists. Everything that does not currently apply is hidden (LiteGraph has
// no show/hide API, so the shared setWidgetVisible swaps the widget's type to
// "hidden" and keeps its value so it can come back).

const NODE = "MMUpscaleCrop";

const TARGET_MODES = ["scale_percent", "scale_factor", "shortest_side",
                      "longest_side", "megapixels", "custom"];
// 'custom' takes the target_width x target_height box and always crops to it
// -- typing a resolution and not getting it back would be no use to anyone --
// so the crop toggle and the shape picker both fold away there.
const BOX_MODES = ["custom"];
const METHODS = ["upscaler model", "lanczos", "bicubic", "bilinear", "nearest", "area"];
const RATIOS = ["custom", "4:5", "1:1", "4:3", "3:2", "DIN", "16:10", "16:9", "2:1", "21:9"];
const ORIENTATIONS = ["auto", "portrait", "landscape"];
const CROP_POSITIONS = ["center", "top", "bottom", "left", "right",
                        "top-left", "top-right", "bottom-left", "bottom-right",
                        "random"];
const RESIZE_CONDITIONS = ["always", "downscale_only", "upscale_only"];
// the absolute target modes, the only ones a resize_condition can gate
const ABSOLUTE_MODES = ["shortest_side", "longest_side", "megapixels", "custom"];

// The modes whose widget is simply named after them.
const SIZE_WIDGETS = ["scale_percent", "scale_factor", "shortest_side",
                      "longest_side", "megapixels"];

// One width x height box, shared by the box modes and a custom crop shape.
const BOX_WIDGETS = ["target_width", "target_height"];

// Fallbacks for the optional widgets. ComfyUI's frontend can leave an
// optional widget at null until it is touched, and a null fails server-side
// validation before execute() is ever called, so this is fixed on the widget
// itself rather than in Python.
// multiple_of used to be a combo of fixed strings; a workflow saved then
// restores "8" or "off" into what is now a number field, so it is repaired
// alongside the optional widgets even though it is a required one.
const OPTIONAL_DEFAULTS = {
    multiple_of: 8,
    scale_percent: 200.0,
    scale_factor: 2.0,
    shortest_side: 1536,
    longest_side: 2048,
    megapixels: 2.0,
    target_width: 1920,
    target_height: 1080,
    crop: false,
    crop_offset_x: 0,
    crop_offset_y: 0,
    crop_seed: 0,
    snap: false,
};

// resize_condition is repaired here and NOT in OPTIONAL_DEFAULTS: a combo
// keeps every valid saved value as-is, while OPTIONAL_DEFAULTS would coerce
// its strings to numbers and reset a saved "downscale_only" to the fallback.
const COMBO_VALUES = {
    target_mode: TARGET_MODES,
    method: METHODS,
    crop_ratio: RATIOS,
    crop_orientation: ORIENTATIONS,
    crop_position: CROP_POSITIONS,
    resize_condition: RESIZE_CONDITIONS,
};

// Everything the preview depends on. upscale_model is not here: swapping the
// model changes nothing the preview can show, since it draws the cached
// thumbnail rather than re-running the upscale.
const PREVIEW_WIDGETS = [
    "target_mode", "method", "multiple_of",
    "scale_percent", "scale_factor", "shortest_side", "longest_side", "megapixels",
    "target_width", "target_height",
    "crop", "crop_ratio", "crop_orientation",
    "crop_position", "crop_offset_x", "crop_offset_y", "crop_seed",
    "resize_condition", "snap",
];
const PREVIEW_DEBOUNCE_MS = 120;
const PREVIEW_MAX_H = 150;

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// A workflow saved before a widget existed restores its values by position,
// so every later widget receives the wrong one -- which is how a number can
// land in a combo that only accepts strings. Anything not a valid option
// goes back to the first.
function repairCombos(node) {
    for (const [name, valid] of Object.entries(COMBO_VALUES)) {
        const w = widget(node, name);
        if (!w) continue;
        if (!valid.includes(w.value)) w.value = valid[0];
    }
}

function repairOptionals(node) {
    for (const [name, fallback] of Object.entries(OPTIONAL_DEFAULTS)) {
        const w = widget(node, name);
        if (!w) continue;
        if (typeof w.value === "string") {
            const n = w.value.trim().toLowerCase();
            w.value = (n === "off" || n === "" || Number.isNaN(Number(n)))
                ? fallback : Number(n);
        }
        if (w.value === null || w.value === undefined ||
            (typeof w.value === "number" && Number.isNaN(w.value))) {
            w.value = fallback;
        }
        if (w._mmuGuarded) continue;
        w._mmuGuarded = true;
        const orig = w.serializeValue?.bind(w);
        w.serializeValue = async (...args) => {
            let v = orig ? await orig(...args) : w.value;
            if (v === null || v === undefined ||
                (typeof v === "number" && Number.isNaN(v))) v = fallback;
            return v;
        };
    }
}

function updateVisibility(node) {
    const mode = widget(node, "target_mode")?.value;
    const method = widget(node, "method")?.value;
    const cropOn = widget(node, "crop")?.value === true;
    const ratio = widget(node, "crop_ratio")?.value;
    const position = widget(node, "crop_position")?.value;

    const box = BOX_MODES.includes(mode);
    const autoCrops = box;

    // one size field, whichever the mode is named after
    for (const name of SIZE_WIDGETS) setWidgetVisible(node, name, name === mode);
    // one box serves both the box modes and a custom-shaped crop
    const customCrop = cropOn && !box && ratio === "custom";
    for (const name of BOX_WIDGETS) setWidgetVisible(node, name, box || customCrop);

    // the multiple only means anything while snapping
    setWidgetVisible(node, "multiple_of", widget(node, "snap")?.value === true);

    // the model picker only exists for method = model
    setWidgetVisible(node, "upscale_model", method === "upscaler model");

    // A box mode always crops to its box, so the toggle and the shape picker
    // would only be a second crop arguing with the first. They fold away and
    // just the framing controls stay, since those still decide what survives
    const framing = autoCrops || cropOn;
    setWidgetVisible(node, "crop", !autoCrops);
    // A box mode cuts to its box, so a shape picker here would only fight it
    setWidgetVisible(node, "crop_ratio", cropOn && !box);
    setWidgetVisible(node, "crop_position", framing);
    setWidgetVisible(node, "crop_offset_x", framing);
    setWidgetVisible(node, "crop_offset_y", framing);
    // the seed only means anything to the random placement; its
    // control_after_generate companion goes wherever the seed goes
    const seedVis = framing && position === "random";
    setWidgetVisible(node, "crop_seed", seedVis);
    setWidgetVisible(node, "control_after_generate", seedVis);
    // the failsafe only gates absolute targets; the two relative modes always
    // resize, so for them the widget would be a second lie
    setWidgetVisible(node, "resize_condition", ABSOLUTE_MODES.includes(mode));
    // custom takes the box above; a named format takes an orientation
    // instead, and a square has no orientation to take
    setWidgetVisible(node, "crop_orientation",
        cropOn && !box && ratio !== "custom" && ratio !== "1:1");

    fitNodeSize(node);
}

app.registerExtension({
    name: "MM.UpscaleCrop",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            repairCombos(this);
            repairOptionals(this);
            updateVisibility(this);
            const self = this;
            setTimeout(() => updateVisibility(self), 0);
        };

        // Wiring or unwiring an input can change what has to stay visible.
        const onConnections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slot, type, isConnect, linkIds) {
            onConnections?.apply(this, arguments);
            updateVisibility(this);
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== NODE) return;

        repairCombos(node);
        repairOptionals(node);

        // Re-evaluate whenever one of the widgets that steers the others
        // changes. crop_ratio is in here because 'custom' swaps the
        // pixel fields in for the orientation combo.
        for (const name of ["target_mode", "method", "crop", "crop_ratio", "crop_position", "snap"]) {
            const w = widget(node, name);
            if (!w) continue;
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                updateVisibility(node);
                return r;
            };
        }

        buildPreview(node);
        updateVisibility(node);
        // ComfyUI appends a seed's control widget after this hook has run,
        // so one pass here cannot see it. Catch it on the next tick.
        setTimeout(() => updateVisibility(node), 0);
    },
});

// ---------------------------------------------------------------- preview
//
// One image with the discarded area dimmed, the way every crop tool shows it
// -- two thumbnails side by side would each be half the size and the second
// would only repeat what the frame already says.
//
// The backend hands back the cached thumbnail plus the numbers, computed by
// the same functions the node runs, so the frame here is exactly what the
// graph will cut. Nothing is re-rendered per keystroke.

function buildPreview(node) {
    const stage = document.createElement("div");
    Object.assign(stage.style, {
        position: "relative", borderRadius: "3px", overflow: "hidden",
        background: MM_THEME.widgetBg, margin: "0 auto", display: "none",
    });

    const img = document.createElement("img");
    Object.assign(img.style, { display: "block", width: "100%", height: "100%" });

    const window_ = document.createElement("div");
    Object.assign(window_.style, {
        position: "absolute", border: "1px solid " + MM_THEME.accent,
        // neutral dim over the discarded area (no theme token for overlays)
        boxShadow: "0 0 0 999px rgba(0,0,0,.65)", display: "none",
        cursor: "move", touchAction: "none",
    });
    window_.title = "Drag to place the crop";

    const size = document.createElement("span");
    Object.assign(size.style, {
        position: "absolute", left: "3px", bottom: "3px",
        font: "600 8.5px " + MM_THEME.fontMono, color: "#0b1526",
        background: MM_THEME.accent, padding: "1px 5px", borderRadius: "2px",
        whiteSpace: "nowrap",
    });
    window_.appendChild(size);

    // The size the upscale produced, shown on the picture rather than only in
    // the text row -- when a crop_ratio reshapes the box you typed, having
    // both numbers side by side is the difference between "why 832x832?" and
    // "ah, the 1:1 took a square out of it".
    const upSize = document.createElement("span");
    Object.assign(upSize.style, {
        position: "absolute", left: "3px", top: "3px",
        font: "600 8.5px " + MM_THEME.fontMono, color: MM_THEME.labelHi,
        background: MM_THEME.bg, padding: "1px 5px",
        borderRadius: "2px", whiteSpace: "nowrap", pointerEvents: "none",
    });

    stage.appendChild(img);
    stage.appendChild(window_);
    stage.appendChild(upSize);

    const holder = document.createElement("div");
    Object.assign(holder.style, { display: "flex", justifyContent: "center" });
    holder.appendChild(stage);

    const status = document.createElement("div");
    Object.assign(status.style, {
        font: "9.5px " + MM_THEME.fontMono, color: MM_THEME.dim, textAlign: "center",
        padding: "3px 0", wordBreak: "break-word", lineHeight: "1.45",
    });
    status.textContent = "run the graph once to see a preview";

    const wrap = document.createElement("div");
    Object.assign(wrap.style, { display: "flex", flexDirection: "column", gap: "4px" });
    wrap.appendChild(holder);
    wrap.appendChild(status);

    let abort = null;
    let last = null;          // the last preview payload, for the drag maths

    // The whole readout in one place, so the picture-less path can use it too.
    function describe(d) {
        const [uw, uh] = d.up;
        const bits = [
            d.src[0] + "\u00d7" + d.src[1] + " \u2192 " + uw + "\u00d7" + uh,
            d.method,
            "\u00d7" + d.factor,
        ];
        if (d.multiple_of >= 1) bits.push("/" + Math.round(d.multiple_of));
        if (d.skipped) bits.push("skipped (" + d.skipped + ")");
        if (d.crop) {
            const shape = (d.ratio === "custom" || d.ratio === "1:1")
                ? d.ratio : d.ratio + " " + d.orientation;
            bits.push(shape + " " + d.position +
                      " \u2192 " + d.crop[0] + "\u00d7" + d.crop[1]);

            // The box you typed and the crop you get are two numbers, and a
            // named ratio reshapes the first into the second. Say so rather
            // than leaving it looking like a bug.
            const boxW = widget(node, "target_width")?.value;
            const boxH = widget(node, "target_height")?.value;
            const usesBox = d.ratio === "custom" ||
                            BOX_MODES.includes(widget(node, "target_mode")?.value);
            if (!usesBox && boxW && boxH && (d.crop[0] !== boxW || d.crop[1] !== boxH)) {
                bits.push("crop_ratio " + d.ratio + " reshapes the box \u2014 " +
                          "set crop_ratio to custom for " + boxW + "\u00d7" + boxH);
            }
            if (d.png) bits.push("drag to place");
        }
        return bits.join(" | ");
    }

    function draw(d) {
        last = d;

        // A latent-only run has no picture to draw -- a latent is not viewable
        // without the VAE, which this node has no access to. The numbers are
        // the useful part anyway, so they are shown on their own rather than
        // pretending there is nothing to report.
        if (!d.png) {
            stage.style.display = "none";
            status.textContent = describe(d) + " | connect an image for a picture";
            return;
        }
        stage.style.display = "block";
        img.src = "data:image/png;base64," + d.png;

        // fit the real dimensions into the panel, capped both ways, so a
        // portrait source does not stretch the node to twice its height
        const [uw, uh] = d.up;
        const maxW = holder.clientWidth || 260;
        const fit = Math.min(maxW / uw, PREVIEW_MAX_H / uh);
        stage.style.width = Math.max(24, Math.round(uw * fit)) + "px";
        stage.style.height = Math.max(24, Math.round(uh * fit)) + "px";

        upSize.textContent = uw + "\u00d7" + uh;

        if (d.rect && d.crop) {
            const [x0, y0, x1, y1] = d.rect;
            window_.style.left = (x0 * 100).toFixed(2) + "%";
            window_.style.top = (y0 * 100).toFixed(2) + "%";
            window_.style.width = ((x1 - x0) * 100).toFixed(2) + "%";
            window_.style.height = ((y1 - y0) * 100).toFixed(2) + "%";
            window_.style.display = "block";
            size.textContent = d.crop[0] + "\u00d7" + d.crop[1];
        } else {
            window_.style.display = "none";
        }

        status.textContent = describe(d);
    }

    // ---- drag the window ------------------------------------------------
    //
    // The anchor plus two offsets are precise but slow to aim with. Dragging
    // writes back into those same offsets rather than introducing a second
    // way of storing the position, so the widgets stay the single source of
    // truth and a dragged crop is still reproducible from the saved graph.

    function setWidget(name, value) {
        const w = widget(node, name);
        if (!w) return;
        w.value = value;
        w.callback?.(value);
    }

    function beginDrag(ev) {
        if (!last || !last.rect || !last.crop) return;
        ev.preventDefault();
        ev.stopPropagation();

        const [uw, uh] = last.up;
        const [cw, ch] = last.crop;
        const slackX = Math.max(0, uw - cw);
        const slackY = Math.max(0, uh - ch);
        if (slackX < 1 && slackY < 1) return;      // nothing to move

        // screen px -> image px
        const rect = stage.getBoundingClientRect();
        const perPxX = uw / Math.max(1, rect.width);
        const perPxY = uh / Math.max(1, rect.height);

        // A random placement has no anchor to offset from, so taking hold of
        // the window means taking manual control: it becomes a centre anchor
        // at wherever you drop it.
        let position = widget(node, "crop_position")?.value ?? "center";
        if (position === "random") {
            position = "center";
            setWidget("crop_position", "center");
        }
        // the anchor's base point, per position -- the offsets write back the
        // drop position measured from it, so corners offset on both axes
        const baseX = ["left", "top-left", "bottom-left"].includes(position) ? 0
                    : ["right", "top-right", "bottom-right"].includes(position) ? slackX
                    : slackX / 2;
        const baseY = ["top", "top-left", "top-right"].includes(position) ? 0
                    : ["bottom", "bottom-left", "bottom-right"].includes(position) ? slackY
                    : slackY / 2;

        const startX = ev.clientX, startY = ev.clientY;
        const startLeft = last.rect[0] * uw;
        const startTop = last.rect[1] * uh;
        let endX = startLeft, endY = startTop;

        function onMove(e) {
            // draw it locally while dragging -- a round trip per mouse move
            // would lag behind the cursor
            endX = Math.max(0, Math.min(slackX, startLeft + (e.clientX - startX) * perPxX));
            endY = Math.max(0, Math.min(slackY, startTop + (e.clientY - startY) * perPxY));
            window_.style.left = ((endX / uw) * 100).toFixed(3) + "%";
            window_.style.top = ((endY / uh) * 100).toFixed(3) + "%";
        }

        function onUp() {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            setWidget("crop_offset_x", Math.round(endX - baseX));
            setWidget("crop_offset_y", Math.round(endY - baseY));
            node.setDirtyCanvas(true, true);
            refreshNow();        // confirm against the backend's own maths
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
    }

    window_.addEventListener("pointerdown", beginDrag);

    // double-click puts it back to the anchor
    window_.addEventListener("dblclick", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        setWidget("crop_offset_x", 0);
        setWidget("crop_offset_y", 0);
        refreshNow();
    });

    async function fetchPreview() {
        abort?.abort();
        abort = new AbortController();
        const body = { node_id: String(node.id) };
        for (const name of PREVIEW_WIDGETS) {
            const w = widget(node, name);
            if (w) body[name] = w.value;
        }
        try {
            const d = await (await api.fetchApi("/mm/upscale/preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: abort.signal,
            })).json();
            if (!d?.ok) {
                stage.style.display = "none";
                status.textContent = d?.error || "no preview yet";
                return;
            }
            draw(d);
        } catch (err) {
            if (err?.name !== "AbortError") {
                // some frontends throw on non-2xx with a generic message;
                // only show the detail when it carries one
                const msg = String(err?.message ?? err);
                status.textContent = msg.includes("There was a problem with the request")
                    ? "preview failed"
                    : "preview failed: " + msg;
            }
        }
    }

    const schedule = debounce(fetchPreview, PREVIEW_DEBOUNCE_MS);
    const refreshNow = () => fetchPreview();

    // Re-render on every widget that matters, and again once the graph runs --
    // that is when a fresh image lands in the cache.
    for (const name of PREVIEW_WIDGETS) {
        const w = widget(node, name);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (...args) {
            const r = orig?.apply(this, args);
            schedule();
            return r;
        };
    }

    const onExecuted = node.onExecuted;
    node.onExecuted = function (...args) {
        const r = onExecuted?.apply(this, args);
        refreshNow();
        return r;
    };

    node.addDOMWidget("upscale_preview", "preview", wrap, { serialize: false });
    refreshNow();
}
