import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import { MM_THEME, setWidgetVisible, fitNodeSize, debounce } from "./lib/mm_ui.mjs";

// Upscale Crop MM -- widget visibility, the live preview panel, and the
// migration of workflows saved with the old positional layout.
//
// The node is image-only (image in, image + width + height + info out) and
// has two stages, but only a handful of widgets matter at any one time:
// target_mode decides which size field is live, method decides whether the
// model picker is, and crop decides whether the second block exists.
// Everything that does not currently apply is hidden (LiteGraph has no
// show/hide API, so the shared setWidgetVisible swaps the widget's type to
// "hidden" and keeps its value so it can come back).

const NODE = "MMUpscaleCrop";

const TARGET_MODES = ["scale_factor", "scale_percent", "shortest_side",
                      "longest_side", "megapixels", "custom"];
// 'custom' takes the target_width x target_height box and always crops to it
// -- typing a resolution and not getting it back would be no use to anyone --
// so the crop toggle and the shape picker both fold away there.
const BOX_MODES = ["custom"];
// keep in sync with METHODS in py/upscale_crop.py; the first entry is the
// fallback both sides use, so the order matters
const METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "area", "upscaler model"];
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

// The box target_mode 'custom' covers and crops to ...
const TARGET_BOX_WIDGETS = ["target_width", "target_height"];
// ... and the dedicated box crop_ratio 'custom' cuts out of the result.
const CROP_BOX_WIDGETS = ["crop_width", "crop_height"];

// Fallbacks for the numeric widgets. ComfyUI's frontend can leave a widget at
// null until it is touched, and a null fails server-side validation before
// execute() is ever called, so this is fixed on the widget itself rather than
// in Python.
// multiple_of used to be a combo of fixed strings; a workflow saved then
// restores "8" or "off" into what is now a number field, so it is repaired
// alongside the numeric widgets even though it is a required one.
const OPTIONAL_DEFAULTS = {
    multiple_of: 8,
    scale_percent: 200.0,
    scale_factor: 2.0,
    shortest_side: 1536,
    longest_side: 2048,
    megapixels: 2.0,
    target_width: 1920,
    target_height: 1080,
    crop_width: 1024,
    crop_height: 1024,
    crop_offset_x: 0,
    crop_offset_y: 0,
    crop_seed: 0,
    snap: false,
    crop: false,
};

// The combos are repaired against their option lists and NOT against
// OPTIONAL_DEFAULTS: a combo keeps every valid saved value as-is, while
// OPTIONAL_DEFAULTS would coerce its strings to numbers and reset a saved
// "downscale_only" to the fallback.
const COMBO_VALUES = {
    target_mode: TARGET_MODES,
    method: METHODS,
    crop_ratio: RATIOS,
    crop_orientation: ORIENTATIONS,
    crop_position: CROP_POSITIONS,
    resize_condition: RESIZE_CONDITIONS,
};

// Workflows saved before the node became image-only carry their widget values
// positionally in this order (no image input, no crop_width/crop_height).
// migrateLegacyWidgets uses it to re-map such a list onto the current
// widgets by name.
const OLD_WIDGET_ORDER = [
    "target_mode", "scale_factor", "method", "snap", "multiple_of",
    "upscale_model", "scale_percent", "shortest_side", "longest_side",
    "megapixels", "target_width", "target_height", "crop", "crop_ratio",
    "crop_orientation", "crop_position", "crop_offset_x", "crop_offset_y",
    "crop_seed", "control_after_generate", "resize_condition",
];

// Everything the preview depends on. upscale_model is not here: swapping the
// model changes nothing the preview can show, since it draws the cached
// thumbnail rather than re-running the upscale.
const PREVIEW_WIDGETS = [
    "target_mode", "method", "multiple_of",
    "scale_percent", "scale_factor", "shortest_side", "longest_side", "megapixels",
    "target_width", "target_height",
    "crop", "crop_ratio", "crop_width", "crop_height", "crop_orientation",
    "crop_position", "crop_offset_x", "crop_offset_y", "crop_seed",
    "resize_condition", "snap",
];
const PREVIEW_DEBOUNCE_MS = 120;
const PREVIEW_MAX_H = 150;

// Remember the last chosen target_mode and main sizing parameters for newly
// created nodes (saved workflows always keep their own stored values).
const LAST_SETTINGS_KEY = "mm_upscale_crop_last_v1";
const REMEMBERED_WIDGETS = [
    "target_mode", "scale_factor", "scale_percent",
    "shortest_side", "longest_side", "megapixels",
    "target_width", "target_height", "crop_width", "crop_height",
];

function loadLastSettings() {
    try {
        const raw = window.localStorage?.getItem(LAST_SETTINGS_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
        return null;
    }
}

function saveLastSettings(node) {
    try {
        const prev = loadLastSettings() || {};
        for (const name of REMEMBERED_WIDGETS) {
            const w = widget(node, name);
            if (w && w.value !== null && w.value !== undefined) prev[name] = w.value;
        }
        window.localStorage?.setItem(LAST_SETTINGS_KEY, JSON.stringify(prev));
    } catch { /* storage blocked or full */ }
}

function applyLastSettings(node) {
    const saved = loadLastSettings();
    if (!saved) return;
    for (const name of REMEMBERED_WIDGETS) {
        const w = widget(node, name);
        if (!w || saved[name] === undefined || saved[name] === null) continue;
        if (name === "target_mode" && !TARGET_MODES.includes(saved[name])) continue;
        if (name !== "target_mode" && (typeof saved[name] !== "number" || Number.isNaN(saved[name]))) continue;
        w.value = saved[name];
    }
}

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// Write a value into a widget the way a user interaction would, so its
// callback (preview refresh, visibility) fires.
function setWidgetValue(node, name, value) {
    const w = widget(node, name);
    if (!w) return;
    w.value = value;
    w.callback?.(value);
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

// The core onConfigure has already poured the old positional values into the
// current widgets in the NEW order, so every name the old layout knew holds
// the wrong value. Recognise the old fingerprint and pour the right values
// back in by name.
function migrateLegacyWidgets(node, info) {
    if (info?.widgets_values_named) return;   // new-style save: values already by name
    const wv = info?.widgets_values;
    if (!Array.isArray(wv) || wv.length < 20) return;
    if (!METHODS.includes(wv[2]) || typeof wv[3] !== "boolean") return;

    OLD_WIDGET_ORDER.forEach((name, i) => {
        const w = widget(node, name);
        if (w && wv[i] !== undefined) w.value = wv[i];
    });
    // the new image input did not exist in the old layout, so it holds the
    // first old value (a target_mode string) instead of a file name
    const iw = widget(node, "image");
    if (iw && typeof iw.value === "string" && iw.value) iw.value = "";
    // the dedicated crop box did not exist either: start it clean and only
    // then carry over the old shared box, which a custom crop shape used to
    // borrow from target_width/height
    const cw = widget(node, "crop_width");
    const ch = widget(node, "crop_height");
    if (cw) cw.value = 1024;
    if (ch) ch.value = 1024;
    const ratio = widget(node, "crop_ratio")?.value;
    const mode = widget(node, "target_mode")?.value;
    if (ratio === "custom" && !BOX_MODES.includes(mode)) {
        const tw = widget(node, "target_width");
        const th = widget(node, "target_height");
        if (cw && tw?.value != null) cw.value = tw.value;
        if (ch && th?.value != null) ch.value = th.value;
    }
}

function updateVisibility(node) {
    const mode = widget(node, "target_mode")?.value;
    const method = widget(node, "method")?.value;
    const cropOn = widget(node, "crop")?.value === true;
    const ratio = widget(node, "crop_ratio")?.value;
    const position = widget(node, "crop_position")?.value;

    const box = BOX_MODES.includes(mode);

    // one size field, whichever the mode is named after
    for (const name of SIZE_WIDGETS) setWidgetVisible(node, name, name === mode);
    // the target box belongs to 'custom', the crop box to a custom-shaped crop
    for (const name of TARGET_BOX_WIDGETS) setWidgetVisible(node, name, box);
    for (const name of CROP_BOX_WIDGETS) {
        setWidgetVisible(node, name, cropOn && !box && ratio === "custom");
    }

    // the multiple only means anything while snapping
    setWidgetVisible(node, "multiple_of", widget(node, "snap")?.value === true);

    // the model picker only exists for method = model
    setWidgetVisible(node, "upscale_model", method === "upscaler model");

    // A box mode always crops to its box, so the toggle and the shape picker
    // would only be a second crop arguing with the first. They fold away and
    // just the framing controls stay, since those still decide what survives
    const framing = box || cropOn;
    const isRandom = position === "random";
    setWidgetVisible(node, "crop", !box);
    setWidgetVisible(node, "crop_ratio", cropOn && !box);
    // custom takes the box above; a named format takes an orientation
    // instead, and a square has no orientation to take
    setWidgetVisible(node, "crop_orientation",
        cropOn && !box && ratio !== "custom" && ratio !== "1:1");
    setWidgetVisible(node, "crop_position", framing);
    // fixed pixel offsets contradict random placement: hide them in random mode
    setWidgetVisible(node, "crop_offset_x", framing && !isRandom);
    setWidgetVisible(node, "crop_offset_y", framing && !isRandom);
    // the seed only means anything to the random placement; its
    // control_after_generate companion goes wherever the seed goes
    const seedVis = framing && isRandom;
    setWidgetVisible(node, "crop_seed", seedVis);
    const ctrlw = widget(node, "control_after_generate");
    if (ctrlw) {
        if (seedVis) {
            // it was parked on 'fixed' while hidden: bring the saved mode back
            const stashed = node.properties?.mm_crop_ctrl;
            if (stashed && ctrlw.value !== stashed) {
                ctrlw.value = stashed;
                ctrlw.callback?.(ctrlw.value);
            }
            if (stashed && node.properties) delete node.properties.mm_crop_ctrl;
            setWidgetVisible(node, "control_after_generate", true);
        } else {
            // a non-fixed mode on a hidden widget would still fire on runs
            // that never use it, so park it on 'fixed' and remember it
            if (ctrlw.value && ctrlw.value !== "fixed") {
                (node.properties ??= {}).mm_crop_ctrl = ctrlw.value;
                ctrlw.value = "fixed";
                ctrlw.callback?.(ctrlw.value);
            }
            setWidgetVisible(node, "control_after_generate", false);
        }
    }
    // the failsafe only gates absolute targets; the two relative modes always
    // resize, so for them the widget would be a second lie
    setWidgetVisible(node, "resize_condition", ABSOLUTE_MODES.includes(mode));

    fitNodeSize(node, { shrinkHeight: true });
}

app.registerExtension({
    name: "MM.UpscaleCrop",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;  // the MM logo is attached centrally by web/mm_logo.js

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            // flags the callbacks: a restored crop_position must not wipe the
            // offsets that are about to be restored right behind it
            this._mmuConfiguring = true;
            onConfigure?.apply(this, arguments);
            migrateLegacyWidgets(this, info);
            this._mmuConfiguring = false;
            repairCombos(this);
            repairOptionals(this);
            updateVisibility(this);
            const self = this;
            setTimeout(() => {
                // ComfyUI appends a seed's control widget after this hook has
                // run, so one pass here cannot see it; and the preview cache
                // is empty until the graph has run once
                updateVisibility(self);
                self._mmPreviewRefresh?.();
            }, 0);
        };

        // Wiring or unwiring an input can change what has to stay visible.
        const onConnections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slot, type, isConnect, linkIds) {
            onConnections?.apply(this, arguments);
            updateVisibility(this);
        };
    },

    nodeCreated(node) {
        if ((node.comfyClass || node.type) !== NODE) return;

        // Freshly added nodes restore the last chosen mode & sizing values
        // (defaults to scale_factor = 2.0 when nothing was saved yet); saved
        // workflows overwrite these in onConfigure right after.
        if (!app.configuringGraph) applyLastSettings(node);
        repairCombos(node);
        repairOptionals(node);

        // Re-evaluate whenever one of the widgets that steers the others
        // changes. crop_ratio is in here because 'custom' swaps the
        // crop box in for the orientation combo.
        for (const name of ["target_mode", "method", "crop", "crop_ratio", "snap"]) {
            const w = widget(node, name);
            if (!w) continue;
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                updateVisibility(node);
                return r;
            };
        }
        for (const name of REMEMBERED_WIDGETS) {
            const w = widget(node, name);
            if (!w) continue;
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                if (!node._mmuConfiguring && !app.configuringGraph) saveLastSettings(node);
                return r;
            };
        }
        // A new anchor makes the saved offsets meaningless -- they were
        // written relative to the old anchor -- so they go back to zero.
        // Except while a workflow is restoring, when the saved offsets are
        // the truth the node has to keep.
        const posw = widget(node, "crop_position");
        if (posw) {
            const orig = posw.callback;
            posw.callback = function (...args) {
                const r = orig?.apply(this, args);
                if (!node._mmuConfiguring) {
                    setWidgetValue(node, "crop_offset_x", 0);
                    setWidgetValue(node, "crop_offset_y", 0);
                }
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
    // the text row -- the numbers side by side are the difference between
    // "why 832x832?" and "ah, the 1:1 took a square out of it".
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

    // padding/boxSizing/overflow make the wrap's border box exactly the
    // content, because the ComfyUI host around a DOM widget adds its own
    // 10px top and bottom margin and the height contract below has to absorb
    // it: the host sizes the panel from getMinHeight/getMaxHeight, and the
    // node resizes from the host, so any leftover slack shows up as empty
    // space inside the node.
    const wrap = document.createElement("div");
    Object.assign(wrap.style, {
        display: "flex", flexDirection: "column", gap: "4px",
        padding: "0", boxSizing: "border-box", overflow: "hidden",
    });
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
            if (d.png) bits.push("drag to place");
        }
        if (d.warning) bits.push("\u26a0 " + d.warning);
        return bits.join(" | ");
    }

    // The exact height of what the panel currently holds. The host adds its
    // own 20px of margin and the wrap 4px of gap on top of the two rows, so
    // the panel asks for that much more than the rows measure. With no image
    // yet there is only the status row, which gets a fixed line.
    function panelHeight() {
        if (stage.style.display === "none") return 44;
        const sh = parseInt(stage.style.height, 10) || stage.offsetHeight || 0;
        const th = Math.max(20, status.offsetHeight || status.scrollHeight || 20);
        return sh + th + 28;
    }

    function draw(d) {
        last = d;

        if (!d.png) {
            stage.style.display = "none";
            status.textContent = describe(d) + " | connect an image for a picture";
            fitNodeSize(node, { shrinkHeight: true });
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

        status.style.color = d.oom ? "#ff6b6b" : d.warning ? "#f5b041" : MM_THEME.dim;
        status.textContent = describe(d);
        fitNodeSize(node, { shrinkHeight: true });
    }

    // ---- drag the window ------------------------------------------------
    //
    // The anchor plus two offsets are precise but slow to aim with. Dragging
    // writes back into those same offsets rather than introducing a second
    // way of storing the position, so the widgets stay the single source of
    // truth and a dragged crop is still reproducible from the saved graph.

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
            setWidgetValue(node, "crop_position", "center");
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
            setWidgetValue(node, "crop_offset_x", Math.round(endX - baseX));
            setWidgetValue(node, "crop_offset_y", Math.round(endY - baseY));
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
        setWidgetValue(node, "crop_offset_x", 0);
        setWidgetValue(node, "crop_offset_y", 0);
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
                fitNodeSize(node, { shrinkHeight: true });
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
                fitNodeSize(node, { shrinkHeight: true });
            }
        }
    }

    const schedule = debounce(fetchPreview, PREVIEW_DEBOUNCE_MS);
    const refreshNow = () => fetchPreview();
    node._mmPreviewRefresh = refreshNow;

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

    const dw = node.addDOMWidget("upscale_preview", "preview", wrap, { serialize: false });
    // keep the widget out of the workflow and follow the node width on every
    // layout; the height is exactly the panel content, dynamic both ways
    dw.serialize = false;
    dw.options = { ...(dw.options || {}), serialize: false };
    dw.options.getMinHeight = () => panelHeight();
    dw.options.getMaxHeight = () => panelHeight();
    Object.defineProperty(dw, "width", { get: () => node.size[0], set() { }, configurable: true });
    refreshNow();
}

