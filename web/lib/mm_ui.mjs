// MM Nodepack – shared frontend UI helpers, imported by all MM web scripts
// (mm_nodepack.js, mm_free_gpu.js, lib/comparer.mjs).
// Theme tokens, toasts, debounce, /view URL building, widget hide/dim, node sizing.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Theme: LiteGraph values with consistent dark fallbacks (used by all MM scripts).
// ---------------------------------------------------------------------------
const LG = (typeof window !== "undefined" && window.LiteGraph) || {};
const T = (LG.CONFIG && typeof LG.CONFIG === "object") ? LG.CONFIG : {};

export const MM_THEME = {
  accent: "#4c9aff",
  bg: T.NODE_BGCOLOR || "rgb(34, 34, 34)",
  bgHover: T.NODE_BGCOLOR_2 || "rgb(52, 52, 52)",
  widgetBg: T.WIDGET_BGCOLOR || "rgb(34, 34, 34)",
  outline: T.WIDGET_OUTLINE_COLOR || "#666",
  text: T.WIDGET_TEXT_COLOR || "#DDD",
  label: "rgba(170,170,170,.85)",
  labelHi: "rgba(225,225,225,1)",
  dim: "rgba(150,150,150,.6)",
  fontSans: "Inter, Arial, sans-serif",
  fontMono: 'Consolas, "Cascadia Code", "Liberation Mono", monospace',
};

// ---------------------------------------------------------------------------
// /view URLs: the single builder for all MM scripts. format/rand default to the
// frontend's own preview params (explicit "" disables them, e.g. for re-encoding
// a preview blob without the format param).
// ---------------------------------------------------------------------------
export function viewQuery(entry, { format, rand } = {}) {
  const q = new URLSearchParams({
    filename: entry.filename,
    subfolder: entry.subfolder || "",
    type: entry.type || "temp",
  });
  const fmt = format ?? (app.getPreviewFormatParam?.() ?? "");
  const rnd = rand ?? (app.getRandParam?.() ?? "");
  return q.toString() + fmt + rnd;
}

export function viewUrl(entry, opts) {
  return api.apiURL("/view?" + viewQuery(entry, opts));
}

// relative query for api.fetchApi("/view?" + viewQuery(...))
export async function viewBlob(entry, opts) {
  return (await api.fetchApi("/view?" + viewQuery(entry, opts))).blob();
}

export function loadImage(entry, opts) {
  const img = new Image();
  img.src = viewUrl(entry, opts);
  return img;
}

// ---------------------------------------------------------------------------
// Toast: single helper for all MM scripts (console fallback).
// ---------------------------------------------------------------------------
export function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary, detail, life: severity === "error" ? 6000 : 3000 });
  else console[severity === "error" ? "error" : "log"](`[MM] ${summary}: ${detail}`);
}

// ---------------------------------------------------------------------------
// Debounce
// ---------------------------------------------------------------------------
export function debounce(fn, ms) {
  let t = null;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

// ---------------------------------------------------------------------------
// Widget visibility: hide with w.type = "hidden" (new LiteGraph derives it from the
// type: the row is skipped in draw/layout), plus computeSize/y for older frontends.
// A widget whose matching input socket is linked is kept visible.
// ---------------------------------------------------------------------------
export function setWidgetVisible(node, name, visible) {
  const w = node?.widgets?.find((x) => x.name === name);
  if (!w) return;
  // wired input: the frontend manages this widget (e.g. image combo next to its link)
  const sock = (node.inputs || []).find((i) => i.name === name);
  const show = visible || sock?.link != null;
  if (!w._mmVis) {
    w._mmVis = true;
    w._mmType = w.type;
    w._mmY = w.y;
    w._mmHasOwnComputeSize = Object.prototype.hasOwnProperty.call(w, "computeSize");
    w._mmOrigComputeSize = w.computeSize;
    if (typeof w.drawWidget === "function") {
      const origDraw = w.drawWidget.bind(w);
      w.drawWidget = function (ctx, opts) {
        if (this.type === "hidden") return;
        return origDraw(ctx, opts);
      };
    }
  }
  if (show) {
    if (w.type === "hidden") w.type = w._mmType;
    if (w._mmHasOwnComputeSize) w.computeSize = w._mmOrigComputeSize;
    else if ("computeSize" in w) delete w.computeSize;
    if (w._mmY !== undefined) w.y = w._mmY;
  } else {
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w.y = -100000;
  }
}

// ---------------------------------------------------------------------------
// Widget dimming: one drawWidget opacity wrapper (globalAlpha * 0.35), replaces
// both the auto-state dim and the mode-button active/inactive shading.
// Set widget.disabled separately when the widget should also stop taking clicks.
// ---------------------------------------------------------------------------
export function setWidgetDim(widget, dimmed) {
  if (!widget) return;
  if (!widget._mmDimWrap && typeof widget.drawWidget === "function") {
    const orig = widget.drawWidget.bind(widget);
    widget.drawWidget = function (ctx, opts) {
      if (!this._mmDimmed) return orig(ctx, opts);
      const a = ctx.globalAlpha;
      ctx.globalAlpha = a * 0.35;
      try { return orig(ctx, opts); } finally { ctx.globalAlpha = a; }
    };
    widget._mmDimWrap = true;
  }
  widget._mmDimmed = Boolean(dimmed);
}

// ---------------------------------------------------------------------------
// Node sizing: grow the node to fit its widgets (never shrink the width; with
// shrinkHeight the height may also shrink back to the minimum).
// ---------------------------------------------------------------------------
export function fitNodeSize(node, { shrinkHeight = false } = {}) {
  const min = node.computeSize?.();
  if (!min || !node.size) return;
  const w = Math.max(node.size[0], min[0]);
  const h = shrinkHeight ? min[1] : Math.max(node.size[1], min[1]);
  node.setSize?.([w, h]);
  node.arrange?.();
  node.setDirtyCanvas?.(true, true);
}
