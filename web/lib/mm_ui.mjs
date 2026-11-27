// MM Nodepack – shared frontend UI helpers, imported by all MM web scripts
// (mm_nodepack.js, mm_free_gpu.js, lib/comparer.mjs).
// Theme tokens, toasts, debounce, /view URL building, widget hide/dim, node sizing.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ---------------------------------------------------------------------------
// Theme: LiteGraph values with consistent dark fallbacks (used by all MM scripts).
// ---------------------------------------------------------------------------
const LG = (typeof window !== "undefined" && window.LiteGraph) || {};
const T = (LG.CONFIG && typeof LG.CONFIG === "object") ? LG.CONFIG : {};

// Matches ComfyUI core's built-in $$canvas-image-preview minHeight (220 px) so
// Load Image MM and Preview & Save MM enforce the exact same minimum image size
// as core LoadImage / PreviewImage across all aspect ratios.
export const MIN_PREVIEW_HEIGHT = 220;

export const MM_THEME = {
  accent: "#4c9aff",
  // WIDGET_BGCOLOR / WIDGET_OUTLINE_COLOR / WIDGET_TEXT_COLOR / NODE_BGCOLOR /
  // NODE_BGCOLOR_2 live on window.LiteGraph directly, not only on LG.CONFIG
  bg: T.NODE_BGCOLOR || LG.NODE_BGCOLOR || "rgb(34, 34, 34)",
  bgHover: T.NODE_BGCOLOR_2 || LG.NODE_BGCOLOR_2 || "rgb(52, 52, 52)",
  widgetBg: T.WIDGET_BGCOLOR || LG.WIDGET_BGCOLOR || "rgb(34, 34, 34)",
  outline: T.WIDGET_OUTLINE_COLOR || LG.WIDGET_OUTLINE_COLOR || "#666",
  text: T.WIDGET_TEXT_COLOR || LG.WIDGET_TEXT_COLOR || "#DDD",
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
// Toast: single helper for all MM scripts (current object API -> older
// callable API -> console fallback).
// ---------------------------------------------------------------------------
export function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) {
    t.add({ severity, summary, detail, life: severity === "error" ? 6000 : 3000 });
  } else if (typeof t === "function") {
    t({ severity, summary, detail });
  } else {
    console[severity === "error" ? "error" : "log"](`[MM] ${summary}: ${detail}`);
  }
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
// Widget visibility (by widget object): hide with w.type = "hidden" (new
// LiteGraph derives it from the type: the row is skipped in draw/layout), plus
// computeSize/y for older frontends. A widget whose matching input socket is
// linked is kept visible. Works for schema widgets AND runtime-created buttons,
// so mode-dependent buttons can be hidden just like schema widgets.
// ---------------------------------------------------------------------------
export function setWidgetObjVisible(node, widget, visible) {
  const w = widget;
  if (!node || !w) return;
  // wired input: the frontend manages this widget (e.g. image combo next to its link)
  const sock = (node.inputs || []).find((i) => i.name === w.name);
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
    // Restore type BEFORE setting hidden=false: in ComfyUI frontend v1.58+,
    // `set hidden(false)` skips updating `_rawOptions.hidden` while type is
    // "hidden", and `set type(orig)` then re-applies stale `_rawOptions.hidden`.
    if (w.type === "hidden") w.type = w._mmType;
    w.hidden = false;
    if (w._mmHasOwnComputeSize) w.computeSize = w._mmOrigComputeSize;
    else if ("computeSize" in w) delete w.computeSize;
    if (typeof w._mmY === "number" && w._mmY >= 0) w.y = w._mmY;
    // _mmY was undefined/negative at creation: never leave y stuck at -100000
    else if (typeof w.y === "number" && w.y < 0) w.y = 0;
  } else {
    // Set type="hidden" BEFORE hidden=true so `_rawOptions.hidden` is not latched true
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
    w.y = -100000;
  }
}

// By name: finds the widget and delegates to setWidgetObjVisible (zero duplicated logic).
export function setWidgetVisible(node, name, visible) {
  const w = node?.widgets?.find((x) => x.name === name);
  if (w) setWidgetObjVisible(node, w, visible);
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
  if (!node) return;
  // Widget layout needs the owning graph. During nodeCreated the node is not
  // attached yet: queue one deferred retry (latest shrinkHeight wins).
  if (!node.graph) {
    if (node._mmFitQueued) {
      node._mmFitShrink = shrinkHeight;
      return;
    }
    node._mmFitQueued = true;
    node._mmFitShrink = shrinkHeight;
    requestAnimationFrame(() => {
      node._mmFitQueued = false;
      if (node.graph) fitNodeSize(node, { shrinkHeight: node._mmFitShrink });
    });
    return;
  }
  const min = node.computeSize?.();
  if (!min || !node.size) return;
  const w = Math.max(node.size[0], min[0]);
  const h = shrinkHeight ? min[1] : Math.max(node.size[1], min[1]);
  node.setSize?.([w, h]);
  node.arrange?.();
  node.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------------------
// MM Node titlebar icon: 15x16 pixel-perfect red film-strip mark (assets/logo.svg,
// approved design) drawn right-aligned in every MM node's title bar
// (inline data URI, zero HTTP requests).
// ---------------------------------------------------------------------------
const MM_LOGO_SVG =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 15 16" width="15" height="16" shape-rendering="crispEdges">' +
    '<g fill="#FF0000">' +
    '<path d="M0 0h15v6h-1V2H8v4H7V2H1v4H0Z"/>' +
    '<path transform="translate(0 16) scale(1 -1)" d="M0 0h15v6h-1V2H8v4H7V2H1v4H0Z"/>' +
    "</g></svg>"
  );
let _mmLogoImg = null;
function getMMLogo() {
  if (!_mmLogoImg && typeof Image !== "undefined") {
    _mmLogoImg = new Image();
    _mmLogoImg.onload = () => app.canvas?.setDirty(true, true);
    _mmLogoImg.src = MM_LOGO_SVG;
  }
  return _mmLogoImg;
}

export function attachMMLogo(nodeType) {
  if (!nodeType?.prototype || nodeType.prototype._mmLogoAttached) return;
  nodeType.prototype._mmLogoAttached = true;
  const orig = nodeType.prototype.onDrawForeground;
  nodeType.prototype.onDrawForeground = function (ctx) {
    const r = orig?.apply(this, arguments);
    const img = getMMLogo();
    if (!img?.complete || !img.naturalWidth || this.flags?.collapsed) return r;
    const titleH = (typeof window !== "undefined" && window.LiteGraph?.NODE_TITLE_HEIGHT) || 30;
    const { naturalWidth, naturalHeight } = img;
    const x = (this.size?.[0] ?? 200) - naturalWidth - 8;
    const y = -Math.round((titleH + naturalHeight) / 2);
    const prevSmooth = ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, x, y);
    ctx.imageSmoothingEnabled = prevSmooth;
    return r;
  };
}
