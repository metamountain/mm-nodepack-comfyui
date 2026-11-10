// Compare slider for Preview Image MM - loaded on demand (dynamic import) by mm_nodepack.js
// the first time a Preview Image MM node is created. Slider logic adapted from
// rgthree-comfy image_comparer.js (MIT). .mjs so ComfyUI does not load it as an extension.
// Modes (button under the image, node property mm_compare): slide (rgthree "Slide"),
// hold (rgthree "Click": original while the mouse button is held), off.
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const MODES = ["slide", "hold", "off"];
const MODE_LABEL = { slide: "compare slide", hold: "compare hold", off: "compare off" };

export function compareMode(node) {
  const m = node.properties?.mm_compare;
  if (m === false) return "off";
  return MODES.includes(m) ? m : "slide";
}

const LABEL = "rgba(170,170,170,.85)", LABEL_HI = "rgba(225,225,225,1)", DIM = "rgba(150,150,150,.6)";

function loadImage(entry) {
  const img = new Image();
  const q = new URLSearchParams({ filename: entry.filename, subfolder: entry.subfolder || "", type: entry.type || "temp" });
  img.src = api.apiURL("/view?" + q + (app.getPreviewFormatParam?.() ?? "") + (app.getRandParam?.() ?? ""));
  return img;
}
// ---------------------------------------------------------------------------
// Compare slider widget (slide mode: original full, result up to the pointer x).
// Ported from rgthree-comfy image_comparer.js (MIT): drawImage with cropX,
// variant bar, pointer handling.
// ---------------------------------------------------------------------------
export class ComparerWidget {
  constructor(node) {
    this.name = "comparer";
    this.type = "custom";
    this.serialize = false;
    this.options = { serialize: false };   // graphToPrompt checks options.serialize
    this.value = { a: [], b: [] }; // a = original entries, b = result entries
    this.selected = 0;             // selected result variant (index into b)
    this.imgA = null;              // HTMLImageElement
    this.imgB = null;
    this.node = node;
    this.barBounds = [];           // [{x, y, w, i}] set during draw
    this.draw = (ctx, node, width, y) => this._draw(ctx, node, width, y);
  }

  // flexible height: the frontend gives this widget the free node space and lays out
  // target/saved/buttons below it (LGraphNode.arrange -> computedHeight, like the core image preview)
  // clicks over a widget go to the widget (LiteGraph legacy `mouse`), not node.onMouseDown
  mouse(e, pos) {
    const t = e?.type;
    if (t !== "pointerdown" && t !== "mousedown") return false;
    if (this.hit(pos)) return true;
    // hold mode: show the original while the button is held over the image (rgthree "Click")
    const ib = this.imageBounds;
    if (compareMode(this.node) === "hold" && ib && pos[1] >= ib.top && pos[1] <= ib.bottom) {
      holdDown(this.node);
      return true;
    }
    return false;
  }

  // click on the variant bar (C1 C2 ...) or on the compare toggle under the image
  hit(pos) {
    for (const b of this.barBounds) {
      if (pos[0] >= b.x && pos[0] <= b.x + b.w && pos[1] >= b.y && pos[1] <= b.y + 16) {
        this.selectVariant(b.i);
        return true;
      }
    }
    const tb = this.toggleBounds;
    if (tb && pos[0] >= tb.x && pos[0] <= tb.x + tb.w && pos[1] >= tb.y && pos[1] <= tb.y + tb.h) {
      const p = (this.node.properties ??= {});
      p.mm_compare = MODES[(MODES.indexOf(compareMode(this.node)) + 1) % MODES.length];   // display only
      this.node.setDirtyCanvas?.(true, true);
      return true;
    }
    return false;
  }

  computeLayoutSize() {
    return { minHeight: this.value.b.length ? 160 : 40, minWidth: 1 };
  }

  setData(aEntries, bEntries) {
    this.value = { a: aEntries || [], b: bEntries || [] };
    this.selected = 0;
    this.imgA = this.value.a.length ? loadImage(this.value.a[0]) : null;
    this.imgB = this.value.b.length ? loadImage(this.value.b[this.selected]) : null;
    const fit = () => {
      // grow the node once so the image fits at the current width (never shrink what the user set)
      const img = this.imgB;
      const free = this.computedHeight ?? 0;
      const bars = (this.value.b.length > 1 ? 16 : 0) + 16 + this._reserve() + 6;
      const want = Math.min(900, Math.round((this.node.size[0] - 20) * img.naturalHeight / img.naturalWidth) + bars);
      if (want > free) this.node.setSize([this.node.size[0], this.node.size[1] + (want - free)]);
      this.node.setDirtyCanvas?.(true, true);
    };
    if (this.imgB) this.imgB.onload = fit;
    if (this.imgA) this.imgA.onload = () => this.node.setDirtyCanvas?.(true, true);
    this.node.setDirtyCanvas?.(true, true);
  }

  // height of the widgets that sit below the comparer (target/saved/buttons)
  _belowHeight() {
    const ws = this.node.widgets || [];
    const idx = ws.indexOf(this);
    if (idx < 0) return 0;
    let h = 0;
    for (let i = idx + 1; i < ws.length; i++) {
      const w = ws[i];
      if (w?.computeSize) {
        const s = w.computeSize(this.node.size[0]);
        h += Array.isArray(s) ? (s[1] || 0) : 0;
      }
    }
    return h;
  }

  // space under the image: compare button (only with an original) + size line
  _reserve() {
    return this.imgA?.naturalWidth ? 44 : 14;
  }

  selectVariant(i) {
    if (i === this.selected || !this.value.b[i]) return;
    this.selected = i;
    this.imgB = loadImage(this.value.b[i]);
    this.node.setDirtyCanvas?.(true, true);
    this.node._rtRefreshButtons?.();
  }

  _draw(ctx, node, width, y) {
    this.barBounds = [];
    this.toggleBounds = null;
    this._bottom = this.computedHeight ? y + this.computedHeight - 4 : undefined;
    const b = this.value.b;
    const W = node.size[0];
    ctx.save();
    ctx.textBaseline = "top";
    ctx.font = "10px Arial";
    if (b.length > 1) {
      // variant bar "C1 C2 C3 ..." (clickable, like rgthree image_comparer.js:265-275)
      const parts = b.map((_, i) => { const label = `C${i + 1}`; return { label, w: ctx.measureText(label).width + 8 }; });
      let x = Math.max(4, (W - parts.reduce((acc, p) => acc + p.w + 4, -4)) / 2);
      ctx.textAlign = "left";
      parts.forEach((p, i) => {
        ctx.fillStyle = i === this.selected ? LABEL_HI : "rgba(150,150,150,.6)";
        ctx.fillText(p.label, x + 4, y + 3);
        this.barBounds.push({ x, y, w: p.w, i });
        x += p.w + 4;
      });
      y += 16;
    }
    const mode = compareMode(node);   // button under the image
    const compare = mode !== "off";
    const haveA = !!this.imgA?.naturalWidth;
    const hasA = compare && haveA;
    const held = mode === "hold" && hasA && !!node._mmDown;
    const over = mode === "slide" && !!node.isPointerOver && hasA;
    const cropX = over ? node.pointerOverPos?.[0] : null;
    const creation = b.length > 1 ? `CREATION ${this.selected + 1}` : "CREATION";
    // label line ABOVE the image (outside, subtle): left = original, right = creation
    if (b.length) {
      const leftShare = cropX != null ? cropX / W : 0;
      ctx.textAlign = "left";
      if (hasA) {
        ctx.fillStyle = held || (over && leftShare > 0.5) ? LABEL_HI : LABEL;
        const sid = node._rtSrcId || node._rtSrcIdLive;
        ctx.fillText(sid ? `◀ ORIGINAL #${sid}` : "◀ ORIGINAL", 8, y + 3);
      } else if (compare && (node._rtNote || node._rtSrcNote)) {
        ctx.fillStyle = LABEL;
        ctx.fillText(node._rtNote || node._rtSrcNote, 8, y + 3);
      }
      ctx.textAlign = "right";
      ctx.fillStyle = !held && (!over || leftShare <= 0.5) ? LABEL_HI : LABEL;
      ctx.fillText(creation + " ▶", W - 8, y + 3);
      y += 16;
    }
    ctx.restore();
    // creation full, original clipped up to the pointer -> original left of the line, creation right
    const reserve = this._reserve();
    const r = this._drawImage(ctx, held ? this.imgA : this.imgB, y, null, reserve);
    if (over) this._drawImage(ctx, this.imgA, y, cropX, reserve, true);
    this.imageBounds = r;
    if (r) {
      // line directly under the image: compare toggle (only with an original) + sizes
      const sa = haveA ? `Original ${this.imgA.naturalWidth}×${this.imgA.naturalHeight} · ` : "";
      const sb = this.imgB?.naturalWidth ? `Creation ${this.imgB.naturalWidth}×${this.imgB.naturalHeight}` : "";
      ctx.save();
      ctx.font = "10px Arial";
      ctx.textBaseline = "top";
      if (haveA) {
        // big centered toggle button directly under the image, sizes below it
        const label = MODE_LABEL[mode];
        ctx.font = "bold 13px Arial";
        const bw = Math.min(W - 16, Math.max(150, ctx.measureText(label).width + 32)), bh = 24;
        const bx = (W - bw) / 2, by = r.bottom + 4;
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(bx, by, bw, bh, 12) : ctx.rect(bx, by, bw, bh);
        ctx.fillStyle = compare ? "rgba(70,120,190,.9)" : "rgba(60,60,60,.9)";
        ctx.fill();
        ctx.strokeStyle = compare ? "rgba(140,180,240,.9)" : "rgba(120,120,120,.8)";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = compare ? "#fff" : "rgba(190,190,190,.9)";
        ctx.fillText(label, W / 2, by + bh / 2 + 1);
        this.toggleBounds = { x: bx, y: by, w: bw, h: bh };
        ctx.font = "10px Arial";
        ctx.textBaseline = "top";
        ctx.fillStyle = LABEL;
        ctx.fillText(sa + sb, W / 2, by + bh + 3);
      } else {
        ctx.textAlign = "center";
        ctx.fillStyle = LABEL;
        ctx.fillText(sb, W / 2, r.bottom + 2);
      }
      ctx.restore();
    }
  }

  // rgthree image_comparer.js:297-344 (drawImage with cropX)
  _drawImage(ctx, image, y, cropX, reserve = 0, handle = false) {
    if (!image?.naturalWidth || !image?.naturalHeight) return null;
    const [nodeWidth, nodeHeight] = this.node.size;
    const imageAspect = image.naturalWidth / image.naturalHeight;
    const height = (this._bottom ?? nodeHeight - this._belowHeight()) - y - reserve;
    if (height <= 0) return null;
    const widgetAspect = nodeWidth / height;
    let targetWidth, targetHeight, offsetX = 0;
    if (imageAspect > widgetAspect) {
      targetWidth = nodeWidth;
      targetHeight = nodeWidth / imageAspect;
    } else {
      targetHeight = height;
      targetWidth = height * imageAspect;
      offsetX = (nodeWidth - targetWidth) / 2;
    }
    const widthMultiplier = image.naturalWidth / targetWidth;
    const sourceWidth = cropX != null ? (cropX - offsetX) * widthMultiplier : image.naturalWidth;
    const destX = (nodeWidth - targetWidth) / 2;
    const destY = y + (height - targetHeight) / 2;
    const destWidth = cropX != null ? cropX - offsetX : targetWidth;
    const destHeight = targetHeight;
    ctx.save();
    ctx.beginPath();
    const gco = ctx.globalCompositeOperation;
    if (cropX != null) {
      ctx.rect(destX, destY, destWidth, destHeight);
      ctx.clip();
    }
    ctx.drawImage(image, 0, 0, sourceWidth, image.naturalHeight, destX, destY, destWidth, destHeight);
    if (cropX != null && cropX >= destX && cropX <= destX + destWidth) {
      ctx.beginPath();
      ctx.moveTo(cropX, destY);
      ctx.lineTo(cropX, destY + destHeight);
      ctx.globalCompositeOperation = "difference";
      ctx.strokeStyle = "rgba(255,255,255,1)";
      ctx.lineWidth = 1;
      ctx.stroke();
      if (handle) {
        ctx.beginPath();
        ctx.arc(cropX, destY + destHeight / 2, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,1)";
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = gco;
    ctx.restore();
    return { top: destY, bottom: destY + destHeight };
  }
}

// rgthree setIsPointerDown: stays "down" until the canvas reports the button released
function holdDown(node) {
  node._mmDown = true;
  node.setDirtyCanvas?.(true, false);
  const tick = () => {
    if (app.canvas?.pointer_is_down || app.canvas?.pointer?.isDown) return requestAnimationFrame(tick);
    node._mmDown = false;
    node.setDirtyCanvas?.(true, false);
  };
  requestAnimationFrame(tick);
}

async function copyImage(img) {
  try {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
  } catch (e) {
    console.warn("[MM Nodepack] copy image failed (clipboard needs localhost/https)", e);
  }
}

export function wirePointer(node, widget) {
  // right-click entries for the shown images. Not via node.imgs: that makes the frontend add its
  // own $$canvas-image-preview widget below ours (unsafeUpdatePreviews -> showCanvasImagePreview).
  const origMenu = node.getExtraMenuOptions;
  node.getExtraMenuOptions = function (canvas, options) {
    const r = origMenu?.apply(this, arguments);
    const add = [];
    if (widget.imgB?.naturalWidth) {
      add.push({ content: "Open creation", callback: () => window.open(widget.imgB.src, "_blank") });
      add.push({ content: "Copy creation", callback: () => copyImage(widget.imgB) });
    }
    if (widget.imgA?.naturalWidth) add.push({ content: "Open original", callback: () => window.open(widget.imgA.src, "_blank") });
    if (add.length) options.unshift(...add, null);
    return r;
  };
  // pointer tracking for the slide compare (rgthree image_comparer.js:77-115)
  const origEnter = node.onMouseEnter;
  const origLeave = node.onMouseLeave;
  const origMove = node.onMouseMove;
  const origDown = node.onMouseDown;
  node.onMouseEnter = (e) => {
    origEnter?.call(node, e);
    node.isPointerOver = true;
    node.setDirtyCanvas?.(true, false);
  };
  node.onMouseLeave = (e) => {
    origLeave?.call(node, e);
    node.isPointerOver = false;
    node._mmDown = false;
    node.setDirtyCanvas?.(true, false);
  };
  node.onMouseMove = (e, pos) => {
    origMove?.call(node, e, pos);
    node.pointerOverPos = [...pos];
    node.setDirtyCanvas?.(true, false);
  };
  node.onMouseDown = (e, pos) => {
    // variant bar / compare toggle hit test (bounds are set during draw)
    if (widget.hit(pos)) return false; // consumed
    return origDown?.call(node, e, pos);
  };
}
