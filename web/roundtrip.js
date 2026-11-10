// Image Roundtrip (metamountain/image): target preview, per-variant save buttons,
// and a compare slider (original vs. result).
// slider logic adapted from rgthree-comfy image_comparer.js (MIT)
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SAVE_NODES = new Set(["RoundtripPreviewSave"]);
const LOADER = "RoundtripLoadImage";
const LABEL = "rgba(170,170,170,.85)", LABEL_HI = "rgba(225,225,225,1)";

function viewUrl(entry) {
  return api.apiURL("/view?" + new URLSearchParams({ filename: entry.filename, subfolder: entry.subfolder || "", type: entry.type || "temp" }));
}

function loadImage(entry) {
  const img = new Image();
  img.src = viewUrl(entry);
  return img;
}

function debounce(fn, ms) {
  let t = null;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

// ---------------------------------------------------------------------------
// Compare slider widget (slide mode: original full, result up to the pointer x).
// Ported from rgthree-comfy image_comparer.js (MIT): drawImage with cropX,
// variant bar, pointer handling.
// ---------------------------------------------------------------------------
class ComparerWidget {
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
    for (const b of this.barBounds) {
      if (pos[0] >= b.x && pos[0] <= b.x + b.w && pos[1] >= b.y && pos[1] <= b.y + 16) {
        this.selectVariant(b.i);
        return true;
      }
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
      const bars = (this.value.b.length > 1 ? 16 : 0) + 16 + 14 + 6;
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

  selectVariant(i) {
    if (i === this.selected || !this.value.b[i]) return;
    this.selected = i;
    this.imgB = loadImage(this.value.b[i]);
    this.node.setDirtyCanvas?.(true, true);
    this.node._rtRefreshButtons?.();
  }

  _draw(ctx, node, width, y) {
    this.barBounds = [];
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
    const compare = node.widgets?.find((w) => w.name === "compare")?.value !== false;
    const hasA = compare && !!this.imgA?.naturalWidth;
    const over = !!node.isPointerOver && hasA;
    const cropX = over ? node.pointerOverPos?.[0] : null;
    const creation = b.length > 1 ? `CREATION ${this.selected + 1}` : "CREATION";
    // label line ABOVE the image (outside, subtle): left = original, right = creation
    if (b.length) {
      const leftShare = cropX != null ? cropX / W : 0;
      ctx.textAlign = "left";
      if (hasA) {
        ctx.fillStyle = over && leftShare > 0.5 ? LABEL_HI : LABEL;
        const sid = node._rtSrcId || node._rtSrcIdLive;
        ctx.fillText(sid ? `◀ ORIGINAL #${sid}` : "◀ ORIGINAL", 8, y + 3);
      } else if (compare && (node._rtNote || node._rtSrcNote)) {
        ctx.fillStyle = LABEL;
        ctx.fillText(node._rtNote || node._rtSrcNote, 8, y + 3);
      }
      ctx.textAlign = "right";
      ctx.fillStyle = !over || leftShare <= 0.5 ? LABEL_HI : LABEL;
      ctx.fillText(creation + " ▶", W - 8, y + 3);
      y += 16;
    }
    ctx.restore();
    // creation full, original clipped up to the pointer -> original left of the line, creation right
    const r = this._drawImage(ctx, this.imgB, y, null, 14);
    if (over) this._drawImage(ctx, this.imgA, y, cropX, 14, true);
    if (r) {
      const sa = hasA ? `Original ${this.imgA.naturalWidth}×${this.imgA.naturalHeight} · ` : "";
      const sb = this.imgB?.naturalWidth ? `Creation ${this.imgB.naturalWidth}×${this.imgB.naturalHeight}` : "";
      ctx.save();
      ctx.font = "10px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = LABEL;
      ctx.fillText(sa + sb, W / 2, r.bottom + 2);
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
      ctx.strokeStyle = "rgba(255,255,255,.55)";
      ctx.lineWidth = 1;
      ctx.stroke();
      if (handle) {
        ctx.beginPath();
        ctx.arc(cropX, destY + destHeight / 2, 3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,.85)";
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = gco;
    ctx.restore();
    return { top: destY, bottom: destY + destHeight };
  }
}

// ---------------------------------------------------------------------------
// Shared node wiring: target preview widget + source lookup + pointer handling.
// ---------------------------------------------------------------------------
// Read-only text widget (custom: ComfyUI "string" widgets are editable inputs).
function makeTextWidget(node, name) {
  const w = {
    name,
    type: "custom",
    serialize: false,
    options: { serialize: false },
    value: "",
    callback: () => {},
    computeSize: (width) => [width, 20],
    draw: function (ctx, node, width, y) {
      ctx.save();
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.font = "12px monospace";
      ctx.fillStyle = "rgba(200,200,200,0.9)";
      const text = String(this.value || "");
      const maxW = width - 24;
      let t = text;
      while (t.length > 2 && ctx.measureText(t).width > maxW) t = t.slice(1);
      if (t !== text) t = "…" + t;
      ctx.fillText(t, 10, y + 4);
      this.tooltip = text;
      ctx.restore();
    },
  };
  node.addCustomWidget(w);
  return w;
}

// Which Load Image MM feeds this preview (same rules as find_loader in roundtrip_nodes.py):
// sync link > source "#id" > auto (the one loader upstream of images, else the only loader).
function findLoader(node) {
  const g = node.graph || app.graph;
  const linkOf = (id) => (id == null ? null : (g.links?.get?.(id) ?? g.links?.[id]));
  const syncIn = (node.inputs || []).find((i) => i.name === "sync");
  if (syncIn?.link != null) {
    const n = g.getNodeById(linkOf(syncIn.link)?.origin_id);
    return n?.comfyClass === LOADER ? { node: n, note: "" } : { node: null, note: "" };
  }
  const loaders = (g._nodes || []).filter((n) => n.comfyClass === LOADER && n.mode !== 2 && n.mode !== 4);
  const sel = String(node.widgets?.find((w) => w.name === "source")?.value ?? "auto").trim().replace(/^#/, "").trim();
  if (sel && sel.toLowerCase() !== "auto") {
    const n = loaders.find((x) => String(x.id) === sel);
    return n ? { node: n, note: "" } : { node: null, note: `source #${sel}: no Load Image MM` };
  }
  const found = new Set();
  const seen = new Set();
  const imgIn = (node.inputs || []).find((i) => i.name === "images");
  const stack = imgIn?.link != null ? [imgIn.link] : [];
  while (stack.length) {
    const l = linkOf(stack.pop());
    if (!l || seen.has(l.id)) continue;
    seen.add(l.id);
    const src = g.getNodeById(l.origin_id);
    if (!src) continue;
    if (src.comfyClass === LOADER) { found.add(src); continue; }
    for (const i of src.inputs || []) if (i.link != null) stack.push(i.link);
  }
  if (found.size === 1) return { node: [...found][0], note: "" };
  if (!found.size && loaders.length === 1) return { node: loaders[0], note: "" };
  const many = found.size > 1 ? [...found] : loaders;
  if (many.length > 1) return { node: null, note: `several loaders (${many.map((n) => "#" + n.id).join(" ")}) – set source` };
  return { node: null, note: "" };
}

function sourceFromNode(node) {
  const { node: n, note } = findLoader(node);
  node._rtSrcNote = note;
  node._rtSrcIdLive = n ? String(n.id) : "";
  if (!n) return null;
  const w = (name) => n.widgets?.find((x) => x.name === name)?.value ?? "";
  return { path: String(w("path") || ""), image: String(w("image") || ""), id: String(n.id) };
}

function addTargetWidget(node) {
  const setTarget = (t) => {
    node._rtTarget = t;
    if (node._rtButtons?.[0]) node._rtButtons[0].tooltip = t;
    node.setDirtyCanvas?.(true, true);  // label line shows the live source #id / note
  };
  node._rtRefreshTarget = debounce(() => {
    const s = sourceFromNode(node);
    const body = {
      path: s?.path || "",
      image: s?.image || "",
      suffix: node.widgets?.find((x) => x.name === "suffix")?.value ?? "_#",
      save_as: node.widgets?.find((x) => x.name === "save_as")?.value ?? "",
      format: node.widgets?.find((x) => x.name === "format")?.value ?? "same as source",
    };
    if (!body.path && !body.image) {
      // no source: the saver falls back to the output dir, stem "preview"
      setTarget("output/preview_# (no Load Image MM found)");
      return;
    }
    api.fetchApi("/roundtrip/target", { method: "POST", body: JSON.stringify(body) })
      .then((r) => r.json())
      .then((j) => {
        setTarget(j.ok ? j.target : (j.error || "target?"));
      })
      .catch((e) => {
        setTarget(String(e));
      });
  }, 300);
}

function wirePointer(node, widget) {
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
    node.setDirtyCanvas?.(true, false);
  };
  node.onMouseMove = (e, pos) => {
    origMove?.call(node, e, pos);
    node.pointerOverPos = [...pos];
    node.setDirtyCanvas?.(true, false);
  };
  node.onMouseDown = (e, pos) => {
    // variant bar hit test (the bar bounds are set during draw)
    for (const b of widget.barBounds) {
      if (pos[0] >= b.x && pos[0] <= b.x + b.w && pos[1] >= b.y && pos[1] <= b.y + 16) {
        widget.selectVariant(b.i);
        return false; // consumed
      }
    }
    return origDown?.call(node, e, pos);
  };
}

// ---------------------------------------------------------------------------
// Save buttons (Preview & Save): apply to the SELECTED variant (NACHTRAG C).
// ---------------------------------------------------------------------------
function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary, detail, life: severity === "error" ? 6000 : 3000 });
  else console[severity === "error" ? "error" : "log"](`[Image Roundtrip] ${summary}: ${detail}`);
}

const PICK_TYPES = [
  { description: "PNG", accept: { "image/png": [".png"] } },
  { description: "JPEG", accept: { "image/jpeg": [".jpg", ".jpeg"] } },
  { description: "WebP", accept: { "image/webp": [".webp"] } },
];

// preview file -> blob in the format of the chosen extension (PNG keeps the embedded workflow)
async function previewBlob(entry, name, quality) {
  const q = new URLSearchParams({ filename: entry.filename, subfolder: entry.subfolder || "", type: entry.type || "temp" });
  const src = await (await api.fetchApi(`/view?${q}`)).blob();
  const ext = (name.match(/\.([^.]+)$/)?.[1] || "png").toLowerCase();
  const mime = ext === "jpg" || ext === "jpeg" ? "image/jpeg" : ext === "webp" ? "image/webp" : "image/png";
  if (mime === "image/png" && src.type === "image/png") return src;
  const bmp = await createImageBitmap(src);
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  c.getContext("2d").drawImage(bmp, 0, 0);
  return await new Promise((res) => c.toBlob(res, mime, Math.max(0.01, Math.min(1, quality / 100))));
}

function addSaveButtons(node, entry, sourcePath, variant) {
  node._rtRemoveButtons?.();
  const mk = (label) => {
    const b = node.addWidget("button", label, null, () => {});
    b.serialize = false;
    b.options = { ...(b.options || {}), serialize: false };
    return b;
  };
  const wv = (name, d) => node.widgets?.find((x) => x.name === name)?.value ?? d;
  // server side: next free name next to the source (never overwrites)
  const serverSave = async (saveAs) => {
    const body = {
      filename: entry.filename, subfolder: entry.subfolder || "", type: entry.type || "temp",
      source: sourcePath, suffix: wv("suffix", "_#"), save_as: saveAs,
      format: wv("format", "same as source"), quality: wv("quality", 95),
    };
    let j;
    try {
      j = await (await api.fetchApi("/roundtrip/save", { method: "POST", body: JSON.stringify(body) })).json();
    } catch (e) {
      j = { ok: false, error: String(e) };
    }
    if (j.ok) {
      toast("success", "Saved", j.path);
      node._rtRefreshTarget?.(); // next number is now taken
    } else toast("error", "Save failed", j.error || "unknown error");
    return j;
  };
  const stem = sourcePath ? String(sourcePath).replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "") : "image";
  const btn1 = mk("💾 save");
  btn1.tooltip = node._rtTarget || "next free _# name next to the source";
  btn1.callback = () => serverSave("");
  const btn2 = mk("💾 save as…");
  btn2.tooltip = "choose folder and name";
  btn2.callback = async () => {
    const suggested = (node._rtTarget ? String(node._rtTarget).replace(/\\/g, "/").split("/").pop() : "") || `${stem}_edit.png`;
    if (typeof window.showSaveFilePicker === "function") {
      // real file dialog (Chrome/Edge, secure context: localhost or https)
      let handle;
      try {
        handle = await window.showSaveFilePicker({ id: "image-roundtrip", startIn: "pictures", suggestedName: suggested, types: PICK_TYPES });
      } catch (e) {
        if (e?.name !== "AbortError") toast("error", "Save as", String(e));
        return;
      }
      try {
        const blob = await previewBlob(entry, handle.name, wv("quality", 95));
        const wr = await handle.createWritable();
        await wr.write(blob);
        await wr.close();
        toast("success", "Saved", handle.name);
      } catch (e) {
        toast("error", "Save failed", String(e));
      }
      return;
    }
    // no file dialog (plain http on a LAN address): ask for a name, saved next to the source
    let name = "";
    const prompt = app.extensionManager?.dialog?.prompt;
    if (typeof prompt === "function") name = await prompt({ title: "Save as (next to the source)", message: "File name", defaultValue: `${stem}_edit` });
    else name = window.prompt("Save as (next to the source):", `${stem}_edit`);
    if (name != null && String(name).trim() !== "") await serverSave(String(name).trim());
  };
  const btn3 = mk("↩ send to loader");
  btn3.tooltip = "save as _# and load it in the Load Image MM (next run works on this creation)";
  btn3.callback = async () => {
    const loader = findLoader(node).node;
    if (!loader) {
      toast("error", "Send to loader", node._rtSrcNote || "no Load Image MM found – set source");
      return;
    }
    const j = await serverSave("");
    if (!j?.ok) return;
    const pw = loader.widgets?.find((w) => w.name === "path");
    if (!pw) return;
    pw.value = j.path;
    pw.callback?.(j.path);
    loader.setDirtyCanvas?.(true, true);
    toast("info", `Loader #${loader.id}`, j.path);
  };
  node._rtButtons = [btn1, btn2, btn3];
  node._rtRemoveButtons = () => {
    for (const b of node._rtButtons || []) node.widgets?.splice(node.widgets.indexOf(b), 1);
    node._rtButtons = null;
    node.computeSize?.();
    node.setDirtyCanvas?.(true, true);
  };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "ImageRoundtrip",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!SAVE_NODES.has(nodeData?.name)) return;

    const origExecuted = nodeType.prototype.onExecuted;
    const origConnections = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);
      const a = Array.isArray(message?.a_images) ? message.a_images : [];
      const b = Array.isArray(message?.b_images) ? message.b_images : (Array.isArray(message?.images) ? message.images : []);
      const source = Array.isArray(message?.source) ? message.source[0] || "" : "";
      this._rtSrcId = Array.isArray(message?.source_id) ? message.source_id[0] || "" : "";
      this._rtNote = Array.isArray(message?.note) ? message.note[0] || "" : "";
      const cmp = this.widgets?.find((w) => w.name === "comparer");
      if (cmp) cmp.setData(a, b);
      const saved = Array.isArray(message?.saved) ? message.saved : [];
      if (saved.length) toast("success", "Auto-saved", saved.length > 1 ? `${saved[saved.length - 1]} (+${saved.length - 1})` : saved[0]);
      // buttons for the selected creation (save _# / save as…)
      const sel = cmp?.selected ?? 0;
      if (b[sel]) addSaveButtons(this, b[sel], source, sel);
      this._rtRefreshButtons = () => {
        const c = this.widgets?.find((w) => w.name === "comparer");
        if (c && b[c.selected]) addSaveButtons(this, b[c.selected], source, c.selected);
      };
      this._rtRefreshTarget?.();
      this.setDirtyCanvas?.(true, true);
    };

    nodeType.prototype.onConnectionsChange = function (slot, type, isConnect, linkIds) {
      origConnections?.apply(this, arguments);
      this._rtRefreshTarget?.();  // sync or images link changed -> source may change
      this.setDirtyCanvas?.(true, true);
    };
  },
  nodeCreated(node) {
    if (node.comfyClass === LOADER) {
      // size info: text line + subtle overlay bottom right on the node
      const info = makeTextWidget(node, "info");
      // picking/uploading an image clears the path field (path would win otherwise)
      const iw = node.widgets?.find((w) => w.name === "image");
      const pw = node.widgets?.find((w) => w.name === "path");
      if (iw && pw) {
        const cb = iw.callback;
        iw.callback = function () {
          const r = cb?.apply(this, arguments);
          if (pw.value) {
            pw.value = "";
            pw.callback?.("");
            node.setDirtyCanvas?.(true, true);
          }
          return r;
        };
        // only the source that is used is active: path set -> image picker + upload greyed out
        const upw = node.widgets?.find((w) => w.name === "upload" || w.type === "button");
        const sync = () => {
          const usePath = String(pw.value || "").trim() !== "";
          iw.disabled = usePath;
          if (upw) upw.disabled = usePath;
          node.setDirtyCanvas?.(true, true);
        };
        // the thumbnail belongs to the picker; with a path it would show the wrong image
        // (after a run the node shows the path image from the backend)
        const clearThumb = () => {
          if (String(pw.value || "").trim() === "") return;
          node.imgs = undefined;
          if (node.images) node.images = undefined;
          try { app.nodeOutputs && delete app.nodeOutputs[String(node.id)]; } catch (_) { /* ignore */ }
        };
        const pcb = pw.callback;
        pw.callback = function () {
          const r = pcb?.apply(this, arguments);
          sync();
          clearThumb();
          return r;
        };
        const origCfg = node.onConfigure;
        node.onConfigure = function () {
          const r = origCfg?.apply(this, arguments);
          sync();
          return r;
        };
        sync();
      }
      const origExec = node.onExecuted;
      node.onExecuted = function (message) {
        origExec?.apply(this, arguments);
        info.value = Array.isArray(message?.info) ? message.info[0] || "" : "";
        this.setDirtyCanvas?.(true, true);
      };
      const origFg = node.onDrawForeground;
      node.onDrawForeground = function (ctx) {
        origFg?.apply(this, arguments);
        const t = String(info.value || "").split(" · ")[0];
        if (!t || this.flags?.collapsed) return;
        ctx.save();
        ctx.font = "10px Arial";
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillStyle = "rgba(255,255,255,.55)";
        ctx.fillText(t, this.size[0] - 6, this.size[1] - 4);
        ctx.restore();
      };
      return;
    }
    if (!SAVE_NODES.has(node.comfyClass)) return;
    const cmp = new ComparerWidget(node);
    node.addCustomWidget(cmp);
    wirePointer(node, cmp);
    addTargetWidget(node);
    // target preview updates when suffix/save_as/format change
    const cw = node.widgets?.find((x) => x.name === "compare");
    if (cw) { const cb0 = cw.callback; cw.callback = (...a) => { cb0?.(...a); node.setDirtyCanvas?.(true, true); }; }
    for (const name of ["suffix", "save_as", "format", "source"]) {
      const w = node.widgets?.find((x) => x.name === name);
      if (!w) continue;
      const cb = w.callback;
      w.callback = (...a) => {
        cb?.(...a);
        node._rtRefreshTarget?.();
      };
    }
    node._rtRefreshTarget?.();
  },
});
