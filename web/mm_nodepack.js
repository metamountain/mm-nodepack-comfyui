// MM Nodepack (category "MM Nodepack"): target preview, per-variant save buttons,
// loader helpers. The compare slider lives in lib/comparer.mjs and is only loaded
// when a Preview & Save MM node exists (dynamic import).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  toast, debounce, viewBlob, loadImage, setWidgetVisible, setWidgetObjVisible, setWidgetDim, fitNodeSize,
} from "./lib/mm_ui.mjs";
import { createRefreshChannel } from "./lib/mm_race.mjs";

const SAVE_NODES = new Set(["MMPreviewSave"]);
const LOADER = "MMLoadImage";

// runtime-created nodes can lack comfyClass (only .type is set): check both
const nodeClass = (n) => n?.comfyClass || n?.type;

// compare slider module, loaded once on first use (?v= busts the browser cache)
let comparerModule = null;
const loadComparer = () => (comparerModule ??= import("./lib/comparer.mjs?v=7"));

// ---------------------------------------------------------------------------
// Shared node wiring: target preview widget + source lookup + pointer handling.
// ---------------------------------------------------------------------------
// The loader's 6-field request body, shared by every /mm/image/* call.
function loaderBody(loader) {
  const w = (name) => loader?.widgets?.find((x) => x.name === name)?.value;
  return {
    mode: String(w("mode") ?? "single") || "single",
    path: String(w("path") ?? "").trim(),
    image: String(w("image") ?? "").trim(),
    folder: String(w("folder") ?? "").trim(),
    batch_mode: Boolean(w("batch_mode")),
    index: Number(w("index") ?? 1) || 1,
  };
}

// Which Load Image MM feeds this preview (same rules as find_loader in py/image_io.py):
// source "#id" > auto (the one loader upstream of images, else the only loader).
function findLoader(node) {
  const g = node.graph || app.graph;
  const linkOf = (id) => (id == null ? null : (g.links?.get?.(id) ?? g.links?.[id]));
  const loaders = (g._nodes || []).filter((n) => nodeClass(n) === LOADER && n.mode !== 2 && n.mode !== 4);
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
    if (nodeClass(src) === LOADER) { found.add(src); continue; }
    for (const i of src.inputs || []) if (i.link != null) stack.push(i.link);
  }
  if (found.size === 1) return { node: [...found][0], note: "" };
  if (!found.size && loaders.length === 1) return { node: loaders[0], note: "" };
  const many = found.size > 1 ? [...found] : loaders;
  if (many.length > 1) return { node: null, many, note: `several loaders (${many.map((n) => "#" + n.id).join(" ")}) – set source` };
  return { node: null, note: "" };
}

function loaderLabel(n) {
  const w = (name) => String(n.widgets?.find((x) => x.name === name)?.value ?? "");
  const isFolder = w("mode") === "folder";
  const raw = isFolder ? w("folder") : (w("path") || w("image"));
  const f = raw.replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop();
  return `#${n.id}  ${isFolder ? f + "/" : f}`;
}

// When the loader's file or folder selection changes,
// refresh the compare slider's "original" (imgA) on every preview node that
// resolves to this loader — so compare-hold always shows the current source.
// Race-safe channel (web/lib/mm_race.mjs): one debounce + revision per loader
// (editing A and then B within the window refreshes BOTH previews), the
// revision is re-checked after the fetch (a slower older response never
// replaces a newer original), and a cleared source shows the empty state even
// when a fetch is still in flight.
const compareRefreshChannel = createRefreshChannel({
  delay: 400,
  async fetch(loaderNode) {
    const graph = loaderNode.graph || app.graph;
    const body = loaderBody(loaderNode);
    if (body.mode === "folder" ? !body.folder : (!body.path && !body.image)) {
      return { entry: null, graph };   // cleared source: the write shows the empty state
    }
    const j = await (await api.fetchApi("/mm/image/preview", {
      method: "POST",
      body: JSON.stringify(body),
    })).json();
    if (!j?.ok || !j.view) throw new Error("no preview for the current source");
    return { entry: j.view, graph };
  },
  write(loaderNode, payload) {
    if (!payload.ok) return;  // fetch failed: keep the current original
    const { entry, graph } = payload.result;
    if (!graph || loaderNode.graph !== graph) return;  // the node moved in between
    for (const n of graph._nodes || []) {
      if (nodeClass(n) !== "MMPreviewSave") continue;
      const { node: src } = findLoader(n);
      if (src !== loaderNode) continue;
      n._rtRefreshTarget?.();  // folder/index/batch_mode changed -> live target tooltip
      const cmp = n.widgets?.find((w) => w.name === "comparer");
      if (!cmp) continue;
      cmp.value.a = entry ? [entry] : [];
      cmp.imgA = entry ? loadImage(entry) : null;
      if (cmp.imgA) cmp.imgA.onload = () => n.setDirtyCanvas?.(true, true);
      n.setDirtyCanvas?.(true, true);
    }
  },
});
const _refreshCmp = (loaderNode) => compareRefreshChannel.schedule(loaderNode);

// last pointer position (the button callback has no DOM event to anchor the menu)
const lastPtr = { clientX: 200, clientY: 200 };
window.addEventListener("pointerdown", (e) => { lastPtr.clientX = e.clientX; lastPtr.clientY = e.clientY; }, true);

// ambiguous source: ask once with a small menu, remember the choice in the 'source' widget
function chooseLoader(node, many) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (n) => { if (!done) { done = true; resolve(n); } };
    const items = many.map((n) => ({ content: loaderLabel(n), callback: () => {
      const sw = node.widgets?.find((w) => w.name === "source");
      if (sw) { sw.value = `#${n.id}`; sw.callback?.(sw.value); }
      node.setDirtyCanvas?.(true, true);
      finish(n);
    } }));
    // open after the click has finished, else its pointerup closes the menu at once
    setTimeout(() => {
      try {
        const ev = new MouseEvent("contextmenu", { clientX: lastPtr.clientX, clientY: lastPtr.clientY });
        new LiteGraph.ContextMenu(items, { title: "which Load Image MM?", event: ev, callback: () => {} });
      } catch (e) {
        toast("error", "Choose source", String(e));
        finish(null);
      }
    }, 150);
    // closed without a choice -> give up after a while
    setTimeout(() => finish(null), 30000);
  });
}

async function resolveLoader(node) {
  const r = findLoader(node);
  if (r.node) return r.node;
  if (r.many?.length > 1) return await chooseLoader(node, r.many);
  toast("error", "No source", "no Load Image MM found – add one or set source");
  return null;
}

function addTargetWidget(node) {
  const setTarget = (t) => {
    node._rtTarget = t;
    if (node._rtButtons?.[0]) node._rtButtons[0].tooltip = t;
    node.setDirtyCanvas?.(true, true);  // label line shows the live source #id / note
  };
  // race-safe channel per saver (same guarantees as the compare channel): a
  // cleared source invalidates the in-flight fetch immediately (revision),
  // and both the empty-source branch and the fetch error handler write only
  // when their revision is still current.
  const targetChannel = createRefreshChannel({
    delay: 300,
    async fetch() {
      const { node: loader, note } = findLoader(node);
      node._rtSrcNote = note;
      node._rtSrcIdLive = loader ? String(loader.id) : "";
      const body = {
        ...loaderBody(loader),
        suffix: node.widgets?.find((x) => x.name === "suffix")?.value ?? "_#",
        save_as: node.widgets?.find((x) => x.name === "save_as")?.value ?? "",
        format: node.widgets?.find((x) => x.name === "format")?.value ?? "same as source",
      };
      if (body.mode === "folder" ? !body.folder : (!body.path && !body.image)) {
        // no source: the saver falls back to the output dir, stem "preview"
        return { target: "output/preview_# (no Load Image MM found)" };
      }
      const j = await (await api.fetchApi("/mm/image/target", { method: "POST", body: JSON.stringify(body) })).json();
      return { target: j.ok ? j.target : (j.error || "target?") };
    },
    write(_key, payload) {
      if (!payload.ok) setTarget(payload.error);   // fetch error (newest revision only)
      else setTarget(payload.result.target);       // live target or the empty-source state
    },
  });
  node._rtRefreshTarget = () => targetChannel.schedule(node);
}


// ---------------------------------------------------------------------------
// Save buttons (Preview & Save): apply to the SELECTED variant (NACHTRAG C).
// ---------------------------------------------------------------------------
const PICK_TYPES = [
  { description: "PNG", accept: { "image/png": [".png"] } },
  { description: "JPEG", accept: { "image/jpeg": [".jpg", ".jpeg"] } },
  { description: "WebP", accept: { "image/webp": [".webp"] } },
];

// preview file -> blob in the format of the chosen extension (PNG keeps the embedded workflow)
async function previewBlob(entry, name, quality) {
  const src = await viewBlob(entry, { format: "", rand: "" });
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

// A button whose job an auto switch already does is dimmed and not clickable
// (disabled: the frontend skips it on click and drops its outline).
function applyAutoState(node) {
  const on = (name) => node.widgets?.find((x) => x.name === name)?.value === true;
  const [save, , send] = node._rtButtons || [];
  if (save) { save.disabled = on("auto_save"); setWidgetDim(save, on("auto_save")); }
  if (send) { send.disabled = on("auto_send"); setWidgetDim(send, on("auto_send")); }
  node.setDirtyCanvas?.(true, true);
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
  // server side: next free name next to the source (never overwrites); with no
  // loader in the graph the server saves to saveAs or output/preview_#.png
  const serverSave = async (saveAs, loader) => {
    if (!sourcePath && !loader) {
      const r = findLoader(node);
      if (r.node) {
        loader = r.node;
      } else if (r.many?.length > 1) {
        loader = await chooseLoader(node, r.many);
        if (!loader) return { ok: false };
      }
    }
    const base = loader ? loaderBody(loader) : {};
    if (sourcePath) { base.mode = "single"; base.path = ""; base.image = ""; base.folder = ""; }
    const body = {
      ...base,
      filename: entry.filename, subfolder: entry.subfolder || "", type: entry.type || "temp",
      source: sourcePath, suffix: wv("suffix", "_#"), save_as: saveAs,
      format: wv("format", "same as source"), quality: wv("quality", 95),
    };
    let j;
    try {
      j = await (await api.fetchApi("/mm/image/save", { method: "POST", body: JSON.stringify(body) })).json();
    } catch (e) {
      j = { ok: false, error: String(e) };
    }
    if (j.ok) {
      node._rtLastSaved = j.path;
      toast("success", "Saved", j.path);
      node._rtRefreshTarget?.(); // next number is now taken
    } else toast("error", "Save failed", j.error || "unknown error");
    return j;
  };
  const stem = sourcePath ? String(sourcePath).replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "") : "image";
  const btn1 = mk("save");
  btn1.tooltip = node._rtTarget || "next free _# name next to the source";
  // no known home folder -> save behaves like save as: pasted images (Ctrl+V -> input/pasted/) and
  // uploads whose original was not found (picking/uploading leaves the loader's path empty then).
  // With no loader at all there is no home to be unknown: 'save' goes to output/preview_#.png,
  // 'save as…' opens the dialog.
  const isPasted = () => {
    const ln = findLoader(node)?.node;
    if (!ln && !sourcePath) return false;
    const lw = (name) => String(ln?.widgets?.find((x) => x.name === name)?.value ?? "");
    // folder mode: 'path' is empty by design – only a missing folder means 'unknown home'
    if (ln && lw("mode") === "folder") return !lw("folder").trim();
    if (ln && !lw("path").trim()) return true;
    const p = (sourcePath || lw("path") || lw("image")).replace(/\\/g, "/");
    return /(^|\/)pasted\//i.test(p);
  };
  btn1.callback = () => (isPasted() ? btn2.callback() : serverSave(""));
  const btn2 = mk("save as…");
  btn2.tooltip = "choose folder and name";
  btn2.callback = async () => {
    const suggested = (node._rtTarget ? String(node._rtTarget).replace(/\\/g, "/").split("/").pop() : "") || `${stem}_edit.png`;
    // the normal OS "save as" dialog (browser on the ComfyUI computer), starts in the target folder;
    // a loader in folder mode has no path: start in its folder instead
    const ln = findLoader(node)?.node;
    const lw = (name) => String(ln?.widgets?.find((x) => x.name === name)?.value ?? "");
    const lnFolder = String(ln?.widgets?.find((x) => x.name === "folder")?.value ?? "");
    const start = node._rtTarget || sourcePath || (ln && lw("mode") === "folder" ? lnFolder : "") || lw("path");
    let d;
    try {
      d = await (await api.fetchApi("/mm/image/dialog", { method: "POST", body: JSON.stringify({ mode: "save", path: start, name: suggested }) })).json();
    } catch (e) { d = { ok: false, error: String(e) }; }
    if (d.ok) {
      if (d.path) await serverSave(d.path);
      return;
    }
    if (d.local !== false) return toast("error", "Save as", d.error || "failed");
    // browser on another computer: the browser's own save dialog
    if (typeof window.showSaveFilePicker === "function") {
      // real file dialog (Chrome/Edge, secure context: localhost or https)
      let handle;
      try {
        handle = await window.showSaveFilePicker({ id: "mm-nodepack", startIn: "pictures", suggestedName: suggested, types: PICK_TYPES });
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
  const btn3 = mk("send to loader");
  btn3.tooltip = "save as _# and load it in the Load Image MM (next run works on this creation)";
  // savedPath: auto_save already wrote this creation -> send that file instead of saving a second copy
  const sendToLoader = async (savedPath) => {
    const loader = await resolveLoader(node);
    if (!loader) return;
    const j = savedPath ? { ok: true, path: savedPath } : await serverSave("", loader);
    if (!j?.ok) return;
    const pw = loader.widgets?.find((w) => w.name === "path");
    if (!pw) return;
    loader._mmSetMode?.("single");
    pw.value = j.path;
    pw.callback?.(j.path);
    // show the sent creation in the loader right away (the file = this preview image)
    const img = loadImage(entry);
    img.onload = () => {
      loader._rtPathShown = j.path;
      loader.imgs = [img];
      loader.imageIndex = 0;
      loader._mmEnsureSize?.();  // the loader shows an image now: give the thumbnail room
      app.graph.setDirtyCanvas(true, true);
    };
    loader.setDirtyCanvas?.(true, true);
    toast("info", `Loader #${loader.id}`, j.path);
  };
  btn3.callback = () => sendToLoader();
  node._rtSendToLoader = sendToLoader;
  // the most used ones at the bottom: save and save as last
  const order = [btn3, btn1, btn2];
  for (const b of order) {
    const idx = node.widgets.indexOf(b);
    if (idx !== -1) node.widgets.splice(idx, 1);
  }
  node.widgets.push(...order);
  node._rtButtons = [btn1, btn2, btn3];
  applyAutoState(node);
  fitNodeSize(node);
  node._rtRemoveButtons = () => {
    for (const b of node._rtButtons || []) {
      const idx = node.widgets?.indexOf(b);
      if (idx != null && idx !== -1) node.widgets.splice(idx, 1);
    }
    node._rtButtons = null;
    fitNodeSize(node, { shrinkHeight: true });
    node.setDirtyCanvas?.(true, true);
  };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "MM.Nodepack",
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
      else this._rtPending = [a, b];   // comparer module still loading
      const saved = Array.isArray(message?.saved) ? message.saved : [];
      if (saved.length) this._rtLastSaved = saved[saved.length - 1];
      if (saved.length) toast("success", "Auto-saved", saved.length > 1 ? `${saved[saved.length - 1]} (+${saved.length - 1})` : saved[0]);
      // buttons for the selected creation (save _# / save as…)
      const sel = cmp?.selected ?? 0;
      if (b[sel]) addSaveButtons(this, b[sel], source, sel);
      else this._rtRemoveButtons?.();  // nothing selected (e.g. 0 images): stale buttons must go
      if (b[sel] && this.widgets?.find((w) => w.name === "auto_send")?.value === true) {
        // never auto-overwrite an ambiguous or folder-batch loader
        const r = findLoader(this);
        if (!r.node) {
          if (r.many?.length > 1) {
            toast("warn", "Auto-send skipped", "Multiple Load Image MM nodes found — pick one in 'source'");
          }
        } else if (String(r.node.widgets?.find((x) => x.name === "mode")?.value || "single") === "folder") {
          toast("warn", "Auto-send skipped", `Loader #${r.node.id} is in folder mode — click "send to loader" to switch to single mode`);
        } else {
          this._rtSendToLoader?.(saved[sel] || "");
        }
      }
      this._rtRefreshButtons = () => {
        const c = this.widgets?.find((w) => w.name === "comparer");
        if (c && b[c.selected]) addSaveButtons(this, b[c.selected], source, c.selected);
      };
      this._rtRefreshTarget?.();
      this.setDirtyCanvas?.(true, true);
    };

    nodeType.prototype.onConnectionsChange = function (slot, type, isConnect, linkIds) {
      origConnections?.apply(this, arguments);
      this._rtRefreshTarget?.();  // images link changed -> source may change
      this.setDirtyCanvas?.(true, true);
    };

    // reloading a workflow restores the auto_save/auto_send values: refresh the live
    // target label and the dimmed state of the buttons the switches already cover
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origConfigure?.apply(this, arguments);
      this._rtRefreshTarget?.();
      applyAutoState(this);
      return r;
    };
  },
  nodeCreated(node) {
    if (nodeClass(node) === LOADER) {
      const iw = node.widgets?.find((w) => w.name === "image");
      const pw = node.widgets?.find((w) => w.name === "path");
      const mw = node.widgets?.find((w) => w.name === "mode");
      const fw = node.widgets?.find((w) => w.name === "folder");
      const bmw = node.widgets?.find((w) => w.name === "batch_mode");
      const idxw = node.widgets?.find((w) => w.name === "index");
      const ctrlw = node.widgets?.find((w) => w.name === "control_after_generate");

      // pre-0.2.0 workflows saved other mode values ("image" etc.): normalize to
      // single/folder at every entry point and on serialize
      const repairMode = () => {
        if (!mw) return;
        if (mw.value !== "single" && mw.value !== "folder") mw.value = "single";
      };
      repairMode();
      const mcb = mw?.callback;
      if (mw) {
        mw.serializeValue = () => (String(mw.value) === "folder" ? "folder" : "single");
        // route user combo changes through setMode (setMode calls mcb directly, no recursion)
        mw.callback = function (v) {
          const r = mcb?.apply(this, arguments);
          setMode(String(v ?? mw.value));
          return r;
        };
      }

      // mirrors the Python _clean: strip + remove matching surrounding quotes
      const cleanPath = (s) => {
        let t = String(s ?? "").trim();
        if (t.length >= 2 && ((t[0] === '"' && t[t.length - 1] === '"') || (t[0] === "'" && t[t.length - 1] === "'"))) {
          t = t.slice(1, -1).trim();
        }
        return t;
      };

      // move an already-added widget directly after `anchor` without ever duplicating it
      const moveAfter = (widget, anchor) => {
        if (!widget || !anchor || !node.widgets) return;
        const cur = node.widgets.indexOf(widget);
        if (cur !== -1) node.widgets.splice(cur, 1);
        const at = node.widgets.indexOf(anchor);
        if (at !== -1) node.widgets.splice(at + 1, 0, widget);
        else node.widgets.push(widget);
      };

      // the core upload button (if this frontend ships one) keeps its natural slot right
      // below 'image' and is relabeled "choose file" (the click handler prefers the OS
      // dialog; browser upload is only the fallback for a remote browser)
      let upw = node.widgets?.find((w) => w.name === "upload")
        || node.widgets?.find((w) => w.type === "button" && w.name !== "choose folder");
      const origUpload = upw?.callback;
      if (!upw) {
        // this frontend ships no core upload button: own fallback
        upw = node.addWidget("button", "choose file", null, () => {});
      }
      upw.label = "choose file";
      upw.serialize = false;
      upw.options = { ...(upw.options || {}), serialize: false };
      // Always place "choose file" directly below 'image' (core appends 'upload' at the tail)
      moveAfter(upw, iw);

      // the "choose folder" button lives right below the 'folder' widget (visible only in folder mode)
      const folderBtn = node.addWidget("button", "choose folder", null, () => {});
      folderBtn.label = "choose folder";
      folderBtn.serialize = false;
      folderBtn.options = { ...(folderBtn.options || {}), serialize: false };
      moveAfter(folderBtn, fw);

      // one shared stale-thumbnail guard for both modes
      let thumbSeq = 0;

      // Match original ComfyUI LoadImage & Preview & Save MM: let ComfyUI's native
      // setSizeForImage handle image sizing (same 220 px preview minimum), and never
      // shrink what the user set (mode switches never crush the thumbnail).
      const sizeLoaderNode = (forImage) => {
        if (!node.graph) {
          if (node._mmSizeQueued) {
            if (forImage) node._mmSizeForImage = true;
            return;
          }
          node._mmSizeQueued = true;
          node._mmSizeForImage = Boolean(forImage);
          requestAnimationFrame(() => {
            node._mmSizeQueued = false;
            if (node.graph) sizeLoaderNode(node._mmSizeForImage);
          });
          return;
        }
        const prevW = node.size?.[0] ?? 0;
        const prevH = node.size?.[1] ?? 0;
        if (forImage && node.imgs?.length) {
          node.setSizeForImage?.();
        }
        const min = node.computeSize?.();
        if (!min || !node.size) return;
        const w = Math.max(prevW, node.size[0], min[0]);
        const minWithImg = (forImage && node.imgs?.length) ? min[1] : min[1];
        const h = Math.max(prevH, node.size[1], minWithImg);
        node.setSize?.([w, h]);
        node.arrange?.();
        node.setDirtyCanvas?.(true, true);
      };
      node._mmEnsureSize = () => sizeLoaderNode(true);

      const getMode = () => (String(mw?.value || "single") === "folder" ? "folder" : "single");

      const applyMode = () => {
        const isFolder = getMode() === "folder";
        // the mode selector is always visible; the other schema widgets and the two mode
        // buttons follow the mode. Widgets stay in node.widgets (widgets_values is
        // positional): hide via type/visibility.
        setWidgetVisible(node, "mode", true);
        setWidgetVisible(node, "image", !isFolder);
        setWidgetVisible(node, "path", !isFolder);
        setWidgetVisible(node, "folder", isFolder);
        setWidgetVisible(node, "batch_mode", isFolder);
        setWidgetVisible(node, "index", isFolder);
        setWidgetVisible(node, "control_after_generate", isFolder);
        if (!isFolder) {
          // the index control only means anything in folder mode: park it on fixed
          if (ctrlw && ctrlw.value !== "fixed") {
            (node.properties ??= {}).mm_folder_ctrl = ctrlw.value;
            ctrlw.value = "fixed";
            ctrlw.callback?.(ctrlw.value);
          }
        } else if (ctrlw) {
          ctrlw.value = node.properties?.mm_folder_ctrl || (bmw?.value ? "randomize" : "increment");
          ctrlw.callback?.(ctrlw.value);
        }
        setWidgetObjVisible(node, upw, !isFolder);
        setWidgetObjVisible(node, folderBtn, isFolder);
        sizeLoaderNode(false);
      };

      const setMode = (next) => {
        const val = next === "folder" ? "folder" : "single";
        if (mw && mw.value !== val) {
          mw.value = val;
          mcb?.call(mw, val);
          node.graph?.change?.();
        }
        applyMode();
        _refreshCmp(node);
        if (val === "folder") refreshFolderThumb();  // switching back shows the right thumbnail
        else refreshSingleThumb();
      };
      node._mmSetMode = setMode;

      // one click = OS file dialog on the ComfyUI computer (real server paths, external
      // folders included via the temp folder). Cancel = nothing changes. A remote browser
      // (local === false) falls back to its own browser upload.
      upw.callback = async function (...args) {
        let d;
        try {
          d = await (await api.fetchApi("/mm/image/dialog", {
            method: "POST",
            body: JSON.stringify({ mode: "open", path: cleanPath(pw?.value) }),
          })).json();
        } catch (e) {
          d = { ok: false, error: String(e) };
        }
        if (d?.ok) {
          if (d.path) {
            if (pw) pw.value = d.path;
            node.graph?.change?.();
            const seq = ++thumbSeq;
            node._rtPathShown = cleanPath(d.path);
            node.imgs = undefined;
            if (d.view) {
              const img = loadImage(d.view);
              const applyThumb = () => {
                if (seq !== thumbSeq || getMode() !== "single") return;
                node.imgs = [img];
                node.imageIndex = 0;
                sizeLoaderNode(true);  // the thumbnail needs room: grow once, never crush
                node.setDirtyCanvas?.(true, true);
              };
              img.onload = applyThumb;
              if (img.complete && img.naturalWidth) applyThumb();
            } else {
              refreshSingleThumb();
            }
            node.setDirtyCanvas?.(true, true);
            _refreshCmp(node);
          }
          return;  // dialog cancelled (no path): nothing changes
        }
        if (d?.local === false) {
          if (origUpload) {
            const r = await origUpload.apply(this, args);
            _refreshCmp(node);
            return r;
          }
          toast("error", "Choose file", "no OS file dialog here and no browser upload (remote server)");
          return;
        }
        toast("error", "Choose file", d?.error || "failed");
      };

      // one click = OS folder dialog; cancel = nothing changes. A folder dialog exists only
      // on the ComfyUI computer: a remote browser gets an error toast (no browser fallback).
      folderBtn.callback = async () => {
        let d;
        try {
          d = await (await api.fetchApi("/mm/image/dialog", {
            method: "POST",
            body: JSON.stringify({ mode: "folder", path: cleanPath(fw?.value) }),
          })).json();
        } catch (e) {
          d = { ok: false, error: String(e) };
        }
        if (!d?.ok) {
          if (d?.local === false) toast("error", "Choose folder", "folder dialogs only work from a browser on the ComfyUI computer");
          else toast("error", "Choose folder", d?.error || "failed");
          return;
        }
        if (!d.path) return;  // dialog cancelled: nothing changes
        if (fw) {
          fw.value = d.path;
          fw.callback?.(d.path);
          node.graph?.change?.();
        }
        if (typeof d.count === "number") {
          toast(d.count > 0 ? "info" : "warn", "Folder selected", `${d.count} image${d.count === 1 ? "" : "s"} in folder`);
        }
        // fw.callback above already triggered refreshFolderThumb: it loads the exact image for
        // the current index/batch_mode (external folders included via the temp-folder thumbnails)
        _refreshCmp(node);
      };

      if (iw && pw) {
        const cb = iw.callback;
        iw.callback = function (v) {
          // the pick is the source now: clear the old path BEFORE the core loads the thumbnail,
          // else onDrawBackground drops it as "belongs to another path" (resolve sets the new one)
          pw.value = "";
          node._rtPathShown = "";
          const r = cb?.apply(this, arguments);
          const name = String(v ?? iw.value ?? "");
          if (name) {
            api.fetchApi("/mm/image/resolve", { method: "POST", body: JSON.stringify({ image: name }) })
              .then((res) => res.json())
              .then((j) => {
                if (!j?.ok || String(iw.value) !== name) return;   // picked again meanwhile
                const setPath = (p) => {
                  node._rtPathShown = p;   // the picker thumbnail is the right image
                  pw.value = p;
                  node.graph?.change?.();
                  node.setDirtyCanvas?.(true, true);
                  _refreshCmp(node);
                };
                setPath(j.path);
                if (!cb || !node.imgs?.length) refreshSingleThumb();  // the picker loaded no thumbnail of its own
                // identical copies in several folders: the browser never tells which one -> say so
                if (j.candidates?.length > 1) toast("warn", "Same image in several folders", `${j.candidates.join(" · ")} → use "choose file" to pick the exact one`);
              })
              .catch(() => {});
          }
          return r;
        };
        const clearThumb = () => {
          if (cleanPath(pw.value) === "") return;
          node.imgs = undefined;
          if (node.images) node.images = undefined;
          try { app.nodeOutputs && delete app.nodeOutputs[String(node.id)]; } catch (_) { /* ignore */ }
        };
        const pcb = pw.callback;
        pw.callback = function () {
          const r = pcb?.apply(this, arguments);
          clearThumb();
          refreshSingleThumb();  // pasting/typing a path shows the image right away
          _refreshCmp(node);
          node.setDirtyCanvas?.(true, true);
          return r;
        };
      }

      // Both modes keep the loader's own thumbnail on the resolved image,
      // so stepping the index or typing a path immediately shows the file.
      // Debounced with a shared sequence guard: with a rapid change (1 -> 2 -> 3 -> 4)
      // a slow response for an older selection must never overwrite the newer thumbnail.
      const refreshFolderThumb = debounce(() => {
        if (getMode() !== "folder" || !cleanPath(fw?.value)) return;
        const seq = ++thumbSeq;
        const body = { ...loaderBody(node), mode: "folder" };
        api.fetchApi("/mm/image/preview", { method: "POST", body: JSON.stringify(body) })
          .then((res) => res.json())
          .then((j) => {
            if (seq !== thumbSeq || getMode() !== "folder") return;
            if (!j?.ok || !j.view) return;
            const img = loadImage(j.view);
            img.onload = () => {
              if (seq !== thumbSeq) return;
              node._rtPathShown = cleanPath(j.path);
              node.imgs = [img];
              node.imageIndex = 0;
              sizeLoaderNode(true);  // a visible image needs room in the node
              node.setDirtyCanvas?.(true, true);
            };
          })
          .catch(() => {});
      }, 150);
      const refreshSingleThumb = debounce(() => {
        if (getMode() !== "single") return;
        if (!cleanPath(pw?.value) && !String(iw?.value || "").trim()) return;
        const seq = ++thumbSeq;
        const body = { ...loaderBody(node), mode: "single" };
        api.fetchApi("/mm/image/preview", { method: "POST", body: JSON.stringify(body) })
          .then((res) => res.json())
          .then((j) => {
            if (seq !== thumbSeq || getMode() !== "single") return;
            if (!j?.ok || !j.view) return;
            const img = loadImage(j.view);
            img.onload = () => {
              if (seq !== thumbSeq) return;
              node._rtPathShown = cleanPath(j.path);
              node.imgs = [img];
              node.imageIndex = 0;
              sizeLoaderNode(true);  // a visible image needs room in the node
              node.setDirtyCanvas?.(true, true);
            };
          })
          .catch(() => {});
      }, 150);

      if (fw) {
        const fcb = fw.callback;
        fw.callback = function () {
          const r = fcb?.apply(this, arguments);
          _refreshCmp(node);
          refreshFolderThumb();
          node.setDirtyCanvas?.(true, true);
          return r;
        };
      }

      if (bmw) {
        const bmcb = bmw.callback;
        bmw.callback = function (v) {
          const r = bmcb?.apply(this, arguments);
          if (ctrlw) {
            ctrlw.value = v ? "randomize" : "increment";
            ctrlw.callback?.(ctrlw.value);
          }
          _refreshCmp(node);
          refreshFolderThumb();
          node.setDirtyCanvas?.(true, true);
          return r;
        };
      }

      if (idxw) {
        const icb = idxw.callback;
        idxw.callback = function () {
          const r = icb?.apply(this, arguments);
          _refreshCmp(node);
          refreshFolderThumb();
          return r;
        };
      }

      const origCfg = node.onConfigure;
      node.onConfigure = function () {
        const r = origCfg?.apply(this, arguments);
        repairMode();  // legacy workflows: normalize the mode value
        applyMode();
        // reloaded workflow: show its current source (single: path or the image widget, e.g. pasted)
        if (getMode() === "folder") {
          if (cleanPath(fw?.value)) refreshFolderThumb();
        } else if (cleanPath(pw?.value) || String(iw?.value || "").trim()) {
          refreshSingleThumb();
        }
        return r;
      };
      applyMode();

      const origExec = node.onExecuted;
      node.onExecuted = function (message) {
        origExec?.apply(this, arguments);
        // the backend previews the resolved file -> this thumbnail is the right one
        const shownPath = Array.isArray(message?.path) && message.path.length ? String(message.path[0] || "") : String(this.widgets?.find((w) => w.name === "path")?.value || "");
        this._rtPathShown = Array.isArray(message?.images) && message.images.length ? cleanPath(shownPath) : "";
        this.setDirtyCanvas?.(true, true);
      };
      const origBg = node.onDrawBackground;
      node.onDrawBackground = function () {
        // single path mode: drop the picker's thumbnail if it belongs to another path;
        // both sides cleaned, so a quoted path never wipes its own thumbnail
        const isFolder = getMode() === "folder";
        if (!isFolder) {
          const pv = cleanPath(this.widgets?.find((w) => w.name === "path")?.value);
          if (pv && this.imgs?.length && cleanPath(this._rtPathShown) !== pv) this.imgs = undefined;
        }
        return origBg?.apply(this, arguments);
      };
      return;
    }
    if (!SAVE_NODES.has(nodeClass(node))) return;
    // comparer goes right after the input widgets (before target line and buttons)
    const at = node.widgets?.length ?? 0;
    loadComparer().then(({ ComparerWidget, wirePointer }) => {
      const cmp = new ComparerWidget(node);
      node.addCustomWidget(cmp);
      const ws = node.widgets, i = ws.indexOf(cmp);
      if (i > at) { ws.splice(i, 1); ws.splice(at, 0, cmp); }
      wirePointer(node, cmp);
      if (node._rtPending) { cmp.setData(...node._rtPending); node._rtPending = null; }
      fitNodeSize(node);
      node.setDirtyCanvas?.(true, true);
    }).catch((e) => console.error("[MM Nodepack] comparer not loaded", e));
    addTargetWidget(node);
    // target preview updates when suffix/save_as/format change
    for (const name of ["suffix", "save_as", "format", "source"]) {
      const w = node.widgets?.find((x) => x.name === name);
      if (!w) continue;
      const cb = w.callback;
      w.callback = (...a) => {
        cb?.(...a);
        if (name === "source") node._rtNote = "";  // the run's note is outdated now
        node._rtRefreshTarget?.();
      };
    }
    for (const name of ["auto_save", "auto_send"]) {
      const w = node.widgets?.find((x) => x.name === name);
      if (!w) continue;
      const cb = w.callback;
      w.callback = (...a) => {
        const r = cb?.(...a);
        applyAutoState(node);
        if (name === "auto_send" && w.value === true) {
          // several loaders: let the user pick once, so the auto-send has a target
          const rl = findLoader(node);
          if (!rl.node && rl.many?.length > 1) chooseLoader(node, rl.many);
        }
        return r;
      };
    }
    node._rtRefreshTarget?.();
  },
});
