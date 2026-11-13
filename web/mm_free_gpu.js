// MM Nodepack · free GPU – one button in the top action bar (2026-11-13).
// Unloads all models and frees the cache (ComfyUI's own POST /free), like the button the old ComfyUI-Manager menu had.
// Top right: the GPU memory ComfyUI itself holds (PyTorch reserved) – the monitor beside it shows the whole GPU,
// other programs included. No polling: refreshed on page load, after the click and when a run starts or ends.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TIP = "Unload all models and free the cache (GPU and RAM). Takes effect after the running job.";
const B = `[data-testid="action-bar-buttons"] button[aria-label="${TIP}"]`;

// Like the monitor bars next to it: slim label (Inter 100, 10 px) bottom left, value (500, 11 px) top right
const style = document.createElement("style");
style.textContent = `${B}, ${B} * { font: 100 10px/12px Inter, Arial, sans-serif; color: #fff; }
${B} { gap: 4px; position: relative; min-width: 104px !important; height: 30px !important; justify-content: flex-start !important;
  padding: 0 6px 0 4px !important; border-radius: 0 !important; background: rgb(34, 34, 34) !important; }
${B}:hover { background: rgb(52, 52, 52) !important; }
${B} > .mm-gb { position: absolute; top: 2px; right: 6px; font: 500 11px/14px Inter, Arial, sans-serif; }
${B} > i { width: 22px; height: 22px; }
${B} > span { align-self: flex-end; margin-bottom: 2px; }`;
document.head.appendChild(style);

function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary, detail, life: 2500 });
  else console.log(`[MM free GPU] ${summary}: ${detail}`);
}

async function showUsage() {
  const span = document.querySelector(`${B} > span`);
  if (!span) return;
  try {
    const d = (await (await api.fetchApi("/system_stats")).json())?.devices?.[0];
    const gb = d && d.torch_vram_total != null ? d.torch_vram_total / 1073741824 : null;
    const btn = span.parentElement;
    let val = btn.querySelector(".mm-gb");
    if (!val) {
      val = document.createElement("div");
      val.className = "mm-gb";
      btn.appendChild(val);
    }
    val.textContent = gb == null ? "" : `${gb.toFixed(1)} GB`;
    btn.title = TIP + (gb == null ? "" : `\nComfyUI holds ${gb.toFixed(1)} GB of GPU memory now.`);
  } catch { /* server restarting */ }
}

async function freeGpu() {
  try {
    const r = await api.fetchApi("/free", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ unload_models: true, free_memory: true }),
    });
    if (r.ok) toast("success", "free GPU", "Models unloaded, cache freed.");
    else toast("error", "free GPU", `HTTP ${r.status}`);
  } catch (e) {
    toast("error", "free GPU", String(e?.message || e));
  }
  setTimeout(showUsage, 1500);
}

for (const ev of ["execution_start", "execution_success", "execution_error", "execution_interrupted"]) {
  api.addEventListener(ev, () => setTimeout(showUsage, 500));
}

app.registerExtension({
  name: "MM.FreeGpu",
  actionBarButtons: [
    { icon: "icon-[lucide--cpu] size-4", label: "free GPU", tooltip: TIP, onClick: freeGpu },
  ],
  setup() {
    // once: wait (at most 60 s) until the action bar has rendered the button
    (function first(tries = 60) {
      if (document.querySelector(`${B} > span`)) showUsage();
      else if (tries > 0) setTimeout(() => first(tries - 1), 1000);
    })();
  },
});
