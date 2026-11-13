// MM Nodepack · free GPU – one button in the top action bar (2026-11-13).
// Unloads all models and frees the cache (ComfyUI's own POST /free), like the button the old ComfyUI-Manager menu had.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function toast(severity, summary, detail) {
  const t = app.extensionManager?.toast;
  if (t?.add) t.add({ severity, summary, detail, life: 2500 });
  else console.log(`[MM free GPU] ${summary}: ${detail}`);
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

// What ComfyUI itself holds on the GPU (memory PyTorch reserved), next to the label – the monitor beside it shows the
// whole GPU, other programs included.
async function showUsage() {
  const span = document.querySelector(`${B} > span`);
  if (!span) return;
  try {
    const d = (await (await api.fetchApi("/system_stats")).json())?.devices?.[0];
    const gb = d && d.torch_vram_total != null ? d.torch_vram_total / 1073741824 : null;
    span.textContent = gb == null ? "free GPU" : `free GPU · ${gb.toFixed(1)} GB`;
    span.parentElement.title = TIP + (gb == null ? "" : `\nComfyUI holds ${gb.toFixed(1)} GB of GPU memory now.`);
  } catch { /* server restarting */ }
}
setInterval(showUsage, 3000);

const TIP = "Unload all models and free the cache (GPU and RAM). Takes effect after the running job.";

// Like the VRAM monitor next to it: slim label (Inter 100, 10 px) at the bottom, bigger icon
const B = `[data-testid="action-bar-buttons"] button[aria-label="${TIP}"]`;
const style = document.createElement("style");
style.textContent = `${B}, ${B} * { font: 100 10px/12px Inter, Arial, sans-serif; color: #fff; }
${B} { gap: 4px; }
${B} > i { width: 22px; height: 22px; }
${B} > span { align-self: flex-end; margin-bottom: 2px; }`;
document.head.appendChild(style);

app.registerExtension({
  name: "MM.FreeGpu",
  actionBarButtons: [
    {
      icon: "icon-[lucide--cpu] size-4",
      label: "free GPU",
      tooltip: TIP,
      onClick: freeGpu,
    },
  ],
});
