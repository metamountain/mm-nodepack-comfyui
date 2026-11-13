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
}

app.registerExtension({
  name: "MM.FreeGpu",
  actionBarButtons: [
    {
      icon: "icon-[lucide--cpu] size-4",
      label: "free GPU",
      tooltip: "Unload all models and free the cache (GPU and RAM). Takes effect after the running job.",
      onClick: freeGpu,
    },
  ],
});
