// MM Nodepack titlebar logo: the pack mark on every MM node (all node_ids start
// with "MM"; the prefix rule also covers future MM* nodes). Single place for the
// logo wiring, so no per-node attach calls are needed elsewhere.
import { app } from "../../scripts/app.js";
import { attachMMLogo } from "./lib/mm_ui.mjs";

app.registerExtension({
  name: "MM.Logo",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (typeof nodeData?.name === "string" && nodeData.name.startsWith("MM")) {
      attachMMLogo(nodeType);
    }
  },
});
