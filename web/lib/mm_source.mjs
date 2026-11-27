// Pure helpers for the Preview & Save "source" picker (web/mm_nodepack.js):
// which Load Image MM node feeds the preview. Kept import-free so the rule can
// be unit-tested with Node (tests/js/test_source_visibility.mjs).

// "  #12  " -> "12"; undefined or empty -> "auto" (lower-cased for comparison)
export function normalizeSource(value) {
  const s = String(value ?? "auto").trim().replace(/^#/, "").trim();
  return (s || "auto").toLowerCase();
}

// The picker is shown when the value is not auto (a pinned #id must stay
// visible so the user can reset it), when auto resolution is ambiguous
// (several candidate loaders), or when the source input socket is linked.
export function sourceShouldShow({ value, many = false, socketLinked = false } = {}) {
  return normalizeSource(value) !== "auto" || Boolean(many) || Boolean(socketLinked);
}

// Dropdown entries: "auto" plus one label per current Load Image MM. If the
// stored value pins a #id that no longer exists, an extra "<#id> (missing)"
// entry is appended and selected so nothing is silently changed.
export function sourceOptions({ value, loaders = [] } = {}) {
  const options = ["auto"];
  const seen = new Set(options);
  for (const l of loaders) {
    if (l && l.label && !seen.has(l.label)) {
      seen.add(l.label);
      options.push(l.label);
    }
  }
  const sel = normalizeSource(value);
  let selected = "auto";
  if (sel !== "auto") {
    const hit = loaders.find((l) => String(l?.id) === sel);
    if (hit && hit.label) {
      selected = hit.label;
    } else {
      const missing = `#${sel} (missing)`;
      if (!seen.has(missing)) options.push(missing);
      selected = missing;
    }
  }
  return { options, selected };
}

// Map a dropdown label back to the stored value: "auto" (or empty) -> "auto",
// "#12  file.png" / "#12 (missing)" -> "#12".
export function sourceValueFromLabel(label) {
  const s = String(label ?? "").trim();
  if (!s || s.toLowerCase() === "auto") return "auto";
  const m = s.match(/^#([^ ]+)/);
  return m ? `#${m[1]}` : s;
}
