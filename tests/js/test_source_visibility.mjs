// Unit tests for the Preview & Save source-picker rule
// (web/lib/mm_source.mjs, imported by web/mm_nodepack.js).
import assert from "node:assert/strict";
import {
  normalizeSource,
  sourceShouldShow,
  sourceOptions,
  sourceValueFromLabel,
} from "../../web/lib/mm_source.mjs";

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("normalizeSource: trims, strips leading #, lower-cases", () => {
  assert.equal(normalizeSource("auto"), "auto");
  assert.equal(normalizeSource("AUTO"), "auto");
  assert.equal(normalizeSource("  auto  "), "auto");
  assert.equal(normalizeSource("#12"), "12");
  assert.equal(normalizeSource("  #12  "), "12");
  assert.equal(normalizeSource("# 12"), "12");
  assert.equal(normalizeSource(undefined), "auto");
  assert.equal(normalizeSource(""), "auto");
});

test("sourceShouldShow: hidden while auto and unambiguous", () => {
  assert.equal(sourceShouldShow({ value: "auto" }), false);
  assert.equal(sourceShouldShow({ value: "AUTO" }), false);
  assert.equal(sourceShouldShow({ value: "  auto  " }), false);
  assert.equal(sourceShouldShow({ value: undefined }), false);
  assert.equal(sourceShouldShow({}), false);
  assert.equal(sourceShouldShow({ value: "auto", many: false, socketLinked: false }), false);
});

test("sourceShouldShow: shown for a pinned value", () => {
  assert.equal(sourceShouldShow({ value: "#12" }), true);
  assert.equal(sourceShouldShow({ value: "12" }), true);
  assert.equal(sourceShouldShow({ value: " #12 " }), true);
});

test("sourceShouldShow: shown when auto is ambiguous or the socket is linked", () => {
  assert.equal(sourceShouldShow({ value: "auto", many: true }), true);
  assert.equal(sourceShouldShow({ value: "auto", socketLinked: true }), true);
  assert.equal(sourceShouldShow({ value: "auto", many: true, socketLinked: true }), true);
});

test("sourceOptions: auto with loaders lists every label once", () => {
  const loaders = [
    { id: 12, label: "#12  a.png" },
    { id: 34, label: "#34  b/" },
  ];
  const r = sourceOptions({ value: "auto", loaders });
  assert.deepEqual(r.options, ["auto", "#12  a.png", "#34  b/"]);
  assert.equal(r.selected, "auto");
});

test("sourceOptions: pinned #id selects the matching loader label", () => {
  const loaders = [
    { id: 12, label: "#12  a.png" },
    { id: 34, label: "#34  b/" },
  ];
  const r = sourceOptions({ value: "#34", loaders });
  assert.deepEqual(r.options, ["auto", "#12  a.png", "#34  b/"]);
  assert.equal(r.selected, "#34  b/");
});

test("sourceOptions: missing #id keeps a (missing) entry selected", () => {
  const r = sourceOptions({ value: "#9", loaders: [{ id: 12, label: "#12  a.png" }] });
  assert.deepEqual(r.options, ["auto", "#12  a.png", "#9 (missing)"]);
  assert.equal(r.selected, "#9 (missing)");
});

test("sourceOptions: no loaders", () => {
  assert.deepEqual(sourceOptions({ value: "auto" }).options, ["auto"]);
  assert.equal(sourceOptions({ value: "auto" }).selected, "auto");
  const r = sourceOptions({ value: "#7" });
  assert.deepEqual(r.options, ["auto", "#7 (missing)"]);
  assert.equal(r.selected, "#7 (missing)");
});

test("sourceValueFromLabel: maps labels back to stored values", () => {
  assert.equal(sourceValueFromLabel("auto"), "auto");
  assert.equal(sourceValueFromLabel("Auto"), "auto");
  assert.equal(sourceValueFromLabel(""), "auto");
  assert.equal(sourceValueFromLabel("#12  a.png"), "#12");
  assert.equal(sourceValueFromLabel("#9 (missing)"), "#9");
  assert.equal(sourceValueFromLabel(undefined), "auto");
});

let failed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
    console.log(`ok - ${name}`);
  } catch (e) {
    failed += 1;
    console.error(`FAIL - ${name}\n  ${e.message}`);
  }
}
if (failed) process.exit(1);
console.log(`${tests.length} source visibility tests passed`);
