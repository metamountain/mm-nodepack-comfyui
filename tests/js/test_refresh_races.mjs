// Node test for the SHIPPED race-safe refresh channels: web/lib/mm_race.mjs is
// the exact module web/mm_nodepack.js uses for the compare-preview refresh
// (400 ms, per-loader) and the live target tooltip (300 ms, per-saver).
//
// The test drives createRefreshChannel with fake timers, deferred fetches and
// a minimal fake graph. The wiring below mirrors how mm_nodepack.js consumes
// the channel (fetch = the /mm/image/* request + body from the loader's
// widgets; write = the comparer widget / target tooltip update) — the
// debounce, per-key revision and the post-fetch stale check all execute in
// the shipped module, not here.
//
// Run: node tests/js/test_refresh_races.mjs
import assert from "node:assert/strict";
import { createRefreshChannel } from "../../web/lib/mm_race.mjs";

// ---------------------------------------------------------------------------
// deterministic fake timer clock (the shipped debounce uses the globals)
// ---------------------------------------------------------------------------
let now = 0;
let seq = 0;
let queue = [];
globalThis.setTimeout = (fn, ms = 0) => {
  const id = ++seq;
  queue.push({ id, at: now + Number(ms || 0), fn });
  return id;
};
globalThis.clearTimeout = (id) => {
  queue = queue.filter((t) => t.id !== id);
};
const advance = (ms) => {
  const target = now + ms;
  for (;;) {
    queue.sort((a, b) => a.at - b.at);
    const t = queue[0];
    if (!t || t.at > target) break;
    now = t.at;
    queue.shift();
    t.fn();
  }
  now = Math.max(now, target);
};
// flush the microtask queues created by the channel's awaited fetches
const tick = () => new Promise((r) => setImmediate(r));

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};

// ---------------------------------------------------------------------------
// minimal fake graph
// ---------------------------------------------------------------------------
const makeLoader = (id, { path = "", image = "", folder = "", mode = "single" } = {}) => ({
  id,
  type: "MMLoadImage",
  widgets: [
    { name: "mode", value: mode },
    { name: "path", value: path },
    { name: "image", value: image },
    { name: "folder", value: folder },
    { name: "batch_mode", value: false },
    { name: "index", value: 1 },
  ],
  setDirtyCanvas: () => {},
});
const makeSaver = (id, source) => ({
  id,
  type: "MMPreviewSave",
  widgets: [
    { name: "source", value: source },
    { name: "comparer", value: { a: [] }, imgA: null, writes: [] },
  ],
  _rtRefreshTarget: () => {},
  setDirtyCanvas: () => {},
});
const makeGraph = (nodes) => {
  const g = { _nodes: nodes, links: new Map(), getNodeById: (id) => nodes.find((n) => n.id === id) };
  for (const n of nodes) n.graph = g;
  return g;
};
// the test-side stand-in for findLoader(): an explicit "source #id" selects
// that loader (the graph-level rules of mm_nodepack.js are not race-relevant)
const findLoader = (saver) => {
  const sel = String(saver.widgets.find((w) => w.name === "source")?.value ?? "auto").replace(/^#/, "").trim();
  const loaders = (saver.graph._nodes || []).filter((n) => n.type === "MMLoadImage");
  if (sel && sel.toLowerCase() !== "auto") {
    const n = loaders.find((x) => String(x.id) === sel);
    return { node: n ?? null, note: n ? "" : `source #${sel}: no Load Image MM` };
  }
  return loaders.length === 1 ? { node: loaders[0], note: "" } : { node: null, note: "no loader" };
};
const bodyOf = (loader) => {
  const w = (name) => loader?.widgets?.find((x) => x.name === name)?.value;
  return {
    mode: String(w("mode") ?? "single") || "single",
    path: String(w("path") ?? "").trim(),
    image: String(w("image") ?? "").trim(),
    folder: String(w("folder") ?? "").trim(),
  };
};
const emptyBody = (b) => (b.mode === "folder" ? !b.folder : (!b.path && !b.image));

// compare channel wired like mm_nodepack.js (400 ms, per-loader key)
const makeCompareChannel = (previewFetch) => createRefreshChannel({
  delay: 400,
  async fetch(loaderNode) {
    const graph = loaderNode.graph;
    const body = bodyOf(loaderNode);
    if (emptyBody(body)) return { entry: null, graph };
    const view = await previewFetch(loaderNode, body);
    if (!view) throw new Error("no preview for the current source");
    return { entry: view, graph };
  },
  write(loaderNode, payload) {
    if (!payload.ok) return;  // fetch failed: keep the current original
    const { entry, graph } = payload.result;
    if (!graph || loaderNode.graph !== graph) return;
    for (const n of graph._nodes || []) {
      if (n.type !== "MMPreviewSave") continue;
      const { node: src } = findLoader(n);
      if (src !== loaderNode) continue;
      const cmp = n.widgets.find((w) => w.name === "comparer");
      cmp.value.a = entry ? [entry] : [];
      cmp.imgA = entry ? { entry } : null;
      cmp.writes.push(entry ? [entry] : []);
    }
  },
});
// target channel wired like addTargetWidget() in mm_nodepack.js (300 ms, per-saver)
const makeTargetChannel = (node, targetFetch) => createRefreshChannel({
  delay: 300,
  async fetch() {
    const { node: loader } = findLoader(node);
    const body = { ...bodyOf(loader) };
    if (!loader || emptyBody(body)) {
      return { target: "output/preview_# (no Load Image MM found)" };
    }
    const j = await targetFetch(loader, body);
    return { target: j.ok ? j.target : (j.error || "target?") };
  },
  write(_key, payload) {
    if (!payload.ok) node._rtTarget = payload.error;   // fetch error (newest revision only)
    else node._rtTarget = payload.result.target;        // live target or empty-source state
  },
});

// ---------------------------------------------------------------------------
// the race cases (rows of the repair-4 table + the extra cases)
// ---------------------------------------------------------------------------
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("edit loader A then B within the 400 ms window: BOTH previews update", async () => {
  const A = makeLoader("A", { path: "a.png" });
  const B = makeLoader("B", { path: "b.png" });
  const sA = makeSaver("sA", "#A");
  const sB = makeSaver("sB", "#B");
  makeGraph([A, B, sA, sB]);
  const pending = {};
  const previewFetch = (loader) => { pending[loader.id] = deferred(); return pending[loader.id].promise; };
  const channel = makeCompareChannel(previewFetch);
  channel.schedule(A);        // t=0  (fires at t=400)
  advance(100);               // t=100
  channel.schedule(B);        // t=100 (fires at t=500) — inside A's window
  advance(300);               // t=400: A's fetch starts
  advance(100);               // t=500: B's fetch starts
  assert.equal(Object.keys(pending).length, 2, "both loaders issued a preview request");
  pending.A.resolve({ filename: "a.png", type: "input" });
  await tick();
  pending.B.resolve({ filename: "b.png", type: "input" });
  await tick();
  assert.deepEqual(sA.widgets.find((w) => w.name === "comparer").value.a, [{ filename: "a.png", type: "input" }]);
  assert.deepEqual(sB.widgets.find((w) => w.name === "comparer").value.a, [{ filename: "b.png", type: "input" }]);
  assert.equal(sA.widgets.find((w) => w.name === "comparer").writes.length, 1);
  assert.equal(sB.widgets.find((w) => w.name === "comparer").writes.length, 1);
});

test("two preview requests for A, the newer finishes first: the newer original stays", async () => {
  const A = makeLoader("A", { path: "a.png" });
  const sA = makeSaver("sA", "#A");
  makeGraph([A, sA]);
  const pending = [];
  const previewFetch = () => { const d = deferred(); pending.push(d); return d.promise; };
  const channel = makeCompareChannel(previewFetch);
  channel.schedule(A);        // rev 1, fires at t=400
  advance(400);               // fetch 1 (older) starts, still in flight
  A.widgets.find((w) => w.name === "path").value = "a2.png";
  channel.schedule(A);        // rev 2 (new edit while fetch 1 is in flight), fires at t=800
  advance(400);               // fetch 2 (newer) starts
  assert.equal(pending.length, 2, "two preview requests are in flight for A");
  pending[1].resolve({ filename: "a2.png", type: "input" });  // the NEWER one finishes first
  await tick();
  const cmp = sA.widgets.find((w) => w.name === "comparer");
  assert.deepEqual(cmp.value.a, [{ filename: "a2.png", type: "input" }], "the newer original is visible");
  pending[0].resolve({ filename: "a.png", type: "input" });   // the OLDER one arrives late
  await tick();
  assert.deepEqual(cmp.value.a, [{ filename: "a2.png", type: "input" }], "the older response must not replace it");
  assert.equal(cmp.writes.length, 1, "only the newest revision wrote to the comparer");
});

test("two target requests, the newer finishes first: the newer tooltip stays", async () => {
  const A = makeLoader("A", { path: "a.png" });
  const s = makeSaver("s", "#A");
  makeGraph([A, s]);
  const pending = [];
  const targetFetch = () => { const d = deferred(); pending.push(d); return d.promise; };
  const channel = makeTargetChannel(s, targetFetch);
  s._rtRefreshTarget = () => channel.schedule(s);
  s._rtRefreshTarget();       // rev 1, fires at t=300
  advance(300);               // fetch 1 (older) starts, still in flight
  s._rtRefreshTarget();       // rev 2, fires at t=600
  advance(300);               // fetch 2 (newer) starts
  assert.equal(pending.length, 2, "two target requests are in flight");
  pending[1].resolve({ ok: true, target: "a2_1.png" });        // newer first
  await tick();
  assert.equal(s._rtTarget, "a2_1.png");
  pending[0].resolve({ ok: true, target: "a_1.png" });         // older arrives late
  await tick();
  assert.equal(s._rtTarget, "a2_1.png", "the older target must not replace the newer tooltip");
});

test("an older error after a newer success is ignored", async () => {
  const A = makeLoader("A", { path: "a.png" });
  const s = makeSaver("s", "#A");
  makeGraph([A, s]);
  const pending = [];
  const targetFetch = () => { const d = deferred(); pending.push(d); return d.promise; };
  const channel = makeTargetChannel(s, targetFetch);
  s._rtRefreshTarget = () => channel.schedule(s);
  s._rtRefreshTarget();       // rev 1, fires at t=300
  advance(300);               // fetch 1 (older) starts, still in flight
  s._rtRefreshTarget();       // rev 2, fires at t=600
  advance(300);               // fetch 2 (newer) starts
  pending[1].resolve({ ok: true, target: "a2_1.png" });        // newer succeeds
  await tick();
  assert.equal(s._rtTarget, "a2_1.png");
  pending[0].reject(new Error("network down"));               // older fails late
  await tick();
  assert.equal(s._rtTarget, "a2_1.png", "the older error must not overwrite the newer success");
});

test("clearing the source while a fetch is pending shows the empty state", async () => {
  const A = makeLoader("A", { path: "a.png" });
  const sA = makeSaver("sA", "#A");
  makeGraph([A, sA]);
  const pending = [];
  const previewFetch = () => { const d = deferred(); pending.push(d); return d.promise; };
  const channel = makeCompareChannel(previewFetch);
  channel.schedule(A);        // rev 1, fires at t=400 (body: a.png)
  advance(400);               // the fetch starts and is still in flight
  // the user clears the source while the fetch is still in flight
  A.widgets.find((w) => w.name === "path").value = "";
  A.widgets.find((w) => w.name === "image").value = "";
  channel.schedule(A);        // rev 2, fires at t=800 (empty body -> empty state, no fetch)
  advance(400);               // the empty-state refresh runs
  await tick();               // flush the channel's async run (the empty-state write)
  const cmp = sA.widgets.find((w) => w.name === "comparer");
  assert.equal(cmp.writes.length, 1, "the cleared source wrote the empty state");
  assert.deepEqual(cmp.value.a, [], "the cleared source shows the empty state");
  assert.equal(cmp.imgA, null);
  const writesBefore = cmp.writes.length;
  pending[0].resolve({ filename: "a.png", type: "input" });   // the stale fetch finally answers
  await tick();
  assert.deepEqual(cmp.value.a, [], "the stale preview must not reappear after the source was cleared");
  assert.equal(cmp.writes.length, writesBefore, "the stale fetch wrote nothing");
});

test("no timers or revisions are attached to the nodes (workflow serialization)", async () => {
  const A = makeLoader("A", { path: "a.png" });
  const s = makeSaver("s", "#A");
  makeGraph([A, s]);
  const channel = makeCompareChannel(() => Promise.resolve(null));
  const target = makeTargetChannel(s, () => Promise.resolve({ ok: false, error: "x" }));
  channel.schedule(A);
  target.schedule(s);
  advance(500);
  await tick();
  for (const n of [A, s]) {
    for (const k of Object.keys(n)) {
      assert.ok(!/revision|timer|debounce|channel/i.test(k), `node property ${k} must not be serialization state`);
    }
    assert.ok(!("_revision" in n) && !("_timers" in n));
  }
});

// ---------------------------------------------------------------------------
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
if (failed) {
  console.error(`${failed} of ${tests.length} race tests failed`);
  process.exit(1);
}
console.log(`${tests.length} race tests passed`);
