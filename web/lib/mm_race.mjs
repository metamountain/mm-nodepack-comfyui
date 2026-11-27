// MM Nodepack – race-safe refresh channels (NO ComfyUI imports: the pure
// debounce/revision logic is unit-testable in Node, see tests/js/test_refresh_races.mjs;
// mm_nodepack.js wires the real /mm/image/* fetches and widgets to these channels).
//
// Every "edit -> debounced fetch -> write" flow (the compare slider's original
// preview, the live save-target tooltip) must never let an older async result
// overwrite a newer state:
//   * one debouncer + one revision counter per key (the node object), so
//     editing loader A and then loader B within the debounce window schedules
//     BOTH refreshes (one per key) – not only the last edit;
//   * the revision is bumped the moment an edit schedules a refresh, so a
//     cleared/changed source invalidates an outstanding fetch immediately;
//   * the revision is re-checked AFTER the fetch (and any other async work)
//     and BEFORE the write: a slower older preview never replaces a newer
//     original, an older target response never replaces a newer tooltip, and
//     an older error after a newer success is ignored.
//
// Timers and revision counters live in the channel state (WeakMap / closure),
// never on the node, so nothing is serialized into workflows.

export function debounce(fn, delay) {
  let timer = null;
  const wrapped = (...args) => {
    if (timer != null) clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn(...args); }, delay);
  };
  wrapped.cancel = () => {
    if (timer != null) {
      clearTimeout(timer);
      timer = null;
    }
  };
  return wrapped;
}

/**
 * One refresh channel.
 * `fetch(key)` is awaited after the debounce delay; whatever it resolves to
 * (or rejects with) is written by `write(key, payload)` ONLY when the key's
 * revision is still the one the fetch was scheduled with:
 *   payload = { ok: true, result } | { ok: false, error: string }
 */
export function createRefreshChannel({ delay = 0, fetch, write } = {}) {
  const states = new WeakMap();
  async function run(key, revision) {
    let payload;
    try {
      payload = { ok: true, result: await fetch(key) };
    } catch (e) {
      payload = { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
    const st = states.get(key);
    // a newer edit (or a cleared source) superseded this fetch: drop the
    // result, whatever it is (newer original, empty state, or error)
    if (!st || st.revision !== revision) return;
    write(key, payload);
  }
  function schedule(key) {
    let st = states.get(key);
    if (!st) {
      st = { revision: 0, timer: debounce((rev) => { void run(key, rev); }, delay) };
      states.set(key, st);
    }
    st.revision += 1;
    st.timer(st.revision);
  }
  return {
    schedule,
    /** current revision of the key (0 = never scheduled) */
    revision: (key) => (states.has(key) ? states.get(key).revision : 0),
  };
}
