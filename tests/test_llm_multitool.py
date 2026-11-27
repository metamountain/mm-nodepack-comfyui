"""Unit tests for the LLM Multitool MM node (port of the standalone reference
suite, adapted to the V3 node API). No GPU and no LLM server are needed: every
HTTP request is faked, and the local llama-server code paths are only touched
through the fakes and monkeypatches.

Run:
  $env:CUDA_VISIBLE_DEVICES='-1'; D:\\ComfyUI_windows_portable\\python_embeded\\python.exe -B -m pytest -p no:cacheprovider tests/ -q

The module under test imports comfy_api, so the whole suite is skipped in
interpreters without ComfyUI's site-packages.
"""
import asyncio
import json
import math
import os
import sys
import time

import numpy as np
import pytest
import requests

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# some interpreters have the legacy `py` library in site-packages (or a stale import
# of it in sys.modules): make sure it cannot shadow the `py/` package of this pack
sys.modules.pop("py", None)
sys.path.insert(0, REPO_ROOT)
# in --import-mode=importlib pytest does not put the tests/ directory on
# sys.path, so make conftest_helpers importable explicitly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from py import llm_multitool  # noqa: E402
    COMFY_AVAILABLE = True
except Exception:  # noqa: BLE001 – plain venv without ComfyUI's site-packages
    llm_multitool = None
    COMFY_AVAILABLE = False

needs_comfy = pytest.mark.skipif(
    not COMFY_AVAILABLE, reason="comfy_api not available in this interpreter")

# The node class, looked up rather than spelled, so a rename cannot break
# the suite.
Node = llm_multitool.NODE_LIST[0] if COMFY_AVAILABLE else None


@pytest.fixture(autouse=True)
def _clear_node_state():
    """The model scan cache is module-level: seed it so neither this test's
    define_schema() call nor a previous test's leftovers ever walk the disk,
    and no scan result can leak between tests."""
    if llm_multitool is not None:
        llm_multitool._local_models = {}
        llm_multitool._local_scan_t = time.time()
    yield


# -- fakes --------------------------------------------------------------------

class FakeTensor:
    """Stands in for a torch IMAGE batch [B,H,W,C]."""
    def __init__(self, b=1, h=8, w=8):
        self.arr = np.random.rand(b, h, w, 3).astype(np.float32)
        self.shape = self.arr.shape

    def __getitem__(self, i):
        a = self.arr[i]

        class _T:
            def cpu(self):
                return self

            def numpy(self):
                return a
        return _T()


class FakeResp:
    def __init__(self, data=None, code=200, text="", body=None):
        self._data, self.status_code = data, code
        self.text = text if text else ("" if body is not None else str(data))
        self._body = body          # raw body bytes (for capped reads); derived from data/json otherwise
        self.closed = False

    def _raw_body(self):
        if self._body is not None:
            return self._body
        if self._data is not None:
            return json.dumps(self._data).encode("utf-8")
        return self.text.encode("utf-8")

    def json(self):
        if self._body is not None:
            return json.loads(self._body.decode("utf-8"))
        return self._data

    def iter_content(self, chunk_size=None, decode_content=False):
        body = self._raw_body()
        size = chunk_size or 65536
        for i in range(0, len(body), size):
            yield body[i:i + size]

    def close(self):
        self.closed = True

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


@pytest.fixture
def net(monkeypatch):
    """Records POSTs; replies are configured via net['post'] / net['get']."""
    state = {"posts": [], "post": FakeResp({"choices": [{"message": {"content": "ok"}}]}),
             "get": {}}

    def fake_post(url, json=None, **kw):
        state["posts"].append((url, json))
        return state["post"]

    def fake_get(url, **kw):
        for suffix, resp in state["get"].items():
            if url.endswith(suffix):
                return resp
        raise requests.exceptions.ConnectionError(url)

    monkeypatch.setattr(llm_multitool.requests, "post", fake_post)
    monkeypatch.setattr(llm_multitool.requests, "get", fake_get)
    return state


def kw(**over):
    base = dict(
        api_url="http://127.0.0.1:8080", mode="text",
        system_prompt="sys", user_prompt="hi",
        temperature=1.0, top_p=0.95, top_k=64, min_p=0.0,
        repetition_penalty=1.1, presence_penalty=0.0,
        max_tokens=64, seed=-1, timeout_s=30, max_retries=0,
        enable_thinking=False, reasoning_effort="none",
        model_id="m", backend="llama.cpp", api_key_env="", local_model=llm_multitool.SERVER_CHOICE,
        clean_output=False, image_mode="single", unload_after=False, keep_alive=0,
        extra_body="",
    )
    base.update(over)
    return base


def run(**over):
    """One execute() call. The V3 node's execute is a classmethod, so the
    reference suite's instance-method call maps to a direct class call; the
    NodeOutput exposes its result tuple via .result exactly like the old
    {"result": ...} dict."""
    return Node.execute(**kw(**over)).result


# -- helpers ------------------------------------------------------------------

@needs_comfy
def test_clean_keeps_normal_words():
    assert llm_multitool._clean_response("I think the weekend is fine") == "I think the weekend is fine"


@needs_comfy
def test_clean_removes_think_and_markdown():
    assert llm_multitool._clean_response("<think>abc</think>Hello **world**") == "Hello world"


@needs_comfy
def test_clean_drops_orphan_closing_tag():
    assert llm_multitool._clean_response("reasoning here</think> Final") == "Final"


@needs_comfy
def test_normalize_url():
    assert llm_multitool._normalize_url("127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert llm_multitool._normalize_url("") == "http://127.0.0.1:8080"
    assert llm_multitool._normalize_url("https://api.openai.com/v1") == "https://api.openai.com"


@needs_comfy
def test_pick_router_model():
    pick = llm_multitool._pick_router_model
    assert pick([{"id": "a", "status": {"value": "unloaded"}},
                 {"id": "b", "status": {"value": "sleeping"}}]) == "b"
    assert pick([{"id": "a", "status": {"value": "unloaded"}}]) is None
    assert pick([{"id": "x"}]) == "x"
    assert pick([]) is None


@needs_comfy
def test_fingerprint_inputs():
    v = Node.fingerprint_inputs(seed=-1)
    assert isinstance(v, float) and math.isnan(v)
    assert Node.fingerprint_inputs(seed=3) == ""


# -- requests -----------------------------------------------------------------

@needs_comfy
def test_ollama_vision_payload(net):
    net["post"] = FakeResp({"message": {"content": "hi", "thinking": "t"}})
    res = run(backend="ollama", mode="vision", image=FakeTensor(), seed=5,
              enable_thinking=True, api_url="http://127.0.0.1:11434")
    url, payload = net["posts"][-1]
    assert url.endswith("/api/chat")
    assert "images" not in payload
    assert len(payload["messages"][1]["images"]) == 1
    assert payload["options"]["seed"] == 5
    assert payload["think"] is True
    assert res == ("hi", "t", "m", "vision")


@needs_comfy
def test_router_auto_picks_loaded_model(net):
    net["get"]["/v1/models"] = FakeResp({"data": [
        {"id": "a", "status": {"value": "unloaded"}},
        {"id": "b", "status": {"value": "sleeping"}}]})
    run(model_id="auto")
    assert net["posts"][-1][1]["model"] == "b"


@needs_comfy
def test_empty_response_return_text(net):
    net["post"] = FakeResp({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    res = run(on_error="return text")
    assert res[0].startswith("ERROR: Empty response") and "max_tokens" in res[0]


@needs_comfy
def test_empty_response_stops_by_default(net):
    net["post"] = FakeResp({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
    with pytest.raises(RuntimeError):
        run()
    with pytest.raises(RuntimeError):
        run(on_error="stop")


@needs_comfy
def test_vision_400_mentions_mmproj(net):
    net["post"] = FakeResp({"error": "x"}, code=400, text="image input is not supported")
    res = run(mode="vision", image=FakeTensor(), on_error="return text")
    assert "mmproj" in res[0]


@needs_comfy
def test_url_with_v1_is_fixed(net):
    run(api_url="127.0.0.1:8080/v1")
    assert net["posts"][-1][0] == "http://127.0.0.1:8080/v1/chat/completions"


# -- unload button --------------------------------------------------------------

@needs_comfy
def test_unload_router_only_loaded_models(net, monkeypatch):
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: None)
    net["post"] = FakeResp({"success": True})
    net["get"]["/v1/models"] = FakeResp({"data": [
        {"id": "a", "status": {"value": "unloaded"}},
        {"id": "b", "status": {"value": "loaded"}}]})
    done, msg = llm_multitool._unload_all(api_url="127.0.0.1:8080", backend="llama.cpp")
    assert done and "b" in msg
    assert net["posts"] == [("http://127.0.0.1:8080/models/unload", {"model": "b"})]


@needs_comfy
def test_unload_plain_llama_server_only_notes(net, monkeypatch):
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: None)
    net["get"]["/v1/models"] = FakeResp({"data": [{"id": "x"}]})
    done, msg = llm_multitool._unload_all(api_url="http://h:8080", backend="llama.cpp")
    assert not done and "no unload API" in msg and net["posts"] == []


@needs_comfy
def test_unload_ollama_keep_alive_0(net, monkeypatch):
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: None)
    net["post"] = FakeResp({})
    net["get"]["/api/ps"] = FakeResp({"models": [{"name": "llava"}]})
    done, msg = llm_multitool._unload_all(api_url="http://h:11434", backend="ollama")
    assert done
    assert net["posts"] == [("http://h:11434/api/generate", {"model": "llava", "keep_alive": 0})]


@needs_comfy
def test_unload_openai_does_nothing(net, monkeypatch):
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: None)
    done, msg = llm_multitool._unload_all(api_url="https://api.openai.com/v1", backend="openai")
    assert not done and "nothing to unload" in msg and net["posts"] == []


# -- model scan -----------------------------------------------------------------

@needs_comfy
def test_scan_lists_text_only_and_skips_non_chat(tmp_path, monkeypatch):
    vis, txt = tmp_path / "vis", tmp_path / "txt"
    vis.mkdir()
    txt.mkdir()
    for f in ("Model-A-Q4.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"):
        (vis / f).write_bytes(b"x")
    for f in ("Chat-B-Q4.gguf", "t5-xxl-encoder-Q8.gguf", "X-DFlash2-Q4.gguf", "mtp-head.gguf"):
        (txt / f).write_bytes(b"x")
    monkeypatch.setattr(llm_multitool, "_model_dirs", lambda: [str(tmp_path)])
    found = llm_multitool._scan_local_models()
    assert set(found) == {"Model-A-Q4", "Chat-B-Q4 (text)"}
    assert found["Model-A-Q4"][1].endswith("mmproj-F16.gguf")
    assert found["Chat-B-Q4 (text)"][1] is None


@needs_comfy
def test_text_only_model_rejects_image(net):
    res = run(local_model="Chat-B-Q4 (text)", mode="vision", image=FakeTensor(), on_error="return text")
    assert "text only" in res[0] and net["posts"] == []


# -- schema ---------------------------------------------------------------------

@needs_comfy
def test_schema_id_name_category_and_aliases():
    schema = Node.define_schema()
    assert schema.node_id == "MMLLMMultitool"
    assert schema.display_name == "LLM Multitool MM"
    assert schema.category == llm_multitool.CATEGORY
    assert schema.search_aliases == ["mm", "metamountain", "mm nodepack",
                                     "llm mm", "vision mm", "vlm"]


@needs_comfy
def test_schema_input_and_output_contracts():
    schema = Node.define_schema()
    assert [i.id for i in schema.inputs] == [
        "local_model", "api_url", "backend", "model_id", "api_key_env",
        "mode", "system_prompt", "user_prompt",
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
        "presence_penalty", "max_tokens", "seed",
        "enable_thinking", "reasoning_effort",
        "clean_output", "image_mode", "unload_after", "keep_alive",
        "on_error", "timeout_s", "max_retries", "extra_body",
        "image", "text"]
    assert [o.id for o in schema.outputs] == ["response", "thinking",
                                              "active_model", "active_mode"]
    optional = {i.id for i in schema.inputs if i.optional}
    assert optional == {"image", "text"}


@needs_comfy
def test_inputs_have_no_legacy_fields():
    schema = Node.define_schema()
    names = {i.id for i in schema.inputs}
    assert not {"api_key", "_header_system_prompt", "_header_user_prompt"} & names
    assert "beellama" not in llm_multitool.BACKENDS
    assert [i.id for i in schema.inputs][0] == "local_model"


# -- shared-server policy: profiles ------------------------------------------
# Dummy profile + dummy credential + in-memory transport (the net fixture or a
# hand-rolled recorder) + mocked lifecycle functions: no real process or
# network is ever touched.

KEY_ENV = "MM_TEST_LLM_KEY"
KEY_VAL = "dummy-secret-key"


def dummy_profile(**over):
    p = {
        "id": "dummy-ninfer",
        "backends": ["llama.cpp"],
        "urls": ["http://127.0.0.1:8080"],
        "api_key_env": KEY_ENV,
        "allow_unload": True,
        "allow_local_unload": True,
        "allow_local_model": True,
        "local_models": ["dummy-local"],
        "max_response_bytes": 4096,
    }
    p.update(over)
    return p


_set_profiles_n = 0


def set_profiles(monkeypatch, tmp_path, profiles, shared=True):
    """Write a dummy profile file, point MM_NODEPACK_LLM_PROFILES at it and
    (optionally) force the shared-server switch; reset the profile cache.
    Each call gets its own file name so the mtime-keyed cache can never
    confuse two generations of the file within one test."""
    global _set_profiles_n
    _set_profiles_n += 1
    f = tmp_path / f"profiles_{_set_profiles_n}.json"
    f.write_text(json.dumps({"profiles": profiles}), encoding="utf-8")
    monkeypatch.setenv(llm_multitool._PROFILE_ENV, str(f))
    if shared:
        monkeypatch.setenv("MM_NODEPACK_SHARED_SERVER", "1")
    else:
        monkeypatch.delenv("MM_NODEPACK_SHARED_SERVER", raising=False)
    monkeypatch.setattr(llm_multitool, "_profile_cache",
                        {"path": None, "mtime": None, "profiles": None})
    return str(f)


def _block_transport(monkeypatch):
    """Transport recorder that fails the test if any request goes out."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("transport must not be called before authorization")

    monkeypatch.setattr(llm_multitool.requests, "get", boom)
    monkeypatch.setattr(llm_multitool.requests, "post", boom)
    return calls


# -- node execution (queued): policy active ----------------------------------

@needs_comfy
def test_policy_approved_discovery_and_generation(net, monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    net["get"]["/v1/models"] = FakeResp({"data": [{"id": "m"}]})
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp", api_key_env=KEY_ENV)
    assert res == ("ok", "", "m", "text")
    url, payload = net["posts"][-1]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    assert payload["model"] == "m"


@needs_comfy
def test_policy_approved_retries_then_succeeds(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    monkeypatch.setattr(llm_multitool.time, "sleep", lambda s: None)
    posts = []

    def fake_post(url, json=None, **kw):
        posts.append((url, json, kw.get("headers")))
        if len(posts) == 1:
            return FakeResp({}, code=503)
        return FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm_multitool.requests, "post", fake_post)
    monkeypatch.setattr(llm_multitool.requests, "get",
                        lambda url, **kw: FakeResp({"data": [{"id": "m"}]}))
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env=KEY_ENV, max_retries=2, on_error="return text")
    assert res[0] == "ok"
    assert len(posts) == 2
    assert posts[0][2].get("Authorization") == f"Bearer {KEY_VAL}"


@needs_comfy
def test_policy_unapproved_url_rejects_before_transport(monkeypatch, tmp_path):
    calls = _block_transport(monkeypatch)
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    res = run(api_url="http://127.0.0.1:9999", backend="llama.cpp",
              api_key_env=KEY_ENV, on_error="return text")
    assert res[0].startswith("ERROR:") and "no approved LLM profile" in res[0]
    assert "127.0.0.1:9999" in res[0]          # redacted host:port is reported
    assert KEY_VAL not in res[0]                # ...but never the key value
    assert calls["n"] == 0


@needs_comfy
def test_policy_unapproved_backend_or_credential_var_rejects(monkeypatch, tmp_path):
    calls = _block_transport(monkeypatch)
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    res = run(api_url="http://127.0.0.1:8080", backend="openai",
              api_key_env=KEY_ENV, on_error="return text")
    assert "no approved LLM profile" in res[0]
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env="SOME_OTHER_KEY", on_error="return text")
    assert "no approved LLM profile" in res[0]
    assert calls["n"] == 0


@needs_comfy
def test_policy_shared_server_without_profiles_fails_closed(monkeypatch, tmp_path):
    calls = _block_transport(monkeypatch)
    set_profiles(monkeypatch, tmp_path, None)   # shared on, no profile file
    monkeypatch.setenv(llm_multitool._PROFILE_ENV, "")
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env="", on_error="return text")
    assert res[0].startswith("ERROR:") and "no LLM profiles configured" in res[0]
    assert calls["n"] == 0


@needs_comfy
def test_policy_oversized_response_controlled_error(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])     # cap 4096
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    posts = []

    def fake_post(url, json=None, **kw):
        posts.append((url, json))
        return FakeResp(body=b"x" * 100000)

    monkeypatch.setattr(llm_multitool.requests, "post", fake_post)
    monkeypatch.setattr(llm_multitool.requests, "get",
                        lambda url, **kw: FakeResp({"data": [{"id": "m"}]}))
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env=KEY_ENV, on_error="return text")
    assert res[0].startswith("ERROR:") and "exceeds the 4096 byte limit" in res[0]
    assert len(posts) == 1


@needs_comfy
def test_policy_oversized_error_body_stays_bounded(monkeypatch, tmp_path):
    """A huge 400 body must not leak into the (bounded) error message."""
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    monkeypatch.setattr(llm_multitool.requests, "post",
                        lambda url, json=None, **kw: FakeResp(code=400, body=b"Y" * 100000))
    monkeypatch.setattr(llm_multitool.requests, "get",
                        lambda url, **kw: FakeResp({"data": [{"id": "m"}]}))
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env=KEY_ENV, on_error="return text")
    assert res[0].startswith("ERROR: 400 Bad request")
    assert "Y" * 1000 not in res[0]
    assert len(res[0]) < 1000


@needs_comfy
def test_policy_redirect_with_credentials_not_followed(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    posts = []

    def fake_post(url, json=None, **kw):
        posts.append((url, kw.get("allow_redirects")))
        return FakeResp({}, code=302, text="moved")

    monkeypatch.setattr(llm_multitool.requests, "post", fake_post)
    monkeypatch.setattr(llm_multitool.requests, "get",
                        lambda url, **kw: FakeResp({"data": [{"id": "m"}]}))
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env=KEY_ENV, on_error="return text")
    assert res[0].startswith("ERROR:")
    assert len(posts) == 1                 # the 302 hop was never followed
    assert posts[0][1] is False            # allow_redirects disabled for credentialed calls


# -- node execution (queued): local model lifecycle ---------------------------

@needs_comfy
def test_policy_local_model_approved_starts_mocked_server(net, monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    starts = []
    monkeypatch.setattr(llm_multitool, "_ensure_local_server",
                        lambda label: starts.append(label) or (llm_multitool.LOCAL_URL, "alias", None))
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(local_model="dummy-local", mode="text")
    assert res[0] == "ok" and starts == ["dummy-local"]
    assert net["posts"][-1][0] == llm_multitool.LOCAL_URL + "/v1/chat/completions"


@needs_comfy
def test_policy_local_model_denied_leaves_process_mock_untouched(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile(allow_local_model=False, local_models=None)])
    starts = []
    monkeypatch.setattr(llm_multitool, "_ensure_local_server",
                        lambda label: starts.append(label) or (llm_multitool.LOCAL_URL, "a", None))
    monkeypatch.setattr(llm_multitool, "_stop_local_server", lambda: starts.append("stop"))
    _block_transport(monkeypatch)
    res = run(local_model="dummy-local", mode="text", on_error="return text")
    assert res[0].startswith("ERROR:") and "not approved" in res[0]
    assert starts == []


@needs_comfy
def test_policy_local_model_label_not_in_profile_list_rejects(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path,
                 [dummy_profile(allow_local_model=True, local_models=["other-model"])])
    starts = []
    monkeypatch.setattr(llm_multitool, "_ensure_local_server",
                        lambda label: starts.append(label) or (llm_multitool.LOCAL_URL, "a", None))
    _block_transport(monkeypatch)
    res = run(local_model="dummy-local", mode="text", on_error="return text")
    assert "not approved" in res[0] and starts == []


@needs_comfy
def test_policy_local_run_needs_no_credential(monkeypatch, tmp_path):
    """A profile credential variable need not exist for local-model runs:
    the managed llama-server takes no API key."""
    set_profiles(monkeypatch, tmp_path, [dummy_profile(local_models=None)])
    monkeypatch.delenv(KEY_ENV, raising=False)
    starts = []
    monkeypatch.setattr(llm_multitool, "_ensure_local_server",
                        lambda label: starts.append(label) or (llm_multitool.LOCAL_URL, "a", None))
    monkeypatch.setattr(llm_multitool.requests, "post",
                        lambda url, json=None, **kw: FakeResp({"choices": [{"message": {"content": "ok"}}]}))

    def no_get(url, **kw):
        raise AssertionError("a local-model run must not query any remote server")

    monkeypatch.setattr(llm_multitool.requests, "get", no_get)
    res = run(local_model="dummy-local", mode="text")
    assert res[0] == "ok" and starts == ["dummy-local"]


@needs_comfy
def test_policy_unload_after_skipped_without_permission(net, monkeypatch, tmp_path):
    """Generation succeeds; the follow-up /models/unload must never go out."""
    set_profiles(monkeypatch, tmp_path, [dummy_profile(allow_unload=False, allow_local_unload=False)])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    net["get"]["/v1/models"] = FakeResp({"data": [{"id": "m"}]})
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env=KEY_ENV, unload_after=True)
    assert res[0] == "ok"
    assert [u for u, _ in net["posts"] if u.endswith("/models/unload")] == []


@needs_comfy
def test_policy_local_model_options_are_filtered(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path,
                 [dummy_profile(allow_local_model=True, local_models=["Model-A"])])
    monkeypatch.setattr(llm_multitool, "_local_models",
                       {"Model-A": ("/a.gguf", None), "Chat-B": ("/b.gguf", None)})
    monkeypatch.setattr(llm_multitool, "_local_scan_t", time.time())
    opts = llm_multitool._local_model_options()
    assert opts == [llm_multitool.SERVER_CHOICE, "Model-A"]
    # fail closed: shared server, no profile file
    set_profiles(monkeypatch, tmp_path, None)
    assert llm_multitool._local_model_options() == [llm_multitool.SERVER_CHOICE]
    # private server: everything stays listed
    monkeypatch.setenv("MM_NODEPACK_SHARED_SERVER", "0")
    assert llm_multitool._local_model_options() == [llm_multitool.SERVER_CHOICE, "Model-A", "Chat-B"]


# -- extra_body reserved fields (applied on every server) ---------------------

@needs_comfy
def test_policy_unapproved_error_redacts_credential_url():
    host = llm_multitool._redacted_host("http://user:pass@127.0.0.1:8080/v1/")
    assert host == "127.0.0.1:8080"
    assert "user" not in host and "pass" not in host


@needs_comfy
def test_policy_unload_after_dispatched_with_permission(net, monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp",
              api_key_env=KEY_ENV, unload_after=True)
    assert res[0] == "ok"
    assert ("http://127.0.0.1:8080/models/unload", {"model": "m"}) in net["posts"]


def _shared_extra(monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [
        dummy_profile(),
        dummy_profile(id="dummy-ollama", backends=["ollama"], urls=["http://127.0.0.1:11434"])])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    return {"api_key_env": KEY_ENV, "on_error": "return text"}


@needs_comfy
def test_extra_body_reserved_model_rejected(net, monkeypatch, tmp_path):
    k = _shared_extra(monkeypatch, tmp_path)
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp", extra_body='{"model": "hacker"}', **k)
    assert res[0].startswith("ERROR:") and "model" in res[0] and net["posts"] == []


@needs_comfy
def test_extra_body_stream_and_urlish_and_nested_rejected(net, monkeypatch, tmp_path):
    k = _shared_extra(monkeypatch, tmp_path)
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp", extra_body='{"stream": true}', **k)
    assert "stream" in res[0] and net["posts"] == []
    res = run(api_url="http://127.0.0.1:8080", backend="llama.cpp", extra_body='{"proxy_url": "http://x"}', **k)
    assert "proxy_url" in res[0] and net["posts"] == []
    res = run(backend="ollama", api_url="http://127.0.0.1:11434",
              extra_body='{"options": {"num_predict": 8}}', **k)
    assert "num_predict" in res[0] and net["posts"] == []


@needs_comfy
def test_extra_body_reserved_fields_allowed_on_private_server(net):
    # policy inactive: the old behavior is unchanged (no reserved-field gate)
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(extra_body='{"max_tokens": 64}')
    assert res[0] == "ok"
    assert net["posts"][-1][1]["max_tokens"] == 64


@needs_comfy
def test_policy_local_unload_after_needs_permission(net, monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile(allow_local_unload=False)])
    stopped = []
    monkeypatch.setattr(llm_multitool, "_stop_local_server", lambda: stopped.append(1))
    monkeypatch.setattr(llm_multitool, "_ensure_local_server",
                        lambda label: ("http://127.0.0.1:8081", "dummy-local", None))
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(local_model="dummy-local", unload_after=True)
    assert res[0] == "ok" and stopped == []


@needs_comfy
def test_extra_body_supported_extension_still_merges(net):
    net["post"] = FakeResp({"choices": [{"message": {"content": "ok"}}]})
    res = run(extra_body='{"cache_prompt": true, "top_k": 20}')
    assert res[0] == "ok"
    _, payload = net["posts"][-1]
    assert payload["cache_prompt"] is True and payload["top_k"] == 20


# -- /mm/llm/unload route ------------------------------------------------------

class _FakeUnloadRequest:
    def __init__(self, remote, body=None, malformed=False):
        self.remote = remote
        self._body, self._malformed = body, malformed

    async def json(self, content_type=None):
        if self._malformed:
            raise ValueError("malformed JSON body")
        return self._body


class _RouteTable:
    def __init__(self):
        self.handlers = {}

    def post(self, path):
        def deco(fn):
            self.handlers[path] = fn
            return fn
        return deco


@pytest.fixture
def unload_route(monkeypatch):
    """Register /mm/llm/unload against a stub `server` module and hand the
    async handler back (aiohttp is part of the ComfyUI interpreter)."""
    pytest.importorskip("aiohttp")
    import types
    ps = types.SimpleNamespace(routes=_RouteTable())
    mod = types.ModuleType("server")
    mod.PromptServer = type("PromptServerStub", (), {"instance": ps})
    monkeypatch.setitem(sys.modules, "server", mod)
    llm_multitool.register_routes()
    return ps.routes.handlers["/mm/llm/unload"]


def _post_unload(handler, remote, body=None, malformed=False):
    return asyncio.run(handler(_FakeUnloadRequest(remote, body, malformed)))


def _json_of(resp):
    return json.loads(resp.body)


@needs_comfy
def test_route_unload_malformed_json_is_400_before_dispatch(unload_route, monkeypatch):
    dispatch = []
    monkeypatch.setattr(llm_multitool, "_unload_all",
                        lambda **kw: dispatch.append(kw) or (False, "x"))
    resp = _post_unload(unload_route, "127.0.0.1", malformed=True)
    assert resp.status == 400 and not dispatch
    resp = _post_unload(unload_route, "127.0.0.1", body=["not", "an", "object"])
    assert resp.status == 400 and not dispatch
    resp = _post_unload(unload_route, "127.0.0.1", body=None)
    assert resp.status == 400 and not dispatch


@needs_comfy
def test_route_unload_remote_forced_is_profile_bound(unload_route, monkeypatch, tmp_path):
    """Non-local request on a PRIVATE server: profile gate still applies."""
    set_profiles(monkeypatch, tmp_path, [dummy_profile()], shared=False)
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: None)
    calls = {"get": 0, "post": 0}

    def fake_get(url, **kw):
        calls["get"] += 1
        return FakeResp({"data": [{"id": "m", "status": {"value": "loaded"}}]})

    def fake_post(url, json=None, **kw):
        calls["post"] += 1
        return FakeResp({"ok": True})

    monkeypatch.setattr(llm_multitool.requests, "get", fake_get)
    monkeypatch.setattr(llm_multitool.requests, "post", fake_post)
    body = {"api_url": "http://127.0.0.1:8080", "backend": "llama.cpp", "api_key_env": KEY_ENV}
    resp = _post_unload(unload_route, "203.0.113.7", body)
    assert resp.status == 200 and _json_of(resp)["ok"] is True
    assert calls["get"] == 1 and calls["post"] == 1
    # unapproved target: 403, transport untouched
    calls["get"] = calls["post"] = 0
    bad = dict(body, api_url="http://127.0.0.1:9999")
    resp = _post_unload(unload_route, "203.0.113.7", bad)
    assert resp.status == 403 and calls["get"] == 0 and calls["post"] == 0


@needs_comfy
def test_route_unload_local_stop_needs_profile_permission(unload_route, monkeypatch, tmp_path):
    set_profiles(monkeypatch, tmp_path, [dummy_profile()])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: "alias")
    stops, remotes = [], []
    monkeypatch.setattr(llm_multitool, "_stop_local_server", lambda: stops.append(1))
    monkeypatch.setattr(llm_multitool, "_unload_remote",
                        lambda *a, **k: remotes.append((a, k)) or ([], ""))
    body = {"api_url": "http://127.0.0.1:8080", "backend": "llama.cpp", "api_key_env": KEY_ENV}
    resp = _post_unload(unload_route, "127.0.0.1", body)
    assert resp.status == 200 and stops == [1] and len(remotes) == 1
    # same target, profile without any unload permission: 403, nothing stopped, no remote call
    set_profiles(monkeypatch, tmp_path,
                 [dummy_profile(allow_unload=False, allow_local_unload=False)])
    stops.clear()
    remotes.clear()
    resp = _post_unload(unload_route, "127.0.0.1", body)
    assert resp.status == 403 and stops == [] and remotes == []


@needs_comfy
def test_route_unload_remote_forbidden_by_profile_only_skips_remote(unload_route, monkeypatch, tmp_path):
    """allow_local_unload without allow_unload: the own server may be stopped,
    the remote URL may not be contacted."""
    set_profiles(monkeypatch, tmp_path, [dummy_profile(allow_unload=False, allow_local_unload=True)])
    monkeypatch.setenv(KEY_ENV, KEY_VAL)
    monkeypatch.setattr(llm_multitool, "_local_served", lambda: "alias")
    monkeypatch.setattr(llm_multitool, "_stop_local_server", lambda: None)
    calls = _block_transport(monkeypatch)
    body = {"api_url": "http://127.0.0.1:8080", "backend": "llama.cpp", "api_key_env": KEY_ENV}
    resp = _post_unload(unload_route, "127.0.0.1", body)
    assert resp.status == 200
    assert calls["n"] == 0



@needs_comfy
def test_device_args_place_layers():
    assert llm_multitool._device_args("auto") == ["--fit", "on"]
    assert llm_multitool._device_args("gpu") == ["--n-gpu-layers", "99", "--fit", "off"]
    assert llm_multitool._device_args("cpu") == ["--n-gpu-layers", "0", "--fit", "off"]
    assert llm_multitool._device_args("bogus") == ["--fit", "on"]
    assert llm_multitool.LOCAL_DEVICE in llm_multitool._DEVICES


def _write_gguf(path, kv):
    """Minimal GGUF v3 header: no tensors, the given (key, type, value) pairs."""
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, t, v in kv:
        out += s(key) + struct.pack("<I", t)
        if t == 4:
            out += struct.pack("<I", v)
        elif t == 8:
            out += s(v)
        elif t == 9:   # array: (elem_type, values)
            et, vals = v
            out += struct.pack("<IQ", et, len(vals))
            for e in vals:
                out += s(e) if et == 8 else struct.pack("<I", e)
    path.write_bytes(out + b"\0" * 1000)


@needs_comfy
def test_llm_vram_need_reads_kv_shape(tmp_path):
    p = tmp_path / "m.gguf"
    _write_gguf(p, [
        ("general.architecture", 8, "llama"),
        ("tokenizer.ggml.tokens", 9, (8, ["a", "b", "c"])),
        ("llama.block_count", 4, 32),
        ("llama.attention.head_count", 4, 32),
        ("llama.attention.head_count_kv", 9, (4, [8, 4, 8])),   # per-layer array -> max
        ("llama.embedding_length", 4, 4096),
    ])
    meta = llm_multitool._gguf_meta(str(p), ("block_count", "attention.head_count_kv"))
    assert meta == {"block_count": 32, "attention.head_count_kv": 8}
    need = llm_multitool._llm_vram_need(str(p), "", ctx=16384)
    kv = 16384 * 32 * 8 * (128 + 128) * 2
    assert need == p.stat().st_size + kv + llm_multitool._COMPUTE_MARGIN


@needs_comfy
def test_llm_vram_need_falls_back_without_header(tmp_path):
    p = tmp_path / "broken.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 4000)
    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"\0" * 1000)
    weights = 4004 + 1000
    assert llm_multitool._llm_vram_need(str(p), str(proj)) == weights + weights // 4 + llm_multitool._COMPUTE_MARGIN


def _fake_mm(monkeypatch, free_seq):
    import types
    calls = []
    seq = list(free_seq)
    fake = types.SimpleNamespace(
        get_torch_device=lambda: types.SimpleNamespace(type="cuda"),
        get_free_memory=lambda dev=None: seq.pop(0),
        free_memory=lambda need, dev, **kw: calls.append(("free", need)),
        soft_empty_cache=lambda force=False: calls.append(("empty",)),
    )
    pkg = types.ModuleType("comfy")
    pkg.model_management = fake
    monkeypatch.setitem(sys.modules, "comfy", pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake)
    return calls


@needs_comfy
def test_make_room_frees_only_when_short(monkeypatch):
    calls = _fake_mm(monkeypatch, [2 * 2**30, 9 * 2**30])
    assert llm_multitool._make_room(8 * 2**30) == (2 * 2**30, 9 * 2**30)
    assert calls == [("free", 8 * 2**30), ("empty",)]
    calls = _fake_mm(monkeypatch, [12 * 2**30, 12 * 2**30])
    assert llm_multitool._make_room(8 * 2**30) == (12 * 2**30, 12 * 2**30)
    assert calls == []   # enough room: ComfyUI keeps its models


@needs_comfy
def test_make_room_without_comfy(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    assert llm_multitool._make_room(1) is None
