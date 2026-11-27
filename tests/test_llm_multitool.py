"""Unit tests for the LLM Multitool MM node (port of the standalone reference
suite, adapted to the V3 node API). No GPU and no LLM server are needed: every
HTTP request is faked, and the local llama-server code paths are only touched
through the fakes and monkeypatches.

Run:
  $env:CUDA_VISIBLE_DEVICES='-1'; D:\\ComfyUI_windows_portable\\python_embeded\\python.exe -B -m pytest -p no:cacheprovider tests/ -q

The module under test imports comfy_api, so the whole suite is skipped in
interpreters without ComfyUI's site-packages.
"""
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
    def __init__(self, data=None, code=200, text=""):
        self._data, self.status_code, self.text = data, code, text or str(data)

    def json(self):
        return self._data

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

