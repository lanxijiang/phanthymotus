"""Voice selection must reuse MOSS engines and keep admitted speech stable."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

import plugins.tts as tts
from plugins.moss_tts import MossTTSAdapter
from test_moss_tts_adapter import FakeRuntime
from vision_stubs import _FakeExecutor


@pytest.fixture
def moss_plugin(tmp_path, monkeypatch):
    builds, closed = [], []

    def load_runtimes(model_dir, asset_url, instances):
        assert instances == 1
        runtime = FakeRuntime(f"runtime-{len(builds)}")
        pool = SimpleNamespace(close=lambda: closed.append(runtime))
        builds.append((runtime, pool))
        # Exercise the actual cached voice order instead of a hardcoded ID map.
        return (runtime,), ("Junhao", "Lingyu", "Ava"), pool

    monkeypatch.setattr(MossTTSAdapter, "_load_runtimes", staticmethod(load_runtimes))
    plugin = tts.TTSPlugin({
        "engine": "moss-multi", "device": "gpu", "speaker_id": 0,
        "moss_language": "zh", "warmup": False,
        "model_dir": str(tmp_path / "assets"),
    }, _FakeExecutor())
    assert plugin._impl is not None
    yield plugin, builds, closed
    plugin._impl._nodes.clear()
    plugin._impl._adapter.close()


def test_speaker_config_reuses_pool_nodes_and_changes_next_request(moss_plugin):
    plugin, builds, closed = moss_plugin
    impl = plugin._impl
    adapter = impl._adapter
    node = object()
    impl._nodes["existing"] = node
    assert plugin.synthesize_raw("Before.")

    result = plugin.dispatch("tts", {
        "action": "config", "tts_engine": "moss-multi", "speaker_id": "1",
    })
    assert result == {"status": "configured", "rebuilt": False, "engine": "moss-multi"}
    assert plugin._impl is impl and impl._adapter is adapter
    assert impl._nodes["existing"] is node
    assert len(builds) == 1 and not closed
    assert adapter._pool is builds[0][1]
    assert plugin.synthesize_raw("After.")
    assert [call["voice"] for call in builds[0][0].calls] == ["Junhao", "Lingyu"]

    plugin.dispatch("tts", {"action": "config", "speaker_id": 0})
    assert plugin.synthesize_raw("Back to zero.")
    assert builds[0][0].calls[-1]["voice"] == "Junhao"
    assert len(builds) == 1 and not closed


@pytest.mark.parametrize("invalid", [-1, 3, "unknown", 1.5, True, None, float("nan")])
@pytest.mark.parametrize("entry", ["facade", "implementation"])
def test_invalid_speaker_keeps_both_configs_voice_and_pool(moss_plugin, invalid, entry):
    plugin, builds, closed = moss_plugin
    impl = plugin._impl
    adapter = impl._adapter
    previous_outer, previous_inner = dict(plugin._cfg), dict(impl._cfg)
    target = plugin if entry == "facade" else impl
    with pytest.raises(ValueError):
        target.dispatch("tts", {"action": "config", "speaker_id": invalid})
    assert plugin._cfg == previous_outer and impl._cfg == previous_inner
    assert impl._adapter is adapter and len(builds) == 1 and not closed
    assert plugin.synthesize_raw("Still the original voice.")
    assert builds[0][0].calls[-1]["voice"] == "Junhao"


def test_waiting_request_keeps_voice_and_language_snapshot(moss_plugin):
    plugin, builds, closed = moss_plugin
    adapter = plugin._impl._adapter
    original_queue = adapter._available
    entered, proceed = threading.Event(), threading.Event()

    class GatedQueue:
        def get(self, **kwargs):
            entered.set()
            assert proceed.wait(3)
            return original_queue.get(**kwargs)

        def put_nowait(self, item):
            return original_queue.put_nowait(item)

    adapter._available = GatedQueue()
    with ThreadPoolExecutor(max_workers=1) as workers:
        request = workers.submit(plugin.synthesize_raw, "Already admitted.")
        try:
            assert entered.wait(3)
            # Producer has not started, so a late read of self._voice would
            # incorrectly observe the new speaker even for the first request.
            result = plugin.dispatch("tts", {
                "action": "config", "speaker_id": 2, "moss_language": "en",
            })
            assert result["rebuilt"] is False
            assert len(builds) == 1 and not closed
        finally:
            proceed.set()
        assert request.result(timeout=3)
    assert plugin.synthesize_raw("Next request.")
    assert [(call["voice"], call["language"]) for call in builds[0][0].calls] == [
        ("Junhao", "zh"), ("Ava", "en"),
    ]


def test_moss_other_session_keys_still_rebuild(moss_plugin, tmp_path):
    plugin, builds, _closed = moss_plugin
    old = plugin._impl._adapter
    try:
        result = plugin.dispatch("tts", {
            "action": "config", "model_dir": str(tmp_path / "changed"), "speaker_id": 2,
        })
        assert result["rebuilt"] is True and len(builds) == 2
        assert plugin._impl._adapter is not old
        assert plugin.synthesize_raw("New resident engine.")
        assert builds[1][0].calls[-1]["voice"] == "Ava"
    finally:
        old.close()


def test_moss_default_asset_path_matches_frame_release():
    assert tts.ENGINE_MODEL_DIRS["moss-multi"] == "/models/moss-frame-v4/assets"


@pytest.mark.parametrize("engine, has_setter", [("moss-multi", False), ("mms-th", True)])
def test_cheap_selection_requires_both_moss_and_the_capability(monkeypatch, engine, has_setter):
    builds, speaker_calls = [], []

    def build(cfg, **kwargs):
        adapter = SimpleNamespace()
        if has_setter:
            adapter.set_speaker_id = speaker_calls.append
        builds.append(adapter)
        return adapter

    monkeypatch.setattr(tts, "_build_tts_adapter", build)
    plugin = tts.SherpaOnnxTTSPlugin({
        "engine": engine, "speaker_id": 0, "warmup": False,
    }, _FakeExecutor())
    result = plugin.dispatch("tts", {"action": "config", "speaker_id": 1})
    assert result["rebuilt"] is True
    assert len(builds) == 2 and not speaker_calls
