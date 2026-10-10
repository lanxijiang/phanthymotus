"""Fused graph routing must preserve controls, RNG ownership and provenance."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.moss_deploy.runtime import (
    _make_frame_generation_method, _run_sampled_frame, _validate_frame_sampling,
)


CONTROLS = dict(text_temperature=1.5, text_top_p=1.0, text_top_k=50,
                audio_temperature=1.7, audio_top_p=.8, audio_top_k=25,
                audio_repetition_penalty=1.0)
SAMPLE_MODE_FIXED = "fixed"


def generation_branch(self, request_rows, on_frame=None):
    # Same branch predicate as the original generator, isolated from model I/O.
    defaults = self.manifest["generation_defaults"]
    if "local_fixed_sampled_frame" in self.sessions and defaults["sample_mode"] == SAMPLE_MODE_FIXED:
        return "fused", request_rows, on_frame
    return "cached", request_rows, on_frame


def test_fused_branch_keeps_public_full_mode_and_source_globals_private():
    runtime = SimpleNamespace(
        manifest={"generation_defaults": {**CONTROLS, "sample_mode": "full", "do_sample": True}},
        sessions={"local_fixed_sampled_frame": object()},
    )
    method = _make_frame_generation_method(generation_branch, {"generation_defaults": CONTROLS})
    rows, callback = object(), object()
    assert method(runtime, rows, callback) == ("fused", rows, callback)
    assert generation_branch(runtime, rows, callback) == ("cached", rows, callback)
    assert generation_branch.__globals__["SAMPLE_MODE_FIXED"] == "fixed"
    assert runtime.manifest["generation_defaults"]["sample_mode"] == "full"


@pytest.mark.parametrize("name", list(CONTROLS) + ["sample_mode", "do_sample"])
def test_fused_parameter_mismatch_fails_before_generation(name):
    defaults = {**CONTROLS, "sample_mode": "full", "do_sample": True}
    defaults[name] = "fixed" if name == "sample_mode" else False if name == "do_sample" else defaults[name] + .01
    runtime = SimpleNamespace(manifest={"generation_defaults": defaults},
                              sessions={"local_fixed_sampled_frame": object()})
    method = _make_frame_generation_method(generation_branch, {"generation_defaults": CONTROLS})
    with pytest.raises(ValueError, match="sampling does not match"):
        method(runtime, {})


def test_frame_limits_are_not_sampling_constants_and_missing_session_fails():
    defaults = {**CONTROLS, "sample_mode": "full", "do_sample": True, "max_new_frames": 999}
    _validate_frame_sampling(defaults, {"generation_defaults": CONTROLS})
    method = _make_frame_generation_method(generation_branch, {"generation_defaults": CONTROLS})
    runtime = SimpleNamespace(manifest={"generation_defaults": defaults}, sessions={})
    with pytest.raises(RuntimeError, match="no matching session"):
        method(runtime, {})


class FrameSession:
    def __init__(self, continuation=1):
        self.outputs = [np.array([[continuation]], dtype=np.int32),
                        np.arange(16, dtype=np.int32)[None],
                        np.zeros((1, 2), dtype=np.float32),
                        np.zeros((1, 16, 1024), dtype=np.float32)]
        self.feeds = []

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in
                ("should_continue", "frame_token_ids", "text_logits", "audio_logits")]

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return self.outputs


def runtime_for(session, seed=7):
    return SimpleNamespace(manifest={"tts_config": {"n_vq": 16}},
                           sessions={"local_fixed_sampled_frame": session},
                           rng=np.random.default_rng(seed))


def run_frame(runtime):
    return _run_sampled_frame(runtime, np.zeros((1, 768), dtype=np.float32),
                              previous_token_sets_by_channel=[set() for _ in range(16)])


@pytest.mark.parametrize("continues,consumed", [(1, 17), (0, 1)])
def test_frame_and_stop_consume_the_same_draws_as_host_generation(continues, consumed):
    session = FrameSession(continues)
    runtime = runtime_for(session)
    expected = np.random.default_rng(7)
    draws = expected.random(17)
    result = run_frame(runtime)
    assert result == (bool(continues), list(range(16)))
    assert set(session.feeds[0]) == {"global_hidden", "random_u"}
    np.testing.assert_array_equal(session.feeds[0]["random_u"],
                                  np.clip(draws.astype(np.float32)[None], 0., .99999994))
    assert session.feeds[0]["random_u"].dtype == np.float32
    expected = np.random.default_rng(7)
    expected.random(consumed)
    assert runtime.rng.bit_generator.state == expected.bit_generator.state


def test_stop_on_one_runtime_cannot_advance_another_runtime_rng():
    first, second = runtime_for(FrameSession(0)), runtime_for(FrameSession(1))
    untouched = copy.deepcopy(second.rng.bit_generator.state)
    run_frame(first)
    assert second.rng.bit_generator.state == untouched
    run_frame(second)
    expected = np.random.default_rng(7)
    expected.random(17)
    assert second.rng.bit_generator.state == expected.bit_generator.state


@pytest.mark.parametrize("field,value", [
    (0, np.array([[1.0]], dtype=np.float32)), (0, np.array([1], dtype=np.int32)),
    (0, np.array([[2]], dtype=np.int32)), (0, np.array([[True]], dtype=bool)),
    (1, np.zeros((1, 16), dtype=np.float32)), (1, np.zeros((16,), dtype=np.int32)),
    (1, np.full((1, 16), -1, dtype=np.int32)), (1, np.full((1, 16), 1024, dtype=np.int32)),
])
def test_invalid_fused_tokens_fail_without_consuming_rng(field, value):
    session = FrameSession()
    session.outputs[field] = value
    runtime = runtime_for(session)
    before = copy.deepcopy(runtime.rng.bit_generator.state)
    with pytest.raises(ValueError, match="Fused frame returned"):
        run_frame(runtime)
    assert runtime.rng.bit_generator.state == before


def test_graph_failure_preserves_rng_for_a_retry():
    session = FrameSession()
    session.run = lambda *args: (_ for _ in ()).throw(RuntimeError("GPU execution failed"))
    runtime = runtime_for(session)
    before = copy.deepcopy(runtime.rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="GPU execution failed"):
        run_frame(runtime)
    assert runtime.rng.bit_generator.state == before


@pytest.mark.parametrize("fused", [False, True])
def test_optional_graph_has_separate_lazy_contexts_without_mutating_assets(monkeypatch, tmp_path, fused):
    from plugins.moss_deploy import trt_pool

    class Driver:
        def __init__(self, device):
            self.device, self.context, self.stream, self.lib = device, object(), object(), SimpleNamespace()
        def call(self, *args):
            pass
        def _bind(self, *args):
            pass

    class Session:
        def configure_profiling(self, graph, **kwargs):
            self.graph = graph
        def run(self, feeds):
            return {"session": self, "graph": self.graph, "feeds": feeds}
        def close(self):
            pass

    class Logger:
        WARNING = 1
        def __init__(self, level):
            pass

    deserialized = []
    def deserialize(data):
        deserialized.append(data)
        return SimpleNamespace(create_execution_context=object)
    trt = SimpleNamespace(Logger=Logger, Runtime=lambda log: SimpleNamespace(
        deserialize_cuda_engine=deserialize))
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    module = SimpleNamespace(CudaDriver=Driver, TensorRTSession=Session)
    monkeypatch.setattr(trt_pool.importlib.util, "spec_from_file_location", lambda *args:
                        SimpleNamespace(loader=SimpleNamespace(exec_module=lambda loaded: None)))
    monkeypatch.setattr(trt_pool.importlib.util, "module_from_spec", lambda spec: module)
    base, local, extra = (tmp_path / name for name in
                          ("global_unified.plan", "local_cached_step.plan", "fused.plan"))
    base.write_bytes(b"global")
    local.write_bytes(b"original local weights")
    extra.write_bytes(b"fused")
    original_graph_paths = {"global_unified": base, "local_cached_step": local}
    assets = SimpleNamespace(graph_paths=dict(original_graph_paths), streaming_path=None,
                              precomputed_reference_codes={}, voice_paths={})
    pool = trt_pool.TensorRTGraphPool(assets, tmp_path, driver_tool=base,
                                     optional_engine_paths={"local_fixed_sampled_frame": extra} if fused else None)
    try:
        active_local = "local_fixed_sampled_frame" if fused else "local_cached_step"
        assert assets.graph_paths == original_graph_paths
        assert pool.effective_plan_paths == {"global_unified": base, active_local: extra if fused else local}
        assert set(pool.engines) == set(pool.effective_plan_paths)
        assert deserialized == ([b"global", b"fused"] if fused else [b"global", b"original local weights"])
        if fused:
            assert "superseded" in pool.omitted_graphs["local_cached_step"]
        else:
            assert pool.omitted_graphs == {}
        first, second = pool.create_client("first"), pool.create_client("second")
        assert pool.instance_sessions == {"first": {}, "second": {}}
        if fused:
            with pytest.raises(FileNotFoundError, match="local step inference is disabled"):
                first.run("local_cached_step", {})
            assert pool.instance_sessions == {"first": {}, "second": {}}
        a, b = first.run(active_local, {}), second.run(active_local, {})
        assert a["session"] is not b["session"]
        assert a["session"].engine is b["session"].engine
        assert a["session"].cuda is not b["session"].cuda
        assert a["graph"] == b["graph"] == active_local
    finally:
        pool.close()


def test_adapted_warmup_uses_fused_graph_when_old_local_runner_is_disabled(monkeypatch):
    from plugins.moss_deploy import runtime
    monkeypatch.setenv("MOSS_FAST_SAMPLING", "0")
    original = type("Original", (), {"generate_audio_frames": generation_branch})
    monkeypatch.setattr(runtime, "load_original_source", lambda path: (
        SimpleNamespace(OnnxTtsRuntime=original),
        SimpleNamespace(OrtCpuRuntime=original, _extract_last_hidden=lambda value: value),
    ))
    pool = SimpleNamespace(assets=SimpleNamespace(source_dir=None, default_voice="Junhao"),
                           frame_contract={"generation_defaults": CONTROLS},
                           omitted_graphs={"local_cached_step": "superseded", "codec_decoder": "streaming only"})
    cls = runtime._create_original_adapter_class(pool)
    model = object.__new__(cls)
    model.manifest = {"generation_defaults": {**CONTROLS, "do_sample": True, "sample_mode": "full"},
                      "tts_config": {"n_vq": 16}}
    model.rng = np.random.default_rng(1234)
    model.resolve_prompt_audio_codes = lambda **kwargs: [[0] * 16]
    model.encode_text = lambda text: [5]
    model.build_voice_clone_request_rows = lambda *args: {"inputIds": [[0] * 17], "attentionMask": [[1]]}
    model.run_local_cached_step = lambda *args, **kwargs: pytest.fail("Warmup reached the disabled old local graph")
    frame = FrameSession()
    model.sessions = {"prefill": SimpleNamespace(run=lambda *args: [np.zeros((1, 768), dtype=np.float32)]),
                      "local_fixed_sampled_frame": frame}
    codec_calls = []
    model.codec_streaming_session = SimpleNamespace(reset=lambda: codec_calls.append("reset"),
                                                    run_frames=lambda rows: codec_calls.append(rows))
    model.warmup()
    assert len(frame.feeds) == 1
    assert codec_calls == ["reset", [[0] * 16], "reset"]


@pytest.mark.parametrize("fault", ["unknown", "missing", "collision"])
def test_optional_graph_config_is_checked_before_importing_tensorrt(tmp_path, fault):
    from plugins.moss_deploy.trt_pool import TensorRTGraphPool
    base = tmp_path / "global_unified.plan"
    base.write_bytes(b"global")
    assets = SimpleNamespace(graph_paths={"global_unified": base}, streaming_path=None,
                              precomputed_reference_codes={}, voice_paths={})
    optional = {"unknown": base} if fault == "unknown" else {"local_fixed_sampled_frame": tmp_path / "missing.plan"}
    if fault == "collision":
        assets.graph_paths["local_fixed_sampled_frame"] = base
        optional = {"local_fixed_sampled_frame": base}
    with pytest.raises((ValueError, FileNotFoundError)):
        TensorRTGraphPool(assets, engine_paths={key: base for key in assets.graph_paths},
                          optional_engine_paths=optional)
