"""Selected projection keeps the original token choices and rejects bad plans."""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.moss_deploy.sampling import make_fast_generation_method
from plugins.moss_deploy.selected_logits import (
    SelectedAudioScores, SelectedTextScores, load_selected_contract,
    wrap_selected_outputs,
)
from test_moss_sampling import ReferenceRuntime, _sample_assistant_text_token


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("sample", [False, True])
def test_selected_text_matches_full_logits_tokens_and_rng(fast, sample):
    sampler = _sample_assistant_text_token
    if fast:
        sampler = make_fast_generation_method(ReferenceRuntime.generate_audio_frames).__globals__["_sample_assistant_text_token"]
    manifest = {"continue": 9, "end": 7}
    generator = np.random.default_rng(123)
    original_rng, selected_rng = np.random.default_rng(456), np.random.default_rng(456)
    for index in range(120):
        values = generator.standard_normal(16384).astype(np.float32)
        if index % 3 == 0:
            values[9] = values[7]  # Candidate order must also preserve ties.
        defaults = dict(do_sample=sample, text_temperature=(.8, 1., 1.5)[index % 3],
                        text_top_k=(0, 1, 50)[index % 3],
                        text_top_p=(0., .5, .95, 1.)[index % 4])
        selected = SelectedTextScores(values[[9, 7]][None, :], [9, 7])
        expected = _sample_assistant_text_token(values, manifest, defaults, original_rng)
        assert sampler(selected.reshape(-1), manifest, defaults, selected_rng) == expected
    assert original_rng.bit_generator.state == selected_rng.bit_generator.state


def test_selected_scores_reject_uncomputed_logits_and_wrong_shapes():
    text = SelectedTextScores([[3, 5]], [9, 7])
    np.testing.assert_array_equal(text[np.array([9, 7], dtype=np.int32)], [3, 5])
    for ids in ([7, 9], [9], [9, 8], 9):
        with pytest.raises(ValueError, match="uncomputed text logit"):
            text[ids]
    with pytest.raises(ValueError, match="flatten"):
        text.reshape(1, -1)
    with pytest.raises(ValueError, match="two candidates"):
        SelectedTextScores([1, 2, 3], [9, 7])
    with pytest.raises(ValueError, match="1024"):
        SelectedAudioScores(np.zeros(1023), 0)


@pytest.mark.parametrize("step,previous,channel", [(0, 0, 0), (1, 0, 0)] + [(2, i, i + 1) for i in range(15)])
def test_local_steps_wrap_the_consumed_audio_channel(step, previous, channel):
    audio = np.arange(1024, dtype=np.float32)[None, :]
    cache = object()
    raw = {"text_logits": np.array([[9., 7.]], dtype=np.float32),
           "audio_logits": audio, "local_present_key_0": cache}
    wrapped = wrap_selected_outputs(raw, {"step_type": np.array([step]),
                                         "channel_index": np.array([previous])},
                                    {"text_token_ids": [9, 7], "audio_channels": 16})
    np.testing.assert_array_equal(wrapped["audio_logits"].for_channel(channel), audio[0])
    assert wrapped["local_present_key_0"] is cache
    assert raw["audio_logits"] is audio
    with pytest.raises(ValueError, match="uncomputed audio channel"):
        wrapped["audio_logits"].for_channel((channel + 1) % 16)


@pytest.mark.parametrize("step,channel", [(3, 0), (-1, 0), (2, 15), (2, -1), (0, 1), (1, 1)])
def test_invalid_step_and_channel_combinations_fail(step, channel):
    outputs = {"text_logits": np.zeros((1, 2)), "audio_logits": np.zeros((1, 1024))}
    with pytest.raises(ValueError):
        wrap_selected_outputs(outputs, {"step_type": [step], "channel_index": [channel]},
                              {"text_token_ids": [9, 7], "audio_channels": 16})


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "original.onnx"
    source.write_bytes(b"original local graph")
    plan = tmp_path / "local_cached_step.plan"
    plan.write_bytes(b"selected target engine")
    contract_path = tmp_path / "selected_contract.json"
    report_path = tmp_path / "build_report.json"
    contract = dict(version=1, text_token_ids=[9, 7], audio_channels=16,
                    audio_vocab_size=1024, source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    selected_onnx_sha256=hashlib.sha256(b"selected graph").hexdigest())
    report = {"status": "built", "engines": [{"name": "local_cached_step",
              "onnx_sha256": contract["selected_onnx_sha256"],
              "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
              "validation": [{"outputs": {name: {"numerical_pass": True} for name in
                              ("text_logits", "audio_logits", "local_present_key_0", "local_present_value_0")}}]}]}
    model = dict(audio_assistant_slot_token_id=9, audio_end_token_id=7, n_vq=16,
                 audio_codebook_sizes=[1024] * 16, vocab_size=16384, local_layers=1)
    assets = SimpleNamespace(tts_meta={"model_config": model},
                             graph_paths={"local_cached_step": source}, voice_files={"Junhao": ()})

    def save():
        contract_path.write_text(json.dumps(contract))
        report_path.write_text(json.dumps(report))

    save()
    return SimpleNamespace(source=source, plan=plan, contract_path=contract_path,
                           report_path=report_path, contract=contract, report=report,
                           assets=assets, model=model, save=save, root=tmp_path)


def test_valid_contract_is_accepted(bundle):
    assert load_selected_contract(bundle.plan, bundle.contract_path, bundle.assets) == bundle.contract


@pytest.mark.parametrize("fault", ["missing_plan", "source_hash", "selected_hash", "plan_hash",
                                  "failed_build", "missing_engine", "duplicate_engine",
                                  "no_validation", "empty_outputs", "failed_output",
                                  "version", "token_order", "channels", "vocab", "model_vocab"])
def test_contract_rejects_incompatible_or_unvalidated_plan(bundle, fault):
    entry = bundle.report["engines"][0]
    if fault == "missing_plan":
        bundle.plan.unlink()
    elif fault == "source_hash":
        bundle.source.write_bytes(b"another original graph")
    elif fault == "selected_hash":
        entry["onnx_sha256"] = "bad"
    elif fault == "plan_hash":
        bundle.plan.write_bytes(b"another engine")
    elif fault == "failed_build":
        bundle.report["status"] = "failed"
    elif fault == "missing_engine":
        entry["name"] = "another_graph"
    elif fault == "duplicate_engine":
        bundle.report["engines"].append(dict(entry))
    elif fault == "no_validation":
        entry["validation"] = []
    elif fault == "empty_outputs":
        entry["validation"] = [{"outputs": {}}]
    elif fault == "failed_output":
        entry["validation"][0]["outputs"]["audio_logits"]["numerical_pass"] = False
    elif fault == "version":
        bundle.contract["version"] = 2
    elif fault == "token_order":
        bundle.contract["text_token_ids"] = [7, 9]
    elif fault == "channels":
        bundle.contract["audio_channels"] = 8
    elif fault == "vocab":
        bundle.contract["audio_vocab_size"] = 512
    elif fault == "model_vocab":
        bundle.model["audio_codebook_sizes"][0] = 512
    bundle.save()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_selected_contract(bundle.plan, bundle.contract_path, bundle.assets)


@pytest.fixture
def adapter_loader(monkeypatch, bundle):
    import plugins.moss_tts as adapter
    import plugins.moss_deploy.assets as assets_module
    import plugins.moss_deploy.trt_pool as pool_module

    monkeypatch.delenv("MOSS_TRT_SELECTED_LOCAL_PLAN", raising=False)
    monkeypatch.delenv("MOSS_TRT_SELECTED_LOCAL_CONTRACT", raising=False)
    mapping = {"local_cached_step": "original.plan", "global_unified": "global.plan"}
    (bundle.root / "engine-map.json").write_text(json.dumps(mapping))
    monkeypatch.setattr(adapter, "ensure_moss_assets", lambda *args, **kwargs: bundle.root)
    monkeypatch.setattr(assets_module, "ModelAssets", lambda *args, **kwargs: bundle.assets)
    state = SimpleNamespace(pools=[], runtimes=[])

    def make_pool(assets, **kwargs):
        pool = SimpleNamespace(assets=assets, kwargs=kwargs, closed=False)
        pool.close = lambda: setattr(pool, "closed", True)
        state.pools.append(pool)
        return pool

    def make_runtime(pool, **kwargs):
        runtime = SimpleNamespace(pool=pool, kwargs=kwargs, defaults={})
        runtime.set_generation_defaults = lambda **values: runtime.defaults.update(values)
        state.runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(pool_module, "TensorRTGraphPool", make_pool)
    monkeypatch.setitem(sys.modules, "plugins.moss_deploy.runtime", SimpleNamespace(MossRuntime=make_runtime))
    state.load = lambda: adapter.MossTTSAdapter._load_runtimes(bundle.root, None, 3)
    return state


def test_adapter_keeps_original_plan_without_selected_environment(adapter_loader, bundle):
    runtimes, voices, pool = adapter_loader.load()
    assert len(runtimes) == 3 and voices == ("Junhao",)
    assert pool.selected_local_contract is None
    assert pool.kwargs["engine_paths"]["local_cached_step"] == bundle.root / "original.plan"


def test_adapter_passes_selected_contract_and_only_overrides_local_plan(adapter_loader, bundle, monkeypatch):
    monkeypatch.setenv("MOSS_TRT_SELECTED_LOCAL_PLAN", str(bundle.plan))
    monkeypatch.setenv("MOSS_TRT_SELECTED_LOCAL_CONTRACT", str(bundle.contract_path))
    runtimes, _voices, pool = adapter_loader.load()
    assert pool.selected_local_contract == bundle.contract
    assert pool.kwargs["engine_paths"] == {"local_cached_step": bundle.plan, "global_unified": bundle.root / "global.plan"}
    assert all(runtime.pool is pool for runtime in runtimes)


@pytest.mark.parametrize("environment", ["MOSS_TRT_SELECTED_LOCAL_PLAN", "MOSS_TRT_SELECTED_LOCAL_CONTRACT"])
def test_partial_selected_environment_fails_before_pool_creation(adapter_loader, bundle, monkeypatch, environment):
    monkeypatch.setenv(environment, str(bundle.plan))
    with pytest.raises(ValueError, match="both its plan and contract"):
        adapter_loader.load()
    assert not adapter_loader.pools and not adapter_loader.runtimes


def test_bad_selected_contract_fails_before_pool_creation(adapter_loader, bundle, monkeypatch):
    monkeypatch.setenv("MOSS_TRT_SELECTED_LOCAL_PLAN", str(bundle.plan))
    monkeypatch.setenv("MOSS_TRT_SELECTED_LOCAL_CONTRACT", str(bundle.contract_path))
    bundle.report["status"] = "failed"
    bundle.save()
    with pytest.raises(ValueError, match="successful numerical build report"):
        adapter_loader.load()
    assert not adapter_loader.pools and not adapter_loader.runtimes
