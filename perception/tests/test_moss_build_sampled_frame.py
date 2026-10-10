"""Offline sampling and fail-closed publication gates for the GPU frame builder."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "tools/moss_build_sampled_frame.py"
    spec = importlib.util.spec_from_file_location("_moss_frame_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topk_keeps_threshold_ties_and_samples_token_order(builder):
    # Token 0 has a lower score than token 1; u=.05 still selects token 0,
    # proving this is token-ID CDF rather than descending-score CDF.
    token, _ = builder.sample_f32(np.array([2, 4, 2, -10], np.float32), .05,
                                  temperature=1, top_k=2, top_p=1)
    assert token == 0
    token, detail = builder.sample_f32(np.array([2, 4, 2, -10], np.float32), .99,
                                       temperature=1, top_k=2, top_p=1)
    assert token == 2
    assert detail["unmasked_tokens"] == 3


def test_top_p_keeps_equal_previous_cdf_boundary(builder):
    token, detail = builder.sample_f32(np.zeros(4, np.float32), .9,
                                       temperature=1, top_k=4, top_p=.5)
    assert token == 2  # previous CDF of token 2 equals .5 and is retained.
    assert detail["unmasked_tokens"] == 3


def test_zero_uniform_preserves_original_first_index_edge(builder):
    token, _ = builder.sample_f32(np.array([-100, 10, 9], np.float32), 0,
                                  temperature=1, top_k=1, top_p=.8)
    assert token == 0


def test_near_one_uses_last_positive_mass_not_argmax_or_masked_tail(builder, monkeypatch):
    # A valid softmax rounded/accumulated slightly below one. The final slot is
    # masked. Normalizing only the final CDF element would incorrectly pick it;
    # the previous overflow fallback would incorrectly choose argmax token 0.
    probabilities = np.array([.2, .3, .49999988, 0], dtype=np.float32)
    monkeypatch.setattr(builder, "softmax_f32", lambda _scores: probabilities.copy())
    token, diagnostic = builder.sample_f32(
        np.array([100, 90, 80, -100], np.float32), np.nextafter(np.float32(1), np.float32(0)),
        temperature=1, top_k=0, top_p=1)
    assert token == 2
    assert diagnostic["cdf_total_before_normalization"] < 1
    assert diagnostic["cdf_last"] == 1 and not diagnostic["rounding_fallback"]


@pytest.mark.parametrize("uniform", [-1, 1, np.inf, np.nan])
def test_invalid_uniform_is_rejected(builder, uniform):
    with pytest.raises(ValueError, match="uniform"):
        builder.sample_f32(np.zeros(2, np.float32), uniform, temperature=1, top_k=2, top_p=1)


def test_nonfinite_logits_and_invalid_output_tokens_fail(builder):
    with pytest.raises(ValueError, match="finite"):
        builder.sample_f32(np.array([0, np.nan]), .5, temperature=1, top_k=2, top_p=1)
    outputs = {"should_continue": np.ones((1, 1), np.int32),
               "frame_token_ids": np.zeros((1, 16), np.int32),
               "text_logits": np.zeros((1, 2), np.float32),
               "audio_logits": np.zeros((1, 16, 1024), np.float32)}
    outputs["frame_token_ids"][0, 15] = 1024
    with pytest.raises(ValueError, match="vocabulary"):
        builder.check_outputs(outputs)


def test_sampler_precision_constraints_include_float_outputs_only(builder):
    class Layer:
        def __init__(self, name, types, inputs=()):
            self.name, self.types, self.num_outputs = name, types, len(types)
            self.input_types, self.num_inputs = inputs, len(inputs)
            self.outputs, self.precision = {}, None
        def get_output(self, index):
            return SimpleNamespace(dtype=self.types[index])
        def get_input(self, index):
            return SimpleNamespace(dtype=self.input_types[index])
        def set_output_type(self, index, dtype):
            self.outputs[index] = dtype
    layers = [Layer("sample/step_01/temperature", ["fp16"]),
              Layer("sample/step_01/TopK", ["fp32", "int32"]),
              Layer("sample/step_01/comparison", ["bool"], ["fp32", "fp32"]),
              Layer("model", ["fp16"]), Layer("sample/step_01/integer_comparison", ["bool"], ["int32"])]
    network = SimpleNamespace(num_layers=len(layers), get_layer=lambda i: layers[i])
    flags = []
    config = SimpleNamespace(clear_flag=lambda x: flags.append(("clear", x)),
                             set_flag=lambda x: flags.append(("set", x)))
    trt = SimpleNamespace(float16="fp16", float32="fp32",
                          BuilderFlag=SimpleNamespace(PREFER_PRECISION_CONSTRAINTS="prefer", OBEY_PRECISION_CONSTRAINTS="obey"))
    assert len(builder.force_sampler_fp32(network, config, trt)) == 3
    assert layers[0].precision == "fp32"
    assert layers[1].outputs == {0: "fp32"}
    assert layers[2].precision == "fp32" and layers[2].outputs == {}
    assert layers[3].precision is None and layers[4].precision is None
    assert flags == [("clear", "prefer"), ("set", "obey")]
    with pytest.raises(ValueError, match="No named"):
        builder.force_sampler_fp32(SimpleNamespace(num_layers=0), config, trt)


def test_sampler_constraint_tracks_unnamed_loop_and_stops_at_transformer(builder):
    def tensor(name):
        return SimpleNamespace(name=name, dtype="fp32")
    tensors = {name: tensor(name) for name in ("hidden", "model_logits", "probabilities", "iterator",
                                              "initial", "recurrence", "sum", "cdf", "mask", "other")}
    class Layer:
        def __init__(self, name, inputs, outputs):
            self.name, self.inputs, self.outputs = name, inputs, outputs
            self.num_inputs, self.num_outputs = len(inputs), len(outputs)
            self.precision = None
            self.output_types = {}
        def get_input(self, index):
            return tensors[self.inputs[index]]
        def get_output(self, index):
            return tensors[self.outputs[index]]
        def set_output_type(self, index, dtype):
            self.output_types[index] = dtype
    layers = [Layer("moss_frame/step_01/projection", ["hidden"], ["model_logits"]),
              Layer("sample/step_01/probabilities", ["model_logits"], ["probabilities"]),
              Layer("(Unnamed Layer* 2) [Iterator]", ["probabilities"], ["iterator"]),
              Layer("(Unnamed Layer* 3) [Fill]", [], ["initial"]),
              Layer("(Unnamed Layer* 4) [Recurrence]", ["initial", "sum"], ["recurrence"]),
              Layer("(Unnamed Layer* 5) [ElementWise]", ["recurrence", "iterator"], ["sum"]),
              Layer("(Unnamed Layer* 6) [LoopOutput]", ["sum"], ["cdf"]),
              Layer("sample/step_01/cdf_below_u", ["cdf"], ["mask"]),
              Layer("(Unnamed Layer* 8) [Unrelated]", ["hidden"], ["other"])]
    tensors["mask"].dtype = "bool"
    assert builder._sampler_layer_indices(layers) == set(range(1, 8))
    trt = SimpleNamespace(float16="fp16", float32="fp32",
                          BuilderFlag=SimpleNamespace(PREFER_PRECISION_CONSTRAINTS=0, OBEY_PRECISION_CONSTRAINTS=1))
    config = SimpleNamespace(clear_flag=lambda _: None, set_flag=lambda _: None)
    result = builder.force_sampler_fp32(SimpleNamespace(num_layers=len(layers), get_layer=lambda i: layers[i]), config, trt)
    assert len(result) == 7
    assert all(layer.precision == "fp32" for layer in layers[1:8])
    assert layers[0].precision is None and layers[8].precision is None
    assert layers[7].output_types == {}  # Boolean output remains boolean.


def prepare_run(builder, monkeypatch, directory, *, bad_sampling=False, bad_model=False):
    assets, candidate = directory / "assets", directory / "candidate"
    assets.mkdir()
    candidate.mkdir()
    (assets / "original.onnx").write_bytes(b"original graph")
    (candidate / "frame.onnx").write_bytes(b"fused graph")
    (candidate / "weights.data").write_bytes(b"original weights")
    original = {"name": "local_cached_step", "file": "original.onnx", "validation_samples": []}
    for i in range(4):
        name = f"sample-{i}.npz"
        np.savez(assets / name, input__global_hidden=np.zeros((1, 768), np.float32),
                 input__local_past_key_0=np.zeros((1, i, 1, 2), np.float32),
                 input__local_past_value_0=np.zeros((1, i, 1, 2), np.float32))
        original["validation_samples"].append(name)
    (assets / "analysis.json").write_text(json.dumps({"graphs": [original]}))
    plan_dir = assets / "plans/local_cached_step"
    plan_dir.mkdir(parents=True)
    plan = plan_dir / "local_cached_step.plan"
    plan.write_bytes(b"verified original engine")
    (plan_dir / "build_report.json").write_text(json.dumps({"status": "built", "engines": [{
        "name": "local_cached_step", "sha256": builder.sha256(plan),
        "onnx_sha256": builder.sha256(assets / "original.onnx"),
        "validation": [{"outputs": {"audio_logits": {"numerical_pass": True}}}]}]}))
    contract = {"version": 1, "graph_name": builder.GRAPH_NAME, "audio_channels": 16,
                "audio_vocab_size": 1024, "steps_per_frame": 17, "batch_size": 1,
                "text_token_ids": [9, 7], "sampling": builder.EXPECTED_SAMPLING,
                "sampling_semantics": {"cdf_normalization": builder.CDF_NORMALIZATION},
                "source_onnx_sha256": builder.sha256(assets / "original.onnx"),
                "onnx_sha256": builder.sha256(candidate / "frame.onnx"),
                "external_weights": {"weights.data": {"bytes": 16, "sha256": builder.sha256(candidate / "weights.data")}}}
    (candidate / "frame-contract.json").write_text(json.dumps(contract))
    graph = {"name": builder.GRAPH_NAME, "file": "frame.onnx"}
    (candidate / "analysis.json").write_text(json.dumps({"graphs": [graph]}))
    output = directory / "plans"
    commands = ["--analysis", str(candidate / "analysis.json"), "--contract", str(candidate / "frame-contract.json"),
                "--assets", str(assets), "--output", str(output)]

    def fake_build(graph, source, staging, args):
        assert not list(output.glob("*.plan"))
        assert args.strongly_typed is True
        assert getattr(args, "configure_network", None) is None
        path = staging / (builder.GRAPH_NAME + ".plan")
        path.write_bytes(b"unchecked candidate")
        return {"name": graph["name"], "file": path.name, "bytes": path.stat().st_size,
                "sha256": builder.sha256(path), "onnx_sha256": contract["onnx_sha256"]}

    class Session:
        def __init__(self, path, _driver):
            self.candidate = path.stem == builder.GRAPH_NAME
        def run(self, feeds):
            assert not list(output.glob("*.plan"))
            audio = np.tile(np.linspace(-2, 2, 1024, dtype=np.float32), (1, 16, 1))
            if self.candidate:
                uniforms = feeds["random_u"][0]
                indices = [builder.sample_f32(logits, uniform,
                           temperature=builder.EXPECTED_SAMPLING[("text" if i == 0 else "audio") + "_temperature"],
                           top_k=builder.EXPECTED_SAMPLING[("text" if i == 0 else "audio") + "_top_k"],
                           top_p=builder.EXPECTED_SAMPLING[("text" if i == 0 else "audio") + "_top_p"])[0]
                           for i, (logits, uniform) in enumerate(zip([np.zeros(2, np.float32), *audio[0]], uniforms))]
                frame = np.asarray([indices[1:]], dtype=np.int32)
                if bad_sampling:
                    frame[0, -1] = (frame[0, -1] + 1) % 1024
                return {"should_continue": np.asarray([[int(indices[0] == 0)]], np.int32),
                        "frame_token_ids": frame, "text_logits": np.zeros((1, 2), np.float32), "audio_logits": audio}
            step = int(feeds["past_valid_lengths"].item())
            assert feeds["local_past_key_0"].shape[1] == step
            return {"text_logits": np.zeros((1, 16384), np.float32), "audio_logits": audio + (100 if bad_model else 0),
                    "local_present_key_0": np.zeros((1, step + 1, 1, 2), np.float32),
                    "local_present_value_0": np.zeros((1, step + 1, 1, 2), np.float32)}
        def close(self):
            pass

    runner_path = Path(__file__).resolve().parents[1] / "tools/moss_trt_session.py"
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="test-only"))
    spec = importlib.util.spec_from_file_location("_frame_runner_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setitem(sys.modules, "moss_build_trt", SimpleNamespace(build=fake_build))
    monkeypatch.setitem(sys.modules, "moss_trt_session", SimpleNamespace(
        CudaDriver=lambda device: SimpleNamespace(info=lambda: {"name": "mock only"}),
        TensorRTSession=Session, compare=runner.compare))
    monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", "0")
    monkeypatch.setenv("MOSS_TRT_REUSE_LOCAL_BUFFERS", "0")
    return output, commands


def test_actual_plan_published_only_after_all_32_checks(builder, monkeypatch, tmp_path):
    output, commands = prepare_run(builder, monkeypatch, tmp_path)
    builder.main(commands)
    report = json.loads((output / "build_report.json").read_text())
    assert report["status"] == "built" and report["validation_pass"]
    assert report["sample_count"] == 32 and report["continuation_values_covered"] == [0, 1]
    assert len(list(output.glob("*.plan"))) == 1
    assert all(len(row["logit_checks"]) == 17 and len(row["sampling_checks"]) == 17
               for row in report["engines"][0]["validation"])


@pytest.mark.parametrize("kind", ["bad_sampling", "bad_model"])
def test_integer_or_model_failure_never_publishes_plan(builder, monkeypatch, tmp_path, kind):
    output, commands = prepare_run(builder, monkeypatch, tmp_path, **{kind: True})
    with pytest.raises(RuntimeError, match="validation failed"):
        builder.main(commands)
    assert not list(output.rglob("*.plan"))
    report = json.loads((output / "build_report.json").read_text())
    assert report["status"] == "failed" and not report["validation_pass"]
    key = "sampling_exact" if kind == "bad_sampling" else "model_logits_pass"
    assert all(not row[key] for row in report["engines"][0]["validation"])


def test_existing_report_cannot_be_overwritten(builder, tmp_path):
    output = tmp_path / "plans"
    output.mkdir()
    report = output / "build_report.json"
    report.write_text("retained evidence")
    with pytest.raises(FileExistsError):
        builder.main(["--analysis", "missing", "--contract", "missing", "--assets", "missing", "--output", str(output)])
    assert report.read_text() == "retained evidence"
