"""Offline checks that unchecked TensorRT plans cannot become runtime assets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def builder(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="10.mock"))
    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace())
    runner_spec = importlib.util.spec_from_file_location("moss_trt_session", tools / "moss_trt_session.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    monkeypatch.setitem(sys.modules, "moss_trt_session", runner)
    spec = importlib.util.spec_from_file_location("_moss_builder_test", tools / "moss_build_trt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CudaDriver", lambda device: SimpleNamespace(info=lambda: {"name": "mock GPU"}))
    return module


def prepare(monkeypatch, tmp_path, builder, *, fail_graph=None, empty_samples=False, extra_args=()):
    graphs = []
    for name in ("first", "second"):
        np.savez(tmp_path / f"{name}.npz", input__signal=np.array([1.0]), ref__audio=np.array([1.0]))
        graphs.append({"name": name, "file": f"{name}.onnx",
                       "validation_samples": [] if empty_samples else [f"{name}.npz"]})
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"graphs": graphs}))
    output = tmp_path / "plans"
    builds = []

    def fake_build(graph, directory, staging, args):
        assert staging.parent == output
        assert staging.name.startswith(".moss-build-")
        assert not list(output.glob("*.plan")), "Plans must stay hidden until all validation passes"
        name = graph["name"]
        path = staging / f"{name}.plan"
        path.write_bytes(b"mock engine")
        builds.append(name)
        return {"name": name, "file": path.name, "bytes": path.stat().st_size,
                "exact_validation_outputs": []}

    class FakeSession:
        def __init__(self, path, cuda):
            assert path.is_file()
            assert not list(output.glob("*.plan"))
            self.name = path.stem

        def run(self, inputs):
            return {"audio": inputs["signal"] + (1.0 if self.name == fail_graph else 0.0)}

    monkeypatch.setattr(builder, "build", fake_build)
    monkeypatch.setattr(builder, "TensorRTSession", FakeSession)
    monkeypatch.setattr(sys, "argv", ["moss_build_trt.py", "--analysis", str(analysis),
                                      "--output", str(output), *extra_args])
    return output, builds


def test_publish_only_after_every_selected_graph_passes(builder, monkeypatch, tmp_path):
    output, builds = prepare(monkeypatch, tmp_path, builder)
    builder.main()
    assert builds == ["first", "second"]
    assert sorted(path.name for path in output.glob("*.plan")) == ["first.plan", "second.plan"]
    assert not list(output.glob(".moss-build-*"))
    report = json.loads((output / "build_report.json").read_text())
    assert report["status"] == "built"
    assert all(row["validation"][0]["outputs"]["audio"]["numerical_pass"] for row in report["engines"])


def test_second_graph_numeric_failure_publishes_neither_plan(builder, monkeypatch, tmp_path):
    output, builds = prepare(monkeypatch, tmp_path, builder, fail_graph="second")
    with pytest.raises(RuntimeError, match="Numerical validation failed for second"):
        builder.main()
    assert builds == ["first", "second"]
    assert not list(output.rglob("*.plan"))
    assert not list(output.glob(".moss-build-*"))
    assert json.loads((output / "build_report.json").read_text())["status"] == "failed"


def test_missing_validation_samples_rejected_before_gpu_build(builder, monkeypatch, tmp_path):
    output, builds = prepare(monkeypatch, tmp_path, builder, empty_samples=True)
    monkeypatch.setattr(builder, "CudaDriver", lambda device: pytest.fail("GPU must not start without validation samples"))
    with pytest.raises(ValueError, match="No validation samples"):
        builder.main()
    assert not builds
    assert not list(output.rglob("*.plan"))
    assert json.loads((output / "build_report.json").read_text())["status"] == "failed"


def test_existing_plan_and_report_are_preserved(builder, monkeypatch, tmp_path):
    output, builds = prepare(monkeypatch, tmp_path, builder)
    output.mkdir()
    (output / "first.plan").write_bytes(b"previous verified engine")
    (output / "build_report.json").write_text('{"status": "built"}')
    with pytest.raises(FileExistsError, match="choose a new output directory"):
        builder.main()
    assert not builds
    assert (output / "first.plan").read_bytes() == b"previous verified engine"
    assert (output / "build_report.json").read_text() == '{"status": "built"}'


def test_explicit_size_gate_failure_publishes_no_plan(builder, monkeypatch, tmp_path):
    output, builds = prepare(monkeypatch, tmp_path, builder, extra_args=("--max-model-bytes", "1"))
    with pytest.raises(RuntimeError, match="Engine bundle exceeds"):
        builder.main()
    assert builds == ["first", "second"]
    assert not list(output.rglob("*.plan"))
    report = json.loads((output / "build_report.json").read_text())
    assert report["status"] == "failed"
    assert report["engine_budget_pass"] is False


def test_failed_publication_removes_plans_from_current_attempt(builder, monkeypatch, tmp_path):
    output, _ = prepare(monkeypatch, tmp_path, builder)
    real_link = builder.os.link

    def fail_second_link(source, destination):
        if destination.name == "second.plan":
            raise OSError("mock publication failure")
        real_link(source, destination)

    monkeypatch.setattr(builder.os, "link", fail_second_link)
    with pytest.raises(OSError, match="mock publication failure"):
        builder.main()
    assert not list(output.rglob("*.plan"))
    assert json.loads((output / "build_report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("mode", ["default", "weak", "strong"])
def test_real_build_branch_honors_explicit_types_without_precision_overrides(builder, monkeypatch, tmp_path, mode):
    """Exercise build(), not a replacement build function; GPU API is mocked."""
    events = SimpleNamespace(flags=[], cleared=[], callbacks=[], network_flags=None)
    unary = SimpleNamespace(type="unary", num_inputs=1, precision=None,
                            get_input=lambda _: SimpleNamespace(dtype="float32"))
    network = SimpleNamespace(num_layers=1, num_inputs=0, get_layer=lambda _: unary)
    config = SimpleNamespace(set_memory_pool_limit=lambda *_: None,
                              set_flag=lambda value: events.flags.append(value),
                              clear_flag=lambda value: events.cleared.append(value))
    class Logger:
        WARNING = 2
        def __init__(self, _level):
            pass
    class Builder:
        # A strongly typed model does not query this weakly typed capability
        # flag; the ONNX graph determines whether FP16 operations are needed.
        platform_has_fast_fp16 = mode != "strong"
        def __init__(self, _logger):
            pass
        def create_network(self, flags):
            events.network_flags = flags
            return network
        def create_builder_config(self):
            return config
        def create_optimization_profile(self):
            return object()
        def build_serialized_network(self, passed_network, passed_config):
            assert passed_network is network and passed_config is config
            return b"target-specific mocked engine"
    trt = SimpleNamespace(Logger=Logger, Builder=Builder,
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        OnnxParser=lambda *_: SimpleNamespace(parse_from_file=lambda _: True),
        MemoryPoolType=SimpleNamespace(WORKSPACE=0),
        BuilderFlag=SimpleNamespace(TF32="tf32", FP16="fp16", PREFER_PRECISION_CONSTRAINTS="prefer"),
        LayerType=SimpleNamespace(UNARY="unary"), float16="float16", float32="float32",
        ProfilingVerbosity=SimpleNamespace(DETAILED="detailed"))
    monkeypatch.setattr(builder, "trt", trt)
    monkeypatch.setattr(builder, "exact_output_names", lambda *_: [])
    args = SimpleNamespace(fp32_graph=[], precision="fp16", workspace_mb=64,
                           disable_jit_convolutions=False,
                           configure_network=lambda n, c: events.callbacks.append((n, c)),
                           detailed_profiling=True)
    if mode != "default":
        args.strongly_typed = mode == "strong"
    (tmp_path / "model.onnx").write_bytes(b"ONNX with explicit FP16 model and FP32 sampler")
    entry = builder.build({"name": "model", "file": "model.onnx", "precision": "fp16_mixed"},
                          tmp_path, tmp_path, args)
    assert events.cleared == ["tf32"]
    assert config.profiling_verbosity == "detailed"
    assert entry["strongly_typed"] is (mode == "strong")
    if mode == "strong":
        assert events.network_flags == 3
        assert not events.flags and not events.callbacks
        assert unary.precision is None
        assert entry["precision"] == "onnx_explicit_types"
    else:
        assert events.network_flags == 1
        assert events.flags == ["fp16", "prefer"]
        assert events.callbacks == [(network, config)]
        assert unary.precision == "float32"
        assert entry["precision"] == "fp16"
