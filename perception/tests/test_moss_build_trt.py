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
