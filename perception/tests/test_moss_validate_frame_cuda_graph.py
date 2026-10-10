"""Offline tests for the exact GPU A/B gate; no TensorRT/CUDA model execution."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def validator():
    path = Path(__file__).resolve().parents[1] / "tools/moss_validate_frame_cuda_graph.py"
    spec = importlib.util.spec_from_file_location("_moss_cuda_graph_validation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path, validator, monkeypatch):
    plan = tmp_path / "local_fixed_sampled_frame.plan"
    plan.write_bytes(b"test engine; never sent to a GPU")
    rows = []
    for hidden in range(4):
        for uniform in range(8):
            label = f"hidden-{hidden}-uniform-{uniform}"
            np.savez(tmp_path / (label + ".npz"),
                     input__global_hidden=np.full((1, 768), hidden, np.float32),
                     input__random_u=np.full((1, 17), uniform / 8, np.float32))
            rows.append({"sample": label, "model_logits_pass": True, "sampling_exact": True,
                         "finite": True, "outputs_valid": True})
    (tmp_path / "build_report.json").write_text(json.dumps({
        "status": "built", "validation_pass": True, "sample_count": 32,
        "engines": [{"name": validator.GRAPH_NAME, "sha256": validator.file_sha256(plan), "validation": rows}]}))

    class Driver:
        def __init__(self, device):
            assert device == 3
        def info(self):
            return {"test_double": True}

    class Session:
        instances = []
        fault = None

        def __init__(self, candidate, cuda):
            assert candidate == plan
            self.capture = os.environ["MOSS_TRT_FRAME_CUDA_GRAPH"] == "1"
            assert os.environ["MOSS_TRT_REUSE_FRAME_BUFFERS"] == "1"
            assert os.environ["MOSS_TRT_RESIDENT_STATE"] == "0"
            self.calls, self.closed, self.saved = 0, False, None
            self.instances.append(self)

        def configure_profiling(self, graph, *, enabled):
            assert graph == validator.GRAPH_NAME and enabled

        def run(self, feeds):
            self.calls += 1
            value = np.float32(feeds["global_hidden"][0, 0] + feeds["random_u"][0, 0])
            result = {"should_continue": np.array([[int(value) % 2]], np.int32),
                      "frame_token_ids": np.full((1, 16), int(value * 16), np.int32),
                      "text_logits": np.full((1, 2), value, np.float32),
                      "audio_logits": np.full((1, 16, 1024), value, np.float32)}
            if self.capture and self.fault == "float_ulp":
                result["audio_logits"][0, 3, 17] = np.nextafter(value, np.float32(np.inf))
            if self.capture and self.fault == "token":
                result["frame_token_ids"][0, 4] += 1
            if self.fault == "nondeterministic" and self.calls > 32:
                # Both sessions drift identically; A/B alone must not pass.
                result["text_logits"][0, 0] += 1
            if self.fault == "reuse_output":
                if self.saved is None:
                    self.saved = result
                for key, value in result.items():
                    self.saved[key][...] = value
                result = self.saved
            return result

        def cuda_graph_snapshot(self):
            active = self.capture and self.calls > 0 and self.fault != "no_capture"
            return {"enabled": self.capture, "captures": int(active),
                    "replays": self.calls - 1 if active else 0,
                    "warmup_enqueues": int(active), "capture_failures": 0,
                    "last_error": None, "ready": bool(active and not self.closed), "closed": self.closed}

        def profile_snapshot(self):
            graph = self.cuda_graph_snapshot()
            return {"runs": self.calls, "errors": 0, "cuda_allocations": 6,
                    "cuda_frees": 6 if self.closed else 0,
                    **{"cuda_graph_" + key: graph[key] for key in
                       ("captures", "replays", "warmup_enqueues", "capture_failures")}}

        def close(self):
            self.closed = True

    runner = SimpleNamespace(CudaDriver=Driver, TensorRTSession=Session, __file__=__file__)
    monkeypatch.setitem(sys.modules, "moss_trt_session", runner)
    args = SimpleNamespace(plan=plan, samples_dir=tmp_path, output=tmp_path / "result", device=3)
    return args, Session


def test_actual_capture_replay_exact_outputs_and_environment_restoration(validator, prepared, monkeypatch):
    args, session = prepared
    monkeypatch.setenv("MOSS_TRT_FRAME_CUDA_GRAPH", "external-value")
    monkeypatch.delenv("MOSS_TRT_PROFILE", raising=False)
    report = validator.run_validation(args)
    assert report["ok"] and report["validation_pass"] and report["status"] == "passed"
    assert report["unique_samples"] == 32 and report["completed_pairs"] == 72
    assert [instance.capture for instance in session.instances] == [False, True]
    assert all(instance.closed for instance in session.instances)
    assert all(report["execution_checks"].values())
    assert report["sessions"]["captured"]["before_close"]["graph"]["replays"] == 71
    assert os.environ["MOSS_TRT_FRAME_CUDA_GRAPH"] == "external-value"
    assert "MOSS_TRT_PROFILE" not in os.environ
    assert len({row["sample"] for row in report["runs"][:32]}) == 32
    assert {tuple(row["session_order"]) for row in report["runs"]} == {("plain", "captured"), ("captured", "plain")}
    assert not list(args.output.glob("mismatch-*.npz"))


@pytest.mark.parametrize("fault", ["float_ulp", "token"])
def test_even_one_float_ulp_or_one_token_difference_fails_and_keeps_evidence(validator, prepared, fault):
    args, session = prepared
    session.fault = fault
    report = validator.run_validation(args)
    assert not report["ok"] and not report["all_outputs_exact"]
    assert all(report["execution_checks"].values())
    assert report["runs"][0]["mismatch_artifact"]
    with np.load(args.output / report["runs"][0]["mismatch_artifact"]) as mismatch:
        assert "plain__audio_logits" in mismatch and "captured__audio_logits" in mismatch
    assert json.loads((args.output / "report.json").read_text())["status"] == "failed"


def test_equal_output_without_real_capture_is_rejected(validator, prepared):
    args, session = prepared
    session.fault = "no_capture"
    report = validator.run_validation(args)
    assert report["all_outputs_exact"] and not report["ok"]
    assert not report["execution_checks"]["capture_really_executed"]


def test_identical_drift_in_both_sessions_is_not_hidden_by_ab_comparison(validator, prepared):
    args, session = prepared
    session.fault = "nondeterministic"
    report = validator.run_validation(args)
    assert report["all_outputs_exact"] and not report["ok"]
    assert not report["repeat_inputs_exact"]
    assert "mismatch_artifact" in report["runs"][32]


def test_output_buffer_aliasing_is_rejected(validator, prepared):
    args, session = prepared
    session.fault = "reuse_output"
    report = validator.run_validation(args)
    assert report["all_outputs_exact"] and report["repeat_inputs_exact"]
    assert not report["ok"] and report["output_lifetime_errors"]


def test_signed_zero_is_a_byte_difference_even_when_values_compare_equal(validator):
    result = validator.compare_exact(np.array([0.0], np.float32), np.array([-0.0], np.float32))
    assert not result["exact"] and result["values_exact"]
    assert result["max_abs_error"] == 0 and result["first_different_flat_index"] == 0


def test_wrong_plan_hash_fails_before_constructing_gpu_session(validator, prepared):
    args, session = prepared
    args.plan.write_bytes(b"different engine")
    report = validator.run_validation(args)
    assert not report["ok"] and not session.instances
    assert "successful fused-frame build report" in report["errors"][0]


def test_invalid_uniform_fails_before_constructing_gpu_session(validator, prepared):
    args, session = prepared
    np.savez(args.samples_dir / "hidden-3-uniform-7.npz",
             input__global_hidden=np.zeros((1, 768), np.float32),
             input__random_u=np.ones((1, 17), np.float32))
    report = validator.run_validation(args)
    assert not report["ok"] and not session.instances
    assert "outside [0, 1)" in report["errors"][0]


def test_previous_report_is_never_overwritten(validator, prepared):
    args, session = prepared
    args.output.mkdir()
    existing = args.output / "report.json"
    existing.write_text("preserve this evidence")
    with pytest.raises(FileExistsError):
        validator.run_validation(args)
    assert existing.read_text() == "preserve this evidence" and not session.instances
