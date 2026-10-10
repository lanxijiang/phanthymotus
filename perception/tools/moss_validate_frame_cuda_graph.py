#!/usr/bin/env python3
"""Require bit-identical fused-frame outputs with ordinary and CUDA Graph runs.

Both sessions deserialize the same validated plan on the requested CUDA device.
This diagnostic executes model inference only through TensorRT on that GPU.
The NumPy operations below inspect returned arrays; they do not run the model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


GRAPH_NAME = "local_fixed_sampled_frame"
INTERFACE = {
    "should_continue": ((1, 1), np.dtype("int32")),
    "frame_token_ids": ((1, 16), np.dtype("int32")),
    "text_logits": ((1, 2), np.dtype("float32")),
    "audio_logits": ((1, 16, 1024), np.dtype("float32")),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_info(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "finite": bool(np.isfinite(value).all())}


def compare_exact(plain, captured):
    """No tolerance, including signed-zero bits; diagnostics never relax the gate."""
    left, right = array_info(plain), array_info(captured)
    compatible = plain.shape == captured.shape and plain.dtype == captured.dtype
    result = {"plain": left, "captured": right,
              "exact": bool(compatible and left["finite"] and right["finite"]
                            and left["sha256"] == right["sha256"])}
    if compatible:
        result["values_exact"] = bool(np.array_equal(plain, captured))
        result["unequal_elements"] = int(np.count_nonzero(plain != captured))
        if left["finite"] and right["finite"]:
            delta = np.abs(plain.astype(np.float64) - captured.astype(np.float64))
            result["max_abs_error"] = float(delta.max(initial=0))
            result["rms_error"] = float(np.sqrt(np.mean(delta * delta)))
            if not result["exact"]:
                byte_width = plain.dtype.itemsize
                byte_differences = (np.ascontiguousarray(plain).view(np.uint8).reshape(-1, byte_width)
                                    != np.ascontiguousarray(captured).view(np.uint8).reshape(-1, byte_width))
                index = int(np.flatnonzero(byte_differences.any(axis=1))[0])
                result["first_different_flat_index"] = index
                result["first_different_values"] = [plain.flat[index].item(), captured.flat[index].item()]
    return result


@contextmanager
def session_environment(capture):
    changes = {"MOSS_TRT_FRAME_CUDA_GRAPH": "1" if capture else "0",
               "MOSS_TRT_REUSE_FRAME_BUFFERS": "1", "MOSS_TRT_PROFILE": "1",
               "MOSS_TRT_RESIDENT_STATE": "0"}
    original = {key: os.environ.get(key) for key in changes}
    try:
        os.environ.update(changes)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_samples(plan, directory):
    if plan.stem != GRAPH_NAME:
        raise ValueError(f"Plan filename must be {GRAPH_NAME}.plan so session graph policy is unambiguous")
    build_report_path = plan.parent / "build_report.json"
    build_report = json.loads(build_report_path.read_text())
    entries = [entry for entry in build_report.get("engines", []) if entry.get("name") == GRAPH_NAME]
    digest = file_sha256(plan)
    if (build_report.get("status") != "built" or build_report.get("validation_pass") is not True
            or len(entries) != 1 or entries[0].get("sha256") != digest):
        raise ValueError("Plan is not bound to a successful fused-frame build report")
    rows = entries[0].get("validation", [])
    labels = [f"hidden-{hidden}-uniform-{uniform}" for hidden in range(4) for uniform in range(8)]
    if (build_report.get("sample_count") != 32 or len(rows) != 32
            or {row.get("sample") for row in rows} != set(labels)
            or not all(all(row.get(key) is True for key in
                           ("model_logits_pass", "sampling_exact", "finite", "outputs_valid")) for row in rows)):
        raise ValueError("Expected all 32 successful original builder validation samples")
    samples = []
    for label in labels:
        path = directory / (label + ".npz")
        with np.load(path, allow_pickle=False) as source:
            feeds = {name: source["input__" + name].copy() for name in ("global_hidden", "random_u")}
        for name, shape in (("global_hidden", (1, 768)), ("random_u", (1, 17))):
            value = feeds[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} in {path}")
        if np.any(feeds["random_u"] < 0) or np.any(feeds["random_u"] >= 1):
            raise ValueError(f"Uniforms outside [0, 1) in {path}")
        samples.append({"label": label, "path": str(path.resolve()), "sha256": file_sha256(path),
                        "feeds": feeds, "inputs": {name: array_info(value) for name, value in feeds.items()}})
    return samples, {"plan": str(plan.resolve()), "plan_sha256": digest,
                     "build_report": str(build_report_path.resolve()),
                     "build_report_sha256": file_sha256(build_report_path),
                     "onnx_sha256": entries[0].get("onnx_sha256"),
                     "contract_sha256": entries[0].get("contract_sha256")}


def validate_outputs(outputs):
    if set(outputs) != set(INTERFACE):
        raise ValueError(f"Unexpected output names: {sorted(outputs)}")
    for name, (shape, dtype) in INTERFACE.items():
        value = outputs[name]
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid output shape, dtype or finite values: {name}")
    if not np.isin(outputs["should_continue"], [0, 1]).all():
        raise ValueError("should_continue is not binary")
    if np.any(outputs["frame_token_ids"] < 0) or np.any(outputs["frame_token_ids"] >= 1024):
        raise ValueError("frame_token_ids is outside the original vocabulary")


def execution_order():
    # First cover all originals. Then vary both hidden vectors and uniforms,
    # and finally alternate distant inputs to expose stale captured buffers.
    return list(range(32)) + [hidden * 8 + uniform for uniform in reversed(range(8))
                             for hidden in (2, 0, 3, 1)] + [0, 31, 0, 31, 7, 24, 7, 24]


def graph_execution_checks(sessions, expected_runs):
    plain, captured = (sessions[name]["before_close"] for name in ("plain", "captured"))
    checks = {
        "plain_uses_ordinary_enqueue": (plain["graph"]["enabled"] is False
                                        and plain["graph"]["captures"] == 0 and plain["graph"]["replays"] == 0),
        "capture_really_executed": (captured["graph"]["enabled"] is True and captured["graph"]["ready"] is True
                                   and captured["graph"]["captures"] == 1
                                   and captured["graph"]["warmup_enqueues"] == 1
                                   and captured["graph"]["replays"] == expected_runs - 1
                                   and captured["graph"]["capture_failures"] == 0
                                   and captured["graph"]["last_error"] is None),
        "expected_run_counts": all(s["before_close"]["profile"]["runs"] == expected_runs
                                   and s["before_close"]["profile"]["errors"] == 0 for s in sessions.values()),
        "profile_agrees_with_actual_capture": all(
            captured["profile"]["cuda_graph_" + key] == captured["graph"][key]
            for key in ("captures", "replays", "warmup_enqueues", "capture_failures")),
        "sessions_closed": all(s["after_close"]["graph"]["closed"] is True
                               and s["after_close"]["graph"]["ready"] is False for s in sessions.values()),
        "device_allocations_released": all(s["after_close"]["profile"]["cuda_allocations"]
                                           == s["after_close"]["profile"]["cuda_frees"] for s in sessions.values()),
    }
    return checks


def run_validation(args):
    # Refuse overwrites, including a previous failed diagnosis.
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "ok": False, "validation_pass": False,
              "comparison": "exact dtype, shape and C-order bytes for all four outputs; no tolerance",
              "device": args.device, "samples": [], "runs": [], "errors": [], "sessions": {},
              "tool_sha256": file_sha256(__file__)}
    sessions, retained, first_results = {}, [], {}
    order = execution_order()
    report["sample_order"] = order
    started = time.perf_counter()
    try:
        samples, provenance = load_samples(args.plan, args.samples_dir)
        report.update(provenance)
        report["samples"] = [{key: value for key, value in sample.items() if key != "feeds"} for sample in samples]
        from moss_trt_session import CudaDriver, TensorRTSession
        import moss_trt_session
        report["runner_sha256"] = file_sha256(moss_trt_session.__file__)
        cuda = CudaDriver(args.device)
        report["gpu"] = cuda.info()
        for name, capture in (("plain", False), ("captured", True)):
            with session_environment(capture):
                session = TensorRTSession(args.plan, cuda)
                sessions[name] = session
                session.configure_profiling(GRAPH_NAME, enabled=True)
            if session.cuda_graph_snapshot()["enabled"] is not capture:
                raise RuntimeError(f"{name} did not freeze the requested CUDA Graph mode")
        for sequence, index in enumerate(order):
            sample = samples[index]
            names = ("plain", "captured") if sequence % 2 == 0 else ("captured", "plain")
            row = {"sequence": sequence, "sample": sample["label"], "session_order": list(names),
                   "all_outputs_exact": False, "repeat_exact": {}, "outputs": {}, "wall_seconds": {}}
            report["runs"].append(row)
            outputs = {}
            for name in names:
                began = time.perf_counter()
                outputs[name] = sessions[name].run(sample["feeds"])
                row["wall_seconds"][name] = time.perf_counter() - began
                validate_outputs(outputs[name])
                # Capture only the first 32 results as repetition references.
                key = (name, index)
                if key in first_results:
                    row["repeat_exact"][name] = all(compare_exact(first_results[key][tensor], value)["exact"]
                                                     for tensor, value in outputs[name].items())
                else:
                    first_results[key] = {tensor: value.copy() for tensor, value in outputs[name].items()}
                    row["repeat_exact"][name] = True
                retained.extend((sequence, name, tensor, value, array_info(value))
                                for tensor, value in outputs[name].items())
            row["outputs"] = {name: compare_exact(outputs["plain"][name], outputs["captured"][name])
                              for name in INTERFACE}
            row["all_outputs_exact"] = all(value["exact"] for value in row["outputs"].values())
            row["inputs_unchanged"] = all(array_info(value) == sample["inputs"][name]
                                           for name, value in sample["feeds"].items())
            if not row["all_outputs_exact"] or not all(row["repeat_exact"].values()) or not row["inputs_unchanged"]:
                artifact = f"mismatch-{sequence:03d}-{sample['label']}.npz"
                np.savez(args.output / artifact,
                         **{"input__" + name: value for name, value in sample["feeds"].items()},
                         **{mode + "__" + name: value for mode, values in outputs.items() for name, value in values.items()},
                         **{"first_" + mode + "__" + name: value for mode in names
                            for name, value in first_results[(mode, index)].items()})
                row["mismatch_artifact"] = artifact
            print(json.dumps({"sequence": sequence, "sample": sample["label"],
                              "all_outputs_exact": row["all_outputs_exact"], "repeat_exact": row["repeat_exact"]}), flush=True)
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    finally:
        for name, session in sessions.items():
            record = report["sessions"].setdefault(name, {})
            try:
                record["before_close"] = {"graph": session.cuda_graph_snapshot(), "profile": session.profile_snapshot()}
                session.close()
                record["after_close"] = {"graph": session.cuda_graph_snapshot(), "profile": session.profile_snapshot()}
            except Exception as error:
                report["errors"].append(f"{name} close/snapshot: {type(error).__name__}: {error}")
        report["output_lifetime_errors"] = [
            {"sequence": sequence, "session": mode, "tensor": name}
            for sequence, mode, name, value, expected in retained if array_info(value) != expected]
        try:
            report["execution_checks"] = graph_execution_checks(report["sessions"], len(order))
        except (KeyError, TypeError) as error:
            report["execution_checks"] = {"complete_session_evidence": False}
            report["errors"].append(f"Incomplete execution evidence: {error}")
        report["all_outputs_exact"] = (len(report["runs"]) == len(order)
                                        and all(row["all_outputs_exact"] for row in report["runs"]))
        report["repeat_inputs_exact"] = (len(report["runs"]) == len(order)
                                          and all(row["repeat_exact"] == {"plain": True, "captured": True}
                                                  for row in report["runs"]))
        report["inputs_unchanged"] = (len(report["runs"]) == len(order)
                                       and all(row.get("inputs_unchanged") is True for row in report["runs"]))
        report["ok"] = (not report["errors"] and not report["output_lifetime_errors"]
                        and report["all_outputs_exact"] and report["repeat_inputs_exact"]
                        and report["inputs_unchanged"] and all(report["execution_checks"].values()))
        report.update(status="passed" if report["ok"] else "failed", validation_pass=report["ok"],
                      elapsed_seconds=time.perf_counter() - started,
                      unique_samples=len(report["samples"]), completed_pairs=len(report["runs"]))
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"),
                          "all_outputs_exact": report["all_outputs_exact"],
                          "repeat_inputs_exact": report["repeat_inputs_exact"],
                          "execution_checks": report["execution_checks"], "errors": report["errors"]}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic directory; never overwritten")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    report = run_validation(args)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
