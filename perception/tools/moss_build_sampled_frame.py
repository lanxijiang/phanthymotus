#!/usr/bin/env python3
"""Build and validate the actual fused sampled-frame engine on the target GPU.

The candidate's sampled history is replayed through the original step engine,
so numerical comparisons always have the same token context. Independently,
every emitted integer is checked against sampling from the candidate's actual
logits and supplied uniforms. A failed check never publishes an engine.

The NumPy sampler is a validation oracle, not CPU model inference. Its FP32
contract intentionally does not promise old FP64 host-sampler seed equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np


GRAPH_NAME = "local_fixed_sampled_frame"
CDF_NORMALIZATION = "divide_by_last_cumulative_probability"
EXPECTED_SAMPLING = {
    "do_sample": True, "text_temperature": 1.5, "text_top_p": 1.0,
    "text_top_k": 50, "audio_temperature": 1.7, "audio_top_p": 0.8,
    "audio_top_k": 25, "audio_repetition_penalty": 1.0,
}
OUTPUTS = {"should_continue", "frame_token_ids", "text_logits", "audio_logits"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def softmax_f32(scores):
    shifted = np.asarray(scores, dtype=np.float32) - np.max(scores)
    exps = np.exp(shifted, dtype=np.float32)
    return exps / np.sum(exps, dtype=np.float32)


def sample_f32(logits, random_u, *, temperature, top_k, top_p):
    """Independent scalar sampler matching the exported FP32 filtering contract."""
    logits = np.asarray(logits, dtype=np.float32)
    uniform = np.float32(random_u)
    if logits.ndim != 1 or not logits.size or not np.isfinite(logits).all():
        raise ValueError("Sampling logits must be a nonempty finite vector")
    if not np.isfinite(uniform) or not 0 <= uniform < 1 or not temperature > 0:
        raise ValueError("Invalid uniform or temperature")
    scores = logits / np.float32(temperature)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite temperature-scaled logits")
    if 0 < top_k < scores.size or 0 < top_p < 1:
        order = np.argsort(-scores, kind="stable")
        ordered = scores[order].copy()
        if 0 < top_k < scores.size:
            ordered[ordered < ordered[top_k - 1]] = -np.inf
        if 0 < top_p < 1:
            probabilities = softmax_f32(ordered)
            # ONNX CumSum(exclusive=1), not inclusive_sum - current_probability.
            previous = np.concatenate((np.zeros(1, dtype=np.float32),
                                       np.cumsum(probabilities[:-1], dtype=np.float32)))
            ordered[previous > np.float32(top_p)] = -np.inf
        scores[order] = ordered
    probabilities = softmax_f32(scores)
    cdf = np.cumsum(probabilities, dtype=np.float32)
    cumulative_total = float(cdf[-1])
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("Invalid cumulative probability total")
    # The reduction used by softmax and the prefix-sum kernel can accumulate
    # differently. Normalize the entire CDF by its OWN last cumulative value.
    # This also sets every trailing zero-probability plateau to exactly one;
    # changing only the final array element could select a masked token.
    cdf = cdf / cdf[-1]
    selected = int(np.count_nonzero(cdf < uniform))
    fallback = selected >= scores.size
    if fallback:
        selected = int(np.argmax(scores))
    return selected, {
        "uniform": float(uniform), "cdf_boundary_distance": float(np.min(np.abs(cdf - uniform))),
        "cdf_last": float(cdf[-1]), "rounding_fallback": fallback,
        "cdf_total_before_normalization": cumulative_total,
        "unmasked_tokens": int(np.isfinite(scores).sum()),
    }


def uniform_vectors():
    """Predeclared fixtures; never select uniforms after observing model output."""
    rng = np.random.default_rng(20261010)
    near_one = np.nextafter(np.float32(1), np.float32(0))
    rows = rng.random((8, 17), dtype=np.float32)
    rows[0] = 0
    rows[1] = near_one
    rows[2, 0] = 0
    rows[3, 0] = near_one
    rows[4] = np.linspace(0, near_one, 17, dtype=np.float32)
    rows[5] = np.linspace(near_one, 0, 17, dtype=np.float32)
    return rows


def check_outputs(outputs):
    expected = {"should_continue": ((1, 1), np.int32),
                "frame_token_ids": ((1, 16), np.int32),
                "text_logits": ((1, 2), np.float32),
                "audio_logits": ((1, 16, 1024), np.float32)}
    if set(outputs) != OUTPUTS:
        raise ValueError("Unexpected sampled-frame output names")
    for name, (shape, dtype) in expected.items():
        value = outputs[name]
        if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape, dtype or finite values: {name}")
    if not np.isin(outputs["should_continue"], [0, 1]).all():
        raise ValueError("Continuation output is not binary")
    if np.any(outputs["frame_token_ids"] < 0) or np.any(outputs["frame_token_ids"] >= 1024):
        raise ValueError("Audio code outside its original vocabulary")


def check_sampling(outputs, uniforms):
    check_outputs(outputs)
    vectors = [outputs["text_logits"][0], *outputs["audio_logits"][0]]
    actual = [0 if int(outputs["should_continue"].item()) else 1,
              *map(int, outputs["frame_token_ids"][0])]
    checks = []
    for step, (logits, uniform, token) in enumerate(zip(vectors, uniforms[0], actual, strict=True)):
        prefix = "text" if step == 0 else "audio"
        expected, diagnostic = sample_f32(
            logits, uniform, temperature=EXPECTED_SAMPLING[prefix + "_temperature"],
            top_k=EXPECTED_SAMPLING[prefix + "_top_k"], top_p=EXPECTED_SAMPLING[prefix + "_top_p"],
        )
        checks.append({"step": step, "actual": token, "expected": expected,
                       "exact": token == expected, **diagnostic})
    return checks


def _sampler_layer_indices(layers):
    """Include parser-generated loop bodies without crossing model boundaries.

    ONNX CumSum can expand into unnamed iterator/recurrence/elementwise layers.
    Those ancestors must receive the same precision constraint as the named
    sampler. Named non-sampler producers (model logits, uniforms, constants)
    are explicit boundaries, so the Transformer is not promoted to FP32.
    """
    producers = {}
    named = set()
    for index, layer in enumerate(layers):
        outputs = [layer.get_output(i) for i in range(layer.num_outputs)]
        if "sample/step_" in layer.name or any(
                tensor is not None and "sample/step_" in getattr(tensor, "name", "") for tensor in outputs):
            named.add(index)
        for tensor in outputs:
            if tensor is not None and getattr(tensor, "name", ""):
                producers[tensor.name] = index
    selected, pending = set(named), list(named)
    while pending:
        layer = layers[pending.pop()]
        for position in range(layer.num_inputs):
            tensor = layer.get_input(position)
            if tensor is None:
                continue
            producer = producers.get(getattr(tensor, "name", ""))
            if producer is None or producer in selected:
                continue
            candidate = layers[producer]
            if (producer in named or not candidate.name
                    or candidate.name.startswith("(Unnamed Layer")):
                selected.add(producer)
                pending.append(producer)
    return selected


def force_sampler_fp32(network, config, trt):
    """Constrain float arithmetic AND float-input comparisons to FP32."""
    config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    constrained = []
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    selected = _sampler_layer_indices(layers)
    for index in sorted(selected):
        layer = layers[index]
        floating_outputs = [i for i in range(layer.num_outputs)
                            if layer.get_output(i) is not None
                            and layer.get_output(i).dtype in (trt.float16, trt.float32)]
        floating_inputs = any(layer.get_input(i) is not None
                              and layer.get_input(i).dtype in (trt.float16, trt.float32)
                              for i in range(layer.num_inputs))
        if not floating_outputs and not floating_inputs:
            continue
        # Less/Greater produce bool but must compare FP32 scores: comparing
        # their half-rounded inputs changes kth ties and top-p membership.
        # Pure routing layers do not perform floating arithmetic, but their
        # floating outputs must still remain FP32 between arithmetic layers.
        layer.precision = trt.float32
        for output_index in floating_outputs:
            layer.set_output_type(output_index, trt.float32)
        constrained.append(layer.name)
    if not constrained:
        raise ValueError("No named sampler float layers found; refusing unconstrained sampling")
    return constrained


def numerical_result(actual, reference, compare):
    result = compare(actual, reference)
    result["reference_finite"] = bool(np.isfinite(reference).all())
    result["numerical_pass"] = bool(
        result.get("shape_match") and result.get("finite") and result["reference_finite"]
        and (result.get("max_abs_error", float("inf")) <= 1e-4
             or result.get("relative_rms_error", float("inf")) <= 0.05))
    return result


def replay_reference(reference, sample, outputs, text_ids, compare):
    """Replay candidate token history, carrying the original engine's own KV."""
    pairs = {key: key.replace("local_past_", "local_present_", 1)
             for key in sample if key.startswith("local_past_")}
    if not pairs:
        raise ValueError("Original validation sample has no local KV inputs")
    states = {key: np.zeros((1, 0, *sample[key].shape[2:]), dtype=np.float32) for key in pairs}
    hidden = np.asarray(sample["global_hidden"], dtype=np.float32)
    text_token = text_ids[0 if int(outputs["should_continue"].item()) else 1]
    frame = outputs["frame_token_ids"][0]
    checks = []
    for step in range(17):
        feeds = {"global_hidden": hidden, "past_valid_lengths": np.asarray([step], dtype=np.int32),
                 "step_type": np.asarray([min(step, 2)], dtype=np.int32),
                 "channel_index": np.asarray([max(step - 2, 0)], dtype=np.int32),
                 "text_token_id": np.asarray([text_token if step == 1 else 0], dtype=np.int32),
                 "audio_token_id": np.asarray([int(frame[step - 2]) if step >= 2 else 0], dtype=np.int32),
                 **states}
        result = reference.run(feeds)
        states = {key: result[value] for key, value in pairs.items()}
        baseline = (result["text_logits"].reshape(-1)[text_ids] if step == 0
                    else result["audio_logits"].reshape(16, 1024)[step - 1])
        candidate = outputs["text_logits"][0] if step == 0 else outputs["audio_logits"][0, step - 1]
        check = numerical_result(candidate, baseline, compare)
        check.update(step=step, scope="assistant_logits" if step == 0 else f"audio_channel_{step - 1}")
        checks.append(check)
    return checks


def validate_contract(contract, graph, source_graph, assets, analysis_directory):
    if (contract.get("version") != 1 or contract.get("graph_name") != GRAPH_NAME
            or contract.get("audio_channels") != 16 or contract.get("audio_vocab_size") != 1024
            or contract.get("steps_per_frame") != 17 or contract.get("batch_size") != 1
            or contract.get("text_token_ids") != [9, 7]
            or contract.get("sampling") != EXPECTED_SAMPLING
            or contract.get("sampling_semantics", {}).get("cdf_normalization") != CDF_NORMALIZATION
            or graph.get("name") != GRAPH_NAME):
        raise ValueError("Unsupported sampled-frame contract")
    if contract.get("source_onnx_sha256") != sha256(assets / source_graph["file"]):
        raise ValueError("Candidate refers to another original step graph")
    if contract.get("onnx_sha256") != sha256(analysis_directory / graph["file"]):
        raise ValueError("Candidate ONNX/contract hash mismatch")
    if not contract.get("external_weights"):
        raise ValueError("Candidate contract has no original external-weight hashes")
    for name, expected in contract["external_weights"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe external weight name")
        actual = analysis_directory / path
        if actual.stat().st_size != expected["bytes"] or sha256(actual) != expected["sha256"]:
            raise ValueError(f"Candidate external-weight hash mismatch: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=1)
    args = parser.parse_args(argv)
    # Preserve successful AND failed prior evidence rather than overwriting it.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {args.output}")
    report = {"status": "failed", "validation_pass": False, "engines": [],
              "numerical_gate": "Same token context; max_abs_error <= 1e-4 OR relative_rms_error <= 0.05; matching shapes and finite values",
              "integer_gate": "All 17 sampler outputs must exactly match independent FP32 sampling of the actual candidate logits",
              "validation_scope": "Actual production plan. 17 aligned logits; candidate internal KV is checked indirectly through subsequent logits, not directly returned. Does not establish eight-language quality or old FP64 sampler seed equivalence.",
              "uniform_seed": 20261010, "uniforms_per_hidden": 8}
    sessions, published = [], None
    try:
        contract = json.loads(args.contract.read_text())
        analysis = json.loads(args.analysis.read_text())
        if len(analysis["graphs"]) != 1:
            raise ValueError("Expected exactly one fused-frame graph")
        graph = analysis["graphs"][0]
        original = json.loads((args.assets / "analysis.json").read_text())
        source_graph = next(row for row in original["graphs"] if row["name"] == "local_cached_step")
        validate_contract(contract, graph, source_graph, args.assets, args.analysis.parent)
        samples = []
        for name in source_graph["validation_samples"]:
            with np.load(args.assets / name, allow_pickle=False) as archive:
                sample = {key.removeprefix("input__"): archive[key].copy()
                          for key in archive.files if key.startswith("input__")}
            if sample["global_hidden"].shape != (1, 768) or not np.isfinite(sample["global_hidden"]).all():
                raise ValueError("Invalid real original global-hidden sample")
            samples.append((name, sample))
        if len(samples) != 4:
            raise ValueError("Expected all four original local-step validation samples")
        baseline_plan = args.assets / "plans/local_cached_step/local_cached_step.plan"
        baseline_report = json.loads((baseline_plan.parent / "build_report.json").read_text())
        baseline_entries = [row for row in baseline_report.get("engines", []) if row.get("name") == "local_cached_step"]
        if (baseline_report.get("status") != "built" or len(baseline_entries) != 1
                or baseline_entries[0].get("sha256") != sha256(baseline_plan)
                or baseline_entries[0].get("onnx_sha256") != contract["source_onnx_sha256"]
                or not baseline_entries[0].get("validation")
                or not all(row.get("outputs") and all(value.get("numerical_pass") for value in row["outputs"].values())
                           for row in baseline_entries[0]["validation"])):
            raise ValueError("Original local-step plan lacks matching verified provenance")
        report.update(reference_plan_sha256=sha256(baseline_plan),
                      source_onnx_sha256=contract["source_onnx_sha256"],
                      contract_sha256=sha256(args.contract), sampling=EXPECTED_SAMPLING,
                      cdf_normalization=CDF_NORMALIZATION,
                      reference_samples=[{"path": name, "sha256": sha256(args.assets / name)} for name, _ in samples])
        # Import GPU dependencies only after all artifact provenance checks.
        import tensorrt as trt
        from moss_build_trt import build
        from moss_trt_session import CudaDriver, TensorRTSession, compare
        os.environ["MOSS_TRT_RESIDENT_STATE"] = "0"
        os.environ["MOSS_TRT_REUSE_LOCAL_BUFFERS"] = "0"
        cuda = CudaDriver(args.device)
        report.update(gpu=cuda.info(), tensorrt_version=trt.__version__)
        print(json.dumps({"gpu": report["gpu"], "building": GRAPH_NAME}), flush=True)
        build_args = SimpleNamespace(fp32_graph=[], precision="fp16", workspace_mb=args.workspace_mb,
                                     builder_optimization_level=args.builder_optimization_level,
                                     disable_jit_convolutions=False, strongly_typed=True,
                                     detailed_profiling=True)
        with tempfile.TemporaryDirectory(prefix=".moss-frame-build-", dir=args.output) as staging_name:
            staging = Path(staging_name)
            entry = build(graph, args.analysis.parent, staging, build_args)
            entry.update(contract_sha256=report["contract_sha256"],
                         sampler_precision="FP32 explicit ONNX tensor types in a STRONGLY_TYPED network; model FP16 types retained",
                         sampler_precision_layers=[], validation=[])
            report["engines"].append(entry)
            report["strongly_typed"] = True
            candidate = TensorRTSession(staging / entry["file"], cuda)
            sessions.append(candidate)
            reference = TensorRTSession(baseline_plan, cuda)
            sessions.append(reference)
            for sample_index, (sample_name, sample) in enumerate(samples):
                for uniform_index, vector in enumerate(uniform_vectors()):
                    label = f"hidden-{sample_index}-uniform-{uniform_index}"
                    row = {"sample": label, "original_sample": sample_name,
                           "model_logits_pass": False, "sampling_exact": False,
                           "finite": False, "outputs_valid": False}
                    entry["validation"].append(row)
                    uniforms = vector.reshape(1, 17)
                    outputs = candidate.run({"global_hidden": sample["global_hidden"], "random_u": uniforms})
                    check_outputs(outputs)
                    row.update(finite=True, outputs_valid=True,
                               should_continue=int(outputs["should_continue"].item()))
                    row["sampling_checks"] = check_sampling(outputs, uniforms)
                    row["sampling_exact"] = all(check["exact"] for check in row["sampling_checks"])
                    row["logit_checks"] = replay_reference(reference, sample, outputs, contract["text_token_ids"], compare)
                    row["model_logits_pass"] = all(check["numerical_pass"] for check in row["logit_checks"])
                    # Keep mismatches in a portable NPZ/report; never overwrite a
                    # reference fixture with values produced by the candidate.
                    np.savez(args.output / (label + ".npz"), input__global_hidden=sample["global_hidden"],
                             input__random_u=uniforms, **outputs)
                    print(json.dumps({"validation": label, "model_logits_pass": row["model_logits_pass"],
                                      "sampling_exact": row["sampling_exact"],
                                      "should_continue": row["should_continue"]}), flush=True)
            checks = entry["validation"]
            decisions = sorted({row["should_continue"] for row in checks})
            report.update(sample_count=len(checks), continuation_values_covered=decisions,
                          audio_channels_covered=list(range(16)))
            report["validation_pass"] = (len(checks) == 32 and decisions == [0, 1]
                                         and all(all(row[key] for key in ("model_logits_pass", "sampling_exact", "finite", "outputs_valid")) for row in checks))
            if not report["validation_pass"]:
                raise RuntimeError("Fused-frame validation failed; see per-sample logits, exact sampling and continuation coverage")
            for session in reversed(sessions):
                session.close()
            sessions.clear()
            destination = args.output / entry["file"]
            os.link(staging / entry["file"], destination)
            published = destination
            report.update(status="built", total_engine_bytes=entry["bytes"])
    except BaseException as error:
        if published is not None:
            published.unlink(missing_ok=True)
        report.update(status="failed", validation_pass=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for session in reversed(sessions):
            try:
                session.close()
            except Exception as error:
                report.setdefault("close_errors", []).append(str(error))
        (args.output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
