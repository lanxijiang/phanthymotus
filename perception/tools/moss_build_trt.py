"""Build the prepared ONNX graphs on the current GPU; plans are target specific."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import onnx
import tensorrt as trt

from moss_trt_session import CudaDriver, TensorRTSession, compare


def exact_output_names(graph, source):
    """Floating Ceil durations control frame counts and require exact values."""
    model = onnx.load(source, load_external_data=False)
    outputs = {value.name for value in model.graph.output}
    exact = {name for node in model.graph.node if node.op_type == "Ceil"
             for name in node.output if name in outputs}
    duration = graph.get("semantic_outputs", {}).get("duration")
    if duration is not None:
        if duration not in outputs:
            raise ValueError(f"Unknown semantic duration output: {duration}")
        exact.add(duration)
    return sorted(exact)


def build(graph, directory, output, args):
    # The encoder was deliberately kept in FP32 because FP16 changed discrete
    # audio codes. Do not undo that tested fallback when building the bundle.
    source_precision = graph.get("precision", "")
    precision = ("fp32" if graph["name"] in args.fp32_graph
                 or source_precision.startswith("fp32") else args.precision)
    source = directory / graph["file"]
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder.max_threads = 2
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(source)):
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i))
                                                            for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024 * 1024)
    config.builder_optimization_level = 1
    config.avg_timing_iterations = 1
    if args.disable_jit_convolutions:
        sources = config.get_tactic_sources()
        config.set_tactic_sources(sources & ~(1 << int(trt.TacticSource.JIT_CONVOLUTIONS)))
    # This also protects FP32-sensitive operations in the mixed FP16 graphs.
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("GPU does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    # Prefer accurate EXP/LOG and reductions: duration errors can change audio length.
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # get_layer returns the base ILayer in TensorRT 10.1, without .op.
            # Keep floating unary math in FP32; boolean/integer shape math
            # retains its actual type.
            if (layer.type == trt.LayerType.UNARY and layer.num_inputs
                    and layer.get_input(0).dtype in (trt.float16, trt.float32)):
                layer.precision = trt.float32
    profile = builder.create_optimization_profile()
    profiles = graph.get("profiles", {})
    shape_values = graph.get("shape_values", {})
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if tensor.is_shape_tensor:
            if tensor.name not in shape_values:
                raise RuntimeError(f"Shape tensor {tensor.name} requires explicit value profiles")
            values = shape_values[tensor.name]
            if profile.set_shape_input(tensor.name, values["min"], values["opt"], values["max"]) is False:
                raise ValueError(f"Invalid shape values: {tensor.name}")
        if any(dimension < 0 for dimension in tensor.shape):
            if tensor.name not in profiles:
                raise RuntimeError(f"Dynamic input {tensor.name} requires an explicit dimension profile")
            shapes = profiles[tensor.name]
            if profile.set_shape(tensor.name, shapes["min"], shapes["opt"], shapes["max"]) is False:
                raise ValueError(f"Invalid profile: {tensor.name}")
    if profiles or shape_values:
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError(f"Cannot add optimization profile: {graph['name']}")
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed: {graph['name']}")
    engine_bytes = bytes(serialized)
    destination = output / (graph["name"] + ".plan")
    destination.write_bytes(engine_bytes)
    return {"name": graph["name"], "file": destination.name, "bytes": len(engine_bytes), "precision": precision,
            "source_precision": source_precision, "tf32_enabled": False,
            "jit_convolutions_disabled": args.disable_jit_convolutions,
            "exact_validation_outputs": exact_output_names(graph, source),
            "sha256": hashlib.sha256(engine_bytes).hexdigest(),
            "onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "build_seconds": time.perf_counter() - started, "profiles": profiles, "shape_values": shape_values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--fp32-graph", action="append", default=[], help="Keep these graphs in FP32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--disable-jit-convolutions", action="store_true", help="Exclude JIT convolution tactics when measuring compact engine packages")
    parser.add_argument("--max-model-bytes", type=int, default=None, help="Optional explicit engine size gate; latest requirements do not impose a fixed size cap")
    parser.add_argument("--graph", action="append", help="Build only these graph names")
    parser.add_argument("--allow-over-budget", action="store_true", help="Keep experimental plans and record a budget failure")
    parser.add_argument("--max-relative-rms", type=float, default=0.05)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing_plans = list(args.output.glob("*.plan"))
    if existing_plans:
        raise FileExistsError(
            f"Refusing to overwrite existing TensorRT plans in {args.output}; "
            "choose a new output directory. Existing plans and reports are unchanged."
        )
    analysis = json.loads(args.analysis.read_text())
    selected = [graph for graph in analysis["graphs"] if not args.graph or graph["name"] in args.graph]
    if args.graph:
        unknown = set(args.graph) - {graph["name"] for graph in analysis["graphs"]}
        if unknown:
            raise ValueError(f"Unknown graph names: {sorted(unknown)}")
    if not selected:
        raise ValueError("No graphs selected")
    report = {"tensorrt_version": trt.__version__, "gpu": None, "precision": args.precision,
              "workspace_bytes": args.workspace_mb * 1024 * 1024,
              "max_model_bytes": args.max_model_bytes, "engines": [],
              "deployment_note": "Built on the listed GPU. Rebuild on the deployment platform; "
                                 "A100 plans cannot be used as Jetson Orin deployment plans."}
    published_paths = []
    try:
        for graph in selected:
            if not graph.get("validation_samples"):
                raise ValueError(f"No validation samples for {graph['name']}; refusing to publish an unchecked plan")
        cuda = CudaDriver(args.device)
        report["gpu"] = cuda.info()
        print(json.dumps({"tensorrt": trt.__version__, "gpu": report["gpu"]}, ensure_ascii=False), flush=True)
        # Keep unchecked engines away from the paths used by the TTS adapter.
        # The staging directory is on the destination filesystem, not /tmp.
        with tempfile.TemporaryDirectory(prefix=".moss-build-", dir=args.output) as staging_name:
            staging = Path(staging_name)
            for graph in selected:
                entry = build(graph, args.analysis.parent, staging, args)
                report["engines"].append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                for sample in graph["validation_samples"]:
                    with np.load(args.analysis.parent / sample, allow_pickle=False) as data:
                        inputs = {name.removeprefix("input__"): data[name] for name in data.files
                                  if name.startswith("input__")}
                        session = TensorRTSession(staging / entry["file"], cuda)
                        try:
                            actual = session.run(inputs)
                        finally:
                            del session
                        expected_outputs = {name.removeprefix("ref__") for name in data.files
                                            if name.startswith("ref__")}
                        if set(actual) != expected_outputs:
                            raise RuntimeError(f"Validation output mismatch for {graph['name']}: "
                                               f"missing={sorted(expected_outputs - set(actual))}, "
                                               f"unexpected={sorted(set(actual) - expected_outputs)}")
                        comparison = {name: compare(value, data["ref__" + name])
                                      for name, value in actual.items()}
                        for name, result in comparison.items():
                            reference = data["ref__" + name]
                            if (name in entry["exact_validation_outputs"]
                                    or np.issubdtype(reference.dtype, np.integer) or reference.dtype == np.bool_):
                                match = bool(np.array_equal(actual[name], reference))
                                result["exact_value_match_required"] = True
                            else:
                                match = (result.get("max_abs_error", float("inf")) <= 1e-4
                                         or result.get("relative_rms_error", float("inf")) <= args.max_relative_rms)
                            result["numerical_pass"] = bool(result.get("shape_match") and result.get("finite") and match)
                    entry.setdefault("validation", []).append({"sample": sample, "outputs": comparison})
                    np.savez(staging / (Path(sample).stem + "_trt.npz"), **actual)
                    print(json.dumps({"validation": sample, "outputs": comparison}), flush=True)
                    if not comparison or not all(result["numerical_pass"] for result in comparison.values()):
                        raise RuntimeError(f"Numerical validation failed for {graph['name']}: {comparison}")
            report["total_engine_bytes"] = sum(entry["bytes"] for entry in report["engines"])
            report["engine_budget_pass"] = (report["total_engine_bytes"] <= args.max_model_bytes
                                            if args.max_model_bytes is not None else None)
            if (args.max_model_bytes is not None and not report["engine_budget_pass"]
                    and not args.allow_over_budget):
                raise RuntimeError(f"Engine bundle exceeds {args.max_model_bytes} bytes")
            for entry in report["engines"]:
                # Hard links publish each validated file atomically and refuse
                # an existing destination, including a concurrent builder's plan.
                destination = args.output / entry["file"]
                os.link(staging / entry["file"], destination)
                published_paths.append(destination)
            for sample_output in staging.glob("*.npz"):
                sample_output.replace(args.output / sample_output.name)
            report["status"] = "built"
    except Exception as error:
        for path in published_paths:
            path.unlink(missing_ok=True)
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (args.output / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
