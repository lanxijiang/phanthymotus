"""One TensorRT engine per graph with distinct instance execution contexts.

This module is imported only for an explicit TensorRT backend request. GPU
validation/building belongs to the caller; plans must match the target GPU.
"""

from __future__ import annotations

import importlib.util
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

from .pool import SharedGraphPool
from .selected_logits import _sha256


def _validate_fp32_plans(assets, plan_paths):
    """Reject stale mixed-precision artifacts before TensorRT/CUDA is loaded.

    Export/package checks own ONNX tensor auditing and GPU validation. Here
    bind the selected plan bytes to their build/source metadata; the fused
    source has already passed the caller's versioned frame-contract checks.
    """
    reports = {}
    for name, path in plan_paths.items():
        path = Path(path)
        report_path = path.parent / "build_report.json"
        if report_path not in reports:
            reports[report_path] = json.loads(report_path.read_text(encoding="utf-8"))
        report = reports[report_path]
        entries = [row for row in report.get("engines", []) if row.get("name") == name]
        if report.get("status") != "built" or len(entries) != 1:
            raise ValueError(f"MOSS FP32 plan requires one successful build entry: {name}")
        entry = entries[0]
        if (entry.get("sha256") != _sha256(path)
                or entry.get("bytes") != path.stat().st_size
                or entry.get("tf32_enabled") is not False):
            raise ValueError(f"MOSS FP32 plan hash/size or TF32 provenance mismatch: {name}")
        precision = entry.get("precision")
        requested = entry.get("requested_precision")
        if "requested_precision" not in entry and name.startswith("codec_") and precision == "fp32":
            requested = "fp32"  # Reviewed original codec reports predate this field.
        if (requested != "fp32" or precision not in {"fp32", "onnx_explicit_types"}
                or precision == "onnx_explicit_types" and entry.get("strongly_typed") is not True):
            raise ValueError(f"MOSS active engine must use full FP32 computation: {name}")
        source_precision = entry.get("source_precision", "")
        if (not isinstance(source_precision, str) or not source_precision.startswith("fp32")
                or any(marker in source_precision.lower() for marker in
                       ("fp16", "bf16", "float16", "bfloat16", "mixed", "int8", "int4"))):
            raise ValueError(f"MOSS active source graph must be FP32: {name}")
        if name == "local_fixed_sampled_frame":
            if report.get("model_precision") != "fp32" or entry.get("model_precision") != "fp32":
                raise ValueError("MOSS fused frame must have FP32 model provenance")
            continue
        if source_precision != assets.graphs[name].get("precision", ""):
            raise ValueError(f"MOSS active source graph must be FP32: {name}")
        if entry.get("onnx_sha256") != _sha256(assets.graph_paths[name]):
            raise ValueError(f"MOSS FP32 engine/source hash mismatch: {name}")


def resolve_graph_plans(assets, plan_dir: str | Path | None, engine_paths: dict | None, encoder_context_mode: str):
    """Check plan availability before starting CUDA or deserializing weights.

    The original ONNX graph assets remain mandatory in ModelAssets. A missing
    full decoder plan is safe only when the verified native streaming step has
    both its metadata and a plan. Explicit full decoding stays unavailable.
    """
    if encoder_context_mode not in {"shared-serial", "per-instance", "disabled"}:
        raise ValueError("encoder_context_mode must be shared-serial, per-instance or disabled")
    references_cached = set(assets.precomputed_reference_codes) == set(assets.voice_paths)
    if encoder_context_mode == "disabled" and not references_cached:
        raise ValueError("Disabling the original encoder requires verified cached codes for every advertised built-in voice")
    supplied = engine_paths or {}
    def plan_path(name):
        return Path(supplied[name]) if name in supplied else Path(plan_dir) / (name + ".plan") if plan_dir is not None else None
    step_path = plan_path("codec_decode_step") if "codec_decode_step" in assets.graph_paths else None
    streaming_plan_ready = bool(assets.streaming_path is not None and step_path is not None and step_path.is_file())
    plan_paths = {}
    omitted = {}
    for name in assets.graph_paths:
        path = plan_path(name)
        if name == "codec_encoder" and encoder_context_mode == "disabled":
            omitted[name] = "custom reference encoding explicitly disabled; use verified original cached references"
            continue
        if path is None or not path.is_file():
            if name == "codec_encoder" and references_cached:
                omitted[name] = "encoder plan unavailable; all original built-in voices have verified cached reference codes"
                continue
            if name == "codec_decoder" and streaming_plan_ready:
                omitted[name] = "full decoder plan unavailable; use the verified codec_decode_step plan for streaming synthesis"
                continue
            raise FileNotFoundError(f"Missing original MOSS TensorRT plan for {name}: {path}")
        plan_paths[name] = path
    return plan_paths, omitted


class TensorRTGraphPool(SharedGraphPool):
    def __init__(self, assets, plan_dir: str | Path | None = None, *, device: int = 3, engine_paths: dict | None = None, optional_engine_paths: dict | None = None, driver_tool: str | Path | None = None, encoder_context_mode: str = "shared-serial"):
        self.encoder_context_mode = encoder_context_mode
        plan_paths, omitted = resolve_graph_plans(assets, plan_dir, engine_paths, encoder_context_mode)
        optional_engine_paths = dict(optional_engine_paths or {})
        if set(optional_engine_paths) - {"local_fixed_sampled_frame"}:
            raise ValueError("Unsupported optional MOSS TensorRT graph")
        if set(optional_engine_paths) & set(assets.graph_paths):
            raise ValueError("Optional MOSS TensorRT graph cannot replace an asset graph")
        for name, path in optional_engine_paths.items():
            path = Path(path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing optional MOSS TensorRT plan for {name}: {path}")
            plan_paths[name] = path
        if "local_fixed_sampled_frame" in optional_engine_paths and "local_cached_step" in plan_paths:
            # The caller validates the fused contract before constructing the
            # pool. Keep the original graph in ModelAssets for provenance, but
            # do not retain duplicate local weights in each inference process.
            # Its runner remains present and fails explicitly if reached.
            del plan_paths["local_cached_step"]
            omitted["local_cached_step"] = (
                "superseded by local_fixed_sampled_frame; local step inference is disabled")
        self.effective_plan_paths = {name: Path(path).resolve() for name, path in plan_paths.items()}
        _validate_fp32_plans(assets, self.effective_plan_paths)
        graph_names = tuple(assets.graph_paths) + tuple(optional_engine_paths)
        import tensorrt as trt
        tool_path = Path(driver_tool) if driver_tool else Path(__file__).resolve().parents[2] / "tools/trt_session.py"
        if not tool_path.is_file():
            raise FileNotFoundError(f"Existing CUDA/NumPy TensorRT runner is required: {tool_path}")
        spec = importlib.util.spec_from_file_location("_moss_deploy_trt_session", tool_path)
        runner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner_module)
        self.runner_module = runner_module
        self.device = int(device)
        self.profile_enabled = os.environ.get("MOSS_TRT_PROFILE") == "1"
        # Freeze the policy before lazy contexts are created. Only this MOSS
        # pool defaults to resident state; generic builder sessions do not.
        self.resident_state_enabled = os.environ.get("MOSS_TRT_RESIDENT_STATE", "1") == "1"
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.logger)
        self.loader_cuda = runner_module.CudaDriver(self.device)
        self.engines = {}
        self.omitted_graphs = omitted
        self.closed = False
        self.instance_cudas = {}
        self.instance_sessions = {}
        self.context_inventory_lock = threading.Lock()
        self.shared_encoder_cuda = None
        self.shared_encoder_session = None
        self.shared_encoder_lock = threading.Lock()
        try:
            for name, path in plan_paths.items():
                engine = self.trt_runtime.deserialize_cuda_engine(path.read_bytes())
                if engine is None:
                    raise RuntimeError(f"Cannot deserialize target-specific MOSS plan: {path}")
                self.engines[name] = engine
        except Exception as error:
            try:
                self.loader_cuda.call("cuCtxSetCurrent", self.loader_cuda.context)
                engine = None  # Drop any last successful local engine reference.
                self.close()
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"TensorRT constructor cleanup failed: {cleanup_error}")
            raise

        def create_session(engine, cuda, label, graph_name):
            session = object.__new__(runner_module.TensorRTSession)
            session.cuda = cuda
            session.logger = self.logger
            session.runtime = self.trt_runtime
            session.engine = engine
            session.context = engine.create_execution_context()
            if session.context is None:
                raise RuntimeError(f"Cannot lazily create TensorRT context for {label}")
            session.configure_profiling(graph_name, enabled=self.profile_enabled,
                                        resident_state=self.resident_state_enabled)
            return session

        def shared_encoder(feeds):
            # Encoder is stateless request preprocessing. One explicitly
            # serialized context/stream avoids three high-water allocations.
            with self.shared_encoder_lock:
                if self.closed:
                    raise RuntimeError("MOSS TensorRT graph pool is closed")
                if self.shared_encoder_cuda is None:
                    self.shared_encoder_cuda = runner_module.CudaDriver(self.device)
                cuda = self.shared_encoder_cuda
                cuda.call("cuCtxSetCurrent", cuda.context)
                if self.shared_encoder_session is None:
                    self.shared_encoder_session = create_session(self.engines["codec_encoder"], cuda, "shared-serial codec_encoder", "codec_encoder")
                return self.shared_encoder_session.run(feeds)

        def factory(instance_id):
            if self.closed:
                raise RuntimeError("MOSS TensorRT graph pool is closed")
            cuda = runner_module.CudaDriver(self.device)
            sessions = {}
            runners = {}
            for name in graph_names:
                if name in self.omitted_graphs:
                    def missing(feeds, name=name):
                        raise FileNotFoundError(f"Original MOSS graph {name} is unavailable: {self.omitted_graphs[name]}; no substitute encoder or audio will be used")
                    runners[name] = missing
                    continue
                if name == "codec_encoder" and self.encoder_context_mode == "shared-serial":
                    runners[name] = shared_encoder
                    continue
                lock = threading.Lock()
                def run(feeds, name=name, lock=lock, cuda=cuda, sessions=sessions, instance_id=instance_id):
                    with lock:
                        if self.closed:
                            raise RuntimeError("MOSS TensorRT graph pool is closed")
                        # HTTP/worker threads need this device context current;
                        # each instance owns its own stream and TRT contexts.
                        cuda.call("cuCtxSetCurrent", cuda.context)
                        session = sessions.get(name)
                        if session is None:
                            session = create_session(self.engines[name], cuda, f"{instance_id}/{name}", name)
                            with self.context_inventory_lock:
                                sessions[name] = session
                        return session.run(feeds)
                runners[name] = run
            with self.context_inventory_lock:
                self.instance_cudas[instance_id] = cuda
                self.instance_sessions[instance_id] = sessions
            return runners
        super().__init__(assets, factory, backend=f"tensorrt-device-{self.device}")

    def performance_snapshot(self) -> dict | None:
        """Aggregate opt-in CPU timing counters after active requests finish."""
        if not self.profile_enabled:
            return None
        with self.context_inventory_lock:
            sessions = [(instance_id, name, session)
                        for instance_id, graph_sessions in self.instance_sessions.items()
                        for name, session in graph_sessions.items()]
        if self.shared_encoder_session is not None:
            sessions.append(("shared-encoder", "codec_encoder", self.shared_encoder_session))
        by_instance = []
        by_graph = {}
        for instance_id, name, session in sessions:
            row = session.profile_snapshot()
            if row is None:
                continue
            row["instance_id"] = instance_id
            by_instance.append(row)
            totals = by_graph.setdefault(name, {key: 0 for key in row if key not in {"graph", "instance_id"}})
            for key in totals:
                totals[key] += row[key]
        return {
            "timing_scope": "CPU wall clock around the existing synchronous CUDA/TensorRT calls; graph totals can overlap across instances",
            "by_graph": by_graph,
            "by_instance": by_instance,
        }

    def memory_snapshot(self) -> dict:
        """Observed device free bytes plus lazy context inventory, not estimates."""
        if self.closed:
            raise RuntimeError("MOSS TensorRT graph pool is closed")
        self.loader_cuda.call("cuCtxSetCurrent", self.loader_cuda.context)
        gpu = self.loader_cuda.info()
        observed_at = time.perf_counter()
        with self.context_inventory_lock:
            contexts = {instance: sorted(sessions) for instance, sessions in self.instance_sessions.items()}
            sessions = [(instance, name, session)
                        for instance, graph_sessions in self.instance_sessions.items()
                        for name, session in graph_sessions.items()]
        graph_states = [dict(session.cuda_graph_snapshot(), instance_id=instance, graph=name)
                        for instance, name, session in sessions
                        if callable(getattr(session, "cuda_graph_snapshot", None))]
        return {"gpu": gpu, "observed_at_monotonic_seconds": observed_at, "snapshot_scope": "GPU free and context inventory are adjacent observations, not an atomic per-process peak", "context_allocation": "lazy on first graph run", "encoder_context_mode": self.encoder_context_mode, "encoder_parallelism": 0 if "codec_encoder" in self.omitted_graphs else 1 if self.encoder_context_mode == "shared-serial" else "one per active instance", "materialized_instance_contexts": contexts, "shared_encoder_context_materialized": self.shared_encoder_session is not None, "loaded_shared_engines": sorted(self.engines), "omitted_graphs": dict(self.omitted_graphs), "cuda_graph_sessions": graph_states}

    def close(self):
        """After inference has stopped, release contexts, weights and streams."""
        if self.closed:
            return
        trace_enabled = os.environ.get("MOSS_TRT_SHUTDOWN_TRACE") == "1"
        self.shutdown_trace = []
        def trace(phase):
            self.shutdown_trace.append({"phase": phase, "elapsed_seconds": time.perf_counter() - began})
            if trace_enabled:
                print(f"[moss-trt-close] {phase}", file=sys.stderr, flush=True)
        began = time.perf_counter()
        self.closed = True
        # A shutdown/constructor-error caller can be a different thread from
        # the model loader. Activate this primary context before GPU destructors.
        trace("activate_primary_context")
        self.loader_cuda.call("cuCtxSetCurrent", self.loader_cuda.context)
        trace("drop_execution_contexts")
        if self.shared_encoder_session is not None:
            self.shared_encoder_session.close()
            self.shared_encoder_session.context = None
            self.shared_encoder_session.engine = None
            self.shared_encoder_session.runtime = None
            self.shared_encoder_session.logger = None
            self.shared_encoder_session.cuda = None
            self.shared_encoder_session = None
        with self.context_inventory_lock:
            instance_sessions = list(self.instance_sessions.values())
            instance_cudas = list(self.instance_cudas.values())
            self.instance_sessions.clear()
            self.instance_cudas.clear()
        for sessions in instance_sessions:
            for session in sessions.values():
                # Runners can retain the session wrapper; remove its GPU
                # objects explicitly instead of relying on closure GC.
                session.close()
                session.context = None
                session.engine = None
                session.runtime = None
                session.logger = None
                session.cuda = None
            sessions.clear()
        trace("drop_engines_and_runtime")
        self.engines.clear()
        self.trt_runtime = None
        self.logger = None
        drivers = [*instance_cudas, self.loader_cuda]
        if self.shared_encoder_cuda is not None:
            drivers.append(self.shared_encoder_cuda)
        for index, cuda in enumerate(drivers):
            trace(f"synchronize_stream_{index}")
            cuda.call("cuCtxSetCurrent", cuda.context)
            cuda.call("cuStreamSynchronize", cuda.stream)
            trace(f"destroy_stream_{index}")
            cuda.call("cuStreamDestroy_v2", cuda.stream)
            cuda.stream = ctypes.c_void_p()
        # Each CudaDriver retained the same primary context. Remove it from
        # this thread before the last release can destroy it; all TensorRT
        # wrappers and streams have already been destroyed above.
        trace("clear_current_context")
        self.loader_cuda.call("cuCtxSetCurrent", ctypes.c_void_p())
        for index, cuda in enumerate(drivers):
            trace(f"release_primary_context_{index}")
            release = "cuDevicePrimaryCtxRelease_v2" if hasattr(cuda.lib, "cuDevicePrimaryCtxRelease_v2") else "cuDevicePrimaryCtxRelease"
            cuda._bind(release, [ctypes.c_int])
            cuda.call(release, cuda.device)
            cuda.context = ctypes.c_void_p()
        self.shared_encoder_cuda = None
        trace("closed")
