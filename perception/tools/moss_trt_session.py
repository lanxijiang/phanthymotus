"""Small TensorRT 10 runtime using NumPy and the CUDA driver API."""
from __future__ import annotations

import ctypes as ct
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt


class CudaDriver:
    def __init__(self, device_id: int = 0):
        self.lib = ct.CDLL("libcuda.so.1")
        self._bind("cuInit", [ct.c_uint])
        self._bind("cuDeviceGet", [ct.POINTER(ct.c_int), ct.c_int])
        self._bind("cuDeviceGetName", [ct.c_char_p, ct.c_int, ct.c_int])
        self._bind("cuDeviceGetAttribute", [ct.POINTER(ct.c_int), ct.c_int, ct.c_int])
        self._bind("cuDevicePrimaryCtxRetain", [ct.POINTER(ct.c_void_p), ct.c_int])
        self._bind("cuCtxSetCurrent", [ct.c_void_p])
        self._bind("cuMemAlloc_v2", [ct.POINTER(ct.c_uint64), ct.c_size_t])
        self._bind("cuMemFree_v2", [ct.c_uint64])
        self._bind("cuMemcpyHtoD_v2", [ct.c_uint64, ct.c_void_p, ct.c_size_t])
        self._bind("cuMemcpyDtoH_v2", [ct.c_void_p, ct.c_uint64, ct.c_size_t])
        self._bind("cuMemGetInfo_v2", [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)])
        self._bind("cuStreamCreate", [ct.POINTER(ct.c_void_p), ct.c_uint])
        self._bind("cuStreamSynchronize", [ct.c_void_p])
        self._bind("cuStreamDestroy_v2", [ct.c_void_p])
        self.call("cuInit", 0)
        self.device = ct.c_int()
        self.call("cuDeviceGet", ct.byref(self.device), device_id)
        self.context = ct.c_void_p()
        self.call("cuDevicePrimaryCtxRetain", ct.byref(self.context), self.device)
        self.call("cuCtxSetCurrent", self.context)
        self.stream = ct.c_void_p()
        self.call("cuStreamCreate", ct.byref(self.stream), 0)

    def _bind(self, name, arguments):
        method = getattr(self.lib, name)
        method.argtypes = arguments
        method.restype = ct.c_int

    def call(self, name, *arguments):
        result = getattr(self.lib, name)(*arguments)
        if result:
            raise RuntimeError(f"{name} failed with CUDA status {result}")

    def info(self):
        name = ct.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), self.device)
        major, minor = ct.c_int(), ct.c_int()
        self.call("cuDeviceGetAttribute", ct.byref(major), 75, self.device)
        self.call("cuDeviceGetAttribute", ct.byref(minor), 76, self.device)
        free, total = ct.c_size_t(), ct.c_size_t()
        self.call("cuMemGetInfo_v2", ct.byref(free), ct.byref(total))
        return {"name": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}",
                "free_bytes": free.value, "total_bytes": total.value}

    def malloc(self, count):
        address = ct.c_uint64()
        self.call("cuMemAlloc_v2", ct.byref(address), max(1, count))
        return address.value


class TensorRTSession:
    def __init__(self, path, cuda):
        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot load engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.configure_profiling(Path(path).stem)

    _PROFILE_COUNTERS = (
        "runs", "errors", "h2d_bytes", "d2h_bytes", "cuda_allocations",
        "cuda_frees", "host_allocations", "h2d_ns", "d2h_ns",
        "cuda_alloc_ns", "cuda_free_ns", "host_alloc_ns", "execute_ns",
        "synchronize_ns", "run_total_ns",
    )

    def configure_profiling(self, graph_name: str, *, enabled: bool | None = None) -> None:
        """Opt-in CPU timings for the existing synchronous TensorRT path."""
        self.profile_graph_name = graph_name
        self.profile_enabled = os.environ.get("MOSS_TRT_PROFILE") == "1" if enabled is None else enabled
        self.profile_counters = {name: 0 for name in self._PROFILE_COUNTERS} if self.profile_enabled else None

    def profile_snapshot(self) -> dict | None:
        if not getattr(self, "profile_enabled", False):
            return None
        counters = self.profile_counters.copy()
        result = {"graph": self.profile_graph_name}
        for name, value in counters.items():
            if name.endswith("_ns"):
                result[name[:-3] + "_seconds"] = value / 1e9
            else:
                result[name] = value
        result["execute_sync_seconds"] = (counters["execute_ns"] + counters["synchronize_ns"]) / 1e9
        return result

    def run(self, inputs):
        allocations, arrays, outputs = {}, {}, {}
        profile = {name: 0 for name in self._PROFILE_COUNTERS} if getattr(self, "profile_enabled", False) else None
        began_ns = time.perf_counter_ns() if profile is not None else 0
        completed = False
        try:
            expected_inputs = {self.engine.get_tensor_name(index)
                               for index in range(self.engine.num_io_tensors)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
                               == trt.TensorIOMode.INPUT}
            if set(inputs) != expected_inputs:
                raise ValueError(f"TensorRT input mismatch: missing={sorted(expected_inputs - set(inputs))}, "
                                 f"unexpected={sorted(set(inputs) - expected_inputs)}")
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                    # ascontiguousarray promotes rank-0 scalars to shape (1,).
                    # Inflect length_scale and noise_scale must remain scalars.
                    arrays[name] = np.asarray(inputs[name], dtype=dtype)
                    if not arrays[name].flags.c_contiguous:
                        arrays[name] = np.ascontiguousarray(arrays[name])
                    if not self.context.set_input_shape(name, arrays[name].shape):
                        raise ValueError(f"Unsupported shape for {name}: {arrays[name].shape}")
                    if self.engine.get_tensor_location(name) == trt.TensorLocation.HOST:
                        address = arrays[name].ctypes.data
                    else:
                        if profile is not None:
                            phase_ns = time.perf_counter_ns()
                        address = self.cuda.malloc(arrays[name].nbytes)
                        if profile is not None:
                            profile["cuda_alloc_ns"] += time.perf_counter_ns() - phase_ns
                            profile["cuda_allocations"] += 1
                        allocations[name] = address
                        if arrays[name].nbytes:
                            if profile is not None:
                                phase_ns = time.perf_counter_ns()
                            self.cuda.call("cuMemcpyHtoD_v2", address, arrays[name].ctypes.data,
                                           arrays[name].nbytes)
                            if profile is not None:
                                profile["h2d_ns"] += time.perf_counter_ns() - phase_ns
                                profile["h2d_bytes"] += arrays[name].nbytes
                    if not self.context.set_tensor_address(name, address):
                        raise RuntimeError(f"Cannot bind {name}")
            unspecified = self.context.infer_shapes()
            if unspecified:
                raise RuntimeError(f"Unspecified TensorRT shapes: {unspecified}")
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    shape = tuple(self.context.get_tensor_shape(name))
                    if any(n < 0 for n in shape):
                        raise RuntimeError(f"Data-dependent output {name}: {shape}; split the graph first")
                    dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                    if profile is not None:
                        phase_ns = time.perf_counter_ns()
                    outputs[name] = np.empty(shape, dtype=dtype)
                    if profile is not None:
                        profile["host_alloc_ns"] += time.perf_counter_ns() - phase_ns
                        profile["host_allocations"] += 1
                    if self.engine.get_tensor_location(name) == trt.TensorLocation.HOST:
                        address = outputs[name].ctypes.data
                    else:
                        if profile is not None:
                            phase_ns = time.perf_counter_ns()
                        address = self.cuda.malloc(outputs[name].nbytes)
                        if profile is not None:
                            profile["cuda_alloc_ns"] += time.perf_counter_ns() - phase_ns
                            profile["cuda_allocations"] += 1
                        allocations[name] = address
                    if not self.context.set_tensor_address(name, address):
                        raise RuntimeError(f"Cannot bind {name}")
            if profile is not None:
                phase_ns = time.perf_counter_ns()
            executed = self.context.execute_async_v3(self.cuda.stream.value)
            if profile is not None:
                profile["execute_ns"] += time.perf_counter_ns() - phase_ns
            if not executed:
                raise RuntimeError("TensorRT execution failed")
            if profile is not None:
                phase_ns = time.perf_counter_ns()
            self.cuda.call("cuStreamSynchronize", self.cuda.stream)
            if profile is not None:
                profile["synchronize_ns"] += time.perf_counter_ns() - phase_ns
            for name, output in outputs.items():
                if name in allocations and output.nbytes:
                    if profile is not None:
                        phase_ns = time.perf_counter_ns()
                    self.cuda.call("cuMemcpyDtoH_v2", output.ctypes.data, allocations[name], output.nbytes)
                    if profile is not None:
                        profile["d2h_ns"] += time.perf_counter_ns() - phase_ns
                        profile["d2h_bytes"] += output.nbytes
            completed = True
            return outputs
        finally:
            free_completed = False
            try:
                for address in allocations.values():
                    if profile is not None:
                        phase_ns = time.perf_counter_ns()
                    self.cuda.call("cuMemFree_v2", address)
                    if profile is not None:
                        profile["cuda_free_ns"] += time.perf_counter_ns() - phase_ns
                        profile["cuda_frees"] += 1
                free_completed = True
            finally:
                if profile is not None:
                    profile["runs"] = 1
                    profile["errors"] = int(not completed or not free_completed)
                    profile["run_total_ns"] = time.perf_counter_ns() - began_ns
                    for name, value in profile.items():
                        self.profile_counters[name] += value


def compare(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        return {"shape_match": False, "actual_shape": list(actual.shape),
                "reference_shape": list(reference.shape), "finite": bool(np.isfinite(actual).all())}
    if not actual.size:
        return {"shape_match": True, "shape": list(actual.shape), "finite": True,
                "max_abs_error": 0.0, "rms_error": 0.0, "relative_rms_error": 0.0}
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    return {"shape_match": True, "shape": list(actual.shape),
            "finite": bool(np.isfinite(actual).all()), "max_abs_error": float(np.max(np.abs(delta), initial=0)),
            "rms_error": rms, "relative_rms_error": rms / max(reference_rms, 1e-12)}
