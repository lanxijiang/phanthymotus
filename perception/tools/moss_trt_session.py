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

    def bind_graph_api(self):
        """Bind actual driver symbols, not the C header's versioned macros."""
        if getattr(self, "_graph_api_bound", False):
            return
        for name, arguments in (
            ("cuMemAllocHost_v2", [ct.POINTER(ct.c_void_p), ct.c_size_t]),
            ("cuMemFreeHost", [ct.c_void_p]),
            ("cuMemcpyHtoDAsync_v2", [ct.c_uint64, ct.c_void_p, ct.c_size_t, ct.c_void_p]),
            ("cuStreamBeginCapture_v2", [ct.c_void_p, ct.c_int]),
            ("cuStreamEndCapture", [ct.c_void_p, ct.POINTER(ct.c_void_p)]),
            ("cuGraphInstantiateWithFlags", [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_ulonglong]),
            ("cuGraphLaunch", [ct.c_void_p, ct.c_void_p]),
            ("cuGraphExecDestroy", [ct.c_void_p]),
            ("cuGraphDestroy", [ct.c_void_p]),
        ):
            self._bind(name, arguments)
        self._graph_api_bound = True


class DeviceTensor:
    """Session-owned state valid until its two-buffer slot is reused.

    The generation check makes an old or foreign state fail explicitly instead
    of silently consuming overwritten memory. Ordinary public outputs are still
    fresh NumPy arrays. materialize() is for validation, not the inference loop.
    """
    _moss_device_tensor = True

    def __init__(self, owner, slot, generation, shape, dtype):
        self.owner, self.slot, self.generation = owner, slot, generation
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def __array__(self, *args, **kwargs):
        raise TypeError("GPU state cannot be implicitly copied; use materialize() for validation")

    def materialize(self):
        address = self.owner._state_address(self)
        result = np.empty(self.shape, dtype=self.dtype)
        if result.nbytes:
            cuda = self.owner.cuda
            cuda.call("cuCtxSetCurrent", cuda.context)
            cuda.call("cuMemcpyDtoH_v2", result.ctypes.data, address, result.nbytes)
        return result


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
        "cuda_graph_captures", "cuda_graph_replays", "cuda_graph_warmup_enqueues",
        "cuda_graph_capture_failures", "cuda_graph_capture_ns", "cuda_graph_launch_ns",
    )

    def configure_profiling(self, graph_name: str, *, enabled: bool | None = None,
                            resident_state: bool | None = None) -> None:
        """Configure timings and state ownership once; generic sessions opt in.

        The MOSS pool supplies its frozen resident-state policy explicitly.
        Direct builder/validation sessions retain the environment opt-in and
        return NumPy outputs by default.
        """
        self.profile_graph_name = graph_name
        self.profile_enabled = os.environ.get("MOSS_TRT_PROFILE") == "1" if enabled is None else enabled
        self.profile_counters = {name: 0 for name in self._PROFILE_COUNTERS} if self.profile_enabled else None
        # The pool constructs sessions without __init__. Initialize here once,
        # and preserve allocations if profiling counters are reset after warmup.
        if not hasattr(self, "_device_buffers"):
            self._device_buffers = {}
            self._reuse_device_buffers = (
                (graph_name == "local_cached_step"
                 and os.environ.get("MOSS_TRT_REUSE_LOCAL_BUFFERS") == "1")
                # The optional fused frame has fixed input/output dimensions
                # and no external KV state. Reuse ordinary IO allocations;
                # it must not enter the resident-state path below.
                or (graph_name == "local_fixed_sampled_frame"
                    and os.environ.get("MOSS_TRT_REUSE_FRAME_BUFFERS", "1") == "1")
            )
            self._closed = False
            self._resident_enabled = (
                graph_name in {"global_unified", "local_cached_step", "codec_decode_step"}
                and (os.environ.get("MOSS_TRT_RESIDENT_STATE") == "1"
                     if resident_state is None else bool(resident_state))
            )
            self._resident_ready = False
            self._state_slots = {}
            self._state_generation = 0
            self._graph_enabled = (graph_name == "local_fixed_sampled_frame"
                                   and os.environ.get("MOSS_TRT_FRAME_CUDA_GRAPH") == "1")
            self._graph_stream = ct.c_void_p()
            self._graph_aux_streams = []
            self._graph = ct.c_void_p()
            self._graph_exec = ct.c_void_p()
            self._graph_capturing = False
            self._graph_io = None
            self._graph_pinned_inputs = {}
            self._graph_status = {"enabled": self._graph_enabled, "captures": 0,
                                  "replays": 0, "warmup_enqueues": 0,
                                  "capture_failures": 0, "last_error": None}
            if self._graph_enabled and not self._reuse_device_buffers:
                raise ValueError("Fused CUDA Graph requires MOSS_TRT_REUSE_FRAME_BUFFERS=1")

    def cuda_graph_snapshot(self):
        """Actual session lifetime counts, including when profiling is off."""
        return {**self._graph_status, "ready": bool(self._graph_exec.value),
                "closed": self._closed}

    def _prepare_cuda_graph(self, profile):
        # Captured context state and device addresses must never change during
        # replay. This path intentionally supports only the fixed fused frame.
        expected = {
            "global_hidden": (True, (1, 768), np.dtype(np.float32)),
            "random_u": (True, (1, 17), np.dtype(np.float32)),
            "should_continue": (False, (1, 1), np.dtype(np.int32)),
            "frame_token_ids": (False, (1, 16), np.dtype(np.int32)),
            "text_logits": (False, (1, 2), np.dtype(np.float32)),
            "audio_logits": (False, (1, 16, 1024), np.dtype(np.float32)),
        }
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        if set(names) != set(expected):
            raise ValueError("CUDA Graph requires the exact fixed fused-frame interface")
        for name in names:
            is_input, shape, dtype = expected[name]
            if ((self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT) != is_input
                    or self.engine.get_tensor_location(name) == trt.TensorLocation.HOST
                    or tuple(self.engine.get_tensor_shape(name)) != shape
                    or np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))) != dtype):
                raise ValueError(f"Unsupported fixed CUDA Graph tensor: {name}")
        self.cuda.bind_graph_api()
        # Use a session-owned nonblocking stream, leaving the existing stable
        # runner stream and its implicit default-stream copy ordering unchanged.
        self.cuda.call("cuStreamCreate", ct.byref(self._graph_stream), 1)
        for _ in range(int(getattr(self.engine, "num_aux_streams", 0))):
            stream = ct.c_void_p()
            self.cuda.call("cuStreamCreate", ct.byref(stream), 1)
            self._graph_aux_streams.append(stream)
        for name, (is_input, shape, dtype) in expected.items():
            nbytes = int(np.prod(shape)) * dtype.itemsize
            address = self._device_address(name, nbytes, profile)
            if is_input:
                if not self.context.set_input_shape(name, shape):
                    raise ValueError(f"Unsupported CUDA Graph input shape: {name}")
                pinned = ct.c_void_p()
                self.cuda.call("cuMemAllocHost_v2", ct.byref(pinned), nbytes)
                self._graph_pinned_inputs[name] = (pinned, nbytes)
            if not self.context.set_tensor_address(name, address):
                raise RuntimeError(f"Cannot bind fixed CUDA Graph tensor: {name}")
        if self.context.infer_shapes():
            raise RuntimeError("Fused CUDA Graph contains unspecified tensor shapes")
        self._graph_io = expected

    def _capture_cuda_graph(self, profile):
        began = time.perf_counter_ns() if profile is not None else 0
        try:
            if self._graph_aux_streams:
                self.context.set_aux_streams([stream.value for stream in self._graph_aux_streams])
            # THREAD_LOCAL avoids rejecting another runtime's independent CUDA
            # allocations; our nonblocking streams have no legacy-stream edges.
            self.cuda.call("cuStreamBeginCapture_v2", self._graph_stream, 1)
            self._graph_capturing = True
            try:
                if not self.context.execute_async_v3(self._graph_stream.value):
                    raise RuntimeError("TensorRT fused CUDA Graph capture enqueue failed")
            finally:
                self._graph_capturing = False
                self.cuda.call("cuStreamEndCapture", self._graph_stream, ct.byref(self._graph))
            if not self._graph.value:
                raise RuntimeError("CUDA Graph capture returned no graph")
            self.cuda.call("cuGraphInstantiateWithFlags", ct.byref(self._graph_exec), self._graph, 0)
            if not self._graph_exec.value:
                raise RuntimeError("CUDA Graph instantiation returned no executable")
            self._graph_status["captures"] += 1
            if profile is not None:
                profile["cuda_graph_captures"] += 1
        except BaseException:
            self._graph_status["capture_failures"] += 1
            if profile is not None:
                profile["cuda_graph_capture_failures"] += 1
            raise
        finally:
            if profile is not None:
                profile["cuda_graph_capture_ns"] += time.perf_counter_ns() - began

    def _release_cuda_graph(self):
        if self._graph_capturing:
            self._graph_capturing = False
            self.cuda.call("cuStreamEndCapture", self._graph_stream, ct.byref(self._graph))
        if self._graph_stream.value:
            self.cuda.call("cuStreamSynchronize", self._graph_stream)
        # No stream can reference graph/context buffers after this point.
        for attr, destroy in (("_graph_exec", "cuGraphExecDestroy"), ("_graph", "cuGraphDestroy")):
            handle = getattr(self, attr)
            if handle.value:
                self.cuda.call(destroy, handle)
                setattr(self, attr, ct.c_void_p())
        for name, (pinned, _nbytes) in list(self._graph_pinned_inputs.items()):
            self.cuda.call("cuMemFreeHost", pinned)
            del self._graph_pinned_inputs[name]
        while self._graph_aux_streams:
            self.cuda.call("cuStreamDestroy_v2", self._graph_aux_streams[-1])
            self._graph_aux_streams.pop()
        if self._graph_stream.value:
            self.cuda.call("cuStreamDestroy_v2", self._graph_stream)
            self._graph_stream = ct.c_void_p()
        self._graph_io = None

    def _run_cuda_graph(self, inputs):
        profile = {key: 0 for key in self._PROFILE_COUNTERS} if self.profile_enabled else None
        began = time.perf_counter_ns() if profile is not None else 0
        completed = False
        try:
            if set(inputs) != {"global_hidden", "random_u"}:
                raise ValueError("Fused CUDA Graph requires global_hidden and random_u")
            arrays = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
            if arrays["global_hidden"].shape != (1, 768) or arrays["random_u"].shape != (1, 17):
                raise ValueError("Fused CUDA Graph input dimensions cannot change")
            if self._graph_io is None:
                self._prepare_cuda_graph(profile)
            for name, array in arrays.items():
                array = np.ascontiguousarray(array)
                pinned, nbytes = self._graph_pinned_inputs[name]
                phase = time.perf_counter_ns() if profile is not None else 0
                ct.memmove(pinned, array.ctypes.data, nbytes)
                self.cuda.call("cuMemcpyHtoDAsync_v2", self._device_buffers[name][0],
                               pinned, nbytes, self._graph_stream)
                if profile is not None:
                    profile["h2d_bytes"] += nbytes
                    profile["h2d_ns"] += time.perf_counter_ns() - phase
            phase = time.perf_counter_ns() if profile is not None else 0
            replay = bool(self._graph_exec.value)
            if replay:
                self.cuda.call("cuGraphLaunch", self._graph_exec, self._graph_stream)
                self._graph_status["replays"] += 1
                if profile is not None:
                    profile["cuda_graph_replays"] += 1
                    profile["cuda_graph_launch_ns"] += time.perf_counter_ns() - phase
            else:
                if self._graph_aux_streams:
                    self.context.set_aux_streams([stream.value for stream in self._graph_aux_streams])
                if not self.context.execute_async_v3(self._graph_stream.value):
                    raise RuntimeError("TensorRT fused CUDA Graph warmup enqueue failed")
                self._graph_status["warmup_enqueues"] += 1
                if profile is not None:
                    profile["cuda_graph_warmup_enqueues"] += 1
            if profile is not None:
                profile["execute_ns"] += time.perf_counter_ns() - phase
            phase = time.perf_counter_ns() if profile is not None else 0
            self.cuda.call("cuStreamSynchronize", self._graph_stream)
            if profile is not None:
                profile["synchronize_ns"] += time.perf_counter_ns() - phase
            if not replay:
                # Capture records work without executing it. The first call's
                # already-computed warmup outputs remain valid for this request.
                self._capture_cuda_graph(profile)
            outputs = {}
            for name, (is_input, shape, dtype) in self._graph_io.items():
                if is_input:
                    continue
                phase = time.perf_counter_ns() if profile is not None else 0
                array = np.empty(shape, dtype=dtype)
                if profile is not None:
                    profile["host_allocations"] += 1
                    profile["host_alloc_ns"] += time.perf_counter_ns() - phase
                phase = time.perf_counter_ns() if profile is not None else 0
                self.cuda.call("cuMemcpyDtoH_v2", array.ctypes.data,
                               self._device_buffers[name][0], array.nbytes)
                if profile is not None:
                    profile["d2h_bytes"] += array.nbytes
                    profile["d2h_ns"] += time.perf_counter_ns() - phase
                outputs[name] = array
            completed = True
            return outputs
        except BaseException as error:
            self._graph_status["last_error"] = f"{type(error).__name__}: {error}"
            # Never replay a graph after any failed copy/launch/shape check, and
            # never silently substitute ordinary enqueue or CPU inference.
            try:
                self._release_cuda_graph()
                self._release_device_buffers(profile)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Fused CUDA Graph cleanup also failed: {cleanup_error}")
            raise
        finally:
            if profile is not None:
                profile["runs"] = 1
                profile["errors"] = int(not completed)
                profile["run_total_ns"] = time.perf_counter_ns() - began
                for key, value in profile.items():
                    self.profile_counters[key] += value

    def _prepare_resident(self):
        self._io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self._io.append((name, self.engine.get_tensor_mode(name),
                             np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))),
                             self.engine.get_tensor_location(name) == trt.TensorLocation.HOST))
        self._input_io = [row for row in self._io if row[1] == trt.TensorIOMode.INPUT]
        self._output_io = [row for row in self._io if row[1] != trt.TensorIOMode.INPUT]
        self._expected_inputs = {row[0] for row in self._input_io}
        output_info = {row[0]: row for row in self._output_io}
        self._state_pairs = {}
        for name, _, dtype, host in self._input_io:
            output = (name.replace("local_past_", "local_present_", 1)
                      if name.startswith("local_past_") else
                      name.replace("past_", "present_", 1)
                      if name.startswith("past_") else "present_" + name)
            if output in output_info and not host and not output_info[output][3]:
                if dtype != output_info[output][2]:
                    raise ValueError(f"State dtype mismatch: {name} / {output}")
                self._state_pairs[name] = output
        if not self._state_pairs:
            raise ValueError(f"No device state pairs for {self.profile_graph_name}")
        self._state_inputs = {output: name for name, output in self._state_pairs.items()}
        self._bound_shapes, self._bound_addresses = {}, {}
        self._resident_ready = True

    def _state_address(self, value, input_name=None):
        if value.owner is not self or self._closed:
            raise ValueError("Foreign or closed TensorRT state")
        record = self._state_slots.get(value.slot)
        if record is None or record[2] != value.generation:
            raise ValueError("Stale TensorRT state: its buffer has been reused")
        if input_name is not None and self._state_pairs.get(input_name) != value.slot[0]:
            raise ValueError(f"State is not the output corresponding to {input_name}")
        return record[0]

    def _state_buffer(self, output_name, bank, nbytes, profile):
        slot = (output_name, bank)
        # Round larger growing caches to 64 KiB; never allocate profile maxima.
        capacity = max(1, nbytes)
        if capacity > 4096:
            capacity = ((capacity + 65535) // 65536) * 65536
        previous = self._state_slots.get(slot)
        if previous is not None and previous[1] < capacity:
            self._free_device_buffer(previous[0], profile)
            del self._state_slots[slot]
            previous = None
        if previous is None:
            phase = time.perf_counter_ns() if profile is not None else 0
            address = self.cuda.malloc(capacity)
            if profile is not None:
                profile["cuda_allocations"] += 1
                profile["cuda_alloc_ns"] += time.perf_counter_ns() - phase
        else:
            address, capacity, _ = previous
        self._state_generation += 1
        self._state_slots[slot] = (address, capacity, self._state_generation)
        return address, self._state_generation

    def _release_state_buffers(self, profile=None):
        error = None
        for slot, (address, _, _) in list(self._state_slots.items()):
            try:
                self._free_device_buffer(address, profile)
            except Exception as exc:
                error = error or exc
            else:
                del self._state_slots[slot]
        if error is not None:
            raise error

    def _run_resident(self, inputs):
        if not self._resident_ready:
            self._prepare_resident()
        if set(inputs) != self._expected_inputs:
            raise ValueError("TensorRT resident input names do not match the engine")
        profile = {key: 0 for key in self._PROFILE_COUNTERS} if self.profile_enabled else None
        began = time.perf_counter_ns() if profile is not None else 0
        arrays, outputs, output_addresses, state_banks = {}, {}, {}, {}
        completed = False

        def bind(name, address):
            if self._bound_addresses.get(name) != address:
                if not self.context.set_tensor_address(name, address):
                    raise RuntimeError(f"Cannot bind {name}")
                self._bound_addresses[name] = address

        def persistent(name, nbytes):
            # Persistent ordinary IO is independent of both state banks.
            previous = self._device_buffers.get(name)
            if previous is not None and previous[1] >= max(1, nbytes):
                return previous[0]
            if previous is not None:
                self._free_device_buffer(previous[0], profile)
                del self._device_buffers[name]
            phase = time.perf_counter_ns() if profile is not None else 0
            address = self.cuda.malloc(max(1, nbytes))
            self._device_buffers[name] = (address, max(1, nbytes))
            if profile is not None:
                profile["cuda_allocations"] += 1
                profile["cuda_alloc_ns"] += time.perf_counter_ns() - phase
            return address

        # Reject foreign/stale handles before touching this session's valid
        # buffers. Execution failures below invalidate and reclaim all state.
        for name, _, dtype, host in self._input_io:
            value = inputs[name]
            if isinstance(value, DeviceTensor):
                self._state_address(value, name)
                if host or value.dtype != dtype:
                    raise ValueError(f"Invalid device state dtype/location: {name}")
        try:
            for name, _, dtype, host in self._input_io:
                value = inputs[name]
                if isinstance(value, DeviceTensor):
                    if host or value.dtype != dtype:
                        raise ValueError(f"Invalid device state dtype/location: {name}")
                    shape, address = value.shape, self._state_address(value, name)
                    state_banks[self._state_pairs[name]] = 1 - value.slot[1]
                else:
                    array = np.asarray(value, dtype=dtype)
                    if not array.flags.c_contiguous:
                        array = np.ascontiguousarray(array)
                    arrays[name] = array
                    shape = array.shape
                    if host:
                        address = array.ctypes.data
                    else:
                        if name in self._state_pairs:
                            # Explicit host state always resets/uploads; it is
                            # never ignored because a previous request exists.
                            output_name = self._state_pairs[name]
                            address, _ = self._state_buffer(output_name, 1, array.nbytes, profile)
                            state_banks[output_name] = 0
                        else:
                            address = persistent(name, array.nbytes)
                        if array.nbytes:
                            phase = time.perf_counter_ns() if profile is not None else 0
                            self.cuda.call("cuMemcpyHtoD_v2", address, array.ctypes.data, array.nbytes)
                            if profile is not None:
                                profile["h2d_ns"] += time.perf_counter_ns() - phase
                                profile["h2d_bytes"] += array.nbytes
                if self._bound_shapes.get(name) != shape:
                    if not self.context.set_input_shape(name, shape):
                        raise ValueError(f"Unsupported shape for {name}: {shape}")
                    self._bound_shapes[name] = shape
                bind(name, address)
            unspecified = self.context.infer_shapes()
            if unspecified:
                raise RuntimeError(f"Unspecified TensorRT shapes: {unspecified}")
            for name, _, dtype, host in self._output_io:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(n < 0 for n in shape):
                    raise RuntimeError(f"Data-dependent output {name}: {shape}")
                nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if name in self._state_inputs:
                    bank = state_banks[name]
                    address, generation = self._state_buffer(name, bank, nbytes, profile)
                    outputs[name] = DeviceTensor(self, (name, bank), generation, shape, dtype)
                else:
                    phase = time.perf_counter_ns() if profile is not None else 0
                    array = np.empty(shape, dtype=dtype)
                    if profile is not None:
                        profile["host_allocations"] += 1
                        profile["host_alloc_ns"] += time.perf_counter_ns() - phase
                    outputs[name] = array
                    address = array.ctypes.data if host else persistent(name, array.nbytes)
                    if not host:
                        output_addresses[name] = address
                bind(name, address)
            phase = time.perf_counter_ns() if profile is not None else 0
            executed = self.context.execute_async_v3(self.cuda.stream.value)
            if profile is not None:
                profile["execute_ns"] += time.perf_counter_ns() - phase
            if not executed:
                raise RuntimeError("TensorRT resident execution failed")
            phase = time.perf_counter_ns() if profile is not None else 0
            self.cuda.call("cuStreamSynchronize", self.cuda.stream)
            if profile is not None:
                profile["synchronize_ns"] += time.perf_counter_ns() - phase
            for name, address in output_addresses.items():
                array = outputs[name]
                if array.nbytes:
                    phase = time.perf_counter_ns() if profile is not None else 0
                    self.cuda.call("cuMemcpyDtoH_v2", array.ctypes.data, address, array.nbytes)
                    if profile is not None:
                        profile["d2h_ns"] += time.perf_counter_ns() - phase
                        profile["d2h_bytes"] += array.nbytes
            completed = True
            return outputs
        finally:
            try:
                if not completed:
                    try:
                        self.cuda.call("cuStreamSynchronize", self.cuda.stream)
                    finally:
                        try:
                            self._release_state_buffers(profile)
                        finally:
                            self._release_device_buffers(profile)
                            self._bound_shapes.clear()
                            self._bound_addresses.clear()
            finally:
                if profile is not None:
                    profile["runs"] = 1
                    profile["errors"] = int(not completed)
                    profile["run_total_ns"] = time.perf_counter_ns() - began
                    for key, value in profile.items():
                        self.profile_counters[key] += value

    def _free_device_buffer(self, address, profile):
        phase_ns = time.perf_counter_ns() if profile is not None else 0
        self.cuda.call("cuMemFree_v2", address)
        if profile is not None:
            profile["cuda_free_ns"] += time.perf_counter_ns() - phase_ns
            profile["cuda_frees"] += 1

    def _device_address(self, name, nbytes, profile):
        capacity = max(1, nbytes)
        if self._reuse_device_buffers:
            previous = self._device_buffers.get(name)
            if previous is not None:
                address, previous_capacity = previous
                if previous_capacity >= capacity:
                    return address
                # Every successful run synchronizes before returning. Free the
                # old buffer before growth to avoid a temporary memory spike.
                self._free_device_buffer(address, profile)
                del self._device_buffers[name]
        phase_ns = time.perf_counter_ns() if profile is not None else 0
        address = self.cuda.malloc(capacity)
        if profile is not None:
            profile["cuda_alloc_ns"] += time.perf_counter_ns() - phase_ns
            profile["cuda_allocations"] += 1
        if self._reuse_device_buffers:
            self._device_buffers[name] = (address, capacity)
        return address

    def _release_device_buffers(self, profile=None):
        error = None
        for name, (address, _capacity) in list(self._device_buffers.items()):
            try:
                self._free_device_buffer(address, profile)
            except Exception as exc:
                if error is None:
                    error = exc
            else:
                del self._device_buffers[name]
        if error is not None:
            raise error

    def close(self):
        """Release session buffers/context and any private capture streams."""
        if self._closed:
            return
        if self._graph_enabled:
            self.cuda.call("cuCtxSetCurrent", self.cuda.context)
            self._release_cuda_graph()
        if self._device_buffers or self._state_slots:
            self.cuda.call("cuCtxSetCurrent", self.cuda.context)
            try:
                self.cuda.call("cuStreamSynchronize", self.cuda.stream)
            finally:
                try:
                    self._release_state_buffers(self.profile_counters)
                finally:
                    self._release_device_buffers(self.profile_counters)
        self.context = None
        self.engine = None
        self.runtime = None
        self.logger = None
        self._closed = True

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
        result["cuda_graph_enabled"] = int(self._graph_enabled)
        result["cuda_graph_ready"] = int(bool(self._graph_exec.value))
        return result

    def run(self, inputs):
        if self._closed:
            raise RuntimeError("TensorRT session is closed")
        if self._graph_enabled:
            return self._run_cuda_graph(inputs)
        if self._resident_enabled:
            return self._run_resident(inputs)
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
                        address = self._device_address(name, arrays[name].nbytes, profile)
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
                        address = self._device_address(name, outputs[name].nbytes, profile)
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
                if self._reuse_device_buffers:
                    if not completed:
                        # A failed enqueue may have queued work. Reclaim all
                        # cached addresses only after draining the stream.
                        try:
                            self.cuda.call("cuStreamSynchronize", self.cuda.stream)
                        finally:
                            self._release_device_buffers(profile)
                else:
                    for address in allocations.values():
                        self._free_device_buffer(address, profile)
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
