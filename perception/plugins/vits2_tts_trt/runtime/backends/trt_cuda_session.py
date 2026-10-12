"""Direct TensorRT runner using libcudart and NumPy host arrays."""

from __future__ import annotations

import hashlib
import ctypes
import ctypes.util
import math
from pathlib import Path

import numpy as np

from utils.tensorrt_runtime import trt_dtype_to_numpy


def _load_cuda_runtime():
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    try:
        return ctypes.CDLL(library)
    except OSError as exc:
        raise RuntimeError(f"Unable to load CUDA Runtime library {library!r}") from exc


def _check_cuda(error: int, operation: str) -> None:
    if error != 0:
        raise RuntimeError(f"CUDA {operation} failed with error {error}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CudaRuntime:
    """One CUDA stream shared by all three sequential TensorRT engines."""

    def __init__(self):
        self.api = _load_cuda_runtime()
        self._configure_api()
        stream = ctypes.c_void_p()
        _check_cuda(self.api.cudaStreamCreate(ctypes.byref(stream)), "stream creation")
        self.stream = int(stream.value)
        self._closed = False

    def _configure_api(self) -> None:
        self.api.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.api.cudaStreamCreate.restype = ctypes.c_int
        self.api.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.api.cudaStreamDestroy.restype = ctypes.c_int
        self.api.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.api.cudaStreamSynchronize.restype = ctypes.c_int
        self.api.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.api.cudaMalloc.restype = ctypes.c_int
        self.api.cudaFree.argtypes = [ctypes.c_void_p]
        self.api.cudaFree.restype = ctypes.c_int
        self.api.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.api.cudaMemcpyAsync.restype = ctypes.c_int
        self.api.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.api.cudaDeviceGetAttribute.restype = ctypes.c_int

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        _check_cuda(self.api.cudaMalloc(ctypes.byref(pointer), size), "allocation")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        if pointer:
            _check_cuda(self.api.cudaFree(ctypes.c_void_p(pointer)), "free")

    def copy_to_device(self, pointer: int, array: np.ndarray) -> None:
        _check_cuda(
            self.api.cudaMemcpyAsync(
                ctypes.c_void_p(pointer),
                ctypes.c_void_p(array.ctypes.data),
                array.nbytes,
                1,
                ctypes.c_void_p(self.stream),
            ),
            "host-to-device copy",
        )

    def copy_to_host(self, array: np.ndarray, pointer: int) -> None:
        _check_cuda(
            self.api.cudaMemcpyAsync(
                ctypes.c_void_p(array.ctypes.data),
                ctypes.c_void_p(pointer),
                array.nbytes,
                2,
                ctypes.c_void_p(self.stream),
            ),
            "device-to-host copy",
        )

    def synchronize(self) -> None:
        _check_cuda(
            self.api.cudaStreamSynchronize(ctypes.c_void_p(self.stream)),
            "stream synchronize",
        )

    def compute_capability(self) -> str:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        _check_cuda(
            self.api.cudaDeviceGetAttribute(ctypes.byref(major), 75, 0),
            "compute capability major query",
        )
        _check_cuda(
            self.api.cudaDeviceGetAttribute(ctypes.byref(minor), 76, 0),
            "compute capability minor query",
        )
        return f"{major.value}.{minor.value}"

    def close(self) -> None:
        if not self._closed:
            _check_cuda(
                self.api.cudaStreamDestroy(ctypes.c_void_p(self.stream)),
                "stream destruction",
            )
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _DeviceBuffer:
    def __init__(self, cuda: CudaRuntime):
        self.cuda = cuda
        self.pointer = 0
        self.capacity = 0

    def reserve(self, size: int) -> int:
        if size < 0:
            raise ValueError("CUDA buffer size must be non-negative")
        # TensorRT still needs a non-null binding for empty tensors.
        size = max(size, 1)
        if size > self.capacity:
            # Release first so a growing buffer never temporarily consumes both
            # capacities on the memory-limited Orin. Reset ownership before
            # malloc: if allocation fails, close/retry must not free the stale
            # pointer a second time.
            self.close()
            self.pointer = self.cuda.malloc(size)
            self.capacity = size
        return self.pointer

    def close(self) -> None:
        self.cuda.free(self.pointer)
        self.pointer = 0
        self.capacity = 0


class TensorRTCudaSession:
    def __init__(
        self,
        engine_path: str | Path,
        cuda: CudaRuntime,
        expected_sha256: str | None = None,
    ):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT Python bindings are required") from exc

        self.trt = trt
        self.cuda = cuda
        self.path = Path(engine_path)
        if expected_sha256 and _sha256(self.path) != expected_sha256:
            raise RuntimeError(f"TensorRT engine checksum mismatch: {self.path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {self.path}")
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("The CUDA/NumPy backend requires TensorRT tensor API v3")
        self._buffers: dict[str, _DeviceBuffer] = {}
        self._input_shapes: dict[str, tuple[int, ...]] = {}
        inputs, outputs = [], []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            # Cache static metadata once. The shared mapper also avoids TRT
            # 8.5's nptype(BOOL), which accesses removed NumPy np.bool.
            dtype = np.dtype(trt_dtype_to_numpy(trt, self.engine.get_tensor_dtype(name)))
            target = inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
            target.append((name, dtype))
        self._input_specs = tuple(inputs)
        self._output_specs = tuple(outputs)

    def _buffer(self, name: str, size: int) -> int:
        buffer = self._buffers.get(name)
        if buffer is None:
            buffer = self._buffers[name] = _DeviceBuffer(self.cuda)
        return buffer.reserve(size)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return independently owned host arrays after CUDA completion.

        All engine outputs retain a device binding and are copied to the host.
        Buffers grow only to observed tensor sizes, never profile maxima.
        The session/context is sequential, as is the owning TTS engine.
        """
        host_inputs = {}
        for name, dtype in self._input_specs:
            if name not in inputs:
                raise KeyError(f"Missing TensorRT input {name!r}")
            array = np.ascontiguousarray(inputs[name], dtype=dtype)
            if self._input_shapes.get(name) != array.shape:
                if not self.context.set_input_shape(name, array.shape):
                    raise ValueError(f"Shape outside TensorRT profile: {name}={array.shape}")
                self._input_shapes[name] = array.shape
            host_inputs[name] = array

        outputs = {}
        for name, array in host_inputs.items():
            pointer = self._buffer(name, array.nbytes)
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Failed to bind TensorRT tensor {name!r}")
        for name, dtype in self._output_specs:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"Unresolved TensorRT output shape: {name}={shape}")
            size = math.prod(shape) * dtype.itemsize
            pointer = self._buffer(name, size)
            if not self.context.set_tensor_address(name, pointer):
                raise RuntimeError(f"Failed to bind TensorRT tensor {name!r}")
            outputs[name] = np.empty(shape, dtype=dtype)

        # Resolve/bind every tensor before scheduling copies. Keep the host
        # arrays alive until the stream is drained even on an execution error.
        try:
            for name, array in host_inputs.items():
                self.cuda.copy_to_device(self._buffers[name].pointer, array)
            if not self.context.execute_async_v3(self.cuda.stream):
                raise RuntimeError(f"TensorRT execution failed: {self.path.name}")
            for name, array in outputs.items():
                self.cuda.copy_to_host(array, self._buffers[name].pointer)
        except BaseException:
            try:
                self.cuda.synchronize()
            except Exception:
                pass  # Keep the original failure for diagnosis.
            raise
        self.cuda.synchronize()
        return outputs

    def close(self) -> None:
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
