"""Reviewed JP6 cuDNN9.4 FP32 bindings and explicit resource ownership.

Importing this module performs no library loading, CUDA calls or model work.
The runtime loads the eight pinned definitions with its original ABI globals.
"""
import ctypes as ct
import hashlib
import json
import platform

P, I, S, U = ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_uint32
PP, PI, PS = ct.POINTER(P), ct.POINTER(I), ct.POINTER(S)
H = 256
CAP_BYTES = 512 * 2**20


def need(value, message):
    if not value:
        raise RuntimeError(message)


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class APIError(RuntimeError):
    def __init__(self, library, operation, status, text):
        self.details = {"library": library, "operation": operation,
                        "status_code": int(status), "status_name": text}
        self.unsupported = library == "cuDNN" and "NOT_SUPPORTED" in text
        super().__init__(json.dumps(self.details))


class API:
    """Signatures are from the versioned NVIDIA docs and read-only headers above."""
    def __init__(self):
        need(ct.sizeof(P) == ct.sizeof(S) == 8 and ct.sizeof(I) == 4, "64-bit Linux ABI required")
        need(platform.system() == "Linux" and platform.machine() == "aarch64", "Reviewed JP6 target only")
        self.dnn = ct.CDLL("libcudnn.so.9")
        self.cuda = ct.CDLL("libcudart.so.12")
        self.bind(self.dnn, "cudnnGetVersion", S, [])
        self.bind(self.dnn, "cudnnGetErrorString", ct.c_char_p, [I])
        self.bind(self.cuda, "cudaGetErrorString", ct.c_char_p, [I])
        need(int(self.dnn.cudnnGetVersion()) == 90400, "Exact cuDNN 9.4.0 required; no version fallback")
        ds = {
            "cudnnCreate": [PP], "cudnnDestroy": [P], "cudnnSetStream": [P, P],
            "cudnnGetStream": [P, PP], "cudnnCreateRNNDescriptor": [PP],
            "cudnnDestroyRNNDescriptor": [P], "cudnnCreateRNNDataDescriptor": [PP],
            "cudnnDestroyRNNDataDescriptor": [P], "cudnnCreateTensorDescriptor": [PP],
            "cudnnDestroyTensorDescriptor": [P], "cudnnCreateDropoutDescriptor": [PP],
            "cudnnDestroyDropoutDescriptor": [P], "cudnnDropoutGetStatesSize": [P, PS],
            "cudnnSetDropoutDescriptor": [P, P, ct.c_float, P, S, ct.c_ulonglong],
            "cudnnSetRNNDescriptor_v8": [P] + [I] * 8 + [ct.c_int32] * 4 + [P, U],
            "cudnnGetRNNDescriptor_v8": [P] + [PI] * 12 + [PP, ct.POINTER(U)],
            "cudnnSetRNNDataDescriptor": [P, I, I, I, I, I, PI, P],
            "cudnnGetRNNTempSpaceSizes": [P, P, I, P, PS, PS],
            "cudnnGetRNNWeightSpaceSize": [P, P, PS],
            "cudnnGetRNNWeightParams": [P, P, ct.c_int32, S, P, ct.c_int32, P, PP, P, PP],
            "cudnnSetTensorNdDescriptor": [P, I, I, PI, PI],
            "cudnnGetTensorNdDescriptor": [P, I, PI, PI, PI, PI],
            "cudnnRNNForward": [P, P, I, P, P, P, P, P, P, P, P, P, P, P, S, P, S, P, S, P],
        }
        cs = {
            "cudaStreamCreate": [PP], "cudaStreamDestroy": [P], "cudaStreamSynchronize": [P],
            "cudaMalloc": [PP, S], "cudaFree": [P], "cudaMemcpyAsync": [P, P, S, I, P],
            "cudaMemsetAsync": [P, I, S, P], "cudaMemGetInfo": [PS, PS],
            "cudaRuntimeGetVersion": [PI], "cudaGetDevice": [PI],
            "cudaEventCreate": [PP], "cudaEventDestroy": [P],
            "cudaEventRecord": [P, P], "cudaEventSynchronize": [P],
            "cudaEventElapsedTime": [ct.POINTER(ct.c_float), P, P],
        }
        for name, args in ds.items():
            self.bind(self.dnn, name, I, args)
        for name, args in cs.items():
            self.bind(self.cuda, name, I, args)

    @staticmethod
    def bind(lib, name, result, args):
        function = getattr(lib, name)
        function.restype, function.argtypes = result, args

    def call(self, name, *args):
        is_dnn = name.startswith("cudnn")
        lib = self.dnn if is_dnn else self.cuda
        status = int(getattr(lib, name)(*args))
        if status:
            text = (self.dnn.cudnnGetErrorString(status) if is_dnn
                    else self.cuda.cudaGetErrorString(status))
            raise APIError("cuDNN" if is_dnn else "CUDA", name, status,
                           text.decode(errors="replace") if text else "unknown status")

    def memory(self):
        free, total = S(), S()
        self.call("cudaMemGetInfo", ct.byref(free), ct.byref(total))
        return {"free_bytes": free.value, "total_bytes": total.value}


class Resources:
    def __init__(self, api, report, stream):
        self.api, self.report = api, report
        self.descriptors, self.allocations, self.events = [], [], []
        self.total, self.peak = 0, 0
        self.stream, self.owns_stream, self.handle = P(), stream is None, P()
        if self.owns_stream:
            api.call("cudaStreamCreate", ct.byref(self.stream))
        else:
            need(isinstance(stream, int) and stream >= 0, "Invalid borrowed stream")
            self.stream = P(stream)
        try:
            api.call("cudnnCreate", ct.byref(self.handle))
            api.call("cudnnSetStream", self.handle, self.stream)
            actual = P()
            api.call("cudnnGetStream", self.handle, ct.byref(actual))
            need(actual.value == self.stream.value, "cuDNN stream readback mismatch")
        except BaseException:
            self.close()
            raise

    def descriptor(self, kind):
        value = P()
        self.api.call("cudnnCreate" + kind + "Descriptor", ct.byref(value))
        self.descriptors.append((kind, value))
        return value

    def allocate(self, name, size):
        size = int(size)
        need(0 <= size <= CAP_BYTES and self.total + size <= CAP_BYTES,
             "Explicit CUDA allocation cap 512MiB exceeded")
        if not size:
            return P()
        ptr = P()
        self.api.call("cudaMalloc", ct.byref(ptr), size)
        self.allocations.append((name, ptr, size))
        self.total += size
        self.peak = max(self.peak, self.total)
        return ptr

    def event(self):
        event = P()
        self.api.call("cudaEventCreate", ct.byref(event))
        self.events.append(event)
        return event

    def close(self):
        errors = []
        def attempt(name, *args):
            try:
                self.api.call(name, *args)
                return True
            except BaseException as error:
                errors.append(str(error))
                return False
        if self.stream or not self.owns_stream:
            attempt("cudaStreamSynchronize", self.stream)
        for event in reversed(self.events):
            attempt("cudaEventDestroy", event)
        for kind, value in reversed(self.descriptors):
            attempt("cudnnDestroy" + kind + "Descriptor", value)
        for _, ptr, size in reversed(self.allocations):
            if attempt("cudaFree", ptr):
                self.total -= size
        if self.handle:
            attempt("cudnnDestroy", self.handle)
        if self.owns_stream and self.stream:
            attempt("cudaStreamDestroy", self.stream)
        self.report["explicit_cuda_allocations"] = [
            {"name": name, "bytes": size} for name, _, size in self.allocations]
        self.report["explicit_cuda_peak_bytes"] = self.peak
        self.report["explicit_cuda_allocation_total_bytes"] = sum(size for _, _, size in self.allocations)
        self.report["explicit_cuda_unfreed_bytes"] = self.total
        self.report["cleanup_errors"] = errors
        self.report["cleanup_complete"] = not errors and self.total == 0
        self.events.clear()
        self.descriptors.clear()
        self.allocations.clear()
        self.handle = P()
        self.stream = P()


def tensor_shape(api, descriptor):
    dtype, ndim = I(), I()
    dims, strides = (I * 8)(), (I * 8)()
    api.call("cudnnGetTensorNdDescriptor", descriptor, 8, ct.byref(dtype),
             ct.byref(ndim), dims, strides)
    need(0 < ndim.value <= 8, "Unexpected tensor descriptor rank")
    return dtype.value, list(dims)[:ndim.value], list(strides)[:ndim.value]


def dense_strides(shape):
    result = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        result[i] = result[i + 1] * shape[i + 1]
    return result


def rnn_readback(api, descriptor):
    values, dropout, flags = [I() for _ in range(12)], P(), U()
    api.call("cudnnGetRNNDescriptor_v8", descriptor,
             *[ct.byref(v) for v in values], ct.byref(dropout), ct.byref(flags))
    names = ["algorithm", "cell_mode", "bias_mode", "direction", "input_mode",
             "data_type", "math_precision", "math_type", "input_size",
             "hidden_size", "projection_size", "num_layers"]
    return dict(zip(names, [v.value for v in values])), flags.value
