"""TensorRT GPU execution only, with serialized shared activation memory.

No Torch/ORT/CPU neural implementation is imported. Named V3 and explicit-binding
V2 APIs are separated for version-specific engine packs; a plan is never rebuilt.
"""
import ctypes
import ctypes.util
import gc
import math
import threading

import numpy as np


class Cuda:
    def __init__(self):
        for name in (ctypes.util.find_library('cudart'), 'libcudart.so.12', 'libcudart.so.11.0', 'libcudart.so'):
            if not name:
                continue
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        else:
            raise RuntimeError('V6 requires a local CUDA runtime; no CPU fallback')
        signatures = {
            'cudaGetErrorString': ([ctypes.c_int], ctypes.c_char_p),
            'cudaSetDevice': ([ctypes.c_int], ctypes.c_int),
            'cudaMalloc': ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            'cudaFree': ([ctypes.c_void_p], ctypes.c_int),
            'cudaMemcpyAsync': ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p], ctypes.c_int),
            'cudaStreamCreate': ([ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
            'cudaStreamSynchronize': ([ctypes.c_void_p], ctypes.c_int),
            'cudaStreamDestroy': ([ctypes.c_void_p], ctypes.c_int),
            'cudaMemGetInfo': ([ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(name + ' failed: ' + repr(self.lib.cudaGetErrorString(status)))

    def memory(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call('cudaMemGetInfo', ctypes.byref(free), ctypes.byref(total))
        return {'free_bytes': free.value, 'total_bytes': total.value}


class DeviceTensor:
    def __init__(self, pool, block, shape, dtype):
        self.pool, self.block = pool, block
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.released = False

    @property
    def pointer(self):
        if self.released or self.block['freed'] or self.pool.closed:
            raise RuntimeError('V6 device tensor is no longer live')
        return self.block['pointer']

    def numpy(self):
        return self.pool.download(self)

    def release(self):
        if not self.released:
            self.released = True
            self.pool.release(self.block)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class _API:
    def __init__(self, engine, context, trt):
        self.engine, self.context, self.trt = engine, context, trt
        self.named = hasattr(engine, 'num_io_tensors') and hasattr(context, 'execute_async_v3')
        if self.named:
            self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            self.inputs = {n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT}
        else:
            self.names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
            self.inputs = {n for n in self.names if engine.binding_is_input(engine.get_binding_index(n))}
        if len(set(self.names)) != len(self.names):
            raise RuntimeError('Duplicate TensorRT binding names')

    def index(self, name):
        return self.engine.get_binding_index(name)

    def dtype(self, name):
        value = self.engine.get_tensor_dtype(name) if self.named else self.engine.get_binding_dtype(self.index(name))
        return np.dtype(self.trt.nptype(value))

    def profile(self, name):
        value = self.engine.get_tensor_profile_shape(name, 0) if self.named else self.engine.get_profile_shape(0, name)
        return [list(v) for v in value]

    def require_device_linear(self, name):
        if self.named:
            if self.engine.get_tensor_location(name) != self.trt.TensorLocation.DEVICE:
                raise RuntimeError('V6 binding is not on GPU: ' + name)
            def get(method):
                fn = getattr(self.engine, method)
                try:
                    return fn(name, 0)
                except TypeError:
                    return fn(name)
            fmt = get('get_tensor_format')
            vector = int(get('get_tensor_vectorized_dim'))
            components = int(get('get_tensor_components_per_element'))
            width = int(get('get_tensor_bytes_per_component'))
        else:
            i = self.index(name)
            if self.engine.get_location(i) != self.trt.TensorLocation.DEVICE:
                raise RuntimeError('V6 legacy binding is not on GPU: ' + name)
            fmt = self.engine.get_binding_format(i)
            vector = int(self.engine.get_binding_vectorized_dim(i))
            components = int(self.engine.get_binding_components_per_element(i))
            width = int(self.engine.get_binding_bytes_per_component(i))
        if fmt != self.trt.TensorFormat.LINEAR or vector != -1 or components not in (-1, 1) or width not in (-1, self.dtype(name).itemsize):
            raise RuntimeError('V6 requires scalar LINEAR GPU bindings: ' + name)

    def set_shape(self, name, shape):
        if self.named:
            self.context.set_input_shape(name, shape)
        else:
            self.context.set_binding_shape(self.index(name), shape)
        if self.shape(name) != tuple(shape):
            raise RuntimeError('V6 input shape readback differs: ' + name)

    def shape(self, name):
        value = self.context.get_tensor_shape(name) if self.named else self.context.get_binding_shape(self.index(name))
        return tuple(value)

    def require_shapes(self):
        if self.named:
            if self.context.infer_shapes():
                raise RuntimeError('V6 engine has unspecified shape tensors')
        elif not self.context.all_binding_shapes_specified:
            raise RuntimeError('V6 legacy binding shapes are incomplete')

    def strides(self, name):
        value = self.context.get_tensor_strides(name) if self.named else self.context.get_strides(self.index(name))
        return tuple(value)

    def addresses_and_execute(self, addresses, stream):
        if self.named:
            for name, pointer in addresses.items():
                self.context.set_tensor_address(name, pointer)
                if int(self.context.get_tensor_address(name)) != pointer:
                    raise RuntimeError('V6 tensor address readback mismatch')
            return self.context.execute_async_v3(stream_handle=stream)
        ordered = [addresses[n] for n in self.names]
        return self.context.execute_async_v2(bindings=ordered, stream_handle=stream)


class GPUEnginePool:
    """One GPU engine set and activation workspace, shared by logical sessions.

    Caller serializes jobs. This object additionally locks all allocation/execution
    paths so inference contexts and the shared workspace cannot execute concurrently.
    """
    def __init__(self, root, stages, *, expected_trt_prefix, max_explicit_bytes=1024 * 1024**2):
        import tensorrt as trt
        if not trt.__version__.startswith(expected_trt_prefix + '.') and trt.__version__ != expected_trt_prefix:
            raise RuntimeError('V6 engine pack requires TensorRT ' + expected_trt_prefix)
        self.trt, self.version = trt, trt.__version__
        self.cuda = Cuda()
        self.cuda.call('cudaSetDevice', 0)
        self.lock = threading.RLock()
        self.stream = ctypes.c_void_p()
        self.root, self.specs = root, stages
        self.closed = False
        self.blocks = {}
        self.cached_bytes = 0
        self.io_bytes = 0
        self.peak_explicit_bytes = 0
        self.max_explicit_bytes = int(max_explicit_bytes)
        self.max_cache_bytes = 64 * 1024**2
        self.workspace = None
        self.workspace_bytes = 0
        self.engines = {}
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        engine = context = api = None
        try:
            self.cuda.call('cudaStreamCreate', ctypes.byref(self.stream))
            self.runtime = trt.Runtime(self.logger)
            for name, spec in stages.items():
                engine = self.runtime.deserialize_cuda_engine((root / spec['file']).read_bytes())
                if engine is None or engine.num_optimization_profiles != 1:
                    raise RuntimeError('V6 engine/profile unavailable: ' + name)
                if hasattr(trt, 'ExecutionContextAllocationStrategy'):
                    context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
                else:
                    context = engine.create_execution_context_without_device_memory()
                if context is None:
                    raise RuntimeError('V6 user-managed execution context unavailable')
                api = _API(engine, context, trt)
                if set(api.names) != set(spec['inputs']) | set(spec['outputs']) or api.inputs != set(spec['inputs']):
                    raise RuntimeError('V6 engine I/O name/mode mismatch: ' + name)
                for tensor_name in api.names:
                    if api.dtype(tensor_name) != np.dtype(spec['io_dtypes'][tensor_name]):
                        raise RuntimeError('V6 engine I/O dtype mismatch: ' + tensor_name)
                    api.require_device_linear(tensor_name)
                    if tensor_name in api.inputs and api.profile(tensor_name) != spec['profiles'][tensor_name]:
                        raise RuntimeError('V6 engine profile mismatch: ' + tensor_name)
                self.workspace_bytes = max(self.workspace_bytes, int(engine.device_memory_size))
                self.engines[name] = api
                engine = context = api = None
            if self.workspace_bytes > self.max_explicit_bytes:
                raise RuntimeError('V6 activation workspace exceeds explicit GPU allocation cap')
            if self.workspace_bytes:
                self.workspace = ctypes.c_void_p()
                self.cuda.call('cudaMalloc', ctypes.byref(self.workspace), self.workspace_bytes)
                for obj in self.engines.values():
                    obj.context.device_memory = self.workspace.value
            self.peak_explicit_bytes = self.workspace_bytes
        except BaseException:
            engine = context = api = None
            self.close()
            raise

    def _allocate(self, shape, dtype):
        if self.closed:
            raise RuntimeError('V6 GPU engine pool is closed')
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if size <= 0:
            raise ValueError('V6 cannot allocate an empty dynamic tensor')
        candidates = [b for b in self.blocks.values() if not b['in_use'] and not b['freed'] and b['bytes'] >= size]
        if candidates:
            block = min(candidates, key=lambda b: b['bytes'])
            self.cached_bytes -= block['bytes']
            block['in_use'] = True
        else:
            if self.workspace_bytes + self.io_bytes + size > self.max_explicit_bytes:
                self._trim_cache()
            if self.workspace_bytes + self.io_bytes + size > self.max_explicit_bytes:
                raise RuntimeError('V6 explicit GPU allocation cap exceeded; no fallback')
            pointer = ctypes.c_void_p()
            self.cuda.call('cudaMalloc', ctypes.byref(pointer), size)
            block = {'pointer': pointer, 'bytes': size, 'in_use': True, 'freed': False}
            self.blocks[pointer.value] = block
            self.io_bytes += size
        self.peak_explicit_bytes = max(self.peak_explicit_bytes, self.workspace_bytes + self.io_bytes)
        return DeviceTensor(self, block, shape, dtype)

    def _trim_cache(self):
        for block in list(self.blocks.values()):
            if not block['in_use'] and not block['freed']:
                self.cuda.call('cudaFree', block['pointer'])
                block['freed'] = True
                self.io_bytes -= block['bytes']
                self.cached_bytes -= block['bytes']
                del self.blocks[block['pointer'].value]

    def release(self, block):
        with self.lock:
            if block['freed'] or not block['in_use']:
                return
            self.cuda.call('cudaSetDevice', 0)
            block['in_use'] = False
            self.cached_bytes += block['bytes']
            if self.cached_bytes > self.max_cache_bytes:
                self._trim_cache()

    def upload(self, array):
        with self.lock:
            if self.closed:
                raise RuntimeError('V6 GPU engine pool is closed')
            self.cuda.call('cudaSetDevice', 0)
            a = np.ascontiguousarray(array)
            if not np.isfinite(a).all():
                raise ValueError('V6 upload contains nonfinite data')
            out = self._allocate(a.shape, a.dtype)
            try:
                self.cuda.call('cudaMemcpyAsync', out.pointer, ctypes.c_void_p(a.ctypes.data), a.nbytes, 1, self.stream)
                self.cuda.call('cudaStreamSynchronize', self.stream)
                return out
            except BaseException:
                out.release()
                raise

    def download(self, tensor):
        with self.lock:
            if self.closed or tensor.pool is not self:
                raise RuntimeError('V6 device tensor belongs to a closed/different pool')
            self.cuda.call('cudaSetDevice', 0)
            result = np.empty(tensor.shape, dtype=tensor.dtype)
            self.cuda.call('cudaMemcpyAsync', ctypes.c_void_p(result.ctypes.data), tensor.pointer, result.nbytes, 2, self.stream)
            self.cuda.call('cudaStreamSynchronize', self.stream)
            if not np.isfinite(result).all():
                raise RuntimeError('V6 GPU output is nonfinite')
            return result

    def infer(self, stage, feed, expected_outputs):
        with self.lock:
            if self.closed:
                raise RuntimeError('V6 GPU engine pool is closed')
            self.cuda.call('cudaSetDevice', 0)
            spec, api = self.specs[stage], self.engines[stage]
            if set(feed) != set(spec['inputs']) or set(expected_outputs) != set(spec['outputs']):
                raise ValueError('V6 stage routing differs: ' + stage)
            inputs, temporary, outputs = {}, [], {}
            try:
                for name, value in feed.items():
                    value = value if isinstance(value, DeviceTensor) else self.upload(value)
                    if value is not feed[name]:
                        temporary.append(value)
                    if value.pool is not self:
                        raise ValueError('V6 cannot mix device tensors from different engine pools')
                    if value.dtype != np.dtype(spec['io_dtypes'][name]):
                        raise ValueError('V6 input dtype differs: ' + name)
                    profile = spec['profiles'][name]
                    if len(value.shape) != len(profile[0]) or any(not lo <= x <= hi for x, lo, hi in zip(value.shape, profile[0], profile[2])):
                        raise ValueError('V6 input exceeds the verified GPU profile: ' + name)
                    api.set_shape(name, value.shape)
                    inputs[name] = value
                api.require_shapes()
                for name in api.names:
                    shape = api.shape(name)
                    expected = inputs[name].shape if name in inputs else tuple(expected_outputs[name])
                    if shape != expected or any(int(d) <= 0 for d in shape):
                        raise RuntimeError('V6 actual GPU output shape differs: ' + name)
                    stride = api.strides(name)
                    dense, product = [], 1
                    for d in reversed(shape):
                        dense.append(product)
                        product *= d
                    dense = tuple(reversed(dense))
                    if len(stride) != len(shape) or any(d != 1 and a != b for d, a, b in zip(shape, stride, dense)):
                        raise RuntimeError('V6 GPU binding is not physically dense: ' + name)
                    if name not in inputs:
                        outputs[name] = self._allocate(shape, spec['io_dtypes'][name])
                addresses = {name: tensor.pointer.value for name, tensor in {**inputs, **outputs}.items()}
                if not api.addresses_and_execute(addresses, self.stream.value):
                    raise RuntimeError('V6 TensorRT execution failed: ' + stage)
                self.cuda.call('cudaStreamSynchronize', self.stream)
                return outputs
            except BaseException:
                try:
                    self.cuda.call('cudaStreamSynchronize', self.stream)
                finally:
                    for value in outputs.values():
                        value.release()
                raise
            finally:
                for value in temporary:
                    value.release()

    def stats(self):
        return {'shared_activation_bytes': self.workspace_bytes, 'explicit_io_allocated_bytes': self.io_bytes,
                'explicit_device_peak_bytes': self.peak_explicit_bytes,
                'scope': 'Explicit workspace/I-O only, excludes TRT engine weights/libraries and process RAM; total resource limits require device measurement'}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.cuda.call('cudaSetDevice', 0)
            if self.stream.value:
                self.cuda.call('cudaStreamSynchronize', self.stream)
            for obj in self.engines.values():
                obj.context = None
                obj.engine = None
            self.engines.clear()
            gc.collect()
            for block in self.blocks.values():
                if not block['freed']:
                    self.cuda.call('cudaFree', block['pointer'])
                    block['freed'] = True
            self.blocks.clear()
            if self.workspace is not None:
                self.cuda.call('cudaFree', self.workspace)
                self.workspace = None
            self.runtime = None
            gc.collect()
            if self.stream.value:
                self.cuda.call('cudaStreamDestroy', self.stream)
                self.stream = ctypes.c_void_p()
