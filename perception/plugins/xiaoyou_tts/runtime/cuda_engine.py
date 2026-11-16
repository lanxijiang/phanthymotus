"""TensorRT GPU execution only, with serialized shared activation memory.

No Torch/ORT/CPU neural implementation is imported. Named V3 and explicit-binding
V2 APIs are separated for version-specific engine packs; a plan is never rebuilt.
"""
import ctypes
import ctypes.util
import gc
import math
import os
import threading
import time

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
    def __init__(self, pool, block, shape, dtype, *, _parent=None):
        self.pool, self.block = pool, block
        self._parent = _parent
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.released = False

    @property
    def pointer(self):
        if self.released or self.block['freed'] or self.pool.closed or self.pool.closing or self.pool.poisoned:
            raise RuntimeError('V6 device tensor is no longer live')
        if self._parent is not None:
            return self._parent.pointer
        return self.block['pointer']

    @property
    def readonly(self):
        return self._parent is not None

    def prefix_view(self, shape):
        """Borrow a dense read-only prefix; the parent retains its allocation."""
        with self.pool.lock:
            self.pointer  # Validate this owner before creating any borrowed view.
            shape = tuple(shape)
            if not shape or any(type(d) is not int or d <= 0 for d in shape) or math.prod(shape) > math.prod(self.shape):
                raise ValueError('V6 device prefix requires positive integer dimensions within its parent')
            return DeviceTensor(self.pool, self.block, shape, self.dtype, _parent=self)

    def numpy(self):
        return self.pool.download(self)

    def release(self):
        if not self.released:
            self.released = True
            if self._parent is None:
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
        # infer_shapes() has no stream argument. Only defer host-copy waits when
        # engine metadata proves that input values cannot drive shape inference.
        self.can_defer_upload_sync = False
        query = getattr(engine, 'is_shape_inference_io' if self.named else 'is_shape_binding', None)
        if callable(query):
            try:
                self.can_defer_upload_sync = all(query(n if self.named else self.index(n)) is False for n in self.inputs)
            except Exception:
                self.can_defer_upload_sync = False

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


def _nvtx_label(value):
    if value is None:
        return 'null'
    name = str(getattr(value, 'name', value))
    # Compact metadata alias only; the SDK property retains its original enum.
    return 'NAMES' if name == 'LAYER_NAMES_ONLY' else name


class GPUEnginePool:
    """One GPU engine set and activation workspace, shared by logical sessions.

    Caller serializes jobs. This object additionally locks all allocation/execution
    paths so inference contexts and the shared workspace cannot execute concurrently.
    """
    def __init__(self, root, stages, *, expected_trt_prefix, max_explicit_bytes=1024 * 1024**2,
                 shape_workspace_cache_bytes=None):
        workspace_mode = os.environ.get('XIAOYOU01_WORKSPACE_MODE', 'fixed')
        if workspace_mode not in ('fixed', 'shape'):
            raise ValueError('V6 workspace mode must be fixed or shape')
        if shape_workspace_cache_bytes is None:
            configured_cap = os.environ.get('XIAOYOU01_SHAPE_WORKSPACE_CACHE_BYTES', '0')
            if not configured_cap.isascii() or not configured_cap.isdecimal():
                raise ValueError('V6 shape workspace cache environment cap must be decimal bytes')
            shape_workspace_cache_bytes = int(configured_cap)
        if (type(shape_workspace_cache_bytes) is not int or shape_workspace_cache_bytes < 0
                or shape_workspace_cache_bytes > int(max_explicit_bytes)):
            raise ValueError('V6 shape workspace cache cap must be an integer within the explicit allocation cap')
        if shape_workspace_cache_bytes and workspace_mode != 'shape':
            raise ValueError('V6 shape workspace cache requires explicit shape workspace mode')
        nvtx_mode = os.environ.get('XIAOYOU01_NVTX_VERBOSITY', 'original')
        if nvtx_mode not in ('original', 'none'):
            raise ValueError('V6 NVTX verbosity must be original or none')
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
        self.closing = False
        self.poisoned = False
        self._stream_drained = True
        self._quarantined_refs = []
        self.blocks = {}
        self.cached_bytes = 0
        self.io_bytes = 0
        self._io_cache_primed = False
        self.peak_explicit_bytes = 0
        self.max_explicit_bytes = int(max_explicit_bytes)
        self.max_cache_bytes = 64 * 1024**2
        self.workspace = None
        self.workspace_bytes = 0
        self.workspace_mode = workspace_mode
        # Opt-in only. At most one idle allocation per validated stage is kept,
        # with a strict total byte cap. These allocations remain in every budget.
        self.max_workspace_cache_bytes = shape_workspace_cache_bytes
        self._workspace_cache = {}
        self._workspace_cached_bytes = 0
        self._workspace_stage = None
        self._workspace_requested_bytes = 0
        self.workspace_allocation_count = 0
        self.workspace_reuse_count = 0
        self.workspace_cross_stage_reuse_count = 0
        self.workspace_free_count = 0
        self.workspace_upper_bound_bytes = 0
        self.workspace_peak_bytes = 0
        self.workspace_by_stage = {}
        self.engines = {}
        from . import fused_lstm
        self._fused_lstm = fused_lstm
        fused_lstm.register_creator(trt)
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
                elif self.workspace_mode == 'fixed':
                    context = engine.create_execution_context_without_device_memory()
                else:
                    raise RuntimeError('V6 shape workspace requires USER_MANAGED context allocation; no fallback')
                if context is None:
                    raise RuntimeError('V6 user-managed execution context unavailable')
                nvtx_before = getattr(context, 'nvtx_verbosity', None)
                engine_verbosity = getattr(engine, 'profiling_verbosity', None)
                nvtx_after = nvtx_before
                if nvtx_mode == 'none':
                    quiet = getattr(getattr(trt, 'ProfilingVerbosity', None), 'NONE', None)
                    if nvtx_before is None or quiet is None:
                        raise RuntimeError('V6 explicit NVTX none requires context/SDK support; no fallback')
                    context.nvtx_verbosity = quiet
                    nvtx_after = context.nvtx_verbosity
                    if nvtx_after != quiet:
                        raise RuntimeError('V6 NVTX verbosity readback differs; no fallback')
                api = _API(engine, context, trt)
                if set(api.names) != set(spec['inputs']) | set(spec['outputs']) or api.inputs != set(spec['inputs']):
                    raise RuntimeError('V6 engine I/O name/mode mismatch: ' + name)
                for tensor_name in api.names:
                    if api.dtype(tensor_name) != np.dtype(spec['io_dtypes'][tensor_name]):
                        raise RuntimeError('V6 engine I/O dtype mismatch: ' + tensor_name)
                    api.require_device_linear(tensor_name)
                    if tensor_name in api.inputs and api.profile(tensor_name) != spec['profiles'][tensor_name]:
                        raise RuntimeError('V6 engine profile mismatch: ' + tensor_name)
                upper = engine.device_memory_size
                if self.workspace_mode == 'shape':
                    if (not api.named or type(upper) is not int or upper <= 0
                            or not callable(getattr(context, 'update_device_memory_size_for_shapes', None))
                            or not callable(getattr(context, 'set_device_memory', None))):
                        raise RuntimeError('V6 shape workspace requires the verified TensorRT10.4 memory APIs')
                upper = int(upper)
                self.workspace_upper_bound_bytes = max(self.workspace_upper_bound_bytes, upper)
                self.workspace_by_stage[name] = {'engine_max_bytes': upper, 'query_count': 0,
                    'lease_count': 0, 'allocation_count': 0, 'reuse_count': 0, 'cross_stage_reuse_count': 0,
                    'last_requested_bytes': None, 'peak_requested_bytes': 0,
                    'nvtx': {
                        'before': _nvtx_label(nvtx_before),
                        'after': _nvtx_label(nvtx_after),
                        'engine': _nvtx_label(engine_verbosity)}}
                self.engines[name] = api
                engine = context = api = None
            if self.workspace_mode == 'fixed':
                if self.workspace_upper_bound_bytes > self.max_explicit_bytes:
                    raise RuntimeError('V6 activation workspace exceeds explicit GPU allocation cap')
                if self.workspace_upper_bound_bytes:
                    self.workspace = ctypes.c_void_p()
                    self.cuda.call('cudaMalloc', ctypes.byref(self.workspace), self.workspace_upper_bound_bytes)
                    self.workspace_bytes = self.workspace_upper_bound_bytes
                    self.workspace_allocation_count += 1
                    for obj in self.engines.values():
                        obj.context.device_memory = self.workspace.value
                self.workspace_peak_bytes = self.workspace_bytes
            self.peak_explicit_bytes = self.workspace_bytes
            self._prime_io_cache()
        except BaseException:
            engine = context = api = None
            self.close()
            raise

    def _prime_io_cache(self):
        """Reserve the verified three-stage binding capacities, with no NN/copy.

        Keep every lease live until all reservations exist: releasing each one
        immediately would repeatedly borrow the same best-fit block. One extra
        style slot stays available for the caller's subsequent persistent upload.
        Partial-stage diagnostic helpers retain their original allocation behavior.
        """
        signature = {
            'encoder': ({'tokens', 'style', 'speed'}, {'d', 't_en', 'duration'}),
            'acoustic': ({'d', 't_en', 'indices', 'style'}, {'features', 'f0'}),
            'vocoder': ({'features', 'f0', 'style', 'source_phase', 'excitation_noise'}, {'audio'}),
        }
        if set(self.specs) != set(signature) or any(
                (set(self.specs[k]['inputs']), set(self.specs[k]['outputs'])) != signature[k]
                for k in signature):
            return
        # The constructor calls this only after all original engine/profile/ABI
        # checks. Derive sizes from those actual metadata, never an audio probe.
        if set(self.engines) != set(signature) or self.blocks or self.io_bytes or self.cached_bytes:
            raise RuntimeError('V6 I/O cache priming requires the validated fresh three-stage pool')
        def maximum(stage, name):
            shape = tuple(self.specs[stage]['profiles'][name][-1])
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise RuntimeError('V6 I/O cache profile shape is invalid')
            return shape
        n_max = maximum('encoder', 'tokens')[-1]
        f_max = maximum('vocoder', 'f0')[-1]
        # The original certified operating range ends at F1460; acoustic F=2T.
        # Its physical acoustic profile is wider, so do not reserve T832 audio.
        if f_max != 1460 or f_max % 2:
            raise RuntimeError('V6 I/O cache requires the certified F1460 operating bound')
        source = maximum('vocoder', 'source_phase')
        if source != maximum('vocoder', 'excitation_noise'):
            raise RuntimeError('V6 I/O cache excitation profiles differ')
        symbols = {'N': n_max, 'T': min(maximum('acoustic', 'indices')[0], f_max // 2),
                   'F': f_max, 'samples': source[1]}
        reservations = [('encoder', 'style'), ('encoder', 'tokens'), ('encoder', 'speed'),
                        ('encoder', 'd'), ('encoder', 't_en'), ('encoder', 'duration'),
                        ('acoustic', 'indices'), ('acoustic', 'features'), ('acoustic', 'f0'),
                        ('vocoder', 'source_phase'), ('vocoder', 'excitation_noise'), ('vocoder', 'audio')]
        typed = []
        for stage, name in reservations:
            spec = self.specs[stage]
            if name == 'indices':
                shape = (symbols['T'],)
            elif name in spec['inputs']:
                shape = maximum(stage, name)
            else:
                shape = tuple(symbols[d] if isinstance(d, str) else d for d in spec['outputs'][name])
            dtype = np.dtype(spec['io_dtypes'][name])
            if not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise RuntimeError('V6 I/O cache binding shape is invalid')
            typed.append((shape, dtype))
        total = sum(math.prod(shape) * dtype.itemsize for shape, dtype in typed)
        if total > self.max_cache_bytes:
            raise RuntimeError('V6 I/O cache priming exceeds the unchanged cache cap')
        if max(self.workspace_bytes, self.workspace_upper_bound_bytes) + total > self.max_explicit_bytes:
            raise RuntimeError('V6 I/O cache priming exceeds the unchanged explicit allocation cap')
        # On any failure the constructor's original close transaction owns all
        # allocated blocks, including leases still held by this traceback.
        leases = [self._allocate(shape, dtype) for shape, dtype in typed]
        if len({lease.pointer.value for lease in leases}) != len(typed):
            raise RuntimeError('V6 I/O cache priming did not reserve distinct live blocks')
        for lease in leases:
            lease.release()
        self._io_cache_primed = True

    def _require_live(self):
        if self.closed or self.closing:
            raise RuntimeError('V6 GPU engine pool is closed or closing')
        if self.poisoned:
            raise RuntimeError('V6 GPU engine pool is poisoned; no further inference')

    def _synchronize(self):
        try:
            self.cuda.call('cudaStreamSynchronize', self.stream)
        except BaseException:
            self.poisoned = True
            self._stream_drained = False
            raise
        self._stream_drained = True

    def _retain_until_drain(self, *references):
        # A failed sync cannot prove that DMA/kernel reads have stopped. Keep
        # actual host copies and tensor owners alive until a successful close.
        # Poison blocks new work, so this remains bounded to the failing call.
        self._quarantined_refs.append(references)

    def _activation_allocated_bytes(self):
        return self.workspace_bytes + self._workspace_cached_bytes

    def _evict_workspace(self, stage):
        block = self._workspace_cache.get(stage)
        if block is None:
            return
        # A previous lease was retired only after a drain. A new infer may have
        # queued H2D copies since then; explicitly drain before cudaFree anyway.
        if not self._stream_drained:
            self._synchronize()
        try:
            self.cuda.call('cudaFree', block['pointer'])
        except BaseException:
            self.poisoned = True
            raise
        self._workspace_cached_bytes -= block['bytes']
        del self._workspace_cache[stage]
        self.workspace_free_count += 1

    def _free_workspace_cache(self):
        for stage in list(self._workspace_cache):
            self._evict_workspace(stage)

    def _release_workspace(self, *, retain, _activation_timings=None):
        """Retire an active lease after drain, or discard all on an error."""
        if self.workspace is not None:
            if not self._stream_drained:
                self.poisoned = True
                raise RuntimeError('V6 cannot retire activation memory before a confirmed stream drain')
            if (retain and self.max_workspace_cache_bytes
                    and self.workspace_bytes <= self.max_workspace_cache_bytes):
                stage = self._workspace_stage
                if stage is None or stage in self._workspace_cache:
                    self.poisoned = True
                    raise RuntimeError('V6 invalid shape workspace cache ownership')
                # Insertion order is LRU: renting removes an entry and returning
                # appends it. Never exceed the cap even momentarily while idle.
                while self._workspace_cached_bytes + self.workspace_bytes > self.max_workspace_cache_bytes:
                    self._evict_workspace(next(iter(self._workspace_cache)))
                self._workspace_cache[stage] = {'pointer': self.workspace, 'bytes': self.workspace_bytes}
                self._workspace_cached_bytes += self.workspace_bytes
                self.workspace = None
                self.workspace_bytes = 0
                self._workspace_stage = None
                self._workspace_requested_bytes = 0
            else:
                self._free_workspace(_activation_timings=_activation_timings)
        if not retain:
            self._free_workspace_cache()

    def _rent_workspace(self, stage, api, *, _activation_timings=None):
        if self.workspace is not None or self.workspace_bytes:
            raise RuntimeError('V6 shape workspace still has an outstanding lease')
        context = api.context
        profile = getattr(context, 'active_optimization_profile', None)
        if (not callable(getattr(context, 'update_device_memory_size_for_shapes', None))
                or not callable(getattr(context, 'set_device_memory', None))
                or type(profile) is not int or profile != 0):
            raise RuntimeError('V6 shape workspace API/profile unavailable; no fallback')
        record = self.workspace_by_stage[stage]
        record['query_count'] += 1
        started = time.perf_counter() if _activation_timings is not None else None
        try:
            required = context.update_device_memory_size_for_shapes()
        finally:
            if _activation_timings is not None:
                _activation_timings['query_wall_seconds'] = time.perf_counter() - started
        if type(required) is not int or not 0 < required <= record['engine_max_bytes']:
            raise RuntimeError('V6 actual shape workspace size is invalid or exceeds the engine bound')
        record['last_requested_bytes'] = required
        record['peak_requested_bytes'] = max(record['peak_requested_bytes'], required)
        block = self._workspace_cache.get(stage)
        # Reuse prefers this stage, then best-fit idle memory from any stage. A
        # temporary large shape cannot leave a large high-water block attached
        # to all later small requests: a >2x size drop reallocates at actual size.
        reuse = (block is not None and required <= block['bytes'] <= 2 * required)
        if block is not None and not reuse:
            self._evict_workspace(stage)
        source_stage = stage
        if not reuse:
            compatible = [(name, cached) for name, cached in self._workspace_cache.items()
                          if required <= cached['bytes'] <= 2 * required]
            if compatible:
                source_stage, block = min(compatible, key=lambda item: item[1]['bytes'])
                reuse = True
        if reuse:
            # Each idle block was retired after a confirmed stream drain. Remove
            # its old owner before leasing; return will label it for this stage.
            del self._workspace_cache[source_stage]
            self._workspace_cached_bytes -= block['bytes']
            pointer, capacity = block['pointer'], block['bytes']
            self.workspace_reuse_count += 1
            record['reuse_count'] += 1
            if source_stage != stage:
                self.workspace_cross_stage_reuse_count += 1
                record['cross_stage_reuse_count'] += 1
            if _activation_timings is not None:
                _activation_timings['malloc_wall_seconds'] = 0.0
        else:
            if self.io_bytes + self._activation_allocated_bytes() + required > self.max_explicit_bytes:
                self._trim_cache()
            while (self._workspace_cache
                   and self.io_bytes + self._activation_allocated_bytes() + required > self.max_explicit_bytes):
                self._evict_workspace(next(iter(self._workspace_cache)))
            if self.io_bytes + self._activation_allocated_bytes() + required > self.max_explicit_bytes:
                raise RuntimeError('V6 explicit GPU allocation cap exceeded by shape workspace; no fallback')
            pointer, capacity = ctypes.c_void_p(), required
            try:
                started = time.perf_counter() if _activation_timings is not None else None
                try:
                    self.cuda.call('cudaMalloc', ctypes.byref(pointer), required)
                finally:
                    if _activation_timings is not None:
                        _activation_timings['malloc_wall_seconds'] = time.perf_counter() - started
            except BaseException:
                if pointer.value:
                    self.workspace, self.workspace_bytes = pointer, capacity
                    self._workspace_stage, self._workspace_requested_bytes = stage, required
                    self.workspace_allocation_count += 1
                    record['allocation_count'] += 1
                    allocated = self._activation_allocated_bytes()
                    self.workspace_peak_bytes = max(self.workspace_peak_bytes, allocated)
                    self.peak_explicit_bytes = max(self.peak_explicit_bytes, allocated + self.io_bytes)
                    self.poisoned = True
                raise
            if not pointer.value:
                raise RuntimeError('V6 shape workspace allocation returned a null pointer')
            self.workspace_allocation_count += 1
            record['allocation_count'] += 1
        self.workspace, self.workspace_bytes = pointer, capacity
        self._workspace_stage, self._workspace_requested_bytes = stage, required
        allocated = self._activation_allocated_bytes()
        self.workspace_peak_bytes = max(self.workspace_peak_bytes, allocated)
        self.peak_explicit_bytes = max(self.peak_explicit_bytes, allocated + self.io_bytes)
        record['lease_count'] += 1
        # TensorRT10.4's setter returns None on success. A context may retain
        # this address while idle; every shape-mode execute rebinds a fresh lease.
        started = time.perf_counter() if _activation_timings is not None else None
        try:
            context.set_device_memory(pointer.value, capacity)
        finally:
            if _activation_timings is not None:
                _activation_timings['bind_wall_seconds'] = time.perf_counter() - started

    def _free_workspace(self, *, _activation_timings=None):
        if self.workspace is None:
            return
        if not self._stream_drained:
            self.poisoned = True
            raise RuntimeError('V6 cannot free activation memory before a confirmed stream drain')
        try:
            started = time.perf_counter() if _activation_timings is not None else None
            try:
                self.cuda.call('cudaFree', self.workspace)
            finally:
                if _activation_timings is not None:
                    _activation_timings['free_wall_seconds'] = time.perf_counter() - started
        except BaseException:
            self.poisoned = True
            raise
        self.workspace = None
        self.workspace_bytes = 0
        self._workspace_stage = None
        self._workspace_requested_bytes = 0
        self.workspace_free_count += 1

    def _allocate(self, shape, dtype):
        self._require_live()
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if size <= 0:
            raise ValueError('V6 cannot allocate an empty dynamic tensor')
        candidates = [b for b in self.blocks.values() if not b['in_use'] and not b['freed'] and b['bytes'] >= size]
        if candidates:
            block = min(candidates, key=lambda b: b['bytes'])
            self.cached_bytes -= block['bytes']
            block['in_use'] = True
        else:
            while (self._workspace_cache
                   and self._activation_allocated_bytes() + self.io_bytes + size > self.max_explicit_bytes):
                self._evict_workspace(next(iter(self._workspace_cache)))
            if self._activation_allocated_bytes() + self.io_bytes + size > self.max_explicit_bytes:
                self._trim_cache()
            if self._activation_allocated_bytes() + self.io_bytes + size > self.max_explicit_bytes:
                raise RuntimeError('V6 explicit GPU allocation cap exceeded; no fallback')
            pointer = ctypes.c_void_p()
            self.cuda.call('cudaMalloc', ctypes.byref(pointer), size)
            block = {'pointer': pointer, 'bytes': size, 'in_use': True, 'freed': False}
            self.blocks[pointer.value] = block
            self.io_bytes += size
        self.peak_explicit_bytes = max(self.peak_explicit_bytes, self._activation_allocated_bytes() + self.io_bytes)
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
            if not self._stream_drained:
                block['release_deferred'] = True
                return
            self.cuda.call('cudaSetDevice', 0)
            block['in_use'] = False
            self.cached_bytes += block['bytes']
            if self.cached_bytes > self.max_cache_bytes:
                self._trim_cache()

    def upload(self, array):
        with self.lock:
            self._require_live()
            self.cuda.call('cudaSetDevice', 0)
            a = out = None
            try:
                a = np.ascontiguousarray(array)
                if not np.isfinite(a).all():
                    raise ValueError('V6 upload contains nonfinite data')
                out = self._allocate(a.shape, a.dtype)
                self._stream_drained = False
                self.cuda.call('cudaMemcpyAsync', out.pointer, ctypes.c_void_p(a.ctypes.data), a.nbytes, 1, self.stream)
                self._synchronize()
                return out
            except BaseException:
                try:
                    self._synchronize()
                finally:
                    if self._stream_drained:
                        if out is not None:
                            out.release()
                        self._free_workspace_cache()
                    else:
                        self._retain_until_drain(a, out)
                raise

    def _enqueue_upload(self, array, temporary, host_sources):
        """Queue an infer-owned copy; caller retains buffers through stream sync."""
        self._require_live()
        self.cuda.call('cudaSetDevice', 0)
        a = np.ascontiguousarray(array)
        if not np.isfinite(a).all():
            raise ValueError('V6 upload contains nonfinite data')
        out = self._allocate(a.shape, a.dtype)
        # Register both owners before enqueue: CUDA can still read a temporary
        # contiguous source after this method returns or if a later input fails.
        temporary.append(out)
        host_sources.append(a)
        self._stream_drained = False
        self.cuda.call('cudaMemcpyAsync', out.pointer, ctypes.c_void_p(a.ctypes.data), a.nbytes, 1, self.stream)
        return out

    def download(self, tensor):
        with self.lock:
            self._require_live()
            if tensor.pool is not self:
                self._free_workspace_cache()
                raise RuntimeError('V6 device tensor belongs to a closed/different pool')
            self.cuda.call('cudaSetDevice', 0)
            result = np.empty(tensor.shape, dtype=tensor.dtype)
            try:
                self._stream_drained = False
                self.cuda.call('cudaMemcpyAsync', ctypes.c_void_p(result.ctypes.data), tensor.pointer, result.nbytes, 2, self.stream)
                self._synchronize()
            except BaseException:
                try:
                    self._synchronize()
                finally:
                    if not self._stream_drained:
                        self._retain_until_drain(tensor, result)
                    else:
                        self._free_workspace_cache()
                raise
            if not np.isfinite(result).all():
                self._free_workspace_cache()
                raise RuntimeError('V6 GPU output is nonfinite')
            return result

    def infer(self, stage, feed, expected_outputs, *, _activation_timings=None):
        with self.lock:
            self._require_live()
            self.cuda.call('cudaSetDevice', 0)
            inputs, temporary, outputs = {}, [], {}
            host_sources = []
            completed = False
            try:
                spec, api = self.specs[stage], self.engines[stage]
                if set(feed) != set(spec['inputs']) or set(expected_outputs) != set(spec['outputs']):
                    raise ValueError('V6 stage routing differs: ' + stage)
                for name, value in feed.items():
                    if not isinstance(value, DeviceTensor):
                        if api.can_defer_upload_sync:
                            value = self._enqueue_upload(value, temporary, host_sources)
                        else:
                            value = self.upload(value)
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
                if self.workspace_mode == 'shape':
                    self._rent_workspace(stage, api, _activation_timings=_activation_timings)
                addresses = {name: tensor.pointer.value for name, tensor in {**inputs, **outputs}.items()}
                self._stream_drained = False
                if not api.addresses_and_execute(addresses, self.stream.value):
                    raise RuntimeError('V6 TensorRT execution failed: ' + stage)
                self._synchronize()
                completed = True
                return outputs
            except BaseException:
                try:
                    self._synchronize()
                finally:
                    if self._stream_drained:
                        for value in outputs.values():
                            value.release()
                    else:
                        self._retain_until_drain(inputs, temporary, outputs, host_sources)
                raise
            finally:
                if self._stream_drained:
                    try:
                        if self.workspace_mode == 'shape':
                            self._release_workspace(retain=completed, _activation_timings=_activation_timings)
                    finally:
                        for value in temporary:
                            value.release()

    def stats(self):
        with self.lock:
            allocated = self._activation_allocated_bytes()
            return {'shared_activation_bytes': allocated, 'explicit_io_allocated_bytes': self.io_bytes,
                    'explicit_device_peak_bytes': self.peak_explicit_bytes,
                    'workspace_mode': self.workspace_mode,
                    'activation_upper_bound_bytes': self.workspace_upper_bound_bytes,
                    'activation_current_bytes': allocated, 'activation_peak_bytes': self.workspace_peak_bytes,
                    'activation_allocated_bytes': allocated,
                    'activation_leased_bytes': self.workspace_bytes,
                    'activation_requested_bytes': self._workspace_requested_bytes,
                    'activation_retained_bytes': self._workspace_cached_bytes,
                    'activation_cache_cap_bytes': self.max_workspace_cache_bytes,
                    'activation_allocation_count': self.workspace_allocation_count,
                    'activation_reuse_count': self.workspace_reuse_count,
                    'activation_cross_stage_reuse_count': self.workspace_cross_stage_reuse_count,
                    'activation_free_count': self.workspace_free_count,
                    'activation_by_stage': {name: {**value,
                                            'retained_bytes': self._workspace_cache.get(name, {}).get('bytes', 0),
                                            **({'nvtx': dict(value['nvtx'])} if 'nvtx' in value else {})}
                                            for name, value in self.workspace_by_stage.items()},
                    'poisoned': self.poisoned, 'stream_drained': self._stream_drained,
                    'closed': self.closed, 'closing': self.closing,
                    'quarantined_reference_count': len(self._quarantined_refs),
                    'batched_host_uploads_allowed': {name: bool(getattr(api, 'can_defer_upload_sync', False))
                                                      for name, api in self.engines.items()},
                    'scope': 'Explicit workspace/I-O only, excludes TRT engine weights/libraries and process RAM; total resource limits require device measurement'}

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closing = True
            try:
                self.cuda.call('cudaSetDevice', 0)
                if self.stream.value:
                    self._synchronize()
                # Destruction/free only follows a confirmed drain. On failure,
                # keep owners and the live stream so close() can be retried;
                # the pool stays poisoned and rejects all subsequent inference.
                fused_cleanup = self._fused_lstm.close_all_lifecycle()
                if not fused_cleanup['cleanup_complete']:
                    raise RuntimeError('Private fused LSTM resource cleanup failed')
                for obj in self.engines.values():
                    obj.context = None
                    obj.engine = None
                self.engines.clear()
                gc.collect()
                for block in self.blocks.values():
                    if not block['freed']:
                        self.cuda.call('cudaFree', block['pointer'])
                        block['freed'] = True
                        self.io_bytes -= block['bytes']
                        if not block['in_use']:
                            self.cached_bytes -= block['bytes']
                self.blocks.clear()
                self._free_workspace()
                self._free_workspace_cache()
                self.runtime = None
                gc.collect()
                if self.stream.value:
                    self.cuda.call('cudaStreamDestroy', self.stream)
                    self.stream = ctypes.c_void_p()
                self._quarantined_refs.clear()
            except BaseException:
                self.poisoned = True
                raise
            self.closed = True
            self.closing = False
