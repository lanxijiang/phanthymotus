"""Kokoro GPU inference using Perception's public CUDA runtime.

Plans: encoder -> durations/text features; acoustic -> features/F0;
vocoder -> audio. All three belong to one model.
Intermediate features stay on GPU. Instances serialize infer/close calls.
"""
from pathlib import Path
import threading
from time import perf_counter

import numpy as np


class SequenceTooLong(ValueError):
    """Predicted audio exceeds the deployed decoder profile; re-split text."""


class KokoroTRT:
    def __init__(self, encoder_plan, acoustic_plan, *, vocoder_plan, device_id=0):
        import tensorrt as trt
        from utils.tensorrt_runtime import CudaRuntime, trt_dtype_to_numpy

        self._lock = threading.RLock()
        self._trt, self._numpy_dtype = trt, trt_dtype_to_numpy
        self._buffers, self._engines, self._contexts = {}, [], []
        paths = (encoder_plan, acoustic_plan, vocoder_plan)
        self._profiles = [None] * len(paths)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._cuda = CudaRuntime(device_id)
        try:
            self._runtime = trt.Runtime(self._logger)
            managed = hasattr(trt, 'ExecutionContextAllocationStrategy')
            for path in paths:
                engine = self._runtime.deserialize_cuda_engine(Path(path).read_bytes())
                if engine is None:
                    raise RuntimeError(f"Cannot deserialize TensorRT plan: {path}")
                self._engines.append(engine)
                context = (engine.create_execution_context(
                    strategy=trt.ExecutionContextAllocationStrategy.USER_MANAGED)
                    if managed else engine.create_execution_context_without_device_memory())
                if context is None:
                    raise RuntimeError(f"Cannot create TensorRT context: {path}")
                self._contexts.append(context)
            # One locked CUDA stream runs stages serially, so their activation
            # memory can share an arena. Weights and I/O buffers remain separate.
            sizes = [e.device_memory_size_v2 if managed else e.device_memory_size
                     for e in self._engines]
            self._activation_bytes = max(sizes)
            if self._activation_bytes:
                pointer = self._buffer('activation', self._activation_bytes)
                for context in self._contexts:
                    if managed:
                        context.set_device_memory(pointer, self._activation_bytes)
                    else:
                        context.device_memory = pointer
            contracts = [({'tokens', 'style', 'speed'}, {'d', 't_en', 'duration'}),
                         ({'d', 't_en', 'indices', 'style'}, {'features', 'f0'}),
                         ({'features', 'f0', 'style'}, {'audio'})]
            for engine, (inputs, outputs) in zip(self._engines, contracts):
                names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
                actual = {n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT}
                if actual != inputs or names - actual != outputs:
                    raise ValueError(f"Unexpected TensorRT I/O: {actual}, {names - actual}")
                if any(engine.get_tensor_location(n) != trt.TensorLocation.DEVICE for n in names):
                    raise ValueError("Kokoro plans must expose device I/O")
            for name in ('d', 't_en', 'style'):
                if self._dtype(0, name) != self._dtype(1, name):
                    raise ValueError(f"A/B dtype mismatch for shared GPU tensor {name}")
            for name in ('features', 'f0', 'style'):
                if self._dtype(1, name) != self._dtype(2, name):
                    raise ValueError(f"Acoustic/vocoder dtype mismatch for {name}")
            limits = []
            duration_inputs = [(self._engines[1], 'indices', 1),
                               (self._engines[2], 'f0', 2)]
            for engine, name, ratio in duration_inputs:
                limits.append(max(engine.get_tensor_profile_shape(name, i)[2][-1]
                                  for i in range(engine.num_optimization_profiles)) // ratio)
            self.max_duration = min(limits)
        except BaseException:
            self.close()
            raise

    def _dtype(self, stage, name):
        return self._numpy_dtype(self._trt, self._engines[stage].get_tensor_dtype(name))

    def _shapes(self, stage, shapes):
        engine, context = self._engines[stage], self._contexts[stage]
        for profile in range(engine.num_optimization_profiles):
            covered = True
            for name, shape in shapes.items():
                declared = tuple(engine.get_tensor_shape(name))
                if all(d >= 0 for d in declared):
                    covered &= tuple(shape) == declared
                else:
                    low, _, high = engine.get_tensor_profile_shape(name, profile)
                    covered &= len(shape) == len(low) and all(a <= d <= b for d, a, b in zip(shape, low, high))
            if covered:
                break
        else:
            raise ValueError(f"Stage {stage} input shapes outside engine profiles: {shapes}")
        if self._profiles[stage] != profile:
            if not context.set_optimization_profile_async(profile, self._cuda.stream_handle):
                raise RuntimeError(f"Cannot select stage {stage} profile {profile}")
            self._profiles[stage] = profile
        for name, shape in shapes.items():
            if not context.set_input_shape(name, tuple(shape)):
                raise ValueError(f"Stage {stage} rejected {name} shape {shape}")

    def _buffer(self, name, size):
        size = max(1, int(size))
        pointer, capacity = self._buffers.get(name, (0, 0))
        if size > capacity:
            if pointer:
                self._cuda.synchronize()
                self._cuda.free(pointer)
                self._buffers.pop(name)
            pointer = self._cuda.malloc(size)
            self._buffers[name] = (pointer, size)
        return pointer

    def _bind(self, stage, name, pointer):
        if not self._contexts[stage].set_tensor_address(name, pointer):
            raise RuntimeError(f"Cannot bind stage {stage} tensor {name}")

    def _upload(self, stage, name, array):
        pointer = self._buffer(f'{stage}:{name}', array.nbytes)
        self._bind(stage, name, pointer)
        self._cuda.copy_host_to_device(pointer, array)
        return pointer

    def _output(self, stage, name):
        shape = tuple(self._contexts[stage].get_tensor_shape(name))
        if any(d < 0 for d in shape):
            raise ValueError(f"Unresolved stage {stage} output {name}: {shape}")
        pointer = self._buffer(f'{stage}:{name}', int(np.prod(shape)) * self._dtype(stage, name).itemsize)
        self._bind(stage, name, pointer)
        return shape, pointer

    def infer(self, tokens, style, speed, *, return_timing=False):
        """Return (audio, duration); optional stage wall times include transfers."""
        with self._lock:
            if self._cuda is None:
                raise RuntimeError("KokoroTRT is closed")
            try:
                start = perf_counter()
                inputs = {n: np.ascontiguousarray(v, dtype=self._dtype(0, n))
                          for n, v in (('tokens', tokens), ('style', style), ('speed', speed))}
                if inputs['tokens'].ndim != 2 or inputs['tokens'].shape[0] != 1:
                    raise ValueError("tokens must have shape [1,N]")
                n = inputs['tokens'].shape[1]
                self._shapes(0, {k: v.shape for k, v in inputs.items()})
                input_ptrs = {k: self._upload(0, k, v) for k, v in inputs.items()}
                outputs = {name: self._output(0, name) for name in ('d', 't_en', 'duration')}
                duration = np.empty(outputs['duration'][0], dtype=self._dtype(0, 'duration'))
                if not self._contexts[0].execute_async_v3(self._cuda.stream_handle):
                    raise RuntimeError("Kokoro encoder GPU execution failed")
                self._cuda.copy_device_to_host(duration, outputs['duration'][1])
                self._cuda.synchronize()
                if duration.shape != (n,) or duration.dtype.kind not in 'iu' or np.any(duration < 1):
                    raise ValueError(f"Invalid predicted duration: {duration.shape}, {duration.dtype}")
                if int(duration.sum()) > self.max_duration:
                    raise SequenceTooLong(
                        f"Predicted duration {duration.sum()} exceeds {self.max_duration}")
                stage_a = perf_counter() - start
                start = perf_counter()
                indices = np.repeat(np.arange(n, dtype=np.int32), duration)
                indices = np.ascontiguousarray(indices, dtype=self._dtype(1, 'indices'))
                self._shapes(1, {'d': outputs['d'][0], 't_en': outputs['t_en'][0],
                                 'indices': indices.shape, 'style': inputs['style'].shape})
                for name in ('d', 't_en'):
                    self._bind(1, name, outputs[name][1])
                self._bind(1, 'style', input_ptrs['style'])
                self._upload(1, 'indices', indices)
                names = ('features', 'f0')
                decoder_outputs = {name: self._output(1, name) for name in names}
                if not self._contexts[1].execute_async_v3(self._cuda.stream_handle):
                    raise RuntimeError("Kokoro decoder GPU execution failed")
                self._cuda.synchronize()  # Release the shared arena before the vocoder.
                self._shapes(2, {**{name: decoder_outputs[name][0] for name in names},
                                 'style': inputs['style'].shape})
                for name in names:
                    self._bind(2, name, decoder_outputs[name][1])
                self._bind(2, 'style', input_ptrs['style'])
                shape, pointer = self._output(2, 'audio')
                if not self._contexts[2].execute_async_v3(self._cuda.stream_handle):
                    raise RuntimeError("Kokoro vocoder GPU execution failed")
                audio = np.empty(shape, dtype=self._dtype(2, 'audio'))
                self._cuda.copy_device_to_host(audio, pointer)
                self._cuda.synchronize()
                if not np.isfinite(audio).all():
                    raise ValueError("Kokoro audio contains non-finite values")
                timing = {'stage_a_s': stage_a, 'stage_b_s': perf_counter() - start}
                return (audio, duration, timing) if return_timing else (audio, duration)
            except BaseException:
                self._cuda.synchronize()
                raise

    def close(self):
        with self._lock:
            if self._cuda is None:
                return
            self._cuda.synchronize()
            self._contexts.clear()
            self._engines.clear()
            self._runtime = None
            for pointer, _ in self._buffers.values():
                self._cuda.free(pointer)
            self._buffers.clear()
            self._cuda.close()
            self._cuda = None
