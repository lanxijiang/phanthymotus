"""Non-neural native Kokoro phase/noise DSP, with no Torch or model execution.

The FP64 interpolation product emulates one correctly rounded FP32 fused multiply
add used by the verified CPU reference. All externally visible tensors are FP32.
"""
import numpy as np


def source_phase(f0, initial_phase, *, block_samples=32768):
    f0 = np.asarray(f0)
    initial_phase = np.asarray(initial_phase)
    if f0.dtype != np.float32 or f0.ndim != 2 or f0.shape[0] != 1 or not np.isfinite(f0).all():
        raise ValueError('Phase DSP requires finite FP32 F0[1,F]')
    if initial_phase.shape != (1, 9) or initial_phase.dtype != np.float32 or not np.isfinite(initial_phase).all() or initial_phase[0, 0] != 0:
        raise ValueError('Native initial phase must be finite FP32[1,9] with fundamental zero')
    if int(block_samples) <= 0:
        raise ValueError('Phase DSP block_samples must be positive')
    count = f0.shape[1]
    if count < 2:
        raise ValueError('Native phase DSP requires at least two F0 frames')
    # Native nearest x300 -> rad -> linear /300 samples positions 149.5,
    # 449.5, ...: both taps belong to the same repeated F0 frame. The initial
    # phase addition affects sample0 only, which neither downsample tap reads.
    rad = np.remainder(f0[:, :, None] * np.arange(1, 10, dtype=np.float32)[None, None, :]
                       / np.float32(24000), np.float32(1))
    phase = np.cumsum(rad, axis=1, dtype=np.float64).astype(np.float32)
    phase = phase * np.float32(2) * np.float32(np.pi) * np.float32(300)
    samples = count * 300
    result = np.empty((1, samples, 9), dtype=np.float32)
    for start in range(0, samples, int(block_samples)):
        stop = min(samples, start + int(block_samples))
        at = (np.arange(start, stop, dtype=np.float32) + np.float32(.5)) * np.float32(1 / 300) - np.float32(.5)
        at = np.maximum(at, np.float32(0))
        left = np.floor(at).astype(np.int64)
        right = np.minimum(left + 1, count - 1)
        fraction = (at - left).astype(np.float32)[None, :, None]
        # Match the original FP32 right product followed by FP32 fma(left).
        right_product = phase[:, right, :] * fraction
        result[:, start:stop, :] = (phase[:, left, :].astype(np.float64)
            * (np.float32(1) - fraction).astype(np.float64) + right_product.astype(np.float64)).astype(np.float32)
    return result


def excitation_inputs(f0, *, seed=42):
    """Retain uniform initial phase and independent standard-normal source noise."""
    rng = np.random.default_rng(int(seed))
    initial = rng.random((1, 9), dtype=np.float32)
    initial[:, 0] = 0
    phase = source_phase(f0, initial)
    noise = rng.standard_normal(phase.shape, dtype=np.float32)
    return np.ascontiguousarray(phase), np.ascontiguousarray(noise)
