"""Exact VITS2 duration expansion and iSTFT regressions; no CUDA is loaded."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.runtime.backends import trt_numpy_tts_engine as backend


def dense_expand(duration, x_mask, y_mask, m_p, logs_p):
    """The submitted fc0c2df implementation, used as the reference."""
    attention = backend._generate_path(
        duration, x_mask[:, :, None, :] * y_mask[:, :, :, None]
    )
    return tuple(
        np.matmul(attention[:, 0], prior.transpose(0, 2, 1)).transpose(0, 2, 1)
        for prior in (m_p, logs_p)
    )


def example(counts, channels=192, frames=None):
    duration = np.asarray(counts, dtype=np.float32)[None, None, :]
    if frames is None:
        frames = max(int(duration.sum()), 1)
    rng = np.random.default_rng(831)
    shape = (1, channels, len(counts))
    return [duration, np.ones_like(duration), np.ones((1, 1, frames), np.float32),
            rng.standard_normal(shape).astype(np.float32),
            rng.standard_normal(shape).astype(np.float32)]


class TestPriorExpansion(unittest.TestCase):
    def assert_matches_dense(self, values, *, fast):
        with np.errstate(invalid="ignore", over="ignore"):
            expected = dense_expand(*values)
            with patch.object(backend, "_generate_path", wraps=backend._generate_path) as path:
                actual = backend._expand_priors(*values)
        self.assertEqual(path.call_count, 0 if fast else 1)
        for wanted, got in zip(expected, actual):
            self.assertEqual(got.shape, wanted.shape)
            self.assertEqual(got.dtype, wanted.dtype)
            self.assertTrue(np.array_equal(wanted, got, equal_nan=True))
            if fast:
                self.assertEqual(got.tobytes(), wanted.tobytes())
        return actual

    def test_real_profile_shapes_and_zero_duration_tokens(self):
        rng = np.random.default_rng(57)
        for sources, frames in ((1, 1), (7, 23), (127, 800), (256, 2048), (512, 2048)):
            with self.subTest(sources=sources, frames=frames):
                counts = rng.multinomial(frames, np.full(sources, 1 / sources))
                actual = self.assert_matches_dense(example(counts), fast=True)
                self.assertEqual(actual[0].shape, (1, 192, frames))
        self.assert_matches_dense(example([0, 2, 0, 0, 3, 0]), fast=True)

    def test_padding_and_interior_binary_mask(self):
        values = example([2, 0, 3, 0, 0])
        values[1][0, 0, [1, 3, 4]] = 0
        self.assert_matches_dense(values, fast=True)

    def test_all_zero_duration_keeps_one_zero_frame(self):
        for mask in (0, 1):
            values = example([0, 0, 0])
            values[1].fill(mask)
            expanded = self.assert_matches_dense(values, fast=True)
            for prior in expanded:
                self.assertEqual(prior.shape, (1, 192, 1))
                self.assertFalse(np.any(prior))

    def test_signed_zero_and_noncontiguous_priors(self):
        values = example([1, 0, 2, 3])
        for index in (3, 4):
            values[index] = values[index][:, ::-1, :]
            values[index][0, 2, 0] = -0.0
        self.assert_matches_dense(values, fast=True)

    def test_nonbinary_and_nonfinite_source_masks_use_dense(self):
        for mask in (-1.0, 0.5, 2.0, np.nan, np.inf):
            with self.subTest(mask=mask):
                values = example([1, 2, 3])
                values[1][0, 0, 1] = mask
                self.assert_matches_dense(values, fast=False)

    def test_target_mask_and_masked_nonzero_duration_use_dense(self):
        values = example([2, 0, 3])
        values[2][0, 0, 2] = 0
        self.assert_matches_dense(values, fast=False)
        values = example([2, 0, 3])
        values[1][0, 0, 0] = 0
        self.assert_matches_dense(values, fast=False)

    def test_noninteger_negative_and_nonfinite_durations_use_dense(self):
        for duration in (0.5, -1, np.nan, np.inf):
            with self.subTest(duration=duration):
                values = example([1, 2, 3])
                values[0][0, 0, 1] = duration
                self.assert_matches_dense(values, fast=False)

    def test_truncated_or_extended_target_and_float64_use_dense(self):
        for frames in (4, 8):
            self.assert_matches_dense(example([1, 2, 3], frames=frames), fast=False)
        values = [value.astype(np.float64) for value in example([1, 2, 3])]
        self.assert_matches_dense(values, fast=False)

    def test_other_batches_use_dense(self):
        values = [np.repeat(value, 2, axis=0) for value in example([1, 2, 3])]
        values[2][1, 0, -1] = 0
        self.assert_matches_dense(values, fast=False)

    def test_nonfinite_and_subnormal_priors_preserve_dense_behavior(self):
        for index in (3, 4):
            for prior in (np.nan, np.inf, -np.inf, np.finfo(np.float32).smallest_subnormal):
                with self.subTest(prior=index, value=prior):
                    values = example([2, 0, 3])
                    # Even an unselected source affects the dense NaN result.
                    values[index][0, 0, 1] = prior
                    self.assert_matches_dense(values, fast=False)


class TestInferenceRandomState(unittest.TestCase):
    def test_same_latents_pcm_and_random_draw_order_across_requests(self):
        rng = np.random.default_rng(93)
        shape = (1, 192, 5)
        encoder_outputs = {
            "m_p": rng.normal(size=shape).astype(np.float32),
            "logs_p": rng.normal(size=shape).astype(np.float32),
            "x_mask": np.asarray([[[1, 0, 1, 1, 0]]], dtype=np.float32),
            "logw": np.asarray([[[0, 0, 1, 0, 0]]], dtype=np.float32),
            "g": np.zeros((1, 256, 1), dtype=np.float32),
        }

        class FakeEncoder:
            def run(self, _inputs):
                return encoder_outputs

        class FakeFlow:
            def __init__(self):
                self.calls = []

            def run(self, inputs):
                self.calls.append({key: value.copy() for key, value in inputs.items()})
                return {"z": inputs["z_p"]}

        class RecordingRng:
            def __init__(self):
                self.rng = np.random.default_rng(42)
                self.shapes = []

            def standard_normal(self, shape):
                self.shapes.append(shape)
                return self.rng.standard_normal(shape)

        def make_engine():
            engine = backend.TensorRTNumpyTTSEngine.__new__(backend.TensorRTNumpyTTSEngine)
            engine.max_text_tokens = 512
            engine.max_frames = 2048
            engine.encoder = FakeEncoder()
            engine.flow = FakeFlow()
            engine._rng = RecordingRng()
            engine.n_fft, engine.istft_hop, engine.sr = 4, 1, 16000
            engine._decode = lambda z, g: z[:, :6, :]
            return engine

        original, optimized = make_engine(), make_engine()
        ids = ((1, 2, 3, 4, 5), (0,) * 5, (2,) * 5)
        with patch.object(backend, "_expand_priors", side_effect=dense_expand), \
                patch.object(backend, "_istft", side_effect=frame_loop_istft):
            expected = [original._infer_ids(ids) for _ in range(2)]
        actual = [optimized._infer_ids(ids) for _ in range(2)]
        self.assertEqual(actual, expected)
        self.assertEqual(optimized._rng.shapes, original._rng.shapes)
        self.assertEqual(len(optimized._rng.shapes), 2)
        for left, right in zip(original.flow.calls, optimized.flow.calls):
            for name in left:
                self.assertEqual(left[name].dtype, right[name].dtype)
                self.assertEqual(left[name].tobytes(), right[name].tobytes())
        np.testing.assert_array_equal(original._rng.rng.random(32), optimized._rng.rng.random(32))


def frame_loop_istft(spectrum, n_fft, hop_length):
    """The submitted fc0c2df overlap-add implementation."""
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1).astype(np.float32)
    frame_count = frames.shape[-1]
    output_length = n_fft + hop_length * (frame_count - 1)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    audio = np.zeros((frames.shape[0], output_length), dtype=np.float32)
    envelope = np.zeros(output_length, dtype=np.float32)
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        audio[:, start:start + n_fft] += frames[:, :, frame_index] * window
        envelope[start:start + n_fft] += window * window
    valid = envelope > 1e-11
    audio[:, valid] /= envelope[valid]
    trim = n_fft // 2
    return audio[:, trim:-trim] if trim else audio


class TestIstftOverlapAdd(unittest.TestCase):
    def assert_matches_frame_loop(self, spectrum, n_fft, hop):
        expected = frame_loop_istft(spectrum, n_fft, hop)
        actual = backend._istft(spectrum, n_fft, hop)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.array_equal(actual, expected, equal_nan=True))
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_random_complex_multiple_shapes_and_hops(self):
        rng = np.random.default_rng(847)
        for n_fft in (1, 7, 8, 16, 32):
            for hop in sorted({1, 4, n_fft, n_fft + 3}):
                for frames in (1, 2, 37):
                    for batch in (1, 3):
                        with self.subTest(n_fft=n_fft, hop=hop, frames=frames, batch=batch):
                            shape = (batch, n_fft // 2 + 1, frames)
                            spectrum = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
                            self.assert_matches_frame_loop(spectrum, n_fft, hop)

    def test_release_shape_long_audio_and_complex128(self):
        rng = np.random.default_rng(847)
        # 20.48 seconds at the release's FFT=16, hop=4, 16 kHz settings.
        shape = (1, 9, 81921)
        spectrum = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        self.assert_matches_frame_loop(spectrum, 16, 4)

    def test_empty_frames_zero_spectrum_signed_zero_and_zero_hop(self):
        for count in (0, 1, 2, 11):
            spectrum = np.zeros((2, 9, count), dtype=np.complex64)
            spectrum.real[:, :, 1::2] = -0.0
            spectrum.imag[:, :, ::2] = -0.0
            self.assert_matches_frame_loop(spectrum, 16, 4)
        self.assert_matches_frame_loop(np.ones((1, 9, 11), dtype=np.complex64), 16, 0)

    def test_large_and_small_magnitudes_preserve_rounding(self):
        rng = np.random.default_rng(841)
        shape = (2, 9, 127)
        spectrum = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        spectrum[:, :, ::2] *= 1e10
        spectrum[:, :, 1::2] *= 1e-10
        self.assert_matches_frame_loop(spectrum, 16, 4)


if __name__ == "__main__":
    unittest.main()
