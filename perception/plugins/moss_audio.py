"""Stateful conversion from MOSS's 48 kHz stereo stream to platform PCM.

The filter history and decimation phase survive codec callback boundaries. An
independent converter is created for every utterance, so concurrent requests
cannot change one another's audio state.
"""

from __future__ import annotations

import numpy as np


SOURCE_RATE = 48000
OUTPUT_RATE = 16000
DECIMATION = SOURCE_RATE // OUTPUT_RATE
TAPS = 97


def _lowpass() -> np.ndarray:
    positions = np.arange(TAPS, dtype=np.float64) - (TAPS - 1) / 2
    cutoff = OUTPUT_RATE / (2 * SOURCE_RATE)
    coefficients = 2 * cutoff * np.sinc(2 * cutoff * positions)
    coefficients *= np.blackman(TAPS)
    coefficients /= coefficients.sum()
    return coefficients.astype(np.float32)


_FILTER = _lowpass()


class MossPcm16Converter:
    """Convert arbitrary sized [samples, 2] float32 callbacks continuously."""

    def __init__(self) -> None:
        self._history = np.zeros(TAPS - 1, dtype=np.float32)
        self._input_samples = 0

    def process(self, stereo: np.ndarray) -> bytes:
        audio = np.asarray(stereo, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError(f"MOSS audio must be [samples, 2], got {audio.shape}")
        if not np.all(np.isfinite(audio)):
            raise ValueError("MOSS audio contains a non-finite sample")
        if len(audio) == 0:
            return b""

        mono = audio.mean(axis=1, dtype=np.float32)
        extended = np.concatenate((self._history, mono))
        # `valid` yields one causal filtered value per new input sample.
        filtered = np.convolve(extended, _FILTER, mode="valid")
        positions = np.arange(len(audio), dtype=np.int64) + self._input_samples
        selected = filtered[positions % DECIMATION == 0]
        self._history = extended[-(TAPS - 1):].copy()
        self._input_samples += len(audio)
        return np.rint(np.clip(selected, -1.0, 1.0) * 32767).astype("<i2").tobytes()
