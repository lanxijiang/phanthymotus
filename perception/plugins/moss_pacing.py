"""Bound the amount of PCM sent ahead of a real-time audio consumer."""

from __future__ import annotations

import math
import time


class BoundedAudioPacer:
    """Keep one playback cursor across utterances on the same output topic.

    PCM remains 16 kHz mono S16LE; the bound limits queued audio, not playback
    speed. After the initial bounded burst, a full frame requires 100 ms of
    wall time to replenish its credit. EOF does not empty a speaker's queue,
    so the cursor must not be reset at an utterance boundary or interruption.
    """

    def __init__(self, max_ahead_s=1.0, *, clock=None, sleep=None):
        self.max_ahead_s = float(max_ahead_s)
        if not math.isfinite(self.max_ahead_s) or self.max_ahead_s < 0.1:
            raise ValueError("PCM send-ahead budget must be finite and at least 100 ms")
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self._audio_end_s = None

    def publish_frame(self, frame, publish, cancelled) -> bool:
        """Publish unmodified PCM when its full duration fits the budget.

        Return False if interrupted while waiting, without charging or emitting
        that frame. The short final frame consumes its actual sample duration.
        """
        if not frame or len(frame) % 2 or len(frame) > 3200:
            raise ValueError("Expected 1..1600 mono PCM16 samples")
        duration = len(frame) / 32000.0
        while True:
            if cancelled():
                return False
            now = self._clock()
            end = max(now, self._audio_end_s if self._audio_end_s is not None else now)
            delay = end + duration - self.max_ahead_s - now
            if delay <= 1e-9:
                break
            # A long queued utterance must not make stop/interrupt unresponsive.
            self._sleep(min(delay, 0.02))
        publish(frame)
        # Count only successfully published audio. A slow publish is conservative:
        # anchor to its return time rather than allowing an extra catch-up burst.
        self._audio_end_s = max(end, self._clock()) + duration
        return True
