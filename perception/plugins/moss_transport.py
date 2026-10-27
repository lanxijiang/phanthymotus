"""MOSS-only subscriber readiness and audio delivery policy.

The default ``paced`` mode bounds send-ahead for a live speaker. Explicit
``as_generated`` mode is for consumers that buffer and schedule playback, or
write the stream to a file; its EOF can arrive before playback would end.
"""

from __future__ import annotations

import json
import logging
import math
import time


log = logging.getLogger(__name__)


class SubscriberGate:
    """Request-local, polling-only gate for the first MOSS PCM publication.

    ``count`` returns the publisher's current matched-reader count; ``cancelled``
    reports stop/interrupt/request-abort. No background thread is created.
    The owner should poll once before launching synthesis, keep polling while
    collecting the first audio frames, then call ``wait`` immediately before
    publishing the first frame. Construction alone does not observe a reader.

    ``started_at``, ``deadline``, ``matched_at`` and ``ready_at`` are absolute
    timestamps from the injected monotonic clock. The latter two are ``None``
    until observed, and reset when a poll observes zero matched readers.
    A ready, still-matched reader remains ready after the deadline: slow audio
    generation must not become a subscriber timeout. Losing that reader does
    not grant another timeout budget. Cancellation is terminal for this gate.

    Matching is sampled DDS evidence, not an acknowledgement that a particular
    application consumed a frame. End-to-end PCM checks are still required.
    """

    POLL_INTERVAL_S = 0.02

    def __init__(self, count, cancelled, *, timeout_s=5.0, settle_s=0.1,
                 clock=None, sleep=None):
        self.timeout_s = float(timeout_s)
        self.settle_s = float(settle_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("subscriber timeout must be finite and positive")
        if (not math.isfinite(self.settle_s) or self.settle_s < 0
                or self.settle_s > self.timeout_s):
            raise ValueError("subscriber settle time must be finite and within the timeout")
        self._count = count
        self._cancelled = cancelled
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        self.started_at = self._clock()
        self.deadline = self.started_at + self.timeout_s
        self.matched_at = None
        self.ready_at = None
        self.first_poll_at = None
        self.first_poll_count = None
        self.match_resets = 0
        self.last_count = 0
        self._cancelled_seen = False

    def _timeout(self):
        return TimeoutError(
            "No stable matched MOSS audio subscriber within "
            f"{self.timeout_s:.3f}s (settle={self.settle_s:.3f}s, "
            f"last_count={self.last_count})"
        )

    def poll(self) -> bool:
        """Observe the reader once; return ready/False, or raise on timeout.

        A cancelled request returns False without touching the ROS publisher,
        which may already be on its way to destruction. Publisher query errors
        propagate to the request owner rather than pretending a reader exists.
        """
        if self._cancelled_seen or self._cancelled():
            self._cancelled_seen = True
            return False

        if self.first_poll_at is None:
            self.first_poll_at = self._clock()
        self.last_count = int(self._count())
        if self.first_poll_count is None:
            self.first_poll_count = self.last_count
        now = self._clock()
        if self.last_count <= 0:
            if self.matched_at is not None:
                self.match_resets += 1
            self.matched_at = None
            self.ready_at = None
        else:
            if self.ready_at is not None:
                return True
            # A newly observed reader after the deadline cannot retroactively
            # satisfy readiness, even with a zero settle interval.
            if now > self.deadline:
                raise self._timeout()
            if self.matched_at is None:
                self.matched_at = now
            if now - self.matched_at >= self.settle_s - 1e-9:
                self.ready_at = now
                return True

        if now >= self.deadline:
            raise self._timeout()
        return False

    def wait(self) -> bool:
        """Wait within the original budget; cancellation returns False."""
        while True:
            if self.poll():
                return True
            if self._cancelled_seen:
                return False
            now = self._clock()
            remaining = self.deadline - now
            if self.matched_at is not None:
                remaining = min(remaining, self.matched_at + self.settle_s - now)
            # The next poll handles the boundary; never sleep a negative delay.
            self._sleep(max(0.0, min(self.POLL_INTERVAL_S, remaining)))


class MossAudioTransport:
    """Wait for a reader on each utterance, then deliver its actual PCM.

    The pacer's playback cursor belongs to this transport instance rather
    than an utterance. In paced mode it persists across consecutive requests
    on the same output topic, including requests separated by EOF.
    """

    def __init__(self, mode="paced", max_ahead_s=1.0, subscriber_count=None,
                 settle_s=0.1, *, timeout_s=5.0, clock=None, sleep=None):
        if mode not in {"paced", "as_generated"}:
            raise ValueError("MOSS transport mode must be paced or as_generated")
        if not callable(subscriber_count):
            raise TypeError("subscriber_count must be callable")
        from plugins.moss_pacing import BoundedAudioPacer

        self.mode = mode
        self.subscriber_count = subscriber_count
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self._clock = clock
        self._sleep = sleep
        self._pacer = BoundedAudioPacer(max_ahead_s, clock=clock, sleep=sleep)
        self.gate: SubscriberGate | None = None
        self._utterance_number = 0
        self._first_publish_logged = False
        self._gate_wait_s = 0.0

    def begin_utterance(self, cancelled):
        """Start a fresh subscriber budget before launching synthesis."""
        self._utterance_number += 1
        self._first_publish_logged = False
        self._gate_wait_s = 0.0
        self.gate = SubscriberGate(
            self.subscriber_count, cancelled, timeout_s=self.timeout_s,
            settle_s=self.settle_s, clock=self._clock, sleep=self._sleep,
        )
        self.gate.poll()
        return self.gate

    def poll_ready(self) -> bool:
        """Sample DDS without sleeping while the model produces first audio."""
        if self.gate is None:
            raise RuntimeError("begin_utterance must precede subscriber polling")
        return self.gate.poll()

    def wait_ready(self) -> bool:
        """Recheck DDS immediately before the first PCM frame."""
        if self.gate is None:
            raise RuntimeError("begin_utterance must precede subscriber waiting")
        clock = self._clock or time.monotonic
        before = clock()
        try:
            return self.gate.wait()
        finally:
            self._gate_wait_s += clock() - before

    def _record_first_publish(self, frame):
        # This is the local ROS publish call, not an acknowledgement from the
        # subscriber or player. One record per utterance bounds log volume.
        if self.gate is None or self._first_publish_logged:
            return
        self._first_publish_logged = True
        gate = self.gate
        now = (self._clock or time.monotonic)()
        log.info("[moss] first_publish %s", json.dumps({
            "transport_id": hex(id(self)), "utterance": self._utterance_number,
            "mode": self.mode, "started_monotonic_s": gate.started_at,
            "matched_s": None if gate.matched_at is None else gate.matched_at - gate.started_at,
            "ready_s": None if gate.ready_at is None else gate.ready_at - gate.started_at,
            "blocking_gate_wait_s": self._gate_wait_s,
            "first_publish_s": now - gate.started_at,
            "first_poll_count": gate.first_poll_count, "last_count": gate.last_count,
            "match_resets": gate.match_resets, "frame_bytes": len(frame),
        }, sort_keys=True))

    def publish_frame(self, frame, publish, cancelled) -> bool:
        """Publish real PCM with the configured delivery policy.

        The caller must invoke ``wait_ready`` before the first frame of each
        utterance. The same pacer object handles every utterance in paced mode.
        """
        send = publish
        if not self._first_publish_logged:
            def send(first_frame):
                publish(first_frame)
                self._record_first_publish(first_frame)
        if self.mode == "paced":
            return self._pacer.publish_frame(frame, send, cancelled)
        if not frame or len(frame) % 2 or len(frame) > 3200:
            raise ValueError("Expected 1..1600 mono PCM16 samples")
        if cancelled():
            return False
        send(frame)
        return True
