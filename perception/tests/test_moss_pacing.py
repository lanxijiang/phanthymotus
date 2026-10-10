"""Bounded transport must preserve samples and not accrue lead across speech."""

from types import SimpleNamespace

import pytest

from plugins.moss_pacing import BoundedAudioPacer
from plugins.moss_tts import MossTTSAdapter
import plugins.tts as tts


class Clock:
    def __init__(self, now=0.0):
        self.now = now
        self.waits = []
        self.on_sleep = lambda: None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds
        self.on_sleep()


def make_pacer(clock):
    return BoundedAudioPacer(MossTTSAdapter.transport_lead_seconds,
                             clock=clock.time, sleep=clock.sleep)


def test_long_fast_producer_has_only_ten_frame_burst_and_realtime_steady_rate():
    clock = Clock(.56)
    pacer = make_pacer(clock)
    sent = []
    frame = bytes(range(256)) * 12 + bytes(range(128))
    for _ in range(1000):
        assert pacer.publish_frame(frame, lambda data: sent.append((clock.now, data)), lambda: False)
    assert len(frame) == 3200
    assert [stamp for stamp, _ in sent[:10]] == [.56] * 10
    assert sent[10][0] == pytest.approx(.66)
    assert sent[-1][0] == pytest.approx(.56 + 100 - 1.0)
    assert all(data is frame for _, data in sent)  # No copying/reformatting/resampling.
    for index, (stamp, _) in enumerate(sent):
        assert .56 + (index + 1) * .1 - stamp <= 1.0 + 1e-8
    assert max(clock.waits) <= .02


def test_partial_frames_and_consecutive_utterances_share_the_same_credit():
    clock, sent = Clock(), []
    pacer = make_pacer(clock)
    # Two full utterances: no cursor reset at EOF, including their remainders.
    frames = [[b'\x01\x02' * 1600] * 10 + [b'\x03\x04' * 640],
              [b'\x05\x06' * 1600] * 4 + [b'\x07\x08' * 320]]
    for utterance in frames:
        for frame in utterance:
            assert pacer.publish_frame(frame, lambda pcm: sent.append((clock.now, pcm)), lambda: False)
    assert sent[10][0] == pytest.approx(.04)  # 40 ms, not rounded to 100 ms.
    assert sent[11][0] == pytest.approx(.14)  # Next utterance cannot get a new burst.
    assert sent[-1][0] == pytest.approx(.46)
    assert b''.join(pcm for _, pcm in sent) == b''.join(sum(frames, []))


def test_stalled_source_refills_only_one_bounded_window():
    clock = Clock()
    pacer = make_pacer(clock)
    for _ in range(10):
        assert pacer.publish_frame(b'\x01\x00' * 1600, lambda pcm: None, lambda: False)
    clock.now = 30.0
    times = []
    for _ in range(11):
        assert pacer.publish_frame(b'\x01\x00' * 1600, lambda pcm: times.append(clock.now), lambda: False)
    assert times[:10] == [30.0] * 10
    assert times[10] == pytest.approx(30.1)


def test_cancel_wait_does_not_send_or_charge_frame_and_keeps_already_sent_tail():
    clock, sent = Clock(), []
    pacer = make_pacer(clock)
    stopped = False
    for _ in range(10):
        assert pacer.publish_frame(b'\x01\x00' * 1600, sent.append, lambda: stopped)

    def cancel():
        nonlocal stopped
        stopped = True

    clock.on_sleep = cancel
    assert not pacer.publish_frame(b'\x02\x00' * 1600, sent.append, lambda: stopped)
    assert clock.now == .02 and len(sent) == 10
    stopped = False
    clock.on_sleep = lambda: None
    assert pacer.publish_frame(b'\x03\x00' * 640, sent.append, lambda: stopped)
    assert clock.now == pytest.approx(.04)
    assert len(sent) == 11


def test_failed_publish_does_not_consume_credit():
    clock, sent = Clock(), []
    pacer = make_pacer(clock)

    def fail(frame):
        raise RuntimeError("publisher stopped")

    with pytest.raises(RuntimeError, match="publisher stopped"):
        pacer.publish_frame(b'\x01\x00' * 1600, fail, lambda: False)
    for _ in range(10):
        assert pacer.publish_frame(b'\x01\x00' * 1600, sent.append, lambda: False)
    assert clock.now == 0 and len(sent) == 10


def test_only_capable_adapter_gets_bounded_transport():
    moss = tts._TTSNode('/moss', SimpleNamespace(
        transport_lead_seconds=MossTTSAdapter.transport_lead_seconds))
    legacy = tts._TTSNode('/legacy', SimpleNamespace())
    assert moss._audio_pacer.max_ahead_s == 1.0
    assert legacy._audio_pacer is None


def test_actual_worker_keeps_pcm_eof_and_credit_across_utterances(monkeypatch):
    clock = Clock(.56)
    payloads = {
        "First.": bytes(range(256)) * 155,  # 39680 bytes / 1.24 s, short last frame.
        "Second.": bytes(reversed(range(256))) * 105,  # .84 s.
    }

    class Adapter:
        transport_lead_seconds = MossTTSAdapter.transport_lead_seconds

        def synthesize_stream(self, text):
            pcm = payloads[text]
            for offset in range(0, len(pcm), 1378):
                yield pcm[offset:offset + 1378]

    class InlineThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()  # Both test utterances fit in the 200-frame queue.

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(tts.threading, 'Thread', InlineThread)
    monkeypatch.setattr(tts, 'fire_hook', lambda name: None)
    node = tts._TTSNode('/actual-worker', Adapter())
    node._audio_pacer = make_pacer(clock)
    node.state = 'running'
    node.enqueue("First.")
    node.enqueue("Second.")
    messages, eof_count = [], 0

    def publish(message):
        nonlocal eof_count
        pcm = bytes(message.data)
        messages.append((clock.now, pcm, message.format))
        if pcm == tts.AUDIO_EOF_MAGIC:
            eof_count += 1
            if eof_count == 2:
                node._stop_event.set()

    monkeypatch.setattr(node._pub, 'publish', publish)
    node._worker()
    utterances, pending = [], b''
    for timestamp, pcm, fmt in messages:
        assert fmt == 'audio/pcm-16k'
        if pcm == tts.AUDIO_EOF_MAGIC:
            utterances.append(pending)
            pending = b''
        else:
            pending += pcm
    assert utterances == list(payloads.values()) and not pending
    assert eof_count == 2
    assert clock.now == pytest.approx(.56 + 2.08 - 1.0)
    assert node._audio_pacer._audio_end_s - clock.now == pytest.approx(1.0)
    assert [len(pcm) for _, pcm, _ in messages if pcm != tts.AUDIO_EOF_MAGIC][-1] == 1280
