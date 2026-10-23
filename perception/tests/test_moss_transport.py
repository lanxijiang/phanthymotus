"""DDS first-frame readiness checks without ROS, GPU, threads or real sleeps."""

import math

import pytest

from plugins.moss_transport import MossAudioTransport, SubscriberGate


class Clock:
    def __init__(self, now=10.0):
        self.now = now
        self.waits = []
        self.on_sleep = lambda: None

    def time(self):
        return self.now

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay
        self.on_sleep()


def make_gate(clock, count=lambda: 1, cancelled=lambda: False, **kwargs):
    return SubscriberGate(count, cancelled, clock=clock.time, sleep=clock.sleep,
                          **kwargs)


def test_stable_reader_records_absolute_times_and_waits_at_most_twenty_ms():
    clock = Clock()
    gate = make_gate(clock)
    assert gate.started_at == 10.0 and gate.deadline == 15.0
    assert gate.matched_at is None and gate.ready_at is None
    assert gate.wait()
    assert gate.matched_at == 10.0
    assert gate.ready_at == pytest.approx(10.1)
    assert clock.now == pytest.approx(10.1)
    assert gate.last_count == 1
    assert gate.first_poll_at == 10.0 and gate.first_poll_count == 1
    assert gate.match_resets == 0
    assert all(0 < delay <= .02 for delay in clock.waits)


def test_early_real_match_overlaps_half_second_settle_with_preparation():
    clock = Clock()
    gate = make_gate(clock, settle_s=.5)
    assert not gate.poll()
    # Model/frontend preparation occupies the calling thread before its next
    # chance to poll. The actual earlier observation starts the interval.
    clock.now += .6
    assert gate.wait()
    assert gate.matched_at == 10.0
    assert gate.first_poll_at == 10.0 and gate.first_poll_count == 1
    assert gate.ready_at == pytest.approx(10.6)
    assert clock.waits == []


def test_early_poll_without_reader_does_not_backdate_later_match():
    clock = Clock()
    count = [0]
    gate = make_gate(clock, count=lambda: count[0], settle_s=.5)
    assert not gate.poll()
    clock.now += .6
    count[0] = 1
    assert not gate.poll()
    assert gate.matched_at == pytest.approx(10.6)
    assert gate.wait()
    assert gate.ready_at == pytest.approx(11.1)
    assert sum(clock.waits) == pytest.approx(.5)
    assert gate.first_poll_at == 10.0 and gate.first_poll_count == 0
    assert gate.match_resets == 0


def test_late_reader_waits_its_full_settle_period():
    clock = Clock()
    gate = make_gate(clock, count=lambda: int(clock.now >= 11.2))
    assert gate.wait()
    assert 11.2 <= gate.matched_at < 11.221
    assert gate.ready_at - gate.matched_at == pytest.approx(.1)
    assert gate.deadline == 15.0


def test_transient_match_resets_both_timestamps():
    clock = Clock()
    count = [1]
    gate = make_gate(clock, count=lambda: count[0])
    assert not gate.poll()
    clock.now += .08
    count[0] = 0
    assert not gate.poll()
    assert gate.matched_at is None and gate.ready_at is None
    assert gate.match_resets == 1
    assert not gate.poll()  # Staying unmatched is not another reset.
    assert gate.match_resets == 1
    clock.now += .01
    count[0] = 1
    assert not gate.poll()
    assert gate.matched_at == pytest.approx(10.09)
    clock.now += .09
    assert not gate.poll()
    clock.now += .01
    assert gate.poll()
    assert gate.ready_at == pytest.approx(10.19)
    assert gate.first_poll_at == 10.0 and gate.first_poll_count == 1
    assert gate.match_resets == 1


def test_ready_reader_survives_slow_generation_but_is_rechecked_before_emit():
    clock = Clock()
    count = [1]
    gate = make_gate(clock, count=lambda: count[0])
    assert gate.wait()
    ready_at = gate.ready_at
    clock.now += 20
    assert gate.poll()
    assert gate.ready_at == ready_at
    # Revalidation must not accept a reader that disappeared after readiness.
    count[0] = 0
    with pytest.raises(TimeoutError, match="MOSS audio subscriber"):
        gate.poll()
    assert gate.matched_at is None and gate.ready_at is None


def test_ready_then_disconnected_reader_resettles_without_extending_deadline():
    clock = Clock()
    count = [1]
    gate = make_gate(clock, count=lambda: count[0])
    assert gate.wait()
    count[0] = 0
    clock.now = 14.8
    assert not gate.poll()
    count[0] = 1
    clock.now = 14.95
    assert not gate.poll()
    with pytest.raises(TimeoutError):
        gate.wait()
    assert clock.now == pytest.approx(15.0)
    assert gate.ready_at is None
    assert gate.deadline == 15.0


def test_missing_reader_times_out_and_gate_has_no_publication_side_effect():
    clock = Clock()
    sent = []
    gate = make_gate(clock, count=lambda: 0, timeout_s=.12)
    with pytest.raises(TimeoutError, match="last_count=0"):
        if gate.wait():
            sent.append(b"real pcm")
    assert sent == []
    assert clock.now == pytest.approx(10.12)
    assert gate.matched_at is None and gate.ready_at is None


def test_settling_at_deadline_succeeds_but_new_reader_after_deadline_cannot():
    clock = Clock()
    gate = make_gate(clock, timeout_s=.1)
    assert not gate.poll()
    clock.now = gate.deadline
    assert gate.poll()
    clock = Clock()
    gate = make_gate(clock, timeout_s=.1, settle_s=0)
    clock.now = gate.deadline + .001
    with pytest.raises(TimeoutError):
        gate.poll()


def test_zero_settle_requires_a_match_and_accepts_it_without_sleep():
    clock = Clock()
    count = [0]
    gate = make_gate(clock, count=lambda: count[0], settle_s=0)
    assert not gate.poll()
    count[0] = 2
    assert gate.wait()
    assert gate.last_count == 2
    assert gate.matched_at == gate.ready_at == 10.0
    assert clock.waits == []


def test_cancel_before_poll_avoids_querying_destroyed_publisher_and_is_terminal():
    clock = Clock()
    stopped = [True]

    def destroyed():
        raise AssertionError("must not touch destroyed publisher")

    gate = make_gate(clock, count=destroyed, cancelled=lambda: stopped[0])
    assert not gate.poll()
    assert not gate.wait()
    stopped[0] = False
    assert not gate.poll()
    assert clock.waits == []
    assert gate.first_poll_at is None and gate.first_poll_count is None
    assert gate.match_resets == 0


def test_cancel_during_wait_returns_without_publishing_or_waiting_for_timeout():
    clock = Clock()
    stopped = [False]
    sent = []
    gate = make_gate(clock, count=lambda: 0, cancelled=lambda: stopped[0])
    clock.on_sleep = lambda: stopped.__setitem__(0, True)
    if gate.wait():
        sent.append(b"real pcm")
    assert sent == []
    assert clock.now == pytest.approx(10.02)
    assert clock.waits == [.02]


def test_publisher_query_failure_is_not_a_synthetic_ready_signal():
    clock = Clock()

    def failed():
        raise RuntimeError("publisher destroyed")

    gate = make_gate(clock, count=failed)
    with pytest.raises(RuntimeError, match="publisher destroyed"):
        gate.wait()
    assert gate.ready_at is None and clock.waits == []
    assert gate.first_poll_at == 10.0 and gate.first_poll_count is None


@pytest.mark.parametrize("options", [
    {"timeout_s": 0}, {"timeout_s": -1}, {"timeout_s": math.inf},
    {"timeout_s": math.nan}, {"settle_s": -1}, {"settle_s": math.inf},
    {"settle_s": math.nan}, {"timeout_s": .05, "settle_s": .1},
])
def test_invalid_wait_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        make_gate(Clock(), **options)


def make_transport(clock, *, mode="paced", count=lambda: 1, settle_s=.1,
                   max_ahead_s=1.0, timeout_s=5.0):
    return MossAudioTransport(
        mode=mode, max_ahead_s=max_ahead_s, subscriber_count=count,
        settle_s=settle_s, timeout_s=timeout_s,
        clock=clock.time, sleep=clock.sleep,
    )


def test_default_mode_is_paced_and_new_utterance_gets_a_fresh_gate():
    clock = Clock()
    transport = make_transport(clock)
    assert transport.mode == "paced"
    first = transport.begin_utterance(lambda: False)
    assert first.first_poll_count == 1 and first.matched_at == 10.0
    assert not transport.poll_ready()
    clock.now += .1  # Synthesis overlaps subscriber settling.
    assert transport.wait_ready()
    second = transport.begin_utterance(lambda: False)
    assert second is not first
    assert second.started_at == pytest.approx(10.1)
    assert second.first_poll_count == 1


def test_transport_gate_preserves_original_timeout_and_cancellation():
    clock = Clock()
    count = [0]
    transport = make_transport(clock, count=lambda: count[0], timeout_s=.12)
    transport.begin_utterance(lambda: False)
    clock.now += .06
    count[0] = 1
    assert not transport.poll_ready()
    with pytest.raises(TimeoutError, match="MOSS audio subscriber"):
        transport.wait_ready()
    assert clock.now == pytest.approx(10.12)

    stopped = [True]
    def destroyed():
        raise AssertionError("cancelled gate queried a destroyed publisher")
    transport = make_transport(clock, count=destroyed)
    transport.begin_utterance(lambda: stopped[0])
    assert not transport.wait_ready()


def test_paced_transport_keeps_send_ahead_cursor_across_utterances():
    clock = Clock()
    transport = make_transport(clock, settle_s=0)
    transport.begin_utterance(lambda: False)
    assert transport.wait_ready()
    frame = bytes([1, 0]) * 1600
    sent = []
    for _ in range(10):
        assert transport.publish_frame(frame, sent.append, lambda: False)
    assert len(sent) == 10 and clock.now == 10.0
    # A new utterance/EOF does not grant another 1 s advance budget.
    transport.begin_utterance(lambda: False)
    assert transport.wait_ready()
    assert transport.publish_frame(frame, sent.append, lambda: False)
    assert len(sent) == 11 and clock.now == pytest.approx(10.1)
    assert all(part == frame for part in sent)


def test_as_generated_delivers_exact_pcm_without_wall_clock_pacing():
    clock = Clock()
    transport = make_transport(clock, mode="as_generated", settle_s=0)
    transport.begin_utterance(lambda: False)
    assert transport.wait_ready()
    full = bytes([8, 0]) * 1600
    tail = bytes([3, 0]) * 200
    sent = []
    for frame in [full] * 20 + [tail]:
        assert transport.publish_frame(frame, sent.append, lambda: False)
    assert sent == [full] * 20 + [tail]
    assert b"".join(sent) == full * 20 + tail
    assert clock.now == 10.0 and clock.waits == []


@pytest.mark.parametrize("mode", ["paced", "as_generated"])
def test_both_modes_validate_pcm_and_honor_cancellation(mode):
    clock = Clock()
    transport = make_transport(clock, mode=mode, settle_s=0)
    transport.begin_utterance(lambda: False)
    assert transport.wait_ready()
    sent = []
    for bad in (b"", b"\x01", bytes(3202)):
        with pytest.raises(ValueError, match="PCM16"):
            transport.publish_frame(bad, sent.append, lambda: False)
    assert not transport.publish_frame(bytes(3200), sent.append, lambda: True)
    assert sent == []


def test_invalid_transport_mode_and_missing_reader_counter_are_rejected():
    with pytest.raises(ValueError, match="transport mode"):
        MossAudioTransport(mode="fast", subscriber_count=lambda: 1)
    with pytest.raises(TypeError, match="subscriber_count"):
        MossAudioTransport()
