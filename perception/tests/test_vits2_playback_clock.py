"""Bound PCM delivery across sentences/restarts; ROS/GPU are stubbed."""

from types import SimpleNamespace
import threading
import time

import pytest

from vision_stubs import _FakeExecutor
import plugins.vits2_tts_trt.plugin as vits2


PCM = b"\x01\x00" * 1600


class _Time:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now


class _StopEvent:
    def __init__(self, clock):
        self.clock, self.set_value, self.waits = clock, False, []
        self.on_wait = None

    def is_set(self):
        return self.set_value

    def set(self):
        self.set_value = True

    def clear(self):
        self.set_value = False

    def wait(self, seconds):
        self.waits.append(seconds)
        self.clock.now += seconds
        if self.on_wait:
            self.on_wait()
        return self.set_value


@pytest.fixture
def simulated_time(monkeypatch):
    clock = _Time()
    monkeypatch.setattr(vits2, "time", SimpleNamespace(monotonic=clock.monotonic))
    return clock


def _node(clock, shared=None):
    node = vits2._Vits2TTSNode("/test", object(), playback_clock=shared)
    node._stop_event = _StopEvent(clock)
    return node


@pytest.mark.parametrize("allow_fast, interval_ms", [(False, 0), (False, 100), (True, 100)])
def test_production_clock_requires_both_fast_opt_in_and_fast_interval(
    simulated_time, monkeypatch, allow_fast, interval_ms
):
    monkeypatch.setattr(vits2, "ALLOW_FAST_DELIVERY", allow_fast)
    monkeypatch.setattr(vits2, "FRAME_INTERVAL_MS", interval_ms)
    node = _node(simulated_time)
    initial = simulated_time.now
    for _ in range(20):
        node._publish_pcm_frame(PCM)
        assert node._playback_clock.tail_s - simulated_time.now <= 1.0 + 1e-9
    assert simulated_time.now - initial == pytest.approx(1.0)


def test_explicit_offline_fast_delivery_bypasses_playback_clock(simulated_time, monkeypatch):
    monkeypatch.setattr(vits2, "ALLOW_FAST_DELIVERY", True)
    monkeypatch.setattr(vits2, "FRAME_INTERVAL_MS", 0)
    node = _node(simulated_time)
    initial = simulated_time.now
    for _ in range(100):
        assert node._publish_pcm_frame(PCM) == initial
    assert simulated_time.now == initial
    assert node._stop_event.waits == []
    assert node._playback_clock.tail_s == 0
    assert len(node._pub.messages) == 100
    # Opting out of playback pacing never opts out of cancellation.
    node._stop_event.set()
    assert node._publish_pcm_frame(PCM) is None
    assert len(node._pub.messages) == 100


def test_one_hundred_consecutive_utterances_share_the_same_budget(simulated_time):
    node = _node(simulated_time)
    initial = simulated_time.now
    for _ in range(100):
        for _ in range(5):
            assert node._publish_pcm_frame(PCM) is not None
            assert node._playback_clock.tail_s - simulated_time.now <= 1.0 + 1e-9
        tail = node._playback_clock.tail_s
        node._publish_eof()
        assert node._playback_clock.tail_s == tail
    # 50 seconds of PCM cannot be delivered more than one second early.
    assert simulated_time.now - initial >= 49.0 - 1e-8
    messages = [bytes(value) for value in node._pub.messages]
    assert messages.count(PCM) == 500
    assert messages.count(vits2.AUDIO_EOF_MAGIC) == 100


def test_partial_final_frame_uses_its_actual_duration(simulated_time):
    node = _node(simulated_time)
    for _ in range(10):
        node._publish_pcm_frame(PCM)
    began = simulated_time.now
    partial = PCM[:1600]  # 50 ms, not 100 ms.
    node._publish_pcm_frame(partial)
    assert simulated_time.now - began == pytest.approx(.05)
    assert node._playback_clock.tail_s - simulated_time.now == pytest.approx(1.0)
    assert bytes(node._pub.messages[-1]) == partial


def test_idle_drain_allows_a_new_full_prebuffer(simulated_time):
    node = _node(simulated_time)
    for _ in range(10):
        node._publish_pcm_frame(PCM)
    simulated_time.now += 2
    began = simulated_time.now
    for _ in range(10):
        node._publish_pcm_frame(PCM)
    assert simulated_time.now == pytest.approx(began)


@pytest.mark.parametrize("event_name", ["_stop_event", "_interrupt_event"])
def test_pacing_wait_observes_cancellation_every_twenty_ms(simulated_time, event_name):
    node = _node(simulated_time)
    node._playback_clock.tail_s = simulated_time.now + 1
    node._stop_event.on_wait = getattr(node, event_name).set
    assert node._publish_pcm_frame(PCM) is None
    assert node._stop_event.waits == [.02]
    assert node._pub.messages == []
    # Interrupt is not an acknowledgement that the reader cleared its queue.
    assert node._playback_clock.tail_s == 11.0


class _Timer:
    created = []

    def __init__(self, interval, callback):
        self.interval, self.callback = interval, callback
        self.daemon = False
        self.created.append(self)

    def start(self):
        pass


@pytest.fixture
def resident_plugin(monkeypatch, simulated_time):
    _Timer.created = []
    monkeypatch.setattr(vits2.threading, "Timer", _Timer)

    def start_without_worker(node):
        node.state = "running"
        return node.status()

    monkeypatch.setattr(vits2._Vits2TTSNode, "start", start_without_worker)
    plugin = vits2.TTSPlugin({"backend": "trt"}, _FakeExecutor())
    plugin._adapter = object()
    plugin._adapter_state = "ready"
    return plugin


def _start(plugin, instance, topic="/test"):
    assert plugin.dispatch("tts", {"action": "start", "instance_id": instance,
                                   "input_topic": topic})["state"] == "running"
    return plugin._nodes[instance]


def test_stop_start_on_same_topic_preserves_the_tail(resident_plugin, simulated_time):
    plugin = resident_plugin
    old = _start(plugin, "old")
    old._playback_clock.tail_s = simulated_time.now + .8
    shared = old._playback_clock
    plugin.dispatch("tts", {"action": "stop", "instance_id": "old"})
    assert shared.users == 0
    assert len(_Timer.created) == 1
    new = _start(plugin, "new")
    assert new._playback_clock is shared
    assert shared.users == 1
    assert shared.tail_s == simulated_time.now + .8
    other = _start(plugin, "other", "/other")
    assert other._playback_clock is not shared
    # A previously scheduled cleanup must not remove the newly active clock.
    simulated_time.now += 2
    _Timer.created[0].callback()
    assert plugin._playback_clocks["/test/tts"] is shared
    plugin.dispatch("tts", {"action": "stop"})
    assert not plugin._playback_clocks


def test_shared_topic_references_and_single_idle_cleanup_timer(resident_plugin, simulated_time):
    plugin = resident_plugin
    first = _start(plugin, "one")
    second = _start(plugin, "two")
    assert first._playback_clock is second._playback_clock
    assert first._playback_clock.users == 2
    first._playback_clock.tail_s = simulated_time.now + .8
    plugin.dispatch("tts", {"action": "stop", "instance_id": "one"})
    assert first._playback_clock.users == 1
    assert not _Timer.created
    plugin.dispatch("tts", {"action": "stop", "instance_id": "two"})
    for index in range(100):
        node = _start(plugin, str(index), f"/test{index}")
        node._playback_clock.tail_s = simulated_time.now + .6
        plugin.dispatch("tts", {"action": "stop", "instance_id": str(index)})
    assert len(_Timer.created) == 1
    simulated_time.now += 1
    _Timer.created[0].callback()
    assert not plugin._playback_clocks
    assert plugin._clock_cleanup_timer is None


def test_node_constructor_failure_releases_clock_lease(resident_plugin, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("node construction failed")

    monkeypatch.setattr(vits2, "_Vits2TTSNode", fail)
    answer = resident_plugin.dispatch("tts", {"action": "start", "instance_id": "failed", "input_topic": "/bad"})
    assert answer["state"] == "error"
    assert not resident_plugin._playback_clocks
    assert not _Timer.created


def test_two_concurrent_publishers_cannot_claim_the_same_credit():
    shared = vits2._PlaybackClock()
    nodes = [vits2._Vits2TTSNode("/concurrent", object(), str(i), shared) for i in range(2)]
    barrier = threading.Barrier(3)
    errors = []

    for node in nodes:
        original_publish = node._publish

        def slow_publish(pcm, original=original_publish):
            # Force a scheduling opportunity between quota inspection and
            # accounting; a lock-free check/update can overbook here.
            time.sleep(.005)
            original(pcm)

        node._publish = slow_publish

    def publish(node):
        try:
            barrier.wait(timeout=2)
            for _ in range(6):
                assert node._publish_pcm_frame(PCM) is not None
                with shared.lock:
                    assert shared.tail_s - time.monotonic() <= 1.0 + 1e-6
        except BaseException as error:
            errors.append(error)

    workers = [threading.Thread(target=publish, args=(node,)) for node in nodes]
    for worker in workers:
        worker.start()
    began = time.monotonic()
    barrier.wait(timeout=2)
    for worker in workers:
        worker.join(timeout=2)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert time.monotonic() - began >= .19
    assert sum(len(node._pub.messages) for node in nodes) == 12
    assert all(bytes(message) == PCM for node in nodes for message in node._pub.messages)
