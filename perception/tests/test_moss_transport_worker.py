"""Exercise the real TTS worker with fake ROS and deterministic PCM producers."""

import threading

import pytest

import plugins.tts as tts


EOF = bytes(tts.AUDIO_EOF_MAGIC)
FRAME = bytes([17, 2]) * 1600
TAIL = bytes([31, 4]) * 127


class Publisher:
    def __init__(self, subscribers=1, forbid_queries=False):
        self.subscribers = subscribers
        self.forbid_queries = forbid_queries
        self.queries = 0
        self.messages = []
        self.changed = threading.Condition()
        self.first_pcm = threading.Event()

    def get_subscription_count(self):
        if self.forbid_queries:
            raise AssertionError("non-MOSS worker must not query subscriber readiness")
        with self.changed:
            self.queries += 1
            self.changed.notify_all()
            return self.subscribers

    def set_subscribers(self, count):
        with self.changed:
            self.subscribers = count
            self.changed.notify_all()

    def publish(self, msg):
        data = bytes(msg.data)
        assert msg.format == "audio/pcm-16k"
        with self.changed:
            self.messages.append(data)
            if data != EOF:
                self.first_pcm.set()
            self.changed.notify_all()

    def wait_queries(self, count):
        with self.changed:
            return self.changed.wait_for(lambda: self.queries >= count, timeout=2)

    def snapshot(self):
        with self.changed:
            return list(self.messages)


class Adapter:
    dry_run_text = "worker-dry-run"

    def __init__(self, produce, moss=True):
        self.produce = produce
        if moss:
            self.transport_lead_seconds = 1.0
            self.transport_mode = "as_generated"
            self.subscriber_settle_s = 0.0

    def synthesize_stream(self, text):
        if text == self.dry_run_text:
            yield FRAME
            return
        yield from self.produce(text)


@pytest.fixture
def worker(monkeypatch):
    nodes = []
    releases = []
    completed = {}
    changed = threading.Condition()
    monkeypatch.setattr(tts, "fire_hook", lambda *_args, **_kwargs: None)

    def complete(action_id, text, frames_sent, interrupted):
        with changed:
            completed[action_id] = {
                "text": text, "frames": frames_sent, "interrupted": interrupted,
            }
            changed.notify_all()

    monkeypatch.setattr(tts, "_complete_action", complete)

    def create(produce, *, moss=True, subscribers=1, release=None):
        node = tts._TTSNode("/test/moss-worker", Adapter(produce, moss=moss),
                            node_suffix=str(len(nodes)))
        pub = Publisher(subscribers, forbid_queries=not moss)
        # Keep these reader controls local to this test; shared ROS stubs stay
        # unchanged, including tests of other engines.
        monkeypatch.setattr(node, "_pub", pub)
        nodes.append(node)
        if release is not None:
            releases.append(release)
        assert node.start()["state"] == "running"
        return node, pub

    def wait_complete(action_id):
        with changed:
            assert changed.wait_for(lambda: action_id in completed, timeout=3), action_id
            return dict(completed[action_id])

    create.wait_complete = wait_complete
    yield create
    for release in releases:
        release.set()
    for node in nodes:
        node.stop()
        assert not node._worker_thread.is_alive()
        node.destroy_node()


def assert_pcm_and_one_eof(pub, expected):
    messages = pub.snapshot()
    assert messages[-1] == EOF
    assert messages.count(EOF) == 1
    pcm = messages[:-1]
    assert pcm and all(0 < len(part) <= 3200 and len(part) % 2 == 0 for part in pcm)
    assert b"".join(pcm) == expected
    return pcm


def test_as_generated_worker_preserves_all_pcm_and_one_eof(worker):
    expected = FRAME * 6 + TAIL
    # Codec callback boundaries need not coincide with ROS frame boundaries.
    def produce(_text):
        for start, end in [(0, 222), (222, 4000), (4000, len(expected))]:
            yield expected[start:end]

    node, pub = worker(produce)
    assert node._audio_pacer.mode == "as_generated"
    node.enqueue("exact audio", action_id="complete")
    result = worker.wait_complete("complete")
    pcm = assert_pcm_and_one_eof(pub, expected)
    assert [len(part) for part in pcm] == [3200] * 6 + [len(TAIL)]
    assert result == {"text": "exact audio", "frames": 7, "interrupted": False}


def test_moss_publishes_first_real_frame_before_later_pcm_exists(worker):
    awaiting_tail = threading.Event()
    release = threading.Event()

    def produce(_text):
        yield FRAME
        awaiting_tail.set()
        assert release.wait(2), "test did not release the codec producer"
        yield TAIL

    node, pub = worker(produce, release=release)
    node.enqueue("stream first frame", action_id="first")
    assert awaiting_tail.wait(2)
    # Waiting for five frames or for the complete waveform would fail here.
    assert pub.first_pcm.wait(1)
    assert pub.snapshot() == [FRAME]
    release.set()
    assert not worker.wait_complete("first")["interrupted"]
    assert_pcm_and_one_eof(pub, FRAME + TAIL)


@pytest.mark.parametrize("payload", [FRAME + TAIL, TAIL], ids=["full-and-tail", "sub-frame"])
def test_worker_waits_for_reader_even_for_sub_frame_response(worker, payload):
    produced = threading.Event()

    def produce(_text):
        yield payload
        produced.set()

    node, pub = worker(produce, subscribers=0)
    node.enqueue("wait for reader", action_id="waiting")
    assert produced.wait(2)
    assert pub.wait_queries(3), "worker did not reach its subscriber gate"
    assert pub.snapshot() == []
    pub.set_subscribers(1)
    assert not worker.wait_complete("waiting")["interrupted"]
    assert_pcm_and_one_eof(pub, payload)


@pytest.mark.parametrize("cancel", ["interrupt", "stop"])
def test_gate_cancellation_emits_no_pcm_and_next_request_can_run(worker, cancel):
    node, pub = worker(lambda _text: iter([FRAME + TAIL]), subscribers=0)
    node.enqueue("cancel at gate", action_id="cancelled")
    assert pub.wait_queries(3)
    assert pub.snapshot() == []
    getattr(node, cancel)()
    result = worker.wait_complete("cancelled")
    assert result["interrupted"] is True
    assert result["frames"] == 0
    assert pub.snapshot() == [EOF]
    old_gate = node._audio_pacer.gate

    pub.set_subscribers(1)
    if cancel == "stop":
        assert not node._worker_thread.is_alive()
        assert node.start()["state"] == "running"
    node.enqueue("next request", action_id="next")
    assert not worker.wait_complete("next")["interrupted"]
    assert node._audio_pacer.gate is not old_gate
    assert pub.snapshot() == [EOF, FRAME, TAIL, EOF]


def test_non_moss_keeps_five_frame_prebuffer_and_has_no_subscriber_gate(worker):
    awaiting_fifth = threading.Event()
    release = threading.Event()

    def produce(_text):
        yield FRAME * 4
        awaiting_fifth.set()
        assert release.wait(2), "test did not release the ordinary producer"
        yield FRAME

    node, pub = worker(produce, moss=False, subscribers=0, release=release)
    assert node._audio_pacer is None
    node.enqueue("ordinary engine", action_id="ordinary")
    assert awaiting_fifth.wait(2)
    assert not pub.first_pcm.wait(.05)
    release.set()
    result = worker.wait_complete("ordinary")
    assert result["interrupted"] is False and result["frames"] == 5
    assert pub.queries == 0
    assert_pcm_and_one_eof(pub, FRAME * 5)
