"""Protocol and failure-path tests; no ROS graph, GPU, server or network used."""
import ast
import importlib.util
import io
import json
from pathlib import Path
import queue
from types import SimpleNamespace
import time

import pytest


@pytest.fixture
def smoke():
    path = Path(__file__).resolve().parents[1] / "tools/moss_platform_smoke.py"
    spec = importlib.util.spec_from_file_location("_moss_platform_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audio(smoke, data, at, fmt=None):
    return {"kind": "audio", "format": fmt or smoke.PCM_FORMAT, "data": data,
            "received_monotonic_s": at, "stamp": {"sec": 3, "nanosec": 10}}


def spans(trace="test", frames=2):
    return [{"kind": "perf", "data": {"type": "perf_span", "trace_id": trace,
             "span": name, "start_ts": 100., "end_ts": 101., "meta": meta}}
            for name, meta in (("tts_generate", {"chars": 3}), ("tts_playback", {"frames": frames}))]


def populate(smoke, evidence, frame_count=2):
    for i in range(frame_count):
        evidence.accept(audio(smoke, b"\x01\x00" * 1600, evidence.started + .05 * (i + 1)))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, evidence.started + .15))
    for event in spans(evidence.trace_id, frame_count):
        evidence.accept(event)


def test_constants_match_actual_plugin_and_message_definition(smoke):
    perception = Path(__file__).resolve().parents[1]
    tree = ast.parse((perception / "plugins/tts.py").read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    assert ast.literal_eval(assignments["AUDIO_EOF_MAGIC"]) == smoke.EOF_MAGIC
    assert ast.literal_eval(assignments["SAMPLE_RATE"]) * 2 == smoke.BYTES_PER_SECOND
    assert ast.literal_eval(assignments["CHUNK_BYTES"]) == 3200
    message = (perception.parent / "deploy/ros-base/audio_msgs/msg/AudioChunk.msg").read_text()
    assert message.splitlines() == ["std_msgs/Header header", "string format", "uint8[] data"]


def test_mcp_request_uses_real_tools_call_envelope_and_action(smoke):
    client = smoke.McpClient("http://127.0.0.1:17920/mcp", 30)
    sent = []
    def open_request(request, timeout):
        body = json.loads(request.data)
        sent.append(body)
        assert timeout == 30 and request.full_url.endswith("/mcp")
        return io.BytesIO(json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {
            "content": [{"type": "text", "text": json.dumps({"state": "running"})}]}}).encode())
    client.opener = SimpleNamespace(open=open_request)
    assert client.call("start", input_topic="/smoke/input", instance_id="mine") == {"state": "running"}
    assert sent[0]["params"] == {"name": "tts", "arguments": {
        "action": "start", "input_topic": "/smoke/input", "instance_id": "mine"}}


@pytest.mark.parametrize("payload,match", [
    ({"jsonrpc": "2.0", "id": 99, "result": {}}, "version/id"),
    ({"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "failed"}}, "failed"),
    ({"jsonrpc": "2.0", "id": 1, "result": {"isError": True}}, "tool failed"),
])
def test_mcp_errors_are_not_successful_start(smoke, payload, match):
    client = smoke.McpClient("http://127.0.0.1:17920/mcp", 30)
    client.opener = SimpleNamespace(open=lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(RuntimeError, match=match):
        client.call("start", instance_id="mine")


def test_eof_not_counted_as_pcm_and_timings_start_before_speak(smoke):
    evidence = smoke.AudioEvidence(10., "test")
    populate(smoke, evidence)
    result = evidence.result()
    assert result["pcm_bytes"] == 6400 and result["chunks"] == 2
    assert result["audio_s"] == .2 and result["first_pcm_s"] == pytest.approx(.05)
    assert result["elapsed_to_eof_s"] == pytest.approx(.15)
    assert result["rtf"] == pytest.approx(.75)
    assert result["publisher_frames_match"]


@pytest.mark.parametrize("kind", ["empty", "odd", "oversize", "wrong_format", "stale", "eof_only"])
def test_bad_audio_packets_fail_closed(smoke, kind):
    evidence = smoke.AudioEvidence(10., "test")
    payload = {"empty": b"", "odd": b"a", "oversize": b"a" * 3202,
               "eof_only": smoke.EOF_MAGIC}.get(kind, b"\x01\x00" * 1600)
    packet = audio(smoke, payload, 9. if kind == "stale" else 10.1,
                   "audio/wav" if kind == "wrong_format" else None)
    with pytest.raises(ValueError):
        evidence.accept(packet)


def test_partial_audio_eof_without_success_spans_is_not_success(smoke):
    evidence = smoke.AudioEvidence(10., "test")
    evidence.accept(audio(smoke, b"\x01\x00" * 1600, 10.1))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.2))
    for event in spans("someone_else", 1):
        evidence.accept(event)
    assert not evidence.complete()
    with pytest.raises(ValueError, match="successful TTS perf spans"):
        evidence.result()


def test_short_audio_accepts_publishers_merged_success_span(smoke):
    evidence = smoke.AudioEvidence(10., "short")
    evidence.accept(audio(smoke, b"\x01\x00" * 640, 10.05))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.1))
    generated = spans("short", 1)[0]
    generated["data"]["meta"]["frames"] = 1
    evidence.accept(generated)
    assert evidence.complete()
    result = evidence.result()
    assert result["pcm_bytes"] == 1280 and result["chunks"] == 1
    assert result["audio_s"] == .04 and result["publisher_frames_match"]
    assert result["perf_span_mode"] == "merged_short_audio"
    assert set(result["successful_perf_spans"]) == {"tts_generate"}


@pytest.mark.parametrize("frame_count", [None, 0, 2, True, 1.0, "1"])
def test_short_audio_requires_exact_merged_success_frame_count(smoke, frame_count):
    evidence = smoke.AudioEvidence(10., "short")
    evidence.accept(audio(smoke, b"\x01\x00" * 640, 10.05))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.1))
    generated = spans("short", 1)[0]
    generated["data"]["meta"]["frames"] = frame_count
    evidence.accept(generated)
    assert not evidence.complete()
    with pytest.raises(ValueError, match="successful TTS perf spans"):
        evidence.result()


def test_short_error_eof_without_success_span_still_fails(smoke):
    evidence = smoke.AudioEvidence(10., "short")
    evidence.accept(audio(smoke, b"\x01\x00" * 640, 10.05))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.1))
    assert not evidence.complete()
    with pytest.raises(ValueError, match="successful TTS perf spans"):
        evidence.result()


@pytest.mark.parametrize("pcm_bytes", [3200, 6400])
def test_normal_audio_cannot_substitute_merged_span_for_missing_playback(smoke, pcm_bytes):
    evidence = smoke.AudioEvidence(10., "normal")
    for index in range(pcm_bytes // 3200):
        evidence.accept(audio(smoke, b"\x01\x00" * 1600, 10.05 + index * .05))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.2))
    generated = spans("normal", pcm_bytes // 3200)[0]
    generated["data"]["meta"]["frames"] = pcm_bytes // 3200
    evidence.accept(generated)
    assert not evidence.complete()
    with pytest.raises(ValueError, match="successful TTS perf spans"):
        evidence.result()


def test_dds_dropped_audio_frame_is_detected_from_publisher_count(smoke):
    evidence = smoke.AudioEvidence(10., "test")
    populate(smoke, evidence)
    evidence.accept(spans("test", 3)[1])
    with pytest.raises(ValueError, match="packet loss"):
        evidence.result()


def test_all_zero_pcm_is_not_accepted_even_with_eof_and_success_spans(smoke):
    evidence = smoke.AudioEvidence(10., "test")
    evidence.accept(audio(smoke, bytes(3200), 10.1))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.2))
    for event in spans("test", 1):
        evidence.accept(event)
    with pytest.raises(ValueError, match="Missing real audio"):
        evidence.result()


def test_timeout_does_not_synthesize_fake_audio(smoke):
    receiver = SimpleNamespace(receive=lambda **kw: (_ for _ in ()).throw(queue.Empty()))
    evidence = smoke.AudioEvidence(time.perf_counter() - 10, "test")
    with pytest.raises(TimeoutError, match="first PCM"):
        smoke.collect(receiver, evidence, .1, .2, .1)
    assert not evidence.pcm


@pytest.fixture
def lifecycle(smoke, tmp_path):
    class Receiver:
        instances = []
        def __init__(self, topic, domain):
            self.topic, self.domain, self.closed = topic, domain, False
            self.events = queue.Queue()
            self.instances.append(self)
        def wait_ready(self, timeout):
            return {"audio_publishers": 1}
        def receive(self, timeout):
            return self.events.get(timeout=timeout)
        def close(self):
            self.closed = True

    class Client:
        def __init__(self):
            self.calls = []
            self.start_error = False
            self.busy = False
        def call(self, action, **kwargs):
            self.calls.append((action, kwargs))
            if action == "info":
                return {"state": "running" if self.busy else "idle", "engine": "moss-multi"}
            if action == "config":
                return {"status": "configured"}
            if action == "start":
                if self.start_error:
                    return {"state": "error", "message": "Missing sentencepiece"}
                return {"state": "running", "topic_out": [{"topic": kwargs["input_topic"] + "/tts", "format": smoke.PCM_FORMAT}]}
            if action == "speak":
                receiver = Receiver.instances[-1]
                now = time.perf_counter()
                receiver.events.put(audio(smoke, b"\x01\x00" * 1600, now + .01))
                receiver.events.put(audio(smoke, b"\x01\x00" * 1600, now + .05))
                receiver.events.put(audio(smoke, smoke.EOF_MAGIC, now + .15))
                for event in spans(kwargs["_trace_id"], 2):
                    receiver.events.put(event)
                return {"status": "queued", "action_id": "speak-test"}
            if action == "stop":
                return {"state": "idle"}
            raise AssertionError(action)

    args = SimpleNamespace(output=tmp_path, ros_domain_id=81, topic_prefix="/benchmark/tts/input",
                           ready_timeout=1, discovery_timeout=1, ttft_timeout=1,
                           completion_timeout=1, perf_timeout=1, max_ttft=1, max_rtf=1)
    return args, Client(), Receiver


def test_real_sequence_uses_private_id_topic_and_successful_cleanup(smoke, lifecycle):
    args, client, receiver = lifecycle
    row = smoke.run_case(client, args, {"text": "你好。", "language": "zh", "speaker_id": 0}, 0,
                         receiver_factory=receiver)
    assert row["ok"] and row["start"]["state"] == "running"
    assert row["pcm_bytes"] == 6400 and Path(row["audio_path"]).exists()
    actions = [a for a, _ in client.calls]
    assert actions == ["info", "config", "info", "start", "speak", "stop"]
    assert client.calls[-1][1] == {"instance_id": row["instance_id"]}
    assert row["instance_id"].startswith("moss_smoke_") and receiver.instances[0].closed


def test_start_error_never_calls_speak_and_stops_only_own_attempt(smoke, lifecycle):
    args, client, receiver = lifecycle
    client.start_error = True
    row = smoke.run_case(client, args, {"text": "你好。"}, 0, receiver_factory=receiver)
    assert not row["ok"] and "sentencepiece" in row["error"]
    assert [a for a, _ in client.calls] == ["info", "start", "stop"]
    assert client.calls[-1][1] == {"instance_id": row["instance_id"]}
    assert receiver.instances[0].closed


def test_already_running_endpoint_is_never_configured_or_stopped(smoke, lifecycle):
    args, client, receiver = lifecycle
    client.busy = True
    row = smoke.run_case(client, args, {"text": "你好。", "language": "zh"}, 0, receiver_factory=receiver)
    assert not row["ok"] and "isolated idle server" in row["error"]
    assert client.calls == [("info", {})] and not receiver.instances
