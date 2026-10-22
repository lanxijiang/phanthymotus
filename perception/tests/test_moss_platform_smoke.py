"""Protocol and failure-path tests; no ROS graph, GPU, server or network used."""
import ast
import hashlib
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


def spans(trace="test", frames=2, pcm=None):
    pcm = b"\x01\x00" * (1600 * frames) if pcm is None else pcm
    integrity = {"pcm_bytes": len(pcm), "pcm_sha256": hashlib.sha256(pcm).hexdigest()}
    integrity.update(generated_pcm_bytes=len(pcm), generated_pcm_sha256=integrity["pcm_sha256"])
    return [{"kind": "perf", "data": {"type": "perf_span", "trace_id": trace,
             "span": name, "start_ts": 100., "end_ts": 101., "meta": meta}}
            for name, meta in (("tts_generate", {"chars": 3, **integrity}),
                               ("tts_playback", {"frames": frames, **integrity}))]


def populate(smoke, evidence, frame_count=2):
    for i in range(frame_count):
        evidence.accept(audio(smoke, b"\x01\x00" * 1600, evidence.started + .05 * (i + 1)))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, evidence.started + .15))
    for event in spans(evidence.trace_id, frame_count):
        evidence.accept(event)


PCM_FIELDS = ("pcm_bytes", "pcm_sha256", "generated_pcm_bytes", "generated_pcm_sha256")
PCM_CHECKS = ("publisher_pcm_bytes_match", "publisher_pcm_sha256_match",
              "generated_pcm_bytes_match", "generated_pcm_sha256_match")


def remove_pcm_metadata(evidence):
    for span in evidence.spans.values():
        for field in PCM_FIELDS:
            span["meta"].pop(field, None)


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
    assert result["rtf_to_eof"] == pytest.approx(.75)
    assert result["last_pcm_s"] == pytest.approx(.1)
    assert result["rtf_last_pcm"] == pytest.approx(.5)
    assert result["pcm_sha256"] == hashlib.sha256(b"\x01\x00" * 3200).hexdigest()
    assert result["publisher_pcm_bytes_match"] and result["publisher_pcm_sha256_match"]
    assert result["publisher_frames_match"]
    assert result["pcm_integrity_metadata"] == "verified"


def test_missing_all_pcm_metadata_requires_explicit_legacy_opt_in(smoke):
    evidence = smoke.AudioEvidence(10., "test")
    populate(smoke, evidence)
    remove_pcm_metadata(evidence)
    with pytest.raises(ValueError, match="PCM bytes/hash"):
        evidence.result()


@pytest.mark.parametrize("short", [False, True])
def test_legacy_missing_metadata_reports_unavailable_not_verified(smoke, short):
    evidence = smoke.AudioEvidence(10., "test", allow_legacy_perf_metadata=True)
    if short:
        pcm = b"\x01\x00" * 640
        evidence.accept(audio(smoke, pcm, 10.05))
        evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.1))
        generated = spans("test", 1, pcm=pcm)[0]
        generated["data"]["meta"]["frames"] = 1
        evidence.accept(generated)
    else:
        populate(smoke, evidence)
    remove_pcm_metadata(evidence)
    result = evidence.result()
    assert all(result[field] is None for field in PCM_CHECKS)
    assert result["pcm_integrity_metadata"] == "unavailable"
    assert result["publisher_frames_match"] and result["eof_received"]
    assert result["pcm_sha256"] == hashlib.sha256(evidence.pcm).hexdigest()


@pytest.mark.parametrize("missing", ["eof", "generation", "playback", "frame_count"])
def test_legacy_opt_in_keeps_eof_spans_and_frame_count_checks(smoke, missing):
    evidence = smoke.AudioEvidence(10., "test", allow_legacy_perf_metadata=True)
    populate(smoke, evidence)
    remove_pcm_metadata(evidence)
    if missing == "eof":
        evidence.eof_received = None
    elif missing == "generation":
        del evidence.spans["tts_generate"]
    elif missing == "playback":
        del evidence.spans["tts_playback"]
    else:
        evidence.spans["tts_playback"]["meta"]["frames"] = 3
    with pytest.raises(ValueError):
        evidence.result()


@pytest.mark.parametrize("span_name", ["tts_generate", "tts_playback"])
@pytest.mark.parametrize("field", PCM_FIELDS)
def test_legacy_opt_in_rejects_even_one_present_pcm_field(smoke, span_name, field):
    evidence = smoke.AudioEvidence(10., "test", allow_legacy_perf_metadata=True)
    populate(smoke, evidence)
    original = evidence.spans[span_name]["meta"][field]
    remove_pcm_metadata(evidence)
    evidence.spans[span_name]["meta"][field] = original
    with pytest.raises(ValueError, match="PCM bytes/hash"):
        evidence.result()


@pytest.mark.parametrize("field", PCM_FIELDS)
def test_present_null_pcm_metadata_is_not_legacy(smoke, field):
    evidence = smoke.AudioEvidence(10., "test", allow_legacy_perf_metadata=True)
    populate(smoke, evidence)
    remove_pcm_metadata(evidence)
    evidence.spans["tts_playback"]["meta"][field] = None
    with pytest.raises(ValueError, match="PCM bytes/hash"):
        evidence.result()


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
    generated = spans("short", 1, pcm=b"\x01\x00" * 640)[0]
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


@pytest.mark.parametrize("field,value", [
    ("pcm_bytes", None), ("pcm_bytes", 6408), ("pcm_bytes", 6400.0),
    ("pcm_sha256", None), ("pcm_sha256", "0" * 64),
])
@pytest.mark.parametrize("allow_legacy", [False, True])
def test_same_frame_count_cannot_hide_wrong_or_missing_pcm_evidence(smoke, field, value, allow_legacy):
    evidence = smoke.AudioEvidence(10., "test", allow_legacy_perf_metadata=allow_legacy)
    populate(smoke, evidence)
    metadata = evidence.spans["tts_playback"]["meta"]
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    with pytest.raises(ValueError, match="PCM bytes/hash"):
        evidence.result()


def test_reordered_equal_size_frames_fail_pcm_hash(smoke):
    evidence = smoke.AudioEvidence(10., "test")
    first, second = b"\x01\x00" * 1600, b"\x02\x00" * 1600
    evidence.accept(audio(smoke, second, 10.05))
    evidence.accept(audio(smoke, first, 10.1))
    evidence.accept(audio(smoke, smoke.EOF_MAGIC, 10.15))
    for event in spans("test", 2, pcm=first + second):
        evidence.accept(event)
    with pytest.raises(ValueError, match="PCM bytes/hash"):
        evidence.result()


@pytest.mark.parametrize("field,value", [
    ("generated_pcm_bytes", None), ("generated_pcm_bytes", 6402),
    ("generated_pcm_sha256", None), ("generated_pcm_sha256", "0" * 64),
])
@pytest.mark.parametrize("allow_legacy", [False, True])
def test_producer_pcm_must_match_publish_and_receive(smoke, field, value, allow_legacy):
    evidence = smoke.AudioEvidence(10., "test", allow_legacy_perf_metadata=allow_legacy)
    populate(smoke, evidence)
    metadata = evidence.spans["tts_playback"]["meta"]
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    with pytest.raises(ValueError, match="Generated PCM bytes/hash"):
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
        def __init__(self, topic, domain, *, subscribe_audio=True):
            self.topic, self.domain, self.closed = topic, domain, False
            self.audio_enabled = subscribe_audio
            self.wait_calls = 0
            self.pending = None
            self.subscription_times = []
            self.events = queue.Queue()
            self.instances.append(self)
        def subscribe_audio(self):
            assert not self.audio_enabled
            self.audio_enabled = True
            self.subscription_times.append(time.perf_counter())
            if self.pending:
                self.pending()
        def wait_ready(self, timeout):
            self.wait_calls += 1
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
            self.legacy_perf = False
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
                def emit():
                    now = time.perf_counter()
                    receiver.events.put(audio(smoke, b"\x01\x00" * 1600, now + .01))
                    receiver.events.put(audio(smoke, b"\x01\x00" * 1600, now + .05))
                    receiver.events.put(audio(smoke, smoke.EOF_MAGIC, now + .15))
                    for event in spans(kwargs["_trace_id"], 2):
                        if self.legacy_perf:
                            for field in PCM_FIELDS:
                                event["data"]["meta"].pop(field, None)
                        receiver.events.put(event)
                if receiver.audio_enabled:
                    emit()
                else:
                    receiver.pending = emit
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
    assert receiver.instances[0].wait_calls == 1


@pytest.mark.parametrize("allow_legacy", [False, True])
def test_run_case_forwards_explicit_legacy_setting(smoke, lifecycle, allow_legacy):
    args, client, receiver = lifecycle
    args.allow_legacy_perf_metadata = allow_legacy
    client.legacy_perf = True
    row = smoke.run_case(client, args, {"text": "Hello."}, 0, receiver_factory=receiver)
    assert row["ok"] is allow_legacy
    if allow_legacy:
        assert row["pcm_integrity_metadata"] == "unavailable"
        assert all(row[field] is None for field in PCM_CHECKS)
    else:
        assert "PCM bytes/hash" in row["error"]
    assert client.calls[-1][0] == "stop" and receiver.instances[0].closed


@pytest.mark.parametrize("allow_legacy", [False, True])
def test_cli_records_and_forwards_legacy_opt_in(smoke, tmp_path, monkeypatch, allow_legacy):
    class Client:
        def __init__(self, *args):
            pass
        def rpc(self, method, params=None):
            return {"tools": [{"name": "tts"}]} if method == "tools/list" else {}
    def run_case(client, args, case, index):
        assert args.allow_legacy_perf_metadata is allow_legacy
        return {"ok": True, "transport_success": True}
    monkeypatch.setattr(smoke, "McpClient", Client)
    monkeypatch.setattr(smoke, "run_case", run_case)
    output = tmp_path / "output"
    argv = ["--mcp-url", "http://127.0.0.1:17920/mcp", "--output", str(output), "--text", "Hello."]
    if allow_legacy:
        argv.append("--allow-legacy-perf-metadata")
    assert smoke.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["allow_legacy_perf_metadata"] is allow_legacy


def test_immediate_mode_never_waits_for_discovery(smoke, lifecycle):
    args, client, receiver = lifecycle
    args.discovery_mode = "immediate"
    def after_start(topic, domain):
        assert [a for a, _ in client.calls] == ["info", "start"]
        return receiver(topic, domain)
    row = smoke.run_case(client, args, {"text": "Hello."}, 0, receiver_factory=after_start)
    assert row["ok"] and receiver.instances[0].wait_calls == 0
    assert "subscriber_ready" not in row


@pytest.mark.parametrize("delay", [.8, 1.2])
def test_delayed_mode_subscribes_after_dispatch_and_counts_delay_in_ttft(smoke, lifecycle, monkeypatch, delay):
    args, client, receiver = lifecycle
    clock = SimpleNamespace(now=10.)
    monkeypatch.setattr(smoke.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(smoke.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    args.discovery_mode = "delayed"
    args.subscriber_delay_s = delay
    args.ttft_timeout = args.completion_timeout = 3
    row = smoke.run_case(client, args, {"text": "Hello."}, 0, receiver_factory=receiver)
    assert row["transport_success"] and row["publisher_pcm_sha256_match"]
    assert not row["rtf_pass"]  # Do not hide delayed subscription from the metric.
    assert not row["ok"] and row["ttft_pass"] == (delay < 1)
    assert row["audio_subscription_created_s"] == pytest.approx(delay)
    assert row["first_pcm_s"] == pytest.approx(delay + .01)
    assert receiver.instances[0].subscription_times == pytest.approx([10 + delay])
    assert receiver.instances[0].wait_calls == 0 and receiver.instances[0].closed


def test_no_save_audio_keeps_integrity_evidence_without_wav(smoke, lifecycle):
    args, client, receiver = lifecycle
    args.no_save_audio = True
    row = smoke.run_case(client, args, {"text": "你好。"}, 0, receiver_factory=receiver)
    assert row["ok"] and row["publisher_pcm_sha256_match"]
    assert row["pcm_bytes"] == row["partial_pcm_bytes"] == 6400
    assert row["pcm_sha256"] == row["partial_pcm_sha256"]
    assert "audio_path" not in row and not list(args.output.glob("*.wav"))


def test_reused_topic_keeps_instance_but_has_unique_trace_per_request(smoke, lifecycle):
    args, client, receiver = lifecycle
    args.reuse_topic = True
    first = smoke.run_case(client, args, {"text": "你好。"}, 0, receiver_factory=receiver)
    second = smoke.run_case(client, args, {"text": "Hello."}, 1, receiver_factory=receiver)
    assert first["ok"] and second["ok"]
    assert first["instance_id"] == second["instance_id"]
    assert first["topic_out"] == second["topic_out"]
    assert first["trace_id"] != second["trace_id"]
    assert [a for a, _ in client.calls] == ["info", "start", "speak", "stop"] * 2
    assert all(item.closed for item in receiver.instances)


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
