#!/usr/bin/env python3
"""Exercise the real perception MCP -> ROS2 AudioChunk TTS interface.

Run inside the complete ROS image, on the SAME ROS_DOMAIN_ID as its dedicated
perception server. --mcp-url names that server's /mcp endpoint. WS_PORT belongs
to ASR and is not a TTS audio socket. This tool subscribes to the public DDS
topic directly; the private leaderboard receiver/metric code is unavailable.

Example:
  python3 tools/moss_platform_smoke.py --mcp-url http://127.0.0.1:17920/mcp \
      --ros-domain-id 81 --text '你好，欢迎使用第四范式。' --language zh \
      --rounds 3 --output /results/platform-smoke-81

--cases accepts {"cases": [{"id": "zh", "text": "你好。", "language": "zh",
                           "speaker_id": 0}, ...]} or the bare list.
Language/speaker configuration is shared by a server: use a dedicated idle
endpoint. This tool never launches containers, stops other instances, or uses
/tts/test. Each request owns a random instance/topic and cleans up only it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import threading
import time
import traceback
import urllib.error
import urllib.request
from urllib.parse import urlparse
import uuid
import wave


PCM_FORMAT = "audio/pcm-16k"
EOF_MAGIC = b"\x01\x00\xff\xff\x01\x00\xff\xff"
BYTES_PER_SECOND = 32000


class McpClient:
    def __init__(self, url, timeout):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.path != "/mcp":
            raise ValueError("--mcp-url must be the dedicated perception HTTP(S) /mcp endpoint")
        self.url, self.timeout, self.counter = url, timeout, 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rpc(self, method, params=None):
        self.counter += 1
        request_id = self.counter
        body = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "Accept": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get("jsonrpc") != "2.0" or result.get("id") != request_id:
            raise RuntimeError("MCP response version/id mismatch")
        if "error" in result:
            raise RuntimeError(f"MCP {method} failed: {result['error']}")
        if "result" not in result:
            raise RuntimeError("MCP response has no result")
        return result["result"]

    def call(self, action, **arguments):
        result = self.rpc("tools/call", {"name": "tts", "arguments": {"action": action, **arguments}})
        if result.get("isError"):
            raise RuntimeError(f"MCP TTS tool failed: {result}")
        contents = result.get("content", [])
        if len(contents) != 1 or contents[0].get("type") != "text":
            raise RuntimeError("Unexpected perception MCP text-content envelope")
        value = json.loads(contents[0]["text"])
        if not isinstance(value, dict):
            raise RuntimeError("TTS tool result is not an object")
        return value


class RosReceiver:
    """Subscribe before speak; callback timestamps include transport latency."""
    def __init__(self, topic, domain_id):
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from audio_msgs.msg import AudioChunk
        from std_msgs.msg import String
        self.rclpy, self.context = rclpy, Context()
        self.events, self.stopped = queue.Queue(), threading.Event()
        self.topic = topic
        rclpy.init(args=[], context=self.context)
        self.node = rclpy.create_node("moss_smoke_listener_" + uuid.uuid4().hex[:12], context=self.context)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=200,
                         durability=DurabilityPolicy.VOLATILE)

        def audio(msg):
            self.events.put({"kind": "audio", "received_monotonic_s": time.perf_counter(),
                             "format": msg.format, "data": bytes(msg.data),
                             "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)}})

        def perf(msg):
            try:
                value = json.loads(msg.data)
            except (ValueError, TypeError):
                return  # Other services may also use the shared diagnostic topic.
            self.events.put({"kind": "perf", "received_monotonic_s": time.perf_counter(), "data": value})

        self.audio_sub = self.node.create_subscription(AudioChunk, topic, audio, qos)
        self.perf_sub = self.node.create_subscription(String, "/perception/perf_spans", perf, qos)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True, name="moss-smoke-dds")
        self.thread.start()

    def _spin(self):
        try:
            while not self.stopped.is_set():
                self.executor.spin_once(timeout_sec=.05)
        except Exception as error:
            self.events.put({"kind": "error", "error": f"DDS executor failed: {error}"})

    def wait_ready(self, timeout):
        deadline, stable_since = time.perf_counter() + timeout, None
        while time.perf_counter() < deadline:
            count = self.node.count_publishers(self.topic)
            if count > 1:
                raise RuntimeError(f"Topic collision: {count} publishers on {self.topic}")
            ready = count == 1 and self.node.count_publishers("/perception/perf_spans") >= 1
            stable_since = (stable_since or time.perf_counter()) if ready else None
            if stable_since and time.perf_counter() - stable_since >= .25:
                return {"audio_publishers": count, "stable_discovery_s": .25,
                        "message_type": "audio_msgs/msg/AudioChunk", "qos": "BEST_EFFORT/VOLATILE/KEEP_LAST depth=200"}
            time.sleep(.02)
        raise TimeoutError("DDS audio/perf publisher discovery timed out; check matching ROS_DOMAIN_ID/network")

    def receive(self, timeout):
        return self.events.get(timeout=timeout)

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("DDS listener did not stop")
        self.executor.shutdown()
        self.node.destroy_node()
        self.rclpy.shutdown(context=self.context)


class AudioEvidence:
    def __init__(self, started, trace_id):
        self.started, self.trace_id = started, trace_id
        self.pcm, self.frames, self.spans = bytearray(), [], {}
        self.eof_received = None
        self.eof_stamp = None
        self.first_nonzero = None

    def accept(self, event):
        if event["kind"] == "error":
            raise RuntimeError(event["error"])
        if event["kind"] == "perf":
            data = event["data"]
            if isinstance(data, dict) and data.get("trace_id") == self.trace_id and data.get("type") == "perf_span":
                self.spans[data.get("span")] = data
            return
        if event["kind"] != "audio":
            raise ValueError("Unexpected receiver event")
        if event["format"] != PCM_FORMAT:
            raise ValueError(f"Wrong AudioChunk.format: {event['format']}")
        received = event["received_monotonic_s"]
        if received < self.started:
            raise ValueError("Audio arrived before this speak request; possible stale data/topic collision")
        payload = event["data"]
        if self.eof_received is not None:
            raise ValueError("Audio or duplicate EOF arrived after the utterance EOF")
        if payload == EOF_MAGIC:
            self.eof_received, self.eof_stamp = received, event["stamp"]
            if not self.pcm:
                raise ValueError("Received EOF without PCM; synthesis may have failed")
            return
        if not payload or len(payload) % 2 or len(payload) > 3200:
            raise ValueError("Invalid PCM16 AudioChunk byte count")
        if self.frames and self.frames[-1]["bytes"] != 3200:
            raise ValueError("A short PCM frame was followed by more PCM; expected only a final remainder")
        if self.first_nonzero is None and any(payload):
            self.first_nonzero = received
        self.pcm.extend(payload)
        self.frames.append({"bytes": len(payload), "received_monotonic_s": received, "stamp": event["stamp"]})

    def _merged_short_span(self):
        # _TTSNode._worker publishes only its merged tts_generate span when
        # the utterance is shorter than one full 100 ms PCM frame. An error
        # path also sends EOF, but never this successful frame-count span.
        generated = self.spans.get("tts_generate", {})
        meta = generated.get("meta", {})
        return (0 < len(self.pcm) < 3200 and len(self.frames) == 1
                and isinstance(meta, dict) and type(meta.get("frames")) is int
                and meta["frames"] == 1)

    def complete(self):
        return (self.eof_received is not None and "tts_generate" in self.spans
                and ("tts_playback" in self.spans or self._merged_short_span()))

    def result(self):
        if not self.complete() or not self.pcm or self.first_nonzero is None:
            raise ValueError("Missing real audio, EOF, or matching successful TTS perf spans")
        generated, played = self.spans["tts_generate"], self.spans.get("tts_playback")
        counted = played if played is not None else generated
        if counted.get("meta", {}).get("frames") != len(self.frames):
            raise ValueError("DDS frame count differs from successful publisher count; possible packet loss")
        for span in ((generated, played) if played is not None else (generated,)):
            if not isinstance(span.get("start_ts"), (int, float)) or not isinstance(span.get("end_ts"), (int, float)) or span["end_ts"] < span["start_ts"]:
                raise ValueError("Invalid successful perf-span timestamps")
        first = self.frames[0]["received_monotonic_s"]
        elapsed = self.eof_received - self.started
        duration = len(self.pcm) / BYTES_PER_SECOND
        return {"first_pcm_s": first - self.started, "first_nonzero_pcm_chunk_s": self.first_nonzero - self.started,
                "elapsed_to_eof_s": elapsed, "audio_s": duration, "rtf": elapsed / duration,
                "pcm_bytes": len(self.pcm), "chunks": len(self.frames), "sample_rate": 16000,
                "channels": 1, "sample_width_bytes": 2, "format": PCM_FORMAT,
                "nonempty_nonzero_pcm": True, "eof_received": True, "publisher_frames_match": True,
                "perf_span_mode": "separate_generation_and_playback" if played is not None else "merged_short_audio",
                "started_monotonic_s": self.started, "ended_monotonic_s": self.eof_received,
                "pcm_sha256": hashlib.sha256(self.pcm).hexdigest(), "frames": self.frames,
                "eof_stamp": self.eof_stamp, "successful_perf_spans": self.spans,
                "max_receive_gap_s": max((b["received_monotonic_s"] - a["received_monotonic_s"]
                                           for a, b in zip(self.frames, self.frames[1:])), default=0)}


def collect(receiver, evidence, ttft_timeout, completion_timeout, perf_timeout):
    while not evidence.complete():
        now = time.perf_counter()
        deadline = (evidence.eof_received + perf_timeout if evidence.eof_received is not None
                    else evidence.started + (completion_timeout if evidence.frames else ttft_timeout))
        try:
            event = receiver.receive(timeout=min(.1, max(0, deadline - now)))
        except queue.Empty:
            if now >= deadline:
                stage = "successful perf-span confirmation" if evidence.eof_received else "EOF" if evidence.frames else "first PCM"
                raise TimeoutError(f"Timed out waiting for {stage}")
            continue
        evidence.accept(event)
        if evidence.frames and evidence.frames[0]["received_monotonic_s"] - evidence.started > ttft_timeout:
            raise TimeoutError("First PCM arrived after the TTFT timeout")
        if evidence.eof_received and evidence.eof_received - evidence.started > completion_timeout:
            raise TimeoutError("EOF arrived after the completion timeout")
    return evidence.result()


def wait_idle(client, timeout):
    deadline = time.perf_counter() + timeout
    while True:
        info = client.call("info")
        if info.get("state") == "error":
            raise RuntimeError(f"TTS initialization failed: {info}")
        if info.get("engine") != "moss-multi":
            raise RuntimeError(f"Expected already configured moss-multi engine: {info}")
        if info.get("state") == "running":
            raise RuntimeError("Endpoint already has a running TTS instance; use an isolated idle server")
        if info.get("state") == "idle":
            return info
        if info.get("state") != "loading" or time.perf_counter() >= deadline:
            raise RuntimeError(f"TTS endpoint did not become idle: {info}")
        time.sleep(.2)


def run_case(client, args, case, index, *, receiver_factory=RosReceiver):
    identifier = "moss_smoke_" + uuid.uuid4().hex
    topic = f"{args.topic_prefix.rstrip('/')}/{identifier}"
    trace_id = identifier + "_trace"
    row = {"index": index, "case_id": case.get("id", str(index)), "text": case["text"],
           "language": case.get("language"), "speaker_id": case.get("speaker_id"),
           "instance_id": identifier, "topic_in": topic, "topic_out": topic + "/tts",
           "trace_id": trace_id, "ok": False}
    receiver, owned, evidence = None, False, None
    try:
        row["info_before"] = wait_idle(client, args.ready_timeout)
        settings = {}
        if case.get("language"):
            settings["moss_language"] = case["language"]
        if case.get("speaker_id") is not None:
            settings["speaker_id"] = case["speaker_id"]
        if settings:
            row["config"] = client.call("config", **settings)
            if row["config"].get("state") == "error":
                raise RuntimeError(f"TTS configuration failed: {row['config']}")
            row["info_after_config"] = wait_idle(client, args.ready_timeout)
        receiver = receiver_factory(topic + "/tts", args.ros_domain_id)
        began = time.perf_counter()
        owned = True  # Own random ID even if start fails after creating its node.
        row["start"] = client.call("start", input_topic=topic, instance_id=identifier)
        row["start_seconds"] = time.perf_counter() - began
        if row["start"].get("state") != "running":
            raise RuntimeError(f"TTS start did not return running: {row['start']}")
        expected_out = [{"topic": topic + "/tts", "format": PCM_FORMAT}]
        advertised = [{"topic": item.get("topic"), "format": item.get("format")} for item in row["start"].get("topic_out", [])]
        if advertised != expected_out:
            raise RuntimeError(f"TTS start advertised unexpected output topics: {advertised}")
        row["subscriber_ready"] = receiver.wait_ready(args.discovery_timeout)
        started = time.perf_counter()
        row["started_wall_unix_s"] = time.time()
        evidence = AudioEvidence(started, trace_id)
        row["speak"] = client.call("speak", text=case["text"], instance_id=identifier, _trace_id=trace_id)
        row["speak_rpc_seconds"] = time.perf_counter() - started
        if row["speak"].get("status") != "queued" or not row["speak"].get("action_id"):
            raise RuntimeError(f"TTS speak was not queued: {row['speak']}")
        row.update(collect(receiver, evidence, args.ttft_timeout, args.completion_timeout, args.perf_timeout))
        row["ttft_pass"] = row["first_pcm_s"] <= args.max_ttft
        row["rtf_pass"] = row["rtf"] <= args.max_rtf
        row["transport_success"] = True
        row["ok"] = row["ttft_pass"] and row["rtf_pass"]
    except Exception as error:
        row.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if evidence is not None and evidence.pcm:
            path = args.output / f"case-{index:03d}.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1); destination.setsampwidth(2); destination.setframerate(16000)
                destination.writeframes(evidence.pcm)
            row["audio_path"] = str(path)
            row.setdefault("partial_pcm_bytes", len(evidence.pcm))
        if owned:
            try:
                row["stop"] = client.call("stop", instance_id=identifier)
                if row["stop"].get("state") != "idle":
                    raise RuntimeError(f"Owned TTS instance did not stop: {row['stop']}")
            except Exception as error:
                row.update(ok=False, cleanup_error=str(error))
        if receiver is not None:
            try:
                receiver.close()
            except Exception as error:
                row.update(ok=False, receiver_cleanup_error=str(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ros-domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "0")))
    parser.add_argument("--topic-prefix", default="/benchmark/tts/input")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--cases", type=Path)
    parser.add_argument("--language", choices=["auto", "zh", "ja", "ru", "en", "es", "fr", "th", "de"])
    parser.add_argument("--speaker-id", type=int, choices=range(16))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--rpc-timeout", type=float, default=300)
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--discovery-timeout", type=float, default=15)
    parser.add_argument("--ttft-timeout", type=float, default=60)
    parser.add_argument("--completion-timeout", type=float, default=300)
    parser.add_argument("--perf-timeout", type=float, default=5)
    parser.add_argument("--max-ttft", type=float, default=1)
    parser.add_argument("--max-rtf", type=float, default=1)
    args = parser.parse_args(argv)
    if args.rounds < 1 or not args.topic_prefix.startswith("/") or not 0 <= args.ros_domain_id <= 232:
        parser.error("Invalid rounds, absolute topic prefix, or ROS domain")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"ok": False, "mcp_url": args.mcp_url, "ros_domain_id": args.ros_domain_id, "cases": [],
              "transport": "HTTP JSON-RPC tools/call(name=tts) -> ROS2 audio_msgs/msg/AudioChunk",
              "timing_scope": "Client speak request dispatch to DDS first PCM/EOF; includes publisher prebuffer and real-time pacing. Not pure model inference RTF; private evaluator timing code not available.",
              "validation_scope": "Exact format, nonzero PCM, EOF and matching successful perf spans/frame counts. Does not establish pronunciation, naturalness or transcript completeness.",
              "python": platform.python_version(), "thresholds": {"ttft_s": args.max_ttft, "rtf": args.max_rtf},
              "client_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        loaded = json.loads(args.cases.read_text()) if args.cases else [{"text": args.text}]
        cases = loaded["cases"] if isinstance(loaded, dict) else loaded
        if not isinstance(cases, list) or not cases:
            raise ValueError("Cases must be a nonempty list")
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("text"), str) or not case["text"].strip():
                raise ValueError("Each case needs nonempty text")
            for key, value in (("language", args.language), ("speaker_id", args.speaker_id)):
                if value is not None:
                    case.setdefault(key, value)
            if case.get("language") not in (None, "auto", "zh", "ja", "ru", "en", "es", "fr", "th", "de"):
                raise ValueError("Unsupported case language")
            if case.get("speaker_id") is not None and (type(case["speaker_id"]) is not int or not 0 <= case["speaker_id"] < 16):
                raise ValueError("Invalid case speaker_id")
        client = McpClient(args.mcp_url, args.rpc_timeout)
        report["initialize"] = client.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "moss-platform-smoke", "version": "1"}})
        tools = client.rpc("tools/list")
        if "tts" not in {tool.get("name") for tool in tools.get("tools", [])}:
            raise RuntimeError("Dedicated perception endpoint does not advertise tts")
        report["tts_tool"] = next(tool for tool in tools["tools"] if tool.get("name") == "tts")
        for repetition in range(args.rounds):
            for case in cases:
                row = run_case(client, args, case, len(report["cases"]))
                row["round"] = repetition
                report["cases"].append(row)
                (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps({key: row.get(key) for key in ("index", "case_id", "ok", "error", "first_pcm_s", "rtf", "audio_s")}, ensure_ascii=False), flush=True)
                if not row.get("transport_success") or row.get("cleanup_error") or row.get("receiver_cleanup_error"):
                    raise RuntimeError("Transport/cleanup failed; stopping this smoke client before further requests")
        report["ok"] = all(row["ok"] for row in report["cases"])
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["transport_success"] = bool(report["cases"]) and all(row.get("transport_success") for row in report["cases"])
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
