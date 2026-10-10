"""Startup must not advertise TTS before its fixed warmup completes."""

from __future__ import annotations

import pytest

import vision_stubs  # noqa: F401
import main


def _config(**updates):
    cfg = {"preload_tts": True, "preload_tts_timeout_s": 5,
           "plugins": {"tts": {"enabled": True, "warmup": True}}}
    cfg.update(updates)
    return cfg


class _Plugin:
    PREFIX = "tts"

    def __init__(self, states, *, events=None, stop_result=None):
        self.states = list(states)
        self.calls = []
        self.events = [] if events is None else events
        self.stop_result = {"state": "idle"} if stop_result is None else stop_result

    def dispatch(self, name, args):
        assert name == "tts"
        self.calls.append(dict(args))
        action = args["action"]
        self.events.append(action)
        if action == "stop":
            return self.stop_result
        assert action in ("start", "info"), "preload must never enqueue benchmark speech"
        result = self.states[0]
        if len(self.states) > 1:
            self.states.pop(0)
        return result


def _bundle(plugin=None):
    bundle = main.PerceptionBundle.__new__(main.PerceptionBundle)
    bundle._plugins = [] if plugin is None else [plugin]
    return bundle


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.0

        def sleep(self, duration):
            self.now += duration

    clock = Clock()
    monkeypatch.setattr(main.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(main.time, "sleep", clock.sleep)
    return clock


def test_preload_waits_then_removes_only_its_instance(clock, caplog):
    plugin = _Plugin([{"state": "loading"}, {"state": "loading"}, {"state": "running"}])
    with caplog.at_level("INFO"):
        main._preload_tts(_bundle(plugin), _config())
    assert [call["action"] for call in plugin.calls] == ["start", "info", "info", "stop"]
    assert len({call["instance_id"] for call in plugin.calls}) == 1
    assert plugin.calls[-1]["instance_id"], "stop must never target every instance"
    assert all("text" not in call for call in plugin.calls)
    assert clock.now == pytest.approx(0.2)
    assert "download/load/warmup finished" in caplog.text


def test_disabled_by_default_does_not_dispatch():
    plugin = _Plugin([])
    main._preload_tts(_bundle(plugin), {})
    assert plugin.calls == []


def test_adapter_ready_before_instance_start_is_not_a_false_failure(clock):
    plugin = _Plugin([{"state": "loading"}, {"state": "idle"}, {"state": "running"}])
    main._preload_tts(_bundle(plugin), _config())
    assert [call["action"] for call in plugin.calls] == ["start", "info", "info", "stop"]


def test_missing_plugin_is_fatal():
    with pytest.raises(RuntimeError, match="registered TTS"):
        main._preload_tts(_bundle(), _config())


def test_disabled_warmup_is_rejected():
    cfg = _config()
    cfg["plugins"]["tts"]["warmup"] = False
    with pytest.raises(ValueError, match="warmup=true"):
        main._preload_tts(_bundle(_Plugin([])), cfg)


@pytest.mark.parametrize("states", [
    [{"state": "error", "message": "engine checksum mismatch"}],
    [{"state": "loading"}, {"state": "error", "error": "engine checksum mismatch"}],
])
def test_load_errors_propagate_and_cancel_pending_instance(states, clock, caplog):
    plugin = _Plugin(states)
    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="engine checksum mismatch"):
        main._preload_tts(_bundle(plugin), _config())
    assert plugin.calls[-1]["action"] == "stop"
    assert "warmup finished" not in caplog.text


def test_timeout_is_fatal_and_cancels_pending_instance(clock):
    plugin = _Plugin([{"state": "loading"}])
    with pytest.raises(TimeoutError, match="timed out"):
        main._preload_tts(_bundle(plugin), _config(preload_tts_timeout_s=0.25))
    assert clock.now == pytest.approx(0.25)
    assert plugin.calls[-1]["action"] == "stop"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        main._preload_tts(_bundle(_Plugin([])), _config(preload_tts_timeout_s=timeout))


@pytest.mark.parametrize("status", [None, {"state": "ready"}])
def test_missing_or_unexpected_state_cannot_pass_preload(status):
    plugin = _Plugin([status])
    with pytest.raises(RuntimeError):
        main._preload_tts(_bundle(plugin), _config())
    assert plugin.calls[-1]["action"] == "stop"


def test_preload_cleanup_failure_is_fatal():
    plugin = _Plugin([{"state": "running"}], stop_result={"state": "error"})
    with pytest.raises(RuntimeError, match="did not stop cleanly"):
        main._preload_tts(_bundle(plugin), _config())


@pytest.mark.parametrize("load_fails", [False, True])
def test_ports_and_registration_follow_successful_preload(monkeypatch, clock, load_fails):
    events = []
    result = ({"state": "error", "error": "GPU model unavailable"}
              if load_fails else {"state": "running"})
    bundle = _bundle(_Plugin([{"state": "loading"}, result], events=events))
    monkeypatch.setattr(main, "_load_config", _config)
    monkeypatch.setattr(main, "PerceptionBundle", lambda cfg, executor: bundle)
    monkeypatch.setattr(main.rclpy, "init", lambda: events.append("ros-init"), raising=False)
    monkeypatch.setattr(main.rclpy, "shutdown", lambda: events.append("ros-shutdown"), raising=False)

    class Executor:
        def create_task(self, callback):
            self.callback = callback

        def spin(self):
            events.append("spin")
            self.callback()

        def shutdown(self):
            events.append("executor-shutdown")

    class Thread:
        def __init__(self, *, target, args=(), **kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    class Server:
        def __init__(self, *args):
            events.append("http-listen")

        def serve_forever(self):
            events.append("http-serve")

    monkeypatch.setattr(main.rclpy.executors, "MultiThreadedExecutor", Executor)
    monkeypatch.setattr(main.threading, "Thread", Thread)
    monkeypatch.setattr(main, "_start_ws_thread", lambda *args: events.append("ws-listen"))
    monkeypatch.setattr(main, "_start_registration", lambda *args: events.append("register"))
    monkeypatch.setattr(main, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(main.signal, "signal", lambda *args: None)
    if load_fails:
        with pytest.raises(RuntimeError, match="GPU model unavailable"):
            main.main()
        assert not {"http-listen", "ws-listen", "register"}.intersection(events)
    else:
        main.main()
        assert events.index("spin") < events.index("start")
        assert events.index("stop") < events.index("ws-listen")
        assert events.index("stop") < events.index("register")
        assert events.index("stop") < events.index("http-listen")
    assert events[-2:] == ["executor-shutdown", "ros-shutdown"]
