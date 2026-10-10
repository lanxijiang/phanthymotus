"""Deterministic admission/teardown races without loading model engines."""

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.moss_tts import MossTTSAdapter
from test_moss_tts_adapter import FakeRuntime


def make_adapter(tmp_path, runtime=None, **kwargs):
    runtime = runtime or FakeRuntime("only")
    adapter = MossTTSAdapter(str(tmp_path / "assets"), runtimes=[runtime], **kwargs)
    closed = []
    adapter._pool = SimpleNamespace(close=lambda: closed.append(True))
    return adapter, runtime, closed


def test_admitted_request_blocks_close_before_queue_lease_and_rejection_is_reversible(tmp_path):
    adapter, runtime, closed = make_adapter(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    available = adapter._available

    class GatedQueue:
        def get(self, **kwargs):
            entered.set()
            assert proceed.wait(3)
            return available.get(**kwargs)

        def put_nowait(self, item):
            available.put_nowait(item)

        def qsize(self):
            return available.qsize()

    adapter._available = GatedQueue()
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(adapter.synthesize, "First.")
        try:
            assert entered.wait(3)
            assert available.qsize() == 1  # No engine has been taken yet.
            with pytest.raises(RuntimeError, match="still active"):
                adapter.close()
            assert not closed and not adapter._closed
        finally:
            proceed.set()
        assert future.result(timeout=3)
    assert adapter.synthesize("After rejected close.")
    assert len(runtime.calls) == 2
    adapter.close()
    with pytest.raises(RuntimeError, match="adapter is closed"):
        adapter.synthesize("After successful close.")
    adapter.close()
    assert closed == [True] and len(runtime.calls) == 2


def test_lease_timeout_releases_only_its_own_admission(tmp_path):
    entered, proceed = threading.Event(), threading.Event()

    class BlockedRuntime(FakeRuntime):
        def synthesize_stream(self, **kwargs):
            entered.set()
            assert proceed.wait(3)
            return super().synthesize_stream(**kwargs)

    adapter, _runtime, closed = make_adapter(tmp_path, BlockedRuntime("only"), lease_timeout_s=.01)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(adapter.synthesize, "In progress.")
        try:
            assert entered.wait(3)
            with pytest.raises(RuntimeError, match="instances in this process are busy"):
                adapter.synthesize("Waiting.")
            assert adapter._active_requests == 1
            with pytest.raises(RuntimeError, match="still active"):
                adapter.close()
            assert not closed and not adapter._closed
        finally:
            proceed.set()
        assert future.result(timeout=3)
    assert adapter._active_requests == 0
    assert adapter.synthesize("Retry.")
    adapter.close()
    assert closed == [True]


@pytest.mark.parametrize("phase", ["converter", "thread_start"])
def test_failure_before_producer_starts_returns_lease_and_admission(tmp_path, monkeypatch, phase):
    import plugins.moss_tts as module
    adapter, _runtime, closed = make_adapter(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("injected setup failure")

    with monkeypatch.context() as patch:
        if phase == "converter":
            patch.setattr(module, "MossPcm16Converter", fail)
        else:
            patch.setattr(threading.Thread, "start", fail)
        with pytest.raises(RuntimeError, match="injected setup failure"):
            adapter.synthesize("Failed setup.")
    assert adapter._active_requests == 0 and adapter._available.qsize() == 1
    assert adapter.synthesize("Retry.")
    adapter.close()
    assert closed == [True]


def test_cancelled_consumer_does_not_release_running_producer(tmp_path, monkeypatch):
    proceed = threading.Event()
    producer_threads = []

    class StreamingRuntime(FakeRuntime):
        def synthesize_stream(self, *, on_audio, **kwargs):
            producer_threads.append(threading.current_thread())
            on_audio(np.full((4800, 2), .25, dtype=np.float32))
            assert proceed.wait(3)

    adapter, _runtime, closed = make_adapter(tmp_path, StreamingRuntime("only"))
    stream = adapter.synthesize_stream("Cancel after first audio.")
    try:
        assert next(stream)
        # Emulate the bounded join expiring while GPU inference is still live.
        with monkeypatch.context() as patch:
            patch.setattr(producer_threads[0], "join", lambda timeout: None)
            stream.close()
        assert producer_threads[0].is_alive()
        assert adapter._active_requests == 1 and adapter._available.qsize() == 0
        with pytest.raises(RuntimeError, match="still active"):
            adapter.close()
        assert not closed and not adapter._closed
    finally:
        proceed.set()
        producer_threads[0].join(timeout=3)
    assert not producer_threads[0].is_alive()
    assert adapter._active_requests == 0 and adapter._available.qsize() == 1
    assert adapter.synthesize("Retry after producer completed.")
    adapter.close()
    assert closed == [True]


def test_concurrent_closes_run_teardown_once_and_reject_new_admission(tmp_path):
    adapter, runtime, _closed = make_adapter(tmp_path)
    teardown_entered, teardown_proceed = threading.Event(), threading.Event()
    close_calls = []

    def close_pool():
        close_calls.append(True)
        teardown_entered.set()
        assert teardown_proceed.wait(3)

    adapter._pool = SimpleNamespace(close=close_pool)
    with ThreadPoolExecutor(max_workers=3) as workers:
        first = workers.submit(adapter.close)
        assert teardown_entered.wait(3)
        try:
            request = workers.submit(adapter.synthesize, "Too late.")
            second = workers.submit(adapter.close)
            assert not runtime.calls
        finally:
            teardown_proceed.set()
        first.result(timeout=3)
        second.result(timeout=3)
        with pytest.raises(RuntimeError, match="adapter is closed"):
            request.result(timeout=3)
    assert close_calls == [True] and not runtime.calls
