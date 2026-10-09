"""Offline tests for the MOSS PCM bridge and three isolated runtime leases."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plugins.moss_audio import MossPcm16Converter
from plugins.moss_tts import MossTTSAdapter
from plugins.moss_deploy.trt_pool import resolve_graph_plans


def test_48k_stereo_to_16k_pcm_is_independent_of_callback_boundaries():
    samples = np.arange(4801, dtype=np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 1000 * samples / 48000)
    stereo = np.stack([tone, tone], axis=1)
    expected = MossPcm16Converter().process(stereo)
    converter = MossPcm16Converter()
    cuts = (1, 3, 14, 15, 93, 94, 207, 391, 4801)
    chunks = []
    start = 0
    for stop in cuts:
        chunks.append(converter.process(stereo[start:stop]))
        start = stop
    assert b"".join(chunks) == expected
    assert len(expected) == 2 * 1601
    assert np.max(np.abs(np.frombuffer(expected, dtype="<i2"))) > 10000


def test_stereo_downmix_and_nonfinite_rejection():
    tone = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    opposite = np.stack([tone, -tone], axis=1)
    assert not np.frombuffer(MossPcm16Converter().process(opposite), dtype="<i2").any()
    opposite[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        MossPcm16Converter().process(opposite)


class FakeRuntime:
    def __init__(self, instance_id: str, barrier: threading.Barrier | None = None):
        self.instance_id = instance_id
        self.barrier = barrier
        self.calls = []

    def synthesize_stream(self, *, on_audio, output_audio_path, **kwargs):
        self.calls.append(kwargs)
        Path(output_audio_path).write_bytes(b"fake original wav")
        if self.barrier:
            self.barrier.wait(timeout=5)
        tone = np.full((4800, 2), 0.25, dtype=np.float32)
        on_audio(tone[:1301])
        on_audio(tone[1301:])
        return {"language": kwargs["language"]}


def test_adapter_consumes_codec_callbacks_as_pcm_and_removes_original_wav(tmp_path):
    runtimes = [FakeRuntime(str(i)) for i in range(3)]
    adapter = MossTTSAdapter(
        str(tmp_path / "assets"), output_dir=tmp_path / "outputs",
        speaker_id=5, language="zh", runtimes=runtimes,
    )
    chunks = list(adapter.synthesize_stream("你好。"))
    assert len(chunks) == 2
    assert len(b"".join(chunks)) == 3200  # 100 ms of 16 kHz mono PCM16
    assert runtimes[0].calls[0]["voice"] == "Lingyu"
    assert runtimes[0].calls[0]["language"] == "zh"
    assert list((tmp_path / "outputs").iterdir()) == []
    adapter.close()


def test_three_concurrent_requests_use_three_distinct_runtime_states(tmp_path):
    barrier = threading.Barrier(3)
    runtimes = [FakeRuntime(str(i), barrier) for i in range(3)]
    adapter = MossTTSAdapter(
        str(tmp_path / "assets"), output_dir=tmp_path / "outputs",
        runtimes=runtimes,
    )
    with ThreadPoolExecutor(max_workers=3) as workers:
        audio = list(workers.map(lambda _: adapter.synthesize("Hello."), range(3)))
    assert all(len(row) == 3200 for row in audio)
    assert [len(runtime.calls) for runtime in runtimes] == [1, 1, 1]
    assert list((tmp_path / "outputs").iterdir()) == []
    adapter.close()


def test_warmup_visits_all_three_runtimes_in_order(tmp_path):
    runtimes = [FakeRuntime(str(index)) for index in range(3)]
    adapter = MossTTSAdapter(
        str(tmp_path / "assets"), output_dir=tmp_path / "outputs", runtimes=runtimes,
    )
    assert adapter.warmup() == 3 * 3200
    assert [len(runtime.calls) for runtime in runtimes] == [1, 1, 1]
    assert all(runtime.calls[0]["text"] == adapter.dry_run_text for runtime in runtimes)
    assert list((tmp_path / "outputs").iterdir()) == []
    adapter.close()


def test_warmup_reports_silent_and_failed_runtime(tmp_path):
    class SilentRuntime(FakeRuntime):
        def synthesize_stream(self, *, on_audio, output_audio_path, **kwargs):
            self.calls.append(kwargs)
            on_audio(np.zeros((4800, 2), dtype=np.float32))

    class FailedRuntime(FakeRuntime):
        def synthesize_stream(self, *, on_audio, output_audio_path, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("codec failed")

    for runtime, message in ((SilentRuntime("silent"), "produced silence"),
                             (FailedRuntime("failed"), "MOSS warmup failed")):
        runtimes = [runtime, FakeRuntime("1"), FakeRuntime("2")]
        adapter = MossTTSAdapter(
            str(tmp_path / "assets"), output_dir=tmp_path / "outputs", runtimes=runtimes,
        )
        with pytest.raises(RuntimeError, match=message):
            adapter.warmup()
        assert [len(item.calls) for item in runtimes] == [1, 0, 0]
        adapter.close()


def test_missing_orin_plan_fails_before_loading_cuda(tmp_path):
    assets = SimpleNamespace(
        graph_paths={"global_unified": tmp_path / "global_unified.onnx"},
        precomputed_reference_codes={"Junhao": ()},
        voice_paths={"Junhao": tmp_path / "junhao.wav"},
        streaming_path=None,
    )
    with pytest.raises(FileNotFoundError, match="Missing original MOSS TensorRT plan"):
        resolve_graph_plans(
            assets, None, {"global_unified": tmp_path / "plans/global_unified.plan"},
            "disabled",
        )


def test_unconfigured_asset_url_fails_without_touching_network(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOSS_ASSET_URL"):
        MossTTSAdapter(str(tmp_path / "assets"), output_dir=tmp_path / "outputs")
