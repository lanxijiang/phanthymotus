"""Known incomplete generations must not pass through as successful speech."""

from types import SimpleNamespace

import numpy as np
import pytest

from plugins.moss_deploy import runtime
from plugins.moss_deploy.completeness import IncompleteSynthesisError, validate_chunk_result
from plugins.moss_tts import MossTTSAdapter
from test_moss_tts_adapter import FakeRuntime


@pytest.mark.parametrize("frames,samples,reason", [
    (0, 0, "empty_generation"), (375, 1440000, "frame_limit"),
    (376, 1443840, "frame_limit"), (1, 0, "empty_audio"),
])
def test_runtime_rejects_incomplete_chunk(monkeypatch, frames, samples, reason):
    monkeypatch.setenv("MOSS_FAST_SAMPLING", "0")
    result = {"generated_frames": [[1] * 16] * frames, "audio_samples": samples}
    original = type("Original", (), {"synthesize_single_chunk": lambda self, **kwargs: result})
    monkeypatch.setattr(runtime, "load_original_source", lambda _: (
        SimpleNamespace(OnnxTtsRuntime=original), SimpleNamespace()))
    pool = SimpleNamespace(assets=SimpleNamespace(source_dir=None))
    adapted = object.__new__(runtime._create_original_adapter_class(pool))
    adapted._request_audio_callback = None
    adapted._previous_stream_chunk_text = None
    adapted.manifest = {"generation_defaults": {"max_new_frames": 375}}
    with pytest.raises(IncompleteSynthesisError, match=reason):
        adapted.synthesize_single_chunk(text="Hello.", prompt_audio_codes=[], streaming=True)
    assert adapted._previous_stream_chunk_text is None


def test_normal_chunk_does_not_change_tokens_or_audio():
    frames, waveform = [[1] * 16], np.ones((3840, 2))
    result = {"generated_frames": frames, "waveform": waveform}
    validate_chunk_result(result, {"max_new_frames": 375})
    assert result["generated_frames"] is frames and result["waveform"] is waveform


@pytest.mark.parametrize("partial", [False, True])
def test_generation_error_is_propagated_and_runtime_lease_is_returned(tmp_path, partial):
    class FailedRuntime(FakeRuntime):
        def synthesize_stream(self, *, on_audio, **kwargs):
            if partial:
                on_audio(np.ones((4800, 2), dtype=np.float32))
            raise IncompleteSynthesisError("frame_limit", frames=375, limit=375)
    adapter = MossTTSAdapter(str(tmp_path / "assets"), runtimes=[FailedRuntime("failed")],
                            output_dir=tmp_path / "outputs")
    with pytest.raises(IncompleteSynthesisError):
        list(adapter.synthesize_stream("Hello."))
    assert adapter._available.qsize() == 1 and adapter._active_requests == 0
    adapter.close()


def test_adapter_refuses_no_pcm_even_if_runtime_returns_normally(tmp_path):
    class EmptyRuntime(FakeRuntime):
        def synthesize_stream(self, **kwargs):
            return {"language": kwargs["language"]}
    adapter = MossTTSAdapter(str(tmp_path / "assets"), runtimes=[EmptyRuntime("empty")],
                            output_dir=tmp_path / "outputs")
    with pytest.raises(RuntimeError, match="no PCM"):
        list(adapter.synthesize_stream("Hello."))
    assert adapter._available.qsize() == 1 and adapter._active_requests == 0
    adapter.close()
