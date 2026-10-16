"""In-process adapter for the Jetson VITS2 TensorRT runtime."""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .runtime.backends.trt_numpy_tts_engine import TensorRTNumpyTTSEngine


SAMPLE_RATE = 16000
CHUNK_BYTES = int(os.getenv("MIX_VITS_CHUNK_BYTES", "3200"))
PCM_FRAME_MS = CHUNK_BYTES * 1000.0 / (SAMPLE_RATE * 2)
MAX_CHUNK_TOKENS = int(os.getenv("MIX_VITS_MAX_TEXT_TOKENS", "256"))
CHUNK_PAUSE_MS = int(os.getenv("MIX_VITS_CHUNK_PAUSE_MS", "0"))
MODEL_CONFIG = os.getenv("MIX_VITS_CONFIG_PATH", "/models/vits2-mix/config.json")
ENGINE_DIR = os.getenv("MIX_VITS_TRT_ENGINE_DIR", "/models/vits2-mix/engines")
WARMUP_CASES = (
    "你好，语音服务已经准备好了。",
    "晚上会用FaceTime练习英语。",
    "Lucy喝完coffee后检查PPT，David同时记录GPS数据。"
)
log = logging.getLogger(__name__)

if CHUNK_BYTES <= 0 or CHUNK_BYTES % 2:
    raise ValueError("MIX_VITS_CHUNK_BYTES must be a positive even number")
if not 0 <= CHUNK_PAUSE_MS <= 1000:
    raise ValueError("MIX_VITS_CHUNK_PAUSE_MS must be between 0 and 1000")

class TTSAdapter(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)

    def warmup(self) -> int:
        return 0

    def set_speed(self, speed: float) -> None:
        del speed


class Vits2TensorRTAdapter(TTSAdapter):
    def __init__(
        self,
        speed: float = 1.0,
        *,
        engine=None,
        model_config: str = MODEL_CONFIG,
        engine_dir: str = ENGINE_DIR,
        max_chunk_tokens: int | None = None,
    ):
        self._lock = threading.Lock()
        self.set_speed(speed)
        self._engine = engine or TensorRTNumpyTTSEngine(model_config, engine_dir)
        self.max_chunk_tokens = int(
            MAX_CHUNK_TOKENS if max_chunk_tokens is None else max_chunk_tokens
        )
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        engine_limit = int(getattr(self._engine, "max_text_tokens", self.max_chunk_tokens))
        if self.max_chunk_tokens > engine_limit:
            raise ValueError(
                f"adapter text limit {self.max_chunk_tokens} exceeds engine limit "
                f"{engine_limit}"
            )

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if speed <= 0 or speed > 4:
            raise ValueError("TTS speed must be greater than zero and at most four")
        with self._lock:
            self._speed = speed

    def _token_count(self, text: str) -> int:
        return len(self._engine._get_text_ids(text, normalized=True)[0])

    def iter_text_chunks(self, text: str):
        """Yield the exact normalized text chunks and IDs used in production."""
        text = text.strip()
        if not text:
            raise ValueError("TTS text must not be empty")
        from .frontend.cleaner import normalize_text_mix
        from .frontend.chunking import (
            iter_text_chunks as _shared_iter_text_chunks,
        )

        for chunk in _shared_iter_text_chunks(
            normalize_text_mix(text), self._token_count, self.max_chunk_tokens
        ):
            text_ids = self._engine._get_text_ids(chunk, normalized=True)
            yield chunk, text_ids

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        pause_samples = SAMPLE_RATE * CHUNK_PAUSE_MS // 1000
        silence = b"\x00\x00" * pause_samples
        with self._lock:
            # One deterministic latent stream per complete request. Resetting for
            # every text chunk would repeat the same noise prefix at boundaries.
            reset_random_state = getattr(self._engine, "reset_random_state", None)
            if reset_random_state is not None:
                reset_random_state()
            chunks = list(self.iter_text_chunks(text))

            def synthesize_chunk(item):
                chunk, text_ids = item
                return self._engine.synthesize(
                    chunk,
                    text_ids=text_ids,
                    length_scale=1.0 / self._speed,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(synthesize_chunk, chunks[0])
                for chunk_index, (chunk, text_ids) in enumerate(chunks):
                    pcm = pending.result()
                    if chunk_index + 1 < len(chunks):
                        pending = executor.submit(
                            synthesize_chunk, chunks[chunk_index + 1]
                        )
                    token_count = len(text_ids[0])
                    log.info(
                        "text redacted: chars=%d chunk=%d tokens=%d",
                        len(text.strip()),
                        chunk_index,
                        token_count,
                    )
                    if chunk_index and silence:
                        yield silence
                    for offset in range(0, len(pcm), CHUNK_BYTES):
                        yield pcm[offset : offset + CHUNK_BYTES]

    def warmup(self) -> int:
        warmup_bytes = 0
        for text in WARMUP_CASES:
            case_bytes = sum(len(pcm) for pcm in self.synthesize_stream(text))
            if not case_bytes:
                raise RuntimeError("TensorRT warmup produced no audio")
            warmup_bytes += case_bytes
        return warmup_bytes


class Vits2OnnxCpuAdapter(TTSAdapter):
    """ONNX Runtime CPU backend — low-memory / no-TRT fallback.

    Uses the same shared frontend and chunking as the TensorRT backend, so chunk
    boundaries (and thus synthesized audio) are identical across backends.
    """

    def __init__(
        self,
        model_dir: str,
        speed: float = 1.0,
        num_threads: int = 1,
        max_chunk_tokens: int = 64,
    ):
        # Avoid loading Torch in the normal TensorRT process. The fallback's
        # waveform post-processing is imported only if ONNX is selected.
        from .runtime.backends.onnx import OnnxCpuEngine

        if speed <= 0:
            raise ValueError("TTS speed must be greater than zero")
        root = Path(model_dir)
        os.environ.setdefault("NLTK_DATA", str(root / "nltk_data"))
        os.environ.setdefault("EN_TN_CACHE_DIR", str(root / "tn_cache"))
        os.environ.setdefault("TN_CACHE_DIR", str(root / "tn_cache"))
        os.environ.setdefault("VITS2_FRONTEND_DATA_DIR", str(root / "frontend_data"))
        self._engine = OnnxCpuEngine(
            config_path=root / "config.json",
            model_dir=root / "onnx",
            num_threads=num_threads,
        )
        if self._engine.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"VITS2 sample rate must be {SAMPLE_RATE}, got {self._engine.sample_rate}"
            )
        self._max_chunk_tokens = max(
            16, min(int(max_chunk_tokens), self._engine.max_text_tokens)
        )
        self._length_scale = 1.0 / speed
        self._lock = threading.Lock()

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if speed <= 0 or speed > 4:
            raise ValueError("TTS speed must be greater than zero and at most four")
        with self._lock:
            self._length_scale = 1.0 / speed

    def _token_count(self, text: str) -> int:
        return self._engine.text_token_count(text)

    def iter_text_chunks(self, text: str):
        """Yield the exact normalized text chunks used by the ONNX backend."""
        text = text.strip()
        if not text:
            raise ValueError("TTS text must not be empty")
        from .frontend.cleaner import normalize_text_mix
        from .frontend.chunking import (
            iter_text_chunks as _shared_iter_text_chunks,
        )

        yield from _shared_iter_text_chunks(
            normalize_text_mix(text), self._token_count, self._max_chunk_tokens
        )

    def synthesize_stream(self, text: str):
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        pause_samples = SAMPLE_RATE * CHUNK_PAUSE_MS // 1000
        silence = b"\x00\x00" * pause_samples
        with self._lock:
            chunks = list(self.iter_text_chunks(text))
            for chunk_index, chunk in enumerate(chunks):
                if chunk_index and silence:
                    yield silence
                pcm = self._engine.synthesize(
                    chunk, length_scale=self._length_scale
                )
                for offset in range(0, len(pcm), CHUNK_BYTES):
                    yield pcm[offset : offset + CHUNK_BYTES]

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def warmup(self) -> int:
        with self._lock:
            pcm = self._engine.synthesize("你好。", length_scale=self._length_scale)
        if not pcm:
            raise RuntimeError("VITS2 warmup produced no audio")
        return len(pcm)


def _onnx_adapter(cfg: dict) -> Vits2OnnxCpuAdapter:
    root = Path(cfg.get("vits2_model_dir", "/models/vits2-mix"))
    _configure_frontend_paths(root)
    return Vits2OnnxCpuAdapter(
        model_dir=str(root),
        speed=float(cfg.get("speed", 1.0)),
        num_threads=max(1, int(cfg.get("vits2_num_threads", 1))),
        max_chunk_tokens=int(cfg.get("vits2_max_chunk_tokens", 64)),
    )


def _trt_adapter(cfg: dict) -> Vits2TensorRTAdapter:
    root = Path(cfg.get("vits2_model_dir", "/models/vits2-mix"))
    _configure_frontend_paths(root)
    return Vits2TensorRTAdapter(
        speed=float(cfg.get("speed", 1.0)),
        model_config=str(root / "config.json"),
        engine_dir=str(root / "engines"),
        max_chunk_tokens=int(cfg.get("vits2_max_chunk_tokens", MAX_CHUNK_TOKENS)),
    )


def _configure_frontend_paths(root: Path) -> None:
    """Configure frontend assets from the verified VITS2 release root."""
    from .frontend.release_paths import configure_release_paths

    configure_release_paths(root)


def build_adapter(cfg: dict) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    backend = str(cfg.get("backend", "auto")).lower()
    if backend not in {"auto", "trt", "onnx"}:
        raise ValueError("backend must be one of: auto, trt, onnx")
    if backend == "trt":
        return _trt_adapter(cfg)
    if backend == "onnx":
        return _onnx_adapter(cfg)
    # auto: prefer TensorRT on Jetson GPU, fall back to ONNX CPU if unavailable.
    try:
        return _trt_adapter(cfg)
    except Exception as exc:
        log.warning("TensorRT backend unavailable (%s); falling back to ONNX CPU", exc)
        return _onnx_adapter(cfg)
