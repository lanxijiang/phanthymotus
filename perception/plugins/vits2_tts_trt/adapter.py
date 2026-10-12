"""In-process adapter for the Jetson VITS2 TensorRT runtime."""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .runtime.backends.trt_numpy_tts_engine import TensorRTNumpyTTSEngine
from .frontend.pronunciation_cache import PronunciationCache, use_pronunciation_cache


log = logging.getLogger(__name__)


def _env_int(name: str, default: int, low: int, high: int, even: bool = False) -> int:
    """Read a bounded int from the environment, falling back on bad input.

    Same reasoning as plugin._env_int: this module is imported while the plugin
    list is built, so a bad tuning variable must not abort the whole perception
    process.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("[vits2_tts_trt] %s=%r is not an integer; using %d", name, raw, default)
        return default
    if not low <= value <= high or (even and value % 2):
        log.warning("[vits2_tts_trt] %s=%d is unusable; using %d", name, value, default)
        return default
    return value


SAMPLE_RATE = 16000
# PCM frame size on the wire. Must stay even (16-bit samples) and match what
# the Speaker side expects; 3200 bytes is 100 ms at 16 kHz mono.
CHUNK_BYTES = _env_int("MIX_VITS_CHUNK_BYTES", 3200, 2, 1 << 20, even=True)
PCM_FRAME_MS = CHUNK_BYTES * 1000.0 / (SAMPLE_RATE * 2)
MAX_CHUNK_TOKENS = _env_int("MIX_VITS_MAX_TEXT_TOKENS", 256, 1, 4096)
CHUNK_PAUSE_MS = _env_int("MIX_VITS_CHUNK_PAUSE_MS", 0, 0, 1000)
WARMUP_CASES = (
    "你好，语音服务已经准备好了。",
    "今天可以使用平板电脑查看消息。",
    # Exercise the dictionary-miss path as well as ordinary English. A warmup
    # containing only CMU words leaves the NLTK tagger and G2P predictor cold.
    "The phonemizer is ready for a short test.",
    "语音 phonemizer 已经准备好了。",
)

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
        model_config: str = "",
        engine_dir: str = "",
        max_chunk_tokens: int | None = None,
        first_sentence: bool | None = None,
    ):
        # Both paths come from the installed release (see
        # utils.model_downloader.ensure_vits2_model), never from an env default:
        # engines are per JetPack family, so a hardcoded fallback would happily
        # hand TensorRT 10 a plan built by TensorRT 8.
        if engine is None and not (model_config and engine_dir):
            raise ValueError("model_config and engine_dir are required")
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
        # Opt-in only: changing utterance boundaries can change prosody and
        # latent sampling, so this candidate requires listening comparison.
        self.first_sentence = bool(
            _env_int("MIX_VITS_FIRST_SENTENCE", 0, 0, 1)
            if first_sentence is None else first_sentence
        )

    def set_speed(self, speed: float) -> None:
        speed = float(speed)
        if speed <= 0 or speed > 4:
            raise ValueError("TTS speed must be greater than zero and at most four")
        with self._lock:
            self._speed = speed

    def iter_text_chunks(self, text: str):
        """Yield the exact normalized text chunks and IDs used in production."""
        text = text.strip()
        if not text:
            raise ValueError("TTS text must not be empty")
        from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix
        from plugins.vits2_tts_trt.frontend.chunking import (
            iter_text_chunks as _frontend_iter_text_chunks,
        )

        # Chunk packing asks about a unit more than once. Reuse exactly those
        # IDs for synthesis so a dictionary miss does not run G2P repeatedly.
        # The cache belongs to this invocation, never to a later request.
        text_ids_by_chunk = {}
        pronunciations = PronunciationCache()

        def text_ids_for(chunk):
            if chunk not in text_ids_by_chunk:
                # Overlapping candidates contain the same OOV words. Reuse
                # only their exact predictor output; keep all contextual
                # normalization, tokenization, tones and phone conversion.
                # Exit the context before yielding to a caller.
                with use_pronunciation_cache(pronunciations):
                    text_ids_by_chunk[chunk] = self._engine._get_text_ids(
                        chunk, normalized=True
                    )
            return text_ids_by_chunk[chunk]

        def token_count(chunk):
            return len(text_ids_for(chunk)[0])

        for chunk in _frontend_iter_text_chunks(
            normalize_text_mix(text), token_count, self.max_chunk_tokens,
            first_sentence=self.first_sentence,
        ):
            yield chunk, text_ids_for(chunk)

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        pause_samples = SAMPLE_RATE * CHUNK_PAUSE_MS // 1000
        silence = b"\x00\x00" * pause_samples
        request_started = time.perf_counter()
        with self._lock:
            lock_wait_ms = (time.perf_counter() - request_started) * 1000.0
            # One deterministic latent stream per complete request. Resetting for
            # every text chunk would repeat the same noise prefix at boundaries.
            reset_random_state = getattr(self._engine, "reset_random_state", None)
            if reset_random_state is not None:
                reset_random_state()
            frontend_started = time.perf_counter()
            for chunk_index, (chunk, text_ids) in enumerate(
                self.iter_text_chunks(text)
            ):
                frontend_ms = (time.perf_counter() - frontend_started) * 1000.0
                token_count = len(text_ids[0])
                if chunk_index and silence:
                    yield silence
                engine_started = time.perf_counter()
                pcm = self._engine.synthesize(
                    chunk,
                    text_ids=text_ids,
                    length_scale=1.0 / self._speed,
                )
                log.info(
                    "[vits2_tts_trt] chunk synthesis: chars=%d chunk=%d tokens=%d "
                    "frontend_ms=%.1f engine_ms=%.1f pcm_bytes=%d",
                    len(text.strip()), chunk_index, token_count, frontend_ms,
                    (time.perf_counter() - engine_started) * 1000.0, len(pcm),
                )
                if chunk_index == 0:
                    log.info(
                        "[vits2_tts_trt] first PCM ready: lock_wait_ms=%.1f "
                        "elapsed_ms=%.1f",
                        lock_wait_ms, (time.perf_counter() - request_started) * 1000.0,
                    )
                for offset in range(0, len(pcm), CHUNK_BYTES):
                    yield pcm[offset : offset + CHUNK_BYTES]
                # Consumer backpressure between yields is not frontend work.
                frontend_started = time.perf_counter()

    def warmup(self) -> int:
        warmup_bytes = 0
        for text in WARMUP_CASES:
            case_bytes = sum(len(pcm) for pcm in self.synthesize_stream(text))
            if not case_bytes:
                raise RuntimeError("TensorRT warmup produced no audio")
            warmup_bytes += case_bytes
        return warmup_bytes


def _trt_adapter(cfg: dict) -> Vits2TensorRTAdapter:
    root = Path(cfg.get("model_dir", "/models/vits2"))
    _configure_frontend_paths(root)
    return Vits2TensorRTAdapter(
        speed=float(cfg.get("speed", 1.0)),
        model_config=str(root / "config.json"),
        engine_dir=cfg["engine_dir"],
        max_chunk_tokens=int(cfg.get("max_chunk_tokens", MAX_CHUNK_TOKENS)),
    )


def _configure_frontend_paths(root: Path) -> None:
    """Point inference at the release-controlled frontend data."""
    from .frontend.release_paths import configure_release_paths

    configure_release_paths(root)


def build_adapter(cfg: dict) -> TTSAdapter:
    speaker_id = int(cfg.get("speaker_id", 0))
    if speaker_id != 0:
        raise ValueError("The VITS2 model supports only speaker_id=0")
    backend = str(cfg.get("backend", "trt")).lower()
    if backend != "trt":
        raise ValueError("The JP6 VITS2 plugin supports backend=trt only")
    return _trt_adapter(cfg)
