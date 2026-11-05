"""Kokoro Multi FP32 TensorRT adapter; dictionaries and voices stay external."""

import json
import logging
from pathlib import Path
import re
import threading
from time import monotonic

import numpy as np

from plugins.tts import TTSAdapter
from utils.resample import downsample_24k_to_16k
from .kokoro_multi_frontend import Frontend, LANGUAGES, phoneme_chunks
from .kokoro_multi_trt import KokoroTRT, SequenceTooLong

log = logging.getLogger(__name__)


class KokoroMultiTTSAdapter(TTSAdapter):
    LANGUAGES = LANGUAGES
    dry_run_text = "Hi."
    handles_text_chunking = True

    def __init__(self, model_dir, speaker_id=0, speed=1.0, device="gpu",
                 language="auto", on_status=None):
        if device != "gpu":
            raise ValueError("Kokoro Multi TensorRT requires GPU")
        if speaker_id != 0:
            raise ValueError("Kokoro Multi selects its release voice per language; speaker_id must be 0")
        self._lock = threading.RLock()
        self._runtime = None
        self.set_speed(speed)
        self.set_language(language)
        root = Path(model_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["sample_rate"] != 24000 or manifest["precision"] != "float32":
            raise ValueError("Kokoro Multi requires its 24kHz FP32 release")
        self._vocab = json.loads((root / "vocab.json").read_text())
        self._voices = {lang: np.load(root / manifest["voices"][lang], allow_pickle=False)
                        for lang in LANGUAGES}
        if any(v.shape != (510, 1, 256) or v.dtype != np.float32
               or not np.isfinite(v).all() for v in self._voices.values()):
            raise ValueError("Kokoro Multi voice packs must be finite float32 [510,1,256]")
        if on_status:
            on_status("Loading eight-language frontend")
        self._frontend = Frontend(root)
        if on_status:
            on_status("Loading FP32 TensorRT")
        self._runtime = KokoroTRT(root / "encoder.plan", root / "acoustic.plan",
                                 vocoder_plan=root / "vocoder.plan")

    def set_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or not 0.5 <= speed <= 2.0:
            raise ValueError("Kokoro Multi speed must be between 0.5 and 2.0")
        with self._lock:
            self._speed = speed

    def set_language(self, language):
        language = str(language).lower().replace("_", "-").split("-")[0]
        if language not in ("auto", *LANGUAGES):
            raise ValueError(f"Unsupported Kokoro Multi language: {language}")
        with self._lock:
            self._language = language

    def synthesize(self, text):
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text):
        started = monotonic()
        with self._lock:
            language = (self._frontend.detect_language(text)
                        if self._language == "auto" else self._language)
            phones = self._frontend.convert(text, language)
            phones = re.sub(r"\s+", " ", phones).strip()
            speed = np.array([self._speed], dtype=np.float32)
        unknown = set(phones) - self._vocab.keys()
        decorations = {char for char in unknown if char in text and (
            0x1F000 <= ord(char) <= 0x1FAFF or char in "\u200d\ufe0e\ufe0f")}
        if decorations:
            log.info("[kokoro-multi] ignoring unsupported decorations: %s", sorted(decorations))
            phones = "".join(char for char in phones if char not in decorations)
            unknown -= decorations
        if unknown:
            raise ValueError(f"Kokoro Multi unsupported phonemes ({language}): {sorted(unknown)}")
        chunks = phoneme_chunks(phones)
        samples, count = 0, 0
        while chunks:
            part = chunks.pop(0)
            ids = [self._vocab[c] for c in part]
            tokens = np.array([[0, *ids, 0]], dtype=np.int64)
            style = self._voices[language][len(ids) - 1]
            try:
                audio, duration = self._runtime.infer(tokens, style, speed)
            except SequenceTooLong:
                # Nothing from this chunk has been published. Retry smaller
                # whole-word chunks; never truncate audio or repeat old PCM.
                limit = len(part) // 2
                if limit < 1:
                    raise
                chunks[:0] = phoneme_chunks(part, limit=limit,
                                           preferred=limit, first_limit=0)
                continue
            audio = audio.reshape(-1)
            if audio.size != 600 * int(duration.sum()):
                raise RuntimeError("Kokoro Multi output length differs from predicted duration")
            pcm = downsample_24k_to_16k(audio)
            if count == 0:
                log.info("[kokoro-multi] first_pcm language=%s seconds=%.3f", language,
                         monotonic() - started)
            samples += audio.size
            count += 1
            yield pcm
        log.info("[kokoro-multi] complete language=%s chunks=%d audio_s=%.3f wall_s=%.3f",
                 language, count, samples / 24000, monotonic() - started)

    def close(self):
        if self._runtime is not None:
            self._runtime.close()
