"""MOSS TTS adapter for the existing perception TTS tool and ROS publisher.

Three original-model runtimes have separate RNG, codec and KV state while one
TensorRT graph pool owns the immutable engines. Each synthesis leases one
runtime and forwards its real codec callbacks as 16 kHz mono PCM16 bytes.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from plugins.moss_assets import ensure_moss_assets
from plugins.moss_audio import MossPcm16Converter


log = logging.getLogger(__name__)
_DONE = object()


class _Cancelled(Exception):
    pass


class MossTTSAdapter:
    """TTSAdapter-compatible facade over exactly three independent runtimes."""

    dry_run_text = "你好。"

    def __init__(self, model_dir: str, *, speaker_id: int = 0,
                 language: str = "auto", instances: int = 3,
                 asset_url: str | None = None, output_dir: str | Path | None = None,
                 runtimes=None, lease_timeout_s: float = 10.0):
        if instances != 3:
            raise ValueError("The MOSS deployment is configured for exactly three inference instances")
        self._model_dir = Path(model_dir).resolve()
        self._output_dir = Path(output_dir or self._model_dir.parent / "outputs").resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lease_timeout_s = float(lease_timeout_s)
        self._config_lock = threading.Lock()
        self._closed = False
        self._pool = None
        self._language = "auto"
        self.set_language(language)

        if runtimes is None:
            self._runtimes, voices, self._pool = self._load_runtimes(
                self._model_dir, asset_url or os.environ.get("MOSS_ASSET_URL"), instances
            )
        else:
            self._runtimes = tuple(runtimes)
            if len(self._runtimes) != instances or len({id(x) for x in self._runtimes}) != instances:
                raise ValueError("MOSS needs three distinct runtime objects")
            voices = ("Junhao", "Zhiming", "Weiguo", "Xiaoyu", "Yuewen", "Lingyu",
                      "Trump", "Ava", "Bella", "Adam", "Nathan", "Sakura", "Yui",
                      "Aoi", "Hina", "Mei")
        if not 0 <= int(speaker_id) < len(voices):
            if self._pool is not None:
                self._pool.close()
            raise ValueError(f"MOSS speaker_id must be 0..{len(voices) - 1}")
        self._voice = voices[int(speaker_id)]
        self._available: queue.Queue = queue.Queue(maxsize=instances)
        for runtime in self._runtimes:
            self._available.put_nowait(runtime)

    @staticmethod
    def _load_runtimes(model_dir: Path, asset_url: str | None, instances: int):
        root = ensure_moss_assets(model_dir, url=asset_url)
        from plugins.moss_deploy.assets import ModelAssets
        from plugins.moss_deploy.runtime import MossRuntime
        from plugins.moss_deploy.trt_pool import TensorRTGraphPool

        assets = ModelAssets(
            root / "analysis.json", root / "source/MOSS-TTS-Nano",
            streaming_codec_meta=root / "onnx/codec/codec_browser_onnx_meta.json",
        )
        mapping = json.loads((root / "engine-map.json").read_text(encoding="utf-8"))
        engine_paths = {name: root / value for name, value in mapping.items()}
        driver_tool = Path(__file__).resolve().parents[1] / "tools/moss_trt_session.py"
        # resolve_graph_plans checks missing Orin plans before importing TRT.
        # In particular, the portable ONNX tar alone must not appear runnable.
        pool = TensorRTGraphPool(
            assets, device=0, engine_paths=engine_paths,
            driver_tool=driver_tool, encoder_context_mode="disabled",
        )
        try:
            runtimes = tuple(MossRuntime(
                pool, instance_id=f"perception-moss-{index}", seed=1234 + index,
                output_dir=root.parent / "outputs",
            ) for index in range(instances))
            for runtime in runtimes:
                runtime.set_generation_defaults(
                    text_temperature=1.5, text_top_p=1.0, text_top_k=50,
                    audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
                    audio_repetition_penalty=1.0,
                )
            return runtimes, tuple(assets.voice_files), pool
        except BaseException:
            pool.close()
            raise

    def set_language(self, language: str) -> None:
        normalized = str(language or "auto").lower()
        if normalized not in {"auto", "zh", "ja", "ru", "en", "es", "fr", "th", "de"}:
            raise ValueError(f"Unsupported MOSS benchmark language: {language}")
        with self._config_lock:
            self._language = normalized

    def set_speed(self, speed: float) -> None:
        if float(speed) != 1.0:
            raise ValueError("MOSS has no validated speed control; use speed=1.0")

    def warmup(self) -> int:
        # Each completed synthesis returns its lease to the tail of _available.
        # At startup this visits all three runtimes in turn, materializing each
        # runtime's lazy TensorRT contexts before the first scored request.
        total_bytes = 0
        for index in range(len(self._runtimes)):
            try:
                audio = self.synthesize(self.dry_run_text)
            except Exception as error:
                raise RuntimeError(
                    f"MOSS warmup failed for instance {index + 1}/{len(self._runtimes)}"
                ) from error
            if not audio or not any(audio):
                raise RuntimeError(
                    f"MOSS warmup instance {index + 1}/{len(self._runtimes)} produced silence"
                )
            total_bytes += len(audio)
        return total_bytes

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        if self._closed:
            raise RuntimeError("MOSS adapter is closed")
        if not str(text).strip():
            return
        try:
            runtime = self._available.get(timeout=self._lease_timeout_s)
        except queue.Empty as error:
            raise RuntimeError("All three MOSS inference instances are busy") from error
        with self._config_lock:
            language = self._language
        audio_queue: queue.Queue = queue.Queue(maxsize=32)
        cancelled = threading.Event()
        state: dict = {}
        converter = MossPcm16Converter()
        output_path = self._output_dir / f"moss-{uuid.uuid4().hex}.wav"

        def offer(item) -> None:
            while not cancelled.is_set():
                try:
                    audio_queue.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise _Cancelled()

        def on_audio(stereo) -> None:
            if cancelled.is_set():
                raise _Cancelled()
            pcm = converter.process(stereo)
            if pcm:
                offer(pcm)

        def produce() -> None:
            try:
                runtime.synthesize_stream(
                    on_audio=on_audio, text=text, voice=self._voice,
                    language=language, sample_mode="full", do_sample=True,
                    max_new_frames=375, voice_clone_max_text_tokens=75,
                    enable_wetext=True, output_audio_path=output_path,
                )
            except _Cancelled:
                pass
            except BaseException as error:
                state["error"] = error
            finally:
                try:
                    output_path.unlink(missing_ok=True)
                except OSError:
                    log.warning("Could not remove MOSS output WAV %s", output_path,
                                exc_info=True)
                self._available.put(runtime)
                try:
                    offer(_DONE)
                except _Cancelled:
                    pass

        producer = threading.Thread(target=produce, name="moss-tts-producer", daemon=True)
        producer.start()
        try:
            while True:
                item = audio_queue.get()
                if item is _DONE:
                    if "error" in state:
                        raise state["error"]
                    return
                yield item
        finally:
            cancelled.set()
            producer.join(timeout=5)

    def close(self) -> None:
        self._closed = True
        if self._pool is None:
            return
        deadline = time.monotonic() + 5
        while self._available.qsize() < len(self._runtimes) and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._available.qsize() != len(self._runtimes):
            raise RuntimeError("MOSS inference still active; refusing to free shared TensorRT pool")
        self._pool.close()
        self._pool = None
