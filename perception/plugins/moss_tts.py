"""MOSS TTS adapter for the existing perception TTS tool and ROS publisher.

Original-model runtimes have separate RNG, codec and KV state while one
TensorRT graph pool per process owns the immutable engines. Each synthesis
leases one runtime and forwards its real codec callbacks as 16 kHz mono PCM16.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from pathlib import Path

from plugins.moss_assets import ensure_moss_assets
from plugins.moss_audio import MossPcm16Converter


log = logging.getLogger(__name__)
_DONE = object()


class _Cancelled(Exception):
    pass


class MossTTSAdapter:
    """TTSAdapter-compatible facade over independent in-process runtimes."""

    dry_run_text = "你好。"
    # Keep at most 1 s of PCM queued downstream. Including the browser's 200 ms
    # startup margin, this remains below its 1.5 s drop threshold. The generic
    # ROS publisher retains its cursor across consecutive utterances.
    transport_lead_seconds = 1.0

    def __init__(self, model_dir: str, *, speaker_id: int = 0,
                 language: str = "auto", instances: int = 1,
                 asset_url: str | None = None, output_dir: str | Path | None = None,
                 runtimes=None, lease_timeout_s: float = 10.0):
        if not 1 <= instances <= 3:
            raise ValueError("MOSS instances per process must be 1..3")
        self._model_dir = Path(model_dir).resolve()
        self._output_dir = Path(output_dir or self._model_dir.parent / "outputs").resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._lease_timeout_s = float(lease_timeout_s)
        self._config_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._active_requests = 0
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
                raise ValueError(f"MOSS needs {instances} distinct runtime objects")
            voices = ("Junhao", "Zhiming", "Weiguo", "Xiaoyu", "Yuewen", "Lingyu",
                      "Trump", "Ava", "Bella", "Adam", "Nathan", "Sakura", "Yui",
                      "Aoi", "Hina", "Mei")
        self._voices = tuple(voices)
        try:
            self.set_speaker_id(speaker_id)
        except (TypeError, ValueError):
            if self._pool is not None:
                self._pool.close()
            raise
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
        selected_plan = os.environ.get("MOSS_TRT_SELECTED_LOCAL_PLAN")
        selected_contract_path = os.environ.get("MOSS_TRT_SELECTED_LOCAL_CONTRACT")
        if bool(selected_plan) != bool(selected_contract_path):
            raise ValueError("Selected local engine needs both its plan and contract")
        selected_contract = None
        if selected_plan:
            from plugins.moss_deploy.selected_logits import load_selected_contract
            selected_contract = load_selected_contract(selected_plan, selected_contract_path, assets)
            engine_paths["local_cached_step"] = Path(selected_plan)
        frame_plan = os.environ.get("MOSS_TRT_FRAME_PLAN")
        frame_contract_path = os.environ.get("MOSS_TRT_FRAME_CONTRACT")
        if bool(frame_plan) != bool(frame_contract_path):
            raise ValueError("Fused frame engine needs both its plan and contract")
        if frame_plan and selected_plan:
            raise ValueError("Choose either the fused frame or selected local candidate")
        frame_contract = None
        if frame_plan:
            from plugins.moss_deploy.frame_contract import load_frame_contract
            frame_contract = load_frame_contract(frame_plan, frame_contract_path, assets)
        driver_tool = Path(__file__).resolve().parents[1] / "tools/moss_trt_session.py"
        # resolve_graph_plans checks missing Orin plans before importing TRT.
        # In particular, the portable ONNX tar alone must not appear runnable.
        optional = {"local_fixed_sampled_frame": Path(frame_plan)} if frame_plan else {}
        pool_kwargs = {"optional_engine_paths": optional} if optional else {}
        pool = TensorRTGraphPool(
            assets, device=0, engine_paths=engine_paths,
            driver_tool=driver_tool, encoder_context_mode="disabled",
            **pool_kwargs,
        )
        pool.selected_local_contract = selected_contract
        pool.frame_contract = frame_contract
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

    def validate_speaker_id(self, speaker_id: int) -> int:
        """Validate a cached built-in voice without changing any runtime state."""
        try:
            index = int(speaker_id)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"MOSS speaker_id must be an integer in 0..{len(self._voices) - 1}") from error
        if (isinstance(speaker_id, bool)
                or (not isinstance(speaker_id, str) and speaker_id != index)
                or not 0 <= index < len(self._voices)):
            raise ValueError(f"MOSS speaker_id must be an integer in 0..{len(self._voices) - 1}")
        return index

    def set_speaker_id(self, speaker_id: int) -> None:
        """Choose cached reference codes for subsequent requests; keep engines."""
        index = self.validate_speaker_id(speaker_id)
        with self._config_lock:
            self._voice = self._voices[index]

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
        # At startup this visits every in-process runtime in turn, materializing
        # its lazy TensorRT contexts before the first scored request.
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

    def _release_runtime(self, runtime) -> None:
        with self._lifecycle_lock:
            if runtime is not None:
                self._available.put_nowait(runtime)
            self._active_requests -= 1

    def synthesize_stream(self, text: str):
        # Count admitted callers before queue.get: the queue can still look
        # full while a caller is about to lease an engine during shutdown.
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MOSS adapter is closed")
            if not str(text).strip():
                return
            self._active_requests += 1
        runtime = None
        producer_started = False
        try:
            # A queued/admitted request keeps the configuration it started
            # with, even if the operator changes voice while it awaits a lease.
            with self._config_lock:
                language, voice = self._language, self._voice
            try:
                runtime = self._available.get(timeout=self._lease_timeout_s)
            except queue.Empty as error:
                raise RuntimeError("All MOSS inference instances in this process are busy") from error
            audio_queue: queue.Queue = queue.Queue(maxsize=32)
            cancelled = threading.Event()
            state: dict = {"pcm_bytes": 0}
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
                    state["pcm_bytes"] += len(pcm)

            def produce() -> None:
                try:
                    runtime.synthesize_stream(
                        on_audio=on_audio, text=text, voice=voice,
                        language=language, sample_mode="full", do_sample=True,
                        max_new_frames=375, voice_clone_max_text_tokens=75,
                        enable_wetext=True, output_audio_path=output_path,
                    )
                    if not state["pcm_bytes"]:
                        raise RuntimeError("MOSS synthesis returned no PCM for nonempty text")
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
                    # The producer owns this lease until model inference has
                    # returned, even if the consumer cancelled or join timed out.
                    self._release_runtime(runtime)
                    try:
                        offer(_DONE)
                    except _Cancelled:
                        pass

            producer = threading.Thread(target=produce, name="moss-tts-producer", daemon=True)
            producer.start()
            producer_started = True
            while True:
                item = audio_queue.get()
                if item is _DONE:
                    if "error" in state:
                        raise state["error"]
                    return
                yield item
        finally:
            if producer_started:
                cancelled.set()
                producer.join(timeout=5)
            else:
                # Includes queue timeouts and exceptions before Thread.start.
                self._release_runtime(runtime)

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._active_requests:
                raise RuntimeError("MOSS inference still active; refusing to free shared TensorRT pool")
            # Rejected closes leave admission open. Once teardown begins,
            # serialize concurrent closes and prohibit any new engine leases.
            self._closed = True
            if self._pool is not None:
                self._pool.close()
                self._pool = None
