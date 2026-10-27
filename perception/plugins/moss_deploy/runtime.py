"""Glue for original MOSS prompting/sampling and the validated unified graphs."""

from __future__ import annotations

import copy
import logging
import os
import threading
from functools import wraps
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Callable

import numpy as np
import sentencepiece as spm

from .assets import load_original_prompting, load_original_source
from .chunking import prompt_text_capacity, split_preserving_boundaries
from .thai import thai_word_breaks
from .completeness import (generate_with_early_eos_guard, minimum_chunk_frames,
                           validate_chunk_result)
from .si_units import prepare_with_request_language, resolve_request_language
from .pool import SessionAdapter, SharedGraphPool
from .selected_logits import SelectedAudioScores, wrap_selected_outputs


log = logging.getLogger(__name__)


GENERATION_DEFAULTS = {
    "max_new_frames": 375,
    "do_sample": True,
    "sample_mode": "full",
    "text_temperature": 1.0,
    "text_top_p": 1.0,
    "text_top_k": 50,
    "audio_temperature": 0.8,
    "audio_top_p": 0.95,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.2,
}


_FRAME_SAMPLING_KEYS = (
    "text_temperature", "text_top_p", "text_top_k", "audio_temperature",
    "audio_top_p", "audio_top_k", "audio_repetition_penalty",
)


def _validate_frame_sampling(defaults, contract):
    expected = contract["generation_defaults"]
    mismatches = [key for key in _FRAME_SAMPLING_KEYS
                  if key not in expected or defaults.get(key) != expected[key]]
    if defaults.get("sample_mode") != "full" or defaults.get("do_sample") is not True:
        mismatches.append("sample_mode=full/do_sample=True")
    if mismatches:
        raise ValueError("Fused frame sampling does not match its validated contract: "
                         + ", ".join(mismatches))


def _make_frame_generation_method(original_method, contract):
    """Select the original fused branch without changing public full sampling.

    The closure owns its globals: source modules, other runtimes and their
    sampling-mode constants retain their original values.
    """
    if not isinstance(original_method, FunctionType):
        raise TypeError("Expected the original Python MOSS generation method")
    namespace = dict(original_method.__globals__)
    if namespace.get("SAMPLE_MODE_FIXED") != "fixed":
        raise ValueError("Original MOSS fixed-frame generation branch has changed")
    namespace["SAMPLE_MODE_FIXED"] = "full"
    generate = FunctionType(original_method.__code__, namespace,
                            original_method.__name__, original_method.__defaults__,
                            original_method.__closure__)
    generate.__kwdefaults__ = original_method.__kwdefaults__

    def generate_frames(self, request_rows, on_frame=None):
        _validate_frame_sampling(self.manifest["generation_defaults"], contract)
        if "local_fixed_sampled_frame" not in self.sessions:
            raise RuntimeError("Fused frame contract has no matching session")
        return generate(self, request_rows, on_frame=on_frame)

    return generate_frames


def _run_sampled_frame(runtime, global_hidden, *, previous_token_sets_by_channel):
    """Run one frame; consume only the assistant draw when generation stops."""
    n_vq = int(runtime.manifest["tts_config"]["n_vq"])
    if n_vq != 16 or len(previous_token_sets_by_channel) != n_vq:
        raise ValueError("Fused frame requires the original 16 audio codebooks")
    hidden = np.asarray(global_hidden, dtype=np.float32)
    if hidden.shape != (1, 768):
        raise ValueError("Fused frame global_hidden must have shape [1,768]")
    # Repetition penalties apply to each channel's generated history, never
    # to reference codes or another request. Build a fresh mask per frame so
    # neither the caller's sets nor a previously submitted feed can change.
    repetition_seen_mask = np.zeros((1, n_vq, 1024), dtype=np.float32)
    for channel, previous_tokens in enumerate(previous_token_sets_by_channel):
        for token in previous_tokens:
            if (isinstance(token, (bool, np.bool_))
                    or not isinstance(token, (int, np.integer))
                    or not 0 <= int(token) < 1024):
                raise ValueError("Fused frame history requires integer audio tokens in 0..1023")
            repetition_seen_mask[0, channel, int(token)] = 1.0
    initial_rng_state = copy.deepcopy(runtime.rng.bit_generator.state)
    assistant_u = float(runtime.rng.random())
    after_assistant = copy.deepcopy(runtime.rng.bit_generator.state)
    draws = [assistant_u] + [float(runtime.rng.random()) for _ in range(n_vq)]
    random_u = np.clip(np.asarray([draws], dtype=np.float32), 0.0, 0.99999994)
    try:
        session = runtime.sessions["local_fixed_sampled_frame"]
        outputs = session.run(None, {"global_hidden": hidden, "random_u": random_u,
                                     "repetition_seen_mask": repetition_seen_mask})
        names = [output.name for output in session.get_outputs()]
        named = dict(zip(names, outputs, strict=True))
        continuation = np.asarray(named["should_continue"])
        tokens = np.asarray(named["frame_token_ids"])
        if (continuation.shape != (1, 1) or tokens.shape != (1, n_vq)
                or not np.issubdtype(continuation.dtype, np.integer)
                or not np.issubdtype(tokens.dtype, np.integer)):
            raise ValueError("Fused frame returned invalid integer output shapes or dtypes")
        if not np.isin(continuation, [0, 1]).all() or np.any(tokens < 0) or np.any(tokens >= 1024):
            raise ValueError("Fused frame returned out-of-range continuation or audio tokens")
        should_continue = bool(continuation[0, 0])
        if not should_continue:
            runtime.rng.bit_generator.state = after_assistant
        return should_continue, [int(token) for token in tokens[0]]
    except BaseException:
        runtime.rng.bit_generator.state = initial_rng_state
        raise


def _require_full_codec_decoder(pool: SharedGraphPool) -> None:
    reason = getattr(pool, "omitted_graphs", {}).get("codec_decoder")
    if reason:
        raise FileNotFoundError(
            f"Original full codec_decoder TensorRT plan is unavailable: {reason}. "
            "Use streaming=True or synthesize_stream() with the original codec_decode_step plan, "
            "or build codec_decoder.plan for explicit full decoding."
        )


class _ObservedStreamingCodec:
    def __init__(self, session):
        self.session = session
        self.on_audio = None
    def reset(self):
        self.session.reset()
    def run_frames(self, frame_rows):
        result = self.session.run_frames(frame_rows)
        if result is not None and self.on_audio is not None:
            audio, length = result
            if length:
                self.on_audio(np.asarray(audio[0, :, :length], dtype=np.float32).T.copy())
        return result


def _create_original_adapter_class(pool: SharedGraphPool):
    assets = pool.assets
    original, original_ort = load_original_source(assets.source_dir)
    use_fast_sampling = os.environ.get("MOSS_FAST_SAMPLING", "1") == "1"
    frame_contract = getattr(pool, "frame_contract", None)

    class AdaptedMossRuntime(original.OnnxTtsRuntime):
        fast_sampling_enabled = use_fast_sampling
        fused_frame_enabled = frame_contract is not None

        def __init__(self, instance_id: str, seed: int, output_dir: Path):
            # Intentionally bypass the upstream constructor's browser asset
            # downloader and its incompatible split graph session factory.
            self.model_dir = assets.analysis_path.parent
            self.manifest_path = assets.analysis_path
            self.manifest_dir = assets.analysis_path.parent
            self.tts_meta_path = assets.tts_meta_path
            self.tts_meta = copy.deepcopy(assets.tts_meta)
            self.codec_meta = copy.deepcopy(assets.codec_meta)
            self.execution_provider = pool.backend
            self.output_dir = output_dir.resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.sp_model = spm.SentencePieceProcessor(model_file=str(assets.tokenizer_path))
            self._text_normalizer_manager = None
            self._request_normalization_language = None
            self.rng = np.random.default_rng(seed)
            self.client = pool.create_client(instance_id)
            self.instance_id = instance_id
            prompting = load_original_prompting(assets.prompting_path)
            config = SimpleNamespace(**assets.config)
            controls = {key: int(assets.tts_meta["model_config"][key]) for key in ("n_vq", "audio_pad_token_id", "audio_start_token_id", "audio_end_token_id", "audio_user_slot_token_id", "audio_assistant_slot_token_id")}
            self.manifest = {
                "tts_config": controls,
                "generation_defaults": copy.deepcopy(GENERATION_DEFAULTS),
                "prompt_templates": {
                    "user_prompt_prefix_token_ids": prompting.build_user_prompt_prefix(self.sp_model, config),
                    "user_prompt_after_reference_token_ids": prompting.build_user_prompt_after_reference(self.sp_model),
                    "assistant_prompt_prefix_token_ids": prompting.build_assistant_prompt_prefix(self.sp_model, config),
                },
                "builtin_voices": [{"voice": voice, "description": row[1]} for voice, row in assets.voice_files.items()],
                "text_samples": [],
            }
            model = self.tts_meta["model_config"]
            global_outputs = ["global_hidden"] + [name for layer in range(int(model["global_layers"])) for name in (f"present_key_{layer}", f"present_value_{layer}")]
            global_past = [name.replace("present_", "past_") for name in global_outputs[1:]]
            local_outputs = ["text_logits", "audio_logits"] + [name for layer in range(int(model["local_layers"])) for name in (f"local_present_key_{layer}", f"local_present_value_{layer}")]
            self.tts_meta["onnx"] = {
                "prefill_output_names": global_outputs,
                "decode_input_names": ["input_ids", "past_valid_lengths", *global_past],
                "decode_output_names": global_outputs,
                "local_cached_output_names": local_outputs,
            }

            def prefill(feeds):
                ids = np.asarray(feeds["input_ids"], dtype=np.int32)
                mask = np.asarray(feeds["attention_mask"], dtype=np.int32)
                if ids.ndim != 3 or ids.shape[0] != 1 or mask.shape != (1, ids.shape[1]) or not np.all(mask == 1):
                    raise ValueError("Unified global prefill supports exact batch-one all-valid prompts")
                cache_shape = (1, 0, int(model["global_heads"]), int(model["head_dim"]))
                return self.client.run("global_unified", {"input_ids": ids, "attention_mask": mask, **{name: np.zeros(cache_shape, dtype=np.float32) for name in global_past}})

            def decode(feeds):
                ids = np.asarray(feeds["input_ids"], dtype=np.int32)
                past = {name: (feeds[name] if getattr(feeds[name], "_moss_device_tensor", False)
                               else np.asarray(feeds[name], dtype=np.float32)) for name in global_past}
                lengths = {array.shape[1] for array in past.values()}
                if len(lengths) != 1:
                    raise ValueError("Global KV cache lengths do not agree")
                length = next(iter(lengths))
                if np.asarray(feeds["past_valid_lengths"]).tolist() != [length]:
                    raise ValueError("Unified graph cannot consume padded or partly-valid KV caches")
                return self.client.run("global_unified", {"input_ids": ids, "attention_mask": np.ones((1, length + ids.shape[1]), dtype=np.int32), **past})

            def encode(feeds):
                waveform = np.asarray(feeds["waveform"], dtype=np.float32)
                valid = np.asarray(feeds["input_lengths"], dtype=np.int64)
                length = int(valid[0])
                if waveform.shape[:2] != (1, 2) or not 0 < length <= waveform.shape[-1]:
                    raise ValueError("Original codec reference input must be nonempty 48 kHz stereo")
                padded = ((waveform.shape[-1] + 3839) // 3840) * 3840
                if padded != waveform.shape[-1]:
                    waveform = np.pad(waveform, ((0, 0), (0, 0), (0, padded - waveform.shape[-1])))
                outputs = self.client.run("codec_encoder", {"waveform": waveform, "waveform_lengths": valid})
                return {"audio_codes": np.asarray(outputs["audio_codes"], dtype=np.int32).transpose(1, 2, 0), "audio_code_lengths": np.asarray(outputs["audio_codes_lengths"], dtype=np.int32)}

            def codec_decode(feeds):
                codes = np.asarray(feeds["audio_codes"], dtype=np.int64).transpose(2, 0, 1)
                outputs = self.client.run("codec_decoder", {"audio_codes": codes, "audio_codes_lengths": np.asarray(feeds["audio_code_lengths"], dtype=np.int64)})
                return {"audio": np.asarray(outputs["waveform"], dtype=np.float32), "audio_lengths": np.asarray(outputs["waveform_lengths"], dtype=np.int32)}

            def local_step(feeds):
                outputs = self.client.run("local_cached_step", feeds)
                contract = getattr(pool, "selected_local_contract", None)
                return wrap_selected_outputs(outputs, feeds, contract) if contract else outputs

            self.sessions = {
                "prefill": SessionAdapter(self.client, "global_unified", global_outputs, prefill),
                "decode": SessionAdapter(self.client, "global_unified", global_outputs, decode),
                "local_cached_step": SessionAdapter(self.client, "local_cached_step", local_outputs, local_step),
                "codec_encode": SessionAdapter(self.client, "codec_encoder", ["audio_codes", "audio_code_lengths"], encode),
                "codec_decode": SessionAdapter(self.client, "codec_decoder", ["audio", "audio_lengths"], codec_decode),
            }
            if frame_contract is not None:
                self.sessions["local_fixed_sampled_frame"] = SessionAdapter(
                    self.client, "local_fixed_sampled_frame",
                    ["should_continue", "frame_token_ids", "text_logits", "audio_logits"],
                )
            self.codec_streaming_session = None
            self._active_reference_codes = None
            self._request_audio_callback = None
            self._previous_stream_chunk_text = None
            if "codec_decode_step" in assets.graph_paths:
                output_names = ["audio", "audio_lengths"]
                state = self.codec_meta["streaming_decode"]
                for spec in state.get("transformer_offsets", []):
                    output_names.append(spec["output_name"])
                for spec in state.get("attention_caches", []):
                    output_names.extend(spec[key] for key in ("offset_output_name", "cached_keys_output_name", "cached_values_output_name", "cached_positions_output_name"))
                step = SessionAdapter(self.client, "codec_decode_step", output_names)
                self.sessions["codec_decode_step"] = step
                self.codec_streaming_session = _ObservedStreamingCodec(original_ort.CodecStreamingDecodeSession(self.codec_meta, step))

        def slice_audio_channel_logits(self, audio_logits, channel_index):
            if isinstance(audio_logits, SelectedAudioScores):
                return audio_logits.for_channel(channel_index)
            return super().slice_audio_channel_logits(audio_logits, channel_index)

        def run_local_fixed_sampled_frame(self, global_hidden, *, previous_token_sets_by_channel):
            return _run_sampled_frame(
                self, global_hidden,
                previous_token_sets_by_channel=previous_token_sets_by_channel,
            )

        def prepare_synthesis_text(self, *, text, voice="", prompt_text="", enable_wetext=True, enable_normalize_tts_text=True):
            return prepare_with_request_language(
                self, original.OnnxTtsRuntime.prepare_synthesis_text.__get__(self),
                normalization_language=self._request_normalization_language,
                text=text,
                voice=voice,
                prompt_text=prompt_text,
                enable_wetext=enable_wetext,
                enable_normalize_tts_text=enable_normalize_tts_text,
            )

        def resolve_prompt_audio_codes(self, *, voice, prompt_audio_path):
            self._active_reference_codes = None
            if prompt_audio_path:
                if "codec_encoder" in getattr(pool, "omitted_graphs", {}):
                    raise FileNotFoundError("Custom reference audio needs the original encoder plan; this pool currently supports the verified built-in reference voices")
                path = Path(prompt_audio_path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                codes = self.encode_reference_audio(path)
                self._active_reference_codes = codes
                return codes
            chosen = voice or assets.default_voice
            if chosen not in assets.voice_paths:
                raise ValueError(f"Unknown original MOSS voice: {chosen}")
            if chosen in assets.precomputed_reference_codes:
                codes = [list(row) for row in assets.precomputed_reference_codes[chosen]]
                self._active_reference_codes = codes
                return codes
            path = assets.voice_paths[chosen]
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            with pool.reference_lock:
                codes = pool.reference_codes.get(key)
                if codes is None:
                    codes = tuple(tuple(int(value) for value in row) for row in self.encode_reference_audio(path))
                    pool.reference_codes[key] = codes
            result = [list(row) for row in codes]
            self._active_reference_codes = result
            return result

        def split_voice_clone_text(self, text, max_tokens=75):
            codes = self._active_reference_codes
            if codes is None:
                raise RuntimeError("Reference audio codes must be resolved before text chunking")
            prefill_max = int(assets.graphs["global_unified"]["profiles"]["input_ids"]["max"][1])
            capacity = prompt_text_capacity(self, codes, prefill_max)
            chunks = split_preserving_boundaries(
                text, max_tokens,
                original_split=lambda value, budget: original.OnnxTtsRuntime.split_voice_clone_text(self, value, max_tokens=budget),
                count_tokens=self.count_text_tokens,
                text_capacity=capacity,
                thai_word_breaks=thai_word_breaks,
            )
            log.info("[moss] split chars=%d chunks=%d tokens=%s preferred=%d capacity=%d",
                     len(text), len(chunks), [self.count_text_tokens(chunk) for chunk in chunks],
                     max_tokens, capacity)
            return chunks

        def decode_full_audio(self, generated_frames):
            _require_full_codec_decoder(pool)
            return super().decode_full_audio(generated_frames)

        def decode_full_audio_safe(self, generated_frames):
            _require_full_codec_decoder(pool)
            if not generated_frames:
                return np.zeros((0, 2), dtype=np.float32)
            if self.codec_streaming_session is not None:
                # Keep the author's same-codec incremental fallback when a
                # full decode fails; no alternate model or audio is used.
                return super().decode_full_audio_safe(generated_frames)
            channels, _length = self.decode_full_audio(generated_frames)
            return original._merge_audio_channels(channels)

        def warmup(self):
            codes = self.resolve_prompt_audio_codes(voice=assets.default_voice, prompt_audio_path=None)
            rows = self.build_voice_clone_request_rows(codes, self.encode_text("你好。"))
            outputs = self.sessions["prefill"].run(None, {"input_ids": np.asarray([rows["inputIds"]], dtype=np.int32), "attention_mask": np.asarray(rows["attentionMask"], dtype=np.int32)})
            hidden = original_ort._extract_last_hidden(outputs[0])
            if frame_contract is not None:
                _validate_frame_sampling(self.manifest["generation_defaults"], frame_contract)
                self.run_local_fixed_sampled_frame(hidden, previous_token_sets_by_channel=[set() for _ in range(16)])
            else:
                self.run_local_cached_step(hidden, text_token_id=0, audio_token_id=0, channel_index=0, step_type=0, past_valid_lengths=0, local_past_by_name=self.create_empty_local_cached_past())
            if "codec_decoder" in getattr(pool, "omitted_graphs", {}):
                self.codec_streaming_session.reset()
                try:
                    self.codec_streaming_session.run_frames([codes[0]])
                finally:
                    self.codec_streaming_session.reset()
            else:
                self.decode_full_audio([codes[0]])

        def synthesize_single_chunk(self, *, text, prompt_audio_codes, streaming):
            if self._request_audio_callback is not None and self._previous_stream_chunk_text is not None:
                # Match only the original runtime's authored inter-chunk pause.
                pause = self.estimate_voice_clone_inter_chunk_pause_seconds(self._previous_stream_chunk_text)
                self._request_audio_callback(np.zeros((round(48000 * pause), 2), dtype=np.float32))
            self._chunk_minimum_frames = minimum_chunk_frames(text)
            try:
                result = super().synthesize_single_chunk(text=text, prompt_audio_codes=prompt_audio_codes, streaming=streaming)
            finally:
                self._chunk_minimum_frames = 1
            token_ids = result.get("text_token_ids")
            log.info("[moss] chunk chars=%d text_tokens=%s generated_frames=%d frame_limit=%s",
                     len(text), None if token_ids is None else len(token_ids), len(result["generated_frames"]),
                     self.manifest["generation_defaults"].get("max_new_frames"))
            validate_chunk_result(result, self.manifest["generation_defaults"], text=text)
            self._previous_stream_chunk_text = text
            return result

    if use_fast_sampling:
        from .sampling import make_fast_generation_method
        AdaptedMossRuntime.generate_audio_frames = make_fast_generation_method(
            original_ort.OrtCpuRuntime.generate_audio_frames)
    if frame_contract is not None:
        AdaptedMossRuntime.generate_audio_frames = _make_frame_generation_method(
            AdaptedMossRuntime.generate_audio_frames, frame_contract)
    original_generate = AdaptedMossRuntime.generate_audio_frames
    @wraps(original_generate)
    def guarded_generate(self, request_rows, on_frame=None):
        return generate_with_early_eos_guard(
            lambda rows, on_frame: original_generate(self, rows, on_frame=on_frame),
            request_rows, minimum_frames=getattr(self, "_chunk_minimum_frames", 1),
            on_frame=on_frame,
        )
    AdaptedMossRuntime.generate_audio_frames = guarded_generate
    return AdaptedMossRuntime


class MossRuntime:
    """One original-model inference instance with isolated mutable state."""
    def __init__(self, pool: SharedGraphPool, *, instance_id: str = "tts-0", seed: int = 1234, output_dir: str | Path = "conversion/moss-runtime/outputs"):
        self.pool = pool
        self.instance_id = instance_id
        self._lock = threading.Lock()
        self._runtime = _create_original_adapter_class(pool)(instance_id, seed, Path(output_dir))

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def set_generation_defaults(self, **values):
        unknown = set(values) - GENERATION_DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown sampling controls: {sorted(unknown)}")
        with self._lock:
            self._runtime.manifest["generation_defaults"].update(values)

    def _language_kwargs(self, kwargs):
        kwargs = dict(kwargs)
        # The original ONNX API maps an omitted mode back to its historical
        # fixed preset. Pass the accepted full sampler explicitly for ordinary
        # calls while preserving an explicitly requested mode or greedy run.
        if kwargs.get("sample_mode") is None:
            kwargs["sample_mode"] = (GENERATION_DEFAULTS["sample_mode"]
                                     if kwargs.get("do_sample", True) else "greedy")
        language = kwargs.pop("language", "auto")
        language = resolve_request_language(
            self.pool.assets, kwargs.get("text", ""),
            kwargs.get("voice") or "", language,
            normalize=kwargs.get("enable_normalize_tts_text", True),
        )
        if language not in self.pool.assets.languages:
            raise ValueError(f"Unsupported original MOSS request language: {language}")
        # Match zhangjinghong app_gpu: WeText grammars are zh/en only; other
        # scripts use the unchanged original robust normalizer.
        kwargs["enable_wetext"] = kwargs.get("enable_wetext", True) and language in {"zh", "en"}
        return language, kwargs

    def synthesize(self, **kwargs):
        language, kwargs = self._language_kwargs(kwargs)
        # When the true codec step graph is attached, use its streaming decode
        # by default so full-decoder activation need not be materialized.
        kwargs.setdefault("streaming", self._runtime.codec_streaming_session is not None)
        if kwargs.get("streaming") and self._runtime.codec_streaming_session is None:
            raise FileNotFoundError("True incremental codec graph is not attached; supply streaming_codec_meta or use streaming=False")
        if not kwargs.get("streaming"):
            _require_full_codec_decoder(self.pool)
        with self._lock:
            self._runtime._previous_stream_chunk_text = None
            self._runtime._request_normalization_language = language
            try:
                result = self._runtime.synthesize(**kwargs)
                result["language"] = language
                return result
            finally:
                self._runtime._request_normalization_language = None

    def synthesize_stream(self, *, on_audio: Callable[[np.ndarray], None], **kwargs):
        """Blocking synthesis with an actual incremental audio callback.

        Callback receives independent [samples,2] float32 chunks immediately
        after codec_decode_step, before later frames are generated. The caller
        can queue/transport these chunks. This method does not start a server.
        """
        session = self._runtime.codec_streaming_session
        if session is None:
            raise FileNotFoundError("The original incremental codec export is required for audio streaming")
        with self._lock:
            language, kwargs = self._language_kwargs(kwargs)
            kwargs.pop("streaming", None)
            session.on_audio = on_audio
            self._runtime._request_audio_callback = on_audio
            self._runtime._previous_stream_chunk_text = None
            self._runtime._request_normalization_language = language
            try:
                result = self._runtime.synthesize(streaming=True, **kwargs)
                result["language"] = language
                return result
            finally:
                session.on_audio = None
                self._runtime._request_audio_callback = None
                self._runtime._request_normalization_language = None

    def state_summary(self) -> dict:
        return {"instance_id": self.instance_id, "backend": self.pool.backend, "shared_graph_weights": True, "independent_rng": True, "global_local_kv_scope": "request-local", "codec_cache_scope": "instance-local", "streaming_codec_available": self._runtime.codec_streaming_session is not None, "full_codec_decoder_available": "codec_decoder" not in getattr(self.pool, "omitted_graphs", {}), "context_allocation": "lazy" if hasattr(self.pool, "instance_sessions") else "shared stateless ONNX sessions", "encoder_context_mode": getattr(self.pool, "encoder_context_mode", "shared stateless ONNX session"), "custom_reference_encoding_available": "codec_encoder" not in getattr(self.pool, "omitted_graphs", {}), "generation_defaults": copy.deepcopy(self._runtime.manifest["generation_defaults"]), "effective_optimizations": {"resident_state": bool(getattr(self.pool, "resident_state_enabled", False)), "fast_sampling": self._runtime.fast_sampling_enabled, "fused_frame": self._runtime.fused_frame_enabled}}
