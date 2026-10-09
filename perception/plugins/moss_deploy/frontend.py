"""Original tokenizer/prompt construction, with no neural graph execution."""

from __future__ import annotations

from types import SimpleNamespace

import sentencepiece as spm

from .assets import ModelAssets, load_original_prompting, load_original_source
from .chunking import prompt_text_capacity, split_preserving_boundaries
from .si_units import prepare_with_si_unit_guard


class MossFrontend:
    def __init__(self, assets: ModelAssets):
        self.assets = assets
        original, _ort = load_original_source(assets.source_dir)
        # A frontend-only original helper object: its model constructor and
        # downloader are never called, and no synthesis method is exposed.
        self.original = object.__new__(original.OnnxTtsRuntime)
        self.original.sp_model = spm.SentencePieceProcessor(model_file=str(assets.tokenizer_path))
        self.original._text_normalizer_manager = None
        prompting = load_original_prompting(assets.prompting_path)
        config = SimpleNamespace(**assets.config)
        controls = {key: int(assets.tts_meta["model_config"][key]) for key in ("n_vq", "audio_pad_token_id", "audio_start_token_id", "audio_end_token_id", "audio_user_slot_token_id", "audio_assistant_slot_token_id")}
        self.original.manifest = {"tts_config": controls, "prompt_templates": {"user_prompt_prefix_token_ids": prompting.build_user_prompt_prefix(self.original.sp_model, config), "user_prompt_after_reference_token_ids": prompting.build_user_prompt_after_reference(self.original.sp_model), "assistant_prompt_prefix_token_ids": prompting.build_assistant_prompt_prefix(self.original.sp_model, config)}}

    def encode_text(self, text: str) -> list[int]:
        return self.original.encode_text(text)

    def prepare(self, text: str, *, voice: str | None = None, language: str = "auto", max_text_tokens: int = 75, enable_wetext: bool = False) -> dict:
        voice = voice or self.assets.default_voice
        if voice not in self.assets.voice_paths:
            raise ValueError(f"Unknown original MOSS voice: {voice}")
        resolved_language = self.assets.detect_language(text) if language == "auto" else language
        if resolved_language not in self.assets.languages:
            raise ValueError(f"Unsupported original MOSS language: {resolved_language}")
        prepared = prepare_with_si_unit_guard(
            self.original.prepare_synthesis_text,
            text=text,
            voice=voice,
            enable_wetext=enable_wetext and resolved_language in {"zh", "en"},
            enable_normalize_tts_text=True,
        )
        codes = self.assets.precomputed_reference_codes.get(voice)
        prefill_max = int(self.assets.graphs["global_unified"]["profiles"]["input_ids"]["max"][1])
        if codes is not None:
            text_capacity = prompt_text_capacity(self.original, [list(row) for row in codes], prefill_max)
        else:
            # Without a cached voice encoding, reserve the encoder profile's
            # full reference length before planning any longer safe chunks.
            encoder_max_samples = int(self.assets.graphs["codec_encoder"]["profiles"]["waveform"]["max"][-1])
            text_capacity = prompt_text_capacity(self.original, [], prefill_max) - encoder_max_samples // 3840
        chunks = split_preserving_boundaries(
            str(prepared["text"]), max_text_tokens,
            original_split=lambda value, budget: self.original.split_voice_clone_text(value, max_tokens=budget),
            count_tokens=self.original.count_text_tokens,
            text_capacity=text_capacity,
        )
        result = {"voice": voice, "language": resolved_language, "prepared_texts": prepared, "chunks": [{"text": chunk, "text_token_ids": self.encode_text(chunk)} for chunk in chunks], "prompt_templates": self.original.manifest["prompt_templates"], "reference_codes_source": "not loaded"}
        if codes is not None:
            result["reference_codes_source"] = "verified original codec reference export"
            result["reference_valid_frames"] = len(codes)
            result["requests"] = [self.original.build_voice_clone_request_rows([list(row) for row in codes], row["text_token_ids"]) for row in result["chunks"]]
        return result
