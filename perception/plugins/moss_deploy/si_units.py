"""Preserve explicit English SI units through the author's WeText frontend."""

from __future__ import annotations

import re
from collections.abc import Callable
from types import FunctionType


# Only a numeric quantity followed by the lowercase SI spelling m/s is
# rewritten. In particular, x=m/s and uppercase M/s are left to the author
# normalizer rather than guessing whether they denote a unit or an expression.
_METERS_PER_SECOND = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>[+-]?(?:\d+(?:[.,]\d+)?|\.\d+))[ \t\u00a0]*m[ \t\u00a0]*/[ \t\u00a0]*s(?![A-Za-z0-9/])"
)
_STABLE_PHRASE = "meters each second"
_SPOKEN_PHRASE = "meters per second"


def resolve_request_language(assets, text, voice, language, *, normalize=True):
    """Preserve the author's voice-based choice for language-neutral input."""
    if language != "auto":
        return language
    detected = assets.detect_language(text)
    if detected == "en" and not re.search(r"[A-Za-z\u3400-\u9fff]", str(text or "")):
        from text_normalization_pipeline import normalize_tts_text, resolve_text_normalization_language

        raw = str(text or "")
        value = normalize_tts_text(raw) if normalize else raw
        return resolve_text_normalization_language(text=value, voice=voice)
    return detected


def _rewrite_mps(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group('number')} {_STABLE_PHRASE}"

    return _METERS_PER_SECOND.sub(replace, text), count


def _restore_mps(text: str, count: int) -> str:
    if count and text.count(_STABLE_PHRASE) != count:
        raise ValueError(
            "WeText meters-per-second restoration is ambiguous; "
            "the protected phrase was changed or already occurs in the input"
        )
    return text.replace(_STABLE_PHRASE, _SPOKEN_PHRASE) if count else text


def prepare_with_si_unit_guard(
    base_prepare: Callable[..., dict],
    *,
    text: str,
    voice: str = "",
    prompt_text: str = "",
    enable_wetext: bool,
    enable_normalize_tts_text: bool = True,
    normalization_language: str | None = None,
) -> dict:
    """Run the original normalizer, guarding only English numeric ``m/s``.

    WeText reads ``3 m/s`` as ``three minutes per second``. Feeding it the
    literal words ``3 meters per second`` also corrupts ``per second`` in the
    installed grammar. ``meters each second`` survives WeText intact; we
    restore the equivalent conventional wording in the final model text.
    """
    original_text = str(text or "")
    original_prompt = str(prompt_text or "")
    normalized_branch = ""
    if enable_wetext:
        # These functions are from the unchanged author source, imported when
        # ModelAssets loads that source. Match its robust-pre language choice.
        from text_normalization_pipeline import normalize_tts_text, resolve_text_normalization_language

        branch_input = normalize_tts_text(original_text) if enable_normalize_tts_text else original_text
        normalized_branch = normalization_language or resolve_text_normalization_language(text=branch_input, voice=voice)
    if normalized_branch == "en":
        input_text, text_count = _rewrite_mps(original_text)
        input_prompt, prompt_count = _rewrite_mps(original_prompt)
    else:
        input_text, input_prompt = original_text, original_prompt
        text_count = prompt_count = 0

    prepared = base_prepare(
        text=input_text,
        voice=voice,
        prompt_text=input_prompt,
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize_tts_text,
    )
    if text_count or prompt_count:
        for key in ("text", "normalized_text"):
            prepared[key] = _restore_mps(str(prepared[key]), text_count)
        for key in ("prompt_text", "normalized_prompt_text"):
            prepared[key] = _restore_mps(str(prepared[key]), prompt_count)
        prepared["normalization_method"] = str(prepared["normalization_method"]) + "+en_si_mps_guard"
        prepared["si_unit_guard"] = {
            "original_text": original_text,
            "pre_wetext_text": input_text,
            "original_prompt_text": original_prompt,
            "pre_wetext_prompt_text": input_prompt,
            "text_replacements": text_count,
            "prompt_replacements": prompt_count,
            "final_text": prepared["text"],
        }
    return prepared


def prepare_with_request_language(runtime, base_prepare, *, normalization_language=None, **kwargs):
    """Keep WeText's choice local to this request without changing author globals."""
    if kwargs.get("enable_wetext") and normalization_language in {"zh", "en"}:
        prepare_method = base_prepare.__func__
        author_pipeline = prepare_method.__globals__["prepare_tts_request_texts"]
        pipeline_globals = dict(author_pipeline.__globals__)
        pipeline_globals["resolve_text_normalization_language"] = lambda **_kwargs: normalization_language
        request_pipeline = FunctionType(author_pipeline.__code__, pipeline_globals,
                                        author_pipeline.__name__, author_pipeline.__defaults__,
                                        author_pipeline.__closure__)
        request_pipeline.__kwdefaults__ = author_pipeline.__kwdefaults__
        manager = runtime._ensure_text_normalizer(True)

        def prepare(**request):
            return request_pipeline(**request, text_normalizer_manager=manager)
    else:
        prepare = base_prepare
    return prepare_with_si_unit_guard(prepare, normalization_language=normalization_language, **kwargs)
