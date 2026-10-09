"""Preserve explicit English SI units through the author's WeText frontend."""

from __future__ import annotations

import re
from collections.abc import Callable


# Only a numeric quantity followed by the lowercase SI spelling m/s is
# rewritten. In particular, x=m/s and uppercase M/s are left to the author
# normalizer rather than guessing whether they denote a unit or an expression.
_METERS_PER_SECOND = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>[+-]?(?:\d+(?:[.,]\d+)?|\.\d+))[ \t\u00a0]*m[ \t\u00a0]*/[ \t\u00a0]*s(?![A-Za-z0-9/])"
)
_STABLE_PHRASE = "meters each second"
_SPOKEN_PHRASE = "meters per second"


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
        normalized_branch = resolve_text_normalization_language(text=branch_input, voice=voice)
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
