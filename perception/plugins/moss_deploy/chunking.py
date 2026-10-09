"""Keep Thai text runs and number/unit expressions intact during text splitting.

The author tokenizer and prompt builder remain the source of truth. The
author's 75-token splitter is used unchanged whenever it chooses safe cuts.
"""

from __future__ import annotations

import re
from collections.abc import Callable


_THAI = re.compile(r"[\u0e00-\u0e7f]")
_NUMBER_UNIT = re.compile(r"(?<![0-9.])\d+(?:[.,]\d+)?\s*[A-Za-zµμ°]+(?:/[A-Za-zµμ°]+)*\b")
_SAFE_SEPARATORS = re.compile(r"\s+|[。！？!?；;，,、]+")


def prompt_text_capacity(runtime, reference_codes: list[list[int]], prefill_max: int) -> int:
    """Maximum text tokens admitted by this export's global prefill profile."""
    fixed_rows = len(runtime.build_voice_clone_request_rows(reference_codes, [])["inputIds"])
    capacity = prefill_max - fixed_rows
    if capacity < 1:
        raise ValueError(f"Reference prompt uses {fixed_rows} rows, exceeding global prefill profile {prefill_max}")
    return capacity


def _risky_boundary(text: str, left: str, right: str, search_from: int) -> tuple[bool, int]:
    left_at = text.find(left, search_from)
    right_at = text.find(right, left_at + len(left)) if left_at >= 0 else -1
    if right_at < 0:
        # The original splitter can rejoin punctuation/spacing. When the
        # source location is ambiguous, still protect an obvious Thai run or
        # numeric unit across the returned chunk boundary.
        joined = f"{left[-24:]}{right[:24]}"
        boundary = min(24, len(left))
        thai_run = bool(_THAI.fullmatch(left[-1])) and bool(_THAI.fullmatch(right[0]))
        unit_split = any(match.start() < boundary < match.end() for match in _NUMBER_UNIT.finditer(joined))
        return thai_run or unit_split, search_from
    gap = text[left_at + len(left):right_at]
    thai_run = not gap and bool(_THAI.fullmatch(left[-1])) and bool(_THAI.fullmatch(right[0]))
    combined = f"{left[-24:]}{gap}{right[:24]}"
    boundary_offset = min(24, len(left))
    unit_split = any(match.start() < boundary_offset + len(gap) < match.end()
                     for match in _NUMBER_UNIT.finditer(combined))
    return thai_run or unit_split, right_at


def _safe_cuts(text: str) -> list[int]:
    protected = [(match.start(), match.end()) for match in _NUMBER_UNIT.finditer(text)]
    cuts = []
    for match in _SAFE_SEPARATORS.finditer(text):
        cut = match.end()
        if 0 < cut < len(text) and not any(begin < cut < end for begin, end in protected):
            cuts.append(cut)
    return sorted(set(cuts)) + [len(text)]


def split_preserving_boundaries(
    text: str,
    max_tokens: int,
    *,
    original_split: Callable[[str, int], list[str]],
    count_tokens: Callable[[str], int],
    text_capacity: int,
) -> list[str]:
    """Repair unsafe original cuts; never discard text or exceed the prefill profile.

    ``max_tokens`` is the author's preferred segment size. A segment may be
    longer when that is necessary to preserve a Thai run or number/unit group.
    Text with no available safe cut within the engine profile raises an error.
    """
    text = str(text or "").strip()
    if not text:
        return []
    original = original_split(text, max_tokens)
    if not original:
        return []
    position = 0
    unsafe = False
    for left, right in zip(original, original[1:]):
        risk, position = _risky_boundary(text, left, right, position)
        unsafe |= risk
    if not unsafe:
        return original
    if count_tokens(text) <= text_capacity:
        return [text]

    cuts = _safe_cuts(text)
    chunks = []
    start = 0
    while start < len(text):
        feasible = [cut for cut in cuts if cut > start and count_tokens(text[start:cut].strip()) <= text_capacity]
        if not feasible:
            raise ValueError(
                "Text has no safe chunk boundary within the global prefill profile; "
                "add a space or punctuation at a natural phrase boundary"
            )
        end = feasible[-1]
        chunks.append(text[start:end].strip())
        start = end
    if any(count_tokens(chunk) > text_capacity for chunk in chunks):
        raise AssertionError("Safe text splitting exceeded the global prefill profile")
    return chunks
