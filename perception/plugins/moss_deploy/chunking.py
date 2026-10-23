"""Keep Thai text runs and number/unit expressions intact during text splitting.

The author tokenizer and prompt builder remain the source of truth. The
author's 75-token splitter is used unchanged whenever it chooses safe cuts.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable


_THAI = re.compile(r"[\u0e00-\u0e7f]")
_THAI_RUN = re.compile(r"[\u0e00-\u0e7f]+")
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
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


def _original_safe_cuts(text: str, chunks: list[str]) -> list[int]:
    """Reuse original CJK cuts that locate exactly in the source text."""
    protected = [(match.start(), match.end()) for pattern in (_THAI_RUN, _NUMBER_UNIT)
                 for match in pattern.finditer(text)]
    cuts = []
    search_from = 0
    for chunk in chunks[:-1]:
        if not isinstance(chunk, str) or not chunk.strip():
            break
        found = text.find(chunk, search_from)
        if found < 0:
            break
        cut = found + len(chunk)
        if (0 < cut < len(text)
                and not any(begin < cut < end for begin, end in protected)
                and not (_latin_word_character(text[cut - 1])
                         and _latin_word_character(text[cut]))):
            cuts.append(cut)
        search_from = cut
    return cuts


def _punctuation_only(text: str) -> bool:
    piece = text.strip()
    return bool(piece) and all(char.isspace() or unicodedata.category(char).startswith("P")
                               for char in piece)


def _single_protected_piece(text: str) -> bool:
    """Allow one indivisible Thai run or number/unit expression past 75 tokens."""
    piece = text.strip()
    while piece and unicodedata.category(piece[0]).startswith("P"):
        piece = piece[1:].lstrip()
    while piece and unicodedata.category(piece[-1]).startswith("P"):
        piece = piece[:-1].rstrip()
    return bool(_THAI_RUN.fullmatch(piece) or _NUMBER_UNIT.fullmatch(piece))


def _non_whitespace(parts: list[str]) -> str:
    return "".join(character for part in parts for character in part if not character.isspace())


def _author_prepared_text(text: str) -> str:
    # The author's multi-chunk preparation can capitalize the initial Latin
    # letter and append one terminal stop. Admit only those known additions.
    prepared = text
    if _CJK.search(text):
        if prepared[-1] not in ".!?。！？；;":
            prepared += "。"
    else:
        if prepared[0].islower():
            prepared = prepared[0].upper() + prepared[1:]
        if prepared[-1].isalnum():
            prepared += "."
    return prepared


def _author_text_preserved(text: str, chunks: list[str]) -> bool:
    spoken = _non_whitespace(chunks)
    return spoken == _non_whitespace([text]) or (
        len(chunks) > 1 and spoken == _non_whitespace([_author_prepared_text(text)])
    )


def _latin_word_character(character: str) -> bool:
    return character.isdigit() or unicodedata.name(character, "").startswith("LATIN ")


def _latin_spacing_preserved(text: str, chunks: list[str]) -> bool:
    # Across chunk boundaries the author's splitter may remove or insert a
    # space. Inside one chunk, retain whether adjacent Latin word characters
    # were separated: compacting only non-whitespace characters misses both
    # "We are" -> "Weare" and "Weare" -> "We are".
    expected = text if _non_whitespace(chunks) == _non_whitespace([text]) else _author_prepared_text(text)
    source_positions = [index for index, char in enumerate(expected) if not char.isspace()]
    offset = 0
    for chunk in chunks:
        chunk_positions = [index for index, char in enumerate(chunk) if not char.isspace()]
        for index in range(len(chunk_positions) - 1):
            left = source_positions[offset + index]
            right = source_positions[offset + index + 1]
            if _latin_word_character(expected[left]) and _latin_word_character(expected[right]):
                source_space = right > left + 1
                chunk_space = chunk_positions[index + 1] > chunk_positions[index] + 1
                if source_space != chunk_space:
                    return False
        offset += len(chunk_positions)
    return True


def _protected_runs(parts: list[str]) -> tuple[list[str], list[str]]:
    return (
        sorted(match.group() for part in parts for match in _THAI_RUN.finditer(part)),
        sorted(match.group() for part in parts for match in _NUMBER_UNIT.finditer(part)),
    )


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
    if text_capacity < 1:
        raise ValueError("Global prefill profile has no capacity for text")
    preferred = min(max(1, int(max_tokens)), text_capacity)
    original = original_split(text, max_tokens)
    # The original splitter may normalize spaces and add its known terminal
    # punctuation, but losing, duplicating or reordering spoken characters
    # would silently change speech.
    # A nonempty input must also never turn into an empty successful request.
    invalid_original = (
        not original
        or any(not isinstance(chunk, str) or not chunk.strip() for chunk in original)
        or any(_punctuation_only(chunk) for chunk in original)
        or not _author_text_preserved(text, original)
        or not _latin_spacing_preserved(text, original)
        or _protected_runs(original) != _protected_runs([text])
        or any(count_tokens(chunk) > text_capacity for chunk in original)
        or any(count_tokens(chunk) > preferred and not _single_protected_piece(chunk)
               for chunk in original)
    )
    position = 0
    unsafe = invalid_original
    if not invalid_original:
        for left, right in zip(original, original[1:]):
            risk, position = _risky_boundary(text, left, right, position)
            unsafe |= risk
    if not unsafe:
        return original
    if count_tokens(text) <= preferred:
        return [text]

    cuts = sorted(set(_safe_cuts(text) + _original_safe_cuts(text, original)))
    chunks = []
    start = 0
    next_cut = 0
    while start < len(text):
        end = None
        end_index = None
        for index in range(next_cut, len(cuts)):
            cut = cuts[index]
            piece = text[start:cut].strip()
            if not piece or _punctuation_only(piece):
                continue
            tokens = count_tokens(piece)
            if tokens <= preferred:
                end, end_index = cut, index
                continue
            # Tokenizers are not strictly monotonic across appended text. A
            # prefix under budget is safe; scanning the whole remaining text
            # at every step is not. A single protected run is the only case
            # allowed to exceed the preferred budget.
            if end is None and tokens <= text_capacity and _single_protected_piece(piece):
                end, end_index = cut, index
            break
        if end is None:
            if chunks and _punctuation_only(text[start:]):
                joined = chunks[-1] + text[start:].strip()
                joined_tokens = count_tokens(joined)
                if joined_tokens <= text_capacity and (
                    joined_tokens <= preferred or _single_protected_piece(joined)
                ):
                    chunks[-1] = joined
                    break
            raise ValueError(
                "Text has no safe chunk boundary within the global prefill profile; "
                "add a space or punctuation at a natural phrase boundary"
            )
        chunks.append(text[start:end].strip())
        start = end
        next_cut = end_index + 1
    if any(count_tokens(chunk) > text_capacity for chunk in chunks):
        raise AssertionError("Safe text splitting exceeded the global prefill profile")
    return chunks
