"""Keep Thai runs, numbers and addresses intact during text splitting.

The author tokenizer and prompt builder remain the source of truth. Safe
75-token cuts are retained except for the narrow long mixed-sentence repair.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections.abc import Callable


_THAI = re.compile(r"[\u0e00-\u0e7f]")
_THAI_RUN = re.compile(r"[\u0e00-\u0e7f]+")
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_NUMBER_UNIT = re.compile(r"(?<![0-9.])\d+(?:[.,]\d+)?\s*[A-Za-zµμ°]+(?:/[A-Za-zµμ°]+)*\b")
_SAFE_SEPARATORS = re.compile(r"\s+|[。！？!?；;，,、]+")
_STRUCTURED = re.compile(
    # Unicode word boundaries disappear beside Chinese/Japanese letters.
    # Only ASCII token neighbours delimit these ASCII structures.
    # Protect the ASCII URL body, including encoded paths/query strings;
    # adjoining Chinese punctuation/prose must retain its normal safe cuts.
    r"(?a:(?<![A-Za-z0-9_])(?:https?://|www\.)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|(?<![A-Za-z0-9_])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_])(?:[A-Za-z]\.){2,}"
    r"|(?<![A-Za-z0-9_.])v?[0-9]+(?:[.,][0-9]+)+(?![A-Za-z0-9_]|[.,][0-9])",
    re.IGNORECASE,
)


def _structured_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in _STRUCTURED.finditer(text):
        end = match.end()
        if match.group().lower().startswith(("http://", "https://", "www.")):
            # Leave adjoining sentence punctuation outside the address, so
            # the author's permitted terminal stop does not change its value.
            while end > match.start() and text[end - 1] in ".!?;,:。！？\"'”’)]}»›":
                end -= 1
        spans.append((match.start(), end))
    return spans


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
    protected = [(match.start(), match.end()) for match in _NUMBER_UNIT.finditer(text)] + _structured_spans(text)
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
    protected.extend(_structured_spans(text))
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


def _single_protected_piece(text: str, *, allow_thai: bool = True) -> bool:
    """Allow one indivisible Thai run or number/unit expression past 75 tokens."""
    piece = text.strip()
    while piece and unicodedata.category(piece[0]).startswith("P"):
        piece = piece[1:].lstrip()
    while piece and unicodedata.category(piece[-1]).startswith("P"):
        piece = piece[:-1].rstrip()
    return bool((allow_thai and _THAI_RUN.fullmatch(piece)) or _NUMBER_UNIT.fullmatch(piece))


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


def _protected_runs(parts: list[str]) -> tuple[list[str], ...]:
    return (
        sorted(match.group() for part in parts for match in _THAI_RUN.finditer(part)),
        sorted(match.group() for part in parts for match in _NUMBER_UNIT.finditer(part)),
        sorted(part[start:end] for part in parts for start, end in _structured_spans(part)),
    )


def _refine_long_mixed_chunks(chunks, count_tokens, text_capacity):
    """Separate complete strong-stop sentences only in long CJK/Latin chunks.

    Short accepted phrasing and single-language text keep the author's cuts.
    Periods are deliberately ambiguous (addresses, decimals, abbreviations).
    This narrow repair does not change those boundaries or split one sentence.
    """
    result = []
    for chunk in chunks:
        if (count_tokens(chunk) <= 48 or not _CJK.search(chunk) or not re.search(r"[A-Za-z]", chunk)
                or re.search(r"[\u3040-\u30ff\u0400-\u052f\u0e00-\u0e7f]", chunk)):
            result.append(chunk)
            continue
        protected = _structured_spans(chunk) + [(m.start(), m.end()) for m in _NUMBER_UNIT.finditer(chunk)]
        cuts = [m.end() for m in re.finditer(r"[。！？!?]+", chunk)
                if not any(begin < m.end() < end for begin, end in protected)]
        # An unfinished tail or a quote-only fragment is not a trusted full
        # sentence. Keep this author's chunk intact in those ambiguous cases.
        if not cuts or chunk[cuts[-1]:].strip():
            result.append(chunk)
            continue
        pieces, start = [], 0
        for cut in cuts:
            pieces.append(chunk[start:cut].strip())
            start = cut
        if (len(pieces) < 2 or any(not piece or _punctuation_only(piece) for piece in pieces)
                or any(count_tokens(piece) > text_capacity for piece in pieces)
                or _non_whitespace(pieces) != _non_whitespace([chunk])
                or _protected_runs(pieces) != _protected_runs([chunk])
                or not _latin_spacing_preserved(chunk, pieces)):
            result.append(chunk)
        else:
            result.extend(pieces)
    return result


def split_preserving_boundaries(
    text: str,
    max_tokens: int,
    *,
    original_split: Callable[[str, int], list[str]],
    count_tokens: Callable[[str], int],
    text_capacity: int,
    thai_word_breaks: Callable[[str], list[int]] | None = None,
) -> list[str]:
    """Repair unsafe original cuts; never discard text or exceed the prefill profile.

    ``max_tokens`` is the author's preferred segment size. A segment may be
    longer when that is necessary to preserve a Thai run or number/unit group.
    When a Thai dictionary is supplied, long runs instead use its word cuts
    and must respect the preferred budget. Short Thai runs are unchanged.
    Text with no available safe cut within the engine profile raises an error.
    """
    text = str(text or "").strip()
    if not text:
        return []
    if text_capacity < 1:
        raise ValueError("Global prefill profile has no capacity for text")
    preferred = min(max(1, int(max_tokens)), text_capacity)
    thai_cuts = []
    thai_runs = list(_THAI_RUN.finditer(text))
    segmented_thai_runs = set()

    def add_thai_cuts(match):
        segmented_thai_runs.add(match.start())
        for offset in thai_word_breaks(match.group()):
            if not isinstance(offset, int) or not 0 < offset < len(match.group()):
                raise ValueError("Invalid Thai dictionary word boundary")
            thai_cuts.append(match.start() + offset)

    if thai_word_breaks is not None:
        for match in thai_runs:
            if count_tokens(match.group()) > preferred:
                add_thai_cuts(match)
    allow_long_thai = thai_word_breaks is None
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
        or any(count_tokens(chunk) > preferred and not _single_protected_piece(chunk, allow_thai=allow_long_thai)
               for chunk in original)
    )
    position = 0
    unsafe = invalid_original
    if not invalid_original:
        for left, right in zip(original, original[1:]):
            risk, position = _risky_boundary(text, left, right, position)
            unsafe |= risk
    if not unsafe:
        return _refine_long_mixed_chunks(original, count_tokens, text_capacity)
    if count_tokens(text) <= preferred:
        return _refine_long_mixed_chunks([text], count_tokens, text_capacity)

    cuts = sorted(set(_safe_cuts(text) + _original_safe_cuts(text, original) + thai_cuts))
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
            if end is None and tokens <= text_capacity and _single_protected_piece(piece, allow_thai=allow_long_thai):
                end, end_index = cut, index
            break
        if end is None:
            # The run alone may fit while its surrounding punctuation or
            # attached prefix does not. Only request these extra dictionary
            # boundaries when existing cuts cannot fit the current chunk;
            # ordinary short Thai and already-valid phrase cuts stay intact.
            if thai_word_breaks is not None:
                for match in thai_runs:
                    if match.end() > start and match.start() not in segmented_thai_runs:
                        add_thai_cuts(match)
                expanded = sorted(set(cuts + thai_cuts))
                if expanded != cuts:
                    cuts = expanded
                    next_cut = bisect_right(cuts, start)
                    continue
            if chunks and _punctuation_only(text[start:]):
                joined = chunks[-1] + text[start:].strip()
                joined_tokens = count_tokens(joined)
                if joined_tokens <= text_capacity and (
                    joined_tokens <= preferred or _single_protected_piece(joined, allow_thai=allow_long_thai)
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
    return _refine_long_mixed_chunks(chunks, count_tokens, text_capacity)
