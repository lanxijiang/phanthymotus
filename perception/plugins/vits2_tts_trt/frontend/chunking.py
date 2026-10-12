"""Engine-agnostic frontend text chunking for VITS2 (shared across backends).

Pure text-processing helpers only — no engine dependency. Both the TensorRT and
ONNX Runtime backends split text the same way so that chunk boundaries (and thus
synthesized audio) are identical regardless of backend.
"""

from __future__ import annotations

import re

import jieba


_ZH_NUMBER_UNIT_RE = re.compile(
    r"[零〇一二两三四五六七八九十百千万亿点]+"
    r"(?:K\s*B|M\s*B|G\s*B|T\s*B|P\s*B)(?:每[A-Za-z])?",
    re.IGNORECASE,
)
_SPLIT_PUNCTUATION = "，,、。．.｡！!？?；;：:…⋯—–~～\n\r"
_CLOSING_PUNCTUATION = "”’\"'」』》〉】〕〗〙〛）)]｝}"
_PUNCTUATION_UNIT_RE = re.compile(
    rf".*?[{re.escape(_SPLIT_PUNCTUATION)}]+"
    rf"[{re.escape(_CLOSING_PUNCTUATION)}]*|.+$",
    flags=re.DOTALL,
)
# Sentence-first is an experimental latency/quality tradeoff.  These are
# deliberately conservative: an uncertain English period keeps the original
# packing instead of turning an abbreviation into a new utterance.
_SENTENCE_END_RE = re.compile(r"[.!?。！？]+")
_PERIOD_WORD_RE = re.compile(r"[A-Za-z]+(?:\.[A-Za-z]+)*$")
_PERIOD_ABBREVIATIONS = frozenset((
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon",
    "capt", "col", "gen", "lt", "sgt", "maj", "esq", "co", "corp",
    "inc", "ltd", "dept", "fig", "eq", "no", "vs", "etc", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
))


def first_sentence_end(text: str) -> int | None:
    """Find a conservative complete-sentence boundary in normalized text.

    Chinese normalization also emits ASCII periods. Keep closing punctuation
    with its sentence and ignore commas, decimal points, domains, initials,
    common abbreviations and ambiguous ellipses. This does not normalize or
    modify any input characters.
    """
    for match in _SENTENCE_END_RE.finditer(text):
        start, end = match.span()
        punctuation = match.group()
        if start == 0:
            continue
        if punctuation == ".":
            before = text[start - 1]
            after = text[end : end + 1]
            # A period inside an ASCII token is not a sentence boundary.
            if after and before.isascii() and before.isalnum() and after.isascii() and after.isalnum():
                continue
            word = _PERIOD_WORD_RE.search(text[:start])
            if word is not None:
                token = word.group()
                if len(token) == 1 or "." in token or token.lower() in _PERIOD_ABBREVIATIONS:
                    continue
        elif set(punctuation) == {"."}:
            continue
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        if punctuation == ".":
            following = text[end:].lstrip()
            if following:
                first = following[0]
                # Lowercase English after a period can still be an unknown
                # abbreviation. Delay this optimization rather than guess.
                if first.isascii() and (first.islower() or first.isdigit()):
                    continue
                # URL paths and email-like forms can follow a domain dot.
                if first in "/\\@_-":
                    continue
        return end
    return None


def _language_kind(char: str) -> str | None:
    if "一" <= char <= "鿿":
        return "ZH"
    if char.isascii() and char.isalnum():
        return "EN"
    return None


def split_positions(text: str) -> tuple[list[int], list[int], list[int]]:
    """Return (language, word, fallback) safe split positions using jieba.

    ``protected`` spans (e.g. ``2KB``) are never split across.
    """
    protected = [match.span() for match in _ZH_NUMBER_UNIT_RE.finditer(text)]

    def is_safe(index: int) -> bool:
        return not any(start < index < end for start, end in protected)

    boundaries = []
    previous_kind = None
    for index, char in enumerate(text):
        kind = _language_kind(char)
        if kind is None:
            continue
        if previous_kind is not None and kind != previous_kind:
            if is_safe(index):
                boundaries.append(index)
        previous_kind = kind
    usable = [index for index in boundaries if 1 < index < len(text) - 1]
    word_boundaries = []
    offset = 0
    for word in jieba.cut(text):
        offset += len(word)
        if 1 < offset < len(text) - 1 and is_safe(offset):
            word_boundaries.append(offset)
    fallback = [index for index in range(1, len(text)) if is_safe(index)]
    if not fallback:
        raise ValueError("Unable to split protected number-unit expression")
    return usable, word_boundaries, fallback


def split_punctuation_units(text: str) -> list[str]:
    """Split ``text`` into punctuation-bounded units (closing punct kept)."""
    return _PUNCTUATION_UNIT_RE.findall(text)


def iter_text_chunks(
    text: str,
    token_count,
    max_tokens: int,
    *,
    first_sentence: bool = False,
) -> "generator[str, None, None]":
    """Yield text chunks that each fit within ``max_tokens``.

    Shared by every backend (TensorRT / ONNX) so that chunk boundaries — and
    therefore the synthesized audio — are identical regardless of backend.
    ``token_count`` must be a callable mapping ``text -> token count``; the
    packing uses the same punctuation-unit + jieba boundary strategy as the
    rest of the frontend.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if first_sentence:
        sentence_end = first_sentence_end(text)
        if sentence_end is not None and text[sentence_end:].strip():
            prefix = text[:sentence_end]
            if prefix.strip() and token_count(prefix) <= max_tokens:
                # Yield without the normal greedy lookahead into later
                # sentences. Keep the complete remainder on the original
                # packing path. Overlong first sentences also fall back to
                # that path, including its existing safe token-limit splits.
                yield prefix
                yield from iter_text_chunks(text[sentence_end:], token_count, max_tokens)
                return

    def _iter_unit_chunks(unit):
        if token_count(unit) <= max_tokens:
            yield unit
            return
        if len(unit) <= 1:
            raise ValueError("Unable to split text within the token limit")
        language_boundaries, word_boundaries, fallback = split_positions(unit)

        def longest_fitting(positions):
            low, high = 0, len(positions) - 1
            best = None
            while low <= high:
                middle = (low + high) // 2
                position = positions[middle]
                if token_count(unit[:position]) <= max_tokens:
                    best = position
                    low = middle + 1
                else:
                    high = middle - 1
            return best

        split_at = longest_fitting(language_boundaries)
        if split_at is None:
            split_at = longest_fitting(word_boundaries)
        if split_at is None:
            split_at = longest_fitting(fallback)
        if split_at is None:
            raise ValueError("Unable to split text within the token limit")
        left, right = unit[:split_at], unit[split_at:]
        if not left.strip() or not right.strip():
            raise ValueError("Unable to split text within the token limit")
        yield from _iter_unit_chunks(left)
        yield from _iter_unit_chunks(right)

    # 1) Validate/split units on demand. Materializing all their chunks here
    # would run G2P over the entire request before the first chunk can reach
    # the engine. Keep the same unit order and greedy packing, with only the
    # lookahead needed to decide whether the next chunk still fits.
    chunks = (
        chunk
        for unit in split_punctuation_units(text)
        if unit.strip()
        for chunk in _iter_unit_chunks(unit)
    )

    # 2) Greedy-join already-valid units; never re-split a merged unit.
    pending = ""
    for chunk in chunks:
        if not pending:
            pending = chunk
            continue
        if token_count(pending + chunk) <= max_tokens:
            pending += chunk
            continue
        yield pending
        pending = chunk
    if pending:
        yield pending
