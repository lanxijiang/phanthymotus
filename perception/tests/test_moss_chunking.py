"""Pure text checks for the MOSS splitter; no model or GPU is loaded."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "plugins/moss_deploy/chunking.py"
SPEC = importlib.util.spec_from_file_location("moss_chunking_under_test", MODULE_PATH)
chunking = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunking)


def word_count(text):
    return len(text.split())


def split(text, original, *, capacity=20, count_tokens=word_count):
    return chunking.split_preserving_boundaries(
        text, 75, original_split=lambda *_: original,
        count_tokens=count_tokens, text_capacity=capacity,
    )


def test_valid_original_split_is_unchanged():
    original = ["Hello,", "world."]
    assert split("Hello, world.", original) == original
    assert split("  \n  ", original) == []


def test_authors_known_multi_chunk_normalization_stays_unchanged():
    # The author's splitter may capitalize a Latin opening or append one
    # terminal stop only when it returns multiple chunks.
    latin = ["Hello there", "friend."]
    assert split("hello there friend", latin) == latin
    japanese = ["こんにちは。", "次の文。"]
    assert split("こんにちは。次の文", japanese, count_tokens=len) == japanese


def test_real_punctuation_loss_is_repaired():
    source = "Hello, world."
    assert split(source, ["Hello", "world."]) == [source]


@pytest.mark.parametrize("source, original", [
    ("We are ready.", ["Weareready."]),
    ("We are ready.", ["We", "areready."]),
    ("Weareready.", ["We are ready."]),
    ("Café crème.", ["Cafécrème."]),
    ("Cafécrème.", ["Café crème."]),
])
def test_latin_word_spacing_inside_a_chunk_is_preserved(source, original):
    assert split(source, original) == [source]


def test_collapsed_whitespace_and_chunk_boundary_remain_allowed():
    assert split("We  are\tready.", ["We are ready."]) == ["We are ready."]
    assert split("We are ready.", ["We", "are", "ready."]) == [
        "We", "are", "ready."
    ]


@pytest.mark.parametrize("original", [
    [],
    ["one two"],
    ["one two", "two three."],
    ["two", "one three."],
    ["one", "", "two three."],
    ["One two three."],
    ["one two three. extra"],
])
def test_invalid_original_split_cannot_change_spoken_characters(original):
    text = "one two three."
    assert split(text, original) == [text]


def test_corrupt_split_recovers_only_at_safe_source_boundaries():
    text = "one two three four five"
    chunks = split(text, [], capacity=2)
    assert chunks == ["one two", "three four", "five"]
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_original_chunk_above_prefill_capacity_is_repaired():
    text = "one two three four"
    assert split(text, [text], capacity=2) == ["one two", "three four"]


def test_no_safe_boundary_fails_instead_of_silent_truncation():
    text = "abcdefghijkl"
    with pytest.raises(ValueError, match="no safe chunk boundary"):
        split(text, [], capacity=5, count_tokens=len)
    with pytest.raises(ValueError, match="no capacity"):
        split(text, [text], capacity=0, count_tokens=len)


def test_unsafe_thai_cut_and_number_unit_are_kept_intact():
    thai = "ภาษาไทย"
    assert split(thai, ["ภาษา", "ไทย"], count_tokens=len, capacity=20) == [thai]
    assert split(thai, ["ภาษา", " ไทย"], count_tokens=len, capacity=20) == [thai]

    text = "The package weighs 100 kg today."
    chunks = split(text, ["The package weighs 100", "kg today."], capacity=3)
    assert chunks == ["The package weighs", "100 kg today."]
    assert "100 kg" in chunks[1]
    assert split("Dose 100 kg.", ["Dose 100kg."]) == ["Dose 100 kg."]


def test_repair_keeps_authors_75_token_budget_instead_of_joining_chunks():
    words = ["word"] * 74 + ["100", "kg"] + ["more"] * 64
    text = " ".join(words)
    # A split inside the protected number/unit expression is unsafe. The old
    # repair joined all 140 tokens because the prefill profile allowed it.
    original = [" ".join(words[:75]), " ".join(words[75:])]
    chunks = split(text, original, capacity=363)
    assert len(chunks) > 1
    assert all(word_count(chunk) <= 75 for chunk in chunks)
    assert any("100 kg" in chunk for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_oversized_original_chunk_is_repaired_at_preferred_budget():
    text = " ".join(["word"] * 90)
    chunks = split(text, [text], capacity=363)
    assert [word_count(chunk) for chunk in chunks] == [75, 15]


def test_only_indivisible_protected_piece_may_exceed_preferred_budget():
    thai = "ก" * 80
    assert split(thai, [thai[:40], thai[40:]], capacity=200, count_tokens=len) == [thai]
    quoted = "“" + "ภาษาไทย" * 12 + "”"
    assert split(quoted, [], capacity=200, count_tokens=len) == [quoted]
    with pytest.raises(ValueError, match="no safe chunk boundary"):
        split("a" * 80, [], capacity=200, count_tokens=len)


def test_oversized_thai_runs_do_not_create_empty_chunks_between_them():
    text = "ก" * 80 + "! " + "ข" * 80 + "."
    chunks = split(text, [], capacity=200, count_tokens=len)
    assert chunks == ["ก" * 80 + "!", "ข" * 80 + "."]
    assert all(chunks)


def test_repair_reuses_valid_original_chinese_cut():
    text = "中" * 80 + "100 kg"
    original = ["中" * 70, "中" * 10 + "100", "kg"]
    chunks = split(text, original, capacity=200, count_tokens=len)
    assert chunks == ["中" * 70, "中" * 10 + "100 kg"]


@pytest.mark.parametrize("text", [
    "ภาษาไทย" * 12 + " , " + "ภาษาไทย" * 12,
    "“ " + "ภาษาไทย" * 12 + " ”",
])
def test_repaired_thai_never_has_standalone_punctuation(text):
    chunks = split(text, [], capacity=200, count_tokens=len)
    assert all(not chunking._punctuation_only(chunk) for chunk in chunks)
    assert chunking._non_whitespace(chunks) == chunking._non_whitespace([text])


def test_punctuation_cannot_extend_an_unprotected_chunk_past_preferred_budget():
    with pytest.raises(ValueError, match="no safe chunk boundary"):
        split("a" * 75 + " !", [], capacity=200, count_tokens=len)


def test_repair_scans_only_nearby_cuts_as_text_grows():
    def measured(words):
        total_chars = 0

        def tokens(piece):
            nonlocal total_chars
            total_chars += len(piece)
            return len(piece.split())

        text = " ".join(["word"] * words)
        chunks = split(text, [], capacity=200, count_tokens=tokens)
        assert all(word_count(chunk) <= 75 for chunk in chunks)
        return total_chars

    small = measured(600)
    large = measured(1200)
    assert large < 4 * small
