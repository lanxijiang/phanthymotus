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
