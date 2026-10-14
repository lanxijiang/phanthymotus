"""Hotword matching must not change ordinary quantities or longer identifiers."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend.lexicon_rules import apply_lexeme_rules
from plugins.vits2_tts_trt.frontend.lexicon_rules_generated import GENERATED_REPLACEMENTS


@pytest.mark.parametrize("language", ["MIX", "EN"])
@pytest.mark.parametrize("text", [
    "总计310元和910元，共111台，比例为0.01。",
    "现在有2.5公斤和3.14平方米。",
    "We paid 310 dollars for 111 units, with a 0.01 percent error.",
    "2 3 01 111 310 910 0.310 0.910 2.03 3.02",
])
def test_quantities_stay_available_for_number_normalization(text, language):
    assert apply_lexeme_rules(text, language=language) == text


def test_numeric_source_entry_is_retained():
    assert GENERATED_REPLACEMENTS["111"] == "one one one"


@pytest.mark.parametrize("text,expected", [
    ("使用Ascend 310和Ascend 910。", "使用Ascend 三一零和Ascend 九一零。"),
    ("检查昇腾310和升腾 910。", "检查昇腾三一零和升腾 九一零。"),
    ("ascend310 / ASCEND 910", "ascend三一零 / ASCEND 九一零"),
    ("使用A100、MI250X和MLU370。", "使用A一百、M I二五零X和M L U三七零。"),
    ("使用310P、910A和910B。", "使用三一零P、九一零A和九一零B。"),
    ("比较Qwen3和Qwen2.05。", "比较千问三和千问二点零五。"),
])
def test_explicit_model_readings_are_preserved(text, expected):
    assert apply_lexeme_rules(text) == expected


def test_english_model_names_are_left_for_english_normalization():
    text = "We tested Ascend 310, A100, MI250X, Qwen3, and Qwen2.05."
    assert apply_lexeme_rules(text, language="EN") == text


@pytest.mark.parametrize("text", [
    "xAscend310 Ascend3100 Ascend310_extra",
    "Ascend 310.5 Ascend 910,000",
])
def test_context_models_do_not_match_identifier_or_number_fragments(text):
    assert apply_lexeme_rules(text) == text


@pytest.mark.parametrize("language", ["MIX", "EN"])
@pytest.mark.parametrize("text,expected", [
    ("在OpenMLDB里打开FeatInsight。", "在Open M L D B里打开Feat Insight。"),
    ("openmldb and FEATINSIGHT", "Open M L D B and Feat Insight"),
    ("OpenMLDB使用LAST JOIN，FeatInsight提供SQL Playground。",
     "Open M L D B使用LAST JOIN，Feat Insight提供SQL Playground。"),
])
def test_requested_products_keep_their_word_and_letter_parts(text, expected, language):
    assert apply_lexeme_rules(text, language=language) == expected


@pytest.mark.parametrize("text", [
    "myOpenMLDB OpenMLDBextra OpenMLDB2 MyFeatInsight FeatInsightExtra",
    "OpenMLDB_2 my_OpenMLDB FeatInsight_2 my_FeatInsight",
])
def test_requested_products_are_not_substrings_of_longer_identifiers(text):
    assert apply_lexeme_rules(text) == text
