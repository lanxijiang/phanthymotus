"""An explicit Chinese count must not become a telephone fragment in TN."""
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend.special_text import (
    normalize_special_text,
    verbalize_chinese_number,
)


def number(value, language):
    assert language == "ZH"
    return verbalize_chinese_number(value)


@pytest.mark.parametrize("classifier", ["台", "个", "件", "人", "份", "套"])
def test_count_is_protected_from_telephone_style_normalization(classifier):
    plain = Mock(side_effect=lambda text: text.replace("111", "幺幺幺"))
    text = f"共有111{classifier}，请确认。"
    actual = normalize_special_text(text, plain, number)
    assert f"一百一十一{classifier}" in actual
    assert "幺" not in actual
    assert all("111" not in call.args[0] for call in plain.call_args_list)


def test_count_and_decimal_model_boundaries_stay_separate():
    plain = Mock(side_effect=lambda text: text)
    text = "检查A111台设备、0.111台、device_111台、id-111台和111台设备。"
    actual = normalize_special_text(text, plain, number)
    for identifier in ("A111台", "0.111台", "device_111台", "id-111台"):
        assert identifier in actual
    assert actual.count("一百一十一台") == 1


@pytest.mark.parametrize("text", [
    "编号A111台和ID_111台。",
    "版本0.111台和1.2.111台。",
    "共有1,111台设备。",
    "111和310只是普通数字。",
    "There are 111 devices.",
])
def test_ambiguous_or_non_count_text_is_delegated_unchanged(text):
    plain = Mock(side_effect=lambda value: value)
    assert normalize_special_text(text, plain, number) == text
    plain.assert_called_once_with(text)


def test_counts_do_not_overlap_formulas_or_measurements():
    text = "共111台，容量512GB，已知x=111，增加1+111台。"
    actual = normalize_special_text(text, lambda value: value, number)
    assert "一百一十一台" in actual
    assert "五百一十二 G B" in actual
    assert "x 等于 一百一十一" in actual
    assert "一 加 一百一十一 台" in actual


def test_address_digits_keep_literal_reading_next_to_count():
    text = "共111台，请访问https://example.com/111并联系id111@example.com。"
    actual = normalize_special_text(text, lambda value: value, number)
    assert actual.count("一百一十一台") == 1
    assert "斜杠 一 一 一" in actual
    assert "I D 一 一 一 艾特" in actual


def test_spaces_between_count_and_classifier_do_not_change_reading():
    actual = normalize_special_text("共有111 台设备。", lambda value: value, number)
    assert actual == "共有 一百一十一台 设备。"
