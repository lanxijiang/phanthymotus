"""Verify supplemental English pronunciations without model/NLTK downloads.

External CMU/G2P dependencies use small fixtures; production lookup, ARPAbet
conversion, both English routes, and the complete shipped overlay are real.
"""

import importlib.util
import json
from pathlib import Path
import pickle
import sys
import types
from unittest.mock import Mock, patch

import pytest
import numpy  # Import the extension before temporary sys.modules isolation.


FRONTEND = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend"


@pytest.fixture
def frontend(tmp_path):
    package = "_vits2_gold_test"
    modules = {name: types.ModuleType(name) for name in (package, "g2p_en", "nltk", "inflect")}
    modules[package].__path__ = [str(FRONTEND)]
    modules["nltk"].data = types.SimpleNamespace(path=[])
    modules["inflect"].engine = lambda: Mock()
    modules["g2p_en"].G2p = lambda: Mock(return_value=["T", "EH1", "S", "T"])
    cache = tmp_path / "cmudict_cache.pickle"
    cache.write_bytes(pickle.dumps({"KNOWN": [["N", "OW1", "N"]]}))

    def load(name):
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", FRONTEND / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    with patch.dict(sys.modules, modules), patch.dict("os.environ", {"VITS2_CMU_CACHE_PATH": str(cache)}):
        english = load("english")
        mixed = load("unified_g2p")
        lexicon = load("lexicon_rules")
    return english, mixed, lexicon


def test_entire_overlay_is_valid_and_below_submission_file_limit(frontend):
    english, _, _ = frontend
    entries = english.gold_en_pronunciations()
    assert len(entries) == 13083
    assert (FRONTEND / "en_gold_pronunciations.json").stat().st_size < 1_000_000
    assert english.gold_en_pronunciations() is entries
    for word, pronunciation in entries.items():
        phones, tones = english.refine_syllables((pronunciation,))
        assert "UNK" not in [english.post_replace_ph(phone) for phone in phones], word
        assert len(phones) == len(tones) and set(tones) <= {1, 2, 3}, word


@pytest.mark.parametrize("word,phones,tones", [
    ("SQL", ("eh", "s", "k", "y", "uw", "eh", "l"), (3, 3, 3, 3, 3, 2, 3)),
    ("NVIDIA", ("eh", "n", "V", "ih", "d", "iy", "ah"), (2, 3, 3, 2, 3, 1, 1)),
])
def test_gold_phones_and_stress_match_in_english_and_mixed_routes(frontend, word, phones, tones):
    english, mixed, _ = frontend
    for spelling in (word, word.lower(), word.title()):
        assert english.g2p_token(spelling) == (phones, tones)
        actual_phones, actual_tones = mixed._en_token_to_phones(spelling)
        assert (tuple(actual_phones), tuple(actual_tones)) == (phones, tones)
    sentence_phones, sentence_tones, _ = english.g2p(word)
    assert tuple(sentence_phones[1:-1]) == phones
    assert tuple(sentence_tones[1:-1]) == tones
    english._g2p.assert_not_called()


def test_explicit_custom_pronunciation_wins_over_gold_and_cmu(frontend):
    english, mixed, _ = frontend
    english.CUSTOM_EN_PRONUNCIATIONS["NVIDIA"] = [["K", "UW1", "D", "AH0"]]
    english.eng_dict["NVIDIA"] = [["B", "AE1", "D"]]
    expected = (("k", "uw", "d", "ah"), (3, 2, 3, 1))
    assert english.g2p_token("Nvidia") == expected
    phones, tones = mixed._en_token_to_phones("Nvidia")
    assert (tuple(phones), tuple(tones)) == expected


def test_gold_wins_over_cmu(frontend):
    english, mixed, _ = frontend
    english.eng_dict["SQL"] = [["B", "AE1", "D"]]
    assert english.g2p_token("SQL")[0] == ("eh", "s", "k", "y", "uw", "eh", "l")
    assert tuple(mixed._en_token_to_phones("SQL")[0]) == english.g2p_token("SQL")[0]


def test_missing_gold_preserves_cmu_and_g2p_fallback(frontend):
    english, mixed, _ = frontend
    assert "KNOWN" not in english.gold_en_pronunciations()
    assert english.g2p_token("known") == (("n", "ow", "n"), (3, 2, 3))
    assert mixed._en_token_to_phones("known") == (["n", "ow", "n"], [3, 2, 3])
    english._g2p.assert_not_called()
    assert english.g2p_token("unlistedtesttoken") == (("t", "eh", "s", "t"), (3, 2, 3, 3))
    english._g2p.assert_called_once_with("unlistedtesttoken")
    assert mixed._en_token_to_phones("unlistedtesttoken") == (["t", "eh", "s", "t"], [3, 2, 3, 3])
    mixed._g2p_en.assert_called_once_with("unlistedtesttoken")


def test_existing_text_lexicon_remains_ahead_of_phone_lookup(frontend):
    english, _, lexicon = frontend
    assert english.gold_en_pronunciations()["QWEN"] == ("K", "W", "IY1", "N")
    assert lexicon.apply_lexeme_rules("Qwen") == "千问"
    assert lexicon.apply_lexeme_rules("Qwen3") == "千问三"
    assert lexicon.apply_lexeme_rules("使用PhanRouter") == "使用Phan Router"
    assert english.g2p_token("CUDA") == (("k", "uw", "d", "ah"), (3, 2, 3, 1))
    assert english.g2p_token("A") == (("ey",), (2,))
    assert english.g2p_token(".") == ((".",), (0,))


def test_known_english_words_survive_fst_before_g2p(frontend):
    english, mixed, _ = frontend
    fst = Mock(side_effect=lambda text: text.replace("NVIDIA", "英伟达").replace("SQL", "S Q L").strip())
    result = english.normalize_preserving_english_pronunciations(
        "Check NVIDIA and SQL with Qwen.", fst,
    )
    assert result == "Check NVIDIA and SQL with Qwen ."
    assert not any("\u4e00" <= char <= "\u9fff" for char in result)
    assert [call.args[0] for call in fst.call_args_list] == ["Check ", " and ", " with ", "."]
    assert "NVIDIA" in result and "SQL" in result
    phones, tones, _ = english.g2p(result)
    assert "UNK" not in phones
    assert len(phones) == len(tones)
    assert tuple(mixed._en_token_to_phones("NVIDIA")[0]) == english.g2p_token("NVIDIA")[0]


def test_protection_keeps_number_time_unit_and_ordinary_context_in_fst(frontend):
    english, _, _ = frontend
    fst = Mock(side_effect=str.strip)
    text = "Check NVIDIA at 5:30 PM with 2 GB for 12 USD."
    result = english.normalize_preserving_english_pronunciations(text, fst)
    assert result == text
    assert [call.args[0] for call in fst.call_args_list] == ["Check ", " at 5:30 PM with 2 GB for 12 USD."]
    fst.reset_mock()
    ordinary = "The driver needs 2 GB at 5:30 PM."
    assert english.normalize_preserving_english_pronunciations(ordinary, fst) == ordinary
    fst.assert_called_once_with(ordinary)


def test_protection_does_not_split_unknown_identifiers_or_word_substrings(frontend):
    english, _, _ = frontend
    fst = Mock(side_effect=str.strip)
    text = "NVIDIAish myNVIDIA my_NVIDIA Qwen3foo Qwen2.5foo Qwen-3-extra Nvidia_X"
    assert english.normalize_preserving_english_pronunciations(text, fst) == text
    fst.assert_called_once_with(text)


def test_lowercase_hotwords_are_protected_without_fragmenting_date_context(frontend):
    english, _, _ = frontend
    fst = Mock(side_effect=str.strip)
    text = "The nvidia driver uses sql on October 14 at 5:30 PM."
    assert english.normalize_preserving_english_pronunciations(text, fst) == text
    assert [call.args[0] for call in fst.call_args_list] == [
        "The ", " driver uses ", " on October 14 at 5:30 PM.",
    ]
    fst.reset_mock()
    text = "Check Airbnb on October 14 at 5:30 PM."
    assert english.normalize_preserving_english_pronunciations(text, fst) == text
    assert [call.args[0] for call in fst.call_args_list] == ["Check ", " on October 14 at 5:30 PM."]


@pytest.mark.parametrize("token,expected", [
    ("Qwen3", "Qwen three"),
    ("qwen-3", "qwen three"),
    ("Qwen2.05", "Qwen two point zero five"),
    ("Qwen2.5B", "Qwen two point five B"),
    ("Qwen3.0.1", "Qwen three point zero point one"),
    ("NVIDIA_2", "NVIDIA underscore two"),
    ("Nvidia2", "Nvidia two"),
    ("A100", "A one hundred"),
    ("a100", "A one hundred"),
    ("H200", "H two hundred"),
    ("MI250X", "M I two hundred fifty X"),
    ("L40S", "L forty S"),
    ("WSE-2", "WSE two"),
    ("Qwen03", "Qwen zero three"),
])
def test_complete_known_model_versions_bypass_chinese_fst(frontend, token, expected):
    english, _, lexicon = frontend
    # Number normalization itself is existing code; isolate the new routing
    # from the optional inflect dependency in this unit test.
    cardinals = {"2": "two", "3": "three", "40": "forty", "100": "one hundred",
                 "200": "two hundred", "250": "two hundred fifty"}
    fst = Mock(side_effect=lambda text: text.replace("3", "三").replace("100", "幺零零").strip())
    original = "Check " + token + " today."
    with patch.object(english, "normalize_numbers", side_effect=cardinals.__getitem__):
        result = english.normalize_preserving_english_pronunciations(
            lexicon.apply_lexeme_rules(original, language="EN"), fst,
        )
    assert result == "Check " + expected + " today."
    assert [call.args[0] for call in fst.call_args_list] == ["Check ", " today."]
    assert not any("\u4e00" <= char <= "\u9fff" for char in result)


@pytest.mark.parametrize("token", ["NVIDIA", "nvidia", "Nvidia", "Qwen", "qwen", "SQL", "sql", "airbnb"])
def test_hotword_casing_cannot_change_fst_protection(frontend, token):
    english, _, _ = frontend
    fst = Mock(side_effect=lambda text: text.replace(token, "错误中文").strip())
    assert english.normalize_preserving_english_pronunciations("Use " + token + ".", fst) == "Use " + token + " ."
    assert [call.args[0] for call in fst.call_args_list] == ["Use ", "."]


def test_ordinary_lowercase_words_quantities_and_unknown_codes_keep_tn_context(frontend):
    english, _, _ = frontend
    text = "The driver needs 2 GB for 12 usd on October 14 at 5:30 PM; unknown17 uses 2GHz."
    fst = Mock(side_effect=str.strip)
    assert english.normalize_preserving_english_pronunciations(text, fst) == text
    fst.assert_called_once_with(text)


@pytest.mark.parametrize("invalid", [[], {"word": ["W", "ER1", "D"]}, {"WORD": []}, {"WORD": ["NOTAPHONE"]}])
def test_invalid_overlay_fails_explicitly(frontend, tmp_path, invalid):
    english, _, _ = frontend
    (tmp_path / "en_gold_pronunciations.json").write_text(json.dumps(invalid), encoding="utf-8")
    english.current_file_path = str(tmp_path)
    english.gold_en_pronunciations.cache_clear()
    with pytest.raises(ValueError, match="English gold"):
        english.gold_en_pronunciations()
