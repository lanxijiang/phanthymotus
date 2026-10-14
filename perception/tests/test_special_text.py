"""Address integrity and compatibility with the author's normal lexicon."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plugins.vits2_tts_trt.frontend.special_text import normalize_special_text, verbalize_chinese_number
from plugins.vits2_tts_trt.frontend.lexicon_rules import apply_lexeme_rules


def number(value, language):
    return {"ZH": {"2": "二", "3": "三", "7": "七", "2.5": "二点五"},
            "EN": {"2": "two", "3": "three", "7": "seven", "2.5": "two point five"}}[language][value]


class SpecialTextTests(unittest.TestCase):
    def normalize(self, text):
        return normalize_special_text(text, apply_lexeme_rules, number)

    def test_ordinary_text_uses_original_lexicon(self):
        for text in ("使用Qwen3、A100、H800和CNCF。", "Tier 1 IDC AutoML PhanRouter",
                     "在OpenMLDB里打开FeatInsight的SQL Playground。"):
            self.assertEqual(self.normalize(text), apply_lexeme_rules(text))

    def test_mail_local_part_is_not_a_model_name(self):
        text = self.normalize("请联系qwen3+tts_2@4paradigm.com。")
        self.assertIn("Q W E N 三 加号 T T S 下划线 二 艾特 四 P A R A D I G M 点 C O M", text)
        self.assertNotIn("千问", text)
        self.assertTrue(text.endswith("。"))

    def test_english_address_uses_english_symbols(self):
        text = self.normalize("Please email api_2+qa@example.com.")
        self.assertIn("A P I underscore two plus Q A at E X A M P L E dot C O M", text)
        self.assertFalse(any("\u4e00" <= c <= "\u9fff" for c in text))
        self.assertTrue(text.endswith("."))

    def test_url_path_query_and_fragment_are_preserved(self):
        text = self.normalize("请访问https://www.4paradigm.com/v2?a=3&b=2#top。")
        for phrase in ("H T T P S 冒号 斜杠 斜杠 W W W 点 四", "斜杠 V 二",
                       "问号 A 等号 三 与号 B 等号 二 井号 T O P"):
            self.assertIn(phrase, text)

    def test_url_period_is_outside_address(self):
        text = self.normalize("Visit https://example.com/docs.")
        self.assertTrue(text.endswith("D O C S ."))

    def test_preserve_balanced_url_parentheses(self):
        text = self.normalize("Visit https://example.com/a(b).")
        self.assertIn("A left parenthesis B right parenthesis", text)

    def test_quantity_is_cardinal_and_decimal_digits_are_preserved(self):
        for raw, spoken in (("512", "五百一十二"), ("3.05", "三点零五"),
                            ("0.005", "零点零零五"), ("-2.5", "负二点五"),
                            ("10", "十"), ("101", "一百零一"),
                            ("10001", "一万零一"), ("10012", "一万零一十二"),
                            ("100000010", "一亿零一十"), ("0", "零")):
            with self.subTest(raw=raw):
                self.assertEqual(verbalize_chinese_number(raw), spoken)


if __name__ == "__main__":
    unittest.main(verbosity=2)
