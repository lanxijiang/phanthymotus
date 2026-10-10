"""Real VITS2 English frontend regressions; no GPU synthesis is needed.

Run directly with Python, or collect with pytest; no test packages are needed.
Set VITS2_MODEL_DIR to the installed release. Missing dependencies/assets skip
the test class. VITS2_PERCEPTION_DIR optionally points at the perception source
when this file is copied outside the repository for a device test.
Do not import g2p_en before configure_release_paths: that can trigger a download.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

PERCEPTION_DIR = Path(os.getenv("VITS2_PERCEPTION_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PERCEPTION_DIR))
MODEL_DIR = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))

ENGLISH_CASES = [
    ("It is 23 degrees Celsius.", "It is twenty-three degrees Celsius."),
    ("It weighs 3.5 kilograms.", "It weighs three point five kilograms."),
    ("It weighs 3.5kilograms.", "It weighs three point five kilograms."),
    ("The reading is 3.05.", "The reading is three point zero five."),
    ("The reading is 3.14.", "The reading is three point one four."),
    ("The reading is 0.005.", "The reading is zero point zero zero five."),
    ("Please arrive at 9:30.", "Please arrive at nine thirty."),
    ("Please arrive at 9:05.", "Please arrive at nine oh five."),
    ("Please arrive at 9:00.", "Please arrive at nine o'clock."),
    ("Please arrive at 9:30am.", "Please arrive at nine thirty am."),
    ("It costs $3.05.", "It costs three dollars, five cents."),
    ("It costs $3.5.", "It costs three dollars, fifty cents."),
    ("It costs $.05.", "It costs five cents."),
    ("This is the 21st example.", "This is the twenty-first example."),
    ("Open the SQL Playground and check the last join.",
     "Open the S Q L Playground and check the last join."),
    ("Open the S Q L Playground.", "Open the S Q L Playground."),
    ("OpenMLDB and FeatInsight use a catalog login.",
     "OpenMLDB and FeatInsight use a catalog login."),
]

CHINESE_CASES = [
    "今天气温是23度。", "重量是3.5千克。", "请在9:30到达。",
]

MIXED_WORD_CASES = [
    ("重量是 3.5 kilograms。", ("三点五", "kilograms")),
    ("重量是3.5kilograms。", ("三点五", "kilograms")),
    ("打开 catalog 和 login 页面。", ("catalog", "login")),
    ("请看 catalogue、logarithm 和 biology。", ("catalogue", "logarithm", "biology")),
    ("请打开 LOGin 页面。", ("LOGin",)),
    ("请说 open catalog now。", ("open catalog now",)),
]


class TestVits2EnglishNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name for name in
            ("jieba", "pypinyin", "wetext", "kaldifst", "g2p_en", "inflect", "nltk")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise unittest.SkipTest(
                f"VITS2 frontend dependencies missing: {', '.join(missing)}"
            )
        if not (MODEL_DIR / "tn_cache").is_dir():
            raise unittest.SkipTest(f"no VITS2 release at {MODEL_DIR}")

        from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths

        configure_release_paths(MODEL_DIR)

        from plugins.vits2_tts_trt.frontend import chinese
        from plugins.vits2_tts_trt.frontend.cleaner import (
            g2p_normalized_text_mix,
            normalize_text_mix,
        )

        cls.chinese = chinese
        cls.normalize = staticmethod(normalize_text_mix)
        cls.g2p = staticmethod(g2p_normalized_text_mix)

    def test_english_normalization_and_pronunciation(self):
        for text, expected in ENGLISH_CASES:
            with self.subTest(text=text):
                normalized = self.normalize(text)
                self.assertEqual(normalized, expected)
                self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in normalized))
                self.assertFalse(any(char.isdigit() for char in normalized))
                phones, tones, languages, word2ph = self.g2p(normalized)
                self.assertGreater(len(phones), 2)
                self.assertNotIn("UNK", phones)
                self.assertEqual(len(phones), len(tones))
                self.assertEqual(len(phones), len(languages))
                self.assertEqual(len(phones), sum(word2ph))
                self.assertEqual(set(languages), {"EN"})

    def test_chinese_normalization_is_unchanged(self):
        for text in CHINESE_CASES:
            with self.subTest(text=text):
                normalized = self.normalize(text)
                self.assertEqual(normalized, self.chinese.text_normalize(text))
                _, _, languages, _ = self.g2p(normalized)
                self.assertEqual(set(languages), {"ZH"})

    def test_mixed_english_words_are_not_math_operators(self):
        for text, expected_parts in MIXED_WORD_CASES:
            with self.subTest(text=text):
                normalized = self.normalize(text)
                for part in expected_parts:
                    self.assertIn(part, normalized)
                self.assertNotIn("对数", normalized)
                phones, tones, languages, word2ph = self.g2p(normalized)
                self.assertGreater(len(phones), 2)
                self.assertNotIn("UNK", phones)
                self.assertEqual(len(phones), len(tones))
                self.assertEqual(len(phones), len(languages))
                self.assertEqual(len(phones), sum(word2ph))
                self.assertEqual(set(languages), {"ZH", "EN"})

    def test_standalone_log_still_expands_next_to_protected_words(self):
        normalized = self.normalize("请计算 log(10)，然后打开 catalog。")
        self.assertIn("对数", normalized)
        self.assertIn("十", normalized)
        self.assertIn("catalog", normalized)
        self.assertEqual(normalized.count("对数"), 1)
        self.g2p(normalized)

    def test_number_units_keep_their_context(self):
        for text, expected_parts in [
            ("传输 2GB/秒。", ("每秒两G B",)),
            ("重量是 3.5kg。", ("三点五", "千克")),
            ("请在 catalog 中查看，传输 2GB/秒。", ("catalog", "每秒两G B")),
        ]:
            with self.subTest(text=text):
                normalized = self.normalize(text)
                for part in expected_parts:
                    self.assertIn(part, normalized)
                self.g2p(normalized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
