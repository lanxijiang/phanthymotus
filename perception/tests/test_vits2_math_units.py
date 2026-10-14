"""Targeted normalization spans without ROS, GPU, or frontend dependencies."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/frontend/math_units.py"
SPEC = importlib.util.spec_from_file_location("vits2_math_units_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathUnitsTest(unittest.TestCase):
    def setUp(self):
        self.numbers = Mock(side_effect=lambda value, language: f"NUM({value})")

    def spans(self, text, language="ZH"):
        return MODULE.find_spans(text, language, self.numbers)

    def spoken(self, text, language="ZH"):
        return self.spans(text, language)[0][2]

    def test_explicit_math_and_equation(self):
        self.assertEqual(self.spoken("x^2+3=7"), "x 的平方 加 NUM(3) 等于 NUM(7)")
        self.assertEqual(self.spoken("x=2"), "x 等于 NUM(2)")
        self.assertEqual(self.spoken("πr²"), "派 乘以 r 的平方")
        self.assertEqual(self.spoken("x^2+3=7", "EN"), "x squared plus NUM(3) equals NUM(7)")

    def test_decimal_spelling_reaches_number_callback_unchanged(self):
        self.assertEqual(self.spoken("3.05kg"), "NUM(3.05) 千克")
        self.numbers.assert_called_once_with("3.05", "ZH")
        self.numbers.reset_mock()
        self.assertEqual(self.spoken("3.00 kg", "EN"), "NUM(3.00) kilograms")
        self.numbers.assert_called_once_with("3.00", "EN")

    def test_common_units_in_both_languages(self):
        units = {
            "℃": ("摄氏度", "degrees Celsius"),
            "°C": ("摄氏度", "degrees Celsius"),
            "°F": ("华氏度", "degrees Fahrenheit"),
            "kg": ("千克", "kilograms"), "g": ("克", "grams"),
            "mg": ("毫克", "milligrams"), "km/h": ("千米每小时", "kilometers per hour"),
            "m/s": ("米每秒", "meters per second"), "m²": ("平方米", "square meters"),
            "m/s²": ("米每二次方秒", "meters per second squared"),
            "cm²": ("平方厘米", "square centimeters"), "m³": ("立方米", "cubic meters"),
            "kg/m³": ("千克每立方米", "kilograms per cubic meter"),
            "GB": ("G B", "gigabytes"), "MB": ("M B", "megabytes"),
            "KB": ("K B", "kilobytes"), "degrees Celsius": ("摄氏度", "degrees Celsius"),
        }
        for unit, (zh, en) in units.items():
            with self.subTest(unit=unit):
                self.assertEqual(self.spoken(f"2.5 {unit}"), f"NUM(2.5) {zh}")
                self.assertEqual(self.spoken(f"2.5 {unit}", "EN"), f"NUM(2.5) {en}")
        self.assertEqual(self.spoken("2.5%"), "百分之 NUM(2.5)")
        self.assertEqual(self.spoken("2.5%", "EN"), "NUM(2.5) percent")
        self.assertEqual(self.spoken("1kg", "EN"), "NUM(1) kilogram")

    def test_parentheses_comparators_negative_values_and_powers(self):
        self.assertEqual(self.spoken("(x+2)^3=27"),
                         "左括号 x 加 NUM(2) 右括号 的立方 等于 NUM(27)")
        self.assertEqual(self.spoken("x>=-2"), "x 大于等于 负 NUM(2)")
        self.assertEqual(self.spoken("x^-2=0.25"), "x 的 负 NUM(2) 次方 等于 NUM(0.25)")
        self.assertEqual(self.spoken("x!=2", "EN"), "x not equal to NUM(2)")
        self.assertEqual(self.spoken("3-x=2"), "NUM(3) 减 x 等于 NUM(2)")

    def test_model_names_dates_phone_brand_and_ordinary_text_are_untouched(self):
        for text in ["A100", "H800", "Qwen3", "PhanRouter", "2026-10-13", "2026/10/13",
                     "1-800-FLOWERS.COM", "I/O", "The price is 123.",
                     "The API is ready.", "3.05kgModel", "100MBps"]:
            with self.subTest(text=text):
                self.assertEqual(self.spans(text), [])

    def test_unsupported_unit_suffix_is_not_partially_consumed(self):
        self.assertEqual(self.spans("2kg/day"), [])

    def test_spans_preserve_original_offsets_and_do_not_overlap(self):
        text = "公式x=2，重3.05kg，面积πr²。"
        spans = self.spans(text)
        self.assertEqual([text[start:end] for start, end, _ in spans], ["x=2", "3.05kg", "πr²"])
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])))

    def test_excessively_long_expression_stays_with_existing_frontend(self):
        self.assertEqual(self.spans("+".join(["2"] * 1000)), [])


if __name__ == "__main__":
    unittest.main()
