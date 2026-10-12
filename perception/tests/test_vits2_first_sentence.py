"""Opt-in sentence-first packing; no model/voice quality is simulated here."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

PERCEPTION_DIR = Path(os.getenv("VITS2_PERCEPTION_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PERCEPTION_DIR))

from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter
from plugins.vits2_tts_trt.frontend.chunking import first_sentence_end, iter_text_chunks
from plugins.vits2_tts_trt.frontend.pronunciation_cache import predict_english_token


class TestFirstSentenceBoundary(unittest.TestCase):
    def test_complete_sentence_includes_commas_and_closing_quotes(self):
        examples = (
            ("机器人已经准备好,请确认道路安全.接下来开始工作.", "机器人已经准备好,请确认道路安全."),
            ("你好，先检查电量。现在可以出发！", "你好，先检查电量。"),
            ("Please check the battery, then start. Keep watching.", "Please check the battery, then start."),
            ('He said, "Ready!" Then we left.', 'He said, "Ready!"'),
            ("他说：“准备好了吗？”然后我们出发。", "他说：“准备好了吗？”"),
        )
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(text[:first_sentence_end(text)], expected)

    def test_abbreviations_initials_and_domains_do_not_end_first_sentence(self):
        examples = (
            "Dr. Smith is ready. Next task.",
            "Prof. Wang has arrived. Next task.",
            "J. R. Smith is ready. Next task.",
            "The U.S. office is ready. Next task.",
            "The Ph.D. student is ready. Next task.",
            "Use e.g. this example. Next task.",
            "The value is 3.14 today. Next task.",
            "Visit example.com before leaving. Next task.",
            "Contact user@example.com for details. Next task.",
            "Wait... then leave. Next task.",
        )
        for text in examples:
            with self.subTest(text=text):
                expected = text.index(" Next task.")
                self.assertEqual(first_sentence_end(text), expected)

    def test_uncertain_boundary_falls_back_conservatively(self):
        for text in (
            "e.g. this example", "Dr. Smith", "The U.S. team", "Visit example.com/path",
            "Wait... then leave", "This is done. please continue", "No punctuation here",
        ):
            with self.subTest(text=text):
                self.assertIsNone(first_sentence_end(text))


class TestSentenceFirstPacking(unittest.TestCase):
    def test_only_first_sentence_uses_new_boundary(self):
        text = "你好,准备好了.现在开始.接下来检查.完成了."
        actual = list(iter_text_chunks(text, len, 16, first_sentence=True))
        self.assertEqual(actual[0], "你好,准备好了.")
        self.assertEqual(actual[1:], list(iter_text_chunks(text[len(actual[0]):], len, 16)))
        self.assertEqual("".join(actual), text)

    def test_first_yield_does_not_inspect_later_sentence_ids(self):
        visited = []

        def count(text):
            visited.append(text)
            return len(text)

        stream = iter_text_chunks("First sentence. Later sentence. Another sentence.", count, 40, first_sentence=True)
        self.assertEqual(next(stream), "First sentence.")
        self.assertEqual(visited, ["First sentence."])

    def test_overlong_first_sentence_uses_existing_safe_split(self):
        text = "Please check the battery and network before starting. Then continue."
        original = list(iter_text_chunks(text, len, 15))
        candidate = list(iter_text_chunks(text, len, 15, first_sentence=True))
        self.assertEqual(candidate, original)
        self.assertTrue(all(len(chunk) <= 15 for chunk in candidate))
        self.assertEqual("".join(candidate), text)

    def test_single_sentence_or_ambiguous_text_keeps_existing_packing(self):
        for text in ("Please check, then start.", "Dr. Smith", "Keep watching", "你好，检查一下。"):
            with self.subTest(text=text):
                self.assertEqual(
                    list(iter_text_chunks(text, len, 24, first_sentence=True)),
                    list(iter_text_chunks(text, len, 24)),
                )

    def test_default_disabled_path_matches_captured_greedy_output(self):
        examples = (
            ("Hello. Go! Hola.", 12, ["Hello. Go!", " Hola."]),
            ("你好。 Hello. 好的。", 12, ["你好。 Hello.", " 好的。"]),
            ("hi! bye? later.", 7, ["hi!", " bye?", " later."]),
        )
        for text, limit, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(list(iter_text_chunks(text, len, limit)), expected)
                self.assertEqual(list(iter_text_chunks(text, len, limit, first_sentence=False)), expected)

    def test_text_is_preserved_and_each_chunk_fits(self):
        for text in (
            "你好,准备好了.现在开始.下一项任务较长,请认真检查.",
            'He said, "Ready!" Then continue, carefully. We are finished.',
            "Dr. Smith is ready. The next task needs a longer explanation. Continue.",
            "Visit example.com before leaving. Next task.",
        ):
            for limit in (8, 16, 64):
                with self.subTest(text=text, limit=limit):
                    chunks = list(iter_text_chunks(text, len, limit, first_sentence=True))
                    self.assertEqual("".join(chunks), text)
                    self.assertTrue(all(len(chunk) <= limit for chunk in chunks))


class _FakeEngine:
    max_text_tokens = 256

    def __init__(self):
        self.calls = Counter()

    def _get_text_ids(self, text, *, normalized=False):
        assert normalized
        self.calls[text] += 1
        ids = tuple(ord(char) for char in text)
        return ids, (0,) * len(ids), (2,) * len(ids)


class TestAdapterSentenceOption(unittest.TestCase):
    def test_option_is_disabled_by_default_and_constructor_overrides_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Vits2TensorRTAdapter(engine=_FakeEngine()).first_sentence)
        with patch.dict(os.environ, {"MIX_VITS_FIRST_SENTENCE": "1"}):
            self.assertTrue(Vits2TensorRTAdapter(engine=_FakeEngine()).first_sentence)
            self.assertFalse(Vits2TensorRTAdapter(engine=_FakeEngine(), first_sentence=False).first_sentence)
        with patch.dict(os.environ, {"MIX_VITS_FIRST_SENTENCE": "0"}):
            self.assertTrue(Vits2TensorRTAdapter(engine=_FakeEngine(), first_sentence=True).first_sentence)

    def test_adapter_reuses_ids_and_cache_never_leaks_across_yield(self):
        cleaner = types.ModuleType("plugins.vits2_tts_trt.frontend.cleaner")
        cleaner.normalize_text_mix = lambda text: text
        predictions = []

        def predictor(token):
            predictions.append(token)
            return ["AH0"]

        engine = _FakeEngine()
        original = engine._get_text_ids

        def convert(text, *, normalized=False):
            predict_english_token(predictor, "repeated")
            return original(text, normalized=normalized)

        engine._get_text_ids = convert
        adapter = Vits2TensorRTAdapter(engine=engine, first_sentence=True)
        with patch.dict(sys.modules, {cleaner.__name__: cleaner}):
            stream = adapter.iter_text_chunks("First sentence. Next sentence. Last sentence.")
            first, _ = next(stream)
            self.assertEqual(first, "First sentence.")
            self.assertEqual(predictions, ["repeated"])
            predict_english_token(predictor, "repeated")
            remaining = list(stream)
            self.assertEqual(first + "".join(text for text, _ in remaining), "First sentence. Next sentence. Last sentence.")
            self.assertEqual(predictions, ["repeated", "repeated"])
            self.assertTrue(all(count == 1 for count in engine.calls.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
