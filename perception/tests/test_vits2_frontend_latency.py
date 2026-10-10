"""VITS2 frontend reuse regressions, without loading a GPU engine.

Run directly with Python or collect with pytest. The request-cache tests use a
deterministic fake engine. The optional real-frontend tests require the release
and its frontend dependencies; set VITS2_MODEL_DIR and VITS2_PERCEPTION_DIR when
running this file separately on Jetson. Actual PCM equivalence is a device test.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import os
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

PERCEPTION_DIR = Path(os.getenv("VITS2_PERCEPTION_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PERCEPTION_DIR))
MODEL_DIR = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))

from plugins.vits2_tts_trt.adapter import Vits2TensorRTAdapter, WARMUP_CASES
from plugins.vits2_tts_trt.frontend.chunking import iter_text_chunks


class _FakeEngine:
    max_text_tokens = 12

    def __init__(self):
        self.calls = Counter()
        self.generation = 0

    def _get_text_ids(self, text, *, normalized=False):
        assert normalized
        self.calls[text] += 1
        ids = tuple(ord(char) + self.generation * 100000 for char in text)
        return ids, (0,) * len(ids), (2,) * len(ids)

    def synthesize(self, text, *, text_ids, length_scale):
        # Deterministic payload detects changed IDs, order or speed. This is
        # deliberately a fake payload, not a GPU/quality/PCM verification.
        return repr((text, text_ids, length_scale)).encode("utf-8")


class TestRequestLocalIds(unittest.TestCase):
    def setUp(self):
        self.engine = _FakeEngine()
        self.adapter = Vits2TensorRTAdapter(engine=self.engine, max_chunk_tokens=12)
        cleaner = types.ModuleType("plugins.vits2_tts_trt.frontend.cleaner")
        cleaner.normalize_text_mix = lambda text: text
        self.stub = patch.dict(sys.modules, {cleaner.__name__: cleaner})
        self.stub.start()
        self.addCleanup(self.stub.stop)

    def _uncached_chunks(self, text):
        engine = _FakeEngine()
        return [
            (chunk, engine._get_text_ids(chunk, normalized=True))
            for chunk in iter_text_chunks(
                text, lambda value: len(engine._get_text_ids(value, normalized=True)[0]), 12
            )
        ]

    def test_reuses_ids_without_changing_chunks_or_id_values(self):
        for text in ("Hello.", "Hello. Go! Hola.", "你好。 Hello. 好的。"):
            with self.subTest(text=text):
                self.engine.calls.clear()
                actual = list(self.adapter.iter_text_chunks(text))
                self.assertEqual(actual, self._uncached_chunks(text))
                self.assertTrue(self.engine.calls)
                self.assertTrue(all(count == 1 for count in self.engine.calls.values()))

    def test_cache_does_not_survive_a_request(self):
        before = list(self.adapter.iter_text_chunks("Hello."))
        self.engine.generation += 1
        after = list(self.adapter.iter_text_chunks("Hello."))
        self.assertEqual(self.engine.calls["Hello."], 2)
        self.assertNotEqual(before[0][1], after[0][1])

    def test_stream_keeps_engine_payload_and_chunk_order(self):
        text = "Hello. Go! Hola."
        expected = b"".join(
            self.engine.synthesize(chunk, text_ids=ids, length_scale=1.0)
            for chunk, ids in self._uncached_chunks(text)
        )
        with patch("plugins.vits2_tts_trt.adapter.CHUNK_PAUSE_MS", 0):
            self.assertEqual(b"".join(self.adapter.synthesize_stream(text)), expected)


class TestRealMixedFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name for name in
            ("jieba", "pypinyin", "wetext", "kaldifst", "g2p_en", "inflect", "nltk")
            if importlib.util.find_spec(name) is None
        ]
        if missing or not (MODEL_DIR / "tn_cache").is_dir():
            raise unittest.SkipTest(f"real frontend needs release/dependencies: {missing}")
        from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths

        configure_release_paths(MODEL_DIR)
        from plugins.vits2_tts_trt.frontend import english, unified_g2p
        from plugins.vits2_tts_trt.frontend.cleaner import normalize_text_mix

        cls.english = english
        cls.mixed = unified_g2p
        cls.normalize = staticmethod(normalize_text_mix)

    def test_shared_predictor_matches_the_previous_independent_predictor(self):
        from g2p_en import G2p

        self.assertIs(self.mixed._g2p_en, self.english._g2p)
        texts = [
            self.normalize("请检查 phonemizer 的功能。"),
            self.normalize("打开 hyperparameter 页面，读取 GB 和 IELTS。"),
        ]
        # Reproduce the old mixed-only instance while keeping the production
        # mixed tokenization, phone conversion, tones and language tags intact.
        with patch.object(self.mixed, "_g2p_en", G2p()):
            before = [self.mixed.unified_g2p(text) for text in texts]
        after = [self.mixed.unified_g2p(text) for text in texts]
        self.assertEqual(after, before)

    def test_warmup_covers_english_and_mixed_dictionary_misses(self):
        covered = set()
        for text in WARMUP_CASES:
            normalized = self.normalize(text)
            has_zh = any("\u4e00" <= char <= "\u9fff" for char in normalized)
            words = re.findall(r"[A-Za-z]+", normalized)
            if any(word.upper() not in self.english.eng_dict for word in words):
                covered.add("mixed" if has_zh else "english")
        self.assertEqual(covered, {"english", "mixed"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
