"""Request-local English OOV cache regressions; no GPU is needed.

The optional real-frontend class also checks complete phone/tone outputs. Set
VITS2_MODEL_DIR to the release and install its frontend dependencies to run it.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import importlib.util
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch


PERCEPTION_DIR = Path(os.getenv("VITS2_PERCEPTION_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PERCEPTION_DIR))
MODEL_DIR = Path(os.getenv("VITS2_MODEL_DIR", "/models/vits2"))

from plugins.vits2_tts_trt.frontend.pronunciation_cache import (
    PronunciationCache,
    predict_english_token,
    use_pronunciation_cache,
)


class TestPronunciationCache(unittest.TestCase):
    def setUp(self):
        self.calls = Counter()

        def predictor(token):
            self.calls[token] += 1
            return ["HH", "AY1", " ", token]

        self.predictor = predictor

    def predict(self, token="Phonemizer"):
        return predict_english_token(self.predictor, token)

    def test_no_context_keeps_original_calls_and_result_type(self):
        first = self.predict()
        self.assertIsInstance(first, list)
        self.assertEqual(self.predict(), first)
        self.assertEqual(self.calls["Phonemizer"], 2)

    def test_exact_case_and_predictor_are_distinct(self):
        second_calls = []

        def second(token):
            second_calls.append(token)
            return ["DIFFERENT"]

        with use_pronunciation_cache(PronunciationCache()):
            self.assertEqual(self.predict(), self.predict())
            self.assertEqual(self.predict("phonemizer"), self.predict("phonemizer"))
            self.assertEqual(predict_english_token(second, "Phonemizer"), ("DIFFERENT",))
            self.assertEqual(predict_english_token(second, "Phonemizer"), ("DIFFERENT",))
        self.assertEqual(self.calls, {"Phonemizer": 1, "phonemizer": 1})
        self.assertEqual(second_calls, ["Phonemizer"])

    def test_cached_value_is_not_the_predictors_mutable_list(self):
        source = ["AH0", " ", "T"]
        predictor = lambda token: source
        with use_pronunciation_cache(PronunciationCache()):
            phones = predict_english_token(predictor, "token")
            source[0] = "MODIFIED"
            self.assertEqual(phones, ("AH0", " ", "T"))
            self.assertEqual(predict_english_token(predictor, "token"), phones)
            with self.assertRaises(TypeError):
                phones[0] = "MODIFIED"

    def test_lru_bound_and_long_tokens_bypass_cache(self):
        with use_pronunciation_cache(PronunciationCache(max_entries=2, max_token_chars=4)):
            for token in ("one", "two", "one", "tri", "two", "longword", "longword"):
                self.predict(token)
        self.assertEqual(self.calls, {"one": 1, "two": 2, "tri": 1, "longword": 2})

    def test_context_restores_parent_and_clears_after_exception(self):
        with use_pronunciation_cache(PronunciationCache()):
            self.predict()
            with self.assertRaisesRegex(RuntimeError, "test"):
                with use_pronunciation_cache(PronunciationCache()):
                    self.predict()
                    raise RuntimeError("test")
            self.predict()
        self.assertEqual(self.calls["Phonemizer"], 2)
        self.predict()
        with use_pronunciation_cache(PronunciationCache()):
            self.predict()
        self.assertEqual(self.calls["Phonemizer"], 4)

    def test_prediction_error_is_not_cached(self):
        attempts = []

        def predictor(token):
            attempts.append(token)
            if len(attempts) == 1:
                raise RuntimeError("failed")
            return ["AH0"]

        with use_pronunciation_cache(PronunciationCache()):
            with self.assertRaises(RuntimeError):
                predict_english_token(predictor, "word")
            self.assertEqual(predict_english_token(predictor, "word"), ("AH0",))
            self.assertEqual(predict_english_token(predictor, "word"), ("AH0",))
        self.assertEqual(len(attempts), 2)

    def test_copied_contexts_share_cache_safely(self):
        calls = []

        def slow_predictor(token):
            calls.append(token)
            time.sleep(0.02)
            return ["AH0", "T"]

        with use_pronunciation_cache(PronunciationCache()):
            contexts = [copy_context() for _ in range(8)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(
                    lambda context: context.run(predict_english_token, slow_predictor, "word"),
                    contexts,
                ))
        self.assertEqual(results, [("AH0", "T")] * 8)
        self.assertEqual(calls, ["word"])

    def test_context_is_not_implicitly_shared_with_a_new_thread(self):
        with use_pronunciation_cache(PronunciationCache()):
            self.predict()
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertIsInstance(pool.submit(self.predict).result(), list)
            self.predict()
        self.assertEqual(self.calls["Phonemizer"], 2)


class TestRealEnglishPronunciationCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name for name in ("g2p_en", "inflect", "nltk", "numpy")
            if importlib.util.find_spec(name) is None
        ]
        if missing or not (MODEL_DIR / "frontend_data").is_dir():
            raise unittest.SkipTest(f"real frontend needs release/dependencies: {missing}")
        from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths

        configure_release_paths(MODEL_DIR)
        from plugins.vits2_tts_trt.frontend import english, unified_g2p

        cls.english = english
        cls.mixed = unified_g2p

    def test_real_token_phones_and_tones_match_both_uncached_paths(self):
        tokens = ["phonemizer", "Phonemizer", "hyperparameter", "A", "a", "GB", "IELTS"]
        for frontend, function, predictor_name in (
            (self.english, self.english.g2p_token, "_g2p"),
            (self.mixed, self.mixed._en_token_to_phones, "_g2p_en"),
        ):
            with self.subTest(frontend=frontend.__name__):
                before = [function(token) for token in tokens]
                predictor = getattr(frontend, predictor_name)
                with patch.object(frontend, predictor_name, wraps=predictor) as counted:
                    with use_pronunciation_cache(PronunciationCache()):
                        after = [function(token) for token in tokens]
                        calls_once = counted.call_count
                        repeated = [function(token) for token in tokens]
                    self.assertGreater(calls_once, 0)
                    self.assertEqual(counted.call_count, calls_once)
                self.assertEqual(after, before)
                self.assertEqual(repeated, before)

    def test_mutating_mixed_caller_lists_cannot_change_the_cached_pronunciation(self):
        word = "phonemizer"
        before = self.mixed._en_token_to_phones(word)
        with use_pronunciation_cache(PronunciationCache()):
            phones, tones = self.mixed._en_token_to_phones(word)
            phones[:] = ["MODIFIED"]
            tones[:] = [99]
            self.assertEqual(self.mixed._en_token_to_phones(word), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
