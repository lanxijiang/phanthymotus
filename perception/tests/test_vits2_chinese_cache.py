"""Chinese segment reuse: bounded request scope and unchanged pronunciation.

The real frontend tests require its pinned dependencies and VITS2_MODEL_DIR.
The remaining cache tests use no model or GPU.
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
    predict_chinese_segment,
    predict_english_token,
    use_pronunciation_cache,
)


class TestChineseSegmentCache(unittest.TestCase):
    def setUp(self):
        self.calls = Counter()

        def predictor(segment):
            self.calls[segment] += 1
            return list(segment), [0] * len(segment), [1] * len(segment)

        self.predictor = predictor

    def predict(self, segment="你好,"):
        return predict_chinese_segment(self.predictor, segment)

    def test_no_context_keeps_original_calls_and_list_type(self):
        result = self.predict()
        self.assertTrue(all(isinstance(values, list) for values in result))
        self.assertEqual(self.predict(), result)
        self.assertEqual(self.calls["你好,"], 2)

    def test_exact_punctuation_context_and_predictor_identity(self):
        alternate_calls = []

        def alternate(segment):
            alternate_calls.append(segment)
            return ["different"], [5], [1]

        with use_pronunciation_cache(PronunciationCache()):
            for segment in ("你好,", "你好.", "你好", "你好吗,", "你好,", "你好."):
                self.predict(segment)
            self.assertEqual(predict_chinese_segment(alternate, "你好,"),
                             (("different",), (5,), (1,)))
            predict_chinese_segment(alternate, "你好,")
        self.assertEqual(self.calls, {"你好,": 1, "你好.": 1, "你好": 1, "你好吗,": 1})
        self.assertEqual(alternate_calls, ["你好,"])

    def test_values_are_immutable_and_detached_from_predictor_lists(self):
        source = (["n", "i", ","], [2, 2, 0], [2, 1])
        predictor = lambda segment: source
        with use_pronunciation_cache(PronunciationCache()):
            result = predict_chinese_segment(predictor, "你,")
            source[0][:] = ["modified"]
            source[1][0] = 99
            source[2].append(100)
            self.assertEqual(result, (("n", "i", ","), (2, 2, 0), (2, 1)))
            self.assertEqual(predict_chinese_segment(predictor, "你,"), result)
            with self.assertRaises(TypeError):
                result[1][0] = 3

    def test_lru_eviction_and_long_segment_bypass(self):
        cache = PronunciationCache(max_chinese_entries=2, max_chinese_segment_chars=3)
        with use_pronunciation_cache(cache):
            for segment in ("你好,", "您好,", "你好,", "再见,", "您好,", "非常感谢,", "非常感谢,"):
                self.predict(segment)
        self.assertEqual(self.calls, {"你好,": 1, "您好,": 2, "再见,": 1, "非常感谢,": 2})

    def test_chinese_and_english_eviction_are_independent(self):
        english_calls = []

        def english(token):
            english_calls.append(token)
            return ["AH0"]

        cache = PronunciationCache(max_entries=1, max_chinese_entries=1)
        with use_pronunciation_cache(cache):
            predict_english_token(english, "phonemizer")
            self.predict("你好,")
            self.predict("再见,")
            predict_english_token(english, "phonemizer")
            predict_english_token(english, "hyperparameter")
            self.predict("再见,")
        self.assertEqual(self.calls, {"你好,": 1, "再见,": 1})
        self.assertEqual(english_calls, ["phonemizer", "hyperparameter"])

    def test_context_isolation_and_exception_restore(self):
        parent = PronunciationCache()
        with use_pronunciation_cache(parent):
            self.predict()
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with use_pronunciation_cache(PronunciationCache()):
                    self.predict()
                    raise RuntimeError("abort")
            self.predict()
        self.assertEqual(self.calls["你好,"], 2)
        self.predict()
        with use_pronunciation_cache(PronunciationCache()):
            self.predict()
        self.assertEqual(self.calls["你好,"], 4)

    def test_failed_conversion_is_not_cached(self):
        calls = []

        def predictor(segment):
            calls.append(segment)
            if len(calls) == 1:
                raise ValueError("pinyin")
            return ["n", "i"], [3, 3], [2]

        with use_pronunciation_cache(PronunciationCache()):
            with self.assertRaisesRegex(ValueError, "pinyin"):
                predict_chinese_segment(predictor, "你")
            result = predict_chinese_segment(predictor, "你")
            self.assertEqual(predict_chinese_segment(predictor, "你"), result)
        self.assertEqual(calls, ["你", "你"])

    def test_copied_contexts_reuse_one_conversion(self):
        calls = []

        def predictor(segment):
            calls.append(segment)
            time.sleep(0.01)
            return ["n", "i"], [3, 3], [2]

        with use_pronunciation_cache(PronunciationCache()):
            contexts = [copy_context() for _ in range(4)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda context: context.run(
                    predict_chinese_segment, predictor, "你"), contexts))
        self.assertEqual(calls, ["你"])
        self.assertEqual(results, [(("n", "i"), (3, 3), (2,))] * 4)

    def test_invalid_chinese_limits_are_rejected(self):
        for limits in ({"max_chinese_entries": 0}, {"max_chinese_segment_chars": -1}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                PronunciationCache(**limits)


class TestRealChineseSegmentCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in ("jieba", "pypinyin", "numpy", "wetext", "kaldifst")
                   if importlib.util.find_spec(name) is None]
        if missing or not (MODEL_DIR / "tn_cache").is_dir():
            raise unittest.SkipTest(f"real frontend needs release/dependencies: {missing}")
        from plugins.vits2_tts_trt.frontend.release_paths import configure_release_paths

        configure_release_paths(MODEL_DIR)
        from plugins.vits2_tts_trt.frontend import chinese

        cls.chinese = chinese

    def test_complete_outputs_preserve_tones_and_word_alignment(self):
        texts = [
            "你好,你好.你好吗?",
            "我不是不想去,我想一想,买一本书,看一看书.",
            "银行行长重新安排了重庆的行程.银行行长重新安排了重庆的行程.",
            "小朋友很友好,可以给我两碗米饭吗?",
            "妈妈看看小朋友,你好!妈妈看看小朋友,你好!",
            "你好,,你好...",
            "你 好,\n你 好.",
            "",
        ]
        before = [self.chinese.g2p(text) for text in texts]
        with use_pronunciation_cache(PronunciationCache()):
            after = [self.chinese.g2p(text) for text in texts]
            repeated = [self.chinese.g2p(text) for text in texts]
        self.assertEqual(after, before)
        self.assertEqual(repeated, before)

    def test_candidate_merges_reuse_only_original_segment_boundaries(self):
        candidates = ["你好,", "我想买一本书.", "你好,我想买一本书.", "你好!", "你好,再见."]
        before = [self.chinese.g2p(text) for text in candidates]
        with patch.object(self.chinese, "_g2p_segment", wraps=self.chinese._g2p_segment) as counted:
            with use_pronunciation_cache(PronunciationCache()):
                after = [self.chinese.g2p(text) for text in candidates]
            self.assertEqual([call.args[0] for call in counted.call_args_list],
                             ["你好,", "我想买一本书.", "你好!", "再见."])
        self.assertEqual(after, before)

    def test_mutating_complete_outputs_cannot_corrupt_cached_segment(self):
        text = "你好,我想买一本书."
        before = self.chinese.g2p(text)
        with use_pronunciation_cache(PronunciationCache()):
            phones, tones, alignment = self.chinese.g2p(text)
            phones[:] = ["modified"]
            tones[:] = [99]
            alignment[:] = [999]
            self.assertEqual(self.chinese.g2p(text), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
