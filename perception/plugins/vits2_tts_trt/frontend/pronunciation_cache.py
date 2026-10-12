"""Bounded, request-local reuse of existing pronunciation computations.

Only exact tokens passed to ``g2p_en`` are cached. Dictionary overrides, token
case, normalization, Chinese tone sandhi and phone conversion stay with their
existing callers. Chinese reuse covers a complete existing punctuation segment
after its original word segmentation and tone sandhi; it does not split words
or alter those rules. The context is active only during a frontend call, never
across a streaming generator's yield.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import threading


_active_cache = ContextVar("vits2_english_pronunciation_cache", default=None)


class PronunciationCache:
    """Small LRU shared only by the frontend calls of a single request.

    Predictor identity is part of the key, so replacing a predictor cannot
    reuse another predictor's output. ``g2p_en.G2p`` instances are identity
    hashable. English values are immutable phone tuples; Chinese values hold
    immutable phone, tone and alignment tuples in a separate LRU. The lock
    also prevents duplicate work if a context is copied to a thread.
    """

    def __init__(self, max_entries=512, max_token_chars=256,
                 max_chinese_entries=256, max_chinese_segment_chars=512):
        self.max_entries = int(max_entries)
        self.max_token_chars = int(max_token_chars)
        self.max_chinese_entries = int(max_chinese_entries)
        self.max_chinese_segment_chars = int(max_chinese_segment_chars)
        if min(self.max_entries, self.max_token_chars, self.max_chinese_entries,
               self.max_chinese_segment_chars) <= 0:
            raise ValueError("pronunciation cache limits must be positive")
        self._entries = OrderedDict()
        self._chinese_entries = OrderedDict()
        self._lock = threading.Lock()

    def predict(self, predictor, token):
        # Do not retain arbitrarily long tokens merely because entry count is
        # bounded. The uncached path still uses the original predictor.
        if len(token) > self.max_token_chars:
            return tuple(predictor(token))
        key = (predictor, token)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            phones = tuple(predictor(token))
            self._entries[key] = phones
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return phones

    def predict_chinese_segment(self, predictor, segment):
        """Reuse a whole independent Chinese segment, preserving all context.

        Keep the three output sequences immutable. The Chinese caller extends
        fresh lists from them, so downstream mutations cannot poison a hit.
        A separate LRU leaves English OOV reuse unaffected by Chinese input.
        """
        if len(segment) > self.max_chinese_segment_chars:
            return tuple(tuple(values) for values in predictor(segment))
        key = (predictor, segment)
        with self._lock:
            if key in self._chinese_entries:
                self._chinese_entries.move_to_end(key)
                return self._chinese_entries[key]
            result = tuple(tuple(values) for values in predictor(segment))
            self._chinese_entries[key] = result
            if len(self._chinese_entries) > self.max_chinese_entries:
                self._chinese_entries.popitem(last=False)
            return result


@contextmanager
def use_pronunciation_cache(cache):
    """Use one request's cache for this frontend call; restore on every exit."""
    previous = _active_cache.set(cache)
    try:
        yield
    finally:
        _active_cache.reset(previous)


def predict_english_token(predictor, token):
    """Return the original token prediction, reusing it within a request only."""
    cache = _active_cache.get()
    if cache is None:
        return predictor(token)
    return cache.predict(predictor, token)


def predict_chinese_segment(predictor, segment):
    """Reuse only identical full segments within the current request."""
    cache = _active_cache.get()
    if cache is None:
        return predictor(segment)
    return cache.predict_chinese_segment(predictor, segment)
