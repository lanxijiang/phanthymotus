"""Optional faster host sampling; model execution and RNG remain unchanged."""

from __future__ import annotations

from types import FunctionType

import numpy as np


def _make_score_sampler(reference, softmax):
    def sample(values, *, do_sample, temperature, top_k, top_p, rng):
        if not do_sample or not (temperature > 0):
            return reference(values, do_sample=do_sample, temperature=temperature,
                             top_k=top_k, top_p=top_p, rng=rng)
        scores = np.asarray(values, dtype=np.float32).copy() / float(temperature)
        # Preserve the original treatment of unusual input, overflow and NaN.
        # No random number has been consumed before taking this fallback.
        if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
            return reference(values, do_sample=do_sample, temperature=temperature,
                             top_k=top_k, top_p=top_p, rng=rng)
        if 0 < top_k < scores.shape[0]:
            position = scores.shape[0] - top_k
            threshold = float(np.partition(scores, position)[position])
            # The original retains all scores tied at the kth threshold.
            scores[scores < threshold] = float("-inf")
        if 0 < top_p < 1:
            # Python's original stable sort breaks equal-score ties by original
            # index. Negation + stable argsort preserves exactly that order.
            order = np.argsort(-scores, kind="stable")
            sorted_probs = softmax(scores[order])
            remove = np.empty(scores.size, dtype=bool)
            remove[0] = False
            remove[1:] = np.cumsum(sorted_probs, dtype=np.float64)[:-1] > float(top_p)
            scores[order[remove]] = float("-inf")
        probabilities = softmax(scores)
        random_value = float(rng.random())
        # Keep the original boundary behavior even if token zero is masked.
        if random_value == 0:
            return 0
        # Subtract in token-index order, just as the original loop does.
        # Omitting exact zero terms preserves floating-point subtraction order
        # for all nonzero terms; a cumulative-sum search would not do so.
        for index in np.flatnonzero(probabilities):
            random_value -= float(probabilities[index])
            if random_value <= 0:
                return int(index)
        return int(np.argmax(scores))

    return sample


def _clone(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = dict(function.__kwdefaults__) if function.__kwdefaults__ else None
    clone.__annotations__ = dict(function.__annotations__)
    clone.__qualname__ = function.__qualname__
    clone.__dict__.update(function.__dict__)
    return clone


def make_fast_generation_method(original_method):
    """Bind the original loop to private sampling helpers, without patching it.

    The caller opts in with MOSS_FAST_SAMPLING=1 when constructing its adapted
    runtime class. Original model calls, repetition penalties, greedy/fixed
    branches and RNG draw counts are retained. Other runtime classes keep their
    original globals and behavior.
    """
    if not isinstance(original_method, FunctionType):
        raise TypeError("Expected the original Python MOSS generation method")
    namespace = dict(original_method.__globals__)
    helpers = ("_sample_from_scores", "_softmax", "_sample_audio_token", "_sample_assistant_text_token")
    if any(not isinstance(namespace.get(name), FunctionType) for name in helpers):
        raise ValueError("Original MOSS sampling helpers have changed")
    namespace["_sample_from_scores"] = _make_score_sampler(
        namespace["_sample_from_scores"], namespace["_softmax"],
    )
    for name in ("_sample_audio_token", "_sample_assistant_text_token"):
        namespace[name] = _clone(namespace[name], namespace)
    return _clone(original_method, namespace)
