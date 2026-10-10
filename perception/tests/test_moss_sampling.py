"""Compare the optional sampler to the original MOSS sampling algorithm."""

from types import SimpleNamespace

import numpy as np
import pytest

from plugins.moss_deploy.sampling import make_fast_generation_method


# Reference oracle transcribed from the reviewed ort_cpu_runtime.py. Keep its
# Python sort/loops and float64 probability arithmetic independent of the fast
# implementation; production always clones the actual original methods.
def _softmax(values):
    shifted = np.asarray(values - float(np.max(values)), dtype=np.float64)
    exps = np.exp(shifted)
    return exps / np.sum(exps, dtype=np.float64)


def _sample_from_scores(values, *, do_sample, temperature, top_k, top_p, rng):
    if not do_sample:
        return int(np.argmax(values))
    if not temperature > 0:
        raise ValueError("temperature must be positive when do_sample=True")
    scores = np.asarray(values, dtype=np.float32).copy() / float(temperature)
    if top_k > 0 and top_k < scores.shape[0]:
        threshold = float(np.sort(scores)[::-1][top_k - 1])
        scores[scores < threshold] = float("-inf")
    if top_p > 0 and top_p < 1:
        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda item: item[1], reverse=True)
        sorted_probs = _softmax(np.asarray([item[1] for item in indexed], dtype=np.float32))
        remove = [False] * len(indexed)
        cumulative = 0.0
        for index, probability in enumerate(sorted_probs):
            cumulative += float(probability)
            if cumulative > float(top_p):
                remove[index] = True
        for index in range(len(remove) - 1, 0, -1):
            remove[index] = remove[index - 1]
        if remove:
            remove[0] = False
        for index, removed in enumerate(remove):
            if removed:
                scores[indexed[index][0]] = float("-inf")
    probabilities = _softmax(scores)
    value = float(rng.random())
    for index, probability in enumerate(probabilities):
        value -= float(probability)
        if value <= 0:
            return int(index)
    return int(np.argmax(scores))


def _sample_audio_token(values, previous_ids, previous_set, defaults, rng):
    # This fixture's generation state exercises duplicate histories and both
    # sampled and greedy calls. The production clone retains its own original
    # repetition-penalty helper without any replacement.
    scores = values.copy()
    penalty = defaults["audio_repetition_penalty"]
    for index in set(previous_ids):
        if 0 <= index < len(scores):
            scores[index] = scores[index] * penalty if scores[index] < 0 else scores[index] / penalty
    return _sample_from_scores(scores, do_sample=defaults["do_sample"],
                               temperature=defaults["audio_temperature"],
                               top_k=defaults["audio_top_k"], top_p=defaults["audio_top_p"], rng=rng)


def _sample_assistant_text_token(values, manifest, defaults, rng):
    ids = np.asarray([manifest["continue"], manifest["end"]], dtype=np.int32)
    index = _sample_from_scores(values[ids], do_sample=defaults["do_sample"],
                                temperature=defaults["text_temperature"],
                                top_k=min(defaults["text_top_k"], 2),
                                top_p=defaults["text_top_p"], rng=rng)
    return int(ids[index])


class ReferenceRuntime:
    def generate_audio_frames(self, rows):
        history = [[] for _ in range(16)]
        seen = [set() for _ in range(16)]
        result = []
        for text_logits, audio_logits in rows:
            token = _sample_assistant_text_token(text_logits, self.manifest, self.defaults, self.rng)
            if token == self.manifest["end"]:
                break
            frame = []
            for channel in range(16):
                value = _sample_audio_token(audio_logits[channel], history[channel], seen[channel], self.defaults, self.rng)
                history[channel].append(value)
                seen[channel].add(value)
                frame.append(value)
            result.append(frame)
        return result


@pytest.fixture
def fast():
    return make_fast_generation_method(ReferenceRuntime.generate_audio_frames).__globals__["_sample_from_scores"]


class FixedRandom:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def test_exact_tokens_at_ties_cutoffs_and_random_boundaries(fast):
    generator = np.random.default_rng(1234)
    cases = 0
    for size in (2, 1024):
        for top_k in (0, 1, 25, 1024):
            for top_p in (0, 0.05, 0.5, 0.95, 1):
                for temperature in (0.1, 0.8, 1):
                    for form in ("random", "ties", "equal", "extreme"):
                        values = generator.standard_normal(size).astype(np.float32)
                        if form == "ties":
                            values = np.round(values)
                        elif form == "equal":
                            values[:] = 0
                        elif form == "extreme":
                            values *= 100
                        original_values = values.copy()
                        for random_value in (0, 1e-16, 0.1, 0.5, np.nextafter(1., 0.)):
                            left, right = FixedRandom(random_value), FixedRandom(random_value)
                            kwargs = dict(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p)
                            assert fast(values, rng=right, **kwargs) == _sample_from_scores(values, rng=left, **kwargs)
                            assert left.calls == right.calls == 1
                            cases += 1
                        np.testing.assert_array_equal(values, original_values)
    assert cases == 2400


def test_seeded_audio_draws_and_generator_state_are_exact(fast):
    rows = np.random.default_rng(234).standard_normal((50, 1024)).astype(np.float32)
    first, second = np.random.default_rng(567), np.random.default_rng(567)
    kwargs = dict(do_sample=True, temperature=.8, top_k=25, top_p=.95)
    for index in range(1000):
        values = rows[index % len(rows)]
        assert fast(values, rng=second, **kwargs) == _sample_from_scores(values, rng=first, **kwargs)
    assert first.bit_generator.state == second.bit_generator.state


@pytest.mark.parametrize("values,temperature", [
    ([np.nan, 1, 2], 1), ([np.inf, 1, 2], 1), ([-np.inf, 1, 2], 1),
    ([-np.inf, -np.inf], 1), ([1e30, -1e30], 1e-30),
])
def test_nonfinite_and_overflow_use_original_behavior(fast, values, temperature):
    left, right = FixedRandom(.4), FixedRandom(.4)
    kwargs = dict(do_sample=True, temperature=temperature, top_k=1, top_p=.95)
    with np.errstate(all="ignore"):
        assert fast(np.array(values), rng=right, **kwargs) == _sample_from_scores(np.array(values), rng=left, **kwargs)
    assert right.calls == left.calls


def test_greedy_and_invalid_temperature_do_not_consume_rng(fast):
    rng = FixedRandom(.5)
    assert fast(np.array([1, 3, 3]), do_sample=False, temperature=0, top_k=1, top_p=.9, rng=rng) == 1
    for temperature in (0, -1, np.nan):
        with pytest.raises(ValueError, match="temperature must be positive"):
            fast(np.array([1, 2]), do_sample=True, temperature=temperature, top_k=1, top_p=.9, rng=rng)
    assert rng.calls == 0


@pytest.mark.parametrize("do_sample", [False, True])
def test_scoped_generation_matches_history_and_keeps_original_globals(do_sample):
    original = ReferenceRuntime.generate_audio_frames
    before = {name: original.__globals__[name] for name in ("_sample_from_scores", "_sample_audio_token", "_sample_assistant_text_token")}
    optimized = make_fast_generation_method(original)
    assert optimized.__code__ is original.__code__
    assert optimized.__globals__ is not original.__globals__
    assert all(original.__globals__[name] is value for name, value in before.items())
    defaults = dict(do_sample=do_sample, audio_temperature=.8, audio_top_k=25,
                    audio_top_p=.95, audio_repetition_penalty=1.2,
                    text_temperature=1., text_top_k=50, text_top_p=1.)
    first = SimpleNamespace(defaults=defaults, manifest={"continue": 1, "end": 0}, rng=np.random.default_rng(42))
    second = SimpleNamespace(defaults=defaults, manifest=first.manifest, rng=np.random.default_rng(42))
    generator = np.random.default_rng(343)
    rows = [(np.array([-100., 100.]), generator.standard_normal((16, 1024)).astype(np.float32)) for _ in range(16)]
    assert original(first, rows) == optimized(second, rows)
    assert first.rng.bit_generator.state == second.rng.bit_generator.state
    assert all(original.__globals__[name] is value for name, value in before.items())


@pytest.mark.parametrize("environment,expected", [(None, True), ("0", False), ("1", True)])
def test_runtime_sampling_default_is_scoped_and_reported(monkeypatch, environment, expected):
    from plugins.moss_deploy import runtime

    if environment is None:
        monkeypatch.delenv("MOSS_FAST_SAMPLING", raising=False)
    else:
        monkeypatch.setenv("MOSS_FAST_SAMPLING", environment)
    original_method = ReferenceRuntime.generate_audio_frames
    monkeypatch.setattr(runtime, "load_original_source", lambda path: (
        SimpleNamespace(OnnxTtsRuntime=ReferenceRuntime),
        SimpleNamespace(OrtCpuRuntime=ReferenceRuntime),
    ))
    pool = SimpleNamespace(assets=SimpleNamespace(source_dir=None),
                           backend="tensorrt-device-0", resident_state_enabled=True)
    adapted_class = runtime._create_original_adapter_class(pool)
    assert adapted_class.fast_sampling_enabled is expected
    assert (adapted_class.generate_audio_frames is not original_method) is expected
    assert ReferenceRuntime.generate_audio_frames is original_method
    adapted = object.__new__(adapted_class)
    adapted.codec_streaming_session = None
    adapted.manifest = {"generation_defaults": {}}
    model = object.__new__(runtime.MossRuntime)
    model.pool, model.instance_id, model._runtime = pool, "test-runtime", adapted
    monkeypatch.setenv("MOSS_FAST_SAMPLING", "0" if expected else "1")
    monkeypatch.setenv("MOSS_TRT_RESIDENT_STATE", "0")
    assert model.state_summary()["effective_optimizations"] == {
        "resident_state": True, "fast_sampling": expected, "fused_frame": False,
    }
