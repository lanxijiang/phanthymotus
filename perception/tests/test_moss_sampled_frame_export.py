"""Graph and sampler tests; only tiny synthetic arrays, no model inference."""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import pytest


@pytest.fixture
def exporter():
    path = Path(__file__).resolve().parents[1] / 'tools/moss_export_sampled_frame.py'
    spec = importlib.util.spec_from_file_location('_moss_frame_export_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sampler(exporter, vocabulary, *, temperature=1.0, top_k=0, top_p=1.0):
    graph = helper.make_graph([], 'tiny_sampling_test', [
        helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, vocabulary]),
        helper.make_tensor_value_info('u', TensorProto.FLOAT, [1, 1]),
    ], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    build = exporter.GraphBuilder(model)
    output = build.sample('logits', 'u', vocabulary=vocabulary, temperature=temperature,
                          top_k=top_k, top_p=top_p, prefix='sample/test/')
    model.graph.output.append(helper.make_tensor_value_info(output, TensorProto.INT32, [1]))
    onnx.checker.check_model(model)
    runtime = ReferenceEvaluator(model)
    return lambda scores, u: int(runtime.run(None, {
        'logits': np.asarray(scores, dtype=np.float32).reshape(1, vocabulary),
        'u': np.asarray([[u]], dtype=np.float32),
    })[0].item())


def reference_sample(scores, u, *, temperature, top_k, top_p):
    scores = np.asarray(scores, dtype=np.float32).copy() / np.float32(temperature)
    if 0 < top_k < len(scores):
        threshold = np.sort(scores)[-top_k]
        scores[scores < threshold] = -np.inf
    if 0 < top_p < 1:
        order = np.argsort(-scores, kind='stable')
        values = scores[order]
        probabilities = np.exp(values - np.max(values))
        probabilities /= np.sum(probabilities, dtype=np.float32)
        previous = np.concatenate((np.zeros(1, np.float32), np.cumsum(probabilities[:-1], dtype=np.float32)))
        scores[order[previous > np.float32(top_p)]] = -np.inf
    probabilities = np.exp(scores - np.max(scores))
    probabilities /= np.sum(probabilities, dtype=np.float32)
    cdf = np.cumsum(probabilities, dtype=np.float32)
    cdf /= cdf[-1]
    selected = int(np.sum(cdf < np.float32(u)))
    return selected if selected < len(scores) else int(np.argmax(scores))


def test_kth_ties_preserved_in_original_token_order(exporter):
    sample = sampler(exporter, 6, top_k=2)
    values = [9, 8, 8, 8, 7, 0]
    selected = {sample(values, u) for u in np.linspace(.001, .999, 100)}
    assert selected == {0, 1, 2, 3}
    # The highest score is token 3, but u=0.1 still addresses token-ID CDF.
    sample = sampler(exporter, 4, top_k=4)
    assert sample([0, 0, 0, 1], .1) == 0


def test_topp_equal_previous_boundary_is_retained(exporter):
    sample = sampler(exporter, 4, top_p=.5)
    assert sample([0, 0, 0, 0], .9) == 2


def test_zero_uniform_matches_original_masked_token_zero_boundary(exporter):
    sample = sampler(exporter, 4, top_k=1)
    assert sample([-100, -100, 9, -100], 0) == 0
    assert sample([-100, -100, 9, -100], np.nextafter(np.float32(0), np.float32(1))) == 2


def test_seeded_sampling_agrees_with_independent_float32_reference(exporter):
    sample = sampler(exporter, 64, temperature=1.7, top_k=25, top_p=.8)
    rng = np.random.default_rng(19)
    for _ in range(100):
        # Quantization creates deliberate ties around top-k and top-p boundaries.
        scores = np.round(rng.normal(size=64), 1).astype(np.float32)
        u = np.float32(rng.random())
        assert sample(scores, u) == reference_sample(scores, u, temperature=1.7, top_k=25, top_p=.8)


def tiny_step_model():
    """Small stateful decoder that makes incorrect frame wiring observable."""
    rng = np.random.default_rng(52)
    nodes, initializers = [], []
    def constant(name, array):
        initializers.append(numpy_helper.from_array(np.asarray(array), name))
        return name
    axes = constant('axes', np.array([1], np.int64))
    zero = constant('zero', np.array(0, np.float32))
    gain = constant('gain', np.array(.001, np.float32))
    combined = 'global_hidden'
    # Depend on every scalar input, including channel/step/past_valid_lengths.
    for name in ('text_token_id', 'audio_token_id', 'channel_index', 'step_type', 'past_valid_lengths'):
        nodes += [helper.make_node('Cast', [name], [name + '_float'], to=TensorProto.FLOAT),
                  helper.make_node('Mul', [name + '_float', gain], [name + '_small']),
                  helper.make_node('Add', [combined, name + '_small'], [name + '_sum'])]
        combined = name + '_sum'
    nodes += [helper.make_node('ReduceSum', ['local_past_key_0'], ['past_sum'], keepdims=0),
              helper.make_node('Mul', ['past_sum', gain], ['past_scaled']),
              helper.make_node('Add', [combined, 'past_scaled'], ['body']),
              helper.make_node('Cast', ['body'], ['hidden'], to=TensorProto.FLOAT16)]
    head_weights = [rng.normal(size=(4, 10)).astype(np.float16)]
    head_weights += [rng.normal(size=(4, 4)).astype(np.float16) for _ in range(2)]
    for index, weight in enumerate(head_weights):
        name = 'text_lm_head' if index == 0 else 'audio_lm_heads.%d' % (index - 1)
        w = constant('head_weight_%d' % index, weight)
        half = 'head_half_%d' % index
        final = 'text_logits' if index == 0 else 'audio_float_%d' % (index - 1)
        nodes += [helper.make_node('MatMul', ['hidden', w], [half], name='/' + name + '/MatMul'),
                  helper.make_node('Cast', [half], [final], to=TensorProto.FLOAT)]
        if index:
            nodes.append(helper.make_node('Unsqueeze', [final, axes], ['audio_%d' % (index - 1)]))
    nodes += [helper.make_node('Concat', ['audio_0', 'audio_1'], ['audio_logits'], axis=1)]
    expanded_axes = constant('kv_axes', np.array([1, 2], np.int64))
    nodes.append(helper.make_node('Unsqueeze', ['body', expanded_axes], ['kv_current']))
    for kind in ('key', 'value'):
        nodes.append(helper.make_node('Concat', ['local_past_%s_0' % kind, 'kv_current'], ['local_present_%s_0' % kind], axis=1))
    inputs = [helper.make_tensor_value_info('global_hidden', TensorProto.FLOAT, ['batch', 4])]
    inputs += [helper.make_tensor_value_info(name, TensorProto.INT32, ['batch']) for name in
               ('text_token_id', 'audio_token_id', 'channel_index', 'step_type', 'past_valid_lengths')]
    inputs += [helper.make_tensor_value_info('local_past_%s_0' % kind, TensorProto.FLOAT, ['batch', 'past', 1, 4]) for kind in ('key', 'value')]
    outputs = [helper.make_tensor_value_info('text_logits', TensorProto.FLOAT, ['batch', 10]),
               helper.make_tensor_value_info('audio_logits', TensorProto.FLOAT, ['batch', 2, 4])]
    outputs += [helper.make_tensor_value_info('local_present_%s_0' % kind, TensorProto.FLOAT, ['batch', 'total', 1, 4]) for kind in ('key', 'value')]
    model = helper.make_model(helper.make_graph(nodes, 'tiny_decoder', inputs, outputs, initializers),
                              opset_imports=[helper.make_opsetid('', 17)])
    config = {'n_vq': 2, 'hidden_size': 4, 'audio_codebook_sizes': [4, 4], 'vocab_size': 10,
              'audio_assistant_slot_token_id': 9, 'audio_end_token_id': 7}
    return model, config


def test_unrolled_frame_matches_sequential_teacher_forced_tiny_graph(exporter):
    source, config = tiny_step_model()
    unrolled, contract = exporter.rewrite(source, config)
    onnx.checker.check_model(unrolled)
    graph = ReferenceEvaluator(unrolled)
    step_runtime = ReferenceEvaluator(source)
    hidden = np.array([[.2, -.1, .6, -.7]], np.float32)
    random_u = np.array([[.3, .9, .2]], np.float32)
    continuation, tokens, text, audio = graph.run(None, {'global_hidden': hidden, 'random_u': random_u})
    assert continuation.shape == (1, 1)
    assert tokens.shape == (1, 2)
    previous = {f'local_past_{kind}_0': np.empty((1, 0, 1, 4), np.float32) for kind in ('key', 'value')}
    next_text = next_audio = 0
    for step in range(3):
        feeds = dict(global_hidden=hidden, text_token_id=np.array([next_text], np.int32),
                     audio_token_id=np.array([next_audio], np.int32), channel_index=np.array([max(0, step - 2)], np.int32),
                     step_type=np.array([min(step, 2)], np.int32), past_valid_lengths=np.array([step], np.int32), **previous)
        actual = step_runtime.run(None, feeds)
        if not step:
            np.testing.assert_array_equal(actual[0][:, [9, 7]], text)
            next_text = 9 if continuation.item() else 7
        else:
            np.testing.assert_array_equal(actual[1][:, step - 1], audio[:, step - 1])
            next_audio = int(tokens[0, step - 1])
        previous = {'local_past_key_0': actual[2], 'local_past_value_0': actual[3]}
    original_weights = {value.name: value.SerializeToString() for value in source.graph.initializer}
    assert all(value.SerializeToString() == original_weights[value.name] for value in unrolled.graph.initializer if value.name in original_weights)
    assert len({v.name for v in unrolled.graph.initializer}) == len(unrolled.graph.initializer)
    assert contract['generation_defaults']['audio_temperature'] == 1.7
    assert contract['generation_defaults']['audio_top_p'] == .8
    assert contract['generation_defaults']['audio_repetition_penalty'] == 1.0
    assert not contract['embedding_switch_specialized']


def test_existing_directory_and_unexpected_interface_are_rejected(exporter, tmp_path):
    output = tmp_path / 'preserved'
    output.mkdir()
    marker = output / 'important.txt'
    marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        exporter.export(tmp_path, output)
    assert marker.read_text() == 'unchanged'
    model, config = tiny_step_model()
    model.graph.input.append(helper.make_tensor_value_info('surprise', TensorProto.FLOAT, [1]))
    with pytest.raises(ValueError, match='interface'):
        exporter.rewrite(model, config)


def test_near_one_uniform_selects_last_supported_token(exporter):
    sample = sampler(exporter, 1024, temperature=1.7, top_k=25, top_p=.8)
    scores = np.full(1024, -20, np.float32)
    scores[[11, 100, 215, 876]] = [1.37, 1.2, 1.1, 1.0]
    expected = reference_sample(scores, np.nextafter(np.float32(1), np.float32(0)), temperature=1.7, top_k=25, top_p=.8)
    assert sample(scores, np.nextafter(np.float32(1), np.float32(0))) == expected
    assert expected == 876


@pytest.mark.parametrize("size", [1, 3, 6, 1024])
@pytest.mark.parametrize("exclusive", [False, True])
def test_static_prefix_scan_matches_exact_integer_sums_without_loops(exporter, size, exclusive):
    model = helper.make_model(helper.make_graph([], 'prefix_scan_test', [
        helper.make_tensor_value_info('values', TensorProto.FLOAT, [1, size]),
    ], []), opset_imports=[helper.make_opsetid('', 17)])
    build = exporter.GraphBuilder(model)
    result = build.prefix_scan('values', vocabulary=size, name='sample/scan', exclusive=exclusive)
    model.graph.output.append(helper.make_tensor_value_info(result, TensorProto.FLOAT, [1, size]))
    onnx.checker.check_model(model)
    assert not any(node.op_type in ('Loop', 'Scan', 'CumSum') for node in model.graph.node)
    values = np.arange(size, dtype=np.float32).reshape(1, size)
    expected = np.cumsum(values, axis=1, dtype=np.float32)
    if exclusive:
        expected = np.concatenate([np.zeros((1, 1), np.float32), expected[:, :-1]], axis=1)
    actual = ReferenceEvaluator(model).run(None, {'values': values})[0]
    np.testing.assert_array_equal(actual, expected)


def test_sparse_parallel_sampling_cannot_return_masked_tokens_near_one(exporter):
    sample = sampler(exporter, 1024, temperature=1.7, top_k=25, top_p=.8)
    rng = np.random.default_rng(724)
    for _ in range(20):
        scores = rng.normal(size=1024).astype(np.float32)
        uniform = np.nextafter(np.float32(1), np.float32(0))
        expected = reference_sample(scores, uniform, temperature=1.7, top_k=25, top_p=.8)
        assert sample(scores, uniform) == expected
