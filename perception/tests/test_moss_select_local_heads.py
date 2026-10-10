"""Graph rewrite checks use tiny saved weights; never run a model or CUDA."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest


@pytest.fixture
def selector():
    path = Path(__file__).resolve().parents[1] / 'tools/moss_select_local_heads.py'
    spec = importlib.util.spec_from_file_location('_moss_select_local_heads_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def assets(tmp_path):
    directory = tmp_path / 'assets'
    directory.mkdir()
    weights = [np.arange(30, dtype=np.float16).reshape(3, 10),
               np.arange(12, dtype=np.float16).reshape(3, 4),
               (100 + np.arange(12, dtype=np.float16)).reshape(3, 4)]
    initializers = []
    offset = 0
    with (directory / 'models.data').open('wb') as destination:
        for index, array in enumerate(weights):
            destination.write(array.tobytes())
            tensor = TensorProto(name=f'onnx::MatMul_{806 + index}', data_type=TensorProto.FLOAT16,
                                 dims=array.shape, data_location=TensorProto.EXTERNAL)
            for key, value in {'location': 'models.data', 'offset': str(offset),
                               'length': str(array.nbytes)}.items():
                tensor.external_data.add(key=key, value=value)
            initializers.append(tensor)
            offset += array.nbytes
    initializers.append(numpy_helper.from_array(np.asarray([1], dtype=np.int64), 'axes'))
    inputs = [helper.make_tensor_value_info('global_hidden', TensorProto.FLOAT, ['batch', 3])]
    inputs += [helper.make_tensor_value_info(name, TensorProto.INT32, ['batch']) for name in
               ('text_token_id', 'audio_token_id', 'channel_index', 'step_type', 'past_valid_lengths')]
    inputs += [helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', 'past', 3]) for name in
               ('local_past_key_0', 'local_past_value_0')]
    nodes = [helper.make_node('Cast', ['global_hidden'], ['hidden'], name='original_body', to=TensorProto.FLOAT16),
             helper.make_node('MatMul', ['hidden', 'onnx::MatMul_806'], ['text_half'], name='/text_lm_head/MatMul'),
             helper.make_node('Cast', ['text_half'], ['text_logits'], name='text_output', to=TensorProto.FLOAT)]
    for index in range(2):
        nodes += [helper.make_node('MatMul', ['hidden', f'onnx::MatMul_{807 + index}'],
                                   [f'audio_half_{index}'], name=f'/audio_lm_heads.{index}/MatMul'),
                  helper.make_node('Cast', [f'audio_half_{index}'], [f'audio_float_{index}'],
                                   name=f'audio_cast_{index}', to=TensorProto.FLOAT),
                  helper.make_node('Unsqueeze', [f'audio_float_{index}', 'axes'], [f'audio_{index}'], name=f'expand_{index}')]
    nodes.append(helper.make_node('Concat', ['audio_0', 'audio_1'], ['audio_logits'], name='audio_output', axis=1))
    outputs = [helper.make_tensor_value_info('text_logits', TensorProto.FLOAT, ['batch', 10]),
               helper.make_tensor_value_info('audio_logits', TensorProto.FLOAT, ['batch', 2, 4])]
    for kind in ('key', 'value'):
        name = f'local_present_{kind}_0'
        nodes.append(helper.make_node('Identity', [f'local_past_{kind}_0'], [name], name='original_' + kind))
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ['batch', 'past', 3]))
    model = helper.make_model(helper.make_graph(nodes, 'fixture', inputs, outputs, initializers),
                              opset_imports=[helper.make_opsetid('', 17)])
    onnx.save_model(model, directory / 'local_cached_step.onnx')
    config = {'n_vq': 2, 'hidden_size': 3, 'audio_codebook_sizes': [4, 4], 'vocab_size': 10,
              'audio_assistant_slot_token_id': 9, 'audio_end_token_id': 7}
    (directory / 'tts_browser_onnx_meta.json').write_text(json.dumps({'model_config': config}))
    samples = []
    for index, (step, channel) in enumerate(((0, 0), (1, 0), (2, 0), (2, 0))):
        name = f'original-{index}.npz'
        np.savez(directory / name, input__step_type=np.array([step], dtype=np.int32),
                 input__channel_index=np.array([channel], dtype=np.int32),
                 ref__text_logits=np.arange(10, dtype=np.float32).reshape(1, 10),
                 ref__audio_logits=np.arange(8, dtype=np.float32).reshape(1, 2, 4),
                 ref__local_present_key_0=np.array([index], dtype=np.float32),
                 ref__local_present_value_0=np.array([index + 1], dtype=np.float32))
        samples.append(name)
    (directory / 'analysis.json').write_text(json.dumps({'graphs': [{
        'name': 'local_cached_step', 'file': 'local_cached_step.onnx', 'validation_samples': samples,
        'precision': 'fp16_with_fp32_io_and_sensitive_ops', 'profiles': {}}]}))
    return directory


def test_export_preserves_body_and_selects_reference_channels(selector, assets, tmp_path):
    source_before = {path.name: path.read_bytes() for path in assets.iterdir()}
    output = tmp_path / 'selected'
    contract = selector.export(assets, output)
    assert contract['text_token_ids'] == [9, 7]
    assert contract['source_onnx_sha256'] == selector.sha256(assets / 'local_cached_step.onnx')
    assert contract['selected_onnx_sha256'] == selector.sha256(output / 'local_cached_step.onnx')
    model = onnx.load(output / 'local_cached_step.onnx', load_external_data=False)
    assert [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim][1:] == [2]
    assert [dim.dim_value for dim in model.graph.output[1].type.tensor_type.shape.dim][1:] == [4]
    assert sum(node.op_type == 'MatMul' for node in model.graph.node) == 2
    original = onnx.load(assets / 'local_cached_step.onnx', load_external_data=False)
    assert [x.SerializeToString() for x in model.graph.input] == [x.SerializeToString() for x in original.graph.input]
    old_body = {node.name: node.SerializeToString() for node in original.graph.node if node.name.startswith('original_')}
    new_body = {node.name: node.SerializeToString() for node in model.graph.node if node.name.startswith('original_')}
    assert new_body == old_body
    for index, expected_channel in enumerate((0, 0, 1, 1)):
        with np.load(output / f'validation/sample-{index:02d}.npz') as sample:
            np.testing.assert_array_equal(sample['ref__text_logits'], [[9, 7]])
            np.testing.assert_array_equal(sample['ref__audio_logits'], np.arange(8).reshape(1, 2, 4)[:, expected_channel])
            np.testing.assert_array_equal(sample['ref__local_present_key_0'], [index])
    assert (output / 'models.data').read_bytes() == (assets / 'models.data').read_bytes()
    assert {path.name: path.read_bytes() for path in assets.iterdir()} == source_before
    onnx.checker.check_model(str(output / 'local_cached_step.onnx'))


def test_noncontiguous_weights_are_rejected_before_output(selector, assets, tmp_path):
    path = assets / 'local_cached_step.onnx'
    model = onnx.load(path, load_external_data=False)
    weight = next(value for value in model.graph.initializer if value.name == 'onnx::MatMul_808')
    next(value for value in weight.external_data if value.key == 'offset').value = '62'
    onnx.save_model(model, path)
    with pytest.raises(ValueError, match='contiguous'):
        selector.export(assets, tmp_path / 'selected')
    assert not (tmp_path / 'selected').exists()


def test_existing_output_is_not_overwritten(selector, assets, tmp_path):
    output = tmp_path / 'selected'
    output.mkdir()
    sentinel = output / 'evidence.json'
    sentinel.write_text('preserve')
    with pytest.raises(FileExistsError):
        selector.export(assets, output)
    assert sentinel.read_text() == 'preserve'


def test_step_two_selects_next_output_channel(selector):
    assert selector.selected_channel(0, 0, 16) == 0
    assert selector.selected_channel(1, 0, 16) == 0
    assert [selector.selected_channel(2, i, 16) for i in range(15)] == list(range(1, 16))
    for step, channel in ((3, 0), (2, 15), (2, -1), (0, 1), (1, 1)):
        with pytest.raises(ValueError):
            selector.selected_channel(step, channel, 16)
