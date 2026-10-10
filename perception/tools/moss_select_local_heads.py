#!/usr/bin/env python3
"""Export a candidate local decoder that projects only consumed output heads.

No model inference is performed. Inputs, Transformer nodes and KV outputs stay
unchanged. Existing full-head references are sliced into candidate references;
TensorRT numerical validation and deterministic audio comparison remain required.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PREFIX = 'moss_selected/'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_channel(step_type: int, channel_index: int, channels: int) -> int:
    if step_type not in (0, 1, 2):
        raise ValueError(f'Unsupported step_type: {step_type}')
    if step_type in (0, 1):
        if channel_index != 0:
            raise ValueError('Steps zero and one require input channel_index zero')
        return 0
    if not 0 <= channel_index < channels - 1:
        raise ValueError(f'Previous audio channel out of range: {channel_index}')
    return channel_index + 1


def external_info(tensor):
    fields = {entry.key: entry.value for entry in tensor.external_data}
    location = Path(fields.get('location', ''))
    if (tensor.data_type != TensorProto.FLOAT16 or tensor.data_location != TensorProto.EXTERNAL
            or not fields.get('location') or location.is_absolute() or '..' in location.parts):
        raise ValueError(f'Expected relative external FP16 weights: {tensor.name}')
    offset, length = int(fields['offset']), int(fields['length'])
    if offset < 0 or length != int(np.prod(tensor.dims)) * 2:
        raise ValueError(f'Invalid external weight byte span: {tensor.name}')
    return fields['location'], offset, length


def prune_to_outputs(model):
    producers = {value: index for index, node in enumerate(model.graph.node) for value in node.output}
    wanted, used = set(), set()
    pending = [value.name for value in model.graph.output]
    while pending:
        value = pending.pop()
        if value in used:
            continue
        used.add(value)
        if value in producers:
            index = producers[value]
            wanted.add(index)
            pending.extend(model.graph.node[index].input)
    nodes = [node for i, node in enumerate(model.graph.node) if i in wanted]
    weights = [tensor for tensor in model.graph.initializer if tensor.name in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(weights)
    # Remove stale inferred shapes of the deleted full-head branches.
    del model.graph.value_info[:]


def rewrite(model, config: dict, source_directory: Path):
    model = copy.deepcopy(model)
    nodes = {node.name: node for node in model.graph.node}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    channels, hidden = int(config['n_vq']), int(config['hidden_size'])
    sizes = list(map(int, config['audio_codebook_sizes']))
    ids = [int(config['audio_assistant_slot_token_id']), int(config['audio_end_token_id'])]
    if (channels < 2 or len(sizes) != channels or len(set(sizes)) != 1
            or len(set(ids)) != 2 or min(ids) < 0 or max(ids) >= int(config['vocab_size'])):
        raise ValueError('Unexpected output-head configuration')
    if any(name.startswith(PREFIX) for node in model.graph.node for name in node.output):
        raise ValueError('Graph already contains selected-head nodes')
    text = nodes['/text_lm_head/MatMul']
    audio = [nodes[f'/audio_lm_heads.{i}/MatMul'] for i in range(channels)]
    if any(node.op_type != 'MatMul' or len(node.input) != 2 or node.input[0] != text.input[0]
           for node in [text, *audio]):
        raise ValueError('Output heads must share the original hidden vector')
    text_weight = initializers[text.input[1]]
    if list(text_weight.dims) != [hidden, int(config['vocab_size'])]:
        raise ValueError('Unexpected text projection shape')
    external_info(text_weight)
    weights = [initializers[node.input[1]] for node in audio]
    spans = [external_info(weight) for weight in weights]
    location, offset, length = spans[0]
    if any(list(weight.dims) != [hidden, sizes[0]] for weight in weights):
        raise ValueError('Unexpected audio projection shape')
    if any(span != (location, offset + i * length, length) for i, span in enumerate(spans)):
        raise ValueError('Audio weights must be contiguous in the same external file')
    packed_view = np.memmap(source_directory / location, dtype=np.float16, mode='r',
                            offset=offset, shape=(channels, hidden, sizes[0]))
    for index, span in enumerate(spans):
        old_view = np.memmap(source_directory / span[0], dtype=np.float16, mode='r',
                             offset=span[1], shape=(hidden, sizes[0]))
        if not np.array_equal(packed_view[index], old_view):
            raise ValueError(f'Packed weight view differs at audio channel {index}')
    packed = copy.deepcopy(weights[0])
    packed.name = PREFIX + 'audio_weights'
    del packed.dims[:]
    packed.dims.extend([channels, hidden, sizes[0]])
    for entry in packed.external_data:
        if entry.key == 'length':
            entry.value = str(channels * length)
    model.graph.initializer.append(packed)

    def constant(name, values, dtype):
        name = PREFIX + name
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=dtype), name))
        return name

    candidate_ids = constant('text_ids', ids, np.int64)
    axis_one = constant('axis_one', [1], np.int64)
    zero = constant('zero', [0], np.int32)
    one = constant('one', [1], np.int32)
    two = constant('two', [2], np.int32)

    def add(op, inputs, name, **attrs):
        output = PREFIX + name
        model.graph.node.append(helper.make_node(op, inputs, [output], name=output, **attrs))
        return output

    text_selected = add('Gather', [text_weight.name, candidate_ids], 'text_weights', axis=1)
    text_logits = add('MatMul', [text.input[0], text_selected], 'text_projection')
    is_audio = add('Equal', ['step_type', two], 'is_audio_step')
    next_channel = add('Add', ['channel_index', one], 'next_channel')
    channel = add('Where', [is_audio, next_channel, zero], 'output_channel')
    selected = add('Gather', [packed.name, channel], 'audio_weights_selected', axis=0)
    expanded = add('Unsqueeze', [text.input[0], axis_one], 'hidden_expanded')
    projected = add('MatMul', [expanded, selected], 'audio_projection')
    audio_logits = add('Squeeze', [projected, axis_one], 'audio_squeezed')
    for value, name, shape in ((text_logits, 'text_logits', ['batch', 2]),
                               (audio_logits, 'audio_logits', ['batch', sizes[0]])):
        rounded = add('Cast', [value], name + '_fp16', to=TensorProto.FLOAT16)
        model.graph.node.append(helper.make_node('Cast', [rounded], [name],
                                                name=PREFIX + name, to=TensorProto.FLOAT))
        output = next(output for output in model.graph.output if output.name == name)
        output.CopyFrom(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    prune_to_outputs(model)
    return model, {'version': 1, 'text_token_ids': ids, 'audio_channels': channels,
                   'audio_vocab_size': sizes[0], 'step_type_to_output_channel': {
                       '0': '0 (audio output is ignored)', '1': '0', '2': 'channel_index + 1'},
                   'packed_audio_weight_views_exact': True,
                   'graph_inputs_and_transformer_unchanged': True,
                   'numerical_inference_validation': 'not performed by this export tool'}


def export(assets: Path, output: Path, *, copy_weights=False):
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    analysis = json.loads((assets / 'analysis.json').read_text())
    original = next(graph for graph in analysis['graphs'] if graph['name'] == 'local_cached_step')
    source = (assets / original['file']).resolve()
    meta_path = source.parent / 'tts_browser_onnx_meta.json'
    config = json.loads(meta_path.read_text())['model_config']
    model = onnx.load(str(source), load_external_data=False)
    rewritten, contract = rewrite(model, config, source.parent)
    if [x.SerializeToString() for x in model.graph.input] != [x.SerializeToString() for x in rewritten.graph.input]:
        raise ValueError('Graph inputs changed')
    for before, after in zip(model.graph.output[2:], rewritten.graph.output[2:], strict=True):
        if before.SerializeToString() != after.SerializeToString():
            raise ValueError('KV output interface changed')
    original_nodes = {node.name: node.SerializeToString() for node in model.graph.node}
    if any(node.SerializeToString() != original_nodes[node.name]
           for node in rewritten.graph.node if not node.name.startswith(PREFIX)):
        raise ValueError('An original retained Transformer node changed')
    samples = original.get('validation_samples', [])
    if len(samples) != 4:
        raise ValueError('Expected the four original local validation samples')
    output.mkdir(parents=True, exist_ok=False)
    contract.update(source_onnx_sha256=sha256(source), source_onnx=str(source),
                    source_metadata_sha256=sha256(meta_path), external_weights={})
    locations = sorted({external_info(tensor)[0] for tensor in rewritten.graph.initializer
                        if tensor.data_location == TensorProto.EXTERNAL})
    for location in locations:
        weight = (source.parent / location).resolve()
        destination = output / location
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_weights:
            shutil.copy2(weight, destination)
        else:
            try:
                destination.symlink_to(weight)
            except OSError:
                shutil.copy2(weight, destination)
        contract['external_weights'][location] = {'bytes': weight.stat().st_size,
                                                  'sha256': sha256(weight)}
    graph_path = output / 'local_cached_step.onnx'
    onnx.save_model(rewritten, str(graph_path))
    try:
        onnx.checker.check_model(str(graph_path))
    except onnx.checker.ValidationError as error:
        # Newer ONNX checkers reject even valid local external-data symlinks.
        if 'symbolic link' not in str(error):
            raise
        for location in locations:
            destination = output / location
            if destination.is_symlink():
                destination.unlink()
                shutil.copy2(source.parent / location, destination)
        onnx.checker.check_model(str(graph_path))
    contract['selected_onnx_sha256'] = sha256(graph_path)
    candidate = copy.deepcopy(original)
    candidate.update(file=graph_path.name, validation_samples=[])
    candidate.pop('interface', None)
    candidate.pop('source_validation_pass', None)
    (output / 'validation').mkdir()
    for index, sample_name in enumerate(samples):
        with np.load(assets / sample_name, allow_pickle=False) as sample:
            values = {key: sample[key].copy() for key in sample.files}
        step, previous = int(values['input__step_type'].item()), int(values['input__channel_index'].item())
        channel = selected_channel(step, previous, contract['audio_channels'])
        values['ref__text_logits'] = values['ref__text_logits'][:, contract['text_token_ids']]
        values['ref__audio_logits'] = values['ref__audio_logits'][:, channel, :]
        name = f'validation/sample-{index:02d}.npz'
        np.savez(output / name, **values)
        candidate['validation_samples'].append(name)
    candidate_analysis = {'format_version': 1, 'model': analysis.get('model'), 'graphs': [candidate],
                          'local_selected_heads_contract': 'selected-heads-contract.json',
                          'reference_scope': 'Selected original logits and unchanged KV; no new model inference'}
    (output / 'analysis.json').write_text(json.dumps(candidate_analysis, indent=2) + '\n')
    (output / 'selected-heads-contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--copy-weights', action='store_true', help='Copy weights for transport instead of local symlinks')
    args = parser.parse_args()
    print(json.dumps(export(args.assets.resolve(), args.output_dir.resolve(),
                            copy_weights=args.copy_weights), indent=2))


if __name__ == '__main__':
    main()
