#!/usr/bin/env python3
"""Inspect the isolated GPU sampler with actual failed frame logits, no model run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
from moss_export_sampled_frame import GraphBuilder, SAMPLING


PREFIX = 'sample/step_01/'
FLOAT_OUTPUTS = ['temperature', 'sorted_scores', 'kth_threshold', 'top_k',
                 'sorted_probabilities', 'previous_probability', 'top_p',
                 'token_order_scores', 'probabilities', 'cdf_raw', 'cdf_total', 'cdf']
DEFAULT_CASES = ['hidden-0-uniform-3:14', 'hidden-0-uniform-5:9',
                 'hidden-0-uniform-7:5', 'hidden-2-uniform-2:8',
                 'hidden-3-uniform-2:15', 'hidden-0-uniform-1:1']


def export_graph(path):
    model = helper.make_model(helper.make_graph([], 'moss_sampler_diagnostic', [
        helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 1024]),
        helper.make_tensor_value_info('u', TensorProto.FLOAT, [1, 1]),
    ], []), opset_imports=[helper.make_opsetid('', 17)])
    # TensorRT 10.4 accepts the same IR version as the original MOSS exports.
    model.ir_version = 8
    build = GraphBuilder(model)
    token = build.sample('logits', 'u', vocabulary=1024,
                         temperature=SAMPLING['audio_temperature'],
                         top_k=SAMPLING['audio_top_k'], top_p=SAMPLING['audio_top_p'], prefix=PREFIX)
    for name in FLOAT_OUTPUTS:
        shape = [1, 1] if name in ('kth_threshold', 'cdf_total') else [1, 1024]
        model.graph.output.append(helper.make_tensor_value_info(PREFIX + name, TensorProto.FLOAT, shape))
    model.graph.output.append(helper.make_tensor_value_info(PREFIX + 'sorted_ids', TensorProto.INT64, [1, 1024]))
    model.graph.output.append(helper.make_tensor_value_info(token, TensorProto.INT32, [1]))
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def intermediate_reference(logits, uniform):
    scores = np.asarray(logits, np.float32) / np.float32(SAMPLING['audio_temperature'])
    order = np.argsort(-scores, axis=1, kind='stable')
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    threshold = sorted_scores[:, 24:25]
    top_k = np.where(sorted_scores < threshold, -np.inf, sorted_scores)
    def softmax(values):
        exp = np.exp(values - np.max(values, axis=1, keepdims=True), dtype=np.float32)
        return exp / np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
    sorted_probabilities = softmax(top_k)
    previous = np.concatenate((np.zeros((1, 1), np.float32),
                               np.cumsum(sorted_probabilities[:, :-1], axis=1, dtype=np.float32)), axis=1)
    top_p = np.where(previous > np.float32(SAMPLING['audio_top_p']), -np.inf, top_k)
    token_order_scores = scores.copy()
    np.put_along_axis(token_order_scores, order, top_p, axis=1)
    probabilities = softmax(token_order_scores)
    cdf_raw = np.cumsum(probabilities, axis=1, dtype=np.float32)
    cdf_total = cdf_raw[:, -1:]
    cdf = cdf_raw / cdf_total
    selected = np.sum(cdf < uniform, axis=1).astype(np.int32)
    selected = np.where(selected >= 1024, np.argmax(token_order_scores, axis=1), selected).astype(np.int32)
    return dict(temperature=scores, sorted_scores=sorted_scores, kth_threshold=threshold,
                top_k=top_k, sorted_probabilities=sorted_probabilities, previous_probability=previous,
                top_p=top_p, token_order_scores=token_order_scores, probabilities=probabilities, cdf_raw=cdf_raw, cdf_total=cdf_total, cdf=cdf,
                sorted_ids=order.astype(np.int64), token_id=selected)


def run(args):
    if args.output_dir.exists():
        raise FileExistsError('Choose a fresh output directory')
    args.output_dir.mkdir(parents=True)
    graph_path = args.output_dir / 'sampler_debug.onnx'
    export_graph(graph_path)
    if args.export_only:
        return {'graph': str(graph_path), 'inference': 'not run'}
    import tensorrt as trt
    from moss_build_trt import build
    from moss_build_sampled_frame import force_sampler_fp32
    from moss_trt_session import CudaDriver, TensorRTSession
    cuda = CudaDriver(args.device)
    configured = []
    def configure(network, config):
        configured.extend(force_sampler_fp32(network, config, trt))
    graph = dict(name='sampler_debug', file=graph_path.name, profiles={})
    build_args = SimpleNamespace(fp32_graph=[], precision=args.precision,
                                 workspace_mb=64, disable_jit_convolutions=False,
                                 configure_network=configure,
                                 strongly_typed=bool(getattr(args, 'strongly_typed', False)),
                                 detailed_profiling=True)
    entry = build(graph, args.output_dir, args.output_dir, build_args)
    session = TensorRTSession(args.output_dir / entry['file'], cuda)
    report = dict(engine=entry, gpu=cuda.info(), sampler_constraints=configured, cases=[])
    try:
        inspector = session.engine.create_engine_inspector()
        inspector.execution_context = session.context
        inspector_path = args.output_dir / 'engine-inspector.json'
        inspector_path.write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON) + '\n')
        report['engine_inspector_file'] = inspector_path.name
        report['strongly_typed'] = bool(getattr(args, 'strongly_typed', False))
        for specification in args.case or DEFAULT_CASES:
            stem, text_step = specification.rsplit(':', 1)
            step = int(text_step)
            if not 1 <= step <= 16:
                raise ValueError('Audio step must be 1 through 16')
            with np.load(args.samples_dir / (stem + '.npz'), allow_pickle=False) as sample:
                logits = sample['audio_logits'][:, step - 1].copy()
                uniform = sample['input__random_u'][:, step:step + 1].copy()
                original_token = int(sample['frame_token_ids'][0, step - 1])
            actual = session.run(dict(logits=logits, u=uniform))
            reference = intermediate_reference(logits, uniform)
            row = dict(case=specification, full_frame_token=original_token,
                       isolated_token=int(actual[PREFIX + 'token_id'].item()),
                       reference_token=int(reference['token_id'].item()),
                       uniform=float(uniform.item()), intermediate={})
            for name, expected in reference.items():
                value = actual[PREFIX + name]
                finite = np.isfinite(value) & np.isfinite(expected)
                row['intermediate'][name] = dict(
                    exact=bool(np.array_equal(value, expected)),
                    finite_pattern_equal=bool(np.array_equal(np.isfinite(value), np.isfinite(expected))),
                    max_abs_error=float(np.max(np.abs(value[finite] - expected[finite]))) if finite.any() else 0,
                )
            row['retained_gpu'] = np.flatnonzero(np.isfinite(actual[PREFIX + 'token_order_scores'][0])).tolist()
            row['retained_reference'] = np.flatnonzero(np.isfinite(reference['token_order_scores'][0])).tolist()
            report['cases'].append(row)
            data = dict(input__logits=logits, input__u=uniform)
            data.update({'actual__' + key.removeprefix(PREFIX): value for key, value in actual.items()})
            data.update({'reference__' + key: value for key, value in reference.items()})
            np.savez(args.output_dir / (stem + '-step-' + str(step) + '.npz'), **data)
            print(json.dumps(row), flush=True)
    finally:
        session.close()
        (args.output_dir / 'debug-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--strongly-typed', action='store_true',
                        help='Honor explicit ONNX types; no FP16 builder flag or layer precision overrides')
    parser.add_argument('--case', action='append', help='NPZ stem:audio step (1 through 16)')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({'output_dir': str(args.output_dir), 'cases': len(result.get('cases', []))}), flush=True)


if __name__ == '__main__':
    main()
