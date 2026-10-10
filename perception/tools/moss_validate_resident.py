#!/usr/bin/env python3
"""Validate real TensorRT resident states against the identical host-state engines.

Independent NPZ inputs and chained output-to-input states are both checked.
The chains use controlled valid tokens, not a speech-quality/sampling benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from moss_trt_session import CudaDriver, TensorRTSession


GRAPHS = ('global_unified', 'local_cached_step', 'codec_decode_step')


def load_inputs(path):
    with np.load(path, allow_pickle=False) as data:
        return {key.removeprefix('input__'): data[key].copy()
                for key in data.files if key.startswith('input__')}


def state_pairs(name, feeds):
    if name == 'global_unified':
        return {key: key.replace('past_', 'present_', 1)
                for key in feeds if key.startswith('past_key_') or key.startswith('past_value_')}
    if name == 'local_cached_step':
        return {key: key.replace('local_past_', 'local_present_', 1)
                for key in feeds if key.startswith('local_past_')}
    return {key: 'present_' + key for key in feeds
            if key.startswith('attention_') or key.startswith('transformer_')}


def compare_outputs(expected, resident):
    if expected.keys() != resident.keys():
        raise AssertionError('Output names differ')
    differences = {}
    for name, reference in expected.items():
        value = resident[name]
        actual = value.materialize() if hasattr(value, 'materialize') else value
        actual, reference = np.asarray(actual), np.asarray(reference)
        shape_ok = actual.shape == reference.shape
        finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
        exact = np.issubdtype(reference.dtype, np.integer) or reference.dtype == np.bool_
        equal = shape_ok and finite and (
            np.array_equal(actual, reference) if exact else
            np.allclose(actual, reference, rtol=1e-6, atol=1e-6, equal_nan=False))
        maximum = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)), initial=0)) if shape_ok else None
        differences[name] = {'pass': bool(equal), 'shape_match': shape_ok,
                             'finite': finite, 'exact_integer': exact,
                             'max_abs_error': maximum}
    return differences


def validate_graph(name, graph, directory, plan, cuda):
    report = {'graph': name, 'plan': str(plan), 'ok': False, 'checks': [],
              'chain_scope': 'Controlled valid tokens with actual output states fed into subsequent calls; reset and repeat'}
    sessions = []
    try:
        samples = [load_inputs(directory / path) for path in graph['validation_samples']]
        if not samples:
            raise ValueError('No validation samples')
        pairs = state_pairs(name, samples[0])
        if not pairs:
            raise ValueError('No recurrent state mapping found')
        report['state_pairs'] = pairs
        for mode in ('0', '1'):
            os.environ['MOSS_TRT_RESIDENT_STATE'] = mode
            session = TensorRTSession(plan, cuda)
            sessions.append(session)
            session.configure_profiling(name, enabled=True)

        def run_pair(label, host_feeds, resident_feeds):
            expected = sessions[0].run(host_feeds)
            actual = sessions[1].run(resident_feeds)
            differences = compare_outputs(expected, actual)
            device_outputs = sum(hasattr(value, 'materialize') for value in actual.values())
            check = {'label': label, 'device_state_outputs': device_outputs,
                     'differences': differences,
                     'ok': all(item['pass'] for item in differences.values())}
            report['checks'].append(check)
            if not check['ok']:
                raise AssertionError(f'Output differences: {label}')
            if not device_outputs:
                raise AssertionError('Resident path returned no device state handles')
            return expected, actual

        # Every sample remains an independent host-fed invocation, including
        # samples whose recorded states are nonzero. This must override any
        # prior device state, rather than accidentally continue the old chain.
        for index, sample in enumerate(samples):
            run_pair(f'npz-{index}', sample, sample)

        state_name = next(iter(pairs))
        if name != 'codec_decode_step':
            initial = next(sample for sample in samples if sample[state_name].shape[1] == 0)
        else:
            initial = samples[0]
        if name == 'global_unified':
            decode = next(sample for sample in samples if sample['input_ids'].shape[1] == 1)
        for cycle in range(2):
            # Host resets are explicit, including codec -1 sentinel positions.
            host_state = {
                key: np.full_like(initial[key], -1) if key.endswith('_positions')
                else np.zeros_like(initial[key]) for key in pairs
            }
            device_state = {key: value.copy() for key, value in host_state.items()}
            steps = 17 if name == 'local_cached_step' else 6 if name == 'global_unified' else 5
            for step in range(steps):
                common = {key: value.copy() for key, value in initial.items() if key not in pairs}
                if name == 'global_unified':
                    if step:
                        common['input_ids'] = decode['input_ids'].copy()
                    length = host_state[state_name].shape[1] + common['input_ids'].shape[1]
                    common['attention_mask'] = np.ones((1, length), dtype=np.int32)
                elif name == 'local_cached_step':
                    common['past_valid_lengths'] = np.asarray([step], dtype=np.int32)
                    common['step_type'] = np.asarray([min(step, 2)], dtype=np.int32)
                    common['channel_index'] = np.asarray([max(0, step - 2)], dtype=np.int32)
                    common['text_token_id'] = np.asarray([int(samples[1]['text_token_id'][0]) if step == 1 else 0], dtype=np.int32)
                    common['audio_token_id'] = np.asarray([int(samples[-1]['audio_token_id'][0]) if step >= 2 else 0], dtype=np.int32)
                else:
                    frames = (1, 4, 8, 2, 1)[step]
                    common['audio_codes'] = np.repeat(initial['audio_codes'][:, :1, :], frames, axis=1)
                    common['audio_code_lengths'] = np.asarray([frames], dtype=np.int32)
                host_outputs, device_outputs = run_pair(
                    f'chain-{cycle}-step-{step}-reset-{step == 0}',
                    {**common, **host_state}, {**common, **device_state})
                host_state = {key: host_outputs[value] for key, value in pairs.items()}
                device_state = {key: device_outputs[value] for key, value in pairs.items()}
        report['ok'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        report['profiles'] = []
        for session in sessions:
            try:
                session.close()
            except Exception as error:
                report.setdefault('close_errors', []).append(str(error))
                report['ok'] = False
            report['profiles'].append(session.profile_snapshot())
        if len(report['profiles']) == 2:
            plain, resident = report['profiles']
            report['cuda_allocations_released'] = all(
                profile['cuda_allocations'] == profile['cuda_frees']
                for profile in (plain, resident))
            report['state_transfer_reduced'] = (
                resident['h2d_bytes'] < plain['h2d_bytes']
                and resident['d2h_bytes'] < plain['d2h_bytes'])
            report['ok'] &= (report['cuda_allocations_released'] and report['state_transfer_reduced']
                             and plain['errors'] == resident['errors'] == 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--graph', action='append', choices=GRAPHS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'ok': False, 'atol': 1e-6, 'rtol': 1e-6, 'graphs': [],
              'profiling_scope': 'Runner counters exclude validation-only materialize copies; this is numerical and lifecycle validation, not an inference performance benchmark'}
    try:
        analysis_path = args.analysis or args.assets / 'analysis.json'
        graphs = {graph['name']: graph for graph in json.loads(analysis_path.read_text())['graphs']}
        cuda = CudaDriver(0)
        report['gpu'] = cuda.info()
        for name in dict.fromkeys(args.graph or GRAPHS):
            row = validate_graph(name, graphs[name], analysis_path.parent,
                                 args.assets / 'plans' / name / f'{name}.plan', cuda)
            report['graphs'].append(row)
            print(json.dumps({'graph': name, 'ok': row['ok'], 'checks': len(row['checks']),
                              'error': row.get('error')}), flush=True)
        report['ok'] = all(graph['ok'] for graph in report['graphs'])
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'ok': report['ok'], 'report': str(args.output)}), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
