#!/usr/bin/env python3
"""Validate selected local logits and independent recurrent chains on real TRT.

Numerical acceptance matches the builder: exact integer values; floating
maximum absolute error <= 1e-4 OR relative RMS error <= 0.05. Argmax agreement
is reported separately and does not guarantee identical stochastic sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from moss_trt_session import CudaDriver, TensorRTSession, compare


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(path):
    with np.load(path, allow_pickle=False) as data:
        return {key.removeprefix('input__'): data[key].copy()
                for key in data.files if key.startswith('input__')}


def numerical_result(actual, reference):
    result = compare(actual, reference)
    exact = np.issubdtype(reference.dtype, np.integer) or reference.dtype == np.bool_
    equal = bool(np.array_equal(actual, reference))
    result.update(exact_equal=equal, exact_integer_required=exact,
                  reference_finite=bool(np.isfinite(reference).all()))
    within_tolerance = equal if exact else (
        result.get('max_abs_error', float('inf')) <= 1e-4
        or result.get('relative_rms_error', float('inf')) <= 0.05)
    result['pass'] = bool(result['shape_match'] and result['finite']
                          and result['reference_finite'] and within_tolerance)
    return result


def compare_selected(baseline, candidate, feeds, contract):
    if baseline.keys() != candidate.keys():
        raise ValueError('Candidate output names differ from original engine')
    step_type = int(feeds['step_type'].item())
    channel = int(feeds['channel_index'].item()) + 1 if step_type == 2 else 0
    if step_type not in (0, 1, 2) or not 0 <= channel < contract['audio_channels']:
        raise ValueError('Invalid step type or selected audio channel')
    reference = dict(baseline)
    reference['text_logits'] = baseline['text_logits'].reshape(-1)[contract['text_token_ids']]
    reference['audio_logits'] = baseline['audio_logits'].reshape(
        contract['audio_channels'], contract['audio_vocab_size'])[channel]
    logical_candidate = dict(candidate)
    for name in ('text_logits', 'audio_logits'):
        logical_candidate[name] = np.asarray(candidate[name]).reshape(-1)
    differences = {name: numerical_result(logical_candidate[name], value)
                   for name, value in reference.items()}
    for name in ('text_logits', 'audio_logits'):
        differences[name]['baseline_full_shape'] = list(baseline[name].shape)
        differences[name]['candidate_raw_shape'] = list(candidate[name].shape)
        differences[name]['shape_scope'] = 'Contract-selected logical vector; KV outputs retain exact physical shapes'
    text_match = int(np.argmax(reference['text_logits'])) == int(np.argmax(logical_candidate['text_logits']))
    audio_match = int(np.argmax(reference['audio_logits'])) == int(np.argmax(logical_candidate['audio_logits']))
    return {
        'step_type': step_type,
        'audio_channel': channel,
        'differences': differences,
        'ok': all(value['pass'] for value in differences.values()),
        'assistant_decision_argmax_match': text_match,
        'audio_argmax_match': audio_match,
        'consumed_argmax_match': text_match if step_type == 0 else audio_match,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {
        'ok': False, 'checks': [],
        'numerical_gate': 'Exact integers; floating max_abs_error <= 0.0001 OR relative_rms_error <= 0.05, plus matching logical shapes and finite values',
        'argmax_scope': 'Diagnostic only; argmax agreement does not guarantee identical stochastic sampling',
        'chain_scope': 'Both engines feed their own previous KV outputs into all 17 local steps; controlled shared tokens keep the comparison aligned',
    }
    sessions = []
    try:
        contract = json.loads(args.contract.read_text())
        ids = contract['text_token_ids']
        if (contract.get('version') != 1 or len(ids) != 2 or len(set(ids)) != 2
                or any(not isinstance(item, int) or item < 0 for item in ids)
                or contract['audio_channels'] != 16 or contract['audio_vocab_size'] != 1024):
            raise ValueError('Unsupported selected-head contract')
        report['contract'] = contract
        analysis = json.loads((args.assets / 'analysis.json').read_text())
        graph = next(graph for graph in analysis['graphs'] if graph['name'] == 'local_cached_step')
        samples = [load_inputs(args.assets / path) for path in graph['validation_samples']]
        if len(samples) != 4:
            raise ValueError('Expected four original local validation NPZ files')
        plans = [args.assets / 'plans/local_cached_step/local_cached_step.plan', args.plan]
        report['plans'] = [{'path': str(path), 'sha256': sha256(path)} for path in plans]
        # Keep ordinary NumPy states in both paths. This experiment tests the
        # selected projection only, independently of resident-buffer changes.
        os.environ['MOSS_TRT_RESIDENT_STATE'] = '0'
        os.environ['MOSS_TRT_REUSE_LOCAL_BUFFERS'] = '0'
        cuda = CudaDriver(0)
        report['gpu'] = cuda.info()
        for path in plans:
            session = TensorRTSession(path, cuda)
            sessions.append(session)
            session.configure_profiling('local_cached_step', enabled=True)

        def run_pair(label, baseline_feeds, candidate_feeds):
            baseline = sessions[0].run(baseline_feeds)
            candidate = sessions[1].run(candidate_feeds)
            row = compare_selected(baseline, candidate, baseline_feeds, contract)
            row['label'] = label
            report['checks'].append(row)
            return baseline, candidate

        for index, sample in enumerate(samples):
            run_pair(f'npz-{index}', sample, sample)
        pairs = {key: key.replace('local_past_', 'local_present_', 1)
                 for key in samples[0] if key.startswith('local_past_')}
        state_name = next(iter(pairs))
        initial = next(sample for sample in samples if sample[state_name].shape[1] == 0)
        for cycle in range(2):
            states = [{key: np.zeros_like(initial[key]) for key in pairs} for _ in sessions]
            previous_audio_token = 0
            for step in range(17):
                common = {key: value.copy() for key, value in initial.items() if key not in pairs}
                common.update(
                    past_valid_lengths=np.asarray([step], dtype=np.int32),
                    step_type=np.asarray([min(step, 2)], dtype=np.int32),
                    channel_index=np.asarray([max(0, step - 2)], dtype=np.int32),
                    text_token_id=np.asarray([ids[0] if step == 1 else 0], dtype=np.int32),
                    audio_token_id=np.asarray([previous_audio_token if step >= 2 else 0], dtype=np.int32),
                )
                outputs = run_pair(f'chain-{cycle}-step-{step}-reset-{step == 0}',
                                   {**common, **states[0]}, {**common, **states[1]})
                states = [{key: output[value] for key, value in pairs.items()} for output in outputs]
                if step >= 1:
                    channel = step - 1
                    previous_audio_token = int(np.argmax(outputs[0]['audio_logits'].reshape(16, 1024)[channel]))
        report['ok'] = all(row['ok'] for row in report['checks'])
        report['consumed_argmax_mismatches'] = sum(not row['consumed_argmax_match'] for row in report['checks'])
        report['chain_audio_channels_covered'] = sorted({row['audio_channel'] for row in report['checks']
                                                       if row['label'].startswith('chain-') and row['step_type'] != 0})
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
            report['all_allocations_released'] = all(
                item['cuda_allocations'] == item['cuda_frees'] for item in report['profiles'])
            report['ok'] &= report['all_allocations_released'] and all(
                item['runs'] == 38 and item['errors'] == 0 for item in report['profiles'])
        values = [value for row in report['checks'] for value in row['differences'].values()]
        report['max_abs_error'] = max((value.get('max_abs_error', 0) for value in values), default=None)
        report['max_relative_rms_error'] = max((value.get('relative_rms_error', 0) for value in values), default=None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'ok': report['ok'], 'checks': len(report['checks']),
                          'report': str(args.output), 'error': report.get('error'),
                          'max_abs_error': report['max_abs_error'],
                          'max_relative_rms_error': report['max_relative_rms_error'],
                          'consumed_argmax_mismatches': report.get('consumed_argmax_mismatches')}), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
