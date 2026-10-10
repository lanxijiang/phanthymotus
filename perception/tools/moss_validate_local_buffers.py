#!/usr/bin/env python3
"""Compare fresh and reused CUDA buffers on real local TensorRT samples."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from moss_trt_session import CudaDriver, TensorRTSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'ok': False, 'order': [0, 1, 3, 2] * 2, 'atol': 1e-6, 'rtol': 1e-6}
    sessions, retained = [], []
    try:
        assert args.plan.stem == 'local_cached_step'
        graph = next(g for g in json.loads(args.analysis.read_text())['graphs']
                     if g['name'] == 'local_cached_step')
        samples = graph['validation_samples']
        assert len(samples) == 4, f'Expected four validation samples, got {len(samples)}'
        cuda = CudaDriver(0)
        report['gpu'] = cuda.info()
        for enabled in ('0', '1'):
            os.environ['MOSS_TRT_REUSE_LOCAL_BUFFERS'] = enabled
            session = TensorRTSession(args.plan, cuda)
            sessions.append(session)
            session.configure_profiling('local_cached_step', enabled=True)
        assert not sessions[0]._reuse_device_buffers and sessions[1]._reuse_device_buffers
        for index in report['order']:
            with np.load(args.analysis.parent / samples[index], allow_pickle=False) as sample:
                feeds = {key.removeprefix('input__'): sample[key] for key in sample.files
                         if key.startswith('input__')}
            outputs = [session.run(feeds) for session in sessions]
            for original, saved in retained:
                np.testing.assert_array_equal(original, saved)
            assert outputs[0].keys() == outputs[1].keys()
            for name, expected in outputs[0].items():
                actual = outputs[1][name]
                assert expected.shape == actual.shape and np.isfinite(actual).all()
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=False)
            for output in outputs:
                retained.extend((array, array.copy()) for array in output.values())
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
            plain, reused = report['profiles']
            calls_ok = all(p['runs'] == 8 and p['errors'] == 0 for p in (plain, reused))
            released = all(p['cuda_allocations'] == p['cuda_frees'] for p in (plain, reused))
            report['allocations_reduced'] = reused['cuda_allocations'] < plain['cuda_allocations']
            report['all_allocations_released'] = released
            report['ok'] &= calls_ok and released and report['allocations_reduced']
        try:
            for original, saved in retained:
                np.testing.assert_array_equal(original, saved)
        except AssertionError as error:
            report.update(ok=False, output_lifetime_error=str(error))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
