#!/usr/bin/env python3
"""Record real single-process MOSS GPU audio, timings and scoped memory evidence.

Example (inside the Jetson candidate image):
    python3 /work/tools/moss_benchmark.py --model-dir /models/assets \
        --language ja --speaker-id 11 --text 'こんにちは。音声合成のテストです。' \
        --warmup-text 'こんにちは。' --repeat 3 --output-dir /results/japanese

Each process creates one TensorRT-backed adapter. Three processes/containers
must be launched separately to test the leaderboard's independent instances.
Give each worker a new output directory and the same new --start-barrier-dir,
with --worker-id 0, 1, or 2. The barrier releases only after all workers warm up.
For paired runs keep engine hashes, initial seed, warmup and text order equal.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import uuid
import wave


PERCEPTION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PERCEPTION_DIR))

from plugins.moss_tts import MossTTSAdapter  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata(adapter, model_dir: Path) -> dict:
    """Read provenance outside the timed requests; never change runtime state."""
    result = {
        'backend': adapter._pool.backend,
        'python': {'version': platform.python_version(), 'executable': sys.executable},
        'platform': platform.platform(),
        'environment': {name: value for name, value in sorted(os.environ.items())
                        if name.startswith('MOSS_TRT_') or name in {
                            'MOSS_FAST_SAMPLING',
                            'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_PRELOAD'}},
        'packages': {},
        'engines': {},
        'runtimes': [runtime.state_summary() for runtime in getattr(adapter, '_runtimes', ())],
        'code_sha256': {},
    }
    for relative in ('tools/moss_benchmark.py', 'tools/moss_trt_session.py',
                     'plugins/moss_tts.py', 'plugins/moss_deploy/runtime.py',
                     'plugins/moss_deploy/trt_pool.py',
                     'plugins/moss_deploy/sampling.py',
                     'plugins/moss_deploy/selected_logits.py',
                     'plugins/moss_deploy/frame_contract.py'):
        path = PERCEPTION_DIR / relative
        if path.is_file():
            result['code_sha256'][relative] = file_sha256(path)
    for name in ('numpy', 'onnx', 'tensorrt', 'sentencepiece', 'pynini', 'WeTextProcessing'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    mapping_path = model_dir / 'engine-map.json'
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text())
        result['engine_map_sha256'] = file_sha256(mapping_path)
        result['engine_map'] = {'path': str(mapping_path.resolve()), 'entries': mapping}
        result['configured_engines'] = {}

        def describe_file(path):
            path = Path(path).resolve()
            return {'path': str(path), 'bytes': path.stat().st_size,
                    'sha256': file_sha256(path)}

        selected_plan = os.environ.get('MOSS_TRT_SELECTED_LOCAL_PLAN')
        selected_path = os.environ.get('MOSS_TRT_SELECTED_LOCAL_CONTRACT')
        selected_contract = getattr(adapter._pool, 'selected_local_contract', None)
        selected_enabled = selected_contract is not None
        if selected_enabled:
            if not selected_plan or not selected_path:
                raise ValueError('Loaded selected local engine has no plan/contract provenance paths')
            contract_path = Path(selected_path).resolve()
            contract = json.loads(contract_path.read_text())
            if contract != selected_contract:
                raise ValueError('Selected local contract changed after model initialization')
            result['selected_local_contract'] = {**describe_file(contract_path), 'contents': contract}
        elif selected_plan or selected_path:
            raise ValueError('Selected engine environment does not match the initialized runtime')
        frame_plan = os.environ.get('MOSS_TRT_FRAME_PLAN')
        frame_path = os.environ.get('MOSS_TRT_FRAME_CONTRACT')
        frame_contract = getattr(adapter._pool, 'frame_contract', None)
        frame_enabled = frame_contract is not None
        if frame_enabled:
            if not frame_plan or not frame_path:
                raise ValueError('Loaded fused frame engine has no plan/contract provenance paths')
            contract_path = Path(frame_path).resolve()
            contract = json.loads(contract_path.read_text())
            if contract != frame_contract:
                raise ValueError('Fused frame contract changed after model initialization')
            result['frame_contract'] = {**describe_file(contract_path), 'contents': contract}
        elif frame_plan or frame_path:
            raise ValueError('Fused frame engine environment does not match the initialized runtime')
        effective_paths = getattr(adapter._pool, 'effective_plan_paths', None)
        for name in sorted(getattr(adapter._pool, 'engines', {})):
            configured = (model_dir / mapping[name]).resolve() if name in mapping else None
            override = selected_enabled and name == 'local_cached_step'
            result['configured_engines'][name] = (
                describe_file(configured) if configured is not None and configured.is_file() else
                {'path': str(configured) if configured is not None else None,
                 'exists': False, 'optional_graph': configured is None})
            if effective_paths is not None:
                if name not in effective_paths:
                    raise ValueError(f'Loaded engine has no effective plan provenance: {name}')
                effective = Path(effective_paths[name]).resolve()
            elif frame_enabled and name == 'local_fixed_sampled_frame':
                effective = Path(frame_plan).resolve()
            else:
                effective = Path(selected_plan).resolve() if override else configured
            if effective is None:
                raise ValueError(f'Loaded engine has no plan provenance: {name}')
            expected_override = (frame_plan if frame_enabled and name == 'local_fixed_sampled_frame'
                                 else selected_plan if override else None)
            if expected_override and effective != Path(expected_override).resolve():
                raise ValueError(f'Engine environment changed after initialization: {name}')
            result['engines'][name] = describe_file(effective)
        result['engine_provenance_scope'] = (
            'engines records effective loaded-plan paths; configured_engines and engine_map '
            'retain the archive configuration before a selected local override')
    else:
        result['engine_metadata_unavailable'] = str(mapping_path)
    return result


def profile_delta(after: dict | None, before: dict | None) -> dict | None:
    """Subtract cumulative graph counters, treating newly created graphs as zero."""
    if after is None:
        return None
    before_graphs = (before or {}).get('by_graph', {})
    by_graph = {}
    for graph, counters in after.get('by_graph', {}).items():
        previous = before_graphs.get(graph, {})
        row = {name: value - previous.get(name, 0) for name, value in counters.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
        if any(value < 0 for value in row.values()):
            raise ValueError(f'Performance counters reset during benchmark: {graph}')
        by_graph[graph] = row
    return {'timing_scope': after.get('timing_scope'), 'by_graph': by_graph}


def wait_for_start_barrier(directory: Path, worker_id: int, parties: int,
                           timeout_s: float) -> dict:
    """One-use shared-directory barrier after warmup; worker 0 releases peers.

    Use a new directory for every group. Markers are retained for provenance;
    no worker deletes another worker's files. Exclusive creation and per-worker
    nonces prevent a stale release from starting a new benchmark group.
    """
    began = time.monotonic()
    deadline = began + timeout_s
    nonce = uuid.uuid4().hex
    manifest = directory / 'manifest.json'
    own_marker = directory / f'worker-{worker_id}.ready.json'

    def write_exclusive(path, data):
        with path.open('x', encoding='utf-8') as destination:
            json.dump(data, destination)

    def read_complete(path):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def pause():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Start barrier timed out after {timeout_s}s: {directory}')
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    if worker_id == 0:
        directory.mkdir(parents=True, exist_ok=False)
        write_exclusive(manifest, {'parties': parties, 'leader_nonce': nonce})
    try:
        config = read_complete(manifest)
        while config is None:
            pause()
            config = read_complete(manifest)
        if config.get('parties') != parties:
            raise ValueError('Start barrier parties do not match its manifest')
        write_exclusive(own_marker, {'worker_id': worker_id, 'nonce': nonce,
                                     'leader_nonce': config['leader_nonce'], 'pid': os.getpid()})
        while True:
            failures = sorted(directory.glob('worker-*.failed.json'))
            if failures:
                raise RuntimeError(f'Start barrier peer failed: {failures[0].name}')
            ready = [read_complete(directory / f'worker-{i}.ready.json') for i in range(parties)]
            complete = all(row and row.get('worker_id') == index
                           and row.get('leader_nonce') == config['leader_nonce']
                           for index, row in enumerate(ready))
            release_path = directory / 'release.json'
            if worker_id == 0 and complete:
                write_exclusive(release_path, {'leader_nonce': nonce,
                                               'worker_nonces': [row['nonce'] for row in ready]})
            release = read_complete(release_path)
            if release is not None:
                if (release.get('leader_nonce') != config['leader_nonce']
                        or len(release.get('worker_nonces', [])) != parties
                        or release['worker_nonces'][worker_id] != nonce):
                    raise RuntimeError('Stale or invalid start barrier release')
                return {'directory': str(directory), 'worker_id': worker_id, 'parties': parties,
                        'nonce': nonce, 'leader_nonce': config['leader_nonce'],
                        'wait_s': time.monotonic() - began,
                        'released_monotonic_s': time.perf_counter()}
            pause()
    except Exception as error:
        # A failure marker belongs only to this attempt, identified by its nonce.
        # Do not write into an old directory if exclusive registration failed.
        marker = read_complete(own_marker)
        if marker and marker.get('nonce') == nonce:
            write_exclusive(directory / f'worker-{worker_id}.failed.json',
                            {'nonce': nonce, 'error': str(error)})
        raise


def process_memory() -> dict:
    """Linux process RSS and cgroup charges, never labelled GPU-only memory."""
    result = {
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rss_peak_scope": "Linux process lifetime maximum RSS; includes imports and initialization",
        "rss_current_bytes": None,
        "cgroup_v2": None,
    }
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                result['rss_current_bytes'] = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    candidates = [Path('/sys/fs/cgroup')]
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                candidates.insert(0, Path('/sys/fs/cgroup') / line[3:].lstrip('/'))
    except OSError:
        pass
    for directory in candidates:
        if not (directory / 'memory.current').is_file():
            continue
        values = {}
        for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current'):
            try:
                value = (directory / name).read_text().strip()
                values[name] = int(value) if value.isdigit() else value
            except OSError:
                values[name] = None
        result['cgroup_v2'] = {
            'path': str(directory),
            'scope': 'Memory charged to this cgroup, possibly shared by other processes; not GPU-only; null peak means unavailable',
            **values,
        }
        break
    return result


def memory_snapshot(adapter=None) -> dict:
    result = {'process': process_memory()}
    if adapter is not None and adapter._pool is not None:
        result['gpu'] = adapter._pool.memory_snapshot()
        result['gpu_scope'] = (
            'Device-wide free-memory snapshot after engine loading, including other users; '
            'not a per-process allocation measurement or a sampled peak. '
            'Jetson CPU/GPU use shared system RAM.'
        )
    return result


def synthesize_case(adapter, text: str, output_path: Path) -> dict:
    """Time actual adapter callbacks; save the captured PCM after timing ends."""
    pcm_data = bytearray()
    first_pcm = None
    first_nonzero = None
    chunk_count = 0
    started = time.perf_counter()
    for pcm in adapter.synthesize_stream(text):
        arrived = time.perf_counter() - started
        if not isinstance(pcm, bytes):
            raise TypeError(f'Expected PCM bytes, got {type(pcm).__name__}')
        if not pcm:
            continue
        if len(pcm) % 2:
            raise ValueError('Received incomplete PCM16 sample')
        if first_pcm is None:
            first_pcm = arrived
        if first_nonzero is None and any(pcm):
            first_nonzero = arrived
        pcm_data.extend(pcm)
        chunk_count += 1
    ended = time.perf_counter()
    elapsed = ended - started
    if not pcm_data:
        raise RuntimeError('MOSS produced no PCM')
    with wave.open(str(output_path), 'wb') as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16_000)
        destination.writeframes(pcm_data)
    if first_nonzero is None:
        raise RuntimeError(f'MOSS produced only zero PCM; saved {output_path}')
    duration = len(pcm_data) / 32_000
    return {
        'text': text,
        'audio_path': str(output_path),
        'sample_rate': 16_000,
        'channels': 1,
        'pcm_bytes': len(pcm_data),
        'chunks': chunk_count,
        'first_pcm_s': first_pcm,
        'first_nonzero_pcm_chunk_s': first_nonzero,
        'elapsed_s': elapsed,
        'started_monotonic_s': started,
        'ended_monotonic_s': ended,
        'audio_s': duration,
        'rtf': elapsed / duration,
        'nonempty_nonzero_pcm': True,
        'wav_sha256': file_sha256(output_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--language', default='ja')
    parser.add_argument('--speaker-id', type=int, default=11)
    parser.add_argument('--text', action='append', required=True, help='Repeat this argument for multiple texts')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--warmup-text', help='Defaults to the first requested text; no hardcoded Chinese warmup')
    parser.add_argument('--skip-warmup', action='store_true')
    parser.add_argument('--seed', type=int, choices=(1234,), default=1234,
                        help='Record/require the adapter fixed initial seed; other seeds are not implemented')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start-barrier-dir', type=Path,
                        help='New shared directory for a group of independently launched containers')
    parser.add_argument('--worker-id', type=int, help='Barrier worker number, 0 through parties minus one')
    parser.add_argument('--barrier-parties', type=int, default=3)
    parser.add_argument('--barrier-timeout-s', type=float, default=120)
    args = parser.parse_args(argv)
    if args.repeat < 1 or not all(text.strip() for text in args.text):
        parser.error('--repeat must be positive and every --text must be nonempty')
    if args.start_barrier_dir is not None:
        if (args.worker_id is None or args.barrier_parties < 2
                or not 0 <= args.worker_id < args.barrier_parties):
            parser.error('Barrier needs --worker-id in [0, --barrier-parties) and at least two parties')
        if not 0 < args.barrier_timeout_s <= 3600:
            parser.error('--barrier-timeout-s must be positive and at most 3600')
    elif args.worker_id is not None:
        parser.error('--worker-id requires --start-barrier-dir')
    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Output directory already exists; choose a new run directory: {output_dir}')
    report = {
        'ok': False,
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pid': os.getpid(),
        'model_dir': str(Path(args.model_dir).resolve()),
        'language': args.language,
        'speaker_id': args.speaker_id,
        'instances': 1,
        'seed': args.seed,
        'seed_scope': 'Adapter fixed initial RNG seed; not reset between warmup and cases. Paired runs require identical warmup, text order and repeat count.',
        'repeat': args.repeat,
        'requested_texts': args.text,
        'clock': {'name': 'time.perf_counter',
                  'implementation': time.get_clock_info('perf_counter').implementation,
                  'scope': 'Absolute monotonic seconds, comparable only between processes with the same host and time namespace'},
        'scope': 'One real TensorRT adapter in one process; no CPU inference fallback',
        'timing_scope': 'Adapter call to PCM callbacks and synthesis completion; excludes initialization, warmup and final WAV write; includes text preparation, queueing, CPU postprocessing and PCM capture; excludes ROS/MCP transport',
        'first_nonzero_scope': 'Arrival of first PCM chunk containing any nonzero PCM16 sample; not a perceptual audibility threshold',
        'audio_validation_scope': 'Nonempty nonzero PCM only; does not prove pronunciation, naturalness or untruncated completion',
        'memory': {},
        'cases': [],
    }
    adapter = None
    stage = 'initialize'
    try:
        report['memory']['before_initialize'] = memory_snapshot()
        started = time.perf_counter()
        adapter = MossTTSAdapter(args.model_dir, language=args.language,
                                 speaker_id=args.speaker_id, instances=1,
                                 output_dir=output_dir / 'runtime')
        report['initialize_s'] = time.perf_counter() - started
        report['backend'] = adapter._pool.backend
        report['runtime_metadata'] = runtime_metadata(adapter, Path(args.model_dir).resolve())
        for label, path in (('boot_id', '/proc/sys/kernel/random/boot_id'),
                            ('time_namespace', '/proc/self/ns/time')):
            try:
                report['clock'][label] = os.readlink(path) if label == 'time_namespace' else Path(path).read_text().strip()
            except OSError:
                report['clock'][label] = None
        report['memory']['after_initialize'] = memory_snapshot(adapter)
        profile_after_initialize = copy.deepcopy(adapter._pool.performance_snapshot())
        if not args.skip_warmup:
            stage = 'warmup'
            report['warmup'] = synthesize_case(
                adapter, args.warmup_text or args.text[0], output_dir / 'warmup.wav')
            report['memory']['after_warmup'] = memory_snapshot(adapter)
        profile_after_warmup = copy.deepcopy(adapter._pool.performance_snapshot())
        report['performance_warmup'] = profile_delta(profile_after_warmup, profile_after_initialize)
        if args.start_barrier_dir is not None:
            stage = 'start_barrier'
            report['start_barrier'] = wait_for_start_barrier(
                args.start_barrier_dir.resolve(), args.worker_id,
                args.barrier_parties, args.barrier_timeout_s)
        stage = 'timed_cases'
        profile_before_timed = copy.deepcopy(adapter._pool.performance_snapshot())
        for repeat in range(args.repeat):
            for text_index, text in enumerate(args.text):
                index = len(report['cases'])
                row = synthesize_case(adapter, text, output_dir / f'case-{index:03d}.wav')
                row.update(repeat=repeat, text_index=text_index,
                           memory_after=memory_snapshot(adapter))
                report['cases'].append(row)
                print(json.dumps({'event': 'case_complete', **row}, ensure_ascii=False), flush=True)
        report['memory']['after_timed_cases'] = memory_snapshot(adapter)
        profile = adapter._pool.performance_snapshot()
        if profile is not None:
            report['performance'] = profile
            report['performance_scope'] = 'Cumulative graph counters include warmup and all timed cases'
            report['performance_timed_cases'] = profile_delta(profile, profile_before_timed)
            report['performance_timed_cases_scope'] = 'Counter deltas for timed requests only; excludes initialization and warmup'
        report['ok'] = True
    except Exception as error:
        report.update(stage=stage, error_type=type(error).__name__, error=str(error))
        traceback.print_exc(file=sys.stderr)
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                report.update(ok=False, close_error_type=type(error).__name__, close_error=str(error))
                traceback.print_exc(file=sys.stderr)
        report['memory']['after_close'] = memory_snapshot()
        (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
