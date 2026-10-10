"""Offline checks for benchmark timing boundaries and error reporting."""

import importlib.util
import concurrent.futures
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest


@pytest.fixture
def benchmark():
    path = Path(__file__).resolve().parents[1] / 'tools/moss_benchmark.py'
    spec = importlib.util.spec_from_file_location('_moss_benchmark_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_silent_first_chunk_is_separate_from_nonzero_audio(benchmark, monkeypatch, tmp_path):
    class Adapter:
        def synthesize_stream(self, text):
            yield b''
            yield b'\x00\x00\x00\x00'
            yield b'\x01\x00\xff\xff'

    ticks = iter([10.0, 10.1, 10.2, 10.4, 10.7])
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: next(ticks))
    path = tmp_path / 'actual-captured-pcm.wav'
    result = benchmark.synthesize_case(Adapter(), 'test', path)
    assert result['first_pcm_s'] == pytest.approx(0.2)
    assert result['first_nonzero_pcm_chunk_s'] == pytest.approx(0.4)
    assert result['elapsed_s'] == pytest.approx(0.7)
    assert result['started_monotonic_s'] == 10.0
    assert result['ended_monotonic_s'] == 10.7
    assert result['audio_s'] == 4 / 16_000
    assert result['rtf'] == pytest.approx(0.7 / (4 / 16_000))
    with wave.open(str(path)) as audio:
        assert (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) == (1, 16_000, 2)
        assert audio.readframes(4) == b'\x00\x00\x00\x00\x01\x00\xff\xff'
    assert result['wav_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_initialization_failure_is_a_failed_report(benchmark, monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError('Test missing engine')

    monkeypatch.setattr(benchmark, 'MossTTSAdapter', unavailable)
    assert benchmark.main(['--model-dir', str(tmp_path / 'assets'), '--text', 'こんにちは。',
                           '--output-dir', str(tmp_path / 'result')]) == 1
    report = json.loads((tmp_path / 'result/report.json').read_text())
    assert report['ok'] is False
    assert report['stage'] == 'initialize'
    assert report['error'] == 'Test missing engine'
    assert not report['cases']


def test_warmup_uses_requested_japanese_text(benchmark, monkeypatch, tmp_path):
    calls = []

    class Pool:
        backend = 'offline-test-double'

        def memory_snapshot(self):
            return {'test_double': True}

        def performance_snapshot(self):
            return None

    class Adapter:
        def __init__(self, *args, **kwargs):
            assert kwargs['language'] == 'ja'
            assert kwargs['instances'] == 1
            self._pool = Pool()

        def synthesize_stream(self, text):
            calls.append(text)
            yield b'\x01\x00' * 160

        def close(self):
            pass

    monkeypatch.setattr(benchmark, 'MossTTSAdapter', Adapter)
    assert benchmark.main(['--model-dir', str(tmp_path / 'assets'), '--language', 'ja',
                           '--text', '音声合成です。', '--warmup-text', 'こんにちは。',
                           '--repeat', '2', '--output-dir', str(tmp_path / 'result')]) == 0
    assert calls == ['こんにちは。', '音声合成です。', '音声合成です。']
    report = json.loads((tmp_path / 'result/report.json').read_text())
    assert len(report['cases']) == 2
    assert report['warmup']['text'] == 'こんにちは。'
    assert (tmp_path / 'result/warmup.wav').is_file()


def test_existing_output_is_preserved_before_initialization(benchmark, monkeypatch, tmp_path):
    directory = tmp_path / 'result'
    directory.mkdir()
    old_report = directory / 'report.json'
    old_report.write_text('previous evidence')

    def must_not_initialize(*args, **kwargs):
        pytest.fail('Must reject existing output before loading the model')

    monkeypatch.setattr(benchmark, 'MossTTSAdapter', must_not_initialize)
    with pytest.raises(SystemExit) as error:
        benchmark.main(['--model-dir', str(tmp_path / 'assets'), '--text', 'hello',
                        '--output-dir', str(directory)])
    assert error.value.code == 2
    assert old_report.read_text() == 'previous evidence'


def test_engine_provenance_records_loaded_plans_and_scoped_flags(benchmark, monkeypatch, tmp_path):
    plan = tmp_path / 'global.plan'
    plan.write_bytes(b'target GPU engine bytes')
    (tmp_path / 'engine-map.json').write_text(json.dumps({
        'global_unified': 'global.plan', 'codec_encoder': 'optional-missing.plan'}))
    adapter = SimpleNamespace(_pool=SimpleNamespace(
        backend='tensorrt-device-0', engines={'global_unified': object()}))
    monkeypatch.setenv('MOSS_TRT_REUSE_LOCAL_BUFFERS', '1')
    monkeypatch.setenv('MOSS_FAST_SAMPLING', '1')
    monkeypatch.setenv('UNRELATED_PASSWORD', 'must-not-appear')
    metadata = benchmark.runtime_metadata(adapter, tmp_path)
    assert list(metadata['engines']) == ['global_unified']
    assert metadata['engines']['global_unified'] == {
        'path': str(plan), 'bytes': plan.stat().st_size,
        'sha256': hashlib.sha256(plan.read_bytes()).hexdigest()}
    assert metadata['environment']['MOSS_TRT_REUSE_LOCAL_BUFFERS'] == '1'
    assert metadata['environment']['MOSS_FAST_SAMPLING'] == '1'
    for name in ('sampling.py', 'selected_logits.py'):
        relative = 'plugins/moss_deploy/' + name
        assert metadata['code_sha256'][relative] == hashlib.sha256(
            (benchmark.PERCEPTION_DIR / relative).read_bytes()).hexdigest()
    assert 'UNRELATED_PASSWORD' not in metadata['environment']


@pytest.mark.parametrize('keep_original_plan', [True, False])
def test_selected_plan_provenance_keeps_original_mapping(benchmark, monkeypatch, tmp_path, keep_original_plan):
    original = tmp_path / 'original.plan'
    if keep_original_plan:
        original.write_bytes(b'original full local engine')
    selected = tmp_path / 'selected.plan'
    selected.write_bytes(b'actual selected local engine')
    contract = {'version': 1, 'text_token_ids': [9, 10]}
    contract_path = tmp_path / 'selected-contract.json'
    contract_path.write_text(json.dumps(contract))
    mapping = {'local_cached_step': 'original.plan'}
    (tmp_path / 'engine-map.json').write_text(json.dumps(mapping))
    adapter = SimpleNamespace(_pool=SimpleNamespace(
        backend='tensorrt-device-0', engines={'local_cached_step': object()},
        selected_local_contract=contract))
    monkeypatch.setenv('MOSS_TRT_SELECTED_LOCAL_PLAN', str(selected))
    monkeypatch.setenv('MOSS_TRT_SELECTED_LOCAL_CONTRACT', str(contract_path))
    metadata = benchmark.runtime_metadata(adapter, tmp_path)
    assert metadata['engines']['local_cached_step'] == {
        'path': str(selected), 'bytes': selected.stat().st_size,
        'sha256': hashlib.sha256(selected.read_bytes()).hexdigest()}
    assert metadata['engine_map']['entries'] == mapping
    configured = metadata['configured_engines']['local_cached_step']
    assert configured['path'] == str(original)
    if keep_original_plan:
        assert configured['sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
    else:
        assert configured['exists'] is False
    assert metadata['selected_local_contract']['sha256'] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
    assert metadata['selected_local_contract']['contents'] == contract
    # Disk changes must not silently claim to describe the loaded model.
    contract_path.write_text(json.dumps({'version': 1, 'text_token_ids': [9, 11]}))
    with pytest.raises(ValueError, match='changed after model initialization'):
        benchmark.runtime_metadata(adapter, tmp_path)


def test_fused_plan_provenance_hashes_loaded_optional_engine(benchmark, monkeypatch, tmp_path):
    original, loaded, fused = (tmp_path / name for name in ('original.plan', 'loaded.plan', 'fused.plan'))
    original.write_bytes(b'archive configured engine')
    loaded.write_bytes(b'actual loaded global engine')
    fused.write_bytes(b'actual optional fused engine')
    contract = {'version': 1, 'graph_name': 'local_fixed_sampled_frame'}
    contract_path = tmp_path / 'frame-contract.json'
    contract_path.write_text(json.dumps(contract))
    (tmp_path / 'engine-map.json').write_text(json.dumps({'global_unified': 'original.plan'}))
    pool = SimpleNamespace(
        backend='tensorrt-device-0',
        engines={'global_unified': object(), 'local_fixed_sampled_frame': object()},
        effective_plan_paths={'global_unified': loaded, 'local_fixed_sampled_frame': fused},
        frame_contract=contract,
    )
    adapter = SimpleNamespace(_pool=pool)
    monkeypatch.setenv('MOSS_TRT_FRAME_PLAN', str(fused))
    monkeypatch.setenv('MOSS_TRT_FRAME_CONTRACT', str(contract_path))
    metadata = benchmark.runtime_metadata(adapter, tmp_path)
    assert metadata['engines']['global_unified']['sha256'] == hashlib.sha256(loaded.read_bytes()).hexdigest()
    assert metadata['configured_engines']['global_unified']['sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert metadata['engines']['local_fixed_sampled_frame']['sha256'] == hashlib.sha256(fused.read_bytes()).hexdigest()
    assert metadata['configured_engines']['local_fixed_sampled_frame']['optional_graph'] is True
    assert metadata['frame_contract']['contents'] == contract
    monkeypatch.setenv('MOSS_TRT_FRAME_PLAN', str(original))
    with pytest.raises(ValueError, match='environment changed after initialization'):
        benchmark.runtime_metadata(adapter, tmp_path)
    monkeypatch.setenv('MOSS_TRT_FRAME_PLAN', str(fused))
    contract_path.write_text(json.dumps({**contract, 'version': 2}))
    with pytest.raises(ValueError, match='contract changed after model initialization'):
        benchmark.runtime_metadata(adapter, tmp_path)


def test_profile_phases_exclude_warmup_and_keep_cumulative(benchmark, monkeypatch, tmp_path):
    class Pool:
        backend = 'offline-test-double'

        def __init__(self):
            self.profile = {'timing_scope': 'fixture', 'by_graph': {}}

        def memory_snapshot(self):
            return {}

        def performance_snapshot(self):
            # Deliberately return mutable counters to verify independent snapshots.
            return self.profile

    class Adapter:
        def __init__(self, *args, **kwargs):
            self._pool = Pool()

        def synthesize_stream(self, text):
            counters = self._pool.profile['by_graph'].setdefault(
                'local_cached_step', {'runs': 0, 'execute_ns': 0})
            counters['runs'] += 1
            counters['execute_ns'] += 100
            yield b'\x01\x00' * 160

        def close(self):
            pass

    monkeypatch.setattr(benchmark, 'MossTTSAdapter', Adapter)
    directory = tmp_path / 'result'
    assert benchmark.main(['--model-dir', str(tmp_path / 'assets'), '--text', 'hello',
                           '--repeat', '2', '--output-dir', str(directory)]) == 0
    report = json.loads((directory / 'report.json').read_text())
    assert report['performance']['by_graph']['local_cached_step']['runs'] == 3
    assert report['performance_warmup']['by_graph']['local_cached_step'] == {
        'runs': 1, 'execute_ns': 100}
    assert report['performance_timed_cases']['by_graph']['local_cached_step'] == {
        'runs': 2, 'execute_ns': 200}
    assert report['seed'] == 1234
    assert report['repeat'] == 2


def test_three_workers_share_one_release_and_cannot_reuse_markers(benchmark, tmp_path):
    directory = tmp_path / 'barrier'
    # Nonleader workers may arrive before worker zero creates the directory.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
        pending = [workers.submit(benchmark.wait_for_start_barrier, directory, i, 3, 3)
                   for i in (2, 1, 0)]
        rows = [future.result(timeout=5) for future in pending]
    assert {row['worker_id'] for row in rows} == {0, 1, 2}
    assert len({row['leader_nonce'] for row in rows}) == 1
    assert len({row['nonce'] for row in rows}) == 3
    release = json.loads((directory / 'release.json').read_text())
    assert all(release['worker_nonces'][row['worker_id']] == row['nonce'] for row in rows)
    previous_files = {p.name: p.read_bytes() for p in directory.iterdir()}
    for worker_id in (0, 1):
        with pytest.raises(FileExistsError):
            benchmark.wait_for_start_barrier(directory, worker_id, 3, 0.1)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == previous_files


def test_barrier_timeout_is_bounded_and_records_only_own_failure(benchmark, tmp_path):
    directory = tmp_path / 'barrier'
    with pytest.raises(TimeoutError):
        benchmark.wait_for_start_barrier(directory, 0, 2, 0.01)
    assert (directory / 'worker-0.failed.json').is_file()
    assert not (directory / 'worker-1.failed.json').exists()
    assert not (directory / 'release.json').exists()


def test_worker_id_cannot_escape_barrier_marker_names(benchmark, tmp_path):
    with pytest.raises(SystemExit) as error:
        benchmark.main(['--model-dir', str(tmp_path / 'assets'), '--text', 'hello',
                        '--output-dir', str(tmp_path / 'result'),
                        '--start-barrier-dir', str(tmp_path / 'barrier'), '--worker-id', '-1'])
    assert error.value.code == 2
    assert not (tmp_path / 'result').exists()


def test_seed_option_does_not_claim_adapter_supports_other_seeds(benchmark, tmp_path):
    with pytest.raises(SystemExit) as error:
        benchmark.main(['--model-dir', str(tmp_path / 'assets'), '--text', 'hello',
                        '--output-dir', str(tmp_path / 'result'), '--seed', '99'])
    assert error.value.code == 2
    assert not (tmp_path / 'result').exists()
