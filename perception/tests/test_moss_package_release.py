"""Offline release packaging checks with tiny synthetic assets and GPU reports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[1] / 'tools/moss_package_release.py'
    spec = importlib.util.spec_from_file_location('_moss_release_tool_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def release_inputs(tmp_path, tool):
    root = tmp_path / 'assets'
    root.mkdir()
    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path
    controls = {'audio_pad_token_id': 0, 'audio_start_token_id': 6, 'audio_end_token_id': 7,
                'audio_user_slot_token_id': 8, 'audio_assistant_slot_token_id': 9}
    model = {'n_vq': 16, 'row_width': 17, 'audio_codebook_sizes': [1024] * 16, **controls}
    save('onnx/compact/tts_browser_onnx_meta.json', {'model_config': model})
    save('onnx/compact/frontend/config.json', model)
    save('onnx/compact/frontend/tokenizer.model', b'tiny tokenizer fixture')
    save('onnx/compact/frontend/prompting.py', b'# fixture; not executed\n')
    save('source/MOSS-TTS-Nano/ort_cpu_runtime.py', b'# fixture\n')
    save('source/MOSS-TTS-Nano/onnx_tts_runtime.py', b'# fixture\n')
    save('source/MOSS-TTS-Nano/moss_tts_nano_runtime.py', b'_DEFAULT_VOICE_FILES={"voice": ["voice.wav"]}\nDEFAULT_VOICE="voice"\n')
    save('source/MOSS-TTS-Nano/app_gpu.py', b'SUPPORTED_LANGUAGES=["zh","ja","ru","en","es","fr","th","de"]\n_LATIN_STOPWORDS={}\n_LATIN_CHAR_HINTS={}\ndef detect_language(text):\n return "zh"\n')
    voice = save('source/MOSS-TTS-Nano/assets/audio/voice.wav', b'voice bytes')
    codec_config = {'sample_rate': 48000, 'channels': 2, 'num_quantizers': 16}
    save('onnx/codec/codec_browser_onnx_meta.json', {'codec_config': codec_config, 'files': {'decode_step': 'codec_decode_step.onnx'}})
    np.savez(root / 'onnx/codec/reference_voice.npz', audio_codes=np.zeros((16, 1, 1), np.int64), audio_codes_lengths=np.array([1]))
    save('onnx/codec/streaming_validation.json', {'reference_audio': [{'voice': 'voice', 'padding_original_codes_exact': True,
         'source_sha256': tool.sha256(voice), 'codes_file': 'reference_voice.npz', 'valid_code_frames': 1}]})
    graphs = []
    for name in ('global_unified', 'local_cached_step', 'codec_encoder', 'codec_decoder', 'codec_decode_step'):
        graph_file = ('onnx/compact/' if name in ('global_unified', 'local_cached_step') else 'onnx/codec/') + name + '.onnx'
        save(graph_file, (name + ' graph fixture').encode())
        samples = []
        for index in range(4 if name == 'local_cached_step' else 1):
            relative = f'validation/{name}/{index}.npz'
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, ref__demo=np.array([1], np.float32))
            samples.append(relative)
        graphs.append({'name': name, 'file': graph_file, 'validation_samples': samples})
    weights = save('onnx/compact/models.data', b'original external weights fixture')
    save('analysis.json', {'graphs': graphs, 'metadata': codec_config, 'frontend': {'directory': 'frontend'}})
    manifest = {'model': 'tiny fake fixture; never used for model inference', 'files': []}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'analysis.json':
            manifest['files'].append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                                      'sha256': tool.sha256(path), 'required_for_target_build': True})
    save('manifest.json', manifest)
    save('.moss-assets-ready.json', {'archive_sha256': 'obsolete marker; never package'})
    plans = {}
    for name in ('global_unified', 'local_cached_step', 'codec_decode_step', tool.FRAME_NAME):
        directory = tmp_path / 'plans' / name
        directory.mkdir(parents=True)
        plan = directory / (name + '.plan')
        plan.write_bytes((name + ' plan fixture').encode())
        entry = {'name': name, 'file': plan.name, 'bytes': plan.stat().st_size, 'sha256': tool.sha256(plan),
                 'tf32_enabled': False, 'precision': 'fp32' if name == 'codec_decode_step' else 'fp16'}
        report = {'status': 'built', 'tensorrt_version': '10.4.0', 'gpu': {'name': 'Orin', 'compute_capability': '8.7'}, 'engines': [entry]}
        if name != tool.FRAME_NAME:
            graph = next(row for row in graphs if row['name'] == name)
            entry['onnx_sha256'] = tool.sha256(root / graph['file'])
            entry['validation'] = [{'sample': sample, 'outputs': {'demo': {'shape_match': True, 'finite': True,
                  'numerical_pass': True, 'max_abs_error': .001, 'relative_rms_error': .002}}} for sample in graph['validation_samples']]
        (directory / 'build_report.json').write_text(json.dumps(report))
        plans[name] = plan
    _, contract_module = tool.reviewed_runtime_modules()
    frame_onnx = tmp_path / 'candidate.onnx'
    frame_onnx.write_bytes(b'candidate frame graph fixture')
    contract = {'version': 1, 'graph_name': tool.FRAME_NAME, 'n_vq': 16, 'audio_channels': 16,
                'audio_vocab_size': 1024, 'batch_size': 1, 'steps_per_frame': 17, 'text_token_ids': [9, 7],
                'generation_defaults': dict(contract_module.FRAME_GENERATION_DEFAULTS),
                'sampling': {'do_sample': True, **contract_module.FRAME_GENERATION_DEFAULTS},
                'sampling_semantics': {'cdf_normalization': 'divide_by_last_cumulative_probability', 'cdf_accumulation': 'parallel_prefix_scan_fp32'},
                'source_onnx_sha256': tool.sha256(root / 'onnx/compact/local_cached_step.onnx'), 'onnx_sha256': tool.sha256(frame_onnx),
                'random_input': {'name': 'random_u', 'dtype': 'float32', 'shape': [1, 17]},
                'external_weights': {'models.data': {'bytes': weights.stat().st_size, 'sha256': tool.sha256(weights)}}}
    contract_path = tmp_path / 'frame-contract.json'
    contract_path.write_text(json.dumps(contract))
    plan = plans[tool.FRAME_NAME]
    report = json.loads((plan.parent / 'build_report.json').read_text())
    entry = report['engines'][0]
    entry.update(onnx_sha256=tool.sha256(frame_onnx), contract_sha256=tool.sha256(contract_path), strongly_typed=True)
    rows = []
    for index in range(32):
        decision = index % 2
        rows.append({'sample': f'hidden-{index // 8}-uniform-{index % 8}', 'model_logits_pass': True,
                     'sampling_exact': True, 'finite': True, 'outputs_valid': True, 'should_continue': decision,
                     'sampling_checks': [{'step': step, 'exact': True, 'actual': 1 - decision if not step else 20,
                         'expected': 1 - decision if not step else 20} for step in range(17)],
                     'logit_checks': [{'step': step, 'numerical_pass': True, 'shape_match': True, 'finite': True,
                         'reference_finite': True, 'max_abs_error': .001, 'relative_rms_error': .002} for step in range(17)]})
    entry['validation'] = rows
    local_graph = next(row for row in graphs if row['name'] == 'local_cached_step')
    report.update(validation_pass=True, sample_count=32, contract_sha256=tool.sha256(contract_path),
                  source_onnx_sha256=contract['source_onnx_sha256'], reference_plan_sha256=tool.sha256(plans['local_cached_step']),
                  sampling=copy.deepcopy(contract['sampling']), cdf_normalization=contract['sampling_semantics']['cdf_normalization'],
                  continuation_values_covered=[0, 1], audio_channels_covered=list(range(16)),
                  reference_samples=[{'path': name, 'sha256': tool.sha256(root / name)} for name in local_graph['validation_samples']])
    (plan.parent / 'build_report.json').write_text(json.dumps(report))
    return dict(assets=root, global_plan=plans['global_unified'], local_plan=plans['local_cached_step'],
                codec_plan=plans['codec_decode_step'], frame_plan=plan, frame_contract=contract_path, frame_onnx=frame_onnx)


def test_complete_release_is_reproducible_regular_tar_and_preserves_sources(tool, release_inputs, tmp_path):
    before = {path: path.read_bytes() for path in release_inputs['assets'].rglob('*') if path.is_file()}
    first = tool.package_release(output_dir=tmp_path / 'release1', **release_inputs)
    second = tool.package_release(output_dir=tmp_path / 'release2', **release_inputs)
    assert first['asset_sha256'] == second['asset_sha256']
    assert first['published'] is False
    path = tmp_path / 'release1' / first['archive']
    assert tool.sha256(path) == first['asset_sha256']
    assert path.stat().st_size == first['asset_bytes']
    with tarfile.open(path, 'r:') as archive:
        assert all(member.isfile() for member in archive)
        names = archive.getnames()
        assert '.moss-assets-ready.json' not in names
        manifest = json.load(archive.extractfile('release-manifest.json'))
        for row in manifest['files']:
            data = archive.extractfile(row['path']).read()
            assert digest(data) == row['sha256'] and len(data) == row['bytes']
        candidate = json.load(archive.extractfile('frame-candidate.json'))
        assert candidate['cuda_graph'] and candidate['plan'] in names and candidate['contract'] in names
        assert candidate['onnx'] == 'onnx/compact/local_fixed_sampled_frame.onnx'
        mapping = json.load(archive.extractfile('engine-map.json'))
        assert set(mapping) == {'global_unified', 'local_cached_step', 'codec_decode_step'}
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize('fault', ['wrong_gpu', 'wrong_trt', 'wrong_arch', 'bad_plan', 'bad_source', 'incomplete_frame', 'wrong_frame_reference'])
def test_provenance_failures_do_not_create_release(tool, release_inputs, tmp_path, fault):
    plan = release_inputs['global_plan']
    report_path = plan.parent / 'build_report.json'
    report = json.loads(report_path.read_text())
    if fault == 'wrong_gpu':
        report['gpu']['name'] = 'A100'
    elif fault == 'wrong_trt':
        report['tensorrt_version'] = '10.1.0'
    elif fault == 'wrong_arch':
        report['gpu']['compute_capability'] = '8.0'
    elif fault == 'bad_plan':
        plan.write_bytes(b'changed')
    elif fault == 'bad_source':
        (release_inputs['assets'] / 'onnx/compact/models.data').write_bytes(b'changed')
    elif fault in ('incomplete_frame', 'wrong_frame_reference'):
        report_path = release_inputs['frame_plan'].parent / 'build_report.json'
        report = json.loads(report_path.read_text())
        if fault == 'incomplete_frame':
            report['engines'][0]['validation'].pop()
            report['sample_count'] = 31
        else:
            report['reference_plan_sha256'] = '0' * 64
    report_path.write_text(json.dumps(report))
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        tool.package_release(output_dir=output, **release_inputs)
    assert not output.exists()


def test_git_output_and_existing_directory_are_rejected(tool, release_inputs, tmp_path):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    (checkout / '.git').write_text('gitdir: elsewhere')
    with pytest.raises(ValueError, match='Git'):
        tool.package_release(output_dir=checkout / 'models', **release_inputs)
    existing = tmp_path / 'existing'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        tool.package_release(output_dir=existing, **release_inputs)


def test_escaping_manifest_path_and_symlink_are_rejected(tool, release_inputs, tmp_path):
    for name in ('../outside', '/absolute', 'onnx/../outside', 'onnx//duplicate', 'onnx\\bad'):
        with pytest.raises(ValueError):
            tool.relative_name(name)
    outside = tmp_path / 'outside.data'
    outside.write_bytes(b'outside')
    link = release_inputs['assets'] / 'onnx/link.data'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='outside'):
        tool.source_file(release_inputs['assets'], 'onnx/link.data')


def test_optional_chinese_cache_is_hashed_and_keeps_original_source(tool, release_inputs, tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    for filename in ('zh_tn_tagger.fst', 'zh_tn_verbalizer.fst'):
        (cache / filename).write_bytes((filename + ' target fixture').encode())
    report = tool.package_release(output_dir=tmp_path / 'release-cache', wetext_zh_cache=cache, **release_inputs)
    with tarfile.open(tmp_path / 'release-cache' / report['archive']) as archive:
        manifest = json.load(archive.extractfile('release-manifest.json'))
        metadata = manifest['precompiled_chinese_frontend_cache']
        assert metadata['WeTextProcessing'] == '1.2.0' and metadata['Pynini'] == '2.1.7'
        assert metadata['normalizer_options'] == {'remove_interjections': False, 'remove_erhua': False, 'full_to_half': False}
        assert metadata['source_python_modified'] is False
        for row in metadata['files']:
            assert digest(archive.extractfile(row['path']).read()) == row['sha256']
    (cache / 'zh_tn_tagger.fst').write_bytes(b'')
    with pytest.raises(ValueError, match='Empty'):
        tool.package_release(output_dir=tmp_path / 'empty-cache-release', wetext_zh_cache=cache, **release_inputs)
