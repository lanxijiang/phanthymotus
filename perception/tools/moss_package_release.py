#!/usr/bin/env python3
"""Package reviewed MOSS assets and validated Orin engines, without publishing.

The uncompressed tar contains regular files only and works with moss_assets.py.
Its SHA/byte pins must be reviewed and configured by the caller. This utility
never changes source assets, builds an engine, runs a model, or downloads files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from types import ModuleType

import numpy as np


FRAME_NAME = 'local_fixed_sampled_frame'
TARGET_TRT = '10.4.0'
TARGET_CC = '8.7'
BASE_PREFIXES = {'onnx', 'source', 'validation'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def relative_name(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'Invalid asset path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Unsafe asset path: {value!r}')
    return path.as_posix()


def source_file(root, name):
    name = relative_name(name)
    root = Path(root).resolve()
    path = root / name
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing asset or link outside asset root: {name}')
    return path.resolve()


def reject_git_output(directory):
    resolved = Path(directory).resolve()
    if any((parent / '.git').is_file() or (parent / '.git/HEAD').is_file()
           for parent in (resolved, *resolved.parents)):
        raise ValueError('Model packages must remain outside every Git checkout')
    return resolved


def reviewed_runtime_modules():
    # Load only metadata/provenance modules. Importing moss_deploy.__init__ would
    # unnecessarily pull in the speech frontend and inference runtime.
    name = '_moss_release_reviewed_runtime'
    directory = Path(__file__).resolve().parents[1] / 'plugins/moss_deploy'
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package
    elif sys.modules[name].__path__ != [str(directory)]:
        raise RuntimeError('Another deployment source is already loaded')
    return (importlib.import_module(name + '.assets'),
            importlib.import_module(name + '.frame_contract'))


def verify_runtime_assets(root, included_names):
    assets_module, _ = reviewed_runtime_modules()
    assets = assets_module.ModelAssets(root / 'analysis.json', root / 'source/MOSS-TTS-Nano',
                                       streaming_codec_meta=root / 'onnx/codec/codec_browser_onnx_meta.json')
    paths = [*assets.graph_paths.values(), *assets.voice_paths.values(), assets.tts_meta_path,
             assets.config_path, assets.tokenizer_path, assets.prompting_path,
             assets.streaming_path, assets.streaming_source_step_path]
    for path in paths:
        if path is None or not path.resolve().is_relative_to(root):
            raise ValueError('Runtime metadata points outside the portable asset root')
        if path.resolve().relative_to(root).as_posix() not in included_names:
            raise ValueError(f'Runtime-required file is missing from the source manifest: {path}')
    if set(assets.precomputed_reference_codes) != set(assets.voice_paths):
        raise ValueError('Every advertised built-in voice needs verified original reference codes')
    return assets


def verify_target(report):
    gpu = report.get('gpu', {})
    if (report.get('status') != 'built' or report.get('tensorrt_version') != TARGET_TRT
            or gpu.get('name') != 'Orin' or gpu.get('compute_capability') != TARGET_CC):
        raise ValueError('Release engines must have successful TensorRT 10.4.0 / Orin SM87 provenance')


def read_engine(name, plan):
    plan = Path(plan).resolve()
    if not plan.is_file():
        raise FileNotFoundError(plan)
    report_path = plan.parent / 'build_report.json'
    report = json.loads(report_path.read_text())
    verify_target(report)
    entries = [row for row in report.get('engines', []) if row.get('name') == name]
    if len(entries) != 1:
        raise ValueError(f'Expected exactly one build report entry for {name}')
    entry = entries[0]
    if (entry.get('sha256') != sha256(plan) or entry.get('bytes') != plan.stat().st_size
            or entry.get('tf32_enabled') is not False):
        raise ValueError(f'Engine byte/hash/precision provenance mismatch: {name}')
    if name == 'codec_decode_step' and entry.get('precision') != 'fp32':
        raise ValueError('The validated original streaming codec must remain FP32')
    return report, entry, report_path


def verify_step_validation(name, report, entry, assets, root):
    if entry.get('onnx_sha256') != sha256(assets.graph_paths[name]):
        raise ValueError(f'Engine was built from another original graph: {name}')
    expected_samples = assets.graphs[name].get('validation_samples', [])
    rows = entry.get('validation', [])
    if (not expected_samples or len(rows) != len(expected_samples)
            or {row.get('sample') for row in rows} != set(expected_samples)):
        raise ValueError(f'Original numerical validation is incomplete: {name}')
    for row in rows:
        with np.load(source_file(root, row['sample']), allow_pickle=False) as archive:
            expected_outputs = {key.removeprefix('ref__') for key in archive.files if key.startswith('ref__')}
        outputs = row.get('outputs', {})
        if not expected_outputs or set(outputs) != expected_outputs:
            raise ValueError(f'Missing original validation outputs: {name}')
        for output in outputs.values():
            maximum, relative = output.get('max_abs_error'), output.get('relative_rms_error')
            if (output.get('numerical_pass') is not True or output.get('shape_match') is not True
                    or output.get('finite') is not True
                    or not isinstance(maximum, (int, float)) or not isinstance(relative, (int, float))
                    or not math.isfinite(maximum) or not math.isfinite(relative)
                    or min(maximum, relative) < 0 or not (maximum <= 1e-4 or relative <= .05)
                    or output.get('exact_value_match_required') is True and maximum != 0):
                raise ValueError(f'Failed numerical validation output: {name}')


def collect_release(*, assets, global_plan, local_plan, codec_plan, frame_plan, frame_contract, frame_onnx, release_id, wetext_zh_cache=None):
    root = Path(assets).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for row in manifest.get('files', []):
        if row.get('required_for_target_build', True) is not True:
            continue
        name = relative_name(row['path'])
        if name.split('/')[0] not in BASE_PREFIXES or name in files:
            raise ValueError('Unexpected or duplicate portable manifest path')
        path = source_file(root, name)
        if path.stat().st_size != row.get('bytes') or sha256(path) != row.get('sha256'):
            raise ValueError(f'Original portable asset hash/size mismatch: {name}')
        files[name] = path
    if not files:
        raise ValueError('Empty original portable manifest')
    files['manifest.json'] = manifest_path
    files['analysis.json'] = source_file(root, 'analysis.json')
    analysis = json.loads(files['analysis.json'].read_text())
    for graph in analysis['graphs']:
        relative_name(graph['file'])
        for name in graph.get('validation_samples', []):
            relative_name(name)
    runtime_assets = verify_runtime_assets(root, set(files))
    plans = {'global_unified': Path(global_plan).resolve(), 'local_cached_step': Path(local_plan).resolve(),
             'codec_decode_step': Path(codec_plan).resolve(), FRAME_NAME: Path(frame_plan).resolve()}
    reports, entries = {}, {}
    for name, plan in plans.items():
        report, entry, report_path = read_engine(name, plan)
        if name != FRAME_NAME:
            verify_step_validation(name, report, entry, runtime_assets, root)
        reports[name], entries[name] = report, entry
        files[f'plans/{name}/{name}.plan'] = plan
        files[f'plans/{name}/build_report.json'] = report_path
    frame_contract, frame_onnx = Path(frame_contract).resolve(), Path(frame_onnx).resolve()
    _, contract_module = reviewed_runtime_modules()
    contract = contract_module.load_frame_contract(plans[FRAME_NAME], frame_contract, runtime_assets)
    entry, report = entries[FRAME_NAME], reports[FRAME_NAME]
    expected_labels = {f'hidden-{hidden}-uniform-{uniform}' for hidden in range(4) for uniform in range(8)}
    if (report.get('sample_count') != 32 or len(entry.get('validation', [])) != 32
            or {row.get('sample') for row in entry['validation']} != expected_labels
            or report.get('reference_plan_sha256') != entries['local_cached_step']['sha256']
            or entry.get('strongly_typed') is not True
            or contract.get('sampling_semantics', {}).get('cdf_accumulation') != 'parallel_prefix_scan_fp32'):
        raise ValueError('Frame release requires the v4 strongly typed engine and complete 32 x 17 validation')
    references = report.get('reference_samples', [])
    wanted_refs = runtime_assets.graphs['local_cached_step']['validation_samples']
    if len(references) != 4 or {row.get('path') for row in references} != set(wanted_refs):
        raise ValueError('Frame report lacks the four original local reference samples')
    for row in references:
        if sha256(source_file(root, row['path'])) != row.get('sha256'):
            raise ValueError('Frame reference sample hash does not match the original asset')
    if sha256(frame_onnx) != contract.get('onnx_sha256'):
        raise ValueError('Candidate ONNX hash differs from the validated frame contract')
    source_local = runtime_assets.graph_paths['local_cached_step']
    external = contract.get('external_weights', {})
    if not external:
        raise ValueError('Candidate contract must identify its original external weights')
    for name, row in external.items():
        path = source_file(source_local.parent, name)
        bundled = path.relative_to(root).as_posix()
        if bundled not in files or path.stat().st_size != row.get('bytes') or sha256(path) != row.get('sha256'):
            raise ValueError('Candidate external weights do not match the original portable assets')
    candidate_onnx_name = (source_local.parent.relative_to(root) / (FRAME_NAME + '.onnx')).as_posix()
    candidate_contract_name = 'candidates/frame-v4/frame-contract.json'
    for name, path in ((candidate_onnx_name, frame_onnx), (candidate_contract_name, frame_contract)):
        if name in files:
            raise ValueError(f'Release artifact conflicts with original asset: {name}')
        files[name] = path
    # Keep the original step plan because the current pool validates its path
    # before it unloads the superseded engine in fused-frame mode.
    engine_map = {name: f'plans/{name}/{name}.plan' for name in plans if name != FRAME_NAME}
    candidate = {'version': 1, 'graph_name': FRAME_NAME,
                 'plan': f'plans/{FRAME_NAME}/{FRAME_NAME}.plan', 'contract': candidate_contract_name,
                 'onnx': candidate_onnx_name, 'cuda_graph': True, 'resident_state': True, 'fast_sampling': True}
    generated = {'engine-map.json': json_bytes(engine_map), 'frame-candidate.json': json_bytes(candidate)}
    cache_metadata = None
    if wetext_zh_cache is not None:
        cache = Path(wetext_zh_cache).resolve()
        destination_prefix = 'source/MOSS-TTS-Nano/.cache/wetext_zh_no_erhua_keep_punct/'
        cached_files = []
        for filename in ('zh_tn_tagger.fst', 'zh_tn_verbalizer.fst'):
            path = source_file(cache, filename)
            if path.stat().st_size == 0:
                raise ValueError('Empty precompiled Chinese WeText cache')
            name = destination_prefix + filename
            if name in files:
                raise ValueError('Chinese frontend cache conflicts with original source manifest')
            files[name] = path
            cached_files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
        cache_metadata = {'version': 1, 'WeTextProcessing': '1.2.0', 'Pynini': '2.1.7',
                          'normalizer_options': {'remove_interjections': False, 'remove_erhua': False,
                                                 'full_to_half': False},
                          'source_python_modified': False,
                          'purpose': 'Reuse the target-tested Chinese normalizer cache at cold startup',
                          'files': cached_files}
        generated[destination_prefix + 'release-cache-metadata.json'] = json_bytes(cache_metadata)
    inventory = []
    for name, path in sorted(files.items()):
        inventory.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    for name, data in sorted(generated.items()):
        inventory.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    release_manifest = {
        'version': 1, 'release_id': release_id, 'model': manifest.get('model'),
        'target': {'gpu': 'Orin', 'compute_capability': TARGET_CC, 'tensorrt_version': TARGET_TRT, 'jetpack': '6.1'},
        'source_manifest_sha256': sha256(manifest_path), 'source_assets_preserved': True,
        'frame_validation_samples': 32, 'frame_validation_steps_per_sample': 17,
        'active_graphs': ['global_unified', FRAME_NAME, 'codec_decode_step'],
        'retained_inactive_graph': 'local_cached_step; required by current startup availability check',
        'frontend_dependencies': 'Provided by the deployment image, not included in this model archive',
        'precompiled_chinese_frontend_cache': cache_metadata,
        'files': sorted(inventory, key=lambda row: row['path']),
        'self_hash_note': 'The outer tar SHA covers this manifest; its own row is deliberately excluded.',
    }
    generated['release-manifest.json'] = json_bytes(release_manifest)
    return files, generated, release_manifest


class _HashingReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def write_archive(destination, files, generated, manifest):
    destination = Path(destination)
    expected = {row['path']: row for row in manifest['files']}
    temporary = destination.with_suffix(destination.suffix + '.partial')
    with temporary.open('xb') as output:
        try:
            with tarfile.open(fileobj=output, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(set(files) | set(generated)):
                    relative_name(name)
                    info = tarfile.TarInfo(name)
                    info.mode, info.mtime, info.uid, info.gid = 0o644, 0, 0, 0
                    info.uname = info.gname = ''
                    if name in generated:
                        data = generated[name]
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
                    else:
                        path = files[name]
                        info.size = expected[name]['bytes']
                        with path.open('rb') as source:
                            reader = _HashingReader(source)
                            archive.addfile(info, reader)
                            if (reader.count != info.size or reader.digest.hexdigest() != expected[name]['sha256']
                                    or path.stat().st_size != info.size):
                                raise ValueError(f'Source changed while packaging: {name}')
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    # A fresh output directory is owned by this invocation; still use exclusive
    # publication so a concurrently created archive can never be overwritten.
    os.link(temporary, destination)
    temporary.unlink()


def package_release(*, output_dir, release_id='moss-orin-jp61-frame-v4', **sources):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}', release_id):
        raise ValueError('Invalid release identifier')
    output_dir = reject_git_output(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Choose a new release directory: {output_dir}')
    files, generated, manifest = collect_release(release_id=release_id, **sources)
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = output_dir / (release_id + '.tar')
    write_archive(archive, files, generated, manifest)
    report = {'version': 1, 'release_id': release_id, 'archive': archive.name,
              'asset_sha256': sha256(archive), 'asset_bytes': archive.stat().st_size,
              'files': len(files) + len(generated), 'uncompressed_asset_bytes': sum(row['bytes'] for row in manifest['files']) + len(generated['release-manifest.json']),
              'frame_candidate': 'frame-candidate.json', 'published': False,
              'download_pin_update_required': True, 'target': manifest['target']}
    for name, data in generated.items():
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output_dir / 'release.json').write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'global-plan', 'local-plan', 'codec-plan', 'frame-plan', 'frame-contract', 'frame-onnx', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-id', default='moss-orin-jp61-frame-v4')
    parser.add_argument('--wetext-zh-cache', type=Path, help='Target-tested cache containing zh_tn_tagger.fst and zh_tn_verbalizer.fst')
    args = parser.parse_args()
    print(json.dumps(package_release(**vars(args)), indent=2), flush=True)


if __name__ == '__main__':
    main()
