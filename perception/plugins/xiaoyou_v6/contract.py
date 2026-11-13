"""Immutable xiaoyou01/V6 identity and hashing; no model/backend imports."""
import hashlib
import json
import re
from pathlib import Path

SOURCE_SHA = 'a46bd6826d1074b93bc7a17379966d88005cd248136acd24aab4b6883e689bb0'
BASE_FRONTEND_SHA = '548159e3281ffe9bd2040865d50db83e4bb6f49a057199432ce4dd0468e6b53e'
FRONTEND_SHA = '7d7174f2493ca28be641ae3a67cad9256f2cbe874bdd698ff7913b63fc502297'
NORMALIZATION_REVISION = '01yuewen-context-portable-20261113-v4'
NORMALIZATION_MODULE_SHAS = {'context_frontend.py': '4d441a0c9c638effe518974759341831d3c86add96c3045cc530000ad74fb44f', 'context_numeric_normalization.py': '0368c06073f8186bcddf7cc96df6324ddf2768e01bd82c19c1bc274d44bede0e', 'polyphone_context.py': '6c06b7c137278c683728fe97b0bf1787a111521db03fccac9a281bf5fa880758', 'v6_frontend_base.py': 'ad60c0f7966a71339eb7a039945dd596a3c0d9700e563ac1d749a546c1af9cb5', 'text_normalization.py': '9b15c1dfb760e0d4188201c407b1d45c5a0f48145843d2f06372a75acda590a0', 'http_status_codes.py': '1e73e9b3e628c66021cf626679717eae35ffcb9cff1e626657c3dd7cd7af1842', 'math_normalization.py': '35535c52a0a6edb49fce6a272c82f6772fc2e5fd6452c5dbb8166be3c7f6e726', 'context_pronunciation.py': 'e0f8dfbe5270eb890e879c1f57df3fde6494ca2017def36e10875e44d38132dc', 'frontend_base_profile.json': '98de33e38a5cdc26cbe21a82249fe033c7a17c7f7f514fcaf50c1259b5958226', '__init__.py': 'ef867cb875bb64efbf5fb413226d8a9787be4385d09c97edcc3476fa76a104ad', 'text_segmentation.py': '452265f1841dc8f6702883d95ea11be2524a995f9b0ef138fbf79eb79e7ad03c'}
TERMS_SHA = '1d3c68160bd8b94a1eab98042a02b31fe5bbf8fe53bbd6d3ebcf843eb16bfff5'
PHONE_ABI = 'kokoro-v1.0-misaki-0.9.4-xiaoyou-v6'
VOCAB_SHA = '248ee3192acfc27a9907180354a0efaa63389047a6ff78c3f75e2e19466d0b76'
CONFIG_SHA = '5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def profile_sha256(profiles):
    return hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_frontend_files(front, directory):
    """Bind external manifest, fixed-Git contract and actual source files.

    Normalization modules belong to the source package, never model assets.
    Call before importing the frontend or allocating GPU resources.
    """
    if not isinstance(front, dict) or (
            front.get('abi') != PHONE_ABI or front.get('source_sha256') != FRONTEND_SHA
            or front.get('terms_sha256') != TERMS_SHA or front.get('vocab_sha256') != VOCAB_SHA
            or front.get('normalization_revision') != NORMALIZATION_REVISION
            or front.get('normalization_modules') != NORMALIZATION_MODULE_SHAS):
        raise ValueError('V6 frontend/normalization manifest differs from the fixed source contract')
    root = Path(directory).resolve()
    files = {'v6_frontend.py': FRONTEND_SHA, 'v6_terms.json': TERMS_SHA,
             **NORMALIZATION_MODULE_SHAS}
    for name, expected in files.items():
        path = root / name
        if (path.is_symlink() or not path.is_file() or path.resolve().parent != root
                or sha(path) != expected):
            raise ValueError('V6 frontend source missing, linked or SHA-mismatched: ' + name)
    return dict(files)


def frontend_acceptance_metadata(validation):
    """Historical text acceptance does not transfer to a changed frontend."""
    proof = validation.get('frontend_acceptance', {}) if isinstance(validation, dict) else {}
    bound = isinstance(proof, dict) and (
        proof.get('source_sha256') == FRONTEND_SHA
        and proof.get('terms_sha256') == TERMS_SHA
        and proof.get('vocab_sha256') == VOCAB_SHA
        and proof.get('normalization_revision') == NORMALIZATION_REVISION
        and proof.get('normalization_modules') == NORMALIZATION_MODULE_SHAS
        and isinstance(proof.get('report_sha256'), str)
        and re.fullmatch(r'[0-9a-f]{64}', proof['report_sha256']) is not None)
    result = {'frontend_acceptance_bound': bool(bound),
              'frontend_acceptance_report_sha256': proof['report_sha256'] if bound else None}
    for output, field in (
            ('ttft_validated', 'ttft_1s_passed'),
            ('jetson_validated', 'jetson_end_to_end_passed'),
            ('darvin_validated', 'darvin_passed'),
            ('three_process_instances_validated', 'three_process_instances_passed'),
            ('frontend_pcm_parity_validated', 'pcm_parity_exact'),
            ('short_e2e_verified', 'short_e2e_verified')):
        result[output] = bool(bound and proof.get(field) is True)
    return result
