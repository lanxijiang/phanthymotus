"""Immutable xiaoyou01/V6 identity and hashing; no model/backend imports."""
import hashlib
import json
import re
from pathlib import Path

SOURCE_SHA = '451487c68462f0a5400ea49798625d8bf5ea2def858e0cbff56427d7e9d76a14'
BASE_FRONTEND_SHA = '548159e3281ffe9bd2040865d50db83e4bb6f49a057199432ce4dd0468e6b53e'
FRONTEND_SHA = '65568e99370ffcdba21d4f05e725df3efd32e02322717b4b58d09c2af3828887'
NORMALIZATION_REVISION = 'xiaoyou-v6-20261109'
NORMALIZATION_MODULE_SHAS = {'text_normalization.py': 'a0bf72acad08d78211a2bebc76a54a0374d7464e94a14a4251947650339aadad', 'math_normalization.py': '6f38080b448dd7f0974a3d18739e9f7e8856701dac5e88809126e4c76d4a2381', 'context_pronunciation.py': '752e236679bee321226be40120b6d429447fd70856410243d2dbeff845ba9716'}
TERMS_SHA = '3a0f34ad00099a1041a8f9e40e30802c08822049c20d9dfbb74905f8fe91dff4'
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
