"""Fixed model, frontend and TensorRT identities for Xiaoyou speech."""
import hashlib
import json
from pathlib import Path
import re


SOURCE_SHA = 'a46bd6826d1074b93bc7a17379966d88005cd248136acd24aab4b6883e689bb0'

FRONTEND_SHA = '57ce333178b3215cbd259e6b3aa83f4e97149a0208b21052057e1f24c287455e'

NORMALIZATION_REVISION = '01yuewen-safe-punctuation-first-clause-20261116-candidate'

NORMALIZATION_MODULE_SHAS = {'frontend/__init__.py': '994c535e0e75b35f9c711ae8093a4764134d4128975a5a2b4e2b8262bae699ec',
 'frontend/http_status.py': '1e73e9b3e628c66021cf626679717eae35ffcb9cff1e626657c3dd7cd7af1842',
 'frontend/math.py': '35535c52a0a6edb49fce6a272c82f6772fc2e5fd6452c5dbb8166be3c7f6e726',
 'frontend/normalization.py': 'c9fe5cce9d45507ba084af47cefe86739e1d4f2d5a3eeb4745870d2db14cb5a5',
 'frontend/numbers.py': 'ddd46ade4351d5852d7792e7b0a41248265f292f0392de52a237b48259632137',
 'frontend/pronunciation.py': '98f00a03fd66302c7f5f3cbdc6aba2a54fc76b63efb1aefa7b88f17c26497a4a',
 'frontend/segmentation.py': 'd94d38877734783911b10063215e0e8a80ece7729031c0dc2ac3b0e85dcfbd8d',
 'resources/frontend_profile.json': '09e7af5580972e671ec8740b12d11adb0f4daf1f27e88f697e9fb218bf8ac7ca'}

TERMS_SHA = 'be5f147a74286a2d74752c809c2f26564c9c27ef2ee78bf2f2c8c9db97d027b7'

PHONE_ABI = 'kokoro-v1.0-misaki-0.9.4-xiaoyou-v6'

VOCAB_SHA = '248ee3192acfc27a9907180354a0efaa63389047a6ff78c3f75e2e19466d0b76'

CONFIG_SHA = '5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f'

READY = True

GLOBAL_STEP = 26523

VOICE_ID = '01yuewen'

NATIVE_WEIGHTS_SHA = 'f33dcf83714f3c42cbdad5c85a6ec09d1e69bec29e6c173992bff429f72852f9'

PHASE_DSP_SHA = '82e542dc0d002ff5214779ae568b66407cd0f2770495bb7614fc58bc3804a4b4'

STYLE_FILE_SHA = '1c0c128370aa247bbf3eb77bf798c0f7d4d4583074f9d582e314c4a0570ca13c'

STYLE_RAW_SHA = '1aba56cb2955bc9940de8e3a9c68241acbfc79e126bc63b1386cd9953a05667f'

GRAPH_SHAS = {'acoustic': 'b60ba59980a0a9d8c0c67d531944158c7bed56361384d738a7db168b260100f2',
 'encoder': 'ffdbd9dacf30ad951466a51a7b4f364aa83f1f5a52aa5a6ebe7005d0d64b0702',
 'vocoder': 'adabfa70659495c2fc592e3cbe57851eeeb854c4332ce9b7bee29c16e6257995'}

PLAN_SHAS = {'acoustic': 'efe5b8747d41404f6262e648ed23f0da89fd6025e0019361a14837121c5b869e',
 'encoder': '15dc4462470f6990191437cb6f683582b726ba6ed5683090febe3530b899aef9',
 'vocoder': '152e650b0a0dce455845840815afcadabfac8d76da7bb2d4722a0a506806d91d'}

EXPECTED_PROFILES = {'acoustic': {'d': [[1, 640, 3], [1, 640, 128], [1, 640, 512]],
              'indices': [[3], [512], [832]],
              'style': [[1, 256], [1, 256], [1, 256]],
              't_en': [[1, 512, 3], [1, 512, 128], [1, 512, 512]]},
 'encoder': {'speed': [[1], [1], [1]],
             'style': [[1, 256], [1, 256], [1, 256]],
             'tokens': [[1, 3], [1, 128], [1, 512]]},
 'vocoder': {'excitation_noise': [[1, 1800, 9], [1, 307200, 9], [1, 438000, 9]],
             'f0': [[1, 6], [1, 1024], [1, 1460]],
             'features': [[1, 512, 6], [1, 512, 1024], [1, 512, 1460]],
             'source_phase': [[1, 1800, 9], [1, 307200, 9], [1, 438000, 9]],
             'style': [[1, 256], [1, 256], [1, 256]]}}

STREAM_POLICY = {'activation_allocation': {'default': 'fixed',
                           'release': 'after_confirmed_original_stream_sync',
                           'shape_mode': 'per_stage_lease_after_input_shape_validation',
                           'supported': ['fixed', 'shape']},
 'allow_frame_truncation': False,
 'allowed_first_output_policies': ['clause', 'sentence'],
 'default_first_output_policy': 'clause',
 'join_padding': {'kind': 'PCM16_exact_zero_only',
                  'max_leading_samples': 2000,
                  'max_trailing_samples': 2000,
                  'sample_rate': 16000,
                  'scope': 'selected_first_output_to_full_tail_only'},
 'max_T': 730,
 'mode': 'natural_text_streaming',
 'pcm_postprocessing': {'ambiguous_join_fallback': 'complete_reference',
                        'complete_reference_PCM_preserved': True,
                        'default_prefix_enabled': True,
                        'native_context_samples': 12023,
                        'prefix_samples': 8000,
                        'sample_rate': 16000},
 'preserve_full_text': True,
 'proactive_span_limit': 2,
 'speed': 1.0,
 'split_strategy': 'safe_first_punctuation_clause_then_full_tail_with_lossless_semantic_fallback',
 'supported_F_range': [6, 1460],
 'first_clause_selection': {'keep_complete_sentence_spoken_units_max': 24,
                            'minimum_prefix_spoken_units': 12,
                            'spoken_units': 'CJK ideographs plus twice ASCII word/number token '
                                            'count',
                            'punctuation': 'first safe comma, colon or enumeration boundary after two complete items inside the first complete sentence; retain original punctuation',
                            'short_or_ambiguous_fallback': 'complete_sentence'}}

CHAIN_EVIDENCE_SHA = '539feb59614ab9ad10a46fb4d582282ee0359a87b231679aa957917125952048'

OPERATING_F_RANGE = [6, 1460]

PHYSICAL_F_RANGE = [6, 1460]

TESTED_F_VALUES = [6, 1238, 1328, 1460]



def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def profile_sha256(profiles):
    payload = json.dumps(profiles, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_frontend_files(front, directory):
    """Bind model metadata, the fixed Git contract and real package files."""
    if not isinstance(front, dict) or (
            front.get('abi') != PHONE_ABI
            or front.get('source_sha256') != FRONTEND_SHA
            or front.get('terms_sha256') != TERMS_SHA
            or front.get('vocab_sha256') != VOCAB_SHA
            or front.get('normalization_revision') != NORMALIZATION_REVISION
            or front.get('normalization_modules') != NORMALIZATION_MODULE_SHAS):
        raise ValueError('Xiaoyou frontend metadata differs from the fixed source contract')
    root = Path(directory).resolve()
    files = {'frontend/phonemizer.py': FRONTEND_SHA,
             'resources/terms.json': TERMS_SHA, **NORMALIZATION_MODULE_SHAS}
    for name, expected in files.items():
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root) or sha(path) != expected):
            raise ValueError('Xiaoyou frontend source missing, linked or SHA-mismatched: ' + name)
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

REFERENCE_PHASE_DSP_SHA = 'a89ad2f2658563cbcd17cb608306d13494623834f691e530a326cbcf2e76d073'
REFERENCE_PHASE_PREFIX_BYTES = 3060
EXCITATION_EQUIVALENCE_SHA = '95b70515d9fed45b112e332e085322d8af0d3ae7fea64a6a38c3af084b11d446'
