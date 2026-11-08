"""Independent V6 TensorRT FP32 backend. No CPU neural fallback.

Streaming is across complete natural text clauses with bounded prefetch. A single
clause is generated before its PCM is emitted; measured TTFT is reported honestly.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import threading
import time

import numpy as np

from .assets import ensure_runtime_bundle, resolve_runtime_manifest_pin
from .gpu_engine import GPUEnginePool
from .phase_dsp import excitation_inputs
from .contract import (SOURCE_SHA, FRONTEND_SHA, TERMS_SHA, PHONE_ABI, VOCAB_SHA, CONFIG_SHA,
                       NORMALIZATION_REVISION, NORMALIZATION_MODULE_SHAS,
                       sha, profile_sha256, validate_frontend_files, frontend_acceptance_metadata)

STYLE_FILE_SHA = 'a157b56d6fb1242abea3d79d18c278def0066c1db2676805dd651e3b01771068'
STYLE_RAW_SHA = 'd95f2c67525e3ecd83b4f09f9915a13350cc4e0a89cac6caef326507e4fcd78b'
NATIVE_WEIGHTS_SHA = '6a8d13d7b656e44d4d39b051aff22c68a331b2d5c87f7c64551086c017564f8b'
GRAPH_SHAS = {
    'encoder': '8765b7e745defc1ea3871b04312e0a539c905d762f21925cf9961610aa2a1c9b',
    'acoustic': 'ba5ae6186f1365adf8fdac6ca46c7818eaf43dcef4a3a3e4cf7ee178fc602d5d',
    'source': '38d2834c8bf6fc9f0ea5dead632a035455c927d7f6f466648c31ef9d114a3151',
    'ups0': 'f83353c09945a496574b8b89c83a386e0ee2f2ec52bc32b1728e9382c95f42b7',
    'ups1': 'deadfd607356187d8ed8846d9f57f376397e5c5849e2870a697e6b8d8d0fb75b',
    'inverse': '9e5b3cbc26e6fa52422065f6c0c5be91f88e90d0d44528578cb8f0a2315f47b4',
}
SHORT_PLAN_SHAS = {
    'source': 'fb5052e213062fff4a21164ef19f4bd4af57f41b8eaba74f3623f49b5ab61eb2',
    'ups0': '29c9c053c9344c55be066ca6f6c39b0a594b27fe0203dd17624ae4e9e08a7035',
    'ups1': 'a7975f830f1b29b4d9cc53eb47acd2bdb85c9047ef87e34d95410e774987a2c0',
    'inverse': '682f868d5c37d4e83411642645fced57c13800574810f6c4ad095629db5db30f',
}
SHORT_CHAIN_EVIDENCE_SHA = 'e5d2c14a61cda94d5da9d7eda7e9546bf7f5f24c7236727cbfb1e8f2d72be5b5'
STREAM_POLICY = {'mode': 'natural_text_streaming', 'supported_F_range': [6, 382], 'max_T': 191,
                 'split_strategy': 'natural_clause_then_protected_word', 'speed': 1.0,
                 'preserve_full_text': True, 'allow_frame_truncation': False}


def protected_spans(text):
    """V6 segmentation guards only: no pronunciation replacement/legacy rules."""
    spans = []
    # These spans protect intact source dates/times only. They do not rewrite
    # text, select a reading, or import any pronunciation lexicon.
    digits = r'[0-9零〇○一二两三四五六七八九十百千]+'
    daypart = r'(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜间)'
    clock = (rf'(?:{daypart}[ \t]*)?(?:'
             rf'{digits}(?:点|时)(?:(?:{digits}(?:分)?|半|一刻|三刻))?(?:{digits}秒)?'
             r'|\d{1,2}:\d{2}(?::\d{2})?)')
    calendar = (rf'(?:{digits}年{digits}月(?:{digits}(?:日|号))?'
                rf'|{digits}月{digits}(?:日|号))')
    patterns = [
        rf'{calendar}(?:[ \t]*{clock})?',
        clock,
        r"[A-Za-z][A-Za-z0-9_]*(?:[-./:'’+][A-Za-z0-9_]+)*",
        r'(?<![A-Za-z0-9])(?:[+−-]?\d+(?:[,.]\d+)*)(?:年|月|日|点|分|秒|%|GB|MB|GHz|MHz)?',
        r'(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)',
        r'(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)',
        r'\b[A-Za-z][A-Za-z0-9_./-]*(?:[ \t]+(?:Runtime|Toolkit|XC|Driver|Operator))?[ \t]+(?:version[ \t]+|v)?\d+(?:\.\d+){0,3}\b',
        r'\b\d+(?:\.\d+)*[ \t]+(?:GB|MB|GHz|MHz|GPU|CPU|tokens|ms|milliseconds|seconds)\b',
    ]
    phrases = ['CUDA Runtime', 'CUDA Toolkit', 'ModelHub XC', 'RISE vGPU', 'GPU Operator',
               '第四范式', '小优', '下一步', 'SageAIOS', 'PhancyGroup', 'PhanthyMotus', '4Paradigm', 'xiaoyou01']
    for pattern in patterns:
        spans.extend((m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    for phrase in phrases:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE))
    return spans


def safe_boundary(text, end):
    return not any(start < end < stop for start, stop in protected_spans(text))



class _CancelledGeneration(RuntimeError):
    pass


class ProfileLimitError(ValueError):
    """Text is intact, but the verified model profile cannot contain this clause."""


def validate_options(language='auto', speed=1.0, speaker_id=0):
    if language not in ('auto', 'zh', 'en') or float(speed) != 1.0 or int(speaker_id) != 0:
        raise ValueError('V6 permits auto/zh/en, fixed07 speaker0 and speed1 only')


def natural_clauses(text):
    """Natural V6 text slices, protecting numbers/versions and basic initials."""
    boundaries = {m.end() for m in re.finditer(r'[。！？!?；;\n]+|(?<!\d)[，,：:]|[，,：:](?!\d)', text)}
    abbreviations = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs'}
    for found in re.finditer(r'\.(?=\s+[A-Z]|$)', text):
        before = text[:found.start()]
        word = re.search(r'([A-Za-z]+)$', before)
        # These are V6 sentence boundaries, not a pronunciation dictionary.
        if word and (word.group(1).lower() in abbreviations or len(word.group(1)) == 1):
            continue
        if re.search(r'(?:\b[A-Za-z]\.)+[A-Za-z]$', before):
            continue
        boundaries.add(found.end())
    pieces, start = [], 0
    for end in sorted(boundaries):
        if not safe_boundary(text, end):
            continue
        piece = text[start:end]
        if any(c.isalnum() for c in piece):
            pieces.append(piece)
            start = end
    tail = text[start:]
    if any(c.isalnum() for c in tail):
        pieces.append(tail)
    elif pieces:
        pieces[-1] += tail
    yield from pieces


def safe_split(text):
    """Split only at source punctuation/word gaps; no phoneme or audio truncation."""
    choices = [m.end() for m in re.finditer(r'[。！？!?；;，,：:]|(?<!\d)\.(?!\d)|\s+', text)
               if 0 < m.end() < len(text) and text[:m.end()].strip() and text[m.end():].strip()]
    # A comma/colon surrounded by digits is not a source word boundary.
    choices = [x for x in choices if not (x >= 2 and x < len(text)
                and text[x-1] in ',，:：' and text[x-2].isdigit() and text[x].isdigit())]
    choices = [x for x in choices if safe_boundary(text, x)]
    if not choices and re.search(r'[\u3400-\u9fff]', text):
        # V6-native text word boundaries, not a pronunciation override/legacy lexicon.
        # Never cut inside an ASCII identifier, date, version or numeric run.
        try:
            import jieba
            choices = [end for _, _, end in jieba.tokenize(text)
                       if 0 < end < len(text) and text[:end].strip() and text[end:].strip()
                       and not (text[end-1].isascii() and text[end].isascii()
                                and re.match(r"[A-Za-z0-9_./:'’+-]", text[end-1])
                                and re.match(r"[A-Za-z0-9_./:'’+-]", text[end]))]
        except ImportError:
            choices = []
        choices = [x for x in choices if safe_boundary(text, x)]
    if not choices:
        raise ProfileLimitError('V6 clause exceeds its verified profile without a safe text boundary; nothing was truncated')
    cut = min(choices, key=lambda x: abs(x - len(text) / 2))
    return text[:cut], text[cut:]


def _concrete(shape, symbols):
    try:
        result = tuple(symbols[x] if isinstance(x, str) else int(x) for x in shape)
    except KeyError as error:
        raise ValueError('Unknown V6 shape symbol: ' + str(error)) from error
    if any(x <= 0 for x in result):
        raise ValueError('V6 dynamic output is empty')
    return result


def verified_limits(stages, policy):
    """Only the measured F6..382 contract; long requests are text streamed."""
    if policy != STREAM_POLICY:
        raise ValueError('V6 requires the explicit measured natural_text_streaming policy')
    if set(stages) != set(GRAPH_SHAS):
        raise ValueError('V6 requires exactly encoder/acoustic/source/ups0/ups1/inverse')
    for name, stage in stages.items():
        if stage.get('source_graph_sha256') != GRAPH_SHAS[name]:
            raise ValueError('V6 source graph differs: ' + name)
        proof = stage.get('validation', {})
        if proof.get('profile_verified') is not True or proof.get('verified_profile_sha256') != profile_sha256(stage['profiles']):
            raise ValueError('V6 actual profiles differ from their GPU verification digest: ' + name)
        if name in SHORT_PLAN_SHAS:
            if proof.get('verified_F_range') != [6,382] or proof.get('long_profiles_verified') is not False:
                raise ValueError('V6 short-plan proof must retain F6..382 and long=false')
            if stage.get('sha256') != SHORT_PLAN_SHAS[name]:
                raise ValueError('V6 short plan differs from the measured four-plan chain: ' + name)
        if set(stage.get('io_dtypes', {})) != set(stage['inputs']) | set(stage['outputs']) or set(stage['profiles']) != set(stage['inputs']):
            raise ValueError('V6 typed I/O or profile declaration incomplete: ' + name)
        for key, dtype in stage['io_dtypes'].items():
            if key in ('tokens', 'duration'):
                if dtype not in ('int32','int64'):
                    raise ValueError('V6 token/duration binding must be integer')
            elif dtype != ('int32' if key == 'indices' else 'float32'):
                raise ValueError('V6 neural tensors must be FP32, indices INT32')
    if stages['encoder']['profiles']['tokens'][0] != [1,3] or stages['encoder']['profiles']['tokens'][2] != [1,512]:
        raise ValueError('V6 encoder must preserve the complete N3..512 phone ABI')
    for i, f in enumerate((6,6,382)):
        source, up0, up1, inverse = [stages[k]['profiles'] for k in ('source','ups0','ups1','inverse')]
        checks = [(source['f0'][i], [1,f]), (up0['features'][i], [1,512,f]),
                  (up1['/Div_1_output_0'][i], [1,256,10*f]),
                  (inverse['/conv_post/Conv_output_0'][i], [1,22,60*f+1])]
        checks += [(source[k][i], [1,300*f,9]) for k in ('source_phase','excitation_noise')]
        checks += [(d['/Concat_1_output_0'][i], [1,22,60*f+1]) for d in (up0,up1)]
        if any(actual != expected for actual,expected in checks):
            raise ValueError('V6 source/body/OLA profile relation differs from the measured F6..382')
    acoustic = stages['acoustic']['profiles']
    if acoustic['d'][2] != [1,640,512] or acoustic['t_en'][2] != [1,512,512] or acoustic['indices'][2][0] < 191:
        raise ValueError('V6 acoustic profile cannot supply the legal short segments')
    return {'max_tokens':510, 'min_F':6, 'max_F':382, 'max_T':191,
            'scope':'per complete text segment; request text may contain multiple segments'}


def validate_chain_evidence(validation, stages):
    evidence = validation.get('vocoder_chain_evidence', {})
    if (validation.get('vocoder_chain_waveform_passed') is not True
            or evidence.get('report_sha256') != SHORT_CHAIN_EVIDENCE_SHA
            or evidence.get('plan_sha256') != SHORT_PLAN_SHAS
            or evidence.get('supported_F_range') != [6,382]
            or evidence.get('snr_min_db') != 60.0 or evidence.get('max_abs') != 0.001
            or evidence.get('long_profiles_verified') is not False):
        raise ValueError('V6 short-chain waveform evidence/range/original criteria differ')
    if any(stages[name]['sha256'] != digest for name,digest in SHORT_PLAN_SHAS.items()):
        raise ValueError('V6 actual plans do not match the measured chain evidence')


class TRTBackend:
    dry_run_text = 'Hi.'

    def __init__(self, model_dir=None, *, cpu_threads=2, backend='tensorrt_fp32', expected_manifest_sha256=None):
        if backend != 'tensorrt_fp32' or int(cpu_threads) not in (1, 2):
            raise ValueError('V6 production backend requires TensorRT FP32 and one/two CPU threads')
        pin = resolve_runtime_manifest_pin(expected_manifest_sha256)
        self.root = ensure_runtime_bundle(model_dir, pin)
        manifest_path = self.root / 'runtime_manifest.json'
        if sha(manifest_path) != pin:
            raise ValueError('V6 runtime manifest SHA differs')
        m = json.loads(manifest_path.read_text())
        if (m.get('artifact_kind') != 'xiaoyou_v6_gpu_runtime' or m.get('schema_version') != 1
                or m.get('project_version') != 'xiaoyou_v6' or m.get('model_id') != 'xiaoyou01'
                or m.get('voice_id') != 'xiaoyou01' or m.get('backend') != 'tensorrt_fp32'
                or m.get('precision') != 'FP32' or m.get('sample_rate') != 24000
                or m.get('global_step') != 16500 or m.get('source_checkpoint_sha256') != SOURCE_SHA
                or m.get('native_weights_sha256') != NATIVE_WEIGHTS_SHA):
            raise ValueError('Wrong V6 GPU model/lineage/precision declaration')
        front = m.get('frontend', {})
        frontend_source_files = validate_frontend_files(front, Path(__file__).parent)
        frontend_acceptance = frontend_acceptance_metadata(m.get('validation', {}))
        if sha(self.root / 'config.json') != CONFIG_SHA or sha(self.root / 'style.npy') != STYLE_FILE_SHA:
            raise ValueError('V6 fixed configuration/style asset differs')
        stages = m['stages']
        self.limits = verified_limits(stages, m.get('inference_policy'))
        validate_chain_evidence(m.get('validation', {}), stages)
        for stage in stages.values():
            name = stage['file']
            if name not in m['files'] or m['files'][name]['sha256'] != stage['sha256'] or sha(self.root / name) != stage['sha256']:
                raise ValueError('V6 actual plan asset differs: ' + name)
        pack = m.get('platform', {})
        major = pack.get('jetpack_major')
        trt_prefix = pack.get('tensorrt_version_prefix')
        release = Path('/etc/nv_tegra_release').read_text() if Path('/etc/nv_tegra_release').is_file() else ''
        if platform.machine() != 'aarch64' or (major, trt_prefix) not in ((6, '10.4'), (5, '8.5')):
            raise ValueError('V6 requires a separately verified JP6/TRT10.4 or JP5/TRT8.5 engine pack')
        if not re.search(r'^#\s*R' + ('36' if major == 6 else '35') + r'\b', release):
            raise ValueError('V6 engine pack and actual JetPack/L4T differ')
        if major == 6 and (stages['encoder']['io_dtypes']['tokens'] != 'int64' or stages['encoder']['io_dtypes']['duration'] != 'int64'):
            raise ValueError('V6 JP6 encoder integer ABI differs')
        if os.environ.get('NVIDIA_TF32_OVERRIDE') is not None:
            raise ValueError('NVIDIA_TF32_OVERRIDE must be unset for this FP32 pack')
        from importlib.metadata import version
        if version('en-core-web-sm') != '3.8.0':
            raise RuntimeError('Provision the pinned spaCy model locally; no runtime download')
        # Misaki zh uses jieba's module-global tokenizer. Load its real
        # dictionary during service startup, before allocating any GPU engine.
        # This performs no dummy text, frontend call, audio or neural warmup.
        dictionary_started = time.perf_counter()
        import jieba
        jieba.initialize()
        if getattr(jieba.dt, 'initialized', False) is not True:
            raise RuntimeError('V6 Jieba dictionary initialization did not complete')
        self.frontend_dictionary_initialization_seconds = time.perf_counter() - dictionary_started
        # Import only after manifest/contract/actual component hashes have passed.
        from .v6_frontend import Frontend
        self.frontend = Frontend(config_path=self.root / 'config.json')
        if (self.frontend.metadata.get('frontend_sha256') != FRONTEND_SHA
                or self.frontend.metadata.get('terms_sha256') != TERMS_SHA
                or self.frontend.metadata.get('normalization_module_sha256') != NORMALIZATION_MODULE_SHAS):
            raise ValueError('V6 imported frontend component metadata differs from the fixed source contract')
        self.legal_ids = set(self.frontend.vocab.values()) | {0}
        style = np.load(self.root / 'style.npy', allow_pickle=False)
        if style.shape != (1, 256) or style.dtype != np.float32 or not np.isfinite(style).all() or hashlib.sha256(np.ascontiguousarray(style).tobytes()).hexdigest() != STYLE_RAW_SHA:
            raise ValueError('V6 style is not the exact fixed07 vector')
        self._frontend_lock = threading.Lock()
        self._call_lock = threading.RLock()
        self._warmup_lock = threading.Lock()
        self._producer_lock = threading.Lock()
        self._producers = {}
        self._closed = False
        self._warmed = False
        self.last_audio_metrics = {}
        self.pool = None
        self.style = None
        try:
            self.pool = GPUEnginePool(self.root, stages, expected_trt_prefix=trt_prefix)
            self.style = self.pool.upload(np.ascontiguousarray(style))
        except BaseException:
            if self.pool is not None:
                self.pool.close()
            raise
        self.stages = stages
        self.metadata = {'name': 'xiaoyou01', 'model_id': 'xiaoyou01', 'voice_id': 'xiaoyou01', 'backend': 'tensorrt_fp32', 'accelerated': True, 'device': 'cuda:0', 'precision': 'FP32',
            'cpu_neural_fallback': False, 'runtime_manifest_sha256': pin, 'source_checkpoint_sha256': SOURCE_SHA,
            'global_step': 16500, 'jetpack_major': major, 'tensorrt_version': self.pool.version,
            'limits': self.limits, 'inference_policy': m['inference_policy'],
            'long_profiles_verified': False, 'short_e2e_verified': frontend_acceptance['short_e2e_verified'],
            'normalization_revision': NORMALIZATION_REVISION,
            'normalization_module_sha256': dict(NORMALIZATION_MODULE_SHAS),
            'frontend_source_files': frontend_source_files,
            'frontend_acceptance_bound': frontend_acceptance['frontend_acceptance_bound'],
            'frontend_acceptance_report_sha256': frontend_acceptance['frontend_acceptance_report_sha256'],
            'frontend_pcm_parity_validated': frontend_acceptance['frontend_pcm_parity_validated'],
            'frontend': self.frontend.metadata,
            'frontend_dictionary_initialized': True,
            'frontend_dictionary_initialization_seconds': self.frontend_dictionary_initialization_seconds,
            'frontend_dictionary_initialization_scope': 'service startup before GPU allocation; dictionary only, no text/audio calls',
            'streaming': 'complete natural clauses with bounded prefetch',
            'ttft_validated': frontend_acceptance['ttft_validated'],
            'jetson_validated': frontend_acceptance['jetson_validated'],
            'darvin_validated': frontend_acceptance['darvin_validated'],
            'three_process_instances_validated': frontend_acceptance['three_process_instances_validated'],
            'logical_instance_sharing': 'one engine set per process; three processes/containers need independent measured acceptance'}

    validate_options = staticmethod(validate_options)

    def resolve_language(self, text, language):
        validate_options(language)
        return language if language != 'auto' else ('zh' if re.search(r'[\u3400-\u9fff]', text) or not re.search('[A-Za-z]', text) else 'en')

    def _outputs(self, stage, symbols):
        return {name: _concrete(shape, symbols) for name, shape in self.stages[stage]['outputs'].items()}

    def synthesize_wave(self, text, *, language='auto', speed=1.0, _cancel=None, _normalization_language=None):
        validate_options(language, speed)
        requested_language = language if _normalization_language is None else _normalization_language
        validate_options(requested_language, speed)
        if _cancel is not None and _cancel.is_set():
            raise _CancelledGeneration('Request cancelled before generation')
        language = self.resolve_language(text, language)
        with self._frontend_lock:
            phones, ids = self.frontend(text, language, normalization_language=requested_language)
        return self._synthesize_ids(ids, _cancel=_cancel)

    def synthesize_tokens(self, tokens, style=None, speed=1.0, *, _cancel=None):
        """Pure GPU harness API: only complete tokens/style/speed, no CPU NN oracle."""
        token_array = np.asarray(tokens)
        value = np.asarray(speed)
        if value.size != 1 or float(value.reshape(-1)[0]) != 1.0:
            raise ValueError('xiaoyou01 speed is fixed to1')
        if token_array.ndim != 2 or token_array.shape[0] != 1 or not 3 <= token_array.shape[1] <= 512 or token_array.dtype.kind not in 'iu' or token_array[0,0] != 0 or token_array[0,-1] != 0:
            raise ValueError('xiaoyou01 requires complete integer tokens[1,N] with both zero boundaries')
        if style is not None:
            supplied = np.asarray(style)
            if supplied.shape != (1,256) or supplied.dtype != np.float32 or hashlib.sha256(np.ascontiguousarray(supplied).tobytes()).hexdigest() != STYLE_RAW_SHA:
                raise ValueError('xiaoyou01 only accepts its fixed style vector')
        return self._synthesize_ids(token_array[0,1:-1].tolist(), _cancel=_cancel)

    def _synthesize_ids(self, ids, *, _cancel=None):
        if not ids or len(ids) > 510:
            raise ProfileLimitError('V6 complete clause exceeds510 phoneme IDs; no truncation')
        if not set(ids).issubset(self.legal_ids):
            raise ValueError('V6 frontend emitted a token outside the pinned ABI')
        with self._call_lock:
            if self._closed:
                raise RuntimeError('V6 GPU backend is closed')
            live = []
            def execute(stage, inputs, symbols):
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled between GPU stages')
                out = self.pool.infer(stage, inputs, self._outputs(stage, symbols))
                live.extend(out.values())
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled after GPU stage')
                return out
            def keep(out, names):
                for name, value in out.items():
                    if name not in names:
                        value.release()
                return {name: out[name] for name in names}
            try:
                n = len(ids) + 2
                token_type = np.dtype(self.stages['encoder']['io_dtypes']['tokens'])
                enc = execute('encoder', {'tokens': np.array([[0, *ids, 0]], dtype=token_type),
                    'style': self.style, 'speed': np.ones((1,), np.float32)}, {'N': n})
                durations = enc['duration'].numpy()
                enc['duration'].release()
                if durations.shape != (n,) or durations.dtype.kind not in 'iu' or (np.any(durations < 1) or np.any(durations > 50)):
                    raise RuntimeError('V6 GPU predicted invalid integer durations')
                t = int(durations.astype(np.int64).sum())
                if t > self.limits['max_T'] or t < 3:
                    raise ProfileLimitError('V6 predicted duration exceeds the verified GPU range; no waveform was truncated')
                f = 2 * t
                symbols = {'N': n, 'T': t, 'F': f, 'samples': 300*f, 'padded_samples': 300*f+20,
                           'H': 60*f+1, 'Q': 10*f, 'R': 60*f}
                indices = np.repeat(np.arange(n, dtype=np.int32), durations.astype(np.int64))
                ac = execute('acoustic', {'d': enc['d'], 't_en': enc['t_en'], 'indices': indices, 'style': self.style}, symbols)
                enc['d'].release(); enc['t_en'].release()
                f0 = ac['f0'].numpy()
                if f0.shape != (1, f) or f0.dtype != np.float32:
                    raise RuntimeError('V6 GPU F0 frame rate differs')
                phase, noise = excitation_inputs(f0, seed=42)
                source = keep(execute('source', {'f0': ac['f0'], 'source_phase': phase, 'excitation_noise': noise}, symbols), {'/Concat_1_output_0'})
                del phase, noise
                ac['f0'].release()
                har = source['/Concat_1_output_0']
                up0 = keep(execute('ups0', {'features': ac['features'], 'style': self.style, '/Concat_1_output_0': har}, symbols), {'/Div_1_output_0'})
                ac['features'].release()
                up1 = keep(execute('ups1', {'/Div_1_output_0': up0['/Div_1_output_0'], 'style': self.style, '/Concat_1_output_0': har}, symbols), {'/conv_post/Conv_output_0'})
                up0['/Div_1_output_0'].release(); har.release()
                inv = keep(execute('inverse', {'/conv_post/Conv_output_0': up1['/conv_post/Conv_output_0']}, symbols), {'audio'})
                up1['/conv_post/Conv_output_0'].release()
                wave = inv['audio'].numpy()
                if wave.shape != (600*t,) or wave.dtype != np.float32 or not np.isfinite(wave).all():
                    raise RuntimeError('V6 GPU waveform is not complete finite FP32 at24kHz')
                return wave, durations, ids
            finally:
                for value in live:
                    value.release()

    def _audio_clauses(self, text, language, cancel, *, normalization_language=None):
        for clause in natural_clauses(text):
            if cancel is not None and cancel.is_set():
                return
            yield from self._audio_clause(clause, language, cancel, normalization_language=normalization_language)

    def _audio_clause(self, text, language, cancel, *, normalization_language=None):
        if cancel is not None and cancel.is_set():
            return
        try:
            wave, _, _ = self.synthesize_wave(text, language=language, _cancel=cancel,
                                                _normalization_language=normalization_language)
        except _CancelledGeneration:
            return
        except ProfileLimitError:
            left, right = safe_split(text)
            yield from self._audio_clause(left, language, cancel, normalization_language=normalization_language)
            yield from self._audio_clause(right, language, cancel, normalization_language=normalization_language)
            return
        from utils.resample import resample_poly, float_to_pcm16
        converted = resample_poly(wave)
        self.last_audio_metrics = {'source_peak': float(np.max(np.abs(wave))),
            'pcm_source_peak': float(np.max(np.abs(converted))),
            'pcm_clipped_sample_count': int(np.count_nonzero((converted*32767 < -32768) | (converted*32767 > 32767))),
            'native_samples': len(wave), 'pcm_samples': len(converted), 'gpu_allocations': self.pool.stats(),
            'processing': 'natural complete clause; resample24k-to16k and standard saturated PCM16 only'}
        yield float_to_pcm16(converted)

    def iter_pcm(self, text, *, language='auto', speed=1.0, cancel=None):
        validate_options(language, speed)
        if not isinstance(text, str) or not text.strip():
            return
        if not any(c.isalnum() for c in text):
            raise ValueError('xiaoyou01 text contains no spoken content')
        requested_language = language
        # Keep the original G2P route, but normalize the COMPLETE request before
        # splitting: commas in tensors/operators/groups must not be cut first.
        # Unsupported expressions fail before any producer/PCM is started.
        language = self.resolve_language(text, language)
        with self._frontend_lock:
            text = self.frontend.normalize_text(text, requested_language)
        if not isinstance(text, str) or not text.strip():
            raise ValueError('V6 normalization produced no spoken text')
        pending = queue.Queue(maxsize=2)
        stop = threading.Event()
        started = time.perf_counter()
        def offer(item):
            while not stop.is_set() and not (cancel is not None and cancel.is_set()):
                try:
                    pending.put(item, timeout=.05)
                    return True
                except queue.Full:
                    pass
            return False
        class CombinedCancel:
            def is_set(inner):
                return stop.is_set() or self._closed or (cancel is not None and cancel.is_set())
        combined = CombinedCancel()
        def produce():
            try:
                for pcm in self._audio_clauses(text, language, combined, normalization_language=requested_language):
                    if not offer(('pcm', pcm)):
                        return
                offer(('end', None))
            except BaseException as error:
                offer(('error', error))
            finally:
                with self._producer_lock:
                    self._producers.pop(threading.current_thread(), None)
        producer = threading.Thread(target=produce, name='xiaoyou-v6-gpu-prefetch', daemon=True)
        with self._producer_lock:
            if self._closed:
                raise RuntimeError('V6 backend is closed')
            self._producers[producer] = stop
        producer.start()
        first = True
        try:
            while not combined.is_set():
                try:
                    kind, value = pending.get(timeout=.05)
                except queue.Empty:
                    if self._closed or not producer.is_alive():
                        return
                    continue
                if kind == 'error':
                    raise value
                if kind == 'end':
                    return
                if first:
                    self.last_audio_metrics['backend_first_pcm_seconds'] = time.perf_counter() - started
                    self.last_audio_metrics['latency_scope'] = 'backend first available PCM, excludes MCP/ROS subscriber and transport'
                    first = False
                yield value
        finally:
            stop.set()

    def synthesize(self, text, **options):
        return b''.join(self.iter_pcm(text, **options))

    def warmup(self):
        if self._closed:
            raise RuntimeError('V6 backend is closed')
        if self._warmed:
            return 0
        with self._warmup_lock:
            if self._warmed:
                return 0
            count = len(self.synthesize(self.dry_run_text, language='en'))
            if not count:
                raise RuntimeError('V6 GPU warmup produced no PCM')
            self._warmed = True
            return count

    def close(self):
        with self._producer_lock:
            for stop in self._producers.values():
                stop.set()
        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            if self.style is not None:
                self.style.release()
                self.style = None
            if self.pool is not None:
                self.pool.close()


_SHARED = {}
_SHARED_LOCK = threading.Lock()


class _Lease:
    def __init__(self, key, backend):
        self._key, self._backend, self._released = key, backend, False

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def close(self):
        if self._released:
            return
        with _SHARED_LOCK:
            if self._released:
                return
            self._released = True
            entry = _SHARED[self._key]
            entry['references'] -= 1
            if not entry['references']:
                del _SHARED[self._key]
                entry['backend'].close()


def acquire_gpu_backend(model_dir=None, *, cpu_threads=2, expected_manifest_sha256=None):
    root = str(Path(model_dir or os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/xiaoyou01').expanduser().resolve())
    pin = resolve_runtime_manifest_pin(expected_manifest_sha256)
    key = (root, pin, int(cpu_threads))
    with _SHARED_LOCK:
        entry = _SHARED.get(key)
        if entry is None:
            entry = {'backend': TRTBackend(model_dir, cpu_threads=cpu_threads, expected_manifest_sha256=expected_manifest_sha256), 'references': 0}
            _SHARED[key] = entry
        entry['references'] += 1
        return _Lease(key, entry['backend'])
