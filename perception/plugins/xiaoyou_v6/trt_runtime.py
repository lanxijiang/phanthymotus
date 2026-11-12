"""Independent V6 TensorRT FP32 backend. No CPU neural fallback.

Legal utterances are attempted whole; only a token/frame overflow triggers
lossless semantic fallback. Each segment finishes before its PCM is emitted.
Named numerical checks do not imply exhaustive shape or latency acceptance.
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

from .runtime_bindings import (READY, GLOBAL_STEP, VOICE_ID, STYLE_FILE_SHA, STYLE_RAW_SHA,
    NATIVE_WEIGHTS_SHA, PHASE_DSP_SHA, GRAPH_SHAS, PLAN_SHAS, EXPECTED_PROFILES, CHAIN_EVIDENCE_SHA, STREAM_POLICY, OPERATING_F_RANGE, PHYSICAL_F_RANGE, TESTED_F_VALUES)


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
        raise ValueError('V6 permits auto/zh/en, fixed01yuewen speaker0 and speed1 only')


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
    from . import text_segmentation as segmentation
    try:
        left,right=segmentation.safe_split(text)
    except segmentation.ProfileLimitError as error:
        raise ProfileLimitError(str(error)) from error
    if left+right!=text:raise ValueError('Semantic split changed normalized text')
    return left,right


def _concrete(shape, symbols):
    try:
        result = tuple(symbols[x] if isinstance(x, str) else int(x) for x in shape)
    except KeyError as error:
        raise ValueError('Unknown V6 shape symbol: ' + str(error)) from error
    if any(x <= 0 for x in result):
        raise ValueError('V6 dynamic output is empty')
    return result


def verified_limits(stages, policy):
    """Bind typed profiles and named waveform cases; no exhaustive-shape claim."""
    if READY is not True or policy != STREAM_POLICY or set(stages) != {'encoder','acoustic','vocoder'} or set(stages) != set(GRAPH_SHAS):
        raise ValueError('01yuewen requires completed target validation and exact profile policy')
    for name,stage in stages.items():
        if (stage.get('source_graph_sha256') != GRAPH_SHAS[name]
                or stage.get('sha256') != PLAN_SHAS[name]
                or stage.get('profiles') != EXPECTED_PROFILES[name]):
            raise ValueError('01yuewen actual graph/plan/profile binding differs: '+name)
        proof=stage.get('validation',{})
        if (proof.get('profile_verified') is not True
                or proof.get('verified_profile_sha256') != profile_sha256(stage['profiles'])
                or proof.get('actual_gpu_route_report_sha256') != CHAIN_EVIDENCE_SHA
                or proof.get('tested_F_values') != TESTED_F_VALUES):
            raise ValueError('01yuewen lacks actual new-plan target profile evidence: '+name)
        if name == 'vocoder' and proof.get('verified_operating_F_range') != OPERATING_F_RANGE:
            raise ValueError('01yuewen target F range differs: '+name)
        if (set(stage.get('io_dtypes',{})) != set(stage['inputs'])|set(stage['outputs'])
                or set(stage['profiles']) != set(stage['inputs'])):
            raise ValueError('01yuewen typed I/O/profile declaration incomplete: '+name)
        for key,dtype in stage['io_dtypes'].items():
            expected='int32' if key in ('tokens','indices') else 'int64' if key=='duration' else 'float32'
            if dtype != expected:raise ValueError('01yuewen actual integer/FP32 ABI differs: '+key)
    return {'max_tokens':510,'min_F':OPERATING_F_RANGE[0],'max_F':OPERATING_F_RANGE[1],
            'max_T':OPERATING_F_RANGE[1]//2,'physical_max_F':PHYSICAL_F_RANGE[1],'tested_F_values':list(TESTED_F_VALUES),
            'all_intermediate_shapes_tested':False,
            'scope':'per complete utterance; only actual token/frame overflow triggers semantic fallback'}


def validate_chain_evidence(validation, stages):
    evidence=validation.get('vocoder_chain_evidence',{})
    if (READY is not True or not CHAIN_EVIDENCE_SHA
            or validation.get('vocoder_chain_waveform_passed') is not True
            or validation.get('runtime_numeric_ready') is not True
            or evidence.get('report_sha256') != CHAIN_EVIDENCE_SHA
            or evidence.get('reference_kind') != 'same_actual_TRT_features_F0_native_decoder'
            or evidence.get('production_vs_original_gold_waveform_passed') is not False
            or evidence.get('component_strict_passed') is not False
            or evidence.get('plan_sha256') != PLAN_SHAS
            or evidence.get('physical_F_range') != PHYSICAL_F_RANGE
            or evidence.get('operating_F_range') != OPERATING_F_RANGE
            or evidence.get('tested_F_values') != TESTED_F_VALUES
            or evidence.get('production_phase_dsp_sha256') != PHASE_DSP_SHA
            or evidence.get('production_phase_recomputed_from_TRT_F0') is not True
            or evidence.get('snr_min_db') != 60.0 or evidence.get('max_abs') != 0.001
            or not TESTED_F_VALUES):
        raise ValueError('01yuewen new six-plan target waveform evidence is missing or differs')
    if any(stages[name]['sha256']!=digest for name,digest in PLAN_SHAS.items()):
        raise ValueError('01yuewen actual new plans differ from target evidence')


def validate_encoder_dtypes(stages, major):
    # Exact26523 v5 graph ABI: do not relabel ONNX/TensorRT integer outputs.
    actual=stages['encoder']['io_dtypes']
    if major==6 and (actual['tokens']!='int32' or actual['duration']!='int64'):
        raise ValueError('01yuewen JP6 requires actual INT32 tokens and INT64 duration')


class TRTBackend:
    dry_run_text = 'Hi.'

    def __init__(self, model_dir=None, *, cpu_threads=2, backend='tensorrt_fp32', expected_manifest_sha256=None):
        if backend != 'tensorrt_fp32' or int(cpu_threads) not in (1, 2):
            raise ValueError('V6 production backend requires TensorRT FP32 and one/two CPU threads')
        if READY is not True:
            raise ValueError('01yuewen candidate is not target-validated; no GPU allocation or download')
        if sha(Path(__file__).with_name('phase_dsp.py')) != PHASE_DSP_SHA:
            raise ValueError('01yuewen production phase DSP source differs from the release binding')
        pin = resolve_runtime_manifest_pin(expected_manifest_sha256)
        self.root = ensure_runtime_bundle(model_dir, pin)
        manifest_path = self.root / 'runtime_manifest.json'
        if sha(manifest_path) != pin:
            raise ValueError('V6 runtime manifest SHA differs')
        m = json.loads(manifest_path.read_text())
        if (m.get('ready') is not True or m.get('artifact_kind') != 'xiaoyou_v6_gpu_runtime' or m.get('schema_version') != 1
                or m.get('project_version') != 'xiaoyou_v6' or m.get('model_id') != 'xiaoyou01'
                or m.get('voice_id') != VOICE_ID or m.get('backend') != 'tensorrt_fp32'
                or m.get('precision') != 'FP32' or m.get('sample_rate') != 24000
                or m.get('global_step') != GLOBAL_STEP or m.get('source_checkpoint_sha256') != SOURCE_SHA
                or m.get('native_weights_sha256') != NATIVE_WEIGHTS_SHA
                or m.get('runtime_code') != {'phase_dsp.py':PHASE_DSP_SHA}):
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
        if platform.machine() != 'aarch64' or (major, trt_prefix) != (6, '10.4'):
            raise ValueError('V6 requires the separately verified JP6/TRT10.4 new26523 engine pack')
        if not re.search(r'^#\s*R' + ('36' if major == 6 else '35') + r'\b', release):
            raise ValueError('V6 engine pack and actual JetPack/L4T differ')
        validate_encoder_dtypes(stages, major)
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
            raise ValueError('V6 style is not the exact01yuewen vector')
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
        self.metadata = {'name': 'xiaoyou01', 'model_id': 'xiaoyou01', 'voice_id': VOICE_ID, 'backend': 'tensorrt_fp32', 'accelerated': True, 'device': 'cuda:0', 'precision': 'FP32',
            'cpu_neural_fallback': False, 'runtime_manifest_sha256': pin, 'source_checkpoint_sha256': SOURCE_SHA,
            'global_step': GLOBAL_STEP, 'production_phase_dsp_sha256':PHASE_DSP_SHA, 'jetpack_major': major, 'tensorrt_version': self.pool.version,
            'limits': self.limits, 'inference_policy': m['inference_policy'],
            'long_profiles_verified': m.get('validation',{}).get('vocoder_chain_evidence',{}).get('long_profiles_verified') is True,
            'runtime_numeric_ready':m.get('validation',{}).get('runtime_numeric_ready') is True,
            'numeric_reference_kind':m.get('validation',{}).get('numeric_reference_kind'),
            'production_vs_original_gold_waveform_passed':m.get('validation',{}).get('production_vs_original_gold_waveform_passed'),
            'component_original_strict_passed':m.get('validation',{}).get('all_component_original_thresholds_passed'),
            'operating_F_range':list(OPERATING_F_RANGE), 'physical_F_range':list(PHYSICAL_F_RANGE),
            'numeric_tested_F_values':list(TESTED_F_VALUES), 'all_intermediate_shapes_tested':False, 'short_e2e_verified': frontend_acceptance['short_e2e_verified'],
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
            'streaming': 'whole legal utterance; lossless semantic fallback with bounded prefetch',
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

    def synthesize_wave(self, text, *, language='auto', speed=1.0, _cancel=None, _normalization_language=None, _normalized_context=None):
        validate_options(language, speed)
        requested_language = language if _normalization_language is None else _normalization_language
        validate_options(requested_language, speed)
        if _cancel is not None and _cancel.is_set():
            raise _CancelledGeneration('Request cancelled before generation')
        with self._frontend_lock:
            if _normalized_context is None:
                phones, ids = self.frontend(text, language, normalization_language=requested_language)
            else:
                full,start,end=_normalized_context
                if full[start:end]!=text:raise ValueError('Normalized semantic span differs from full request')
                phones,ids=self.frontend.phonemize_normalized(full,language,start=start,end=end)
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
                vocoder = execute('vocoder', {'features': ac['features'], 'style': self.style,
                    'f0': ac['f0'], 'source_phase': phase, 'excitation_noise': noise}, symbols)
                del phase, noise
                ac['f0'].release(); ac['features'].release()
                wave = vocoder['audio'].numpy()
                if wave.shape != (600*t,) or wave.dtype != np.float32 or not np.isfinite(wave).all():
                    raise RuntimeError('V6 GPU waveform is not complete finite FP32 at24kHz')
                return wave, durations, ids
            finally:
                for value in live:
                    value.release()

    def _audio_clauses(self,text,language,cancel,*,normalization_language=None):
        # Legal requests run as one complete utterance. Split only on actual
        # token/duration profile rejection, retaining full normalized context.
        yield from self._audio_clause(text,language,cancel,normalization_language=normalization_language,
                                      _normalized_context=(text,0,len(text)))

    def _audio_clause(self,text,language,cancel,*,normalization_language=None,_normalized_context=None):
        if cancel is not None and cancel.is_set():return
        if _normalized_context is None:
            requested=language if normalization_language is None else normalization_language
            text=self.frontend.normalize_text(text,requested)
            _normalized_context=(text,0,len(text))
        try:
            wave,_,_=self.synthesize_wave(text,language=language,_cancel=cancel,
                    _normalization_language=normalization_language,_normalized_context=_normalized_context)
        except _CancelledGeneration:
            return
        except ProfileLimitError:
            left,right=safe_split(text);full,parent_start,parent_end=_normalized_context;cursor=0
            for piece in (left,right):
                start=text.find(piece,cursor)
                if start<0 or text[cursor:start].strip():raise ValueError('Profile fallback omitted normalized text')
                end=start+len(piece);cursor=end
                yield from self._audio_clause(piece,language,cancel,normalization_language=normalization_language,
                        _normalized_context=(full,parent_start+start,parent_start+end))
            if text[cursor:].strip():raise ValueError('Profile fallback omitted normalized tail')
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
    root = str(Path(model_dir or os.environ.get('XIAOYOU_V6_MODEL_DIR') or '/models/01yuewen').expanduser().resolve())
    pin = resolve_runtime_manifest_pin(expected_manifest_sha256)
    key = (root, pin, int(cpu_threads))
    with _SHARED_LOCK:
        entry = _SHARED.get(key)
        if entry is None:
            entry = {'backend': TRTBackend(model_dir, cpu_threads=cpu_threads, expected_manifest_sha256=expected_manifest_sha256), 'references': 0}
            _SHARED[key] = entry
        entry['references'] += 1
        return _Lease(key, entry['backend'])
