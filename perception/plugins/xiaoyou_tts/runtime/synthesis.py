"""Independent V6 TensorRT FP32 backend. No CPU neural fallback.

Emit a complete first sentence, then its intact tail, retaining full frontend
context. Only excess zero PCM at their real sentence join is shortened. Actual
token/frame overflow keeps the original lossless semantic fallback.
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

from .model_bundle import ensure_runtime_bundle, resolve_runtime_manifest_pin
from .cuda_engine import GPUEnginePool
from .excitation import ExcitationCache, excitation_inputs
from .release_contract import (SOURCE_SHA, FRONTEND_SHA, TERMS_SHA, PHONE_ABI, VOCAB_SHA, CONFIG_SHA,
                       NORMALIZATION_REVISION, NORMALIZATION_MODULE_SHAS,
                       sha, profile_sha256, validate_frontend_files, frontend_acceptance_metadata,
    READY, GLOBAL_STEP, VOICE_ID, STYLE_FILE_SHA, STYLE_RAW_SHA,
    NATIVE_WEIGHTS_SHA, PHASE_DSP_SHA, GRAPH_SHAS, PLAN_SHAS, EXPECTED_PROFILES, CHAIN_EVIDENCE_SHA, STREAM_POLICY, OPERATING_F_RANGE, PHYSICAL_F_RANGE, TESTED_F_VALUES,
    REFERENCE_PHASE_DSP_SHA, REFERENCE_PHASE_PREFIX_BYTES, EXCITATION_EQUIVALENCE_SHA)







# Each side of the one proactive complete-sentence join keeps up to125ms.
# No threshold/VAD, fade, sample scaling, nonzero trim or within-clause trim.
SENTENCE_JOIN_ZERO_KEEP_SAMPLES = 2000


def _trim_sentence_join_pcm(pcm, *, head=False, tail=False):
    if not isinstance(pcm, bytes) or len(pcm) % 2:
        raise ValueError('Sentence join requires aligned PCM16 bytes')
    if head and tail:
        raise ValueError('A nonempty clause cannot lie on both sides of one join')
    if not (head or tail) or not pcm:
        return pcm, 0, 0
    samples = np.frombuffer(pcm, dtype='<i2')
    nonzero = np.flatnonzero(samples)
    head_zeros = int(nonzero[0]) if nonzero.size else len(samples)
    tail_zeros = len(samples) - int(nonzero[-1]) - 1 if nonzero.size else len(samples)
    head_cut = max(0, head_zeros - SENTENCE_JOIN_ZERO_KEEP_SAMPLES) if head else 0
    tail_cut = max(0, tail_zeros - SENTENCE_JOIN_ZERO_KEEP_SAMPLES) if tail else 0
    return pcm[2*head_cut:len(pcm)-2*tail_cut], head_cut, tail_cut


def _join_producers(producers, timeout=30.0):
    """Drain only these owned producers; never wait while holding their locks."""
    deadline = time.monotonic() + timeout
    for producer in producers:
        if producer is threading.current_thread():
            raise RuntimeError('V6 producer cannot dispose its own active backend')
        producer.join(timeout=max(0.0, deadline-time.monotonic()))
        if producer.is_alive():
            raise RuntimeError('V6 producer did not stop within30s; backend resources retained')


class _CancelledGeneration(RuntimeError):
    pass


class ProfileLimitError(ValueError):
    """Text is intact, but the verified model profile cannot contain this clause."""


def validate_options(language='auto', speed=1.0, speaker_id=0):
    if language not in ('auto', 'zh', 'en') or float(speed) != 1.0 or int(speaker_id) != 0:
        raise ValueError('V6 permits auto/zh/en, fixed01yuewen speaker0 and speed1 only')




def safe_split(text):
    from ..frontend import segmentation
    try:
        left,right=segmentation.safe_split(text)
    except segmentation.ProfileLimitError as error:
        raise ProfileLimitError(str(error)) from error
    if left+right!=text:raise ValueError('Semantic split changed normalized text')
    return left,right


def _timing_trace(metrics):
    if not isinstance(metrics, dict):
        return None
    value = metrics.get('timing')
    return value if isinstance(value, dict) and value.get('enabled') is True else None


def _finish_timing_attempt(trace, attempt, status, error=None):
    if trace is None or attempt is None:
        return
    if 'end_seconds' not in attempt:
        attempt['end_seconds'] = time.perf_counter() - trace['origin_perf']
        attempt['elapsed_seconds'] = attempt['end_seconds'] - attempt['start_seconds']
        attempt['status'] = status
        if error is not None:
            attempt['exception_type'] = type(error).__name__
            if isinstance(error, ProfileLimitError):
                attempt['profile_reason'] = str(error)
    if trace.get('_active_attempt') is attempt:
        trace.pop('_active_attempt', None)


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
            or evidence.get('production_phase_dsp_sha256') != REFERENCE_PHASE_DSP_SHA
            or evidence.get('production_phase_recomputed_from_TRT_F0') is not True
            or evidence.get('snr_min_db') != 60.0 or evidence.get('max_abs') != 0.001
            or not TESTED_F_VALUES):
        raise ValueError('01yuewen three-plan reference waveform evidence is missing or differs')
    equivalence = validation.get('runtime_dsp_equivalence')
    expected = {'reference_phase_dsp_sha256': REFERENCE_PHASE_DSP_SHA,
                'reference_prefix_bytes': REFERENCE_PHASE_PREFIX_BYTES,
                'active_phase_dsp_sha256': PHASE_DSP_SHA,
                'CPU_equivalence_report_sha256': EXCITATION_EQUIVALENCE_SHA,
                'scope': 'CPU_bit_exact_cache_proof_not_GPU_or_latency_acceptance'}
    if not re.fullmatch('[0-9a-f]{64}', EXCITATION_EQUIVALENCE_SHA) or equivalence != expected:
        raise ValueError('Current phase cache lacks its exact reference/CPU equivalence binding')
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
        self.timing_enabled = os.environ.get('XIAOYOU01_TTS_TIMING') == '1'
        self.first_output_policy = os.environ.get('XIAOYOU01_FIRST_OUTPUT_POLICY', 'sentence')
        if self.first_output_policy != 'sentence':
            raise ValueError('Only the original sentence output policy is supported')
        prefix_mode = os.environ.get('XIAOYOU01_PCM_PREFIX', '1')
        if prefix_mode not in ('0', '1'):
            raise ValueError('PCM prefix setting must be 0 or 1')
        self.pcm_prefix_enabled = prefix_mode == '1'
        cache_mode = os.environ.get('XIAOYOU01_EXCITATION_CACHE', '1')
        if cache_mode not in ('0', '1'):
            raise ValueError('Excitation cache setting must be 0 or 1')
        gpu_noise_mode = os.environ.get('XIAOYOU01_GPU_NOISE_CACHE', '0')
        if gpu_noise_mode not in ('0', '1'):
            raise ValueError('GPU noise cache setting must be 0 or 1')
        if backend != 'tensorrt_fp32' or int(cpu_threads) not in (1, 2):
            raise ValueError('V6 production backend requires TensorRT FP32 and one/two CPU threads')
        if READY is not True:
            raise ValueError('01yuewen candidate is not target-validated; no GPU allocation or download')
        if sha(Path(__file__).with_name('excitation.py')) != PHASE_DSP_SHA:
            raise ValueError('01yuewen production phase DSP source differs from the release binding')
        reference_prefix = Path(__file__).with_name('excitation.py').read_bytes()[:REFERENCE_PHASE_PREFIX_BYTES]
        if hashlib.sha256(reference_prefix).hexdigest() != REFERENCE_PHASE_DSP_SHA:
            raise ValueError('The unchanged phase reference prefix differs')
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
                or m.get('runtime_code') != {'excitation.py':PHASE_DSP_SHA}):
            raise ValueError('Wrong V6 GPU model/lineage/precision declaration')
        front = m.get('frontend', {})
        frontend_source_files = validate_frontend_files(front, Path(__file__).resolve().parents[1])
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
        from ..frontend.phonemizer import Frontend
        self.frontend = Frontend(config_path=self.root / 'config.json')
        if (self.frontend.metadata.get('frontend_sha256') != FRONTEND_SHA
                or self.frontend.metadata.get('terms_sha256') != TERMS_SHA
                or self.frontend.metadata.get('normalization_module_sha256') != NORMALIZATION_MODULE_SHAS):
            raise ValueError('V6 imported frontend component metadata differs from the fixed source contract')
        self.legal_ids = set(self.frontend.vocab.values()) | {0}
        self.excitation_cache = ExcitationCache() if cache_mode == '1' else None
        if self.excitation_cache is not None and self.excitation_cache.persistent_array_bytes > 24 * 1024**2:
            raise RuntimeError('Exact excitation cache exceeds its bounded CPU storage')
        style = np.load(self.root / 'style.npy', allow_pickle=False)
        if style.shape != (1, 256) or style.dtype != np.float32 or not np.isfinite(style).all() or hashlib.sha256(np.ascontiguousarray(style).tobytes()).hexdigest() != STYLE_RAW_SHA:
            raise ValueError('V6 style is not the exact01yuewen vector')
        self._frontend_lock = threading.Lock()
        self._call_lock = threading.RLock()
        self._warmup_lock = threading.Lock()
        self._producer_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._producers = {}
        self._closed = False
        self._warmed = False
        self.last_audio_metrics = {}
        self.pool = None
        self.style = None
        self.gpu_noise = None
        try:
            self.pool = GPUEnginePool(self.root, stages, expected_trt_prefix=trt_prefix)
            self.style = self.pool.upload(np.ascontiguousarray(style))
            if gpu_noise_mode == '1' and self.excitation_cache is not None:
                noise = self.excitation_cache._noise
                if (noise.dtype != np.float32 or noise.shape != (1, ExcitationCache.MAX_FRAMES * 300, 9)
                        or not noise.flags.c_contiguous or noise.flags.writeable):
                    raise ValueError('GPU noise cache requires the immutable contiguous FP32 seed42 source')
                self.gpu_noise = self.pool.upload(noise)
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
            'streaming': 'complete first sentence then intact tail; join-only excess zero PCM shortened; bounded prefetch and lossless profile fallback',
            'first_output_policy': self.first_output_policy,
            'excitation_cache_enabled': self.excitation_cache is not None,
            'excitation_cache_bytes': self.excitation_cache.persistent_array_bytes if self.excitation_cache is not None else 0,
            'gpu_noise_cache_enabled': self.gpu_noise is not None,
            'gpu_noise_cache_bytes': self.gpu_noise.nbytes if self.gpu_noise is not None else 0,
            'reference_phase_dsp_sha256': REFERENCE_PHASE_DSP_SHA,
            'CPU_excitation_equivalence_report_sha256': EXCITATION_EQUIVALENCE_SHA,
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

    def synthesize_wave(self, text, *, language='auto', speed=1.0, _cancel=None, _normalization_language=None, _normalized_context=None, _request_metrics=None, auto_fallback=None):
        validate_options(language, speed)
        requested_language = language if _normalization_language is None else _normalization_language
        validate_options(requested_language, speed)
        if _cancel is not None and _cancel.is_set():
            raise _CancelledGeneration('Request cancelled before generation')
        trace = _timing_trace(_request_metrics) if getattr(self, 'timing_enabled', False) else None
        attempt = trace.get('_active_attempt') if trace is not None else None
        waiting = time.perf_counter() if attempt is not None else None
        with self._frontend_lock:
            entered = time.perf_counter() if attempt is not None else None
            if attempt is not None:
                attempt['frontend_lock_wait_seconds'] = entered - waiting
            try:
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled while waiting for frontend')
                if _normalized_context is None:
                    phones, ids = self.frontend(text, language, normalization_language=requested_language, auto_fallback=auto_fallback)
                else:
                    full,start,end=_normalized_context
                    if full[start:end]!=text:raise ValueError('Normalized semantic span differs from full request')
                    phones,ids=self.frontend.phonemize_normalized(full,language,start=start,end=end, normalization_language=requested_language, auto_fallback=auto_fallback)
            finally:
                if attempt is not None:
                    attempt['g2p_wall_seconds'] = time.perf_counter() - entered
        if attempt is not None:
            attempt['N'] = len(ids) + 2
        if _cancel is not None and _cancel.is_set():
            raise _CancelledGeneration('Request cancelled after frontend')
        method = self._synthesize_ids
        if trace is not None and getattr(method, '__func__', None) is TRTBackend._synthesize_ids:
            return method(ids, _cancel=_cancel, _request_metrics=_request_metrics)
        if attempt is not None:
            attempt['neural_detail_unavailable'] = 'overridden_synthesize_ids_keeps_original_signature'
        return method(ids, _cancel=_cancel)

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

    def _synthesize_ids(self, ids, *, _cancel=None, _request_metrics=None):
        if not ids or len(ids) > 510:
            raise ProfileLimitError('V6 complete clause exceeds510 phoneme IDs; no truncation')
        if not set(ids).issubset(self.legal_ids):
            raise ValueError('V6 frontend emitted a token outside the pinned ABI')
        trace = _timing_trace(_request_metrics) if getattr(self, 'timing_enabled', False) else None
        attempt = trace.get('_active_attempt') if trace is not None else None
        stages = attempt.setdefault('stages', {}) if attempt is not None else None
        waiting = time.perf_counter() if attempt is not None else None
        with self._call_lock:
            if attempt is not None:
                attempt['call_lock_wait_seconds'] = time.perf_counter() - waiting
            if self._closed:
                raise RuntimeError('V6 GPU backend is closed')
            live = []
            def execute(stage, inputs, symbols):
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled between GPU stages')
                stage_started = time.perf_counter() if stages is not None else None
                activation_timings = None
                try:
                    infer = self.pool.infer
                    if stages is not None and getattr(infer, '__func__', None) is GPUEnginePool.infer:
                        activation_timings = {}
                    expected = self._outputs(stage, symbols)
                    if activation_timings is None:
                        out = infer(stage, inputs, expected)
                    else:
                        out = infer(stage, inputs, expected, _activation_timings=activation_timings)
                finally:
                    if stages is not None:
                        stages[stage + '_infer_wall_seconds'] = time.perf_counter() - stage_started
                        if activation_timings is not None:
                            for name, value in activation_timings.items():
                                stages[stage + '_activation_' + name] = value
                live.extend(out.values())
                if _cancel is not None and _cancel.is_set():
                    raise _CancelledGeneration('Request cancelled after GPU stage')
                return out
            try:
                n = len(ids) + 2
                if attempt is not None:
                    attempt['N'] = n
                token_type = np.dtype(self.stages['encoder']['io_dtypes']['tokens'])
                enc = execute('encoder', {'tokens': np.array([[0, *ids, 0]], dtype=token_type),
                    'style': self.style, 'speed': np.ones((1,), np.float32)}, {'N': n})
                download_started = time.perf_counter() if stages is not None else None
                try:
                    durations = enc['duration'].numpy()
                finally:
                    if stages is not None:
                        stages['encoder_download_wall_seconds'] = time.perf_counter() - download_started
                        stages['encoder_with_download_wall_seconds'] = stages.get('encoder_infer_wall_seconds', 0.0) + stages['encoder_download_wall_seconds']
                enc['duration'].release()
                if durations.shape != (n,) or durations.dtype.kind not in 'iu' or (np.any(durations < 1) or np.any(durations > 50)):
                    raise RuntimeError('V6 GPU predicted invalid integer durations')
                t = int(durations.astype(np.int64).sum())
                if attempt is not None:
                    attempt.update(T=t, F=2*t)
                if t > self.limits['max_T'] or t < 3:
                    raise ProfileLimitError('V6 predicted duration exceeds the verified GPU range; no waveform was truncated')
                f = 2 * t
                symbols = {'N': n, 'T': t, 'F': f, 'samples': 300*f, 'padded_samples': 300*f+20,
                           'H': 60*f+1, 'Q': 10*f, 'R': 60*f}
                indices = np.repeat(np.arange(n, dtype=np.int32), durations.astype(np.int64))
                ac = execute('acoustic', {'d': enc['d'], 't_en': enc['t_en'], 'indices': indices, 'style': self.style}, symbols)
                enc['d'].release(); enc['t_en'].release()
                download_started = time.perf_counter() if stages is not None else None
                try:
                    f0 = ac['f0'].numpy()
                finally:
                    if stages is not None:
                        stages['acoustic_download_wall_seconds'] = time.perf_counter() - download_started
                        stages['acoustic_with_download_wall_seconds'] = stages.get('acoustic_infer_wall_seconds', 0.0) + stages['acoustic_download_wall_seconds']
                if f0.shape != (1, f) or f0.dtype != np.float32:
                    raise RuntimeError('V6 GPU F0 frame rate differs')
                phase_started = time.perf_counter() if stages is not None else None
                try:
                    cache = getattr(self, 'excitation_cache', None)
                    phase, noise = (cache.excitation_inputs(f0, seed=42) if cache is not None
                                    else excitation_inputs(f0, seed=42))
                finally:
                    if stages is not None:
                        stages['CPU_phase_noise_wall_seconds'] = time.perf_counter() - phase_started
                gpu_noise = getattr(self, 'gpu_noise', None)
                if gpu_noise is not None:
                    noise = gpu_noise.prefix_view(phase.shape)
                    live.append(noise)
                vocoder = execute('vocoder', {'features': ac['features'], 'style': self.style,
                    'f0': ac['f0'], 'source_phase': phase, 'excitation_noise': noise}, symbols)
                del phase, noise
                ac['f0'].release(); ac['features'].release()
                download_started = time.perf_counter() if stages is not None else None
                try:
                    wave = vocoder['audio'].numpy()
                finally:
                    if stages is not None:
                        stages['vocoder_download_wall_seconds'] = time.perf_counter() - download_started
                        stages['vocoder_with_download_wall_seconds'] = stages.get('vocoder_infer_wall_seconds', 0.0) + stages['vocoder_download_wall_seconds']
                if wave.shape != (600*t,) or wave.dtype != np.float32 or not np.isfinite(wave).all():
                    raise RuntimeError('V6 GPU waveform is not complete finite FP32 at24kHz')
                return wave, durations, ids
            finally:
                for value in live:
                    value.release()

    def _audio_clauses(self,text,language,cancel,*,normalization_language=None,_request_metrics=None,auto_fallback=None):
        from ..frontend.segmentation import first_complete_sentence_spans
        policy = getattr(self, 'first_output_policy', 'sentence')
        if policy != 'sentence':
            raise ValueError('Only the original sentence output policy is supported')
        spans = first_complete_sentence_spans(text)
        if not 1 <= len(spans) <= 2 or ''.join(text[a:b] for a,b in spans) != text:
            raise ValueError('Complete-sentence policy changed normalized text')
        join = spans[0][1] if len(spans) == 2 else None
        if _request_metrics is None:
            _request_metrics = {}
        trace = _timing_trace(_request_metrics) if getattr(self, 'timing_enabled', False) else None
        if trace is not None:
            trace['first_output_policy'] = policy
            trace['planned_spans'] = [list(span) for span in spans]
            trace['excitation_cache_enabled'] = getattr(self, 'excitation_cache', None) is not None
            trace['pcm_prefix_enabled'] = bool(getattr(self, 'pcm_prefix_enabled', False))
        for start,end in spans:
            if cancel is not None and cancel.is_set():
                return
            # Yield the first complete piece before starting the tail. Recursive
            # profile fallback receives the same absolute join and full context.
            yield from self._audio_clause(text[start:end],language,cancel,
                    normalization_language=normalization_language,auto_fallback=auto_fallback,
                    _normalized_context=(text,start,end),_sentence_join=join,_request_metrics=_request_metrics)

    def _audio_clause(self,text,language,cancel,*,normalization_language=None,_normalized_context=None,_sentence_join=None,_request_metrics=None,auto_fallback=None):
        trace = _timing_trace(_request_metrics) if getattr(self, 'timing_enabled', False) else None
        attempt = None
        if trace is not None:
            span = list(_normalized_context[1:]) if _normalized_context is not None else [0, len(text)]
            attempt = {'index': len(trace['attempts']), 'normalized_span': span,
                       'text_characters': len(text), 'start_seconds': time.perf_counter() - trace['origin_perf'],
                       'status': 'started', 'stages': {}}
            attempt['parent_index'] = next((row['index'] for row in reversed(trace['attempts'])
                if row.get('status') == 'profile_rejected'
                and row['normalized_span'][0] <= span[0] < span[1] <= row['normalized_span'][1]
                and row['normalized_span'] != span), None)
            trace['attempts'].append(attempt)
            trace['_active_attempt'] = attempt
        try:
            if cancel is not None and cancel.is_set():return
            if _request_metrics is None:
                _request_metrics = {}
            if _normalized_context is None:
                requested=language if normalization_language is None else normalization_language
                text=self.frontend.normalize_text(text,requested,auto_fallback=auto_fallback)
                _normalized_context=(text,0,len(text))
            try:
                method = self.synthesize_wave
                if trace is not None and getattr(method, '__func__', None) is TRTBackend.synthesize_wave:
                    wave,_,_=method(text,language=language,_cancel=cancel,
                            _normalization_language=normalization_language,_normalized_context=_normalized_context,
                            _request_metrics=_request_metrics,auto_fallback=auto_fallback)
                else:
                    if attempt is not None:
                        attempt['neural_detail_unavailable'] = 'overridden_synthesize_wave_keeps_original_signature'
                    wave,_,_=method(text,language=language,_cancel=cancel,
                            _normalization_language=normalization_language,_normalized_context=_normalized_context,
                            **({'auto_fallback':auto_fallback} if auto_fallback is not None else {}))
            except _CancelledGeneration as error:
                _finish_timing_attempt(trace, attempt, 'cancelled', error)
                return
            except ProfileLimitError as error:
                _finish_timing_attempt(trace, attempt, 'profile_rejected', error)
                if trace is not None:
                    trace['profile_fallbacks'] = trace.get('profile_fallbacks', 0) + 1
                left,right=safe_split(text);full,parent_start,parent_end=_normalized_context;cursor=0
                for piece in (left,right):
                    start=text.find(piece,cursor)
                    if start<0 or text[cursor:start].strip():raise ValueError('Profile fallback omitted normalized text')
                    end=start+len(piece);cursor=end
                    yield from self._audio_clause(piece,language,cancel,normalization_language=normalization_language,auto_fallback=auto_fallback,
                            _normalized_context=(full,parent_start+start,parent_start+end),_sentence_join=_sentence_join,_request_metrics=_request_metrics)
                if text[cursor:].strip():raise ValueError('Profile fallback omitted normalized tail')
                return
            from utils.resample import resample_poly, float_to_pcm16
            if getattr(self, 'pcm_prefix_enabled', False):
                from .pcm_rendering import iter_exact_pcm
                full, start, end = _normalized_context
                if _sentence_join is not None and not 0 < _sentence_join < len(full):
                    raise ValueError('Sentence join is not an interior absolute text boundary')
                render = {}
                removed_accounted = False

                def sync_render_metrics():
                    nonlocal removed_accounted
                    head_cut = render.get('head_removed_samples', 0)
                    tail_cut = render.get('tail_removed_samples', 0)
                    segment = {name: render.get(name) for name in (
                        'source_peak', 'pcm_source_peak', 'pcm_clipped_sample_count',
                        'native_samples', 'pcm_samples', 'pcm_samples_before_sentence_join_trim')}
                    segment.update(gpu_allocations=self.pool.stats(),
                        metrics_complete=render.get('metrics_complete', False),
                        pcm_prefix_samples=render.get('prefix_pcm_samples', 0),
                        sentence_join_zero_trim={'absolute_normalized_text_offset': _sentence_join,
                            'head_removed_samples': head_cut, 'tail_removed_samples': tail_cut,
                            'kept_zero_samples_per_side_max': SENTENCE_JOIN_ZERO_KEEP_SAMPLES},
                        processing='complete natural clause; exact FIR prefix then original whole tail; exact-zero join trim')
                    _request_metrics.update(segment)
                    _request_metrics['last_segment'] = segment
                    if render.get('metrics_complete') and not removed_accounted:
                        _request_metrics['total_generated_removed_zero_samples'] = (
                            _request_metrics.get('total_generated_removed_zero_samples', 0) + head_cut + tail_cut)
                        removed_accounted = True
                    if attempt is not None:
                        attempt.update(pcm_prefix_samples=render.get('prefix_pcm_samples', 0),
                                       PCM_render_complete=render.get('metrics_complete', False))

                renderer = iter_exact_pcm(wave, resample=resample_poly, pcm16=float_to_pcm16,
                    trim_join=_trim_sentence_join_pcm,
                    head=_sentence_join is not None and start == _sentence_join,
                    tail=_sentence_join is not None and end == _sentence_join,
                    metrics=render, cancel=cancel,
                    stages=attempt.setdefault('stages', {}) if attempt is not None else None)
                try:
                    for pcm in renderer:
                        sync_render_metrics()
                        _request_metrics['total_generated_pcm_samples'] = (
                            _request_metrics.get('total_generated_pcm_samples', 0) + len(pcm) // 2)
                        if render.get('status') == 'complete':
                            _finish_timing_attempt(trace, attempt, 'complete')
                        yield pcm
                finally:
                    renderer.close()
                    sync_render_metrics()
                if render.get('status') == 'complete':
                    _finish_timing_attempt(trace, attempt, 'complete')
                return
            resample_started = time.perf_counter() if attempt is not None else None
            try:
                converted = resample_poly(wave)
            finally:
                if attempt is not None:
                    attempt.setdefault('stages', {})['resample_wall_seconds'] = time.perf_counter() - resample_started
            pcm_started = time.perf_counter() if attempt is not None else None
            pcm = float_to_pcm16(converted)
            head_cut = tail_cut = 0
            if _sentence_join is not None:
                full,start,end = _normalized_context
                if not 0 < _sentence_join < len(full):
                    raise ValueError('Sentence join is not an interior absolute text boundary')
                pcm,head_cut,tail_cut = _trim_sentence_join_pcm(pcm,
                        head=start==_sentence_join,tail=end==_sentence_join)
            if attempt is not None:
                attempt['stages']['PCM_convert_join_wall_seconds'] = time.perf_counter() - pcm_started
            segment_metrics = {'source_peak': float(np.max(np.abs(wave))),
                'pcm_source_peak': float(np.max(np.abs(converted))),
                'pcm_clipped_sample_count': int(np.count_nonzero((converted*32767 < -32768) | (converted*32767 > 32767))),
                'native_samples': len(wave), 'pcm_samples': len(pcm)//2, 'gpu_allocations': self.pool.stats(),
                'pcm_samples_before_sentence_join_trim': len(converted),
                'sentence_join_zero_trim': {'absolute_normalized_text_offset':_sentence_join,
                    'head_removed_samples':head_cut,'tail_removed_samples':tail_cut,
                    'kept_zero_samples_per_side_max':SENTENCE_JOIN_ZERO_KEEP_SAMPLES},
                'processing': 'complete natural clause; original resample/saturated PCM16; exact-zero trim only at the first-sentence join'}
            _request_metrics.update(segment_metrics)
            _request_metrics['last_segment'] = segment_metrics
            _request_metrics['total_generated_pcm_samples'] = _request_metrics.get('total_generated_pcm_samples',0) + len(pcm)//2
            _request_metrics['total_generated_removed_zero_samples'] = _request_metrics.get('total_generated_removed_zero_samples',0) + head_cut + tail_cut
            _finish_timing_attempt(trace, attempt, 'complete')
            yield pcm
        except BaseException as error:
            status = 'cancelled' if isinstance(error, (GeneratorExit, _CancelledGeneration)) else 'error'
            _finish_timing_attempt(trace, attempt, status, error)
            raise
        finally:
            status = 'cancelled' if cancel is not None and cancel.is_set() else 'complete'
            _finish_timing_attempt(trace, attempt, status)

    def iter_pcm(self, text, *, language='auto', speed=1.0, cancel=None, _request_metrics=None, auto_fallback=None):
        timed = getattr(self, 'timing_enabled', False)
        request_metrics = _request_metrics if isinstance(_request_metrics, dict) else {}
        trace = None
        if timed:
            entered = time.perf_counter()
            trace = request_metrics.setdefault('timing', {'enabled': True, 'schema': 'xiaoyou_stage_wall_timing_v1',
                'origin_perf': entered, 'session': {}, 'attempts': [], 'producer_drained': True})
            trace['backend_enter_seconds'] = entered - trace['origin_perf']
            trace['stage_scope'] = 'CPU wall including scheduling, existing H2D/D2H and natural CUDA synchronization; not GPU kernel time'
            trace['backend_status'] = 'entered'
            self.last_audio_metrics = request_metrics
        validate_options(language, speed)
        if not isinstance(text, str) or not text.strip():
            return
        if not any(c.isalnum() for c in text):
            raise ValueError('xiaoyou01 text contains no spoken content')
        requested_language = language
        # Keep the original G2P route, but normalize the COMPLETE request before
        # splitting: commas in tensors/operators/groups must not be cut first.
        # Unsupported expressions fail before any producer/PCM is started.
        waiting = time.perf_counter() if trace is not None else None
        with self._frontend_lock:
            entered = time.perf_counter() if trace is not None else None
            if trace is not None:
                trace['normalization_lock_wait_seconds'] = entered - waiting
            try:
                text = self.frontend.normalize_text(text, requested_language,auto_fallback=auto_fallback)
            finally:
                if trace is not None:
                    trace['normalization_wall_seconds'] = time.perf_counter() - entered
        if not isinstance(text, str) or not text.strip():
            raise ValueError('V6 normalization produced no spoken text')
        request_metrics.update(total_emitted_pcm_samples=0, total_generated_pcm_samples=0,
                               total_generated_removed_zero_samples=0)
        self.last_audio_metrics = request_metrics
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
                for pcm in self._audio_clauses(text, language, combined, normalization_language=requested_language,_request_metrics=request_metrics,auto_fallback=auto_fallback):
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
            try:
                if trace is not None:
                    trace['producer_drained'] = False
                producer.start()
            except BaseException:
                if trace is not None:
                    trace['producer_drained'] = not producer.is_alive()
                self._producers.pop(producer, None)
                raise
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
                    if trace is not None:
                        observed = time.perf_counter()
                        trace.setdefault('first_pcm_seconds', observed - trace['origin_perf'])
                        trace.setdefault('backend_enter_to_first_pcm_seconds', observed - trace['origin_perf'] - trace['backend_enter_seconds'])
                    request_metrics['backend_first_pcm_seconds'] = time.perf_counter() - started
                    request_metrics['latency_scope'] = 'backend first available PCM, excludes MCP/ROS subscriber and transport'
                    first = False
                request_metrics['total_emitted_pcm_samples'] += len(value)//2
                yield value
        except BaseException as error:
            if trace is not None:
                trace['backend_status'] = 'cancelled' if isinstance(error, GeneratorExit) or combined.is_set() else 'error'
                trace['backend_exception_type'] = type(error).__name__
            raise
        finally:
            stop.set()
            try:
                _join_producers([producer])
            finally:
                if trace is not None:
                    trace['producer_drained'] = not producer.is_alive()
                    if trace.get('backend_status') == 'entered':
                        trace['backend_status'] = 'cancelled' if cancel is not None and cancel.is_set() else 'complete'
                    trace['backend_finished_seconds'] = time.perf_counter() - trace['origin_perf']

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
            # Startup-only coverage; consume and discard every complete output.
            # These independent sentences do not change production segmentation.
            count = 0
            for text, language in (
                (self.dry_run_text, 'en'),
                ('今天天气晴朗，欢迎来到我们的工作空间，大家可以先坐下来稍作休息，喝杯热茶，再一起交流新的想法。', 'zh'),
                ('Welcome to our workspace, where the team shares useful ideas and helps visitors explore new projects in a friendly atmosphere.', 'en'),
                ('欢迎来到小优的工作空间，今天一起体验语音服务，Please take a seat.', 'auto'),
            ):
                case_bytes = len(self.synthesize(text, language=language))
                if not case_bytes:
                    raise RuntimeError('V6 GPU warmup produced no PCM')
                count += case_bytes
            self._warmed = True
            return count

    def close(self):
        # Closed blocks new registrations; it does not mean resources are gone.
        # A failed drain/close remains reachable and can be retried safely.
        with self._close_lock:
            with self._producer_lock:
                self._closed = True
                producers = list(self._producers)
                for stop in self._producers.values():
                    stop.set()
            _join_producers(producers)
            with self._call_lock:
                if self.style is not None:
                    self.style.release()
                    self.style = None
                if getattr(self, 'gpu_noise', None) is not None:
                    self.gpu_noise.release()
                    self.gpu_noise = None
                if self.pool is not None:
                    self.pool.close()
                    self.pool = None
                self.excitation_cache = None


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
            entry = _SHARED[self._key]
            if entry['references'] == 1:
                # Do not commit the last release before the backend has drained.
                entry['backend'].close()
                del _SHARED[self._key]
            entry['references'] -= 1
            self._released = True


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
