"""Bit-exact PCM prefix scheduling for an already complete native waveform.

Callbacks must be the unchanged 24k->16k 97-tap helper, PCM16 converter and
exact-zero join trimmer. The full native waveform is already generated. This
only schedules existing postprocessing; it never splits text or calls a model.
The caller owns first-PCM/HTTP/DDS clocks. No TTFT key is written here.
"""
from __future__ import annotations

import time

import numpy as np

PREFIX_SAMPLES = 8000
NATIVE_PREFIX_SAMPLES = 12023
JOIN_ZERO_KEEP_SAMPLES = 2000


def safe_prefix(prefix, *, head=False, tail=False):
    """Return an already join-trimmed prefix, or None when proof needs the tail."""
    if head and tail:
        raise ValueError('One nonempty span cannot be on both sides of the join')
    if not isinstance(prefix, bytes) or len(prefix) != 2 * PREFIX_SAMPLES:
        return None
    nonzero = np.flatnonzero(np.frombuffer(prefix, dtype='<i2'))
    if tail and (not nonzero.size or nonzero[-1] < PREFIX_SAMPLES - JOIN_ZERO_KEEP_SAMPLES - 1):
        return None
    if head and not nonzero.size:
        return None
    removed = max(0, int(nonzero[0]) - JOIN_ZERO_KEEP_SAMPLES) if head else 0
    return prefix[2 * removed:]


def iter_exact_pcm(wave, *, resample, pcm16, trim_join, head=False, tail=False,
                   metrics=None, cancel=None, stages=None):
    """Emit a safe prefix, then the rest of the original complete trimmed PCM.

    Use a fresh per-render metrics mapping. source_peak is a real whole-wave
    maximum computed before the first yield. PCM peak/clipping remain explicitly
    incomplete until the full original resample finishes. Existing request TTFT
    keys, source text, PCM values, pacing and neural calls are untouched.
    """
    if head and tail:
        raise ValueError('One nonempty span cannot be on both sides of the join')
    wave = np.asarray(wave)
    if wave.ndim != 1 or wave.dtype != np.float32 or not wave.size:
        raise ValueError('A complete nonempty FP32 mono waveform is required')
    state = {} if metrics is None else metrics
    def timed(name, function, *args, total=None):
        if stages is None:
            return function(*args)
        started = time.perf_counter()
        try:
            return function(*args)
        finally:
            elapsed = time.perf_counter() - started
            stages[name] = stages.get(name, 0.0) + elapsed
            if total is not None:
                stages[total] = stages.get(total, 0.0) + elapsed

    source_peak = timed('PCM_source_peak_wall_seconds', lambda: float(np.max(np.abs(wave))))
    if not np.isfinite(source_peak):
        raise ValueError('A finite complete waveform is required')
    state.update(status='rendering', metrics_complete=False, native_samples=len(wave),
                 source_peak=source_peak, pcm_source_peak=None,
                 pcm_clipped_sample_count=None, pcm_samples=None,
                 prefix_pcm_samples=0, emitted_pcm_samples=0)
    emitted = b''
    complete = False

    def cancelled():
        return cancel is not None and cancel.is_set()

    try:
        if cancelled():
            state['status'] = 'cancelled'
            return
        if len(wave) >= NATIVE_PREFIX_SAMPLES:
            prefix_wave = timed('resample_prefix_wall_seconds', resample,
                                wave[:NATIVE_PREFIX_SAMPLES], total='resample_wall_seconds')
            admitted = timed('PCM_prefix_convert_join_wall_seconds',
                lambda: safe_prefix(pcm16(prefix_wave[:PREFIX_SAMPLES]), head=head, tail=tail),
                total='PCM_convert_join_wall_seconds')
            if admitted is not None:
                if cancelled():
                    state['status'] = 'cancelled'
                    return
                emitted = admitted
                state['prefix_pcm_samples'] = len(emitted) // 2
                state['emitted_pcm_samples'] = len(emitted) // 2
                yield emitted
        if cancelled():
            state['status'] = 'cancelled'
            return
        # Reuse the original whole-wave FIR including its global phase and EOF
        # padding. Recomputing its tiny prefix avoids changing the FIR engine.
        converted = timed('resample_full_wall_seconds', resample, wave, total='resample_wall_seconds')
        whole, head_cut, tail_cut = timed('PCM_whole_convert_join_wall_seconds',
            lambda: trim_join(pcm16(converted), head=head, tail=tail),
            total='PCM_convert_join_wall_seconds')
        if not whole.startswith(emitted):
            raise RuntimeError('Prefix differs from complete reference PCM')
        state.update(pcm_source_peak=float(np.max(np.abs(converted))),
                     pcm_clipped_sample_count=int(np.count_nonzero(
                         (converted * 32767 < -32768) | (converted * 32767 > 32767))),
                     pcm_samples=len(whole) // 2, pcm_samples_before_sentence_join_trim=len(converted),
                     head_removed_samples=head_cut,
                     tail_removed_samples=tail_cut, metrics_complete=True)
        rest = whole[len(emitted):]
        if cancelled():
            state['status'] = 'cancelled'
            return
        state['emitted_pcm_samples'] += len(rest) // 2
        complete = True
        state['status'] = 'complete'
        if rest:
            yield rest
    except GeneratorExit:
        state['status'] = 'cancelled'
        raise
    except BaseException:
        state['status'] = 'error'
        raise
    finally:
        if not complete and state['status'] == 'rendering':
            state['status'] = 'cancelled'
