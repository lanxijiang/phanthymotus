"""ROS-independent ordered PCM delivery with one EOF per started request."""
from dataclasses import dataclass, field
import queue
import json
import logging
import hashlib
import threading
import time

SAMPLE_RATE = 16000
FRAME_BYTES = 3200
# Current official main transport: one initial500ms cushion, then real time.
PREBUF_FRAMES = 5
FRAME_INTERVAL_S = .1
EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'
log = logging.getLogger(__name__)


_TIMING_LOG_SCHEMA = 'xiaoyou_timing_records_v2'


def _timing_records(request):
    """Preflight every bounded JSON record before emitting any success record.

    main.py wraps stdout/stderr with logsafe. Its cap includes logging's prefix;
    reserve512 bytes for that prefix and leave the public logging policy intact.
    A count plus ordered canonical digest prevents missing/interleaved records
    from being mistaken for a complete trace by the diagnostic controller.
    """
    from utils.logsafe import MAX_LINE_BYTES
    limit = min(3000, MAX_LINE_BYTES - 512)
    if limit < 256:
        raise ValueError('Log framing budget is too small for timing records')
    encode = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False,
                                      separators=(',', ':'), sort_keys=True)
    trace = request.timing['timing']
    attempts = trace.get('attempts', [])
    if not isinstance(attempts, list):
        raise ValueError('Timing attempts must be a complete ordered list')
    summary_trace = {k: v for k, v in trace.items() if k not in ('attempts', '_active_attempt')}
    metrics = {'timing': summary_trace}
    for key in ('total_emitted_pcm_samples', 'total_generated_pcm_samples',
                'total_generated_removed_zero_samples', 'backend_first_pcm_seconds',
                'latency_scope', 'gpu_allocations'):
        if key in request.timing:
            metrics[key] = request.timing[key]
    records = [{'event': 'xiaoyou_request_timing', 'schema': _TIMING_LOG_SCHEMA,
                'record_kind': 'summary', 'metrics': metrics,
                'attempt_count': len(attempts),
                'attempts_sha256': hashlib.sha256(encode(attempts).encode()).hexdigest()}]
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or attempt.get('index') != index:
            raise ValueError('Timing attempt indices must be contiguous and ordered')
        records.append({'event': 'xiaoyou_request_attempt_timing', 'schema': _TIMING_LOG_SCHEMA,
                        'record_kind': 'attempt', 'trace_id': trace.get('trace_id'),
                        'action_id': trace.get('action_id'),
                        'session_request_index': trace.get('session_request_index'),
                        'attempt': attempt})
    result = []
    for record in records:
        # Event first preserves a simple, unambiguous JSON start in prefixed logs.
        ordered = {'event': record['event'], **{k: v for k, v in record.items() if k != 'event'}}
        payload = json.dumps(ordered, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        size = len(payload.encode())
        if size > limit:
            raise ValueError('Timing JSON record exceeds bound: bytes=' + str(size) + ', limit=' + str(limit))
        result.append(payload)
    return result


def _write_request_timing(request):
    try:
        records = _timing_records(request)
    except Exception as error:
        # Diagnostic failure must not change PCM, EOF/ACP or worker lifecycle.
        # No success summary is emitted if even one full attempt cannot fit.
        identity = json.dumps([request.action_id, request.trace_id], ensure_ascii=True,
                              separators=(',', ':')).encode()
        failure = {'event': 'xiaoyou_request_timing_failed', 'schema': _TIMING_LOG_SCHEMA,
                   'complete': False, 'request_identity_sha256': hashlib.sha256(identity).hexdigest(),
                   'error_type': type(error).__name__, 'error': str(error)[:160]}
        log.error('%s', json.dumps(failure, ensure_ascii=True, separators=(',', ':')))
        return
    for payload in records:
        log.info('%s', payload)


@dataclass
class Request:
    text: str
    action_id: str
    language: str = 'auto'
    trace_id: str = ''
    cancel: threading.Event = field(default_factory=threading.Event)
    delivery_lock: threading.RLock = field(default_factory=threading.RLock)
    eof_sent: bool = False
    completed: bool = False
    bytes_sent: int = 0
    timing: dict = field(default=None, repr=False)
    auto_fallback: str = field(default=None, repr=False)


class SpeechSession:
    """Serialize whole calls; interruption cannot publish PCM after its EOF.

    Backend generation may be in flight; cancellation stops delivery immediately.
    It cannot preempt an already executing neural kernel.
    """
    def __init__(self, backend, publish, complete, *, ready=None, pace=True, max_queue=16):
        self.backend, self.publish, self.complete = backend, publish, complete
        self.ready, self.pace = ready, bool(pace)
        self._queue = queue.Queue(maxsize=max_queue)
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._thread = None
        self._active = None
        self.last_error = None
        self._timing_enabled = bool(getattr(backend, 'timing_enabled', False))
        self._timing_counter = 0
        # Only this ordered session owns spoken auto-language context.
        self._auto_language = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self):
        with self._state_lock:
            if self.running:
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError('Previous V6 inference is still stopping')
            self._stop.clear()
            self._wakeup.clear()
            self._thread = threading.Thread(target=self._worker, name='xiaoyou-v6-pcm', daemon=True)
            self._thread.start()

    def enqueue(self, request):
        with self._state_lock:
            if not self.running:
                raise RuntimeError('V6 session is not running')
            if not request.text.strip():
                raise ValueError('Nonempty text is required')
            if self._timing_enabled:
                now = time.perf_counter()
                self._timing_counter += 1
                request.timing = {'timing': {'enabled': True, 'schema': 'xiaoyou_stage_wall_timing_v1',
                    'origin_perf': now, 'trace_id': request.trace_id, 'action_id': request.action_id,
                    'session_request_index': self._timing_counter, 'attempts': [], 'producer_drained': True,
                    'session': {'enqueued_seconds': 0.0, 'pace': self.pace}}}
            self._queue.put_nowait(request)
            self._wakeup.set()

    def _finish(self, request, status, error=None, *, eof=True):
        with request.delivery_lock:
            if eof and not request.eof_sent:
                self.publish(EOF_MAGIC)
                request.eof_sent = True
            if not request.completed:
                request.completed = True
                result = {'bytes': request.bytes_sent, 'frames': (request.bytes_sent + FRAME_BYTES - 1) // FRAME_BYTES,
                          'trace_id': request.trace_id, 'eof_sent': request.eof_sent}
                if error:
                    result['error'] = str(error)
                if request.timing is not None:
                    trace = request.timing['timing']
                    trace['session']['completion_status'] = status
                    trace['session']['completion_seconds'] = time.perf_counter() - trace['origin_perf']
                self.complete(request.action_id, status, result)

    def interrupt(self):
        with self._state_lock:
            active = self._active
            if active is not None:
                with active.delivery_lock:
                    active.cancel.set()
                    self._finish(active, 'cancelled')
            cleared = 0
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                pending.cancel.set()
                self._finish(pending, 'cancelled', eof=False)
                self._queue.task_done()
                cleared += 1
            return cleared

    def stop(self, timeout=30):
        self._stop.set()
        self._wakeup.set()
        self.interrupt()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            raise RuntimeError('V6 generation is still active; resources were not freed')

    def _send(self, request, frame, origin, frame_index):
        if self.pace:
            request.cancel.wait(max(0, origin + frame_index * FRAME_INTERVAL_S - time.monotonic()))
        with request.delivery_lock:
            if request.cancel.is_set() or self._stop.is_set() or request.eof_sent:
                return False
            trace = request.timing['timing'] if request.timing is not None else None
            first = trace is not None and 'first_publish_enter_seconds' not in trace['session']
            if first:
                trace['session']['first_publish_enter_seconds'] = time.perf_counter() - trace['origin_perf']
            self.publish(frame)
            if first:
                trace['session']['first_publish_return_seconds'] = time.perf_counter() - trace['origin_perf']
            request.bytes_sent += len(frame)
            return True

    def _deliver(self, request):
        trace = request.timing['timing'] if request.timing is not None else None
        if self.ready is not None:
            started = time.perf_counter() if trace is not None else None
            try:
                self.ready(request.cancel)
            finally:
                if trace is not None:
                    trace['session']['ready_wait_seconds'] = time.perf_counter() - started
        pending = bytearray()
        prebuf = []
        origin = None
        frame_index = 0

        def emit(frame):
            nonlocal origin, frame_index
            if origin is None:
                origin = time.monotonic()
            if not self._send(request, frame, origin, frame_index):
                return False
            frame_index += 1
            return True

        def flush_prebuf():
            nonlocal origin
            if not prebuf:
                return True
            # Backdate only once: the initial frames are all due now. Later
            # frames keep the same100ms schedule, with no rebase or speedup.
            origin = time.monotonic() - max(0, len(prebuf) - 1) * FRAME_INTERVAL_S
            for frame in prebuf:
                if not emit(frame):
                    return False
            prebuf.clear()
            return True

        options = dict(language=request.language, speed=1.0, cancel=request.cancel)
        if request.auto_fallback is not None:
            options['auto_fallback'] = request.auto_fallback
        if request.timing is not None:
            options['_request_metrics'] = request.timing
        chunks = self.backend.iter_pcm(request.text, **options)
        for chunk in chunks:
            if trace is not None and 'first_backend_chunk_seconds' not in trace['session']:
                trace['session']['first_backend_chunk_seconds'] = time.perf_counter() - trace['origin_perf']
            if request.cancel.is_set() or self._stop.is_set():
                break
            if not isinstance(chunk, bytes) or len(chunk) % 2:
                raise ValueError('V6 backend must yield aligned PCM16 bytes')
            pending.extend(chunk)
            while len(pending) >= FRAME_BYTES:
                frame = bytes(pending[:FRAME_BYTES])
                del pending[:FRAME_BYTES]
                if self.pace and origin is None:
                    prebuf.append(frame)
                    if len(prebuf) >= PREBUF_FRAMES and not flush_prebuf():
                        return
                elif not emit(frame):
                    return
        if not request.cancel.is_set() and not self._stop.is_set():
            if pending:
                tail = bytes(pending)
                if self.pace and origin is None:
                    # A short utterance's final partial joins its EOF flush.
                    # No fabricated/silent frame or extra playback wait.
                    prebuf.append(tail)
                elif not emit(tail):
                    return
            if prebuf and not flush_prebuf():
                return
        if not request.bytes_sent and not request.cancel.is_set() and not self._stop.is_set():
            raise RuntimeError('V6 generated no PCM')

    def _worker(self):
        while not self._stop.is_set():
            # Dequeue and publish active ownership atomically against interrupt.
            with self._state_lock:
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    request = None
                    # enqueue uses this same lock, so its wake cannot be erased.
                    self._wakeup.clear()
                if request is not None:
                    if self._stop.is_set():
                        request.cancel.set()
                    self._active = request
                    if request.timing is not None:
                        trace = request.timing['timing']
                        elapsed = time.perf_counter() - trace['origin_perf']
                        trace['session']['dequeued_seconds'] = elapsed
                        trace['session']['queue_wait_seconds'] = elapsed - trace['session']['enqueued_seconds']
            if request is None:
                # stop may have set its wake before the empty-queue clear.
                if not self._stop.is_set():
                    self._wakeup.wait()
                continue
            try:
                next_language = self._auto_language
                if not request.cancel.is_set():
                    if request.language in ('zh', 'en'):
                        request.auto_fallback = None
                        next_language = request.language
                    else:
                        from ..frontend.numbers import auto_context_language
                        request.auto_fallback = self._auto_language
                        detected, natural = auto_context_language(request.text, self._auto_language or 'zh')
                        if natural:
                            next_language = detected
                    self._deliver(request)
                status = 'cancelled' if request.cancel.is_set() or self._stop.is_set() else 'completed'
                self._finish(request, status)
                if status == 'completed' and request.bytes_sent and not request.cancel.is_set() and not self._stop.is_set():
                    self._auto_language = next_language
            except Exception as error:
                if request.timing is not None:
                    request.timing['timing']['session']['exception_type'] = type(error).__name__
                self.last_error = str(error)
                self._finish(request, 'cancelled', error=error)
            finally:
                with self._state_lock:
                    self._active = None
                self._queue.task_done()
                if request.timing is not None:
                    trace = request.timing['timing']
                    trace['session']['worker_finished_seconds'] = time.perf_counter() - trace['origin_perf']
                    if trace.get('producer_drained') is True:
                        _write_request_timing(request)
                    else:
                        log.warning('Xiaoyou request timing omitted because its producer did not drain')
