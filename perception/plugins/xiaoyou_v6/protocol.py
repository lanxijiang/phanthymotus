"""ROS-independent ordered PCM delivery with one EOF per started request."""
from dataclasses import dataclass, field
import queue
import threading
import time

SAMPLE_RATE = 16000
FRAME_BYTES = 3200
# Current official main transport: one initial500ms cushion, then real time.
PREBUF_FRAMES = 5
FRAME_INTERVAL_S = .1
EOF_MAGIC = b'\x01\x00\xff\xff\x01\x00\xff\xff'


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
        self._thread = None
        self._active = None
        self.last_error = None

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
            self._thread = threading.Thread(target=self._worker, name='xiaoyou-v6-pcm', daemon=True)
            self._thread.start()

    def enqueue(self, request):
        with self._state_lock:
            if not self.running:
                raise RuntimeError('V6 session is not running')
            if not request.text.strip():
                raise ValueError('Nonempty text is required')
            self._queue.put_nowait(request)

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
            self.publish(frame)
            request.bytes_sent += len(frame)
            return True

    def _deliver(self, request):
        if self.ready is not None:
            self.ready(request.cancel)
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

        for chunk in self.backend.iter_pcm(request.text, language=request.language, speed=1.0, cancel=request.cancel):
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
                if request is not None:
                    if self._stop.is_set():
                        request.cancel.set()
                    self._active = request
            if request is None:
                self._stop.wait(.01)
                continue
            try:
                if not request.cancel.is_set():
                    self._deliver(request)
                self._finish(request, 'cancelled' if request.cancel.is_set() or self._stop.is_set() else 'completed')
            except Exception as error:
                self.last_error = str(error)
                self._finish(request, 'cancelled', error=error)
            finally:
                with self._state_lock:
                    self._active = None
                self._queue.task_done()
