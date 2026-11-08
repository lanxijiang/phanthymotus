"""V6 MCP/ROS transport. No old TTS adapter, factory, or frontend is imported."""
import json
import logging
import threading
import time
import uuid

from .protocol import Request, SpeechSession
from .trt_runtime import acquire_gpu_backend, validate_options

log = logging.getLogger(__name__)

TOOLS = [{
    'name': 'tts', 'type': 'processor', 'multiInstance': True,
    'description': 'xiaoyou01 中英文语音，支持有序PCM交付和中断。',
    'inputSchema': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['info', 'config', 'start', 'speak', 'interrupt', 'stop']},
        'text': {'type': 'string'}, 'input_topic': {'type': 'string'},
        'instance_id': {'type': 'string'},
    }, 'required': ['action'], 'x-completion': {'actions': ['speak'], 'timeout': 60},
        'x-hooks': {'on_interrupt_speak': {'action': 'interrupt'}}},
    'configSchema': {'type': 'object', 'properties': {
        'speaker_id': {'type': 'integer', 'enum': [0], 'default': 0, 'scope': 'shared'},
        'speed': {'type': 'number', 'enum': [1.0], 'default': 1.0, 'scope': 'shared'},
        'language': {'type': 'string', 'enum': ['auto', 'zh', 'en'], 'default': 'auto', 'scope': 'shared'},
    }},
    'topic_in': [{'format': 'data/json', 'desc': 'text'}],
    'topic_out': [{'format': 'audio/pcm-16k', 'desc': '16000Hz mono PCM16'}],
}]


def _node(input_topic, backend, suffix, cfg):
    # ROS is needed only for topic delivery, not native loading/raw inference.
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from audio_msgs.msg import AudioChunk
    from utils.acp import complete
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                     depth=200, durability=DurabilityPolicy.VOLATILE)

    class V6Node(Node):
        def __init__(self):
            super().__init__('xiaoyou_v6_' + suffix)
            self._input_topic = input_topic or ''
            self._output_topic = self._input_topic + '/tts' if self._input_topic else '/perception/tts'
            self._pub = self.create_publisher(AudioChunk, self._output_topic, qos)
            self._sub = self.create_subscription(String, self._input_topic, self._text_cb, qos) if self._input_topic else None
            self.session = SpeechSession(backend, self._publish, complete, ready=self._ready,
                                         pace=cfg.get('pace_audio', True))

        @property
        def state(self):
            return 'running' if self.session.running else 'idle'

        def _publish(self, frame):
            msg = AudioChunk()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'audio/pcm-16k'
            msg.data = list(frame)
            self._pub.publish(msg)

        def _ready(self, cancel):
            timeout = float(cfg.get('subscriber_wait_s', 3.0))
            if timeout <= 0:
                return
            deadline = time.monotonic() + timeout
            while self._pub.get_subscription_count() < 1:
                if cancel.wait(.02):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError('No V6 PCM subscriber became ready')

        def _text_cb(self, msg):
            try:
                body = json.loads(msg.data)
                text = body.get('text', '') if isinstance(body, dict) else str(body)
            except (ValueError, TypeError):
                text = msg.data
            if text and self.session.running:
                self.session.enqueue(Request(text, '', cfg.get('language', 'auto')))

    return V6Node()


class TTSPlugin:
    PREFIX = 'tts'

    def __init__(self, plugin_cfg, executor):
        self._cfg = dict(plugin_cfg)
        if self._cfg.get('engine', 'xiaoyou-v6') != 'xiaoyou-v6':
            raise ValueError('V6 plugin cannot use a different engine')
        self._cfg.setdefault('language', 'auto')
        validate_options(self._cfg['language'], self._cfg.get('speed', 1), self._cfg.get('speaker_id', 0))
        self._executor = executor
        self._nodes = {}
        self._lock = threading.RLock()
        self._load_error = None
        self._backend = None
        self._shutdown = False
        # Existing public tts._dispose_impl calls _adapter.close() on engine switches.
        # Reuse this instance instead of introducing another adapter or public tool.
        self._adapter = self
        try:
            backend = self._cfg.get('backend', 'tensorrt_fp32')
            if backend == 'tensorrt_fp32':
                self._backend = acquire_gpu_backend(self._cfg.get('model_dir'),
                    cpu_threads=self._cfg.get('cpu_threads', 2),
                    expected_manifest_sha256=self._cfg.get('runtime_manifest_sha256'))
            else:
                raise ValueError('xiaoyou01 production requires tensorrt_fp32; no CPU backend or fallback')
        except Exception as error:
            self._load_error = str(error)
            log.exception('xiaoyou01 requested backend failed to load; no fallback')

    def get_tools(self):
        return TOOLS

    def _info(self, args):
        identity = {'name': 'xiaoyou01', 'model': 'xiaoyou01', 'model_id': 'xiaoyou01', 'voice_id': 'xiaoyou01'}
        if self._load_error:
            return {**identity, 'state': 'error', 'desc': self._load_error}
        identifier = args.get('instance_id')
        nodes = ([self._nodes[identifier]] if identifier in self._nodes else []) if identifier else list(self._nodes.values())
        metadata = getattr(self._backend, 'metadata', {})
        backend = metadata.get('backend', self._cfg.get('backend', 'tensorrt_fp32'))
        return {**identity, 'state': 'running' if any(n.state == 'running' for n in nodes) else 'idle',
                'backend': backend, 'accelerated': bool(metadata.get('accelerated', False)),
                'generation': metadata.get('streaming', 'natural-text-streaming'),
                'ttft_validated': bool(metadata.get('ttft_validated', False)),
                'jetson_validated': bool(metadata.get('jetson_validated', False)),
                'darvin_validated': bool(metadata.get('darvin_validated', False)),
                'three_process_instances_validated': bool(metadata.get('three_process_instances_validated', False)),
                'language': self._cfg['language'], 'speaker_id': 0, 'speed': 1.0,
                'topic_in': [{'topic': n._input_topic, 'format': 'data/json'} for n in nodes if n._input_topic],
                'topic_out': [{'topic': n._output_topic, 'format': 'audio/pcm-16k'} for n in nodes],
                'desc': 'xiaoyou01 FP32 GPU speech; deployment acceptance is reported explicitly.'}

    def _start(self, args):
        key = args.get('instance_id') or args.get('input_topic') or '_default'
        topic = args.get('input_topic') or ''
        if key in self._nodes and self._nodes[key]._input_topic != topic and topic:
            self._remove(key)
        if key not in self._nodes:
            if self._executor is None:
                raise RuntimeError('A ROS executor is required for V6 topic delivery')
            # One publisher owns any output topic within this plugin.
            output = topic + '/tts' if topic else '/perception/tts'
            for other in list(self._nodes):
                if self._nodes[other]._output_topic == output:
                    self._remove(other)
            suffix = ''.join(c if c.isalnum() else '_' for c in key)
            node = _node(topic, self._backend, suffix, self._cfg)
            self._executor.add_node(node)
            self._nodes[key] = node
        node = self._nodes[key]
        if self._cfg.get('warmup', True):
            self._backend.warmup()
        node.session.start()
        return node

    def _remove(self, key):
        node = self._nodes[key]
        node.session.stop()
        self._executor.remove_node(node)
        node.destroy_node()
        del self._nodes[key]

    def dispatch(self, name, args):
        action = args.get('action') if name == 'tts' else name
        if action not in ('info', 'config', 'start', 'speak', 'interrupt', 'stop'):
            return None
        with self._lock:
            if action == 'info':
                return self._info(args)
            if self._shutdown:
                return {'state': 'error', 'message': 'V6 plugin has shut down'}
            if self._load_error or self._backend is None:
                return {'state': 'error', 'message': 'V6 model unavailable: ' + str(self._load_error)}
            try:
                if action == 'config':
                    language = args.get('language', self._cfg['language'])
                    validate_options(language, args.get('speed', 1), args.get('speaker_id', 0))
                    self._cfg['language'] = language
                    return {'status': 'configured', 'language': language, 'speed': 1.0, 'speaker_id': 0}
                if action == 'start':
                    node = self._start(args)
                    return {'state': node.state, 'topic_in': [{'topic': node._input_topic, 'format': 'data/json'}],
                            'topic_out': [{'topic': node._output_topic, 'format': 'audio/pcm-16k'}]}
                if action == 'speak':
                    text = args.get('text', '')
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('Nonempty text is required')
                    if not args.get('instance_id') and not args.get('input_topic'):
                        running = [n for n in self._nodes.values() if n.state == 'running']
                        if len(running) > 1:
                            raise ValueError('Multiple V6 output nodes exist; select instance_id for speak')
                        node = running[0] if running else self._start(args)
                    else:
                        node = self._start(args)
                    action_id = 'speak-' + uuid.uuid4().hex[:12]
                    node.session.enqueue(Request(text, action_id, self._cfg['language'], args.get('_trace_id', '')))
                    return {'status': 'queued', 'action_id': action_id}
                keys = [args['instance_id']] if args.get('instance_id') in self._nodes else (
                    [] if args.get('instance_id') else list(self._nodes))
                if action == 'interrupt':
                    cleared = sum(self._nodes[k].session.interrupt() for k in keys)
                    return {'status': 'interrupted', 'nodes': len(keys), 'cleared': cleared}
                for key in keys:
                    self._remove(key)
                return {'state': 'idle'}
            except Exception as error:
                log.exception('V6 %s failed', action)
                return {'state': 'error', 'message': str(error)}

    def synthesize_raw(self, text):
        with self._lock:
            if self._shutdown or self._backend is None:
                raise RuntimeError('xiaoyou01 requested model is unavailable')
            language = self._cfg['language']
        return self._backend.synthesize(text, language=language, speed=1)

    def shutdown(self):
        with self._lock:
            if self._shutdown:
                return
            # A failed stop deliberately leaves the backend alive: an active
            # producer must finish before its GPU lease can be released.
            for key in list(self._nodes):
                self._remove(key)
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._shutdown = True

    def close(self):
        """Compatibility with the existing public factory's disposal hook."""
        self.shutdown()
