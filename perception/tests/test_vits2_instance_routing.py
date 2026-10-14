"""Exercise MCP instance routing without importing ROS or inference libraries."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


def load_plugin():
    package = "_vits2_routing_test"
    modules = {
        name: types.ModuleType(name)
        for name in (
            package, f"{package}.adapter", "audio_msgs", "audio_msgs.msg",
            "rclpy", "rclpy.node", "rclpy.qos", "std_msgs", "std_msgs.msg",
        )
    }
    modules[package].__path__ = []
    modules["audio_msgs.msg"].AudioChunk = object
    modules["std_msgs.msg"].String = object
    modules["rclpy.node"].Node = object
    qos = modules["rclpy.qos"]
    qos.QoSProfile = lambda **kwargs: kwargs
    qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE=0)
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=0)
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=0)
    adapter = modules[f"{package}.adapter"]
    adapter.CHUNK_BYTES = 3200
    adapter.PCM_FRAME_MS = 100
    adapter.SAMPLE_RATE = 16000
    adapter.TTSAdapter = object
    adapter.Vits2TensorRTAdapter = object
    adapter.build_adapter = Mock(side_effect=AssertionError("Inference forbidden"))

    source = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt/plugin.py"
    spec = importlib.util.spec_from_file_location(f"{package}.plugin", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class FakeNode:
    def __init__(self, topic, state="running"):
        self._input_topic = topic
        self._output_topic = f"{topic}/tts"
        self.state = state
        self.queued = []

    def enqueue(self, text):
        if self.state != "running":
            raise RuntimeError("not running")
        self.queued.append(text)

    def start(self):
        self.state = "running"


class InstanceRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()

    def setUp(self):
        self.plugin = self.module.TTSPlugin({"backend": "trt"}, Mock())
        self.plugin._ensure_adapter = Mock(return_value=object())
        self.a = FakeNode("/input/a")
        self.b = FakeNode("/input/b")
        self.plugin._nodes = {"a": self.a, "b": self.b}

    def speak(self, **args):
        return self.plugin.dispatch("tts", {"action": "speak", "text": "Hello", **args})

    def test_explicit_instance_uses_its_own_queue(self):
        result = self.speak(instance_id="b")
        self.assertEqual(self.a.queued, [])
        self.assertEqual(self.b.queued, ["Hello"])
        self.assertEqual(result, {"status": "queued", "chars": 5})

    def test_unknown_instance_fails_without_loading_or_rerouting(self):
        with self.assertRaisesRegex(ValueError, "Unknown TTS instance 'missing'"):
            self.speak(instance_id="missing")
        self.plugin._ensure_adapter.assert_not_called()
        self.assertEqual(self.a.queued + self.b.queued, [])

    def test_stopped_instance_fails_without_rerouting(self):
        self.b.state = "idle"
        with self.assertRaisesRegex(RuntimeError, "instance 'b' is not running"):
            self.speak(instance_id="b")
        self.plugin._ensure_adapter.assert_not_called()
        self.assertEqual(self.b.state, "idle")
        self.assertEqual(self.a.queued + self.b.queued, [])

    def test_omitted_instance_keeps_first_running_behavior(self):
        self.a.state = "idle"
        self.speak()
        self.assertEqual(self.a.queued, [])
        self.assertEqual(self.b.queued, ["Hello"])

    def test_omitted_instance_still_creates_and_starts_default(self):
        self.plugin._nodes = {}
        default = FakeNode("/input/default", state="idle")
        self.plugin._create_node = Mock(return_value=default)
        self.speak(input_topic="/input/default")
        self.plugin._create_node.assert_called_once_with("_default", "/input/default")
        self.assertEqual(default.state, "running")
        self.assertEqual(default.queued, ["Hello"])


if __name__ == "__main__":
    unittest.main()
