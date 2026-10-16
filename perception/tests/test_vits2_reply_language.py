"""Independent language-policy and ROS/MCP queue contract regression tests.

ROS and synthesis are replaced at their external boundaries. The actual addon
policy, dispatch, node enqueue and topic callbacks run here; these tests do not
claim to test ASR, LLM translation or acoustic quality.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch


class _RosNode:
    def __init__(self, name):
        self.name = name
        self.publisher = Mock()
        self.subscriptions = []

    def create_publisher(self, message_type, topic, qos):
        return self.publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions.append((topic, callback))
        return Mock()

    def destroy_node(self):
        pass


class _FakeAdapter:
    def __init__(self):
        self.spoken = []
        self.speed = 1.0
        self.speed_updates = []
        self.frontend_inputs = []
        self.normalized_chunks = None

    def set_speed(self, value):
        self.speed = value
        self.speed_updates.append(value)

    def synthesize_stream(self, text):
        self.spoken.append(text)
        yield b"\x01\x00" * 8

    def iter_text_chunks(self, text):
        self.frontend_inputs.append(text)
        for chunk in self.normalized_chunks if self.normalized_chunks is not None else (text,):
            yield chunk, ([1], [1], [0])


def _load_modules():
    name = "_reply_language_test_plugins"
    root = Path(__file__).resolve().parents[1] / "plugins"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    modules = {name: package}
    for stub in (
        "audio_msgs", "audio_msgs.msg", "rclpy", "rclpy.node", "rclpy.qos",
        "std_msgs", "std_msgs.msg", f"{name}.vits2_tts_trt.adapter",
    ):
        modules[stub] = types.ModuleType(stub)
    modules["rclpy.node"].Node = _RosNode
    modules["audio_msgs.msg"].AudioChunk = object
    modules["std_msgs.msg"].String = object
    qos = modules["rclpy.qos"]
    qos.QoSProfile = lambda **kwargs: kwargs
    qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE=0)
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=0)
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=0)
    adapter = modules[f"{name}.vits2_tts_trt.adapter"]
    adapter.CHUNK_BYTES = 3200
    adapter.PCM_FRAME_MS = 100
    adapter.SAMPLE_RATE = 16000
    adapter.TTSAdapter = object
    adapter.Vits2TensorRTAdapter = _FakeAdapter
    adapter.build_adapter = Mock(side_effect=AssertionError("Inference forbidden"))
    with patch.dict(sys.modules, modules):
        policy = importlib.import_module(f"{name}.vits2_tts_language.policy")
        plugin = importlib.import_module(f"{name}.vits2_tts_language.plugin")
    return policy, plugin


def _load_original_adapter():
    """Run the actual adapter method; only constructing a CUDA engine is stubbed."""
    name = "_reply_language_original_adapter"
    root = Path(__file__).resolve().parents[1] / "plugins/vits2_tts_trt"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    engine_module = types.ModuleType(f"{name}.runtime.backends.trt_numpy_tts_engine")
    engine_module.TensorRTNumpyTTSEngine = Mock(
        side_effect=AssertionError("A real GPU engine must not be constructed in this test")
    )
    with patch.dict(sys.modules, {name: package, engine_module.__name__: engine_module}):
        return importlib.import_module(f"{name}.adapter")


class _RecordingEngine:
    """Deterministic acoustic boundary double; never claims real audio quality."""

    max_text_tokens = 256

    def __init__(self):
        self.calls = []
        self.reset_count = 0
        self.chunk_index = 0
        self.adapter = None

    def reset_random_state(self):
        self.reset_count += 1
        self.chunk_index = 0

    def synthesize(self, text, *, text_ids, length_scale):
        self.calls.append((
            text, tuple(tuple(values) for values in text_ids), length_scale,
            self.adapter._lock.locked(), self.chunk_index,
        ))
        self.chunk_index += 1
        # Multiple PCM frames per chunk expose changes in framing/order.
        return bytes((sum(text_ids[0]) % 128 + self.chunk_index, 0)) * 2401


def _recording_adapter(module, chunks_by_text, speed=1.0):
    engine = _RecordingEngine()
    adapter = module.Vits2TensorRTAdapter(engine=engine, speed=speed)
    engine.adapter = adapter
    frontend_inputs = []

    def frontend(text):
        frontend_inputs.append(text)
        yield from chunks_by_text[text]

    adapter.iter_text_chunks = frontend
    return adapter, engine, frontend_inputs


class LanguagePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, _ = _load_modules()

    def setUp(self):
        self.policy = self.module.LanguagePolicy()

    def assert_code(self, code, operation):
        with self.assertRaises(self.module.LanguagePolicyError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_auto_follows_current_input_and_switches_back(self):
        tokens = []
        for text, expected, answer in (
            ("hello", "en", "Hello, how can I help you?"),
            ("你好", "zh", "你好，请问有什么可以帮助你的？"),
            ("hello again", "en", "Welcome back."),
        ):
            result = self.policy.begin_turn(text)
            self.assertEqual(result["reply_language"], expected)
            self.assertEqual(result["detected_language"], expected)
            request = self.policy.prepare(answer, result["turn_id"])
            self.assertEqual((request.text, request.language), (answer, expected))
            tokens.append(result["turn_id"])
        self.assertEqual(len(set(tokens)), 3)

    def test_fixed_modes_override_input_language(self):
        for mode, text, answer in (("zh", "hello", "你好。"), ("en", "你好", "Hello.")):
            with self.subTest(mode=mode):
                policy = self.module.LanguagePolicy(mode)
                result = policy.begin_turn(text)
                self.assertEqual(result["reply_language"], mode)
                self.assertEqual(policy.prepare(answer, result["turn_id"]).language, mode)

    def test_instruction_is_available_before_generating_reply(self):
        for text, language_name in (("hello", "English"), ("你好", "Chinese")):
            result = self.policy.begin_turn(text)
            self.assertIn("mainly in " + language_name, result["instruction"])
            self.assertIn("mixing is allowed", result["instruction"])
            self.assertNotIn("entirely", result["instruction"])
            self.assertIsInstance(result["turn_id"], str)

    def test_speak_cannot_infer_language_from_generated_reply(self):
        self.assert_code("language_turn_required", lambda: self.policy.prepare("Hello", "abc"))

    def test_old_token_is_rejected_after_switch(self):
        first = self.policy.begin_turn("hello")
        second = self.policy.begin_turn("你好")
        self.assert_code("stale_turn", lambda: self.policy.prepare("Hello", first["turn_id"]))
        self.assertEqual(self.policy.prepare("你好", second["turn_id"]).language, "zh")

    def test_new_policy_cannot_reuse_old_policy_token(self):
        old = self.policy.begin_turn("hello")
        replacement = self.module.LanguagePolicy()
        current = replacement.begin_turn("hello")
        self.assertNotEqual(old["turn_id"], current["turn_id"])
        self.assert_code("stale_turn", lambda: replacement.prepare("Hello", old["turn_id"]))

    def test_config_change_invalidates_pending_reply(self):
        first = self.policy.begin_turn("hello")
        self.policy.configure("zh")
        self.assert_code("language_turn_required", lambda: self.policy.prepare("Hello", first["turn_id"]))
        next_turn = self.policy.begin_turn("hello")
        self.assertEqual(next_turn["reply_language"], "zh")

    def test_same_mode_config_keeps_valid_turn(self):
        first = self.policy.begin_turn("hello")
        self.policy.configure("auto")
        self.assertEqual(self.policy.prepare("Hello", first["turn_id"]).language, "en")

    def test_invalid_input_and_config_do_not_mutate_accepted_turn(self):
        accepted = self.policy.begin_turn("hello")
        before = self.policy.snapshot()
        operations = (
            ("empty_text", lambda: self.policy.begin_turn(" ")),
            ("empty_text", lambda: self.policy.begin_turn(None)),
            ("invalid_input_language", lambda: self.policy.begin_turn("hello", "fr")),
            ("invalid_requested_reply_language", lambda: self.policy.begin_turn(
                "hello", requested_reply_language="auto")),
            ("unsupported_script", lambda: self.policy.begin_turn("สวัสดี")),
            ("invalid_reply_language", lambda: self.policy.configure("fr")),
        )
        for code, operation in operations:
            with self.subTest(code=code):
                self.assert_code(code, operation)
                self.assertEqual(self.policy.snapshot(), before)
        self.assertEqual(self.policy.prepare("Hello", accepted["turn_id"]).language, "en")

    def test_explicit_asr_language_resolves_mixed_first_input(self):
        for text in ("你好 hello", "12345"):
            for language in ("zh", "en"):
                with self.subTest(text=text, language=language):
                    policy = self.module.LanguagePolicy()
                    result = policy.begin_turn(text, input_language=language)
                    self.assertEqual(result["reply_language"], language)
                    self.assertEqual(result["detected_language"], language)
                    self.assertEqual(result["language_source"], "asr")

    def test_auto_uses_english_grammar_around_chinese_names(self):
        for text in (
            "Please check 北京 and tell me the result.",
            "We saw Qwen at 北京.",
            "Can you explain 北京交通大学人工智能与机器人研究中心 to me?",
            "What is 北京?",
            "Please explain 北京.",
            "Tell me about 北京.",
            "Please explain 北京交通大学人工智能与机器人研究中心.",
            "北京 is beautiful.",
            "北京 has a large network and is ready.",
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual((turn["reply_language"], turn["detected_language"]), ("en", "en"))
                self.assertEqual(turn["language_source"], "text")

    def test_auto_keeps_chinese_grammar_with_many_english_product_names(self):
        for text in (
            "我们要检查 OpenMLDB Qwen SQL Playground 和 FeatInsight。",
            "请使用 NVIDIA TensorRT OpenMLDB FeatInsight Qwen 进行测试。",
            "我们在waic会议上看到许多产品和模型，qwen、豆包等都有出现",
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertEqual(turn["language_source"], "text")

    def test_ambiguous_names_and_unquoted_titles_do_not_force_english(self):
        for text in ("你好 hello", "北京 server 8.5GB", "请看 The Lord of the Rings。",
                     "我想读 The Lord of the Rings。", "介绍 The Lord of the Rings。",
                     "推荐 The Lord of the Rings。", "讲讲 The Lord of the Rings。",
                     "聊聊 The Lord of the Rings。", "谈谈 The Lord of the Rings。",
                     "读一读 The Lord of the Rings。"):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertIsNone(turn["detected_language"])
                self.assertEqual(turn["language_source"], "default")

    def test_conflicting_bilingual_clauses_do_not_inherit_prior_english(self):
        for text in ("Please check the network. 然后我们检查电量。",
                     "We will check the network, then report the result. 请检查电量。"):
            with self.subTest(text=text):
                previous = self.policy.begin_turn("hello")
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertNotEqual(turn["turn_id"], previous["turn_id"])

    def test_quotes_and_parentheses_do_not_override_outer_language(self):
        for text, expected in (
            ('请读英文“Please check the network and tell me the result.”', "zh"),
            ('Please read “请检查网络然后告诉我们结果。” for me.', "en"),
            ('Please read (中文“请检查网络”) for me.', "en"),
            ('请念英文“Please check the network', "zh"),
            ('《The Lord of the Rings》是什么？', "zh"),
            ('介绍《The Lord of the Rings》。', "zh"),
            ("It's 北京, and the users' network isn't ready.", "en"),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], expected)
                self.assertEqual(turn["language_source"], "text")

    def test_addresses_and_numeric_entities_are_not_language_evidence(self):
        for text, expected, source in (
            ("请查看 https://we.are.at.example.com/this/is/our/path 和 me@our.example.com。", "zh", "text"),
            ("Please check 北京 at https://example.com/qwen3 and tell me the result.", "en", "text"),
            ("https://we.are.at.example.com/this/is/our/path", "zh", "default"),
            ("2026-08-07 14:25 RTX-4090 8.5GB", "zh", "default"),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual((turn["reply_language"], turn["language_source"]), (expected, source))

    def test_asr_label_overrides_conflicting_text_grammar(self):
        for text, language in (
            ("Please check 北京 and tell me the result.", "zh"),
            ("请检查 OpenMLDB 和 Qwen。", "en"),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text, input_language=language)
                self.assertEqual((turn["reply_language"], turn["language_source"]), (language, "asr"))

    def test_current_user_requested_language_overrides_asr_and_fixed_preference(self):
        for mode in ("auto", "zh", "en"):
            for language, opposite in (("zh", "en"), ("en", "zh")):
                with self.subTest(mode=mode, language=language):
                    policy = self.module.LanguagePolicy(mode)
                    turn = policy.begin_turn("hello", input_language=opposite,
                                             requested_reply_language=language)
                    self.assertEqual((turn["reply_language"], turn["language_source"]), (language, "requested"))
                    self.assertEqual(turn["detected_language"], opposite)
                    self.assertEqual(policy.snapshot()["reply_language"], mode)

    def test_requested_language_is_one_turn_only_and_never_changes_configuration(self):
        for mode, following in (("auto", "zh"), ("zh", "zh"), ("en", "en")):
            with self.subTest(mode=mode):
                policy = self.module.LanguagePolicy(mode)
                first = policy.begin_turn("你好", requested_reply_language="en")
                request = policy.prepare("Hello.", first["turn_id"])
                second = policy.begin_turn("你好")
                self.assertEqual(second["reply_language"], following)
                self.assertEqual(policy.snapshot()["reply_language"], mode)
                self.assertEqual((request.text, request.language), ("Hello.", "en"))
                self.assertNotEqual(first["turn_id"], second["turn_id"])

    def test_arbitrary_language_instructions_require_caller_interpretation(self):
        for text, detected, requested in (
            ("Please answer in Chinese.", "en", "zh"),
            ("请用英文回答。", "zh", "en"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.policy.begin_turn(text)["reply_language"], detected)
                turn = self.policy.begin_turn(text, requested_reply_language=requested)
                self.assertEqual((turn["reply_language"], turn["language_source"]), (requested, "requested"))

    def test_first_mixed_or_numeric_auto_input_defaults_to_chinese(self):
        for text in ("你好 hello", "12345", "🙂!?"):
            with self.subTest(text=text):
                policy = self.module.LanguagePolicy()
                result = policy.begin_turn(text)
                self.assertEqual(result["reply_language"], "zh")
                self.assertIsNone(result["detected_language"])
                self.assertEqual(result["language_source"], "default")
                self.assertIn("Chinese", result["instruction"])
                self.assertEqual(policy.prepare("你好。", result["turn_id"]).language, "zh")

    def test_neutral_and_mixed_input_default_to_chinese_after_english(self):
        for text in ("12345", "你好 hello", "🙂!?"):
            with self.subTest(text=text):
                previous = self.policy.begin_turn("hello")
                result = self.policy.begin_turn(text)
                self.assertEqual(result["reply_language"], "zh")
                self.assertIsNone(result["detected_language"])
                self.assertEqual(self.policy.prepare("你好。", result["turn_id"]).language, "zh")
                self.assert_code(
                    "stale_turn", lambda: self.policy.prepare("Hello", previous["turn_id"]),
                )

    def test_fixed_mode_accepts_first_neutral_input(self):
        for mode in ("zh", "en"):
            for text in ("12345", "你好 hello"):
                with self.subTest(mode=mode, text=text):
                    policy = self.module.LanguagePolicy(mode)
                    result = policy.begin_turn(text)
                    self.assertEqual(result["reply_language"], mode)

    def test_wrong_reply_script_rejected_without_text_rewriting(self):
        for user, bad_replies in (
            ("hello", ("你好", "Привет", "สวัสดี", "Hello 你好 สวัสดี")),
            ("你好", ("Hello", "こんにちは", "你好 hello Привет")),
        ):
            result = self.policy.begin_turn(user)
            before = self.policy.snapshot()
            for reply in bad_replies:
                with self.subTest(reply=reply):
                    self.assert_code(
                        "reply_language_mismatch",
                        lambda: self.policy.prepare(reply, result["turn_id"]),
                    )
                    self.assertEqual(self.policy.snapshot(), before)

    def test_mixed_reply_preserves_text_for_either_primary_language(self):
        for user, reply, language in (
            ("你好", "我们在waic会议上看到许多产品和模型，qwen、豆包等都有出现", "zh"),
            ("你好", "你好，please wait here. 我马上回来。", "zh"),
            ("hello", "Hello，欢迎来到北京。Please wait here.", "en"),
        ):
            with self.subTest(user=user, reply=reply):
                turn = self.policy.begin_turn(user)
                request = self.policy.prepare(reply, turn["turn_id"])
                self.assertEqual((request.text, request.language), (reply, language))
                self.assertEqual(request.turn_id, turn["turn_id"])

    def test_request_is_frozen_and_preserves_original_text(self):
        result = self.policy.begin_turn("hello")
        text = "Hello, the value is 1.2."
        request = self.policy.prepare(text, result["turn_id"])
        self.assertEqual(request.text, text)
        self.policy.begin_turn("你好")
        self.assertEqual(request.language, "en")
        self.assertEqual(request.turn_id, result["turn_id"])
        with self.assertRaises(FrozenInstanceError):
            request.language = "zh"

    def test_numeric_or_punctuation_only_reply_requires_explicit_spoken_words(self):
        for utterance in ("hello", "你好"):
            turn = self.policy.begin_turn(utterance)
            for reply in ("123", "1.2", "?!", "🙂", "－１２℃"):
                with self.subTest(utterance=utterance, reply=reply):
                    self.assert_code("ambiguous_reply_language", lambda: self.policy.prepare(
                        reply, turn["turn_id"]
                    ))

    def test_sessions_are_independent(self):
        other = self.module.LanguagePolicy()
        a = self.policy.begin_turn("hello")
        b = other.begin_turn("你好")
        self.assertEqual(self.policy.prepare("Hello", a["turn_id"]).language, "en")
        self.assertEqual(other.prepare("你好", b["turn_id"]).language, "zh")
        self.assert_code("stale_turn", lambda: other.prepare("你好", a["turn_id"]))

    def test_concurrent_turn_switch_never_relabels_an_accepted_reply(self):
        for _ in range(20):
            policy = self.module.LanguagePolicy()
            first = policy.begin_turn("hello")
            barrier = threading.Barrier(2)

            def prepare():
                barrier.wait(timeout=3)
                try:
                    return policy.prepare("Hello", first["turn_id"])
                except self.module.LanguagePolicyError as exc:
                    return exc.code

            def switch():
                barrier.wait(timeout=3)
                return policy.begin_turn("你好")

            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = pool.submit(prepare)
                switched = pool.submit(switch)
                result = pending.result(timeout=5)
                switched.result(timeout=5)
            if isinstance(result, str):
                self.assertEqual(result, "stale_turn")
            else:
                self.assertEqual((result.text, result.language, result.turn_id),
                                 ("Hello", "en", first["turn_id"]))
            self.assertEqual(policy.snapshot()["resolved_language"], "zh")


class LanguagePluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_module, cls.module = _load_modules()

    def setUp(self):
        self.adapter = _FakeAdapter()
        self.plugin = self.module.TTSPlugin({"backend": "trt", "speed": 1.0}, Mock())
        self.plugin._adapter = self.adapter
        self.plugin._ensure_adapter = Mock(return_value=self.adapter)

        def start_without_background_worker(node):
            node.state = "running"
            return node.status()

        self.start_patch = patch.object(self.module._LanguageTTSNode, "start", start_without_background_worker)
        self.start_patch.start()
        self.addCleanup(self.start_patch.stop)

    def call(self, action, **kwargs):
        return self.plugin.dispatch("tts", {"action": action, **kwargs})

    def start(self, instance_id="a", **kwargs):
        self.call("start", instance_id=instance_id, input_topic=f"/input/{instance_id}", **kwargs)
        return self.plugin._nodes[instance_id]

    def begin(self, text, instance_id="a", **kwargs):
        return self.call("language", instance_id=instance_id, text=text, **kwargs)

    def assert_code(self, code, operation):
        with self.assertRaises(self.policy_module.LanguagePolicyError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_standard_tts_tool_exposes_language_contract(self):
        tools = self.plugin.get_tools()
        tool = next(item for item in tools if item["name"] == "tts")
        properties = tool["inputSchema"]["properties"]
        self.assertIn("language", properties["action"]["enum"])
        for key in ("instance_id", "turn_id", "input_language", "requested_reply_language"):
            self.assertIn(key, properties)

    def test_dispatch_requested_language_overrides_this_turn_only(self):
        node = self.start(reply_language="zh")
        turn = self.begin("请用英文回答。", input_language="zh", requested_reply_language="en")
        self.assertEqual((turn["reply_language"], turn["language_source"]), ("en", "requested"))
        self.call("speak", instance_id="a", text="Hello, welcome to 北京.", turn_id=turn["turn_id"])
        request = node._text_queue.get_nowait()
        self.assertEqual(request.language, "en")
        self.assertEqual(self.adapter.frontend_inputs, ["Hello, welcome to 北京."])
        self.assertEqual(self.begin("hello")["reply_language"], "zh")
        self.assertEqual(node.policy.snapshot()["reply_language"], "zh")
        self.assertEqual(request.language, "en")

    def test_invalid_requested_language_keeps_accepted_turn_and_empty_queue(self):
        node = self.start()
        accepted = self.begin("hello")
        self.assert_code("invalid_requested_reply_language", lambda: self.begin(
            "你好", requested_reply_language="fr",
        ))
        self.assertEqual(node.policy.snapshot()["turn_id"], accepted["turn_id"])
        self.assertEqual(node.policy.snapshot()["resolved_language"], "en")
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(self.adapter.frontend_inputs, [])

    def test_opt_in_plugin_does_not_mutate_base_tool_schema(self):
        original = json.dumps(self.module.BASE_TOOLS, sort_keys=True)
        self.plugin.get_tools()
        self.plugin.get_tools()
        self.assertEqual(json.dumps(self.module.BASE_TOOLS, sort_keys=True), original)
        self.assertNotIn("language", self.module.BASE_TOOLS[0]["inputSchema"]["properties"]["action"]["enum"])

    def test_missing_instance_rejected_before_loading_or_creating_nodes(self):
        for action in ("start", "language", "speak"):
            with self.subTest(action=action):
                self.assert_code("instance_required", lambda: self.call(action, text="hello"))
        self.plugin._ensure_adapter.assert_not_called()
        self.assertEqual(self.plugin._nodes, {})

    def test_unknown_instance_never_reroutes_to_running_instance(self):
        node = self.start("a")
        self.plugin._ensure_adapter.reset_mock()
        for action in ("language", "speak"):
            with self.subTest(action=action):
                self.assert_code("unknown_instance", lambda: self.call(
                    action, instance_id="missing", text="hello", turn_id="wrong"
                ))
        self.assertTrue(node._text_queue.empty())
        self.plugin._ensure_adapter.assert_not_called()

    def test_start_fixed_mode_uses_setting_without_changing_other_instance(self):
        node = self.start("a", reply_language="zh")
        self.start("b", reply_language="en")
        self.assertEqual(self.begin("hello", "a")["reply_language"], "zh")
        self.assertEqual(self.begin("你好", "b")["reply_language"], "en")
        self.assertEqual(node.policy.snapshot()["reply_language"], "zh")

    def test_invalid_start_mode_does_not_load_adapter(self):
        self.assert_code("invalid_reply_language", lambda: self.start(reply_language="fr"))
        self.plugin._ensure_adapter.assert_not_called()
        self.assertEqual(self.plugin._nodes, {})

    def test_idempotent_start_without_topic_preserves_topic_node_and_turn(self):
        node = self.start()
        turn = self.begin("hello")
        self.call("start", instance_id="a")
        self.assertIs(self.plugin._nodes["a"], node)
        self.assertEqual(node._input_topic, "/input/a")
        self.assertEqual(node.policy.snapshot()["turn_id"], turn["turn_id"])

    def test_invalid_input_topic_does_not_load_adapter(self):
        with self.assertRaises(ValueError):
            self.call("start", instance_id="a", input_topic=["/input/a"])
        self.plugin._ensure_adapter.assert_not_called()
        self.assertEqual(self.plugin._nodes, {})

    def test_dispatch_switches_and_queues_frozen_matching_replies(self):
        node = self.start()
        for utterance, answer, expected in (
            ("hello", "Hello.", "en"), ("你好", "你好。", "zh"),
            ("hello again", "Welcome back.", "en"),
        ):
            result = self.begin(utterance)
            response = self.call("speak", instance_id="a", text=answer, turn_id=result["turn_id"])
            self.assertEqual(response["status"], "queued")
            request = node._text_queue.get_nowait()
            self.assertIsInstance(request, self.policy_module.SpeechRequest)
            self.assertEqual((request.text, request.language), (answer, expected))

    def test_dispatch_wrong_language_never_reaches_queue(self):
        node = self.start()
        turn = self.begin("hello")
        self.assert_code("reply_language_mismatch", lambda: self.call(
            "speak", instance_id="a", text="你好", turn_id=turn["turn_id"]
        ))
        self.assertTrue(node._text_queue.empty())

    def test_raw_numeric_reply_never_reaches_frontend_queue_or_synthesis(self):
        node = self.start()
        turn = self.begin("你好")
        self.assert_code("ambiguous_reply_language", lambda: self.call(
            "speak", instance_id="a", text="123", turn_id=turn["turn_id"]
        ))
        self.assertEqual(self.adapter.frontend_inputs, [])
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(self.adapter.spoken, [])

    def test_frontend_language_switch_rejects_entire_request_before_queue(self):
        node = self.start()
        turn = self.begin("hello")
        self.adapter.normalized_chunks = ("价格为", "十二美元")
        self.assert_code("frontend_language_mismatch", lambda: self.call(
            "speak", instance_id="a", text="The price is $12.", turn_id=turn["turn_id"]
        ))
        self.assertEqual(self.adapter.frontend_inputs, ["The price is $12."])
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(self.adapter.spoken, [])

    def test_raw_primary_reply_can_normalize_to_supported_mixed_speech(self):
        node = self.start()
        turn = self.begin("hello")
        self.adapter.normalized_chunks = ("The price is", "十二美元")
        self.call("speak", instance_id="a", text="The price is $12.", turn_id=turn["turn_id"])
        request = node._text_queue.get_nowait()
        self.assertEqual(request.language, "en")
        self.assertEqual(tuple(chunk for chunk, _ in request.chunks), self.adapter.normalized_chunks)

    def test_mixed_reply_can_have_separate_chinese_and_english_chunks(self):
        node = self.start()
        for user, reply, chunks, language in (
            ("你好", "Please wait here.请稍等。", ("Please wait here.", "请稍等。"), "zh"),
            ("hello", "北京。Welcome to the city.", ("北京。", "Welcome to the city."), "en"),
        ):
            with self.subTest(language=language):
                turn = self.begin(user)
                self.adapter.normalized_chunks = chunks
                self.call("speak", instance_id="a", text=reply, turn_id=turn["turn_id"])
                request = node._text_queue.get_nowait()
                self.assertEqual((request.text, request.language), (reply, language))
                self.assertEqual(tuple(chunk for chunk, _ in request.chunks), chunks)
                self.assertEqual(self.adapter.spoken, [])

    def test_mixed_reply_cannot_lose_primary_language_in_all_normalized_chunks(self):
        node = self.start()
        for user, reply, chunks in (
            ("你好", "你好，Hello.", ("Hello.", "Welcome.")),
            ("hello", "Hello，你好。", ("你好。", "欢迎。")),
        ):
            with self.subTest(user=user):
                turn = self.begin(user)
                self.adapter.normalized_chunks = chunks
                self.assert_code("frontend_language_mismatch", lambda: self.call(
                    "speak", instance_id="a", text=reply, turn_id=turn["turn_id"],
                ))
                self.assertTrue(node._text_queue.empty())
                self.assertEqual(self.adapter.spoken, [])

    def test_mixed_normalized_reply_rejects_unsupported_script_in_any_chunk(self):
        node = self.start()
        turn = self.begin("你好")
        self.adapter.normalized_chunks = ("你好。", "Hello.", "Привет")
        self.assert_code("frontend_language_mismatch", lambda: self.call(
            "speak", instance_id="a", text="你好，Hello.", turn_id=turn["turn_id"],
        ))
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(self.adapter.spoken, [])

    def test_topic_accepts_mixed_reply_with_separate_language_chunks(self):
        node = self.start()
        turn = self.begin("你好")
        text = "你好，welcome to 北京。"
        self.adapter.normalized_chunks = ("你好。", "welcome to", "北京。")
        node._text_callback(types.SimpleNamespace(data=json.dumps({
            "text": text, "turn_id": turn["turn_id"],
        })))
        request = node._text_queue.get_nowait()
        self.assertEqual((request.text, request.language), (text, "zh"))
        self.assertIsNone(node.policy.snapshot()["last_language_error"])

    def test_topic_also_rejects_frontend_language_switch(self):
        node = self.start()
        turn = self.begin("hello")
        self.adapter.normalized_chunks = ("十二美元",)
        node._text_callback(types.SimpleNamespace(data=json.dumps({
            "text": "The price is $12.", "turn_id": turn["turn_id"],
        })))
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(node.policy.snapshot()["last_language_error"]["code"], "frontend_language_mismatch")
        self.assertEqual(self.adapter.spoken, [])

    def test_frontend_neutral_chunk_allowed_beside_matching_language(self):
        node = self.start()
        turn = self.begin("hello")
        self.adapter.normalized_chunks = ("Hello", "123.")
        self.call("speak", instance_id="a", text="Hello 123.", turn_id=turn["turn_id"])
        self.assertEqual(node._text_queue.get_nowait().language, "en")
        self.assertEqual(self.adapter.spoken, [])

    def test_frontend_empty_or_neutral_output_rejected(self):
        node = self.start()
        turn = self.begin("hello")
        for normalized in ((), ("123",), ("?", "!")):
            self.adapter.normalized_chunks = normalized
            with self.subTest(normalized=normalized):
                self.assert_code("frontend_language_mismatch", lambda: self.call(
                    "speak", instance_id="a", text="Hello", turn_id=turn["turn_id"]
                ))
                self.assertTrue(node._text_queue.empty())
                self.assertEqual(self.adapter.spoken, [])

    def test_frontend_failure_is_sanitized_and_never_queues_audio(self):
        node = self.start()
        turn = self.begin("hello")
        self.adapter.iter_text_chunks = Mock(side_effect=ValueError("private input details"))
        with self.assertRaises(self.policy_module.LanguagePolicyError) as caught:
            self.call("speak", instance_id="a", text="Hello", turn_id=turn["turn_id"])
        self.assertEqual(caught.exception.code, "frontend_validation_failed")
        self.assertNotIn("private", str(caught.exception))
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(self.adapter.spoken, [])

    def test_second_instance_cannot_use_first_instance_turn(self):
        a, b = self.start("a"), self.start("b")
        first, second = self.begin("hello", "a"), self.begin("你好", "b")
        self.assert_code("stale_turn", lambda: self.call(
            "speak", instance_id="b", text="你好", turn_id=first["turn_id"]
        ))
        self.call("speak", instance_id="b", text="你好", turn_id=second["turn_id"])
        self.assertTrue(a._text_queue.empty())
        self.assertEqual(b._text_queue.get_nowait().language, "zh")

    def test_topic_requires_json_turn_and_obeys_same_language_guard(self):
        node = self.start()
        turn = self.begin("hello")
        for payload in ("Hello", json.dumps({"text": "Hello"}),
                        json.dumps({"text": "你好", "turn_id": turn["turn_id"]})):
            node._text_callback(types.SimpleNamespace(data=payload))
            self.assertTrue(node._text_queue.empty())
            self.assertIsNotNone(node.policy.snapshot()["last_language_error"])
        node._text_callback(types.SimpleNamespace(data=json.dumps({
            "text": "Hello from topic.", "turn_id": turn["turn_id"],
        })))
        request = node._text_queue.get_nowait()
        self.assertEqual((request.text, request.language), ("Hello from topic.", "en"))

    def test_topic_errors_are_queryable_and_do_not_log_input(self):
        node = self.start()
        turn = self.begin("hello")
        sensitive_text = "私密内容不得进入日志"
        with self.assertLogs(self.module.log.name, level="WARNING") as captured:
            node._text_callback(types.SimpleNamespace(data=json.dumps({
                "text": sensitive_text, "turn_id": turn["turn_id"],
            })))
        self.assertNotIn(sensitive_text, "\n".join(captured.output))
        result = self.call("info", instance_id="a")
        self.assertEqual(result["last_language_error"]["code"], "reply_language_mismatch")
        self.assertTrue(node._text_queue.empty())

    def test_topic_stale_turn_does_not_cancel_current_reply(self):
        node = self.start()
        first = self.begin("hello")
        second = self.begin("你好")
        node._text_callback(types.SimpleNamespace(data=json.dumps({
            "text": "Hello", "turn_id": first["turn_id"],
        })))
        self.assertTrue(node._text_queue.empty())
        self.assertEqual(node.policy.snapshot()["turn_id"], second["turn_id"])
        node._text_callback(types.SimpleNamespace(data=json.dumps({
            "text": "你好", "turn_id": second["turn_id"],
        })))
        self.assertEqual(node._text_queue.get_nowait().language, "zh")

    def test_queued_request_retains_language_after_switch_and_proxy_unwraps_text(self):
        node = self.start()
        first = self.begin("hello")
        self.call("speak", instance_id="a", text="Hello.", turn_id=first["turn_id"])
        request = node._text_queue.get_nowait()
        self.begin("你好")
        self.assertEqual(request.language, "en")
        list(node._adapter.synthesize_stream(request))
        self.assertEqual(self.adapter.spoken, ["Hello."])

    def test_plain_string_cannot_bypass_request_adapter_or_raw_api(self):
        node = self.start()
        self.assert_code("unvalidated_request", lambda: list(node._adapter.synthesize_stream("Hello")))
        self.assert_code("session_required", lambda: self.plugin.synthesize_raw("Hello"))
        self.assertEqual(self.adapter.spoken, [])

    def test_unprepared_policy_request_cannot_bypass_frontend_validation(self):
        node = self.start()
        turn = self.begin("hello")
        unprepared = node.policy.prepare("Hello", turn["turn_id"])
        self.assert_code(
            "unvalidated_request", lambda: list(node._adapter.synthesize_stream(unprepared))
        )
        self.assertEqual(self.adapter.spoken, [])

    def test_prepared_frontend_matches_original_adapter_bytes_and_calls(self):
        original = _load_original_adapter()
        text = "Hello. Please wait here."
        chunks = {text: (
            ("Hello.", ((1, 2, 3), (0, 0, 0), (1, 1, 1))),
            ("Please wait here.", ((8, 9), (2, 2), (1, 1))),
        )}
        baseline, baseline_engine, baseline_frontend = _recording_adapter(original, chunks, 0.9)
        expected = list(baseline.synthesize_stream(text))
        self.adapter, engine, frontend = _recording_adapter(original, chunks, 0.9)
        self.plugin._adapter = self.adapter
        self.plugin._ensure_adapter.return_value = self.adapter
        node = self.start()
        turn = self.begin("hello")
        self.call("speak", instance_id="a", text=text, turn_id=turn["turn_id"])
        request = node._text_queue.get_nowait()
        self.assertEqual(frontend, [text])
        actual = list(node._adapter.synthesize_stream(request))
        self.assertEqual(frontend, [text], "Synthesis must not repeat TN/G2P")
        self.assertEqual(baseline_frontend, [text])
        self.assertEqual(actual, expected)
        self.assertEqual(engine.calls, baseline_engine.calls)
        self.assertEqual(engine.reset_count, baseline_engine.reset_count)
        self.assertTrue(all(call[3] for call in engine.calls))

    def test_prepared_chunks_are_isolated_between_instances_and_turn_changes(self):
        original = _load_original_adapter()
        chunks = {
            "Hello.": (("Hello.", ((1, 2), (0, 0), (1, 1))),),
            "你好。": (("你好。", ((7, 8, 9), (1, 2, 3), (0, 0, 0))),),
        }
        self.adapter, engine, frontend = _recording_adapter(original, chunks)
        self.plugin._adapter = self.adapter
        self.plugin._ensure_adapter.return_value = self.adapter
        a, b = self.start("a"), self.start("b")
        first, second = self.begin("hello", "a"), self.begin("你好", "b")
        self.call("speak", instance_id="a", text="Hello.", turn_id=first["turn_id"])
        self.call("speak", instance_id="b", text="你好。", turn_id=second["turn_id"])
        request_a, request_b = a._text_queue.get_nowait(), b._text_queue.get_nowait()
        self.begin("你好", "a")
        self.begin("hello", "b")
        list(b._adapter.synthesize_stream(request_b))
        list(a._adapter.synthesize_stream(request_a))
        self.assertEqual(frontend, ["Hello.", "你好。"])
        self.assertEqual([call[:2] for call in engine.calls], [
            chunks["你好。"][0], chunks["Hello."][0],
        ])
        self.assertEqual(engine.reset_count, 2)
        self.assertEqual((request_a.language, request_b.language), ("en", "zh"))

    def test_prepared_chunks_copy_mutable_ids_and_adapter_view_is_read_only(self):
        original = _load_original_adapter()
        ids = ([1, 2], [0, 0], [1, 1])
        self.adapter, engine, frontend = _recording_adapter(
            original, {"Hello.": (("Hello.", ids),)}
        )
        self.plugin._adapter = self.adapter
        self.plugin._ensure_adapter.return_value = self.adapter
        node = self.start()
        turn = self.begin("hello")
        self.call("speak", instance_id="a", text="Hello.", turn_id=turn["turn_id"])
        request = node._text_queue.get_nowait()
        ids[0][0] = 99
        self.assertEqual(request.chunks[0][1][0], (1, 2))
        with self.assertRaises(FrozenInstanceError):
            request.chunks = ()
        view = self.module._PreparedAdapterView(self.adapter, request)
        self.assertIs(view._lock, self.adapter._lock)
        self.assertIs(view._engine, engine)
        with self.assertRaises(FrozenInstanceError):
            view._speed = 4
        self.assert_code("prepared_text_mismatch", lambda: list(view.iter_text_chunks("Other.")))
        # Shared speed retains the original semantics: read it at synthesis,
        # while prepared linguistic inputs remain tied to this request.
        self.call("config", speed=0.8)
        list(node._adapter.synthesize_stream(request))
        self.assertEqual(engine.calls[0][1][0], (1, 2))
        self.assertEqual(engine.calls[0][2], 1.25)
        self.assertEqual(frontend, ["Hello."])

    def test_stopped_topic_ignores_input(self):
        node = self.start()
        turn = self.begin("hello")
        node.state = "idle"
        node._text_callback(types.SimpleNamespace(data=json.dumps({
            "text": "Hello", "turn_id": turn["turn_id"],
        })))
        self.assertTrue(node._text_queue.empty())

    def test_invalid_language_config_preserves_turn_and_speed(self):
        node = self.start()
        self.begin("hello")
        before = node.policy.snapshot()
        self.assert_code("invalid_reply_language", lambda: self.call(
            "config", instance_id="a", reply_language="xx"
        ))
        self.assertEqual(node.policy.snapshot(), before)
        self.assertEqual(self.adapter.speed, 1.0)

    def test_invalid_speed_config_preserves_turn_and_language(self):
        node = self.start()
        self.begin("hello")
        before = node.policy.snapshot()
        for speed in (-1, 0, 4.1, float("nan"), float("inf"), "fast", None):
            with self.subTest(speed=speed):
                self.assert_code("invalid_speed", lambda: self.call("config", speed=speed))
                self.assertEqual(node.policy.snapshot(), before)
                self.assertEqual(self.adapter.speed, 1.0)

    def test_merged_ui_config_accepts_shared_speed_and_instance_language(self):
        node = self.start()
        self.begin("hello")
        before = node.policy.snapshot()
        self.call("config", instance_id="a", reply_language="auto", speed=0.8, speaker_id=0)
        self.assertEqual(node.policy.snapshot(), before)
        self.assertIs(self.plugin._nodes["a"], node)
        self.assertEqual(self.adapter.speed, 0.8)

    def test_invalid_merged_ui_config_is_atomic(self):
        node = self.start()
        self.begin("hello")
        before, before_cfg = node.policy.snapshot(), dict(self.plugin._cfg)
        for fields in (
            {"reply_language": "fr", "speed": 0.8, "speaker_id": 0},
            {"reply_language": "zh", "speed": -1, "speaker_id": 0},
            {"reply_language": "zh", "speed": 0.8, "speaker_id": 1},
        ):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    self.call("config", instance_id="a", **fields)
                self.assertEqual(node.policy.snapshot(), before)
                self.assertEqual(self.plugin._cfg, before_cfg)
                self.assertEqual(self.adapter.speed_updates, [])
                self.assertIs(self.plugin._nodes["a"], node)

    def test_ui_merged_config_before_every_action_keeps_active_turn(self):
        config = {"instance_id": "a", "reply_language": "auto", "speed": 1.0, "speaker_id": 0}
        self.call("config", **config)
        self.plugin._ensure_adapter.assert_not_called()
        node = self.start()
        for utterance, reply, language in (("hello", "Hello.", "en"), ("你好", "你好。", "zh")):
            self.call("config", **config)
            turn = self.begin(utterance)
            self.call("config", **config)
            self.call("speak", instance_id="a", text=reply, turn_id=turn["turn_id"])
            self.assertIs(self.plugin._nodes["a"], node)
            self.assertEqual(node.policy.snapshot()["turn_id"], turn["turn_id"])
            self.assertEqual(node._text_queue.get_nowait().language, language)
        self.assertEqual(self.adapter.speed_updates, [])

    def test_prestart_fixed_language_config_is_applied_by_start(self):
        for key, mode, utterance in (("a", "zh", "hello"), ("b", "en", "你好")):
            self.call("config", instance_id=key, reply_language=mode, speed=1.0, speaker_id=0)
        self.plugin._ensure_adapter.assert_not_called()
        for key, mode, utterance in (("a", "zh", "hello"), ("b", "en", "你好")):
            self.start(key)
            self.assertEqual(self.begin(utterance, key)["reply_language"], mode)

    def test_invalid_prestart_config_cannot_cache_partial_language_setting(self):
        with self.assertRaises(ValueError):
            self.call("config", instance_id="a", reply_language="zh", speed=-1, speaker_id=0)
        self.plugin._ensure_adapter.assert_not_called()
        self.start()
        self.assertEqual(self.begin("hello")["reply_language"], "en")

    def test_changed_merged_mode_only_invalidates_target_instance(self):
        a, b = self.start("a"), self.start("b")
        first, second = self.begin("hello", "a"), self.begin("hello", "b")
        self.call("config", instance_id="b", reply_language="zh", speed=1.0, speaker_id=0)
        self.assertEqual(a.policy.snapshot()["turn_id"], first["turn_id"])
        self.assertIsNone(b.policy.snapshot()["turn_id"])
        self.assertIs(self.plugin._nodes["a"], a)
        self.assertIs(self.plugin._nodes["b"], b)
        self.assertEqual(self.begin("hello", "b")["reply_language"], "zh")
        self.assertEqual(a.policy.prepare("Hello", first["turn_id"]).language, "en")

    def test_shared_speed_change_preserves_all_nodes_and_turns(self):
        a, b = self.start("a"), self.start("b")
        first, second = self.begin("hello", "a"), self.begin("你好", "b")
        self.call("config", instance_id="a", speed=0.9, speaker_id=0)
        self.assertEqual(self.adapter.speed_updates, [0.9])
        self.assertIs(self.plugin._nodes["a"], a)
        self.assertIs(self.plugin._nodes["b"], b)
        self.assertEqual(a.policy.snapshot()["turn_id"], first["turn_id"])
        self.assertEqual(b.policy.snapshot()["turn_id"], second["turn_id"])

    def test_repeated_ui_config_preserves_already_queued_request(self):
        node = self.start()
        turn = self.begin("hello")
        self.call("speak", instance_id="a", text="Hello.", turn_id=turn["turn_id"])
        for _ in range(3):
            self.call("config", instance_id="a", reply_language="auto", speed=1.0, speaker_id=0)
        request = node._text_queue.get_nowait()
        self.assertEqual((request.text, request.turn_id, request.language), ("Hello.", turn["turn_id"], "en"))
        self.assertEqual(self.adapter.speed_updates, [])

    def test_stop_clears_cached_prestart_language_setting(self):
        self.call("config", instance_id="a", reply_language="zh", speed=1.0, speaker_id=0)
        self.start()
        self.assertEqual(self.begin("hello")["reply_language"], "zh")
        self.call("stop", instance_id="a")
        self.start()
        self.assertEqual(self.begin("hello")["reply_language"], "en")

    def test_language_configuration_is_per_instance(self):
        a, b = self.start("a"), self.start("b")
        first, second = self.begin("hello", "a"), self.begin("hello", "b")
        self.call("config", instance_id="a", reply_language="zh")
        self.assertIsNone(a.policy.snapshot()["turn_id"])
        self.assertEqual(b.policy.snapshot()["turn_id"], second["turn_id"])
        self.assertEqual(self.begin("hello", "a")["reply_language"], "zh")
        self.assertEqual(b.policy.prepare("Hello", second["turn_id"]).language, "en")

    def test_stop_restart_cannot_accept_previous_turn_token(self):
        first_node = self.start()
        first = self.begin("hello")
        self.call("stop", instance_id="a")
        second_node = self.start()
        second = self.begin("hello")
        self.assertIsNot(first_node, second_node)
        self.assertNotEqual(first["turn_id"], second["turn_id"])
        self.assert_code("stale_turn", lambda: self.call(
            "speak", instance_id="a", text="Hello", turn_id=first["turn_id"]
        ))
        self.assertTrue(second_node._text_queue.empty())

    def test_stopped_instance_cannot_prepare_or_speak(self):
        node = self.start()
        first = self.begin("hello")
        node.state = "idle"
        for action in ("language", "speak"):
            with self.subTest(action=action):
                self.assert_code("instance_not_running", lambda: self.call(
                    action, instance_id="a", text="Hello", turn_id=first["turn_id"]
                ))
        self.assertTrue(node._text_queue.empty())


if __name__ == "__main__":
    unittest.main(verbosity=2)
