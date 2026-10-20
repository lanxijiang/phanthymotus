"""Standard MCP compatibility with real policy/frontend helpers and boundary doubles.

Reuse the language suite's ROS/model boundaries. No ASR, dialogue model, acoustic
engine, network, or GPU is exercised by these tests.
"""

from dataclasses import FrozenInstanceError
import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import test_vits2_reply_language as boundaries


def _load_compat():
    policy, language = boundaries._load_modules()
    name = language.__name__.split(".")[0]
    root = Path(__file__).resolve().parents[1] / "plugins"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    language_package = types.ModuleType(f"{name}.vits2_tts_language")
    language_package.__path__ = [str(root / "vits2_tts_language")]
    names = types.ModuleType(f"{name}.vits2_tts_language.name_text")
    # Rebind the actual helper captured by the existing loader, without
    # importing a second ROS/model boundary or changing its implementation.
    names.normalize_standalone_name = language.normalize_standalone_name
    with patch.dict(sys.modules, {
        name: package,
        language_package.__name__: language_package,
        language.__name__: language,
        policy.__name__: policy,
        names.__name__: names,
    }):
        compat_package = importlib.import_module(f"{name}.vits2_tts_compat")
        compat = importlib.import_module(f"{name}.vits2_tts_compat.plugin")
        assert compat_package.TTSPlugin is compat.TTSPlugin
    return policy, language, compat


class StandardCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_module, cls.language_module, cls.module = _load_compat()

    def setUp(self):
        self.adapter = boundaries._FakeAdapter()
        self.plugin = self.module.TTSPlugin({"backend": "trt", "speed": 1.0}, Mock())
        self.plugin._adapter = self.adapter
        self.plugin._ensure_adapter = Mock(return_value=self.adapter)

        def start_without_worker(node):
            node.state = "running"
            return node.status()

        start = patch.object(self.language_module._LanguageTTSNode, "start", start_without_worker)
        start.start()
        self.addCleanup(start.stop)

    def call(self, action, **kwargs):
        return self.plugin.dispatch("tts", {"action": action, **kwargs})

    def start(self, instance_id="a", **kwargs):
        self.call("start", instance_id=instance_id, input_topic=f"/input/{instance_id}", **kwargs)
        return self.plugin._nodes[instance_id]

    def assert_code(self, code, operation):
        with self.assertRaises(self.policy_module.LanguagePolicyError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_schema_describes_direct_and_strict_modes_without_mutating_language_schema(self):
        tool = self.plugin.get_tools()[0]
        self.assertEqual((self.plugin.PREFIX, tool["name"]), ("tts", "tts"))
        properties = tool["inputSchema"]["properties"]
        self.assertIn("language", properties["action"]["enum"])
        self.assertNotIn("turn_id", tool["inputSchema"].get("required", []))
        self.assertIn("Omit for direct speech", properties["turn_id"]["description"])
        self.assertIn("Required after an explicit language call", properties["turn_id"]["description"])
        self.assertIn("Text to pronounce", properties["text"]["description"])
        self.assertIn("does not translate or generate replies", tool["description"])
        controls = tool["configSchema"]["properties"]
        self.assertEqual(controls["reply_language"]["enum"], ["自动", "中文", "English"])
        self.assertEqual(controls["reply_language"]["default"], "自动")
        self.assertEqual(controls["reply_language"]["scope"], controls["speed"]["scope"])
        self.assertEqual(controls["reply_language"]["scope"], "shared")
        self.assertIn("current speech", controls["reply_language"]["description"])
        self.assertIn("text and current turn_id", tool["topic_in"][0]["desc"])
        original = self.language_module.TTSPlugin({"backend": "trt"}, Mock()).get_tools()[0]
        self.assertEqual(original["inputSchema"]["properties"]["turn_id"]["description"],
                         "Required for speak; returned by action=language")
        self.call("config", reply_language="English")
        self.assertEqual(self.plugin.get_tools()[0]["configSchema"]["properties"]
                         ["reply_language"]["default"], "English")

    def test_empty_config_accepts_standard_and_alias_calls_before_start(self):
        plugin = self.module.TTSPlugin({"backend": "trt", "speed": 1.0}, Mock())
        self.assertEqual(plugin.get_tools()[0]["configSchema"]["required"], [])
        with patch.object(plugin, "_ensure_adapter") as ensure:
            for name, fields in (("tts", {"action": "config"}), ("config", {})):
                for instance in ({}, {"instance_id": "tts_eval_shared"}):
                    args = {**fields, **instance}
                    with self.subTest(name=name, args=args):
                        self.assertEqual(plugin.dispatch(name, args), {
                            "status": "configured", "speed": 1.0, "speaker_id": 0,
                        })
                        self.assertEqual(args, {**fields, **instance})
            ensure.assert_not_called()
        self.assertIsNone(plugin._adapter)
        self.assertEqual(plugin._nodes, {})
        self.assertEqual(plugin._reply_modes, {})

    def test_empty_config_preserves_current_defaults_and_load_error(self):
        self.call("config", speed=0.9, reply_language="English")
        self.plugin._load_error = "previous model load failure"
        cfg = dict(self.plugin._cfg)
        updates = list(self.adapter.speed_updates)
        self.plugin._ensure_adapter.reset_mock()
        result = self.call("config", instance_id="not_started")
        self.assertEqual(result, {
            "status": "configured", "speed": 0.9, "speaker_id": 0,
        })
        self.assertEqual(self.plugin._cfg, cfg)
        self.assertEqual(self.plugin._default_reply_language, "en")
        self.assertEqual(self.plugin._reply_modes, {})
        self.assertEqual(self.plugin._nodes, {})
        self.assertEqual(self.adapter.speed_updates, updates)
        self.assertEqual(self.plugin._load_error, "previous model load failure")
        self.plugin._ensure_adapter.assert_not_called()

    def test_darvin_empty_config_start_speak_stop_sequence(self):
        for instance in ("tts_eval_shared", "second", "third"):
            with self.subTest(instance=instance):
                self.assertEqual(self.call("config", instance_id=instance)["status"],
                                 "configured")
                node = self.start(instance)
                for text, language in (("你好。", "zh"), ("Hello.", "en")):
                    result = self.call("speak", instance_id=instance, text=text)
                    request = node._text_queue.get_nowait()
                    self.assertEqual(result["status"], "queued")
                    self.assertEqual((request.text, request.language), (text, language))
                self.call("stop", instance_id=instance)
                self.assertNotIn(instance, self.plugin._nodes)
                self.assertNotIn(instance, self.plugin._reply_modes)

    def test_empty_config_preserves_three_instance_dialogue_turns_and_queues(self):
        nodes = {}
        snapshots = {}
        turns = {}
        for instance, mode, text in (
            ("a", "中文", "你好。"), ("b", "English", "Hello."),
            ("c", "自动", "Hello again."),
        ):
            node = nodes[instance] = self.start(instance, reply_language=mode)
            turn = turns[instance] = self.call("language", instance_id=instance, text=text)
            self.call("speak", instance_id=instance, text=text, turn_id=turn["turn_id"])
            snapshots[instance] = node.policy.snapshot()
        cfg, modes = dict(self.plugin._cfg), dict(self.plugin._reply_modes)
        updates = list(self.adapter.speed_updates)
        self.plugin._ensure_adapter.reset_mock()
        for instance in (None, "a", "b", "c", "not_started"):
            fields = {} if instance is None else {"instance_id": instance}
            self.assertEqual(self.call("config", **fields)["status"], "configured")
        self.assertEqual(self.plugin._cfg, cfg)
        self.assertEqual(self.plugin._reply_modes, modes)
        self.assertEqual(set(self.plugin._nodes), set(nodes))
        self.assertEqual(self.adapter.speed_updates, updates)
        self.plugin._ensure_adapter.assert_not_called()
        for instance, node in nodes.items():
            self.assertIs(self.plugin._nodes[instance], node)
            self.assertEqual(node.state, "running")
            self.assertEqual(node.policy.snapshot(), snapshots[instance])
            self.assertTrue(node._explicit_language_contract)
            self.assertEqual(node._text_queue.qsize(), 1)
            request = node._text_queue.get_nowait()
            self.assertEqual(request.turn_id, turns[instance]["turn_id"])
            self.assert_code("stale_turn", lambda: self.call(
                "speak", instance_id=instance, text=request.text,
            ))
            self.assertEqual(self.call(
                "speak", instance_id=instance, text=request.text, turn_id=request.turn_id,
            )["status"], "queued")

    def test_empty_config_compatibility_keeps_invalid_configuration_checks(self):
        cfg = dict(self.plugin._cfg)
        for fields, code in (
            ({"speeed": 1.0}, "empty_config"),
            ({"model_type": "vits2"}, "empty_config"),
            ({"speed": 0}, "invalid_speed"),
            ({"speed": float("nan")}, "invalid_speed"),
            ({"speed": None}, "invalid_speed"),
            ({"speaker_id": 1}, "invalid_speaker"),
            ({"reply_language": "fr"}, "invalid_reply_language"),
        ):
            with self.subTest(fields=fields):
                self.assert_code(code, lambda: self.call(
                    "config", instance_id="tts_eval_shared", **fields,
                ))
                self.assertEqual(self.plugin._cfg, cfg)
                self.assertEqual(self.plugin._nodes, {})
                self.assertEqual(self.plugin._reply_modes, {})
        self.assertEqual(self.adapter.speed_updates, [])
        self.plugin._ensure_adapter.assert_not_called()

    def test_start_then_speak_needs_no_language_call_and_uses_current_text(self):
        node = self.start()
        requests = []
        for raw, language in (("Hello.", "en"), ("你好。", "zh"), ("Hello again.", "en")):
            result = self.call("speak", instance_id="a", text=raw)
            request = node._text_queue.get_nowait()
            self.assertEqual((result["status"], result["speech_mode"], result["language_source"]),
                             ("queued", "direct", "speech_text"))
            self.assertEqual((request.text, request.language, request.turn_id),
                             (raw, language, result["turn_id"]))
            self.assertEqual(self.adapter.frontend_inputs[-1], raw)
            self.assertFalse(node._explicit_language_contract)
            requests.append(request)
        self.assertEqual(len({r.turn_id for r in requests}), 3)
        self.assertEqual(requests[0].language, "en")
        with self.assertRaises(FrozenInstanceError):
            requests[0].language = "zh"
        self.assertEqual(self.adapter.spoken, [])

    def test_three_panel_preferences_select_numeric_date_and_qwen_readings(self):
        node = self.start()
        for setting, language, source in (
            ("自动", "zh", "speech_text"), ("中文", "zh", "fixed"), ("English", "en", "fixed"),
        ):
            self.call("config", reply_language=setting)
            for raw, chinese, english in (
                ("2026 10/ 20", "二零二六年十月二十日",
                 "October twentieth, two thousand twenty-six"),
                ("Qwen3", "千问三", "Qwen three"),
            ):
                with self.subTest(setting=setting, raw=raw):
                    result = self.call("speak", instance_id="a", text=raw)
                    request = node._text_queue.get_nowait()
                    spoken = chinese if language == "zh" else english
                    self.assertEqual((request.text, request.language), (raw, language))
                    self.assertEqual(result["language_source"], source)
                    self.assertEqual(self.adapter.frontend_inputs[-1], spoken)
                    self.assertEqual(tuple(text for text, _ in request.chunks), (spoken,))

    def test_three_instances_keep_preferences_tokens_and_contracts_independent(self):
        nodes = {key: self.start(key, reply_language=mode)
                 for key, mode in (("a", "English"), ("b", "中文"), ("c", "自动"))}
        tokens = {}
        for key, language in (("a", "en"), ("b", "zh"), ("c", "zh")):
            result = self.call("speak", instance_id=key, text="Qwen3")
            tokens[key] = result["turn_id"]
            self.assertEqual(nodes[key]._text_queue.get_nowait().language, language)
        self.assertEqual(len(set(tokens.values())), 3)
        self.call("language", instance_id="a", text="Hello")
        self.assert_code("stale_turn", lambda: self.call("speak", instance_id="a", text="Hello"))
        self.call("config", instance_id="b", reply_language="English")
        self.assertEqual(nodes["c"].policy.snapshot()["turn_id"], tokens["c"])
        for key, language in (("b", "en"), ("c", "zh")):
            self.call("speak", instance_id=key, text="Qwen3")
            self.assertEqual(nodes[key]._text_queue.get_nowait().language, language)
            self.assertFalse(nodes[key]._explicit_language_contract)

    def test_repeated_start_retains_node_and_stop_creates_fresh_direct_contract(self):
        old = self.start()
        turn = self.call("language", instance_id="a", text="Hello")
        self.assertIs(self.start(), old)
        self.assertEqual(old.policy.snapshot()["turn_id"], turn["turn_id"])
        self.assertTrue(old._explicit_language_contract)
        self.assert_code("stale_turn", lambda: self.call("speak", instance_id="a", text="Hello"))
        self.call("stop", instance_id="a")
        new = self.start()
        self.assertIsNot(new, old)
        self.assertFalse(new._explicit_language_contract)
        self.assert_code("language_turn_required", lambda: self.call(
            "speak", instance_id="a", text="Hello", turn_id=turn["turn_id"],
        ))
        self.call("speak", instance_id="a", text="Hello")
        self.assertEqual(new._text_queue.get_nowait().language, "en")

    def test_external_language_latches_strict_after_direct_speech_and_config(self):
        node = self.start()
        direct = self.call("speak", instance_id="a", text="Hello")
        node._text_queue.get_nowait()
        turn = self.call("language", instance_id="a", text="Hello")
        for fields in ({}, {"turn_id": "wrong"}, {"turn_id": direct["turn_id"]}):
            with self.subTest(fields=fields):
                self.assert_code("stale_turn", lambda: self.call(
                    "speak", instance_id="a", text="Hello", **fields,
                ))
                self.assertEqual(node.policy.snapshot()["turn_id"], turn["turn_id"])
        self.call("config", reply_language="auto", speed=0.9)
        self.assert_code("stale_turn", lambda: self.call("speak", instance_id="a", text="Hello"))
        self.call("config", reply_language="中文")
        self.assertTrue(node._explicit_language_contract)
        self.assert_code("language_turn_required", lambda: self.call(
            "speak", instance_id="a", text="你好",
        ))
        self.assertTrue(node._text_queue.empty())
        turn = self.call("language", instance_id="a", text="你好")
        result = self.call("speak", instance_id="a", text="你好", turn_id=turn["turn_id"])
        self.assertNotIn("speech_mode", result)
        self.assertEqual(node._text_queue.get_nowait().language, "zh")

    def test_explicit_null_empty_or_wrong_tokens_never_create_a_direct_turn(self):
        node = self.start()
        for token in (None, "", "wrong", 0):
            with self.subTest(token=token, before_turn=True):
                self.assert_code("language_turn_required", lambda: self.call(
                    "speak", instance_id="a", text="Hello", turn_id=token,
                ))
                self.assertIsNone(node.policy.snapshot()["turn_id"])
        direct = self.call("speak", instance_id="a", text="Hello")
        node._text_queue.get_nowait()
        for token in (None, "", "wrong", 0):
            with self.subTest(token=token, before_turn=False):
                self.assert_code("stale_turn", lambda: self.call(
                    "speak", instance_id="a", text="Hello", turn_id=token,
                ))
                self.assertEqual(node.policy.snapshot()["turn_id"], direct["turn_id"])
        self.assertEqual(self.adapter.frontend_inputs, ["Hello"])
        self.assertTrue(node._text_queue.empty())

    def test_failed_language_does_not_latch_direct_node_or_clear_strict_node(self):
        node = self.start()
        for fields, code in (({"text": ""}, "empty_text"),
                             ({"text": "Hello", "input_language": "fr"}, "invalid_input_language")):
            self.assert_code(code, lambda: self.call("language", instance_id="a", **fields))
            self.assertFalse(node._explicit_language_contract)
        self.call("speak", instance_id="a", text="Hello")
        node._text_queue.get_nowait()
        turn = self.call("language", instance_id="a", text="Hello")
        self.assert_code("empty_text", lambda: self.call("language", instance_id="a", text=""))
        self.assertTrue(node._explicit_language_contract)
        self.assertEqual(node.policy.snapshot()["turn_id"], turn["turn_id"])
        self.assert_code("stale_turn", lambda: self.call("speak", instance_id="a", text="Hello"))

    def test_asr_dialogue_contract_is_opt_in_and_preserves_direct_benchmark_instance(self):
        conversation, benchmark = self.start("conversation"), self.start("benchmark")
        event = {"text": "你好，can you check the robot?", "lang": "en"}
        turn = self.call("language", instance_id="conversation", asr_result=event)
        self.assertEqual((turn["reply_language"], turn["language_source"]), ("en", "asr"))
        self.assertIn("mainly in English", turn["instruction"])
        self.assertEqual(self.adapter.frontend_inputs, [])
        self.assertTrue(conversation._explicit_language_contract)
        self.assertFalse(benchmark._explicit_language_contract)
        # The caller would give instruction to its dialogue model here. This
        # test supplies that boundary's reply; it does not run or simulate an LLM.
        reply = "The robot is ready. 状态正常。"
        self.assert_code("stale_turn", lambda: self.call(
            "speak", instance_id="conversation", text=reply,
        ))
        accepted = self.call("speak", instance_id="conversation", text=reply,
                             turn_id=turn["turn_id"])
        queued = conversation._text_queue.get_nowait()
        self.assertEqual((accepted["status"], queued.text, queued.language, queued.turn_id),
                         ("queued", reply, "en", turn["turn_id"]))
        self.assertNotIn("speech_mode", accepted)
        direct = self.call("speak", instance_id="benchmark", text="这是榜单测试文本。")
        queued = benchmark._text_queue.get_nowait()
        self.assertEqual((direct["speech_mode"], queued.language), ("direct", "zh"))
        self.assertFalse(benchmark._explicit_language_contract)
        self.assertEqual(self.adapter.spoken, [])

    def test_invalid_asr_result_preserves_direct_mode_and_does_not_clear_dialogue_turn(self):
        node = self.start()
        direct = self.call("speak", instance_id="a", text="Hello.")
        node._text_queue.get_nowait()
        for event, code in ((None, "invalid_asr_result"),
                            ({"text": "Hello", "lang": []}, "invalid_asr_language")):
            with self.subTest(event=event):
                self.assert_code(code, lambda: self.call(
                    "language", instance_id="a", asr_result=event,
                ))
                self.assertFalse(node._explicit_language_contract)
                self.assertEqual(node.policy.snapshot()["turn_id"], direct["turn_id"])
                self.assertTrue(node._text_queue.empty())
        resumed = self.call("speak", instance_id="a", text="仍然可以直接播报。")
        self.assertEqual(resumed["speech_mode"], "direct")
        node._text_queue.get_nowait()
        turn = self.call("language", instance_id="a",
                         asr_result={"text": "Hello", "lang": "en"})
        self.assert_code("conflicting_input_language", lambda: self.call(
            "language", instance_id="a", input_language="zh",
            asr_result={"text": "Hello", "lang": "en"},
        ))
        self.assertTrue(node._explicit_language_contract)
        self.assertEqual(node.policy.snapshot()["turn_id"], turn["turn_id"])
        self.assert_code("stale_turn", lambda: self.call(
            "speak", instance_id="a", text="Hello again.",
        ))
        self.call("speak", instance_id="a", text="Hello again.", turn_id=turn["turn_id"])
        self.assertEqual(node._text_queue.get_nowait().language, "en")

    def test_direct_english_quotes_and_addresses_use_speech_not_asr(self):
        node = self.start()
        for raw in ('"Hello there."', "'Hello there.'", "https://example.com/Qwen3",
                    "Qwen3@example.com"):
            with self.subTest(raw=raw), patch.object(
                node.policy, "begin_turn", wraps=node.policy.begin_turn,
            ) as begin:
                result = self.call("speak", instance_id="a", text=raw,
                                   input_language="zh", requested_reply_language="zh")
                request = node._text_queue.get_nowait()
                self.assertEqual((request.text, request.language), (raw, "en"))
                self.assertEqual((result["speech_mode"], result["language_source"]),
                                 ("direct", "speech_text"))
                self.assertEqual(self.adapter.frontend_inputs[-1], raw)
                for call in begin.call_args_list:
                    self.assertEqual(call.args, (raw,))
                    self.assertNotIn("input_language", call.kwargs)
                self.assertFalse(node._explicit_language_contract)

    def test_fixed_language_does_not_translate_opposite_language_speech(self):
        node = self.start(reply_language="中文")
        self.assert_code("reply_language_mismatch", lambda: self.call(
            "speak", instance_id="a", text="This is an English sentence.",
        ))
        self.assertEqual(self.adapter.frontend_inputs, [])
        self.assertTrue(node._text_queue.empty())
        self.assertFalse(node._explicit_language_contract)

    def test_topic_stays_strict_and_never_enters_direct_speech(self):
        node = self.start()
        node._text_callback(types.SimpleNamespace(data=json.dumps({"text": "Hello"})))
        self.assertEqual(node.policy.snapshot()["last_language_error"]["code"],
                         "language_turn_required")
        self.assertIsNone(node.policy.snapshot()["turn_id"])
        self.assertTrue(node._text_queue.empty())
        self.assertFalse(node._explicit_language_contract)
        self.assertEqual(self.adapter.frontend_inputs, [])


if __name__ == "__main__":
    unittest.main()
