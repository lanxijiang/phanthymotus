"""Standard start/speak compatibility around the opt-in language plugin.

Direct speech has no recognized user turn: it selects readings from the supplied
speech text. Once a caller explicitly invokes language, that node requires the
conversation token on every speak until the instance is removed.
"""

from ..vits2_tts_language.plugin import TTSPlugin as LanguageTTSPlugin
from ..vits2_tts_language.policy import _scripts
from ..vits2_tts_language.name_text import normalize_standalone_name


class TTSPlugin(LanguageTTSPlugin):
    PREFIX = "tts"

    def _create_node(self, key, input_topic, reply_language=None):
        node = super()._create_node(key, input_topic, reply_language)
        node._explicit_language_contract = False
        return node

    def get_tools(self):
        tools = super().get_tools()
        tool = tools[0]
        tool["description"] = (
            "VITS2 TensorRT TTS: start an instance, then speak supplied text. "
            "Direct speech does not require language or turn_id. Auto selects "
            "readings from the current speech text, with Chinese as the fallback. "
            "For dialogue, before composing the reply, call language with the current "
            "ASR message as asr_result={text, lang}, or the current user's text and "
            "optional input_language, and "
            "give its instruction to the dialogue model. After language succeeds, "
            "every speak on that instance requires its current turn_id until stop. "
            "The tool does not translate or generate replies."
        )
        properties = tool["inputSchema"]["properties"]
        properties["turn_id"]["description"] = (
            "Required after an explicit language call. Omit for direct speech "
            "before any language call; an invalid supplied token is always rejected."
        )
        properties["text"]["description"] = (
            "Text to pronounce for direct speak, or the dialogue reply after language. "
            "For language, supply the original recognized user text."
        )
        tool["configSchema"]["properties"]["reply_language"]["description"] = (
            "自动 / 中文 / English. In direct speech, auto uses the current speech "
            "text. In dialogue, language uses current-user prose or an optional "
            "ASR tag. Unclear input falls back to Chinese. This is not translation."
        )
        # Topic messages retain the strict JSON text/turn_id contract.
        return tools

    def dispatch(self, name, args):
        action = args.get("action") if name == "tts" else name
        if action == "config":
            # Darvin sends blank remote-service fields even for a local backend.
            # Treat only these known empty strings as absent; preserve supplied
            # settings for normal validation and leave the caller's dict intact.
            args = {
                key: value for key, value in args.items()
                if not (key in {"api_key", "model", "voice", "url"} and value == "")
            }
        with self._instances_lock:
            if action == "config" and not (args.keys() - {"action", "instance_id"}):
                # Standard clients may initialize with no optional settings.
                # Keep current defaults, nodes and turns; do not load the model.
                # Unknown or explicitly supplied settings retain normal validation.
                return {
                    "status": "configured",
                    "speed": float(self._cfg.get("speed", 1.0)),
                    "speaker_id": int(self._cfg.get("speaker_id", 0)),
                }

            if action == "language":
                node = self._node(args)
                with node.policy.lock:
                    result = super().dispatch(name, args)
                    node._explicit_language_contract = True
                    return result

            if action == "speak" and "turn_id" not in args:
                node = self._node(args)
                with node.policy.lock:
                    if not node._explicit_language_contract:
                        text = args.get("text")
                        turn = node.policy.begin_turn(text)
                        # Dialogue detection ignores quoted text and addresses.
                        # Direct TTS must still pronounce a wholly Latin quote
                        # or address. Preserve curated standalone Chinese names
                        # such as Qwen3 and the Chinese fallback for bare numbers.
                        if (turn["reply_language_mode"] == "auto"
                                and turn["language_source"] == "default"
                                and _scripts(normalize_standalone_name(text, "zh")) == {"en"}):
                            turn = node.policy.begin_turn(text, requested_reply_language="en")
                        result = super().dispatch(name, {**args, "turn_id": turn["turn_id"]})
                        result.update(
                            speech_mode="direct",
                            language_source=(
                                "speech_text" if turn["reply_language_mode"] == "auto" else "fixed"
                            ),
                        )
                        return result
            # Explicit null/empty/stale tokens, and tokenless dialogue requests,
            # retain the original validation. No fallback is attempted on error.
            return super().dispatch(name, args)
