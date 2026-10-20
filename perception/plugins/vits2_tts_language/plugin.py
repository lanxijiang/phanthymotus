"""Opt-in reply language contract, reusing the existing VITS2 TRT worker.

The caller sends recognized user text to action=language, gives the returned
instruction to its dialogue model, and submits that model's reply with turn_id.
This plugin never performs ASR, dialogue generation or translation itself.
"""

import copy
from dataclasses import dataclass
import json
import logging
import math
import threading

from ..vits2_tts_trt.plugin import (
    TOOLS as BASE_TOOLS,
    TTSPlugin as BaseTTSPlugin,
    _Vits2TTSNode,
)
from .policy import (
    LanguagePolicy, LanguagePolicyError, SpeechRequest, validate_reply_script,
)
from .neutral_text import InvalidDateError, is_neutral_numeric, normalize_neutral_text
from .name_text import normalize_standalone_name


log = logging.getLogger(__name__)


_REPLY_LANGUAGE_LABELS = {"auto": "自动", "zh": "中文", "en": "English"}
_REPLY_LANGUAGE_ALIASES = {label: mode for mode, label in _REPLY_LANGUAGE_LABELS.items()}
_ASR_LANGUAGE_ALIASES = {
    "zh": "zh", "zh-cn": "zh", "zh-tw": "zh",
    "zh-hans": "zh", "zh-hant": "zh",
    "en": "en", "en-us": "en", "en-gb": "en", "en-au": "en",
}


def _language_input(args):
    """Read one caller-supplied ASR result without retaining or mutating it."""
    if "asr_result" not in args:
        return args.get("text"), args.get("input_language")
    result = args["asr_result"]
    if not isinstance(result, dict):
        raise LanguagePolicyError("invalid_asr_result", "asr_result must be an object")
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise LanguagePolicyError(
            "invalid_asr_result", "asr_result.text must be a nonempty string"
        )
    if "text" in args and (
        not isinstance(args["text"], str) or args["text"].strip() != text.strip()
    ):
        raise LanguagePolicyError(
            "conflicting_user_text", "text and asr_result.text must describe the same input"
        )
    # Main's SenseVoice publisher exposes the detected tag as `lang`.
    # `language` is a configuration value, not evidence about this utterance.
    raw_language = result.get("lang")
    if raw_language is not None and not isinstance(raw_language, str):
        raise LanguagePolicyError("invalid_asr_language", "asr_result.lang must be a string")
    detected = _ASR_LANGUAGE_ALIASES.get(
        (raw_language or "").strip().lower().replace("_", "-")
    )
    supplied = args.get("input_language")
    if supplied is not None and supplied not in ("zh", "en"):
        raise LanguagePolicyError("invalid_input_language", "input_language must be zh or en")
    if supplied is not None and detected is not None and supplied != detected:
        raise LanguagePolicyError(
            "conflicting_input_language", "input_language conflicts with asr_result.lang"
        )
    # Missing/unknown tags follow the existing text detection and Chinese
    # fallback. A valid explicit label remains supported for legacy callers.
    return text, supplied if supplied is not None else detected


def _reply_language_mode(value):
    """Keep display labels at the plugin boundary and canonical modes in policy."""
    if isinstance(value, str):
        value = _REPLY_LANGUAGE_ALIASES.get(value, value)
    LanguagePolicy(value)  # Validate without loading a model.
    return value


@dataclass(frozen=True)
class _PreparedSpeechRequest(SpeechRequest):
    """Validated frontend output owned by exactly one queued request."""

    chunks: tuple


@dataclass(frozen=True)
class _PreparedAdapterView:
    """Read-only adapter view replacing only repeated frontend preparation.

    The original synthesis method still acquires the shared adapter's lock,
    resets the same engine's RNG, reads its current speed and performs exactly
    the same chunk prefetch and PCM framing. No shared attribute is replaced.
    """

    _adapter: object
    _request: _PreparedSpeechRequest

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def iter_text_chunks(self, text):
        if text != self._request.text:
            raise LanguagePolicyError(
                "prepared_text_mismatch", "Prepared speech belongs to a different request"
            )
        yield from self._request.chunks


class _RequestAdapter:
    """Reuse each request's validated frontend output in the original adapter."""

    def __init__(self, adapter):
        self._adapter = adapter

    def synthesize_stream(self, request):
        if not isinstance(request, _PreparedSpeechRequest):
            raise LanguagePolicyError(
                "unvalidated_request", "The language policy must validate queued speech"
            )
        view = _PreparedAdapterView(self._adapter, request)
        yield from type(self._adapter).synthesize_stream(view, request.text)


class _LanguageTTSNode(_Vits2TTSNode):
    def __init__(self, input_topic, adapter, node_suffix="", reply_language="auto"):
        self.policy = LanguagePolicy(reply_language)
        self._inference_adapter = adapter
        super().__init__(input_topic, _RequestAdapter(adapter), node_suffix)

    def enqueue(self, text, turn_id=None):
        with self.policy.lock:
            try:
                if self.state != "running":
                    raise LanguagePolicyError(
                        "instance_not_running", "Start this TTS instance before speaking"
                    )
                request = self.policy.prepare(text, turn_id)
                # Inspect the real, unchanged frontend before accepting speech.
                # Individual chunks may use either supported language. Only the
                # combined speech must retain the selected primary language.
                normalized_parts = []
                prepared_chunks = []
                try:
                    frontend_text = (
                        normalize_neutral_text(text, request.language)
                        if is_neutral_numeric(text)
                        else normalize_standalone_name(text, request.language)
                    )
                    for normalized, text_ids in self._inference_adapter.iter_text_chunks(frontend_text):
                        validate_reply_script(
                            normalized, request.language,
                            allow_neutral=True, frontend=True, require_primary=False,
                        )
                        normalized_parts.append(normalized)
                        # The TRT frontend already returns tuples. Freeze other
                        # sequence implementations too, avoiding references to
                        # mutable producer buffers or another request's IDs.
                        prepared_chunks.append((
                            normalized, tuple(tuple(values) for values in text_ids),
                        ))
                except LanguagePolicyError:
                    raise
                except InvalidDateError as exc:
                    raise LanguagePolicyError("invalid_date", str(exc)) from None
                except Exception:
                    raise LanguagePolicyError(
                        "frontend_validation_failed",
                        "The frontend could not validate this speech request",
                    ) from None
                validate_reply_script(
                    " ".join(normalized_parts), request.language, frontend=True,
                )
                request = _PreparedSpeechRequest(
                    request.text, request.language, request.turn_id,
                    tuple(prepared_chunks),
                )
                # Validation and queue insertion share the same lock as a
                # language switch. Existing queued requests are never rewritten.
                self._text_queue.put(request)
                self.policy.clear_error()
                return request
            except LanguagePolicyError as exc:
                self.policy.record_error(exc)
                raise

    def _text_callback(self, message):
        try:
            try:
                payload = json.loads(message.data)
            except (ValueError, TypeError, AttributeError):
                raise LanguagePolicyError(
                    "structured_message_required",
                    "The input topic requires a JSON object with text and turn_id",
                ) from None
            if not isinstance(payload, dict):
                raise LanguagePolicyError(
                    "structured_message_required",
                    "The input topic requires a JSON object with text and turn_id",
                )
            self.enqueue(payload.get("text"), payload.get("turn_id"))
        except LanguagePolicyError as exc:
            self.policy.record_error(exc)
            # Topic publication has no RPC response. Preserve a queryable error
            # in info/status, and report the code without logging utterance text.
            log.warning("Language-aware TTS topic request rejected: %s", exc.code)

    def stop(self):
        with self.policy.lock:
            return super().stop()

    def status(self):
        return {**super().status(), **self.policy.snapshot()}


class TTSPlugin(BaseTTSPlugin):
    PREFIX = "tts"

    def __init__(self, plugin_cfg, executor):
        cfg = dict(plugin_cfg)
        cfg.setdefault("backend", "trt")
        if cfg["backend"] != "trt":
            raise ValueError("The language-aware plugin requires backend=trt")
        self._default_reply_language = _reply_language_mode(cfg.get("reply_language", "auto"))
        cfg["reply_language"] = self._default_reply_language
        self._instances_lock = threading.RLock()
        self._reply_modes = {}
        super().__init__(cfg, executor)

    def get_tools(self):
        tools = copy.deepcopy(BASE_TOOLS)
        tool = tools[0]
        tool["description"] = (
            "VITS2 TRT with a shared reply-language preference, per-instance turns and Chinese-English "
            "mixing. Before composing a dialogue reply, send the current user's "
            "ASR result {text, lang} as asr_result to language, or supply user text "
            "and optional input_language. "
            "Then give its instruction to the dialogue model and pass the returned "
            "turn_id with that reply. The dialogue caller interprets explicit user "
            "language requests and supplies requested_reply_language; this tool "
            "does not understand arbitrary instructions or retain dialogue history."
        )
        properties = tool["inputSchema"]["properties"]
        properties["action"]["enum"].append("language")
        properties.update({
            "instance_id": {"type": "string", "description": "Conversation instance ID"},
            "asr_result": {
                "type": "object",
                "description": (
                    "Current recognized user message for action=language, forwarded by the caller. "
                    "Reads text and detected lang only; a language configuration field is ignored. "
                    "No ASR subscription or reply generation is performed by this tool."
                ),
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "lang": {
                        "type": "string",
                        "description": "Detected ASR language, normally zh or en; unknown tags use text fallback",
                    },
                },
                "required": ["text"],
                "additionalProperties": True,
            },
            "input_language": {
                "type": "string", "enum": ["zh", "en"],
                "description": "Optional whole-turn ASR language for action=language",
            },
            "requested_reply_language": {
                "type": "string", "enum": ["zh", "en"],
                "description": (
                    "Optional explicit user reply-language request interpreted by the dialogue caller. "
                    "For this language action only, overrides configured preference and ASR; "
                    "does not change configuration or apply to later turns"
                ),
            },
            "turn_id": {
                "type": "string",
                "description": "Required for speak; returned by action=language",
            },
            "reply_language": {
                "type": "string", "enum": ["auto", "zh", "en", "自动", "中文", "English"],
                "description": (
                    "Reply-language preference: auto/自动, zh/中文 or en/English. "
                    "Chinese-English mixing is allowed. Config without instance_id updates "
                    "the shared default and all instances; with instance_id it updates only that instance"
                ),
            },
        })
        properties["text"]["description"] = (
            "Original recognized user text for language; dialogue reply text for speak"
        )
        tool["configSchema"]["properties"]["reply_language"] = {
            "type": "string", "title": "回复语言", "enum": list(_REPLY_LANGUAGE_LABELS.values()),
            "default": _REPLY_LANGUAGE_LABELS[self._default_reply_language], "scope": "shared",
            "description": (
                "Chinese/English preference; mixing is allowed and an explicit "
                "current-turn requested_reply_language takes priority. Auto uses "
                "ASR language first, then current-user prose grammar, with Chinese "
                "as the fallback; previous turns are never used for selection"
            ),
        }
        tool["topic_in"][0]["desc"] = "JSON object containing text and current turn_id"
        return tools

    @staticmethod
    def _instance_id(args):
        value = args.get("instance_id")
        if not isinstance(value, str) or not value.strip():
            raise LanguagePolicyError(
                "instance_required", "An explicit nonempty instance_id is required"
            )
        return value

    def _node(self, args):
        key = self._instance_id(args)
        node = self._nodes.get(key)
        if node is None:
            raise LanguagePolicyError(
                "unknown_instance", "Unknown TTS instance; call start first"
            )
        if node.state != "running":
            raise LanguagePolicyError(
                "instance_not_running", "Start this TTS instance before using it"
            )
        return node

    def _create_node(self, key, input_topic, reply_language=None):
        suffix = key.replace("/", "_").replace("-", "_")
        node = _LanguageTTSNode(
            input_topic or None, self._adapter, suffix,
            self._default_reply_language if reply_language is None else reply_language,
        )
        self._executor.add_node(node)
        self._nodes[key] = node
        return node

    def dispatch(self, name, args):
        action = args.get("action") if name == "tts" else name
        with self._instances_lock:
            if action == "start":
                key = self._instance_id(args)
                mode = _reply_language_mode(args.get(
                    "reply_language", self._reply_modes.get(key, self._default_reply_language)
                ))
                if "input_topic" in args and not isinstance(args["input_topic"], str):
                    raise LanguagePolicyError("invalid_input_topic", "input_topic must be a string")
                try:
                    self._ensure_adapter()
                except RuntimeError:
                    return {"state": "error", "message": self._load_error}
                node = self._nodes.get(key)
                input_topic = args.get("input_topic", node._input_topic if node else "")
                if node is not None and input_topic != node._input_topic:
                    self._remove_node(key)
                    node = None
                if node is None:
                    node = self._create_node(key, input_topic, mode)
                else:
                    node.policy.configure(mode)
                self._reply_modes[key] = mode
                return node.start()

            if action == "language":
                node = self._node(args)
                try:
                    text, input_language = _language_input(args)
                    return node.policy.begin_turn(
                        text, input_language,
                        args.get("requested_reply_language"),
                    )
                except LanguagePolicyError as exc:
                    node.policy.record_error(exc)
                    raise

            if action == "speak":
                node = self._node(args)
                request = node.enqueue(args.get("text"), args.get("turn_id"))
                return {
                    "status": "queued", "chars": len(request.text),
                    "reply_language": request.language, "turn_id": request.turn_id,
                }

            if action == "config":
                has_language = "reply_language" in args
                has_shared = "speed" in args or "speaker_id" in args
                if not has_language and not has_shared:
                    raise LanguagePolicyError("empty_config", "No supported configuration was supplied")
                key = None
                mode = None
                if has_language:
                    if "instance_id" in args:
                        key = self._instance_id(args)
                    mode = _reply_language_mode(args["reply_language"])
                # The product UI sends merged shared/instance settings before
                # start and before each action. Validate the entire payload
                # first, then apply only changes; never tear down active nodes.
                speed = float(self._cfg.get("speed", 1.0))
                if "speed" in args:
                    try:
                        speed = float(args["speed"])
                    except (TypeError, ValueError):
                        raise LanguagePolicyError("invalid_speed", "speed must be in (0, 4]") from None
                    if not math.isfinite(speed) or not 0 < speed <= 4:
                        raise LanguagePolicyError("invalid_speed", "speed must be in (0, 4]")
                if "speaker_id" in args and args["speaker_id"] != 0:
                    raise LanguagePolicyError("invalid_speaker", "This model supports only speaker_id=0")
                if "speed" in args and speed != float(self._cfg.get("speed", 1.0)):
                    if self._adapter is not None:
                        self._adapter.set_speed(speed)
                    self._cfg["speed"] = speed
                if "speaker_id" in args:
                    self._cfg["speaker_id"] = 0
                result = {
                    "status": "configured",
                    "speed": float(self._cfg.get("speed", 1.0)),
                    "speaker_id": int(self._cfg.get("speaker_id", 0)),
                }
                if has_language:
                    if key is None:
                        self._default_reply_language = mode
                        self._cfg["reply_language"] = mode
                        for instance in self._reply_modes.keys() | self._nodes.keys():
                            self._reply_modes[instance] = mode
                        for node in self._nodes.values():
                            node.policy.configure(mode)
                        result.update({"reply_language": mode, "default_reply_language": mode})
                    else:
                        self._reply_modes[key] = mode
                        node = self._nodes.get(key)
                        result.update(
                            node.policy.configure(mode) if node is not None else {
                                "reply_language": mode,
                                "resolved_language": None,
                                "turn_id": None,
                                "last_language_error": None,
                            }
                        )
                return result

            if action == "stop":
                result = super().dispatch(name, args)
                if args.get("instance_id"):
                    self._reply_modes.pop(args["instance_id"], None)
                else:
                    self._reply_modes.clear()
                return result

            if action == "info":
                result = super().dispatch(name, args)
                result["default_reply_language"] = self._default_reply_language
                key = args.get("instance_id")
                if key and key not in self._nodes:
                    result.update({
                        "reply_language": self._reply_modes.get(key, self._default_reply_language),
                        "resolved_language": None,
                        "turn_id": None,
                        "last_language_error": None,
                    })
                if not key:
                    result["reply_language"] = self._default_reply_language
                    result["language_instances"] = {
                        key: node.policy.snapshot() for key, node in self._nodes.items()
                    }
                return result

            return super().dispatch(name, args)

    def synthesize_raw(self, text):
        raise LanguagePolicyError(
            "session_required",
            "Use MCP start, language and speak with instance_id and turn_id; raw synthesis has no session",
        )
