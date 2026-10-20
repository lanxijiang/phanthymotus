VITS2 standard-call compatibility

This deployment uses the existing Liaoqianqian v6 model archive, TensorRT engine,
adapter and audio publisher. It adds no model weights, acceleration backend,
denoiser or ASR model. Default speaker_id=0 and speed=1.0 are unchanged.

The public MCP tool is tts. The internal implementation is vits2_tts_compat.
Docker's DEFAULT_TTS_PLUGIN selects it through the existing loader; main.py and
agent-core are unchanged. The configuration panel exposes the existing speaker
and speed fields plus 自动 / 中文 / English.

Standard evaluation:
  start(instance_id, input_topic)
  speak(instance_id, text)
  stop(instance_id)

No language action or turn_id is required for this direct-speech sequence.
Auto chooses pronunciation from each supplied speech text, with Chinese as the
fallback. A wholly Latin quote/address stays English. This is direct TTS, not
an ASR decision, translation, or a dialogue model generating a response.

Explicit dialogue:
  start(instance_id, input_topic)
  language(instance_id, current_user_text, optional input_language)
  [caller gives the returned instruction to its dialogue model]
  speak(instance_id, dialogue_reply, returned turn_id)

Once language succeeds on an instance, tokenless speak is rejected until that
instance is removed. Repeated start or config cannot disable this check.
Explicit null, empty or incorrect tokens never trigger direct-speech fallback.
The original vits2_tts_language implementation remains available and strict.

Fixed Chinese/English settings choose the primary language and numeric/name
readings; they do not translate an opposite-language sentence. Both scripts
may occur in a reply. The original request text is retained; only the frontend
input copy is normalized before the existing chunking and synthesis.

Compatibility scope: MCP calls with an explicit instance_id. ROS input messages
still use the strict JSON text/turn_id contract, and this wrapper does not enable
the original sessionless /tts/test endpoint or implicit instance startup.
Use the base vits2_tts_trt implementation if those legacy entry points are needed.

Darvin submission keeps TTS_PLUGIN=tts and pins the new code commit. Existing
config.language.yaml deployments should explicitly set
TTS_PLUGIN=vits2_tts_language when selecting the strict implementation.

All eight-language MOSS work and non-TTS modules are outside this change.
