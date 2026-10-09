# MOSS TTS local integration and PDF acceptance status

Acceptance source: `document/具身智能-感知能力-多语种TTS榜.pdf`
(three pages, read in full on 2026-10-09). Its requirements are distinct
from the separate platform notes about JetPack, ports, and Dockerfile edits.

| PDF requirement | Local evidence | Status |
| --- | --- | --- |
| Eight languages: zh, ja, ru, en, es, fr, th, de (p. 1) | Zhang Jinghong's wrapper accepts all eight and historical A100 conversion audio exists for eight cases. The official MOSS model card does not list Thai; Thai pronunciation and stability need native review. | Orin audio quality still untested. |
| Text input and streamed WAV output (p. 2) | The MOSS codec calls back incrementally; the existing ROS TTS topic emits 16 kHz PCM chunks. `/tts/test` wraps complete PCM in a WAV only after synthesis finishes. | Platform WAV streaming contract still unverified. |
| At least five languages to enter scoring (p. 2) | Adapter accepts all eight language codes. | Real end-to-end coverage still untested. |
| Naturalness, fluency, audio quality, content accuracy GSB versus base (p. 2) | Human A/B samples exist outside this fork. | No leaderboard GSB result. |
| First audio <= 1 s, RTF <= 1, failure rate monitored (p. 3) | A fresh A100 GPU 3 single-instance converted ONNX CUDA probe of one short Chinese sentence gave 0.0914 s median first codec callback and 0.9692 median RTF across three warmed repeats. Historical A100 three-instance TensorRT tests failed most RTF cases. | Orin, eight-language, three-instance and platform measurements are open. |
| Three stable inference instances on an Orin 8 GB (p. 1) | Three separate runtime states share one graph pool; local fake-runtime concurrency tests cover routing. | GPU memory and sustained stability untested. |
| Personal public fork, no Git files over 1 MB; model on JuiceFS via specified HTTP host (p. 1) | A 676638720-byte tar is outside Git. JuiceFS copy has the pinned SHA-256 and HTTP HEAD returns 200 with the correct size. | Code branch is a development snapshot; leaderboard submission remains open. |

This branch selects `moss-multi` in `perception/config.yaml` and disables the
other perception plugins for TTS evaluation. It uses the original MOSS prompt,
sampler, 16 built-in reference voices and streaming codec exported from the
zhangjinghong package. The small `plugins/moss_deploy/` files are local
deployment glue; model weights, ONNX graphs and original source stay outside
Git. `plugins/moss_tts.py` leases one of three independent inference runtimes
per utterance while one TensorRT graph pool shares immutable engines.

## External assets

The configured archive URL is
`http://172.28.4.81:34567/i-lanxijiang/phanthymotus-tts/moss-orin-assets.tar`.
The downloader accepts only that internal host
and port, ignores HTTP proxy settings, checks the pinned 676638720-byte size
and SHA-256, rejects unsafe tar members, and atomically extracts to
`/models/moss/assets` under a process lock. It does no network request when
the URL is unset. The JuiceFS copy was read back and matched the pinned
archive SHA-256:

```text
2c17a83c412eeafc193a879fee97b232a25761499e5edc686fcd201d3137d533
```

That archive contains the source assets and ONNX graphs **only**. Build the
TensorRT 10 plans for the selected Orin runtime, then stage them at the paths
in `/models/moss/assets/engine-map.json` before starting TTS. The current
archive does not contain plans and the current fork does not contain the plan
builder, so a fresh image cannot start MOSS yet. The adapter raises a
missing-plan error. A100 plans cannot be used as Orin deployment plans.
Built-in voices use verified reference codes; custom reference encoding is
disabled in this first adapter.

## Validation gates still open

The audio adapter and downloader have offline unit tests. No Jetson image has
been built, and no Orin plan, audio quality, first-audio latency, RTF or
three-instance memory result has been measured from this branch. The current
Dockerfile copies the TensorRT 10 runner but does not prove imports of the
original source's `sentencepiece`, `torch`, `torchaudio`, `onnxruntime` or
WeTextProcessing `tn` modules on ARM. Check those imports in the actual image.
The existing `deploy/build_perception.sh` defaults to JP5.11 and overrides
the Dockerfile default; its JP6.1 argument is `--jp-version 6.1` (Docker build
arg `61`). This TensorRT 10 runner has not been adapted to JP5's TensorRT 8.5.
The PDF itself does not require a specific JetPack release.

Do not invoke `deploy/build_perception.sh` with registry credentials for a
local check: that script also logs in and pushes the image. No image has been
built or published for this branch.

The original MOSS runtime writes a temporary 48 kHz stereo WAV even while it
streams. The plugin forwards each codec callback through a stateful low-pass
converter to the platform's 16 kHz mono PCM16 topic, then removes the temporary
WAV. `/tts/test` returns a completed WAV and cannot measure streaming first
audio. The platform judgeflow transport is not specified in the PDF or this
repository, so the ROS stream must be checked against the real evaluator
before claiming the PDF's streamed-WAV requirement. No platform submission has
been made from this branch.
