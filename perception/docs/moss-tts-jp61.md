# MOSS TTS local integration and PDF acceptance status

**当前状态：开发版本，尚不具备 Darvin 评测条件。** 本次修复 SentencePiece
依赖、每容器实例数，并加入 TRT 构建和短句验证工具。当前 JuiceFS 模型包
仍不含 Orin `.plan`，启动代码也不会自动编译它们。须先在目标环境完成
构建、完整依赖检查、真实音频与三实例验证，再提交新的评测任务。
2026-10-10 相关本地测试共 65 项通过；这些测试不代表 GPU 性能或音质达标。

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
| Three stable inference instances on an Orin 8 GB (p. 1) | The evaluator launches three perception containers. This branch now configures one runtime per container; local fake-runtime tests also cover three within one process. | GPU memory and sustained stability untested. |
| Personal public fork, no Git files over 1 MB; model on JuiceFS via specified HTTP host (p. 1) | A 676638720-byte tar is outside Git. JuiceFS copy has the pinned SHA-256 and HTTP HEAD returns 200 with the correct size. | Code branch is a development snapshot; leaderboard submission remains open. |

This branch selects `moss-multi` in `perception/config.yaml` and disables the
other perception plugins for TTS evaluation. It uses the original MOSS prompt,
sampler, 16 built-in reference voices and streaming codec exported from the
zhangjinghong package. The small `plugins/moss_deploy/` files are local
deployment glue; model weights, ONNX graphs and original source stay outside
Git. `plugins/moss_tts.py` leases an independent inference runtime per
utterance while one TensorRT graph pool per process shares immutable engines.
Judgeflow launches three perception containers, so `moss_instances: 1` in each
gives three model runtimes total. The adapter supports up to three runtimes in
one process for local testing.

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
archive does not contain plans; the builder is in
`perception/tools/moss_build_trt.py`. A fresh image still cannot start MOSS
until the target-built plans are supplied. The adapter raises a
missing-plan error. A100 plans cannot be used as Orin deployment plans.
Built-in voices use verified reference codes; custom reference encoding is
disabled in this first adapter.

## Validation gates still open

The audio adapter and downloader have offline unit tests. The first platform
image built but failed at MOSS import, and no Orin plan, audio quality,
first-audio latency, RTF or three-instance memory result has been measured
from this branch. The Dockerfile now checks the MOSS adapter's SentencePiece
import at build time, but has not proved the original source's `torch`,
`torchaudio`, `onnxruntime` or WeTextProcessing `tn` imports on ARM. Check them
in the actual image before full evaluation.
The existing `deploy/build_perception.sh` defaults to JP5.11 and overrides
the Dockerfile default; its JP6.1 argument is `--jp-version 6.1` (Docker build
arg `61`). This TensorRT 10 runner has not been adapted to JP5's TensorRT 8.5.
The PDF itself does not require a specific JetPack release.

Do not invoke `deploy/build_perception.sh` with registry credentials for a
local check: that script also logs in and pushes the image. No image has been
published from a local build of this branch.

The original MOSS runtime writes a temporary 48 kHz stereo WAV even while it
streams. The plugin forwards each codec callback through a stateful low-pass
converter to the platform's 16 kHz mono PCM16 topic, then removes the temporary
WAV. `/tts/test` returns a completed WAV and cannot measure streaming first
audio. The platform judgeflow transport is not specified in the PDF or this
repository, so the ROS stream must be checked against the real evaluator
before claiming the PDF's streamed-WAV requirement.

## First platform run: startup failure

The 2026-10-09 platform run of commit `2ef676e7e6136f7052383839139158a8126df12c`
failed all 699 requests across three evaluator instances. The candidate
perception container's first traceback was `ModuleNotFoundError: No module named
'sentencepiece'` in `plugins/moss_deploy/runtime.py`; `tts.start` and `tts.speak`
both returned `state=error`. The evaluator then waited 60 seconds for audio on
each case. No first-audio, RTF or GSB result can be inferred from that run.
The Dockerfile now installs SentencePiece and checks the MOSS runtime import at
image-build time. The corrected image has not yet been built or tested.
The first submission also configured three in-process MOSS runtimes per
perception container. With three evaluator containers this would have created
nine runtimes if model initialization had succeeded. This branch now uses one
runtime per container; Orin memory use remains unmeasured.

The asset tar still contains zero Orin TensorRT plans. After the import succeeds,
plan resolution will fail unless target-built plans are staged. Build and verify
at least `global_unified`, `local_cached_step` and `codec_decode_step` on the JP6
Orin, package them into a new pinned JuiceFS archive, then run a short audio
smoke test and three-instance validation before another leaderboard submission.
The small builder is now in `perception/tools/moss_build_trt.py`, alongside its
runtime helper `moss_trt_session.py`; it needs Python 3.10+, TensorRT 10, NumPy
and ONNX in the Jetson test environment. Build inside the candidate's JP6.1
container environment on Orin, with GPU access and the personal checkout and
asset directory mounted. The builder and runtime must use the same TensorRT
version. The commands below run from the root of that mounted personal
checkout. The image currently includes the smoke tool; run the builder from
the checkout. Its NumPy/ONNX/TensorRT dependencies must pass the preflight
before building. Do not install an x86 environment or upgrade the target's
TensorRT/PyTorch packages blindly.

### Orin 转换步骤（仅在已协调的测试时段执行）

2026-10-10 已只读检查共享 Jetson 的现有环境，尚未启动 TRT 构建或模型推理。
以下是构建相关的版本记录；开始测试前应协调时段并重新检查环境：

| 项目 | 只读检查结果 |
| --- | --- |
| 硬件 / BSP | Jetson Orin NX，aarch64，L4T R36.4.3 |
| 该容器的包记录 | Python 3.10.12、NumPy 1.26.4、TensorRT 10.4.0、NVIDIA Torch 2.5.0a0+872d972e41.nv24.8、ONNX Runtime GPU 1.18.0、wetext 0.1.6 |
| 尚未找到的包记录 | onnx、sentencepiece、torchaudio、WeTextProcessing；实际模块导入仍待隔离环境验证，`wetext` 是否提供所需 `tn` 模块也待验证 |

上表来自当前运行容器的包元数据，不代表我们候选镜像的验证结果。
单独编译引擎只需 NumPy、ONNX、TensorRT 与 CUDA；完整音频验证还需原始
MOSS 源码的其他依赖。依赖补充和构建应在个人测试容器中进行，不修改
现有常驻服务；Torchaudio 必须与 NVIDIA Torch 匹配，避免通用安装命令
替换定制 Torch。共享测试时段尚待确认。

先检查当前容器的架构、依赖及 GPU。此步也检查上次缺失的 SentencePiece，
以及原始 MOSS 源码会导入的 Torch、Torchaudio、ONNX Runtime 和文本归一化库。
导入成功只代表依赖可用；最终仍以生成真实音频为准。

```bash
export MOSS_WORK=/home/develop/i-lanxijiang/moss-jp61
python3 - <<'PY'
import platform
import sys
assert platform.machine() == "aarch64", "Run on the Jetson Orin"
assert sys.version_info >= (3, 10), "Python 3.10+ is required"
import numpy, onnx, tensorrt, sentencepiece, torch, torchaudio, onnxruntime
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
assert tensorrt.__version__.split(".")[0] == "10", "This builder targets TensorRT 10"
assert torch.cuda.is_available(), "Expose the Jetson GPU to this container"
print("TensorRT:", tensorrt.__version__, "GPU:", torch.cuda.get_device_name(0))
PY
```

在同一 shell 中使用项目下载器安装资源。它会校验整包大小与 SHA-256，
并创建运行时要求的 `.moss-assets-ready.json`。不要事先创建 `assets`
子目录或只手动解压 tar；下载器会拒绝没有有效标记的现有目录。
如果此路径已有不完整的手动解压目录，另选新的个人测试路径，保留原目录。

```bash
python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path("perception").resolve()))
from plugins.moss_assets import ensure_moss_assets
assets = ensure_moss_assets(
    Path(os.environ["MOSS_WORK"]) / "assets",
    url="http://172.28.4.81:34567/i-lanxijiang/phanthymotus-tts/moss-orin-assets.tar",
)
print(assets)
PY
```

逐个构建下列三张图；每条成功退出后再运行下一条。输出目录分别保存
引擎及 `build_report.json`。当前构建器会拒绝已有 `.plan` 的输出目录，
避免误覆盖；重建使用新的测试目录。

```bash
python3 perception/tools/moss_build_trt.py --analysis "$MOSS_WORK/assets/analysis.json" --output "$MOSS_WORK/assets/plans/global_unified" --graph global_unified --device 0 --workspace-mb 256 --precision fp16
python3 perception/tools/moss_build_trt.py --analysis "$MOSS_WORK/assets/analysis.json" --output "$MOSS_WORK/assets/plans/local_cached_step" --graph local_cached_step --device 0 --workspace-mb 256 --precision fp16
python3 perception/tools/moss_build_trt.py --analysis "$MOSS_WORK/assets/analysis.json" --output "$MOSS_WORK/assets/plans/codec_decode_step" --graph codec_decode_step --device 0 --workspace-mb 256 --precision fp32
```

Each builder invocation validates its plan against the bundled reference
samples before publishing the final `.plan`. Require successful exit and
`status: built` in all three reports. `global_unified` and `local_cached_step`
use mixed FP16 with sensitive operations kept in FP32; `codec_decode_step`
retains FP32 as required by its metadata. `--workspace-mb 256` limits builder
scratch space, not total model/runtime memory. Build sequentially on the 8 GB
device. The 16 built-in voices have verified cached reference codes, so the
codec encoder plan is optional.

先用同一容器、同一资源目录生成一条短句：

```bash
python3 perception/tools/moss_smoke.py --model-dir "$MOSS_WORK/assets" --instances 1 --language zh --text '你好，欢迎使用第四范式。'
```

This reports first PCM callback latency and RTF after warmup; it does not
include the evaluator's ROS/MCP transport overhead or prove eight-language
quality. Compare audio against the original Jinghong inference, then test
three independent one-runtime containers simultaneously, including long
texts and all eight languages. One process with `--instances 3` is a different
memory layout and is not a substitute for the platform test.

Repack the verified plan files with the portable
assets; the current downloader accepts one pinned archive, not a second plans
archive. Exclude `.moss-assets-ready.json`, build staging directories and
generated test audio from the new tar: the downloader creates its own marker
after verifying the new package. Keep the build reports as provenance.
Update `ASSET_SHA256` and `ASSET_BYTES` in `plugins/moss_assets.py` and
the `moss_asset_url` in `config.yaml` for the new JuiceFS object. Use a new
`model_dir` path as well if the test host keeps the old extracted cache.

Once the new archive is available, run `python3 /work/tools/moss_smoke.py`
inside the candidate image with its GPU and `/models` mount. It exits nonzero
on initialization or audio failure and reports first PCM callback and RTF for
one short sentence. This is a single-container check; the platform's three
independent containers still need a simultaneous memory and stability test.

### 首音与 RTF 的后续优化

先记录三实例下的首音、RTF、内存峰值和各子图开销，再改最慢的环节。
当前 `moss_trt_session.py` 每次子图调用都分配/释放 GPU 缓冲区，且输入输出
都经过主机内存，包括反复增长的 KV cache。优先评估复用 GPU 缓冲区、
把 KV cache 留在 GPU 的收益；用 `MOSS_TRT_PROFILE=1` 开启现有统计，
通过图池的 `performance_snapshot()` 读取数据。
再评估首个音频块大小与后续批量大小，平衡首音延迟和吞吐。
转换成功不代表达到首音 <= 1 s 或 RTF <= 1；每次性能修改仍需数值验证和试听，
保留中英术语发音、停顿及八语种自然度。
