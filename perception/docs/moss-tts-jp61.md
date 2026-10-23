# MOSS TTS local integration and PDF acceptance status

**2026-10-10：完整镜像 `0dc645354f12` 在 Orin GPU 0 / JP6.1 /
TensorRT 10.4 上完成三独立容器实测，每容器 20 次，共 60/60 次音频和性能检查通过。**
配置为 v4 整帧引擎、CUDA Graph、1.0 秒有界提前发送，每容器 1 个实例。
测试含自动八语种及 432 字符长文本；实际 MCP→DDS 首 PCM 为
**0.2603612207–0.6498245569 秒**，RTF 为 **0.8390553598–0.9957533818**。
两轮各 30/30 通过，客户端均退出 0，无 OOM。计时从 `tts.speak` 发起开始；
模型启动预热和每次 `tts.start` 的 dry-run 均不计入这些延迟。

请求窗口为 328.94947 秒，三实例共同请求覆盖 269.09139 秒。
整机 547 个、间隔 1 秒的 tegrastats 快照中，RAM 最高 7317/7620 MB，
SWAP 范围 3799–5059 MB，包含四个原后台服务及初始化；这些不是模型独占显存。
**内存余量较紧，这 60 次结果不能保证完整 233 条榜单或长期运行。**
Darvin 私有评估和 GSB 尚未运行，内容准确性与自然度仍待听评。

从空 `/models` 缓存进行 HTTP 下载、校验解包、TRT 加载、中英文 TN 缓存复用、
预热及 MCP 就绪已跑通，发布包包含四份 Orin TRT 10.4 引擎、v4 契约和 TN 缓存。
镜像基础版本已固定 digest，缓存为 `/models/moss-frame-v4/assets`。
报告归档路径为工作区 `jetson-assets/orin-20261010/deployment-validation/summary.json`。
此前 0.8 秒版本有一条 RTF 1.0041256 的记录，下文保留该失败历史与适配器层实验。
代码版本以个人 `lanxijiang/phanthymotus` 的提交记录为准；本文不替代推送或榜单提交记录。

Acceptance source: `document/具身智能-感知能力-多语种TTS榜.pdf`
(three pages, read in full on 2026-10-09). Its requirements are distinct
from the separate platform notes about JetPack, ports, and Dockerfile edits.

### 整帧局部解码候选（2026-10-10）

`moss_export_sampled_frame.py` 把每帧的 17 次局部推理合并为
`local_fixed_sampled_frame`，共享原始权重。它固定当前实际使用的采样参数：
文本温度 1.5、top-p 1、top-k 50；音码温度 1.7、top-p 0.8、top-k 25、
重复惩罚 1。随机数由每实例原 RNG 提供，停止帧恢复未使用的 16 次抽样。
模型保持 FP16，采样显式 FP32；浮点舍入意味着不保证与原作者 ONNX
`ort_cpu_runtime` 的 FP64 概率路径逐种子生成相同 WAV。Jinghong HTTP
服务使用 PyTorch 默认 bf16 与 `torch.multinomial`，与该 ONNX FP64 路径
不同；尚未证明候选与 HTTP 服务逐样本等价，音质必须另行验证。

专用构建器用四份真实 hidden 和八组预定随机数做 32 组 GPU 对照：
原单步引擎沿候选音码逐步计算 logits；另用独立采样实现检查全部
17 个离散输出。它要求数值门槛、音码精确一致、继续/结束与全部 16 通道
覆盖同时通过，才发布 `.plan`。运行时校验模型、参数、引擎与报告哈希。
失败报告保留，不能作为可用引擎加载。

第一版弱类型构建的采样存在意外 FP16 舍入，已被验证拦截。
强类型 v3 在优化级别 0 构建成功（94,986,044 字节），32 组全部通过。
v3 的历史首次计时得到首 PCM 0.230 秒、RTF 1.815（profiling 开启，单次，
音频 3.36 秒）。43 次整帧调用没有新增 GPU IO 分配，但运行耗时仍集中
于 TensorRT enqueue，因此此结果尚不构成提速或榜单达标证明。
无循环 v4 强类型引擎已在优化级别 1 构建（91,259,476 字节），32 组
GPU 对照也全部通过。它把累计概率改为静态并行前缀和，并保护有效概率
支持集与终端 CDF 平台；模型权重不变。这是局部引擎大小，不能当作整个
TTS 系统的模型大小或显存占用。

这里的“32 组通过”有明确范围：共 544 组沿候选音码对齐的逐步 logits
与原单步 GPU 引擎比较，最大绝对差为 0.027832、最大相对 RMS 为 1.3234%，
只有 8 组差值全为 0。544 次离散采样检查全部精确一致，是候选自身 logits
对独立 FP32 采样实现的检查；不是与旧 FP64 路径或 Jinghong HTTP 服务的
token 精确一致。

`MOSS_TRT_FRAME_CUDA_GRAPH=1` 为整帧候选启用 CUDA Graph。首次正常执行
后捕获固定运算，后续重放；每个实例使用私有流、固定 GPU 地址和页锁定
输入内存。失败明确报错。通用运行器默认关闭该开关，当前 JP6.1 镜像以
环境变量显式默认开启；即使关闭 profiling，
`memory_snapshot.cuda_graph_sessions` 仍记录实际捕获与重放次数。
该阶段 MOSS 本地测试 299 项通过；完整 perception 测试在本机 Python 3.13 上因
旧 `test_file_intake.py` 导入已移除的 `cgi` 模块而无法收集，未声称全项目
测试通过。Jetson 使用 Python 3.10。

#### v4 中文单实例与 CUDA Graph 实测

| 同一 v4 引擎的配置 | 三次 RTF | 三次首 PCM，秒 |
| --- | --- | --- |
| CUDA Graph 关闭 | 0.7505 / 0.6823 / 0.6101 | 本表未列 |
| `MOSS_TRT_FRAME_CUDA_GRAPH=1` | 0.4379 / 0.4815 / 0.4851 | 0.1011 / 0.0959 / 0.0862 |

开启 CUDA Graph 的上述中文请求在适配器层达到首音与 RTF 门槛，尚不包含平台 ROS/MCP 传输。
72 对实际 GPU 验证中，CUDA Graph 开关前后的四个整帧输出
（`should_continue`、`frame_token_ids`、`text_logits`、`audio_logits`）
均逐位一致，重复输入也逐位一致；所有执行的实际 capture/replay 检查通过。
端到端 PCM 配对有 65–71 个采样点相差 1 LSB，因此不声称 WAV 逐字节一致。
差异可能来自自适应 codec 路径，但原因尚未证明。

这些检查验证同一 v4 引擎的 CUDA Graph 执行一致性，不能推导出与原作者
ONNX FP64 概率路径或 Jinghong HTTP 服务逐种子等价，也不能代替发音、
自然度和 GSB 评价。
完整镜像后续已跑通实际 MCP→DDS 请求，发送节奏优化的分阶段计时见下文。
适配器计时与接口计时范围不同，正式榜单集成尚未验证。

#### v4 八语种与混合长文补测

以下均为单实例、CUDA Graph 开启、每项 1 次请求，全部成功输出非空且非全零
PCM。首 PCM 是适配器回调时间，不含平台传输，也不等同于实际语音起声时间。

| 输入 | RTF | 首 PCM，秒 | 生成音频时长，秒 |
| --- | ---: | ---: | ---: |
| 中文 zh | 0.501826 | 0.106948 | 6.16 |
| 日语 ja | 0.386366 | 0.095642 | 5.44 |
| 俄语 ru | 0.326917 | 0.081055 | 4.88 |
| 英语 en | 0.343291 | 0.112047 | 5.12 |
| 西班牙语 es | 0.541028 | 0.096377 | 3.76 |
| 法语 fr | 0.515051 | 0.098822 | 4.56 |
| 泰语 th | 0.472483 | 0.118481 | 10.08 |
| 德语 de | 0.574093 | 0.122389 | 3.68 |
| OpenMLDB / LAST JOIN / FeatInsight 中英混合长文 | 0.536628 | 0.170315 | 26.72 |

日语样例首个非零 PCM 块为 0.129914 秒。每种语言只有一条样例，合成成功
不等同于发音、内容准确率或自然度通过。按 20 ms 窗口、-50 dBFS 阈值检查，
泰语音频的 4.72–6.22 秒和 7.10–9.38 秒、中英混合音频的 10.36–11.60 秒
为低能量区间；需结合文本和听评确认，不能仅凭这些区间断定停顿异常。
相同文本、音色、seed 和预热条件下，旧中英混合音频长 26.00 秒，在
12.98–14.40 秒也有 1.42 秒低能量区间；新音频最长为 1.24 秒。
原作者同文本 `samples/th.wav` 长 10.00 秒，已有 2.16 秒低能量区间，
低能量占比 44.8%；新泰语音频长 10.08 秒，占比 47.8%，上述长区间为
1.50 和 2.28 秒。旧泰语文件为 48 kHz 双声道、seed 未核验，不能当作严格
A/B。两组都需要复听，目前没有证据证明这些区间是本轮新引入的停顿回归。
信号检查记录在 `frame-optimization-20261010/audio-signal-audit.json`。
八语种音质与 GSB 仍为 pending。
用户本轮试听反馈为“似乎还不错”；尚未明确试听了哪些语言，这条主观正向反馈
不等于八语种质量或 GSB 验收通过。

报告及音频位于工作区
`jetson-assets/orin-20261010/frame-optimization-20261010/results/frame-v4-acceptance-*/`。

#### v4 三独立容器并发补测

每个容器只有 1 个实例，完成预热后通过 barrier 同步放行；三个报告的
`boot_id` 和 time namespace 相同，可比较单调时钟。首轮起始差为 0.039947 秒，
三路请求共同运行 10.969 秒；第二轮起始差约 0.4903 秒，共同运行 10.460 秒。
两轮共 6 条请求全部成功，RTF 0.595104–0.647720，首 PCM
0.166347–0.243996 秒；三个容器均退出码 0、`OOMKilled=false`。

有效请求窗口总跨度 22.662 秒，每实例只完成 2 条请求。这个短期结果不能代表
233 条榜单任务或持续运行的稳定性。记录到的设备共享内存可用量最低为
311267328 字节（约 297 MiB），包含四个后台服务的影响，是离散快照，
不是全过程峰值，也不是模型独占显存。测试前后系统已用 swap 增加约 84 MiB，
不能据此宣称内存充足；仍需更长时间的三容器压力测试和内存监测。

报告、音频、barrier 记录和内存记录已回收到
`jetson-assets/orin-20261010/frame-v4-concurrent-results.tar.gz`，
解包结果位于上述 `frame-optimization-20261010/` 下。这些是实验资料；
部署使用下文单独列出的固定 SHA-256 发布包，不把测试音频上传为模型资产。

先前的引擎、验证报告、诊断与已有音频备份在工作区
`jetson-assets/orin-20261010/frame-optimization-20261010.tar`，SHA256：
`629c352a136bc46f0b723ea607e71e990cdca70a1d5111bb4673647031487a68`。
此包不是发布包，不包含重复的原始外部权重。

构建和启用示例（目标 Orin 容器内；目录必须为新的空目录）：

```bash
python3 perception/tools/moss_export_sampled_frame.py \
  --assets /moss/assets --output-dir /moss/candidates/frame-new
python3 perception/tools/moss_build_sampled_frame.py \
  --assets /moss/assets \
  --analysis /moss/candidates/frame-new/analysis.json \
  --contract /moss/candidates/frame-new/frame-contract.json \
  --output /moss/candidates/frame-new/plans --workspace-mb 256
export MOSS_TRT_FRAME_PLAN=/moss/candidates/frame-new/plans/local_fixed_sampled_frame/local_fixed_sampled_frame.plan
export MOSS_TRT_FRAME_CONTRACT=/moss/candidates/frame-new/frame-contract.json
export MOSS_TRT_FRAME_CUDA_GRAPH=1
```

`MOSS_TRT_FRAME_PLAN` 与 `MOSS_TRT_FRAME_CONTRACT` 需要同时设置，
并与精选输出头候选互斥；当前 JP6.1 镜像已将两者设为发布包内路径。
在受控对照中取消这两个环境变量，才会使用原单步局部引擎。
镜像默认 `MOSS_TRT_FRAME_CUDA_GRAPH=1`，设为 `0` 可做同引擎对照。
候选加载后省去旧局部引擎反序列化，但保留原资产用于溯源。
关闭 CUDA Graph 后，`MOSS_TRT_REUSE_FRAME_BUFFERS=0` 可关闭候选的固定 IO 缓冲区复用。
上述示例不下载模型、不自动发布 GitHub，也不会更新 JuiceFS 包。

### 完整镜像 MCP→DDS 与发送节奏验证

以下首包与 RTF 从客户端 `tts.speak` 发起计到 DDS 首 PCM/EOF，包含传输、
预缓冲和发布节奏。它们与前面的适配器计时分开记录。

| 阶段 | 请求范围 | 实测结果 |
| --- | --- | --- |
| 原发布节奏，镜像 `1ac9f9fde91b` | 八语种各 1 条及混合长文，共 9 条 | 9/9 音频完整、非零、EOF 与发布端帧数匹配；首包 0.4993–0.5805 s，RTF 1.00428–1.02643，RTF 均超 1。 |
| 0.8 s 有界提前发送，单实例 | 相同请求历史，共 9 条 | 9/9 首包及 RTF 门槛通过；首包 0.50173–0.59935 s，RTF 0.91603–0.98576。 |
| 0.8 s 有界提前发送，三独立容器 | 每实例累计 30 条，共 90 条 | 90/90 音频收齐，89/90 性能门槛通过；首轮一条 `en-auto` 首包 0.82094 s、RTF 1.0041256。第二轮 30/30 通过，首包最大 0.63338 s、RTF 最大 0.997576。 |
| 1.0 s 有界提前发送，镜像 `0dc645354f12` | 三独立容器每实例 20 条，共 60 条，含自动八语种及 432 字符长文本 | 60/60 音频和性能通过；首 PCM 0.2603612207–0.6498245569 s，RTF 0.8390553598–0.9957533818；两轮各 30/30 通过，无 OOM，全部客户端退出 0。 |

最终测试先启动并预热模型，每次 `tts.start` 另有 dry-run，均不计入 `speak`
测量。请求窗口 328.94947 秒，三实例共同请求覆盖 269.09139 秒；第二轮首 PCM
最大 0.629137 秒、RTF 最大 0.994533。547 个一秒间隔整机 tegrastats 快照
覆盖初始化和四个原后台服务，RAM 最高 7317/7620 MB、SWAP 3799–5059 MB。
它们不是模型独占显存，也不能代表采样间隔之间的瞬时峰值；内存余量较紧，
仍不能保证完整 233 条榜单或长期播放稳定。结果归档至工作区
`jetson-assets/orin-20261010/deployment-validation/summary.json`。

发布端保留首 5 帧预缓冲，跨句保持播放进度，避免连续请求积累提前发送量。
1.0 秒预算结合浏览器额外 200 ms 缓冲，按现有缓冲设计名义余量为 0.3 秒；
这不代表真实播放端已通过长时间验证。0.8 秒版本的发布器单元测试确认同一输入
PCM 逐字节保持；整条推理链路的前后 WAV 对照长度一致，每个文件有 60–519 个
采样点相差 1 LSB，无更大差异，因此不声称 WAV 逐字节一致。内容准确性、
自然度和 GSB 仍需听评。

该阶段 442 项定向回归通过；之后 1.0 秒参数调整相关 7 项定向检查通过。
这些是定向检查，不是整个 perception 测试集通过的声明；真实设备的 60 条结果另列如上。

| PDF requirement | Local evidence | Status |
| --- | --- | --- |
| Eight languages: zh, ja, ru, en, es, fr, th, de (p. 1) | One v4 Orin request per language succeeded through the complete image's MCP→DDS interface with nonempty, nonzero PCM. The official MOSS model card does not list Thai; Thai pronunciation and stability need native review. | Eight sample requests passed synthesis and transport checks; pronunciation, naturalness and broader coverage remain open. |
| Text input and streamed WAV output (p. 2) | The final image completed 60 MCP→DDS requests with nonzero 16 kHz PCM, EOF and matching publisher frame counts. `/tts/test` wraps complete PCM in a WAV only after synthesis finishes. | Actual MCP/ROS streaming verified for these requests; the private evaluator's streamed-WAV contract remains unverified. |
| At least five languages to enter scoring (p. 2) | Adapter accepts all eight language codes; one complete-image MCP/ROS request per language produced audio. | Eight-language transport observed; formal scoring eligibility unverified. |
| Naturalness, fluency, audio quality, content accuracy GSB versus base (p. 2) | Human A/B samples exist outside this fork. | No leaderboard GSB result. |
| First audio <= 1 s, RTF <= 1, failure rate monitored (p. 3) | Final image with 1.0 s send-ahead: 60/60 audio and performance checks passed, first PCM 0.2603612207–0.6498245569 s, RTF 0.8390553598–0.9957533818. Startup warmup and each start dry-run are excluded from speak timing. | These 60 requests pass the measured interface thresholds; Darvin private evaluation has not run. The 0.8 s version's one RTF miss remains recorded above. |
| Three stable inference instances on an Orin 8 GB (p. 1) | Three independent one-runtime containers completed 20 requests each, no OOM, client exits 0; request window 328.94947 s and three-way request coverage 269.09139 s. Whole-device snapshots: RAM max 7317/7620 MB, SWAP 3799–5059 MB, including background services. | This bounded run passed; 233-case and long-term stability remain unverified. Memory headroom is tight, and whole-device RAM is not model-exclusive GPU memory. |
| Personal public fork, no Git files over 1 MB; model on JuiceFS via specified HTTP host (p. 1) | The new 1169192960-byte v4 tar is outside Git, published to JuiceFS with its pinned SHA-256. A cold-cache image downloaded, verified and loaded its Orin TRT plans successfully. | Model package and cold startup verified. Code revision is tracked by the personal fork's commits; formal leaderboard submission is a separate step. |

This branch selects `moss-multi` in `perception/config.yaml` and disables the
other perception plugins for TTS evaluation. It uses the original MOSS prompt,
sampling rules, 16 built-in reference voices and streaming codec exported from the
zhangjinghong package. The small `plugins/moss_deploy/` files are local
deployment glue; model weights, ONNX graphs and original source stay outside
Git. `plugins/moss_tts.py` leases an independent inference runtime per
utterance while one TensorRT graph pool per process shares immutable engines.
Judgeflow launches three perception containers, so `moss_instances: 1` in each
gives three model runtimes total. The adapter supports up to three runtimes in
one process for local testing.

## External assets

The configured archive URL is
`http://172.28.4.81:34567/i-lanxijiang/phanthymotus-tts/moss-orin-jp61-frame-v4-20261010.tar`.
The published file is
`/mnt/data/i-lanxijiang/phanthymotus-tts/moss-orin-jp61-frame-v4-20261010.tar`.
The downloader accepts only that internal host
and port, ignores HTTP proxy settings, checks the pinned 1169192960-byte size
and SHA-256, rejects unsafe tar members, and atomically extracts to
`/models/moss-frame-v4/assets` under a process lock. It does no network request
when the URL is unset. Publication and HTTP availability were checked; the
pinned archive SHA-256 is:

```text
1538144d7365ef74af7828a49fb9d02da7130f6abd7b3e8f61f56e87713d2a74
```

This archive contains source assets and ONNX graphs, four verified Orin
TensorRT 10.4 plans (`global_unified`, `local_cached_step`, `codec_decode_step`,
`local_fixed_sampled_frame`), build reports, the v4 contract and the Chinese TN
cache. The default v4 runtime loads the fused local plan and retains the old
local plan for controlled comparisons and provenance. Normal deployment loads
these packaged plans; it does not build TensorRT engines automatically.
The complete image successfully downloaded, verified and unpacked this archive
from an empty `/models` cache, loaded TRT plans, reused the Chinese/English TN
caches and reached MCP readiness after warmup. A100 plans cannot be used as
Orin deployment plans.
Built-in voices use verified reference codes; custom reference encoding is
disabled in this first adapter.

## Validation gates still open

The first platform image failed at MOSS import. An isolated image built from
`perception/Dockerfile.moss-trt` now builds and runs the Orin engines, and the
ARM frontend script has installed and verified Pynini 2.1.7 / WeTextProcessing
1.2.0 imports. Real Chinese audio also succeeded with that frontend. The local
source loader defers torch/torchaudio imports until custom reference loading;
the 16 built-in voices use their cached codes without torchaudio. Jinghong's
source files and model weights were not modified. The complete Darvin
`Dockerfile.jetson` now integrates this frontend installation route and uses the
tested JP6.1 perception base image pinned to
`sha256:0c8d533b94a858fd6c8571c188ead15b95645ead0f79f589272db40dff662f3a`.
It preserves the base ROS/CUDA/TRT stack, installs the locked frontend in
`/opt/moss-frontend`, and starts through `moss_entrypoint.sh`. The `libnvos`
failure in builds without GPU mounts is resolved: the build checks TRT installation,
while the runtime startup performs the real imports and GPU check. Image
`1ac9f9fde91b` built successfully and passed those startup dependency checks,
confirming TensorRT 10.4 and GPU 0 with SM 8.7. Cold-cache startup subsequently
completed the HTTP download, archive verification/extraction, TRT loading,
existing Chinese/English TN cache reuse and warmup, then reached MCP readiness.
The original publisher delivered all nine MCP→DDS requests but slightly
exceeded RTF 1. The subsequent 0.8 s bounded send-ahead path passed 9/9
single-instance requests and 89/90 three-container performance checks, with
all 90 concurrent-test audio results received. Image `0dc645354f12` with the
1.0 s budget subsequently passed all 60 audio and performance checks in three
independent containers. Details and timing exclusions are recorded above.
Audio quality, GSB, Darvin private evaluation and full 233-case/long-term stability
remain open. Whole-device RAM/swap observations include background services
and show tight headroom; they are not model-exclusive GPU-memory measurements.
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
image-build time. The complete candidate image has subsequently passed cold
startup and real MCP/ROS synthesis. Later sending optimizations and their
historical performance miss are recorded above; the final 1.0 s image passed
the subsequent 60-request interface test.
The first submission also configured three in-process MOSS runtimes per
perception container. With three evaluator containers this would have created
nine runtimes if model initialization had succeeded. This branch now uses one
runtime per container. Short concurrent runs and shared-memory snapshots now exist;
full-run memory peaks and sustained three-container stability remain unverified.

The replacement v4 archive now includes the target-built plans and their
validation reports. The bounded complete-image interface test is complete;
audio-quality review, private evaluator scoring and full 233-case/long-term
three-container behavior remain to be checked.
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

### 初始 Orin 构建与优化前基线

测试使用我方 `Dockerfile.moss-trt` 镜像，TensorRT 10.4、GPU compute capability
8.7。宿主目录为 `/home/develop/i-lanxijiang/moss-jp61`，挂载到容器 `/moss`；
个人项目挂载到 `/src`。三份引擎保存在宿主目录 `assets/plans` 下。
最初这批构建测试容器已退出并清理；后续优化使用我方独立容器，原有四个
后台服务继续运行。

| 引擎 | 精度 | `.plan` 字节数 | 构建耗时 | 数值样本 |
| --- | --- | ---: | ---: | ---: |
| `global_unified` | FP16，敏感运算 FP32 | 229751988 | 47.20 s | 5 / 5 通过 |
| `local_cached_step` | FP16，敏感运算 FP32 | 115245404 | 8.16 s | 4 / 4 通过 |
| `codec_decode_step` | FP32 | 52328004 | 47.07 s | 4 / 4 通过 |

引擎及验证输出已备份到工作区 `jetson-assets/orin-20261010/orin-trt-verified.tar`，
SHA256 为 `05ca4e3e1c1b79f1178b4696a3efb0bbcb8b026d33c8fc0552edcf1428136ced`。
它包含验证输出，用于备份；不是已上传的完整运行模型包。

| 单实例短句测试 | 首 PCM 回调，秒 | RTF |
| --- | --- | --- |
| 日语冷运行 | 5.1545（初始化另计 1.4331 s） | 4.4633 |
| 日语热运行，三次 | 0.330344 / 0.313685 / 0.318981 | 3.348712 / 3.316373 / 3.123211 |
| 中文热运行，三次 | 0.372082 / 0.301885 / 0.279546 | 3.320481 / 2.935370 / 3.212721 |

这些数值由真实适配器生成音频获得，不含平台 ROS/MCP 传输开销，也不代表
音质、GSB 或三容器通过。这是初始构建阶段的历史基线；当时 RTF 超过 1，
该阶段没有继续三容器压力测试。最新 v4 候选结果见本文开头。

### Orin 转换步骤（可选重建，需协调测试时段）

当前发布包已经包含可加载的 Orin 引擎，部署不需要执行以下重建步骤。
以下保留最初三张图的构建方法；整帧 v4 的专用构建与验证见本文前半部分。

以下为测试使用的基础环境记录；重新测试前应检查实际容器版本：

| 项目 | 环境记录 |
| --- | --- |
| 硬件 / BSP | Jetson Orin NX，aarch64，L4T R36.4.3 |
| 该容器的包记录 | Python 3.10.12、NumPy 1.26.4、TensorRT 10.4.0、NVIDIA Torch 2.5.0a0+872d972e41.nv24.8、ONNX Runtime GPU 1.18.0、wetext 0.1.6 |
| 我方测试环境补充 | `Dockerfile.moss-trt` 安装 ONNX 1.17.0、SentencePiece 0.2.0；`moss_prepare_frontend.sh` 安装并验证 Pynini 2.1.7 / WeTextProcessing 1.2.0。内置音色路径无需 torchaudio。 |

基础包记录来自测试所用的现有镜像；新增依赖只装在我方测试镜像和独立前缀。
单独编译引擎需要 NumPy、ONNX、TensorRT 与 CUDA。内置音色路径使用缓存
参考编码，完整中英文推理还需独立安装的原作者文本归一化依赖。
完整 Darvin 镜像需再次验证，不应替换共享机器上的定制 Torch 或常驻服务。

以下命令在我方测试容器内、`/src` 项目根目录运行。先检查构建依赖和 GPU；
原作者归一化库稍后单独准备。导入成功只代表依赖可用。

```bash
cd /src
export MOSS_WORK=/moss
source /etc/dla-fallback.env
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
python3 - <<'PY'
import platform
import sys
assert platform.machine() == "aarch64", "Run on the Jetson Orin"
assert sys.version_info >= (3, 10), "Python 3.10+ is required"
import numpy, onnx, tensorrt, sentencepiece, torch, onnxruntime
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
    url="http://172.28.4.81:34567/i-lanxijiang/phanthymotus-tts/moss-orin-jp61-frame-v4-20261010.tar",
)
print(assets)
PY
```

如需重建，逐个构建下列三张图；每条成功退出后再运行下一条。使用全新的
`rebuild-plans` 输出目录，避免覆盖发布包已有的引擎及 `build_report.json`。
这些新目录不会自动替换 `engine-map.json` 或生产镜像的默认计划。

```bash
python3 perception/tools/moss_build_trt.py --analysis "$MOSS_WORK/assets/analysis.json" --output "$MOSS_WORK/rebuild-plans/global_unified" --graph global_unified --device 0 --workspace-mb 256 --precision fp16
python3 perception/tools/moss_build_trt.py --analysis "$MOSS_WORK/assets/analysis.json" --output "$MOSS_WORK/rebuild-plans/local_cached_step" --graph local_cached_step --device 0 --workspace-mb 256 --precision fp16
python3 perception/tools/moss_build_trt.py --analysis "$MOSS_WORK/assets/analysis.json" --output "$MOSS_WORK/rebuild-plans/codec_decode_step" --graph codec_decode_step --device 0 --workspace-mb 256 --precision fp32
```

Each builder invocation validates its plan against the bundled reference
samples before publishing the final `.plan`. Require successful exit and
`status: built` in all three reports. `global_unified` and `local_cached_step`
use mixed FP16 with sensitive operations kept in FP32; `codec_decode_step`
retains FP32 as required by its metadata. `--workspace-mb 256` limits builder
scratch space, not total model/runtime memory. Build sequentially on the 8 GB
device. The 16 built-in voices have verified cached reference codes, so the
codec encoder plan is optional.

构建完成后，在同一容器准备独立的 ARM 文本前端，再生成一条中文短句。
安装脚本固定核心包和校验值，记录传递依赖锁文件，核对原 Torch / TRT 未被
替换；不激活 conda。`runtime-python` 仅为这条命令追加前端库的搜索路径。
首次文本归一化构图可能较慢，与引擎编译分开运行。

```bash
bash perception/tools/moss_prepare_frontend.sh --prefix "$MOSS_WORK/frontend" --python /usr/bin/python3 --lock perception/tools/moss_frontend_aarch64.lock
"$MOSS_WORK/frontend/runtime-python" perception/tools/moss_smoke.py --model-dir "$MOSS_WORK/assets" --instances 1 --language zh --text '你好，欢迎使用第四范式。'
```

This reports first PCM callback latency and RTF after warmup; it does not
include the evaluator's ROS/MCP transport overhead or prove eight-language
quality. Compare audio against the original Jinghong inference, then test
three independent one-runtime containers simultaneously, including long
texts and all eight languages. One process with `--instances 3` is a different
memory layout and is not a substitute for the platform test.

The published v4 archive, downloader pins and fresh cache path are already
configured. Future engine changes require a newly verified package and updated
pins; preserve reports and exclude staging files, test audio and the downloader's
`.moss-assets-ready.json` marker from any new archive.

For adapter-only checks in the complete image, use
`/opt/moss-frontend/runtime-python /work/tools/moss_smoke.py --model-dir /models/moss-frame-v4/assets`,
then use `moss_platform_smoke.py` against its dedicated MCP endpoint and matching
ROS domain. That tool calls real `tts.start`/`tts.speak`, subscribes to
`audio_msgs/msg/AudioChunk`, and checks nonzero PCM, EOF, successful trace spans
and publisher frame counts. Wait for MCP readiness before invoking it. Its
latency includes ROS transport and the publisher's prebuffer/playback pacing;
its RTF must not be substituted for the adapter-only numbers above. The 0.8 s
configuration has completed single-instance and three-container checks; the
latest 1.0 s configuration has passed all 60 requests in three independent
containers. Full 233-case/sustained testing and formal leaderboard evaluation remain open.

### 历史测量：resident + fast 优化与精选输出头候选

以下为同轮中文单实例对照，均使用 `MOSS_TRT_PROFILE=1`，计时不含模型
初始化、预热和最终 WAV 写盘，也不含 ROS/MCP 传输。前四行对应相同的
三次请求，四种配置各次 WAV 逐字节相同；这些短句结果不能代表八语种和
三容器能力。profiling 有额外开销，应与下面关闭 profiling 的最终复测分开查看。

| 配置 | 三次首 PCM，秒 | 三次 RTF |
| --- | --- | --- |
| 本轮 control，关闭两项优化 | 0.332729 / 0.301727 / 0.306646 | 3.279524 / 3.259443 / 3.252614 |
| fast_only，快速采样 | 本表未列 | 3.118 / 3.078 / 3.071 |
| resident_only，GPU 状态常驻 | 本表未列 | 2.372 / 2.356 / 2.326 |
| combined，两项同时开启 | 0.174057 / 0.172063 / 0.165322 | 1.812565 / 1.795065 / 1.560069 |
| selected，精选输出头候选 | 0.149621 / 0.151028 / 0.171288 | 1.419172 / 1.403413 / 1.414971 |

#### 该阶段最终复测：profiling 关闭，原三份引擎

使用同一最终代码、相同初始 seed 1234、预热文本、音色和请求顺序。
control 显式设置两个开关为 0；default 不设置开关，报告中的
`effective_optimizations` 确认两项默认生效。全部请求成功。

| 文本 | 对照 RTF | 默认优化 RTF | 优化后首 PCM，秒 | WAV 对照 |
| --- | --- | --- | --- | --- |
| 中文短句，3 次 | 3.259109 / 3.289643 / 3.247031 | 1.778101 / 1.770389 / 1.787913 | 0.150238 / 0.174121 / 0.209903 | 3 / 3 逐字节一致 |
| OpenMLDB / LAST JOIN / FeatInsight 中英混合长文，26 秒音频，1 次 | 3.408123 | 1.741540 | 0.232136 | 逐字节一致 |
| 日文短句，1 次 | 3.264610 | 1.799998 | 0.181243 | 逐字节一致 |

“首 PCM”是预热后适配器回调，不含平台传输；WAV 一致仅证明这些请求未改变
原有音频，不代表原有发音已通过 GSB 或八语种质量验收。测试后我方容器全部退出，
四个原有后台容器仍在运行，没有停止他人的任务。报告与音频已回收到工作区
`jetson-assets/orin-20261010/results/{zh,mixed,ja}-production-{control,default}/`。
完整实验备份 `optimization-20261010.tar` 的 SHA256：
`db1569f5abf51f084b0eee898ea480be1078849af8dcbc12e8859cd3280d23e0`。
备份包含候选计划和报告，排除了已有本地副本的候选外部权重；不是上线模型包。

GPU 状态常驻完成 69 项真实 GPU 数值与状态传递检查。个人方案的
`TensorRTGraphPool` 已默认启用 resident，适配运行时已默认启用 fast；
分别设置 `MOSS_TRT_RESIDENT_STATE=0`、`MOSS_FAST_SAMPLING=0` 可回退。
通用 `TensorRTSession` / 构建验证器默认仍返回主机 NumPy，不受池默认值影响。
该阶段 141 项 MOSS 本地测试通过。fast 优化主机采样步骤，模型计算仍在 GPU。

selected 候选已通过 38 项数值检查，最大相对 RMS 误差约 0.04663，
38 项中有 1 项音频 argmax 不一致。其中文 WAV 和时长也发生变化，数值
阈值通过不能证明采样和音质等价，需要复听。候选仅在同时显式设置
`MOSS_TRT_SELECTED_LOCAL_PLAN` 与 `MOSS_TRT_SELECTED_LOCAL_CONTRACT` 时使用，
不会随 resident + fast 默认开启。

最小复测示例（我方容器 `/src` 下，`MOSS_WORK=/moss`；每次换新的输出目录）：

```bash
# 原三份引擎的 resident + fast 历史方案，关闭全部候选和 profiling。
env -u MOSS_TRT_SELECTED_LOCAL_PLAN -u MOSS_TRT_SELECTED_LOCAL_CONTRACT \
  -u MOSS_TRT_FRAME_PLAN -u MOSS_TRT_FRAME_CONTRACT -u MOSS_TRT_FRAME_CUDA_GRAPH \
  MOSS_TRT_RESIDENT_STATE=1 MOSS_FAST_SAMPLING=1 MOSS_TRT_PROFILE=0 \
  MOSS_TRT_REUSE_LOCAL_BUFFERS=0 "$MOSS_WORK/frontend/runtime-python" \
  perception/tools/moss_benchmark.py --model-dir "$MOSS_WORK/assets" \
  --language zh --speaker-id 0 --warmup-text '你好。' --text '你好，欢迎使用第四范式。' \
  --repeat 3 --output-dir "$MOSS_WORK/results/combined-confirm"

# 回退到原来的状态传输和采样实现，其他条件相同。
env -u MOSS_TRT_SELECTED_LOCAL_PLAN -u MOSS_TRT_SELECTED_LOCAL_CONTRACT \
  -u MOSS_TRT_FRAME_PLAN -u MOSS_TRT_FRAME_CONTRACT -u MOSS_TRT_FRAME_CUDA_GRAPH \
  MOSS_TRT_RESIDENT_STATE=0 MOSS_FAST_SAMPLING=0 MOSS_TRT_PROFILE=0 \
  MOSS_TRT_REUSE_LOCAL_BUFFERS=0 "$MOSS_WORK/frontend/runtime-python" \
  perception/tools/moss_benchmark.py --model-dir "$MOSS_WORK/assets" \
  --language zh --speaker-id 0 --warmup-text '你好。' --text '你好，欢迎使用第四范式。' \
  --repeat 3 --output-dir "$MOSS_WORK/results/control-confirm"
```

实现位置：`tools/moss_trt_session.py` 管理 GPU 状态与缓冲区；
`plugins/moss_deploy/trt_pool.py` 管理独立实例及默认开关；`runtime.py` 和
`sampling.py` 接入快速采样；`selected_logits.py` 与 `plugins/moss_tts.py`
负责显式候选契约。`tools/moss_benchmark.py` 保存真实 WAV、分阶段计时、
作用域明确的内存数据及实际引擎/代码/配置哈希；相应验证工具检查优化数值。

早期仅复用 local 缓冲区的实验仍不作为默认方案：8 次 GPU 对照通过，分配
96→20，输出容差 `1e-6`；中文 RTF 3.210852 / 3.229614 / 3.212796，未证明
端到端收益。进程 RSS、设备可用内存变化均不能直接等同于模型独占显存峰值。
含 Orin 引擎的新 JuiceFS 运行包已发布，完整镜像的最终 60 条 MCP→DDS
音频和性能检查通过。Darvin 私有评估、GSB、完整 233 条及长期运行仍需验证；
代码提交以个人仓库记录为准。

## 多语种发送配置与分块修复（2026-10-23）

`9f4eca8c` 没有包含此前本地实验镜像中的生成即发送路径，仍使用
`BoundedAudioPacer` 的一秒提前量。长句的收齐音频 RTF 因此接近 1。
这不能直接与旧实验的生成即发送 RTF 比较后归因于模型推理退化。

MOSS 现在支持以下部署配置；修改后重启服务：

| 配置 | 使用场景 |
| --- | --- |
| `moss_transport_mode: paced` | 默认构造行为，适用于实时浏览器/机器人播放器；跨连续请求保留一秒提前发送额度。 |
| `moss_transport_mode: as_generated` | 显式选择，适用于 Darvin 等保存完整 PCM 的接收器。生成后直接发送真实音频，不改变采样率、语速或波形。 |
| `moss_subscriber_settle_s: 0.5` | 首帧前观察稳定 DDS 订阅匹配，等待与合成重叠；无订阅时超时失败，打断/停止可终止等待。 |

本分支 `config.yaml` 明确选择 `as_generated`，用于 Darvin 文件接收。
旧浏览器存在播放缓存上限，因此部署到实时播放器时应选择 `paced`。
接收完成不等于扬声器已播放完成。订阅匹配也不等于应用确认收包，
仍需核对接收端 PCM、帧数和 EOF。共享 TTS 文件仅接入 MOSS 条件路径，
其他引擎保留原预缓冲和发送行为；`main.py` 未修改。

文本分块仍优先沿用原模型的约 75 token 分段。保护数字单位或泰语边界
需要重新分段时，不再把整段直接合并到 prefill 最大容量；有效原切点
会被保留，并过滤空段和独立标点段。单个不可安全切开的保护串仍受
引擎硬容量限制。375 帧不完整生成检查保留，不将截断音频判为完整。

模型和 TRT 引擎、原 zh_5 音色、采样参数均未因此重建或修改。
MOSS 日志新增分块 token 数和生成帧数；启动启用 Python faulthandler
以便将后续原生异常线程栈保存在容器日志。退出 139 的具体原因尚未
确定，本次修改不宣称修复原生崩溃、所有提前 EOS 或所有语种音质问题。
