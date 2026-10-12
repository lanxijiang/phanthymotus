# VITS2 TensorRT TTS

Chinese-English VITS2 speech synthesis (16 kHz, single speaker) for Jetson
Orin. It is one implementation of the standard `tts` plugin, not a second MCP
plugin — `plugins/tts.py` owns the tool contract and picks the engine.

## 本分支部署：JP6.1 中英文

本分支使用 VITS2 单音色中英文模型。完整 TTS 网络分为
`encoder_duration.plan`、`flow.plan`、`decoder.plan` 三个 TRT 文件，
合计 **55,770,748 B**。按 2026-10-10 的项目确认，允许适当超过原 50MB
要求，继续采用已有 FP16 权重；未采用局部 INT8 实验。

前端字典、FST、NLTK/G2P 资源及镜像另计。三份 TTS 网络在 Orin GPU
执行；文本前端与 iSTFT 在 CPU，英文未登录词的 `g2p_en` 还包含 CPU GRU。

镜像复用固定 digest 的官方 JP6.1 perception 运行环境，复制本分支应用代码，
模型通过 JuiceFS HTTP 按字节数和 SHA256 校验下载。构建不复制模型、不执行
模型推理或 TRT 编译。当前这份 Dockerfile 只接受 JP6.1：

```bash
docker build -f perception/Dockerfile.jetson \
  --build-arg JP_VERSION=61 \
  -t lanxijiang-vits2-jp61:rtf-20261012 .
```

`config.yaml` 只启用 TTS，读取 `MCP_PORT` / `WS_PORT` 环境变量。
`preload_tts: true` 使模型安装、加载和固定文本预热在 HTTP/WS 监听前完成；
初始化失败或超时会使进程启动失败。首个用户请求因此使用已加载的模型。

发送保持 100ms PCM 帧，默认预缓冲 10 帧；同一话题共享发送时钟，跨句及
`stop/start` 按实际 PCM 字节数估算剩余播放时长，限制估计领先量为 1 秒。
这不会修改 PCM 或音频播放速度，也不表示收到了播放器实际缓冲反馈。
订阅连续匹配等待默认 100ms，与首段合成并行；可通过
`MIX_VITS_SUBSCRIBER_SETTLE_MS` 配置部署所需的稳定等待时间。

Darvin 采用个人仓库、实际推送 commit 和 `JP_VERSION=6`；通过 Darvin
后再提 PR。适配器测试、MCP/DDS 本地测试和平台正式成绩应分别记录。

### 本次 RTF 发布验证（2026-10-12）

最终应用镜像 `lanxijiang-vits2-jp61:rtf-20261012` 已在 Orin 构建；
使用镜像内代码、原 FP16 GPU 引擎、默认语速 1.0、默认关闭的首句实验，
三个独立容器同步执行四条自编中英文长短文本，各两轮，共 24 次请求：

- 成功 24/24，接收端首 PCM **0.149～0.450 秒**，RTF **0.875～0.988**。
- 全部帧数、字节数和 EOF 完整；每条文本六份 PCM 相同，且与此前
  100ms 等待版本的相同文本输出逐字节一致。
- 最小“音频时长减最后 PCM 接收耗时”余量为 **0.524 秒**。
  三实例请求累计重叠约 181.4 秒，请求发起最大偏差 27.4ms。
- 下载、加载及固定文本预热先于 HTTP 开放，测试结束后三个容器正常退出。

这是最终镜像的本地 MCP/DDS 验证，尚未代表该版本通过 Darvin。
每测试容器限制 2 CPU / 2GiB、使用 bridge 网络；平台 host 网络与独立
评估容器存在差异。四条文本的重复测试不能代替平台完整用例。
共享主机可用内存最低 542136KiB，RAM 峰值 6948/7620MB，包含其他服务；
不据此认定 GPU 占用限制通过。长音频的 RTF 仍接近 1，需要线上复测。

以下为此前版本和各阶段实验的历史记录。

### 提交前验证（2026-10-10）

JP6.1 Orin 上的本分支镜像已构建并完成单实例 MCP/DDS 测试：10 条人工
文本全部成功，TTFT 0.508～0.644 秒，端到端 RTF 0.910～0.996。
反复 `start/speak/stop` 使用同一实例和话题；全部接收帧数、字节数与
服务端相符，PCM 与原 FP16 适配器输出逐字节一致。冷启动约 27.3 秒，
包括真实下载、加载和固定文本预热，完成后才开放 HTTP。

56 项生命周期、发送时钟和预热回归测试通过。上述为提交前的本地验证；
长文本 RTF 余量较小，GPU 占用准入不能据此认定通过。
共享机器采样峰值可超过 10%，包含等待期的平均占用不作为达标依据。

提交旧 TTS 榜时，`PHANTHYMOTUS_REPO` 填个人公开仓库地址，
`PHANTHYMOTUS_COMMIT_ID` 填已推送的完整 Git commit，
`TTS_PLUGIN=tts`、`JP_VERSION=6`。平台按代码构建镜像，本次无需另传镜像。

### Darvin 反馈与首音优化（2026-10-12）

提交 `fc0c2df` 的平台日志显示：三实例共 30 次请求、合成失败 0 次，
但部分 RTF 超过 1。可见的实例 2 中，首音最慢为 8.867 秒；其 10 条
音频中有 7 条通过 RTF 筛选进入标注。该日志的首音筛选阈值为
999999 秒，不能把“TTFT 不达标数 0”解释成首音小于 1 秒。
上传重试后已成功建立标注任务；日志中没有最终音质成绩。

另一份“中英文2”日志也使用旧 `fc0c2df`，为 16 条文本、三实例共 48
次请求，合成失败 0 次。汇总记录 RTF 不达标数 1；可见实例 2 的 case 4
首音 1.659 秒、音频 15.20 秒、发送完成 15.859 秒、RTF 1.0434。
该实例 16 条音频中 15 条进入标注，任务 29627 创建成功。两次平台日志
均不包含下面的本地改进，不能作为改后版本的成绩。

本次在原 FP16 引擎上优化前端等待：

- 分句按需转换音素，准备好首段及合并所需的下一段后即可合成。
  分句、贪心合并和随机采样规则保持一致，不再预先处理所有后续段落。
- 同一次请求内复用相同英文未登录词的原始 G2P 结果，区分 predictor
  和大小写；缓存最多 512 项。中文变调、特殊发音规则继续走原流程。
- 每段记录 `frontend_ms`、`engine_ms`，首段另记录锁等待与首 PCM
  就绪耗时。`engine_ms` 包含 TRT、数据传输和 NumPy/iSTFT，
  不等于纯 GPU kernel 耗时；首 PCM 就绪也不等于接收端 TTFT。
- 镜像默认 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。实测原
  NumPy OpenBLAS 线程数为 6；单实例容器 CPU 配额为 2 核时，限制
  辅助运算线程进一步降低了耗时。三个 TTS 网络继续使用 TensorRT GPU。
- 时长对齐用整数 duration 直接展开，避免构造稠密对齐矩阵再乘两次；
  不能证明等价的异常输入保留原路径。
- iSTFT 保留浮点累加顺序，改成按窗口位置批量重叠相加。当前 FFT16
  只需 16 次 Python 循环，替代数万次逐帧循环；仍使用相同频谱、窗函数、
  归一化与 PCM 转换，无需重新构建 TRT。

预热、10 帧预缓冲及 DDS 发送节奏沿用已有设置。按需分句意味着
尾段若有前端错误，会在处理该段时报告，前段可能已经输出。
三实例并发和正式 Darvin 达标必须另行验证。

在共享 Orin 上，基线 `fc0c2df` 与改后版本分别运行四条自编中英文文本，
各两轮；两版分别用独立进程、同一 FP16 引擎和相同随机种子。18 项真实
前端测试及 15 项对齐/iSTFT 测试通过，无跳过；8 次输出的分句、音素及
完整 PCM 逐字节一致。全部优化后的原始适配器结果（两轮均值，秒）：

| 文本 | 原版首 PCM | 改后首 PCM | 改后原始合成 RTF |
|---|---:|---:|---:|
| 中文短句 | 0.294 | 0.099 | 0.016 |
| 英文含重复未登录词 | 0.537 | 0.157 | 0.014 |
| 中文多句长文 | 0.635 | 0.279 | 0.019 |
| 英文多句长文 | 0.562 | 0.132 | 0.014 |

以上不含 MCP/DDS、发送节奏与三实例竞争，不能用原始合成 RTF 替代平台
RTF；共享机器上的小样本结果也不能保证所有文本获得同等加速。

第一轮（前端/线程优化）以现有 JP6.1 镜像挂载改后源码、注入相同单线程环境，完成四条文本的
真实 MCP/DDS 单实例检查：接收端 TTFT 0.506～0.645 秒、RTF
0.935～0.990，四条全部成功，帧数与字节数完整，接收 PCM 与基线逐字节
一致。这是源码运行验证，本轮未重新构建镜像、未复测三实例。

包含对齐/iSTFT 优化后的同条件复测也为四条全部成功：接收端 TTFT
0.512～0.515 秒、RTF 0.934～0.989，帧数与字节数完整，接收 PCM
与原提交基线逐字节一致。此轮依然是现有镜像挂载源码的单实例验证。
长文本的端到端 RTF 仍受实时发送节奏影响而接近 1；上述单实例结果不能
据此认定正式 Darvin 已达标。后续三实例对照见下文。

### RTF 三实例对照（2026-10-12）

在以上计算优化的基础上，将连续订阅匹配等待由 500ms 降至 100ms。
保持原 FP16 TRT 引擎、文本切分、语速、100ms 帧节奏、10 帧预缓冲和
1 秒共享播放领先上限。三个独立容器同时请求四条自编中英文长短文本，
500ms 组一轮共 12 次，100ms 组两轮共 24 次：

| 指标 | 500ms 等待 | 100ms 等待 |
|---|---:|---:|
| 接收端首 PCM | 0.507～0.523 秒 | 0.148～0.441 秒 |
| 接收端 RTF | 0.934～0.989 | 0.875～0.987 |
| 最小 RTF 余量（音频时长减接收完成耗时） | 0.399 秒 | 0.540 秒 |
| 成功请求 / PCM 与原提交逐字节一致 | 12/12 | 24/24 |

同步屏障位于 `start` 和新 DDS reader 创建之前，未提前等待订阅发现；
三实例实际请求发起时间最大相差 10.2ms，候选组共同处理请求累计约
181.5 秒。候选最大接收帧间隔 103.3ms，全部帧数、字节数完整，六个测试
容器均正常退出。另有无同步屏障的 100ms 单实例检查，四条 PCM 也与原
提交一致；首音 0.119～0.316 秒、RTF 0.870～0.984。

这是现有 JP6.1 镜像挂载源码验证，尚未重建镜像或在 Darvin 复测。
本地采用 bridge 网络、每容器服务和接收端两个进程、2 CPU/2GiB 限额；
平台日志采用 host 网络和分开的服务/评估容器，且未显示相同资源限额。
只有四条文本，不能证明所有文本或长期运行都达标。候选测试全机最小
可用内存 565752KiB，采样 RAM 峰值 6921/7620MB，包含其他后台服务；
这不是模型独占显存，也不是 GPU 占用准入证明。

上述为已完成试验的历史记录；本地汇总 JSON 和旧音频已于 2026-10-12
按用户要求清理。当前先优化 RTF，再进行自然度和流畅度的听评与调整。

### 中文前端复用与自然度试听（2026-10-12）

继续在同一请求内复用标点分隔的完整中文片段，保留原分词、拼音和变调
算法；独立 LRU 最多 256 项，每项最多 512 字符，跨请求不保留。
CUDA session 缓存引擎张量元数据及已设置的输入形状，并修复缓冲增长失败
后的清理。实际 encoder 的五个输出仍全部使用，没有省略输出传输。

在相同原 FP16 引擎、相同单线程配置下，与此前 100ms 等待版本比较：
四条文本各两轮的分段、音素和完整 PCM 全部一致；设备环境内 66 项回归
测试通过、无跳过。中文长文 G2P 两轮均值由 196ms 降至 54ms，原始首
PCM 由 275ms 降至 248ms；英文没有观察到稳定的首音改善。本轮只测
单实例适配器，不包含 DDS 发送，也没有复测这两项改动的三实例平台 RTF。

另有默认关闭的 `MIX_VITS_FIRST_SENTENCE=1` 试听选项：首个完整句子
若未超过 token 限制则单独合成，剩余文本沿用原分段；不确定的英文缩写
和过长首句按原逻辑处理。它会改变模型上下文与采样结果，不能视为 PCM
等价优化，也不能保证自然度提升。生产默认仍为关闭、语速 1.0。

同一中英文问句各生成三版：A 原分段/1.0，B 首句独立/1.0，C 首句独立/0.96。
问号在实际归一化文本中保留；六条 WAV 无满幅削波，主观自然度尚待试听。
略慢语速只调整音素时长，不直接控制重音或语调。用户反馈这组候选仍偏机械；
分句/慢速旧试听、历史日志和快照压缩包已清理，不再保留“验证记录”目录。

后续以相同音素和参数核对原 FP32 ONNX 与现有 FP16 TRT：两条中英文
文本各一个时长位置相差 16ms。TRT 在 Orin 运行，ONNX 在 A100 卡3
运行，关闭 TF32，逐节点确认浮点网络计算走 CUDA；形状辅助运算另行记录。
这不是平台性能测试，也没有给出主观自然度结论；两端共用当前前端/iSTFT，
还不能据此确认机械感来源。部署仍使用原 TRT 与默认语速。
当前四条对照试听直接放在 `evaluation/old-benchmark/results/` 根目录，
按“中文/英文 + 01 当前TRT、02 原始ONNX”命名。精简诊断记录位于
`evaluation/old-benchmark/vits2-precision-audit.json`，脚本为
`evaluation/old-benchmark/compare_vits2_precision.py`，均在实验工作区。

### A landmine worth knowing

`g2p_en` calls `nltk.download()` at **module import** if the corpora are not on
`nltk.data.path`. `frontend/english.py` therefore inserts the release's
`nltk_data` directory before importing it, and nothing may import `g2p_en`
earlier — including build-time checks. This Dockerfile inherits the pinned
runtime dependencies and does not import `g2p_en` during the image build.

## Configure

The engine is a `configSchema` field, so it is visible and switchable in the
device panel:

```yaml
plugins:
  tts:
    enabled: true
    engine: vits2-zh-en
    backend: trt           # VITS2 requires trt
    model_dir: /models/vits2
    speed: 1.0
    warmup: true
```

`action=config` with `tts_engine` switches engines at runtime: the outgoing
engine's nodes are disposed first (two live publishers on one audio topic is
the duplicate-audio failure mode), then the incoming one is built in the
background. `speaker_id` must stay 0 — the model has one voice.

`speak` keeps the standard action-completion contract: a `speak-*` action id,
`interrupt` support, and the end-of-utterance marker on every terminated
utterance, including one abandoned before it reached a node.

## Model release

`utils.model_downloader.ensure_vits2_model()` installs one archive per JetPack
family (~60 MB), pinned by size and SHA256, using the same lock /
staging / retry path as the other verified models. This branch defaults JP6.1
to the verified personal JuiceFS mirror in `utils/model_downloader.py`;
`VITS2_MODEL_BASE_URL` remains an optional override. It returns the engine
directory for the TensorRT that is actually importable, so `engines/jp61` and
`engines/jp511` can never be confused — TensorRT plans are not portable across
TensorRT majors or GPU architectures.

The archive unpacks to `config.json`, `engines/<family>/`, `frontend_data/`,
`tn_cache/` and `nltk_data/`. The NLTK corpora ship with it precisely so the
container never calls `nltk.download()` at runtime. Upstream is
[Starlight777/VITS2-ZH-EN-Male-16k](https://www.modelscope.cn/models/Starlight777/VITS2-ZH-EN-Male-16k)
at revision `14954122c4baf4e80b44436c4b2b167e38db4103`; the runtime-required
files of that revision were repacked and mirrored to COS.

With `preload_tts` disabled, installation happens in a background loader on the first `start`/`speak`, which
returns `{"state": "loading"}` immediately — the download plus three TensorRT
engines plus warmup takes far longer than the 60 s Agent Core allows a
processor `tools/call`. Instances that were requested while the model loaded
come up on their own when it is ready, and utterances queued in the meantime
play then. `info` never blocks and never triggers a load.
