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
  -t lanxijiang-vits2-jp61:zh-en-v2 .
```

`config.yaml` 只启用 TTS，读取 `MCP_PORT` / `WS_PORT` 环境变量。
`preload_tts: true` 使模型安装、加载和固定文本预热在 HTTP/WS 监听前完成；
初始化失败或超时会使进程启动失败。首个用户请求因此使用已加载的模型。

发送保持 100ms PCM 帧，默认预缓冲 10 帧；同一话题共享发送时钟，跨句及
`stop/start` 按实际 PCM 字节数估算剩余播放时长，限制估计领先量为 1 秒。
这不会修改 PCM 或音频播放速度，也不表示收到了播放器实际缓冲反馈。

Darvin 采用个人仓库、实际推送 commit 和 `JP_VERSION=6`；通过 Darvin
后再提 PR。适配器测试、MCP/DDS 本地测试和平台正式成绩应分别记录。

### 提交前验证（2026-10-10）

JP6.1 Orin 上的本分支镜像已构建并完成单实例 MCP/DDS 测试：10 条人工
文本全部成功，TTFT 0.508～0.644 秒，端到端 RTF 0.910～0.996。
反复 `start/speak/stop` 使用同一实例和话题；全部接收帧数、字节数与
服务端相符，PCM 与原 FP16 适配器输出逐字节一致。冷启动约 27.3 秒，
包括真实下载、加载和固定文本预热，完成后才开放 HTTP。

56 项生命周期、发送时钟和预热回归测试通过。上述为本地验证，Darvin
尚未提交；长文本 RTF 余量较小，GPU 占用准入及平台评分仍待正式评测。
共享机器采样峰值可超过 10%，包含等待期的平均占用不作为达标依据。

提交旧 TTS 榜时，`PHANTHYMOTUS_REPO` 填个人公开仓库地址，
`PHANTHYMOTUS_COMMIT_ID` 填已推送的完整 Git commit，
`TTS_PLUGIN=tts`、`JP_VERSION=6`。平台按代码构建镜像，本次无需另传镜像。

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
