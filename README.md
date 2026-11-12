# OpenViking OpenVINO 嵌入后端

给 [OpenViking](https://github.com/volcengine/OpenViking) 增加 OpenVINO 本地嵌入能力，提供两种部署形态，并附带两个让记忆抽取模型自动跟随的工具。

OpenViking 0.5.3 原生只支持 llama-cpp（GGUF）本地嵌入。本仓库补上 OpenVINO Runtime 后端，让 Intel CPU、核显或 NPU 可以直接跑 int8 量化模型；同一份模型既能进程内推理，也能作为独立 HTTP 服务被调用。

## 适用对象

| 你的情况 | 是否适合 |
| --- | --- |
| 用 OpenViking 做本地记忆或知识库，机器是 Intel CPU、核显或 NPU | 适合 |
| 想摆脱 llama.cpp 在 Intel 平台上的性能限制 | 适合 |
| 需要在进程内与 HTTP 两种形态之间切换 | 适合 |
| 用 NVIDIA 显卡 | 不适合。llama.cpp 的 CUDA 后端更直接，OpenVINO 没有优势 |
| 用 AMD 显卡 | 不适合。OpenVINO 对 AMD 的支持有限 |
| 只想用 OpenViking 默认配置 | 不需要。官方内置 bge-small 开箱即用 |
| 不愿承担升级后重打补丁的维护 | 谨慎。本仓库是补丁形态，不是上游内置能力 |

本仓库由三部分组成，各自独立可用，不必全装：

| 部分 | 解决什么 | 谁需要 |
| --- | --- | --- |
| 嵌入后端补丁 | 让 OpenViking 支持 OpenVINO（进程内或 HTTP） | 想在 Intel 平台跑本地嵌入的人 |
| HTTP 嵌入服务 | 把 OpenVINO 包成 OpenAI 兼容接口 | 要用 HTTP 形态、或多实例共享一块 GPU 的人 |
| 记忆模型同步插件 | 让记忆抽取模型跟随主系统 | 同时跑 Hermes 或 DSH 的人 |

## 基础条件

| 项目 | 要求 | 检查方法 |
| --- | --- | --- |
| OpenViking | 0.5.3（补丁锚点按该版本编写） | `pip show openviking` |
| Python | 3.13（用 OpenViking 自带 venv 即可） | — |
| OpenVINO Runtime | 2024 及以上 | `python -c "import openvino; print(openvino.__version__)"` |
| 模型 | OpenVINO IR 格式（xml、bin、tokenizer） | 见下方模型表 |
| 硬件 | Intel CPU 可跑；核显或 NPU 需要对应驱动 | `python -c "import openvino as ov; print(ov.Core().available_devices)"` |
| 磁盘 | 模型约 600 MB 起，编译缓存约 2 GB | — |
| 内存 | 进程内形态常驻约 1.6 GB | 任务管理器观察 |

前端给出的 `available_devices` 里应包含 `GPU` 或 `CPU`。只有 `CPU` 时功能正常，长文本会明显变慢。

## 模型下载

以下模型都已核实仓库地址，均为原始格式，需要转换成 OpenVINO IR。

| 模型 | 维度 | 上下文 | 原始仓库 | 国内镜像 | 特点 |
| --- | --- | --- | --- | --- | --- |
| harrier-oss-v1-0.6b | 1024 | 32768 | [huggingface.co/microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | [hf-mirror.com/microsoft/harrier-oss-v1-0.6b](https://hf-mirror.com/microsoft/harrier-oss-v1-0.6b) | 微软，检索专用，MTEB v2 得分 69.0，查询侧必须加指令前缀 |
| Qwen3-Embedding-0.6B | 1024（可自定义 32 到 1024） | 32768 | [huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B](https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) | 阿里，100 多种语言，月下载 876 万 |
| bge-m3 | 1024 | 8192 | [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | [hf-mirror.com/BAAI/bge-m3](https://hf-mirror.com/BAAI/bge-m3) | 智源，多语言，不需要指令前缀，月下载 2928 万 |
| bge-small-zh-v1.5 | 512 | 512 | [huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf](https://huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf) | 同路径换 hf-mirror | OpenViking 官方默认，GGUF 格式，走 llama.cpp 而非 OpenVINO |

### 转换成 OpenVINO IR

```bash
pip install optimum[openvino]

optimum-cli export openvino \
  --model microsoft/harrier-oss-v1-0.6b \
  --task feature-extraction \
  --weight-format int8 \
  D:/OpenViking/models/harrier-ov-int8
```

导出后按本仓库要求的目录结构摆放：

```
D:/OpenViking/models/harrier-ov-int8/
├── int8/
│   ├── openvino_model.xml
│   └── openvino_model.bin
└── tokenizer.json
```

如果导出工具把 xml 与 bin 直接放在输出目录根下，把它们移到 `int8/` 子目录，或者修改 `ov.conf` 的 `cache_dir` 与模型目录名来匹配。

`--weight-format int8` 是量化开关，不加则导出 fp32，体积约为 int8 的四倍。转换过程需要下载原模型，国内网络建议先设 `HF_ENDPOINT=https://hf-mirror.com`。

## 两种形态

| | 进程内 | HTTP 服务 |
| --- | --- | --- |
| 嵌入调用 | OpenViking 进程内直调 OpenVINO Runtime | 独立 FastAPI 服务（8010），OpenViking 走 `provider: openai` |
| 检索中位延迟 | 43 ms | 273 到 386 ms |
| 额外进程 | 无 | 一个嵌入服务进程 |
| 配置 | `provider: local` | `provider: openai` + `api_base` |
| 适用 | 长期生产 | 回退、跨进程复用、多实例共享一台 GPU |

HTTP 形态慢在 FastAPI 与 Pydantic 的请求处理，实测 6 到 9 倍差距，属于回退方案，不建议长期使用。

## 性能数据

同一模型 `harrier-ov-int8`、同一文本，只切换 OpenVINO 设备（Intel Arc 核显对比 CPU）：

| 文本长度 | CPU | GPU | 提速 |
| --- | --- | --- | --- |
| 30 字 | 55.5 ms | 34.5 ms | 1.6 倍 |
| 1800 字 | 4255.8 ms | 672.6 ms | 6.3 倍 |
| 6600 字（触顶 4096 token） | 30149.6 ms | 3985.5 ms | 7.6 倍 |

端到端效果：重建 200 条记忆的向量库从约 20 分钟降到约 3 分钟。短查询收益有限，因为 GPU 固定开销占比高；长文档与批量嵌入才是 OpenVINO 的价值所在。

## 目录结构

```
openviking-openvino/
├── patches/                     OpenViking 服务本体补丁
│   ├── patch_openvino.py        进程内 OpenVINO 后端（7 处改动）
│   └── patch_http_keepalive.py  HTTP 客户端保活（仅 HTTP 形态需要）
├── http-server/                 HTTP 形态的独立嵌入服务
│   ├── embed_server.py
│   └── start-embed.cmd
├── configs/                     两套配置模板
│   ├── ov.conf.local.example
│   └── ov.conf.http.example
├── plugins/
│   ├── hermes-vlm-follow/       让 OpenViking 的记忆抽取模型跟随 Hermes
│   └── dsh-vlm-follow/          同上，跟随 DSH
├── docs/
│   ├── 部署指南.md
│   ├── 性能数据.md
│   └── 回退与排障.md
└── models/
    └── 模型获取.md
```

## 快速开始：进程内形态

1. 准备好 OpenVINO IR 模型，放到 `cache_dir` 下（见 `models/模型获取.md`）。
2. 应用后端补丁：

```bash
python patches/patch_openvino.py --check    # 先看状态
python patches/patch_openvino.py            # 打补丁
```

3. 用 `configs/ov.conf.local.example` 生成 `ov.conf`，填入自己的模型路径与 VLM 凭据。
4. 重启 OpenViking 服务。启动日志应出现 `OpenVINO embedder ready: model=... device=GPU`。

## 快速开始：HTTP 形态

1. 启动嵌入服务：

```bash
python http-server/embed_server.py --model-dir D:/OpenViking/models/harrier-ov-int8 --port 8010
```

2. 应用保活补丁（否则批量写入会频繁重连）：

```bash
python patches/patch_http_keepalive.py
```

3. 用 `configs/ov.conf.http.example` 生成 `ov.conf`。
4. 重启 OpenViking，确认 `/health` 与嵌入接口可达。

## 升级后必须重放

`pip install --upgrade openviking` 会整包替换 `site-packages/openviking/`，两个补丁都会丢失。升级后按顺序重跑：

```bash
python patches/patch_openvino.py
python patches/patch_http_keepalive.py    # 仅 HTTP 形态
```

两个脚本都幂等，已打过会直接跳过；锚点不匹配时报错退出，不会盲改。

## 记忆抽取模型同步

OpenViking 的记忆抽取（`session_commit`）调用一个大模型生成摘要，端点写在 `ov.conf` 的 `vlm` 段。手工维护很容易与主系统脱节：主系统换了模型，`ov.conf` 没变，记忆抽取持续失败，而且只体现在日志里。

两个插件分别挂在 Hermes 与 DSH 的模型请求钩子上，把当前实际使用的端点写进 `ov.conf`：

| 插件 | 宿主 | 钩子 | 语言 |
| --- | --- | --- | --- |
| `hermes-vlm-follow` | Hermes | `pre_api_request` | Python |
| `dsh-vlm-follow` | DSH | `llm/stream` | JavaScript |

OpenViking 每 30 秒重载一次配置并比对差异，所以改完最多半分钟生效，不需要重启服务。

## 兼容性

- OpenViking 0.5.3（补丁锚点按该版本编写）
- OpenVINO Runtime 2024 及以上
- 模型：OpenVINO IR 格式，需与 `dimension` 配置一致

## 许可

MIT。补丁脚本与 HTTP 服务为本仓库原创；OpenViking 本身遵循其上游许可证。

## 上游

本仓库是对 OpenViking 的第三方扩展，与字节跳动官方无关。如果官方后续内置 OpenVINO 后端，本仓库会标注对应版本并转入维护状态。
