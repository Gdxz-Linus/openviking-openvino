# OpenViking OpenVINO 嵌入后端

给 [OpenViking](https://github.com/volcengine/OpenViking) 增加 OpenVINO 本地嵌入能力，提供两种部署形态，并附带两个让记忆抽取模型自动跟随的工具。

OpenViking 0.5.3 原生只支持 llama-cpp（GGUF）本地嵌入。本仓库补上 OpenVINO Runtime 后端，让 Intel CPU、核显或 NPU 可以直接跑 int8 量化模型；同一份模型既能进程内推理，也能作为独立 HTTP 服务被调用。

## 为什么需要它

OpenViking 官方默认的本地嵌入模型是 `bge-small-zh-v1.5`：512 维、512 token 上下文。开箱即用，但对记忆系统来说上限太低——超过 512 token 的内容会被直接截断，长文档、长会话的摘要与检索都会丢信息。

换成 4096 token 上下文、1024 维的模型后能力上去了，原来的纯 CPU 推理方案却顶不住。实测条件：llama.cpp，677M 参数 Q8 量化模型，22 逻辑核 CPU，吞吐 151.7 token/s，`n_ctx` 与 `n_batch` 都设为 4096。

| 文本长度 | CPU 嵌入耗时 |
| --- | --- |
| 512 token | 2.8 秒 |
| 2520 token | 16.3 秒 |
| 4000 token | 32.2 秒 |

一份接近 4096 token 的文档要 30 秒以上，批量重建向量库时基本不可用。注意力计算是 O(n²)，CPU 上越长越陡。

于是只有两条路：退回 512 上下文，牺牲能力；或者上 GPU，保住能力。本仓库选后者——用 OpenVINO 把同一个 int8 模型跑在 Intel 核显上，同样一份 4096 token 文本从 30.1 秒降到 4.0 秒。

这就是本项目的必要性：不是为了跑分，而是为了让 4096 上下文的本地嵌入真正可用。

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

以下模型都已核实仓库地址，均为原始格式，需要转换成 OpenVINO IR。两款支持 32768 上下文，三款支持 8192。

| 模型 | 维度 | 上下文 | 原始仓库 | 国内镜像 | 特点 |
| --- | --- | --- | --- | --- | --- |
| harrier-oss-v1-0.6b | 1024 | 32768 | [huggingface.co/microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | [hf-mirror.com/microsoft/harrier-oss-v1-0.6b](https://hf-mirror.com/microsoft/harrier-oss-v1-0.6b) | 微软，检索专用，MTEB v2 得分 69.0，查询侧必须加指令前缀 |
| Qwen3-Embedding-0.6B | 1024（可自定义 32 到 1024） | 32768 | [huggingface.co/Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B](https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) | 阿里，100 多种语言，月下载 876 万 |
| bge-m3 | 1024 | 8192 | [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | [hf-mirror.com/BAAI/bge-m3](https://hf-mirror.com/BAAI/bge-m3) | 智源，多语言，不需要指令前缀，月下载 2928 万 |
| gte-multilingual-base | 768 | 8192 | [huggingface.co/Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | [hf-mirror.com/Alibaba-NLP/gte-multilingual-base](https://hf-mirror.com/Alibaba-NLP/gte-multilingual-base) | 阿里通义，70 多种语言，编码器架构推理更快，月下载 88 万 |
| jina-embeddings-v2-base-zh | 768 | 8192 | [huggingface.co/jinaai/jina-embeddings-v2-base-zh](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh) | [hf-mirror.com/jinaai/jina-embeddings-v2-base-zh](https://hf-mirror.com/jinaai/jina-embeddings-v2-base-zh) | Jina AI，中英双语，8192 序列长文本，需登录 Hugging Face 才能下载 |

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

以下数据全部在 `max_input_tokens = 4096` 的设置下测得，也就是模型窗口被限制在 4096 token。同一模型 `harrier-ov-int8`、同一文本，唯一变量是 OpenVINO 设备（Intel Arc 核显对比 CPU）。

| 文本长度 | 说明 | CPU | GPU | 提速 |
| --- | --- | --- | --- | --- |
| 30 字 | 短查询 | 55.5 ms | 34.5 ms | 1.6 倍 |
| 1800 字 | 中等篇幅 | 4255.8 ms | 672.6 ms | 6.3 倍 |
| 6600 字 | 触顶 4096 token，实际被截断 | 30149.6 ms | 3985.5 ms | 7.6 倍 |

三档都在 4096 上下文内完成，最后一档已经顶到上限，再长的文本不会更慢，只会被截断。

端到端效果：重建 200 条记忆的向量库从约 20 分钟降到约 3 分钟。短查询收益有限，因为 GPU 固定开销占比高；长文档与批量嵌入才是 OpenVINO 的价值所在。

### 上下文长度与检索效果

上下文开得越大，能完整入库的内容越多，但嵌入耗时超线性增长。下表是 2026 年 11 月在本机记忆库上的实测。

| 上下文 | 内容完整入库 | 长文排第 1（头/中/尾） | 长文进前 3（头/中/尾） | 单次嵌入 12647 字符 | 重建 135 条 |
| --- | --- | --- | --- | --- | --- |
| 512（OpenViking 默认） | 49.3%，134 个文件中 68 个被截断，被截断者平均只保留 49.6% | — | — | 2.0 秒 | 约 5 分钟 |
| 1024 | — | — | — | 4.5 秒 | 约 10 分钟 |
| 2048 | 覆盖到 P90（1726 token） | — | — | 约 7 秒 | 约 20 分钟 |
| 4096（本仓库采用） | 96.8%，158 个文件中 5 个超限 | 67% / 0% / 0% | 100% / 100% / 100% | 27 秒 | 约 60 分钟 |
| 8192 | 99.1%，只多覆盖 2 个文件 | 未单独测 | 未单独测 | 67 秒 | 约 150 分钟 |

几点说明：

- 512 档的 49.3% 是拆分前的统计口径。当时 134 个记忆文件里 68 个被截断，合计丢失 12.2 万字。
- 4096 档的 96.8% 对应 158 个文件，token 分布为 P50 935、P90 2402、P99 7159、P100 8739。
- 长文检索每档样本 6 个，数量小，只能定性参考，不能当统计结论。
- 「排第 1」指目标文件排在结果首位，「进前 3」指它出现在前三条里。实际使用中只要进前 3 就能读到。

**为什么停在 4096。** 从 4096 提到 8192，单次嵌入从 3.9 秒涨到 12.1 秒，是原来的 3.1 倍，只多覆盖 2 个文件，而每次查询都要付这个代价。更划算的做法是把个别超限文件拆开：实测把 5 个超长文件拆分后，170 个文件全部在 4096 以内，12 个新拆分文件召回全部进前 2（6 个第 1、6 个第 2），内容 100% 参与嵌入且不增加任何耗时。

**8192 与 32768 的差别。** 上表只测到 8192。再往上开，嵌入耗时按平方增长，而本机记忆库最长文件 8739 token，32768 的窗口用不到一半。选 32768 的模型是为了让长文档有充足余量，不代表要配到 32768。

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
