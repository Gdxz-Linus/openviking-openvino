# OpenViking 服务的 OpenVINO 嵌入后端

[English](README.en.md) | 中文

OpenViking 默认的本地嵌入模型是 `bge-small-zh-v1.5`，512 维、512 上下文、CPU 推理。对记忆系统来说这个上限太低：超过 512 的内容会被直接截断，长文档与长会话的摘要和检索都会丢失大量信息，也不适合建知识库。

换用 4096 上下文、1024 维的模型能解决质量问题，但 llama.cpp 纯 CPU 推理太慢。实测条件：677M 参数 Q8 GGUF 模型，22 逻辑核 CPU，吞吐 151.7 token/s，`n_ctx` 与 `n_batch` 都设为 4096。

| 文本长度 | CPU 嵌入耗时 |
| --- | --- |
| 512 token | 2.8 秒 |
| 2520 token | 16.3 秒 |
| 4000 token | 32.2 秒 |

一份接近 4096 上下文的文档要 30 秒以上，批量重建向量库要 40 分钟以上。

本项目用 OpenVINO 把同一个 int8 模型跑在 Intel 核显上，同样的 4096 token 文本从 30.1 秒降到 4.0 秒。Intel CPU、GPU 或 NPU 都可以直接跑 int8 量化模型，提供进程内与 HTTP 两种部署方式，并附带让记忆抽取模型跟随宿主 agent 的工具。

## 适用对象

| 你的情况 | 是否适合 |
| --- | --- |
| 用 OpenViking 做本地记忆或知识库，机器是 Intel CPU、GPU 或 NPU | 适合 |
| 想摆脱 llama.cpp 在 Intel 平台上的性能瓶颈 | 适合 |
| 需要在进程内与 HTTP 两种形态之间切换 | 适合 |
| 用 NVIDIA 显卡 | llama.cpp 的 CUDA 后端更直接，OpenVINO 没有优势 |
| 用 AMD 显卡 | 不适合。OpenVINO 对 AMD 的支持有限 |
| 只想用 OpenViking 默认配置 | 不需要。官方内置 bge-small 开箱即用 |
| 不愿承担升级后重打补丁的维护 | 谨慎。本仓库是补丁形态，不是上游内置能力 |

本仓库由三部分组成：

| 部分 | 解决什么 | 谁需要 |
| --- | --- | --- |
| 嵌入后端补丁 | 让 OpenViking 支持 OpenVINO（进程内或 HTTP） | 想在 Intel 平台跑本地嵌入的人 |
| HTTP 嵌入服务 | 把 OpenVINO 包成 OpenAI 兼容接口 | 要用 HTTP 形态、或多实例共享一块 GPU 的人 |
| 记忆模型同步插件 | 让记忆抽取模型跟随主系统 | 同时跑 Hermes 或 DSH 的人 |

## 基础条件

| 项目 | 要求 | 检查方法 |
| --- | --- | --- |
| OpenViking | 使用对应版本的本项目发行版即可，具体适配版本见该版本的发布说明 | `pip show openviking` |
| Python | 3.13（用 OpenViking 自带 venv 即可） | — |
| OpenVINO Runtime | 2026.4 | `python -c "import openvino; print(openvino.__version__)"` |
| 模型 | OpenVINO IR 格式（xml、bin、tokenizer） | 见下方模型表 |
| 硬件 | Intel CPU 可跑；GPU 或 NPU 需要对应驱动 | `python -c "import openvino as ov; print(ov.Core().available_devices)"` |
| 磁盘 | 模型约 600 MB 起，编译缓存约 2 GB | — |
| 内存 | 进程内形态常驻约 1.6 GB | 任务管理器观察 |

## 平台支持

| 平台 | 支持 | 说明 |
| --- | --- | --- |
| Windows 10/11 (x86_64) | 支持 | 完整支持，含 Intel 核显 GPU 加速 |
| Linux (x86_64) | 支持 | 完整支持，GPU 需装 Intel 显卡驱动 |
| macOS | 不支持 | OpenVINO 在 macOS 只提供 CPU 推理，拿不到 GPU 加速收益 |
| ARM (任何平台) | 不支持 | OpenVINO 在 ARM 上无 GPU 支持 |

## 嵌入模型下载

五款都可以直接用，任选一个。都是 int8 格式，国内网络用下面的镜像地址下载。

| 模型 | 权重 | 维度 | 上下文 | 输出与池化 | 查询前缀 | 语言 | 下载 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | 596M | 1024 | 32768 | 3 维，last-token | `Instruct: …` | 100 多种语言 | [下载](https://hf-mirror.com/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) |
| harrier-oss-v1-0.6b | 596M | 1024 | 32768 | 2 维，模型内置 | `Instruct: …` | 多语言 | [下载](https://hf-mirror.com/doldolggi/harrier-oss-v1-0.6b-OpenVINO) |
| bge-m3 | 568M | 1024 | 8192 | 3 维，CLS | 无 | 多语言 | [下载](https://hf-mirror.com/OpenVINO/bge-m3-int8-ov) |
| SEA-LION-ModernBERT-Embedding-600M | 612M | 1024 | 8192 | 3 维，mean | 无 | 东南亚 11 种语言与中文 | [下载](https://hf-mirror.com/aisingapore/SEA-LION-ModernBERT-Embedding-600M-OV) |
| embeddinggemma-2 | 744M | **768** | 8192 | 2 维，模型内置 | `task: …` | 100 多种语言 | [下载](https://hf-mirror.com/wesjos/embeddinggemma-2-openvino-int8) |

补丁已内置这五款，选哪款就在 `ov.conf` 中写对应的 `model` 与 `dimension`，无需自加表项。

这五款不是同一个模型换了权重，差异必须按模型分别配置，否则会算出错误的向量：

| 差异项 | 说明 |
| --- | --- |
| 维度 | 只有 embeddinggemma-2 是 768，其余四款 1024。写错会启动即报错 |
| 输出结构 | harrier 与 embeddinggemma-2 直接输出句子向量（2 维）；其余三款输出逐 token 隐状态（3 维），需自行池化 |
| 池化方式 | Qwen3 用 last-token，bge-m3 用 CLS，SEA-LION 用 mean。用错会显著降低检索质量 |
| 归一化 | Qwen3、bge-m3、SEA-LION 需自行归一化，否则余弦相似度失真；另两款模型内部已归一化 |
| 查询前缀 | harrier 与 Qwen3 需加 `Instruct: …` 前缀，bge-m3 要求不加，embeddinggemma-2 查询与文档两侧都要加 |
| 模型文件名 | 前四款为 `openvino_model.xml`，embeddinggemma-2 的文本塔为 `text_int8.xml` |

这些差异由补丁与 HTTP 服务按模型自动处理，只需在 `ov.conf` 中写对 `model` 与 `dimension`。


### 换模型要重建向量库

新模型的维度与池化方式可能不同，旧向量不能复用，需要重新嵌入一遍。这是换模型唯一的代价，提前留出时间。

### 模型转换

int8 格式的嵌入模型有限，也可自行转换合适的其他模型。转出来的模型不在补丁支持范围内，用 `custom` 入口挂载：在模型目录下放一个 `openvino_custom.json`（与 `tokenizer.json` 同级），`ov.conf` 里 `model` 写 `custom`、`model_path` 填 IR 文件绝对路径：

```json
{
  "pooling": "last",
  "normalize": true,
  "query_task": "",
  "query_prefix": "",
  "doc_prefix": ""
}
```

完整字段说明与探测方法见 `models/模型获取.md`。

## 两种接口

| | 进程内 | HTTP 服务 |
| --- | --- | --- |
| 嵌入调用 | OpenVINO Runtime | 独立 HTTP 服务（8010） |
| 检索中位延迟 | 43 ms | 173 到 286 ms |
| 额外进程 | 无 | 一个嵌入服务进程 |
| 配置 | `provider: local` | `provider: openai` + `api_base` |
| 适用 | 高效稳定生产 | 跨进程复用、多实例共享服务 |

HTTP 形态在请求处理上多一层开销，属备选方案。

## 性能数据

以下数据全部在上下文 4096 的设置下测得。同一模型 `harrier-ov-int8`，唯一变量是 Intel Arc GPU 对比 CPU。

| 文本长度 | 说明 | CPU | GPU | 提速 |
| --- | --- | --- | --- | --- |
| 30 字 | 短检索 | 55.5 ms | 34.5 ms | 1.6 倍 |
| 1800 字 | 中等篇幅 | 4255.8 ms | 672.6 ms | 6.3 倍 |
| 6600 字 | 触顶 4096 token，实际被截断 | 30149.6 ms | 3985.5 ms | 7.6 倍 |

三档都在 4096 上下文内完成，最后一档已经顶到上限，再长的文本不会更慢，只会被截断。

端到端效果：重建 200 条记忆的向量库从约 20 分钟降到约 3 分钟。检索文本长度越长越有优势。

### 上下文长度与检索效果

上下文开得越大，能完整入库的内容越多，但嵌入耗时增长更快。下表是在 328 个记忆文件上的实测结果。

测法与真实库一致：文件按该上下文长度切成块，每块一个向量。取 6 篇最长的文件，每篇各问 3 个问题，分别问开头、中间、结尾的内容，共 18 个问题。每个提问看前三条结果里有没有相关内容，同主题的其它文件也算，不要求是目标文件本身。

| 上下文 | 内容完整入库 | 开头提问 | 中段提问 | 结尾提问 | 合计 | 分块数 | 单次嵌入 12647 字符 | 重建 328 个文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512（OpenViking 默认） | 40.9%，298 个正文文件中 176 个被截断 | 4/6（1/6） | 2/6（1/6） | 3/6（0/6） | 9/18 | 673 | 2.0 秒 | 104 秒 |
| 1024 | 70.5%，88 个被截断 | 5/6（1/6） | 2/6（1/6） | 4/6（1/6） | 11/18 | 440 | 4.5 秒 | 153 秒 |
| 2048 | 96.3%，11 个被截断 | 4/6（0/6） | 2/6（1/6） | 5/6（0/6） | 11/18 | 343 | 约 7 秒 | 323 秒 |
| 4096 | 100% | 5/6（5/6） | 4/6（2/6） | 6/6（1/6） | 15/18 | 329 | 27 秒 | 983 秒 |
| 8192 | 100% | 5/6（5/6） | 4/6（2/6） | 6/6（1/6） | 15/18 | 329 | 67 秒 | 3434 秒 |

括号里的写法说明：`4/6（1/6）` 表示 6 次提问里有 4 次前三条出现相关内容，其中 1 次命中的正是目标文件本身。

几点说明：

- **4096 档 18 次提问有 15 次能找到相关内容**，其中 8 次直接命中目标文件。512 档只有 9 次能找到，差距明显。
- **开头内容在所有档位都能找到**（4 到 5 次）。文档开头那一块在任何上下文档里都被完整保留，低上下文不影响开头检索。
- **中段是低上下文的主要损失**。512 到 2048 三档只有 2 次能找到中段的相关内容，4096 档是 4 次。中段的具体细节在低上下文下进不了库。
- **结尾在 2048 档起明显改善**（从 3 次升到 5 次），因为 96% 的文件已经完整入库；4096 档 6 次全部能找到。
- **一个容易误读的点。** 低上下文里目标文件本身很少进前三条（512 档只有 3 次），但前三条往往是同主题的其它文件，用户实际能读到相关内容。只看目标文件排名会低估效果，所以上表以「有相关内容」为准，括号里另列目标文件本身的次数。
- **8192 与 4096 结果完全一致。** 库里最长的文件是 3893 token，全都在 4096 以内，8192 没有可多覆盖的内容，代价却是 3.5 倍建库时间。所以推荐 4096。
- 每档样本 6 篇、18 个问题，数量小，只能定性参考，不能当统计结论。

上下文从 4096 提到 8192，单次嵌入从 3.9 秒涨到 12.1 秒，是原来的 3.1 倍，不过是在后台进行，使用影响小。当前记忆库最长文件 3893 token，4096 已能全部覆盖。

## 记忆抽取模型同步

OpenViking 的记忆抽取要调用一个大模型生成摘要，端点写在配置里，没有选项能自动跟随。主系统换了模型而配置没跟着改，记忆抽取就会持续失败，而且只在日志里体现，对话本身看不出异常。

本项目为 Hermes 与 DSH 各提供一个插件，自动把宿主当前使用的模型写进 OpenViking 的配置：

| 插件 | 宿主 | 语言 |
| --- | --- | --- |
| `hermes-vlm-follow` | Hermes | Python |
| `dsh-vlm-follow` | DSH | JavaScript |

OpenViking 每 30 秒重读一次配置，改完最多半分钟生效，不需要重启服务。

## 一键安装

项目提供安装向导（中英双版），自动完成环境探测、配置写入、打补丁、装插件与验证。

### 启动方式

| 系统 | 入口 |
| --- | --- |
| Windows | 双击 `install.cmd`，或运行 `install.ps1` |
| Linux | `python install.py` |

### 向导会做什么

| 步骤 | 内容 |
| --- | --- |
| 1 环境探测 | 检查 OpenViking、OpenVINO Runtime、补丁目标、agent 侧插件，列出可用模型 |
| 2 选择嵌入方式 | 进程内或 HTTP |
| 3 选择模型 | 内置五款任选，或指定自己准备的模型 |
| 4 选择 agent | 选择为哪些已接入 OpenViking 的 agent 安装记忆同步插件 |
| 5 确认清单 | 逐项列出将要执行的动作与目标路径 |
| 6 执行 | 每步改动前备份，任一步失败自动还原 |
| 7 收尾 | 打印重启命令与启动日志应出现的内容 |

OpenVINO Runtime 与 tokenizers 未安装时会自动装上，不需要你另外准备。

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `--dry-run` | 走完全部探测与交互，打印将要执行的动作，不产生任何改动 |
| `--ov-home <路径>` | 指定 OpenViking 安装目录，跳过自动探测 |
| `--lang <auto\|zh\|en>` | 界面语言，默认自动 |
| `--version` | 显示版本 |

电脑上同时装了多个 OpenViking 时，用 `--ov-home` 指明要装到哪一个。

### 首次使用建议

先跑一次 dry-run，确认识别到的路径、模型与 agent 都正确：

```bash
python install.py --dry-run --ov-home /path/to/OpenViking
```

### 前提条件

向导只做检测与提示，不代装这些内容，需要你先自行准备：

| 项目 | 要求 |
| --- | --- |
| OpenViking | 官方原版安装，具体版本见所用发行版的发布说明 |
| agent 侧 OpenViking 插件 | 至少一个 agent 已接入 OpenViking 并运行正常 |
| 嵌入模型 | 内置五款下载到缓存目录，或自备 OpenVINO IR 模型 |

agent 侧插件的判定方式：Hermes 看 `plugins/openviking/` 是否存在，DSH 看 profile 的 `package.json` 依赖里是否含 `@openviking/`。

如果 agent 已安装但都未接入 OpenViking，向导会在环境探测阶段停止并逐个列出缺少什么。

## 快速开始：进程内接口

1. 准备好 OpenVINO IR 模型，放到嵌入模型目录下。
2. 应用后端补丁：

```bash
python patches/patch_openvino.py --check    # 先看状态
python patches/patch_openvino.py            # 打补丁
```

补丁以官方原版为基础适配，装好 OpenViking 后直接应用即可，不需要任何前置改造。`--check` 会告诉你当前是否已经打过本补丁：

| `--check` 报出的状态 | 含义 | 退出码 |
| --- | --- | --- |
| 未打补丁 | 尚未应用 | 2 |
| 已打补丁 | 已是当前版本 | 0 |

脚本可重复执行。目标文件结构与所适配版本不符时，脚本报错停止，不写入文件。

3. 用 `configs/ov.conf.local.example` 复制一份 `ov.conf`，填入模型路径与凭据。
4. 重启 OpenViking 服务。启动日志里应该能看到 `OpenVINO embedder ready: model=... device=GPU`。

## 快速开始：HTTP 接口

1. 启动嵌入服务，用 `--model` 指定模型预设：

```bash
python http-server/embed_server.py --model harrier-ov-int8 --model-dir D:/OpenViking/models/harrier-ov-int8 --port 8010
```

`--model` 可选值与上表五款一致。模型目录里 IR 的摆放位置各款不同，服务按预设自动找：前四款找 `openvino_model.xml`（harrier 在 `int8/` 子目录内），embeddinggemma-2 找 `text_int8.xml`。

2. 应用保活补丁（否则批量写入会频繁重连）：

```bash
python patches/patch_http_keepalive.py
```

3. 用 `configs/ov.conf.http.example` 生成 `ov.conf`。
4. 重启 OpenViking，确认 `/health` 与嵌入接口可达。

## 升级后必须重打补丁

`pip install --upgrade openviking` 会整包替换 `site-packages/openviking/`，两个补丁都会丢失。升级后按顺序重跑：

```bash
python patches/patch_openvino.py
python patches/patch_http_keepalive.py    # 仅 HTTP 形态
```


## 许可

MIT。补丁脚本与 HTTP 服务为本仓库原创；OpenViking 本身遵循其上游许可证。

## 上游

本仓库是对 OpenViking 的第三方扩展，与字节跳动官方无关。如果官方后续内置 OpenVINO 后端，本仓库会标注对应版本并转入维护状态。
