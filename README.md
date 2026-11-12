# OpenViking OpenVINO 嵌入后端

给 [OpenViking](https://github.com/volcengine/OpenViking) 增加 OpenVINO 本地嵌入能力，提供两种部署形态，并附带两个让记忆抽取模型自动跟随的工具。

OpenViking 0.5.3 原生只支持 llama-cpp（GGUF）本地嵌入。本仓库补上 OpenVINO Runtime 后端，让 Intel CPU、核显或 NPU 可以直接跑 int8 量化模型；同一份模型既能进程内推理，也能作为独立 HTTP 服务被调用。

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
