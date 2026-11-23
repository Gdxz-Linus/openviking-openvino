# hermes-vlm-follow

让 OpenViking 的记忆抽取模型自动跟随 Hermes 当前实际使用的模型通道。

## 解决什么

OpenViking 的记忆抽取（`session_commit`）调用一个大模型生成摘要，端点写在 `ov.conf` 的 `vlm` 段。这个值是手工填的。主系统换通道后忘了改，记忆抽取会持续失败，而失败只出现在日志里，对话本身看不出异常。

本插件挂在 Hermes 的 `pre_api_request` 钩子上，每次模型请求前拿到实际使用的 `base_url`、`model`、`provider`，写进 `ov.conf` 的 `vlm` 段。

## 安装

复制到 Hermes 插件目录：

```
<HERMES_HOME>/plugins/hermes-vlm-follow/
├── __init__.py
└── plugin.yaml
```

目录名与 `plugin.yaml` 的 `name` 字段一致，都叫 `hermes-vlm-follow`。

如果 `ov.conf` 不在默认位置（`~/.openviking/ov.conf`），在 Hermes 的 `.env` 里指定：

```
OPENVIKING_CONFIG_FILE=D:\OpenViking\ov.conf
```

然后重启 Hermes 让模块级插件生效。

## 工作方式

| 环节 | 说明 |
| --- | --- |
| 钩子 | `pre_api_request`（Hermes 官方接口） |
| 写入 | 读 `ov.conf` → 改 `vlm` 段 → 写 `.tmp` → 原子替换 |
| 幂等 | 同一 `base_url` 与 `model` 直接命中内存缓存，不读盘不写盘 |
| 生效 | OpenViking 每 30 秒重载配置并比对差异，无需重启 |
| 异常 | 全部吞掉，配置同步失败不影响正在进行的对话 |

## 限制

- 只覆盖 Hermes。DSH 独立运行时需要另一个插件（见 `dsh-vlm-follow`）。
- 两个插件可能同时写同一个 `ov.conf`，谁最后请求谁说了算；两边都带修改时间感知，不会互相永久覆盖。

## 配置解析顺序

```
OPENVIKING_CONFIG_FILE 环境变量
→ Hermes .env 中的同名变量
→ ~/.openviking/ov.conf
```
