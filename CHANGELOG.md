## v0.1.0

适配 OpenViking 0.5.3 官方原版。

### 新增

- OpenVINO 嵌入后端，把 int8 模型跑在 Intel CPU 或 GPU 上。
- 内置五款 int8 嵌入模型列表，各款的维度、池化方式、归一化要求与查询前缀自动处理。
- 自定义模型支持。预留 `custom` 入口挂载，四项参数写在模型目录下的 `openvino_custom.json` 里。
- 支持HTTP 嵌入服务，把 OpenVINO 包成 OpenAI 兼容接口，供跨进程复用或多用户共享。
- 记忆抽取模型同步插件，提供 Hermes 和 DSH，让记忆抽取所用的大模型自动跟随Agent当前通道。
- 一键安装向导，三个入口：`install.cmd`、`install.ps1`、`install.py`。中英双语自动检测。

### 说明

- 补丁以官方原版为适配基础。OpenViking 发布新版本后，本项目跟进适配，具体适配版本见各版发布说明。
- 安装向导不代装 OpenViking、Hermes、DSH 及其插件，也不下载模型，这些需要使用者按需自行准备。

