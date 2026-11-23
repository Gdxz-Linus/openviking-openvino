# 变更记录

每个发行版一节，标题格式为 `## vX.Y.Z`，与该版本的 git tag 一致。发布流程按 tag 抽取对应小节作为发布说明，抽取不到时退回自动生成的提交列表并给出警告。

每节须在开头写明适配的 OpenViking 版本与上游文件指纹，这是用户判断能否直接使用的唯一依据。

## v0.1.0

适配 OpenViking 0.5.3 官方原版（`local_embedders.py` 8,774 字节，SHA-256 `adf69925…`）。

### 新增

- 进程内 OpenVINO 嵌入后端，把 int8 模型跑在 Intel CPU、GPU 或 NPU 上。同样一份 4096 token 文本，嵌入时间从 CPU 的 30.1 秒降到 4.0 秒。
- 内置五款 int8 嵌入模型：harrier-ov-int8、Qwen3-Embedding-0.6B、bge-m3、SEA-LION-ModernBERT-600M、embeddinggemma-2。各款的维度、池化方式、归一化要求与查询前缀由补丁按模型名自动处理。
- 自定义模型支持。内置五款之外的 OpenVINO IR 模型通过 `custom` 入口挂载，四项参数写在模型目录下的 `openvino_custom.json` 里。
- HTTP 嵌入服务，把 OpenVINO 包成 OpenAI 兼容接口，供跨进程复用或多实例共享一块 GPU。
- 记忆抽取模型同步插件，Hermes 与 DSH 各一个，让记忆抽取所用的大模型自动跟随宿主当前通道。
- 一键安装向导，三个入口：`install.cmd`、`install.ps1`、`install.py`。中英双语自动检测，按运行环境选择语言，也可用 `--lang` 强制指定。
- 安装向导支持 `--dry-run`，走完全部探测与交互但不产生任何改动。
- 打 tag 自动打包发布，产物为 zip 与 SHA-256 清单。

### 说明

- 补丁以官方原版为适配基础。OpenViking 发布新版本后，本项目跟进适配，具体适配版本见各版发布说明。
- `pip install --upgrade openviking` 会整包替换安装目录，补丁随之丢失，升级后需重新应用。
- 安装向导不代装 OpenViking、Hermes、DSH 及其插件，也不下载模型，这些需要使用者自行准备。
- 安装向导的逻辑层已用单元测试覆盖；真实下载 OpenVINO、写入生产配置、安装插件这几步未在开发环境中执行，需要在实际环境验证。
