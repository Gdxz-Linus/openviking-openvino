# dsh-vlm-follow

让 OpenViking 的记忆抽取模型自动跟随 DSH 当前实际使用的模型通道。与 Hermes 版结构对称，钩子换成 DSH 的 `llm/stream`。

## 解决什么

同 Hermes 版：`ov.conf` 的 `vlm` 段与主系统实际通道脱节时，记忆抽取静默失败。

## 安装

```bash
dsh plugin --profile web add <本目录路径>
```

插件以 `link:` 形式进入依赖，目录移动或删除会导致 DSH 启动时报找不到插件。

### 一个已知的安装障碍

被 `pnpm ERR_PNPM_NO_AGENT_SKILLS_DIR` 拦住时，在 `~/.dsh/profiles/web/pnpm-workspace.yaml` 补一行：

```yaml
skills:
  dirs: .pnpm-skills
```

## 工作方式

| 环节 | 说明 |
| --- | --- |
| 钩子 | `llm/stream`（cordis waterfall，必须返回 `next()`） |
| 取值 | 从 `cordis.patch.yml` 解析 provider 的 baseURL 与 apiKeyEnv，从 `~/.dsh/.credentials.yaml` 取 key |
| 写入 | 原子改写 `ov.conf`（`.tmp` → rename） |
| 生效 | OpenViking 每 30 秒重载配置 |
| 阻塞 | 同步执行，微秒级，不阻塞请求 |

## 与 Hermes 版的差别

| 项目 | Hermes 版 | DSH 版 |
| --- | --- | --- |
| 钩子 | `pre_api_request` | `llm/stream` |
| 输入 | 直接给 `base_url` 与 `model` | 只给 `provider` 与 `model`，需要自己映射到 baseURL |
| 语言 | Python | JavaScript |

## 配置路径

默认 `D:/OpenViking/ov.conf`，可用环境变量覆盖。与 Hermes 版写同一份文件，互相感知修改时间。

## 验证方法

把 `ov.conf` 的 `vlm.model` 手工改成错误值，然后发起一次模型请求，353 毫秒内应被纠正回正确值。
