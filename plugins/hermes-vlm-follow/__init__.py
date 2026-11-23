# Hermes 按目录名加载本插件；事件回调中的异常必须全部捕获，避免阻断对话。
# ruff: noqa: N999, BLE001, S110
"""让 OpenViking 的记忆抽取通道自动跟随 Hermes 当前实际使用的模型通道。

背景
----

OpenViking 的记忆抽取（session_commit）调用一个大模型来生成摘要，端点写在
``~/.openviking/ov.conf`` 的 ``vlm`` 段。这个值过去是手工填的，一旦 Hermes
换了通道而 ov.conf 没跟着改，记忆抽取就会持续失败，而且失败只体现在日志里，
对话本身看不出异常。

做法
----

``pre_api_request`` 事件回调每次模型请求前都会带来 agent 实际使用的
``base_url``、``model`` 与 ``provider``。本插件把这三个值写入 ov.conf 的
``vlm`` 段，于是 Hermes 切到哪个通道，记忆抽取就用哪个通道。

同一个 base_url 与 model 重复出现的请求直接命中内存缓存，不读盘也不写盘；
只有变化时才写入磁盘，因此热路径开销接近零。

失败处理
--------

事件回调里的任何异常都被吞掉：记忆抽取配置同步失败不应影响正在进行的对话。
下一次请求会重试。
"""

import json
import os
import threading
from typing import Any

if os.name == "nt":
    _LOCAL_APPDATA = os.environ.get("LOCALAPPDATA") or os.path.expanduser(
        os.path.join("~", "AppData", "Local")
    )
    _HERMES_DEFAULT_HOME = os.path.join(_LOCAL_APPDATA, "hermes")
else:
    _HERMES_DEFAULT_HOME = os.path.expanduser(os.path.join("~", ".hermes"))

_HOME = os.environ.get("HERMES_HOME") or _HERMES_DEFAULT_HOME
CONFIG_PATH = os.path.join(_HOME, "config.yaml")
ENV_PATH = os.path.join(_HOME, ".env")


def _ov_conf_path() -> str:
    """OpenViking 配置文件路径。

    解析顺序与 OpenViking 自身一致：显式环境变量 → Hermes .env 中的同名变量
    → 默认 ``~/.openviking/ov.conf``。D 盘部署只需在 .env 里指明
    ``OPENVIKING_CONFIG_FILE``，不必再在 C 盘维护一份内容过时的副本。
    """
    for candidate in (
        os.environ.get("OPENVIKING_CONFIG_FILE"),
        _read_env_value("OPENVIKING_CONFIG_FILE"),
    ):
        raw = (candidate or "").strip()
        if raw:
            return os.path.expanduser(raw)
    return os.path.expanduser(os.path.join("~", ".openviking", "ov.conf"))


_LOCK = threading.Lock()
_LAST_APPLIED = {"base_url": None, "model": None}


def _read_env_value(var_name: str) -> str:
    """从 Hermes 的 .env 里取一个变量的值。"""
    if not var_name or not os.path.exists(ENV_PATH):
        return ""
    try:
        with open(ENV_PATH, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                if key.strip() == var_name:
                    return val.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _resolve_api_key(provider: str) -> str:
    """按 provider 名找出它用的密钥。

    优先读 config.yaml 里该 provider 的 ``key_env`` 指向的 .env 变量；
    没有 key_env 时回退到 ``api_key`` 字段本身。
    """
    if not provider or not os.path.exists(CONFIG_PATH):
        return ""
    try:
        import yaml  # PyYAML 随 Hermes 一起安装
    except Exception:
        return ""

    try:
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        return ""

    providers = cfg.get("providers") or {}
    entry = providers.get(provider)
    if not isinstance(entry, dict):
        return ""

    key_env = entry.get("key_env") or ""
    if key_env:
        value = _read_env_value(key_env)
        if value:
            return value
    inline = entry.get("api_key")
    return inline if isinstance(inline, str) else ""


def _write_vlm(base_url: str, model: str, api_key: str) -> bool:
    """把三个值写进 ov.conf 的 vlm 段，返回是否真的改了。"""
    ov_conf = _ov_conf_path()
    if not os.path.exists(ov_conf):
        return False

    with open(ov_conf, encoding="utf-8") as fh:
        conf = json.load(fh)

    current = conf.get("vlm") or {}
    if (
        current.get("api_base") == base_url
        and current.get("model") == model
        and (not api_key or current.get("api_key") == api_key)
    ):
        return False

    conf["vlm"] = {
        "provider": "openai",
        "model": model,
        "api_base": base_url,
        "api_key": api_key or current.get("api_key") or "",
    }

    # 一次性替换：先写临时文件再改名，避免写一半被 OpenViking 读到残缺 JSON
    tmp_path = ov_conf + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(conf, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, ov_conf)
    return True


def _apply(base_url: str | None, model: str | None, provider: str | None) -> None:
    base_url = base_url.strip() if isinstance(base_url, str) else ""
    model = model.strip() if isinstance(model, str) else ""

    if not base_url or not model:
        return

    # 热路径：同一通道的连续请求直接返回
    if _LAST_APPLIED["base_url"] == base_url and _LAST_APPLIED["model"] == model:
        return

    with _LOCK:
        if _LAST_APPLIED["base_url"] == base_url and _LAST_APPLIED["model"] == model:
            return

        api_key = _resolve_api_key(provider or "")
        changed = _write_vlm(base_url, model, api_key)
        _LAST_APPLIED["base_url"] = base_url
        _LAST_APPLIED["model"] = model

        if changed:
            logger = _safe_logger()
            if logger is not None:
                logger.info(
                    "hermes-vlm-follow: OpenViking 记忆抽取已跟随到 %s / %s",
                    base_url,
                    model,
                )


def _safe_logger():
    try:
        import logging

        return logging.getLogger("plugins.vlm_follow")
    except Exception:
        return None


_PROBE_PATH = os.path.join(_HOME, "logs", "hermes-vlm-follow-probe.log")


def _probe(message: str) -> None:
    """写一行探针记录，用于确认事件回调是否真的被调用。"""
    try:
        import time

        os.makedirs(os.path.dirname(_PROBE_PATH), exist_ok=True)
        with open(_PROBE_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except Exception:
        pass


def register(ctx) -> None:
    """注册 pre_api_request 事件回调。"""

    def on_pre_api_request(
        base_url: Any = None,
        model: Any = None,
        provider: Any = None,
        **_: Any,
    ) -> None:
        _probe(
            f"hook fired base_url={base_url!r} model={model!r} provider={provider!r}"
        )
        try:
            _apply(
                base_url if isinstance(base_url, str) else None,
                model if isinstance(model, str) else None,
                provider if isinstance(provider, str) else None,
            )
        except Exception:
            # 配置同步失败绝不阻断对话；下一个请求会重试
            pass

    try:
        ctx.register_hook("pre_api_request", on_pre_api_request)
        _probe("hook registered")
    except Exception as exc:
        _probe(f"hook registration failed: {type(exc).__name__}: {exc}")
