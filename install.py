"""OpenViking OpenVINO 后端一键安装向导。

本脚本负责：
1. 只读探测 OpenViking、OpenVINO Runtime、补丁目标与 agent 接入状态；
2. 引导选择嵌入形态、模型和 agent 插件；
3. 在用户确认后安装运行时、更新 ov.conf、打补丁并安装记忆抽取同步插件；
4. 加载模型嵌入一条文本，校验实际维度；
5. 任一步失败时用执行前的备份还原已改动文件。

OpenViking 本体、Hermes、DSH 与模型权重均不由本脚本安装或下载。
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from messages import Translator

try:
    import tomllib as _tomllib
except ModuleNotFoundError:
    _tomllib = None

PATCH_MARKER = b"def _ov_pool_vector"
PATCH_STATE_UNPATCHED = "unpatched"
PATCH_STATE_PATCHED = "patched"
PATCH_STATE_MISMATCH = "mismatch"

KEEPALIVE_MARKER = "keepalive_expiry=60.0"
KEEPALIVE_ANCHOR = "keepalive_expiry=5.0,"
KEEPALIVE_PATCHABLE = "patchable"
KEEPALIVE_MISSING = "missing"
KEEPALIVE_DRIFT = "mismatch"

PLUGIN_STATE_MISSING = "missing"
PLUGIN_STATE_DIFFERENT = "different"
PLUGIN_STATE_SAME = "same"
PLUGIN_STATE_NOT_APPLICABLE = "not-applicable"

HTTP_API_BASE = "http://127.0.0.1:8010/v1"
HTTP_API_KEY = "local-openvino-no-auth"
HTTP_ENCODING_FORMAT = "float"
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_VERIFY_TEXT = "OpenVINO 嵌入后端安装验证"
HERMES_PLUGIN_FILES = ("__init__.py", "plugin.yaml")
DSH_PLUGIN_FILES = ("package.json", "index.js", "cordis.patch.yml")
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


class InstallError(RuntimeError):
    """安装流程中可向用户展示的错误。"""


class UsageError(InstallError):
    """命令行参数错误。"""


@dataclass(frozen=True)
class VersionInfo:
    """VERSION.toml 中的项目版本与 OpenViking 适配目标。"""

    project_version: str
    openviking_version: str
    file_bytes: int
    file_sha256: str


def load_version_info(path: Path | None = None) -> dict[str, str | int]:
    """读取 VERSION.toml。缺失或格式错误时抛出 InstallError。"""
    version_path = (
        Path(__file__).resolve().parent / "VERSION.toml" if path is None else Path(path)
    )
    if _tomllib is None:
        raise InstallError(
            "无法读取 VERSION.toml：当前 Python 缺少标准库 tomllib；"
            "请使用 Python 3.11 或更高版本。"
        )
    if not version_path.is_file():
        raise InstallError(f"VERSION.toml 不存在或不是文件：{version_path}")
    try:
        with version_path.open("rb") as handle:
            document = _tomllib.load(handle)
    except (UnicodeDecodeError, _tomllib.TOMLDecodeError) as exc:
        raise InstallError(f"VERSION.toml 格式错误：{version_path}；{exc}") from exc
    except OSError as exc:
        raise InstallError(f"无法读取 VERSION.toml：{version_path}；{exc}") from exc

    if not isinstance(document, dict):
        raise InstallError(f"VERSION.toml 根节点必须是表：{version_path}")
    project = document.get("project")
    if not isinstance(project, dict):
        raise InstallError(f"VERSION.toml 缺少 [project] 段：{version_path}")
    adapt = document.get("adapt")
    if not isinstance(adapt, dict):
        raise InstallError(f"VERSION.toml 缺少 [adapt] 段：{version_path}")

    def required_string(section: dict, section_name: str, key: str) -> str:
        field_name = f"{section_name}.{key}"
        if key not in section:
            raise InstallError(
                f"VERSION.toml 缺少必需字段 {field_name}：{version_path}"
            )
        value = section[key]
        if not isinstance(value, str):
            raise InstallError(
                f"VERSION.toml 字段 {field_name} 必须是字符串：{version_path}"
            )
        return value

    def required_positive_int(section: dict, section_name: str, key: str) -> int:
        field_name = f"{section_name}.{key}"
        if key not in section:
            raise InstallError(
                f"VERSION.toml 缺少必需字段 {field_name}：{version_path}"
            )
        value = section[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise InstallError(
                f"VERSION.toml 字段 {field_name} 必须是整数：{version_path}"
            )
        if value <= 0:
            raise InstallError(
                f"VERSION.toml 字段 {field_name} 必须是正整数：{version_path}"
            )
        return value

    return {
        "project_version": required_string(project, "[project]", "version"),
        "openviking_version": required_string(adapt, "[adapt]", "openviking"),
        "file_bytes": required_positive_int(adapt, "[adapt]", "openviking_file_bytes"),
        "file_sha256": required_string(adapt, "[adapt]", "openviking_file_sha256"),
    }


_FALLBACK_VERSION_INFO = VersionInfo(
    project_version="unknown",
    openviking_version="0.5.3",
    file_bytes=8774,
    file_sha256="adf69925352febc2f9bc88e55a41ada6bcad01acbe1db88d2b7ea89c8f0bfa17",
)
try:
    _LOADED_VERSION_INFO = load_version_info()
except InstallError as exc:
    warnings.warn(
        f"无法读取 VERSION.toml，使用内置默认版本信息：{exc}",
        RuntimeWarning,
        stacklevel=2,
    )
    VERSION_INFO = _FALLBACK_VERSION_INFO
else:
    VERSION_INFO = VersionInfo(
        project_version=str(_LOADED_VERSION_INFO["project_version"]),
        openviking_version=str(_LOADED_VERSION_INFO["openviking_version"]),
        file_bytes=int(_LOADED_VERSION_INFO["file_bytes"]),
        file_sha256=str(_LOADED_VERSION_INFO["file_sha256"]),
    )

ADAPTED_OPENVIKING_VERSION = VERSION_INFO.openviking_version
OFFICIAL_LOCAL_EMBEDDERS_SIZE = VERSION_INFO.file_bytes
OFFICIAL_LOCAL_EMBEDDERS_SHA256 = VERSION_INFO.file_sha256


_DEFAULT_TRANSLATOR = Translator("zh")


def _message(tr: Translator | None, message_key: str, **kwargs: Any) -> str:
    return (tr or _DEFAULT_TRANSLATOR).t(message_key, **kwargs)


def _windows_ui_language() -> int | None:
    """返回 Windows 默认 UI 语言 LCID；非 Windows 或 API 失败时返回 None。"""
    if os.name != "nt":
        return None
    try:
        value = ctypes.windll.kernel32.GetUserDefaultUILanguage()
    except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_language(explicit: str | None) -> str:
    """按参数、专用变量、POSIX locale、Windows UI、默认值的顺序选择语言。"""
    requested = (explicit or "").strip().lower()
    if requested in ("zh", "en"):
        return requested

    configured = os.environ.get("OPENVIKING_OPENVINO_LANG", "").strip().lower()
    if configured in ("zh", "en"):
        return configured

    for name in ("LC_ALL", "LANG", "LC_MESSAGES"):
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if not normalized:
            continue
        return "zh" if normalized.startswith("zh") else "en"

    if os.name == "nt":
        lcid = _windows_ui_language()
        if lcid is not None and (lcid & 0x3FF) == 0x04:
            return "zh"
    return "en"


@dataclass(frozen=True)
class PresetModel:
    key: str
    directory: str
    ir_relative: str
    dimension: int
    context: int
    download_url: str
    pooling: str
    normalize: bool
    query_task: str = ""
    query_prefix: str = ""
    doc_prefix: str = ""


PRESET_MODELS: tuple[PresetModel, ...] = (
    PresetModel(
        key="harrier-ov-int8",
        directory="harrier-ov-int8",
        ir_relative="int8/openvino_model.xml",
        dimension=1024,
        context=32768,
        download_url="https://hf-mirror.com/doldolggi/harrier-oss-v1-0.6b-OpenVINO",
        pooling="model",
        normalize=False,
        query_task=INSTRUCT,
    ),
    PresetModel(
        key="Qwen3-Embedding-0.6B-int8-ov",
        directory="Qwen3-Embedding-0.6B-int8-ov",
        ir_relative="openvino_model.xml",
        dimension=1024,
        context=32768,
        download_url="https://hf-mirror.com/OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
        pooling="last",
        normalize=True,
        query_task=INSTRUCT,
    ),
    PresetModel(
        key="bge-m3-int8-ov",
        directory="bge-m3-int8-ov",
        ir_relative="openvino_model.xml",
        dimension=1024,
        context=8192,
        download_url="https://hf-mirror.com/OpenVINO/bge-m3-int8-ov",
        pooling="cls",
        normalize=True,
    ),
    PresetModel(
        key="SEA-LION-ModernBERT-Embedding-600M-OV",
        directory="SEA-LION-ModernBERT-Embedding-600M-OV",
        ir_relative="openvino_model.xml",
        dimension=1024,
        context=8192,
        download_url=(
            "https://hf-mirror.com/aisingapore/SEA-LION-ModernBERT-Embedding-600M-OV"
        ),
        pooling="mean",
        normalize=True,
    ),
    PresetModel(
        key="embeddinggemma-2-openvino-int8",
        directory="embeddinggemma-2-openvino-int8",
        ir_relative="text_int8.xml",
        dimension=768,
        context=8192,
        download_url="https://hf-mirror.com/wesjos/embeddinggemma-2-openvino-int8",
        pooling="model",
        normalize=False,
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
    ),
)
PRESET_MODEL_BY_KEY = {model.key: model for model in PRESET_MODELS}
CUSTOM_MODEL_NAME = "custom"


@dataclass(frozen=True)
class RuntimeInfo:
    installed: bool
    has_tokenizers: bool
    version: str = ""
    devices: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class PatchTargetInfo:
    path: Path
    exists: bool
    size: int
    sha256: str
    state: str


@dataclass(frozen=True)
class AgentRecord:
    key: str
    name: str
    connected: bool
    missing_plugin: str
    plugin_state: str
    plugin_target: Path | None
    profile: str | None = None


@dataclass(frozen=True)
class Environment:
    ov_home: Path
    version: str
    site_packages: Path
    python_exe: Path
    target_path: Path
    patch_state: str
    runtime: RuntimeInfo
    agents: tuple[AgentRecord, ...]
    cache_dir: Path
    patch_exists: bool = False
    patch_size: int = 0
    patch_sha256: str = ""


@dataclass(frozen=True)
class ModelFileCheck:
    ready: bool
    missing: tuple[Path, ...]
    ir_path: Path
    tokenizer_path: Path


@dataclass(frozen=True)
class CustomModelProbe:
    dimension: int | None
    pooling: str | None
    normalize: bool | None
    query_task: str | None = None
    query_prefix: str | None = None
    doc_prefix: str | None = None


@dataclass(frozen=True)
class CustomModelResolution:
    ir_path: Path
    model_dir: Path
    tokenizer_path: Path
    config_path: Path
    pooling_path: Path


@dataclass(frozen=True)
class CustomModelSettings:
    model_dir: Path
    pooling: str
    normalize: bool
    query_task: str
    query_prefix: str
    doc_prefix: str


@dataclass(frozen=True)
class ModelSelection:
    preset: PresetModel | None
    model_name: str
    dimension: int
    cache_dir: Path
    ir_path: Path
    custom: CustomModelSettings | None = None
    model_path: Path | None = None


@dataclass(frozen=True)
class PlannedAction:
    kind: str
    title: str
    details: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstallPlan:
    actions: tuple[PlannedAction, ...]
    mode: str
    selection: ModelSelection
    agents: tuple[AgentRecord, ...]


@dataclass(frozen=True)
class CliOptions:
    dry_run: bool
    ov_home: Path | None
    show_help: bool
    show_version: bool = False
    lang: str = "auto"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def classify_patch_signature(
    version: str,
    size: int,
    digest: str,
    contains_marker: bool,
) -> str:
    """按版本、字节数、哈希和标记判定补丁目标状态。"""
    if version != ADAPTED_OPENVIKING_VERSION:
        return PATCH_STATE_MISMATCH
    if size == OFFICIAL_LOCAL_EMBEDDERS_SIZE and digest.lower() == (
        OFFICIAL_LOCAL_EMBEDDERS_SHA256
    ):
        return PATCH_STATE_UNPATCHED
    if contains_marker:
        return PATCH_STATE_PATCHED
    return PATCH_STATE_MISMATCH


def patch_target_state(version: str, content: bytes) -> str:
    return classify_patch_signature(
        version,
        len(content),
        _sha256_bytes(content),
        PATCH_MARKER in content,
    )


def inspect_patch_target(path: Path, version: str) -> PatchTargetInfo:
    exists = path.is_file()
    content = path.read_bytes() if exists else b""
    return PatchTargetInfo(
        path=path,
        exists=exists,
        size=len(content),
        sha256=_sha256_bytes(content),
        state=patch_target_state(version, content),
    )


def _site_packages_candidates(ov_home: Path) -> list[Path]:
    venv = ov_home / "venv"
    current_python = (
        venv
        / "lib"
        / f"python{sys.version_info[0]}.{sys.version_info[1]}"
        / "site-packages"
    )
    candidates = [
        venv / "Lib" / "site-packages",
        venv / "lib" / "site-packages",
        current_python,
    ]
    lib_dir = venv / "lib"
    if lib_dir.is_dir():
        candidates.extend(sorted(lib_dir.glob("python*/site-packages")))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def find_site_packages(ov_home: Path) -> Path | None:
    candidates = _site_packages_candidates(ov_home)
    for candidate in candidates:
        if (candidate / "openviking").is_dir():
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def venv_python_path(ov_home: Path) -> Path:
    if os.name == "nt":
        return ov_home / "venv" / "Scripts" / "python.exe"
    return ov_home / "venv" / "bin" / "python"


def _home_from_candidate(candidate: Path) -> Path | None:
    candidate = candidate.expanduser()
    if candidate.name.lower() == "venv":
        candidate = candidate.parent
    site_packages = find_site_packages(candidate)
    if site_packages and (site_packages / "openviking").is_dir():
        return candidate
    return None


def discover_openviking_home(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return _home_from_candidate(explicit)

    candidates: list[Path] = []
    for env_name in ("OPENVIKING_HOME", "OV_HOME"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    prefix = Path(sys.prefix)
    if prefix.name.lower() == "venv":
        candidates.append(prefix.parent)
    candidates.extend(
        [
            Path.cwd(),
            Path.cwd().parent,
            Path.home() / "OpenViking",
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(str(candidate.resolve(strict=False)))
        if normalized in seen:
            continue
        seen.add(normalized)
        home = _home_from_candidate(candidate)
        if home is not None:
            return home
    return None


def read_openviking_version(site_packages: Path) -> str | None:
    for metadata in sorted(site_packages.glob("openviking-*.dist-info/METADATA")):
        try:
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version:"):
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        except OSError:
            continue
    version_file = site_packages / "openviking" / "_version.py"
    if version_file.is_file():
        try:
            text = version_file.read_text(encoding="utf-8")
        except OSError:
            return None
        match = re.search(r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)", text)
        if match:
            return match.group(1)
    return None


RUNTIME_PROBE_SCRIPT = r"""
import json
try:
    import openvino as ov
    try:
        import tokenizers as _tokenizers
        has_tokenizers = True
        tokenizers_error = ""
    except Exception as exc:
        has_tokenizers = False
        tokenizers_error = str(exc)
    core = ov.Core()
    result = {
        "installed": True,
        "has_tokenizers": has_tokenizers,
        "version": str(getattr(ov, "__version__", "")),
        "devices": list(core.available_devices),
        "error": tokenizers_error,
    }
except Exception as exc:
    result = {
        "installed": False,
        "has_tokenizers": False,
        "version": "",
        "devices": [],
        "error": str(exc),
    }
print(json.dumps(result, ensure_ascii=False))
"""


def normalize_runtime_version(version: str) -> str:
    match = re.match(r"(\d+\.\d+\.\d+)", version)
    return match.group(1) if match else version


def probe_openvino_runtime(
    python_exe: Path, tr: Translator | None = None
) -> RuntimeInfo:
    if not python_exe.is_file():
        return RuntimeInfo(
            installed=False,
            has_tokenizers=False,
            error=_message(tr, "ERR_PYTHON_NOT_FOUND", path=python_exe),
        )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["OPENVINO_TELEMETRY_DISABLE"] = "1"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [str(python_exe), "-I", "-B", "-c", RUNTIME_PROBE_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeInfo(installed=False, has_tokenizers=False, error=str(exc))

    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return RuntimeInfo(
            installed=False,
            has_tokenizers=False,
            error=detail or _message(tr, "ERR_PROBE_UNPARSABLE"),
        )
    return RuntimeInfo(
        installed=bool(payload.get("installed")),
        has_tokenizers=bool(payload.get("has_tokenizers")),
        version=normalize_runtime_version(str(payload.get("version") or "")),
        devices=tuple(str(item) for item in payload.get("devices") or []),
        error=str(payload.get("error") or ""),
    )


def check_preset_model(
    cache_dir: Path | str, model_key: str, tr: Translator | None = None
) -> ModelFileCheck:
    preset = PRESET_MODEL_BY_KEY.get(model_key)
    if preset is None:
        raise InstallError(
            _message(tr, "ERR_UNKNOWN_PRESET_MODEL", model_key=model_key)
        )
    root = Path(cache_dir).expanduser()
    model_dir = root / preset.directory
    ir_path = model_dir / preset.ir_relative
    tokenizer_path = model_dir / "tokenizer.json"
    missing = tuple(path for path in (ir_path, tokenizer_path) if not path.is_file())
    return ModelFileCheck(
        ready=not missing,
        missing=missing,
        ir_path=ir_path,
        tokenizer_path=tokenizer_path,
    )


def model_ready_label(
    cache_dir: Path, preset: PresetModel, tr: Translator | None = None
) -> str:
    ready = check_preset_model(cache_dir, preset.key, tr=tr).ready
    return _message(tr, "MODEL_READY" if ready else "MODEL_MISSING")


def load_ov_conf(path: Path, tr: Translator | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise InstallError(
            _message(tr, "ERR_READ_CONFIG", path=path, error=exc)
        ) from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise InstallError(
            _message(tr, "ERR_CONFIG_INVALID_JSON", path=path, error=exc)
        ) from exc
    if not isinstance(data, dict):
        raise InstallError(_message(tr, "ERR_CONFIG_ROOT_OBJECT", path=path))
    return data


def _ensure_mapping(
    parent: dict[str, Any],
    key: str,
    path: str,
    tr: Translator | None = None,
) -> dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise InstallError(_message(tr, "ERR_CONFIG_ITEM_OBJECT", path=path))
    return value


def merge_ov_conf(
    existing: Mapping[str, Any] | None,
    *,
    mode: str,
    model: str,
    dimension: int,
    cache_dir: Path | str,
    model_path: Path | str | None = None,
    api_base: str = HTTP_API_BASE,
    api_key: str = HTTP_API_KEY,
    encoding_format: str = HTTP_ENCODING_FORMAT,
    tr: Translator | None = None,
) -> dict[str, Any]:
    """深合并 ov.conf，只改嵌入相关字段，其余内容原样保留。"""
    if mode not in ("local", "http"):
        raise InstallError(_message(tr, "ERR_UNKNOWN_EMBEDDING_MODE", mode=mode))
    if not model:
        raise InstallError(_message(tr, "ERR_MODEL_EMPTY"))
    if int(dimension) <= 0:
        raise InstallError(_message(tr, "ERR_DIMENSION_POSITIVE"))
    merged = copy.deepcopy(dict(existing or {}))
    embedding = _ensure_mapping(merged, "embedding", "embedding", tr)
    dense = _ensure_mapping(embedding, "dense", "embedding.dense", tr)
    dense["provider"] = "local" if mode == "local" else "openai"
    dense["model"] = model
    dense["dimension"] = int(dimension)
    dense["cache_dir"] = str(cache_dir)
    if model_path is not None:
        dense["model_path"] = str(model_path)
    if mode == "http":
        dense["api_base"] = api_base
        dense["api_key"] = api_key
        dense["encoding_format"] = encoding_format
    return merged


def validate_ov_conf_text(text: str, tr: Translator | None = None) -> None:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise InstallError(
            _message(tr, "ERR_GENERATED_CONFIG_INVALID_JSON", error=exc)
        ) from exc
    if not isinstance(data, dict):
        raise InstallError(_message(tr, "ERR_GENERATED_CONFIG_ROOT_OBJECT"))
    embedding = data.get("embedding")
    dense = embedding.get("dense") if isinstance(embedding, dict) else None
    if not isinstance(dense, dict):
        raise InstallError(_message(tr, "ERR_GENERATED_CONFIG_DENSE_OBJECT"))


def serialize_ov_conf(conf: Mapping[str, Any], tr: Translator | None = None) -> str:
    text = json.dumps(dict(conf), ensure_ascii=False, indent=2) + "\n"
    validate_ov_conf_text(text, tr)
    return text


def new_ov_conf(ov_home: Path) -> dict[str, Any]:
    workspace = (ov_home / "data").as_posix()
    return {
        "server": {"host": "127.0.0.1", "port": 1933, "log_level": "INFO"},
        "storage": {
            "workspace": workspace,
            "vectordb": {"name": "context", "backend": "local"},
            "agfs": {"backend": "local"},
        },
        "embedding": {
            "dense": {},
            "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
        },
    }


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _pooling_from_config(path: Path) -> str | None:
    data = _read_json_file(path)
    if not isinstance(data, dict):
        return None
    flags = (
        ("pooling_mode_lasttoken", "last"),
        ("pooling_mode_cls_token", "cls"),
        ("pooling_mode_mean_tokens", "mean"),
        ("pooling_mode_mean_sqrt_len_tokens", "mean"),
        ("pooling_mode_weightedmean_tokens", "mean"),
    )
    for key, pooling in flags:
        if data.get(key) is True:
            return pooling
    return None


def probe_custom_parameters(model_dir: Path | str) -> CustomModelProbe:
    """从 config.json 与 1_Pooling/config.json 探测可推断参数。"""
    root = Path(model_dir).expanduser()
    config = _read_json_file(root / "config.json")
    dimension = None
    if isinstance(config, dict):
        dimension = _positive_int(config.get("hidden_size"))
        if dimension is None and isinstance(config.get("text_config"), dict):
            dimension = _positive_int(config["text_config"].get("hidden_size"))
    pooling = _pooling_from_config(root / "1_Pooling" / "config.json")
    normalize = None if pooling is None else pooling != "model"
    return CustomModelProbe(
        dimension=dimension,
        pooling=pooling,
        normalize=normalize,
        query_task=None,
        query_prefix=None,
        doc_prefix=None,
    )


def _custom_ir_candidates(path: Path) -> list[Path]:
    candidates = [
        path / "openvino_model.xml",
        path / "text_int8.xml",
        path / "int8" / "openvino_model.xml",
    ]
    if path.is_dir():
        candidates.extend(sorted(item for item in path.glob("*.xml") if item.is_file()))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_custom_model(
    path: Path | str, tr: Translator | None = None
) -> CustomModelResolution:
    raw = Path(path).expanduser()
    if not raw.exists():
        raise InstallError(_message(tr, "ERR_CUSTOM_PATH_MISSING", path=raw))
    if raw.is_file():
        if raw.suffix.lower() != ".xml":
            raise InstallError(_message(tr, "ERR_CUSTOM_IR_XML", path=raw))
        ir_path = raw
    else:
        ir_path = next(
            (item for item in _custom_ir_candidates(raw) if item.is_file()), None
        )
        if ir_path is None:
            raise InstallError(_message(tr, "ERR_CUSTOM_IR_NOT_FOUND"))
    ir_path = ir_path.resolve()
    model_dir = (
        ir_path.parent.parent
        if ir_path.parent.name.lower() == "int8"
        else (ir_path.parent)
    )
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise InstallError(
            _message(
                tr,
                "ERR_CUSTOM_TOKENIZER_MISSING",
                path=model_dir / "tokenizer.json",
            )
        )
    return CustomModelResolution(
        ir_path=ir_path,
        model_dir=model_dir,
        tokenizer_path=tokenizer_path,
        config_path=model_dir / "config.json",
        pooling_path=model_dir / "1_Pooling" / "config.json",
    )


def _files_equal(source: Path, target: Path) -> bool:
    try:
        return source.read_bytes() == target.read_bytes()
    except OSError:
        return False


def _plugin_state_for_files(
    source_dir: Path, target_dir: Path, files: Sequence[str]
) -> str:
    if not target_dir.is_dir():
        return PLUGIN_STATE_MISSING
    for name in files:
        if not (target_dir / name).is_file():
            return PLUGIN_STATE_DIFFERENT
        if not _files_equal(source_dir / name, target_dir / name):
            return PLUGIN_STATE_DIFFERENT
    return PLUGIN_STATE_SAME


def _normalize_path_for_compare(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _read_dependencies(profile_dir: Path) -> dict[str, Any]:
    data = _read_json_file(profile_dir / "package.json")
    if not isinstance(data, dict):
        return {}
    deps = data.get("dependencies")
    return deps if isinstance(deps, dict) else {}


def _dsh_plugin_state(profile_dir: Path, source_dir: Path) -> str:
    deps = _read_dependencies(profile_dir)
    source_key = _normalize_path_for_compare(source_dir)
    dependency = deps.get("dsh-vlm-follow")
    if isinstance(dependency, str) and dependency.startswith(("link:", "file:")):
        target = Path(dependency.split(":", 1)[1])
        if not target.is_absolute():
            target = profile_dir / target
        if _normalize_path_for_compare(target) == source_key:
            return PLUGIN_STATE_SAME
    installed_dir = profile_dir / "node_modules" / "dsh-vlm-follow"
    if installed_dir.is_dir():
        return _plugin_state_for_files(source_dir, installed_dir, DSH_PLUGIN_FILES)
    return (
        PLUGIN_STATE_DIFFERENT if isinstance(dependency, str) else PLUGIN_STATE_MISSING
    )


def build_agent_records(
    hermes_home: Path | None,
    dsh_home: Path | None,
    repo_root: Path,
) -> list[AgentRecord]:
    """构造 agent 接入与项目插件状态列表，只读文件系统。"""
    records: list[AgentRecord] = []
    hermes_source = repo_root / "plugins" / "hermes-vlm-follow"
    dsh_source = repo_root / "plugins" / "dsh-vlm-follow"

    if hermes_home is not None and hermes_home.is_dir():
        connected = (hermes_home / "plugins" / "openviking").is_dir()
        target = hermes_home / "plugins" / "hermes-vlm-follow"
        plugin_state = _plugin_state_for_files(
            hermes_source, target, HERMES_PLUGIN_FILES
        )
        records.append(
            AgentRecord(
                key="hermes",
                name="Hermes",
                connected=connected,
                missing_plugin="" if connected else "openviking",
                plugin_state=plugin_state,
                plugin_target=target,
            )
        )

    if dsh_home is not None and dsh_home.is_dir():
        profiles_dir = dsh_home / "profiles"
        profile_dirs = []
        if profiles_dir.is_dir():
            profile_dirs = sorted(
                item
                for item in profiles_dir.iterdir()
                if item.is_dir() and (item / "package.json").is_file()
            )
        if not profile_dirs:
            records.append(
                AgentRecord(
                    key="dsh",
                    name="DSH",
                    connected=False,
                    missing_plugin="@openviking/*",
                    plugin_state=PLUGIN_STATE_NOT_APPLICABLE,
                    plugin_target=None,
                )
            )
        for profile_dir in profile_dirs:
            deps = _read_dependencies(profile_dir)
            connected = any(str(key).startswith("@openviking/") for key in deps)
            plugin_state = (
                _dsh_plugin_state(profile_dir, dsh_source)
                if connected
                else PLUGIN_STATE_NOT_APPLICABLE
            )
            records.append(
                AgentRecord(
                    key=f"dsh:{profile_dir.name}",
                    name=f"DSH / {profile_dir.name}",
                    connected=connected,
                    missing_plugin="" if connected else "@openviking/*",
                    plugin_state=plugin_state,
                    plugin_target=profile_dir,
                    profile=profile_dir.name,
                )
            )
    return records


def connected_agents(records: Sequence[AgentRecord]) -> list[AgentRecord]:
    return [record for record in records if record.connected]


def unconnected_agents(records: Sequence[AgentRecord]) -> list[AgentRecord]:
    return [record for record in records if not record.connected]


def _find_home_from_command(
    command_name: str,
    marker: str,
    *,
    excluded_names: tuple[str, ...] = (),
) -> Path | None:
    """从命令所在目录向上找第一个直接包含 marker 子目录的目录。"""
    try:
        command = shutil.which(command_name)
    except OSError:
        return None
    if not command:
        return None

    excluded = {name.casefold() for name in excluded_names}
    current = Path(os.path.abspath(command)).parent
    # 命令目录本身算第 0 级，之后最多再向上 6 级。
    for _ in range(7):
        if (current / marker).is_dir() and current.name.casefold() not in excluded:
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def default_hermes_home() -> Path | None:
    value = os.environ.get("HERMES_HOME")
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return candidate

    # hermes-agent/plugins 是运行时内部目录，用户插件位于上一级 HOME。
    candidate = _find_home_from_command(
        "hermes", "plugins", excluded_names=("hermes-agent",)
    )
    if candidate is not None:
        return candidate

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        base = (
            Path(local_appdata).expanduser()
            if local_appdata
            else Path.home() / "AppData" / "Local"
        )
        defaults = [base / "hermes"]
    else:
        defaults = [
            Path.home() / ".hermes",
            Path.home() / ".config" / "hermes",
            Path.home() / ".local" / "share" / "hermes",
        ]
    for candidate in defaults:
        if candidate.is_dir():
            return candidate
    return None


def default_dsh_home() -> Path | None:
    value = os.environ.get("DSH_HOME")
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            return candidate

    candidate = _find_home_from_command("dsh", "profiles")
    if candidate is not None:
        return candidate

    default = Path.home() / ".dsh"
    return default if default.is_dir() else None


def platform_support_error(tr: Translator | None = None) -> str | None:
    """返回当前平台的不支持说明；通过平台检查时返回 None。"""
    if sys.platform == "darwin":
        return _message(tr, "ERR_MACOS_UNSUPPORTED")
    if sys.platform in ("win32", "linux"):
        machine = platform.machine().strip() or "unknown"
        if machine.lower() not in {"amd64", "x86_64", "x64"}:
            return _message(tr, "ERR_UNSUPPORTED_ARCH", machine=machine)
    return None


def _resolve_config_path(value: str | Path, ov_home: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ov_home / path


def detect_cache_dir(ov_home: Path) -> Path:
    config_path = ov_home / "ov.conf"
    try:
        config = load_ov_conf(config_path)
    except InstallError:
        return ov_home / "models"
    embedding = config.get("embedding")
    dense = embedding.get("dense") if isinstance(embedding, dict) else None
    if isinstance(dense, dict):
        value = dense.get("cache_dir")
        if isinstance(value, str) and value.strip():
            return _resolve_config_path(value, ov_home)
    return ov_home / "models"


def _patch_state_label(state: str, tr: Translator | None = None) -> str:
    keys = {
        PATCH_STATE_UNPATCHED: "PATCH_STATE_UNPATCHED",
        PATCH_STATE_PATCHED: "PATCH_STATE_PATCHED",
        PATCH_STATE_MISMATCH: "PATCH_STATE_MISMATCH",
    }
    return _message(tr, keys[state])


def _agent_summary(records: Sequence[AgentRecord], tr: Translator | None = None) -> str:
    names = [record.name for record in connected_agents(records)]
    separator = _message(tr, "LIST_SEPARATOR")
    return separator.join(names) if names else _message(tr, "AGENT_NONE")


def _format_model_line(
    cache_dir: Path,
    index: int,
    preset: PresetModel,
    tr: Translator | None = None,
) -> str:
    return _message(
        tr,
        "MODEL_LINE",
        status=model_ready_label(cache_dir, preset, tr),
        index=index,
        model_key=preset.key,
        dimension=preset.dimension,
        context=preset.context,
    )


def print_model_table(
    cache_dir: Path,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    output("")
    output(_message(tr, "MODEL_TABLE_TITLE", cache_dir=cache_dir))
    for index, preset in enumerate(PRESET_MODELS, 1):
        output(_format_model_line(cache_dir, index, preset, tr))


def print_openviking_missing(
    output: Callable[[str], None], tr: Translator | None = None
) -> None:
    output(_message(tr, "ERR_NO_OPENVIKING", version=ADAPTED_OPENVIKING_VERSION))
    output(
        _message(
            tr,
            "ERR_NO_OPENVIKING_HINT",
            version=ADAPTED_OPENVIKING_VERSION,
        )
    )
    output(f"    pip install openviking=={ADAPTED_OPENVIKING_VERSION}")


def _print_patch_diagnostic(
    environment: Environment,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    output("")
    output(_message(tr, "ERR_PATCH_TARGET_TITLE"))
    output(_message(tr, "PATCH_CURRENT_VERSION", version=environment.version))
    output(_message(tr, "PATCH_CURRENT_FILE", path=environment.target_path))
    if environment.patch_exists:
        output(_message(tr, "PATCH_CURRENT_SIZE", size=environment.patch_size))
        output(
            _message(
                tr,
                "PATCH_CURRENT_SHA256",
                sha256=environment.patch_sha256,
            )
        )
    else:
        output(_message(tr, "PATCH_FILE_MISSING"))
    output(
        _message(
            tr,
            "PATCH_ADAPTED_VERSION",
            version=ADAPTED_OPENVIKING_VERSION,
        )
    )
    output(
        _message(
            tr,
            "PATCH_ADAPTED_SIZE",
            size=OFFICIAL_LOCAL_EMBEDDERS_SIZE,
        )
    )
    output(
        _message(
            tr,
            "PATCH_ADAPTED_SHA256",
            sha256=OFFICIAL_LOCAL_EMBEDDERS_SHA256,
        )
    )
    output("")
    if environment.version != ADAPTED_OPENVIKING_VERSION:
        output(
            _message(
                tr,
                "PATCH_VERSION_WRONG",
                version=ADAPTED_OPENVIKING_VERSION,
            )
        )
        output(
            _message(
                tr,
                "PATCH_RESTORE_OFFICIAL",
                version=ADAPTED_OPENVIKING_VERSION,
            )
        )
    else:
        output(_message(tr, "PATCH_RESTORE_COMMAND"))
    output(
        f'          "{environment.python_exe}" -m pip install '
        f"--force-reinstall --no-deps openviking=={ADAPTED_OPENVIKING_VERSION}"
    )
    if environment.version != ADAPTED_OPENVIKING_VERSION:
        output(_message(tr, "PATCH_REQUEST_ADAPTATION"))


def _print_agent_failure(
    records: Sequence[AgentRecord],
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    output("")
    if not records:
        output(_message(tr, "ERR_AGENT_NONE"))
        output(_message(tr, "ERR_AGENT_NONE_HINT"))
        return
    output(_message(tr, "ERR_AGENT_UNCONNECTED"))
    for record in unconnected_agents(records):
        output(
            _message(
                tr,
                "AGENT_MISSING_PLUGIN",
                name=record.name,
                plugin=record.missing_plugin,
            )
        )


def print_environment_table(
    environment: Environment,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    error_flag = _message(tr, "FLAG_ERROR")
    server_flag = (
        "[OK]" if environment.version == ADAPTED_OPENVIKING_VERSION else error_flag
    )
    output("")
    output(
        _message(
            tr,
            "ENV_SERVER_ROW",
            flag=server_flag,
            version=environment.version,
        )
    )
    output(f"        {environment.ov_home}")

    runtime = environment.runtime
    if runtime.installed and runtime.has_tokenizers:
        output("")
        output(f"  [OK]  OpenVINO Runtime         {runtime.version}")
        devices = (
            "  ".join(runtime.devices)
            if runtime.devices
            else _message(tr, "ENV_DEVICES_UNREPORTED")
        )
        output(_message(tr, "ENV_RUNTIME_DEVICES", devices=devices))
    else:
        output("")
        output(_message(tr, "ENV_RUNTIME_PENDING"))
        output(
            _message(
                tr,
                "ENV_INSTALL_TARGET",
                path=environment.ov_home / "venv",
            )
        )

    patch_flag = (
        "[OK]" if environment.patch_state != PATCH_STATE_MISMATCH else error_flag
    )
    output("")
    output(
        _message(
            tr,
            "ENV_PATCH_ROW",
            flag=patch_flag,
            state=_patch_state_label(environment.patch_state, tr),
        )
    )
    output(f"        {environment.target_path}")

    connected = connected_agents(environment.agents)
    agent_flag = "[OK]" if connected else error_flag
    output("")
    output(
        _message(
            tr,
            "ENV_AGENT_ROW",
            flag=agent_flag,
            summary=_agent_summary(environment.agents, tr),
        )
    )
    if connected:
        names = _message(tr, "LIST_SEPARATOR").join(record.name for record in connected)
        output(_message(tr, "ENV_CONNECTED", names=names))
    for record in unconnected_agents(environment.agents):
        output(
            _message(
                tr,
                "ENV_UNCONNECTED",
                name=record.name,
                plugin=record.missing_plugin,
            )
        )


def collect_environment(
    ov_home_arg: Path | None,
    output: Callable[[str], None] = print,
    tr: Translator | None = None,
) -> Environment | None:
    """只读探测环境；硬性条件不满足时打印原因并返回 None。"""
    output(_message(tr, "STEP1_TITLE"))
    ov_home = discover_openviking_home(ov_home_arg)
    if ov_home is None:
        print_openviking_missing(output, tr)
        return None
    site_packages = find_site_packages(ov_home)
    version = read_openviking_version(site_packages) if site_packages else None
    if site_packages is None or version is None:
        print_openviking_missing(output, tr)
        return None

    python_exe = venv_python_path(ov_home)
    target_path = (
        site_packages / "openviking" / "models" / "embedder" / "local_embedders.py"
    )
    patch_info = inspect_patch_target(target_path, version)
    if tr is None:
        runtime = probe_openvino_runtime(python_exe)
    else:
        runtime = probe_openvino_runtime(python_exe, tr)
    repo_root = Path(__file__).resolve().parent
    agents = tuple(
        build_agent_records(
            default_hermes_home(),
            default_dsh_home(),
            repo_root,
        )
    )
    cache_dir = detect_cache_dir(ov_home)
    environment = Environment(
        ov_home=ov_home,
        version=version,
        site_packages=site_packages,
        python_exe=python_exe,
        target_path=target_path,
        patch_state=patch_info.state,
        runtime=runtime,
        agents=agents,
        cache_dir=cache_dir,
        patch_exists=patch_info.exists,
        patch_size=patch_info.size,
        patch_sha256=patch_info.sha256,
    )
    print_environment_table(environment, output, tr)
    print_model_table(cache_dir, output, tr)

    if version != ADAPTED_OPENVIKING_VERSION:
        _print_patch_diagnostic(environment, output, tr)
        return None
    if patch_info.state == PATCH_STATE_MISMATCH:
        _print_patch_diagnostic(environment, output, tr)
        return None
    if not agents or not connected_agents(agents):
        _print_agent_failure(agents, output, tr)
        return None
    return environment


def _read_answer(
    input_func: Callable[[str], str],
    prompt: str,
    tr: Translator | None = None,
) -> str:
    try:
        return input_func(prompt).strip()
    except EOFError as exc:
        raise InstallError(_message(tr, "INPUT_ENDED")) from exc


def _read_raw_answer(
    input_func: Callable[[str], str],
    prompt: str,
    tr: Translator | None = None,
) -> str:
    try:
        return input_func(prompt)
    except EOFError as exc:
        raise InstallError(_message(tr, "INPUT_ENDED")) from exc


def prompt_embedding_mode(
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> str:
    output("")
    output(_message(tr, "STEP2_TITLE"))
    output(_message(tr, "EMBED_MODE_LOCAL_OPTION"))
    output(_message(tr, "EMBED_MODE_HTTP_OPTION"))
    while True:
        answer = _read_answer(input_func, _message(tr, "EMBED_MODE_PROMPT"), tr)
        if answer in ("", "1"):
            return "local"
        if answer == "2":
            return "http"
        output(_message(tr, "EMBED_MODE_INVALID"))


def _format_preset_layout(cache_dir: Path, preset: PresetModel) -> list[str]:
    model_dir = cache_dir / preset.directory
    if "/" in preset.ir_relative:
        parent = preset.ir_relative.split("/", 1)[0]
        leaf = preset.ir_relative.split("/", 1)[1]
        bin_name = Path(leaf).with_suffix(".bin").name
        return [
            f"{model_dir}/",
            f"├── {parent}/",
            f"│   ├── {leaf}",
            f"│   └── {bin_name}",
            "└── tokenizer.json",
        ]
    bin_name = Path(preset.ir_relative).with_suffix(".bin").name
    return [
        f"{model_dir}/",
        f"├── {preset.ir_relative}",
        f"├── {bin_name}",
        "└── tokenizer.json",
    ]


def format_model_missing(
    cache_dir: Path,
    preset: PresetModel,
    check: ModelFileCheck,
    tr: Translator | None = None,
) -> str:
    lines = [
        _message(tr, "MODEL_FILES_INCOMPLETE"),
        _message(tr, "MODEL_FILES_MISSING_HEADER"),
    ]
    lines.extend(f"  {path}" for path in check.missing)
    lines.append(_message(tr, "MODEL_FILES_LAYOUT_HEADER"))
    lines.extend(f"  {line}" for line in _format_preset_layout(cache_dir, preset))
    lines.extend(
        [
            _message(tr, "MODEL_FILES_DOWNLOAD_HEADER"),
            f"  {preset.download_url}",
            _message(tr, "MODEL_FILES_RERUN"),
        ]
    )
    return "\n".join(lines)


def _prompt_positive_int(
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    label: str,
    current: int | None,
    guidance: str,
    tr: Translator | None = None,
) -> int:
    if current is None:
        output(_message(tr, "PROMPT_VALUE_INFERRED_FAILED", label=label))
        output(_message(tr, "PROMPT_VALUE_GUIDANCE", guidance=guidance))
        prompt = _message(tr, "PROMPT_ENTER_LABEL", label=label)
    else:
        prompt = _message(tr, "PROMPT_LABEL_CURRENT", label=label, current=current)
    while True:
        answer = _read_answer(input_func, prompt, tr)
        if not answer and current is not None:
            return current
        value = _positive_int(answer)
        if value is not None:
            return value
        output(
            _message(
                tr,
                "PROMPT_POSITIVE_INT_REQUIRED",
                label=label,
                guidance=guidance,
            )
        )


def _prompt_pooling(
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    current: str | None,
    tr: Translator | None = None,
) -> str:
    allowed = ("model", "last", "cls", "mean")
    if current is None:
        output(_message(tr, "POOLING_INFERRED_FAILED"))
        output(_message(tr, "POOLING_GUIDANCE"))
        prompt = _message(tr, "PROMPT_POOLING")
    else:
        prompt = _message(tr, "POOLING_CURRENT", current=current)
    while True:
        answer = _read_answer(input_func, prompt, tr).lower()
        if not answer and current is not None:
            return current
        if answer in allowed:
            return answer
        output(_message(tr, "POOLING_INVALID"))


def _parse_bool_answer(answer: str, current: bool | None) -> bool | None:
    normalized = answer.strip().lower()
    if not normalized:
        return current
    if normalized in ("y", "yes", "true", "1", "是", "需要"):
        return True
    if normalized in ("n", "no", "false", "0", "否", "不需要"):
        return False
    return None


def _prompt_normalize(
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    current: bool | None,
    tr: Translator | None = None,
) -> bool:
    recommended = current
    if recommended is None:
        output(_message(tr, "NORMALIZE_INFERRED_FAILED"))
        prompt = _message(tr, "PROMPT_NORMALIZE")
    else:
        prompt = _message(
            tr,
            "NORMALIZE_CURRENT",
            current="True" if current else "False",
        )
    while True:
        answer = _read_answer(input_func, prompt, tr)
        value = _parse_bool_answer(answer, recommended)
        if value is not None:
            return value
        output(_message(tr, "NORMALIZE_INVALID"))


def _prompt_custom_selection(
    environment: Environment,
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> ModelSelection:
    output("")
    output(_message(tr, "CUSTOM_MODEL_PATH_HINT"))
    raw_path = _read_answer(input_func, _message(tr, "PROMPT_MODEL_PATH"), tr)
    if not raw_path:
        raise InstallError(_message(tr, "ERR_CUSTOM_MODEL_EMPTY"))
    resolution = resolve_custom_model(raw_path, tr)
    probe = probe_custom_parameters(resolution.model_dir)

    dimension = _prompt_positive_int(
        input_func,
        output,
        _message(tr, "LABEL_DIMENSION"),
        probe.dimension,
        _message(tr, "GUIDANCE_DIMENSION"),
        tr,
    )
    pooling = _prompt_pooling(input_func, output, probe.pooling, tr)
    normalize = _prompt_normalize(input_func, output, probe.normalize, tr)
    output(_message(tr, "QUERY_PREFIX_UNPROBED"))
    output(_message(tr, "QUERY_PREFIX_GUIDANCE"))
    query_prefix = _read_raw_answer(input_func, _message(tr, "PROMPT_QUERY_PREFIX"), tr)
    output(_message(tr, "CUSTOM_SIDECAR_HINT"))

    settings = CustomModelSettings(
        model_dir=resolution.model_dir,
        pooling=pooling,
        normalize=normalize,
        query_task="",
        query_prefix=query_prefix,
        doc_prefix="",
    )
    return ModelSelection(
        preset=None,
        model_name=CUSTOM_MODEL_NAME,
        dimension=dimension,
        cache_dir=environment.cache_dir,
        ir_path=resolution.ir_path,
        custom=settings,
        model_path=resolution.ir_path,
    )


def prompt_model_selection(
    environment: Environment,
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> ModelSelection:
    output("")
    output(_message(tr, "STEP3_TITLE"))
    for index, preset in enumerate(PRESET_MODELS, 1):
        output(_format_model_line(environment.cache_dir, index, preset, tr))
    output(_message(tr, "MODEL_CUSTOM_OPTION"))
    while True:
        answer = _read_answer(input_func, _message(tr, "PROMPT_MODEL_INDEX"), tr)
        if answer in {str(index) for index in range(1, 7)}:
            break
        output(_message(tr, "MODEL_INDEX_INVALID"))
    if answer == "6":
        return _prompt_custom_selection(environment, input_func, output, tr)

    preset = PRESET_MODELS[int(answer) - 1]
    check = check_preset_model(environment.cache_dir, preset.key, tr)
    if not check.ready:
        raise InstallError(
            format_model_missing(environment.cache_dir, preset, check, tr)
        )
    return ModelSelection(
        preset=preset,
        model_name=preset.key,
        dimension=preset.dimension,
        cache_dir=environment.cache_dir,
        ir_path=check.ir_path,
    )


def _parse_agent_indices(answer: str, count: int) -> list[int]:
    if not answer:
        return list(range(1, count + 1))
    parts = [part for part in re.split(r"[\s,，、]+", answer) if part]
    selected: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError(part)
        index = int(part)
        if index < 1 or index > count:
            raise ValueError(part)
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError(answer)
    return selected


def prompt_agent_selection(
    environment: Environment,
    input_func: Callable[[str], str],
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> list[AgentRecord]:
    agents = connected_agents(environment.agents)
    output("")
    output(_message(tr, "STEP4_TITLE"))
    for index, agent in enumerate(agents, 1):
        suffix = (
            _message(tr, "AGENT_INSTALLED_SKIP")
            if agent.plugin_state == PLUGIN_STATE_SAME
            else ""
        )
        output(f"  {index}. {agent.name}{suffix}")
    output(_message(tr, "AGENT_SELECT_DEFAULT"))
    while True:
        answer = _read_answer(input_func, _message(tr, "PROMPT_AGENT_INDICES"), tr)
        try:
            indices = _parse_agent_indices(answer, len(agents))
        except ValueError:
            output(_message(tr, "AGENT_INDEX_INVALID"))
            continue
        return [agents[index - 1] for index in indices]


def keepalive_status(target: Path) -> str:
    if not target.is_file():
        return KEEPALIVE_MISSING
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return KEEPALIVE_DRIFT
    if KEEPALIVE_MARKER in text:
        return PATCH_STATE_PATCHED
    if text.count(KEEPALIVE_ANCHOR) == 1:
        return KEEPALIVE_PATCHABLE
    return KEEPALIVE_DRIFT


def find_dsh_command(dsh_home: Path | None) -> list[str] | None:
    executable = shutil.which("dsh")
    if executable:
        return [executable]
    if dsh_home is None:
        return None
    bin_dir = dsh_home / "runtime" / "node_modules" / ".bin"
    for name in ("dsh.cmd", "dsh.exe", "dsh"):
        candidate = bin_dir / name
        if candidate.is_file():
            return [str(candidate)]
    ps1 = bin_dir / "dsh.ps1"
    if ps1.is_file():
        powershell = shutil.which("powershell.exe") or "powershell.exe"
        return [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps1),
        ]
    return None


def _custom_sidecar_payload(selection: ModelSelection) -> dict[str, Any] | None:
    if selection.custom is None:
        return None
    settings = selection.custom
    return {
        "pooling": settings.pooling,
        "normalize": settings.normalize,
        "query_task": settings.query_task,
        "query_prefix": settings.query_prefix,
        "doc_prefix": settings.doc_prefix,
    }


# pip 安装源的尝试顺序：官方源优先，失败后依次换国内镜像。
PIP_INDEX_CANDIDATES: tuple[tuple[str, str | None], ...] = (
    ("官方源", None),
    ("清华镜像", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("阿里云镜像", "https://mirrors.aliyun.com/pypi/simple"),
)


_PIP_SOURCE_MESSAGE_KEYS = {
    "官方源": "PIP_SOURCE_OFFICIAL",
    "清华镜像": "PIP_SOURCE_TSINGHUA",
    "阿里云镜像": "PIP_SOURCE_ALIYUN",
}


def _pip_source_label(source_name: str, tr: Translator | None = None) -> str:
    message_key = _PIP_SOURCE_MESSAGE_KEYS.get(source_name)
    return _message(tr, message_key) if message_key else source_name


def _runtime_command_for_python(
    python_exe: Path, index_url: str | None = None
) -> list[str]:
    command = [
        str(python_exe),
        "-I",
        "-B",
        "-m",
        "pip",
        "install",
        "openvino",
        "tokenizers",
    ]
    if index_url:
        command.extend(["--index-url", index_url])
    return command


def _runtime_command(
    environment: Environment, index_url: str | None = None
) -> list[str]:
    return _runtime_command_for_python(environment.python_exe, index_url)


def _config_details(
    config_path: Path,
    mode: str,
    selection: ModelSelection,
    tr: Translator | None = None,
) -> tuple[str, ...]:
    provider = "local" if mode == "local" else "openai"
    mode_name = _message(
        tr,
        "EMBED_MODE_LOCAL_NAME" if mode == "local" else "EMBED_MODE_HTTP_NAME",
    )
    values = [
        _message(tr, "PLAN_TARGET", path=config_path),
        _message(tr, "PLAN_EMBEDDING_MODE", mode=mode_name),
        _message(tr, "PLAN_PROVIDER", value=provider),
        _message(tr, "PLAN_MODEL", value=selection.model_name),
        _message(tr, "PLAN_DIMENSION", value=selection.dimension),
        _message(tr, "PLAN_CACHE_DIR", value=selection.cache_dir),
    ]
    if selection.model_path is not None:
        values.append(_message(tr, "PLAN_MODEL_PATH", value=selection.model_path))
    if mode == "http":
        values.extend(
            [
                _message(tr, "PLAN_API_BASE", value=HTTP_API_BASE),
                _message(tr, "PLAN_API_KEY", value=HTTP_API_KEY),
                _message(tr, "PLAN_ENCODING_FORMAT", value=HTTP_ENCODING_FORMAT),
            ]
        )
    return tuple(values)


def _hermes_action(
    agent: AgentRecord,
    repo_root: Path,
    tr: Translator | None = None,
) -> PlannedAction:
    source_dir = repo_root / "plugins" / "hermes-vlm-follow"
    for name in HERMES_PLUGIN_FILES:
        if not (source_dir / name).is_file():
            raise InstallError(
                _message(
                    tr,
                    "ERR_PLUGIN_FILE_MISSING",
                    path=source_dir / name,
                )
            )
    if agent.plugin_target is None:
        raise InstallError(_message(tr, "ERR_HERMES_TARGET_MISSING"))
    files = _message(tr, "LIST_SEPARATOR").join(HERMES_PLUGIN_FILES)
    details = [
        _message(tr, "PLAN_HERMES_TARGET", path=agent.plugin_target),
        _message(tr, "PLAN_SOURCE", path=source_dir),
        _message(tr, "PLAN_COPY_FILES", files=files),
        _message(tr, "PLAN_BACKUP_EXISTING"),
    ]
    return PlannedAction(
        kind="install_hermes_plugin",
        title=_message(tr, "PLAN_HERMES_TITLE", name=agent.name),
        details=tuple(details),
        payload={
            "agent": agent.name,
            "source_dir": source_dir,
            "target_dir": agent.plugin_target,
            "files": HERMES_PLUGIN_FILES,
        },
    )


def _dsh_action(
    agent: AgentRecord,
    repo_root: Path,
    dsh_home: Path | None,
    tr: Translator | None = None,
) -> PlannedAction:
    source_dir = repo_root / "plugins" / "dsh-vlm-follow"
    if not source_dir.is_dir():
        raise InstallError(_message(tr, "ERR_DSH_SOURCE_MISSING", path=source_dir))
    if agent.profile is None or agent.plugin_target is None:
        raise InstallError(_message(tr, "ERR_DSH_PROFILE_MISSING"))
    dsh_home = agent.plugin_target.parent.parent if agent.plugin_target else dsh_home
    dsh_command = find_dsh_command(dsh_home)
    if dsh_command is None:
        raise InstallError(_message(tr, "ERR_DSH_COMMAND_MISSING"))
    command = dsh_command + [
        "plugin",
        "--profile",
        agent.profile,
        "add",
        str(source_dir.resolve()),
    ]
    command_text = subprocess.list2cmdline(command)
    details = [
        _message(tr, "PLAN_DSH_PROFILE", path=agent.plugin_target),
        _message(tr, "PLAN_PLUGIN_SOURCE", path=source_dir.resolve()),
        _message(tr, "PLAN_COMMAND", command=command_text),
        _message(tr, "PLAN_DSH_BACKUP"),
    ]
    return PlannedAction(
        kind="install_dsh_plugin",
        title=_message(tr, "PLAN_DSH_TITLE", name=agent.name),
        details=tuple(details),
        payload={
            "agent": agent.name,
            "command": command,
            "profile_dir": agent.plugin_target,
            "profile": agent.profile,
            "source_dir": source_dir,
            "dsh_home": dsh_home,
        },
    )


def build_install_plan(
    environment: Environment,
    mode: str,
    selection: ModelSelection,
    selected_agents: Sequence[AgentRecord],
    repo_root: Path | None = None,
    tr: Translator | None = None,
) -> InstallPlan:
    repo_root = repo_root or Path(__file__).resolve().parent
    actions: list[PlannedAction] = []

    if not (environment.runtime.installed and environment.runtime.has_tokenizers):
        command = _runtime_command(environment)
        source_names = _message(tr, "LIST_SEPARATOR").join(
            _pip_source_label(name, tr) for name, _ in PIP_INDEX_CANDIDATES
        )
        actions.append(
            PlannedAction(
                kind="install_runtime",
                title=_message(tr, "PLAN_RUNTIME_TITLE"),
                details=(
                    _message(
                        tr,
                        "PLAN_RUNTIME_TARGET",
                        path=environment.ov_home / "venv",
                    ),
                    _message(
                        tr,
                        "PLAN_COMMAND",
                        command=subprocess.list2cmdline(command),
                    ),
                    _message(
                        tr,
                        "PLAN_INSTALL_SOURCES",
                        sources=source_names,
                    ),
                ),
                payload={
                    "command": command,
                    "python_exe": environment.python_exe,
                    "site_packages": environment.site_packages,
                },
            )
        )

    config_path = environment.ov_home / "ov.conf"
    existing = load_ov_conf(config_path, tr)
    base_config = existing if existing else new_ov_conf(environment.ov_home)
    merged = merge_ov_conf(
        base_config,
        mode=mode,
        model=selection.model_name,
        dimension=selection.dimension,
        cache_dir=selection.cache_dir,
        model_path=selection.model_path,
        tr=tr,
    )
    config_text = serialize_ov_conf(merged, tr)
    actions.append(
        PlannedAction(
            kind="write_config",
            title=_message(tr, "PLAN_CONFIG_TITLE"),
            details=_config_details(config_path, mode, selection, tr),
            payload={"path": config_path, "text": config_text},
        )
    )

    sidecar_payload = _custom_sidecar_payload(selection)
    if sidecar_payload is not None and selection.custom is not None:
        sidecar_path = selection.custom.model_dir / "openvino_custom.json"
        sidecar_text = json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n"
        try:
            json.loads(sidecar_text)
        except ValueError as exc:
            raise InstallError(
                _message(tr, "ERR_SIDECAR_INVALID_JSON", error=exc)
            ) from exc
        details = [_message(tr, "PLAN_TARGET", path=sidecar_path)]
        details.extend(
            _message(tr, "PLAN_SIDECAR_VALUE", field=key, value=value)
            for key, value in sidecar_payload.items()
        )
        actions.append(
            PlannedAction(
                kind="write_custom_params",
                title=_message(tr, "PLAN_CUSTOM_PARAMS_TITLE"),
                details=tuple(details),
                payload={"path": sidecar_path, "text": sidecar_text},
            )
        )

    if environment.patch_state == PATCH_STATE_UNPATCHED:
        patch_script = repo_root / "patches" / "patch_openvino.py"
        if not patch_script.is_file():
            raise InstallError(
                _message(tr, "ERR_PATCH_SCRIPT_MISSING", path=patch_script)
            )
        actions.append(
            PlannedAction(
                kind="apply_openvino_patch",
                title=_message(tr, "PLAN_OPENVINO_PATCH_TITLE"),
                details=(
                    _message(tr, "PLAN_TARGET", path=environment.target_path),
                    _message(tr, "PLAN_SCRIPT", path=patch_script),
                    _message(tr, "PLAN_PATCH_BACKUP"),
                ),
                payload={
                    "script": patch_script,
                    "target": environment.target_path,
                    "python_exe": environment.python_exe,
                    "marker": PATCH_MARKER.decode("ascii"),
                },
            )
        )

    if mode == "http":
        keepalive_target = (
            environment.site_packages
            / "openviking"
            / "models"
            / "embedder"
            / "openai_embedders.py"
        )
        status = keepalive_status(keepalive_target)
        if status == KEEPALIVE_PATCHABLE:
            patch_script = repo_root / "patches" / "patch_http_keepalive.py"
            if not patch_script.is_file():
                raise InstallError(
                    _message(
                        tr,
                        "ERR_KEEPALIVE_SCRIPT_MISSING",
                        path=patch_script,
                    )
                )
            actions.append(
                PlannedAction(
                    kind="apply_keepalive_patch",
                    title=_message(tr, "PLAN_KEEPALIVE_PATCH_TITLE"),
                    details=(
                        _message(tr, "PLAN_TARGET", path=keepalive_target),
                        _message(tr, "PLAN_SCRIPT", path=patch_script),
                        _message(tr, "PLAN_PATCH_BACKUP"),
                    ),
                    payload={
                        "script": patch_script,
                        "target": keepalive_target,
                        "python_exe": environment.python_exe,
                        "marker": KEEPALIVE_MARKER,
                    },
                )
            )
        elif status != PATCH_STATE_PATCHED:
            raise InstallError(
                _message(
                    tr,
                    "ERR_KEEPALIVE_MISMATCH",
                    path=keepalive_target,
                )
            )

    for agent in selected_agents:
        if not agent.connected:
            continue
        if agent.plugin_state == PLUGIN_STATE_SAME:
            continue
        if agent.key == "hermes":
            actions.append(_hermes_action(agent, repo_root, tr))
        elif agent.key.startswith("dsh:"):
            actions.append(_dsh_action(agent, repo_root, None, tr))

    pooling = selection.preset.pooling if selection.preset else selection.custom.pooling
    normalize = (
        selection.preset.normalize if selection.preset else selection.custom.normalize
    )
    actions.append(
        PlannedAction(
            kind="verify_model",
            title=_message(tr, "PLAN_VERIFY_TITLE"),
            details=(
                _message(tr, "PLAN_VERIFY_MODEL", name=selection.model_name),
                _message(tr, "PLAN_VERIFY_MODEL_PATH", path=selection.ir_path),
                _message(
                    tr,
                    "PLAN_VERIFY_EXPECTED_DIMENSION",
                    dimension=selection.dimension,
                ),
                _message(
                    tr,
                    "PLAN_VERIFY_TEXT",
                    text=_message(tr, "VERIFY_TEXT"),
                ),
                _message(
                    tr,
                    "PLAN_VERIFY_PARAMS",
                    pooling=pooling,
                    normalize=normalize,
                ),
            ),
            payload={
                "model_name": selection.model_name,
                "cache_dir": selection.cache_dir,
                "dimension": selection.dimension,
                "model_path": selection.model_path,
                "python_exe": environment.python_exe,
            },
        )
    )
    return InstallPlan(
        actions=tuple(actions),
        mode=mode,
        selection=selection,
        agents=tuple(selected_agents),
    )


def print_install_plan(
    plan: InstallPlan,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    output("")
    output(_message(tr, "STEP5_TITLE"))
    for index, action in enumerate(plan.actions, 1):
        output(f"  [{index}] {action.title}")
        for detail in action.details:
            output(f"      {detail}")


def make_backup_path(path: Path, now: datetime | None = None) -> Path:
    timestamp = now or datetime.now()
    while True:
        suffix = timestamp.strftime("%Y%m%d-%H%M%S")
        candidate = path.with_name(f"{path.name}.bak-{suffix}")
        if not candidate.exists():
            return candidate
        timestamp += timedelta(seconds=1)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


@dataclass
class _JournalEntry:
    path: Path
    backup: Path | None
    existed: bool
    is_dir: bool


class ChangeJournal:
    def __init__(self) -> None:
        self.entries: list[_JournalEntry] = []
        self.created_dirs: list[Path] = []
        self.pip_python: Path | None = None
        self.pip_new_dists: list[str] = []

    def ensure_dir(self, path: Path) -> None:
        missing: list[Path] = []
        current = path
        while not current.exists():
            missing.append(current)
            current = current.parent
        for item in reversed(missing):
            item.mkdir()
            self.created_dirs.append(item)

    def backup(self, path: Path) -> None:
        if path.exists():
            backup = make_backup_path(path)
            if path.is_dir():
                shutil.copytree(path, backup)
            else:
                shutil.copy2(path, backup)
            self.entries.append(
                _JournalEntry(
                    path=path, backup=backup, existed=True, is_dir=path.is_dir()
                )
            )
        else:
            self.entries.append(
                _JournalEntry(path=path, backup=None, existed=False, is_dir=False)
            )

    def rollback(
        self,
        output: Callable[[str], None],
        tr: Translator | None = None,
    ) -> None:
        failures: list[str] = []
        for entry in reversed(self.entries):
            try:
                if entry.existed:
                    _remove_path(entry.path)
                    if entry.is_dir:
                        shutil.copytree(entry.backup, entry.path)
                    else:
                        entry.path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(entry.backup, entry.path)
                else:
                    _remove_path(entry.path)
            except OSError as exc:
                failures.append(
                    _message(
                        tr,
                        "RESTORE_ENTRY_FAILED",
                        path=entry.path,
                        error=exc,
                    )
                )
        for directory in reversed(self.created_dirs):
            try:
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass
        if self.pip_python is not None and self.pip_new_dists:
            names = []
            for dist_info in self.pip_new_dists:
                stripped = dist_info.removesuffix(".dist-info")
                names.append(stripped.rsplit("-", 1)[0])
            if names:
                try:
                    completed = _run_subprocess(
                        [
                            str(self.pip_python),
                            "-I",
                            "-B",
                            "-m",
                            "pip",
                            "uninstall",
                            "-y",
                        ]
                        + names,
                        timeout=600,
                    )
                    if completed.returncode != 0:
                        failures.append(
                            _message(
                                tr,
                                "RESTORE_UNINSTALL_FAILED",
                                detail=_tail_output(completed, tr=tr),
                            )
                        )
                except (OSError, subprocess.SubprocessError) as exc:
                    failures.append(
                        _message(
                            tr,
                            "RESTORE_UNINSTALL_FAILED",
                            detail=exc,
                        )
                    )
        if failures:
            output(_message(tr, "RESTORE_PARTIAL_WARNING"))
            for failure in failures:
                output(f"        {failure}")
        else:
            output(_message(tr, "RESTORE_COMPLETED"))


def _dist_info_names(site_packages: Path) -> set[str]:
    if not site_packages.is_dir():
        return set()
    return {path.name for path in site_packages.glob("*.dist-info") if path.is_dir()}


def _tail_output(
    completed: subprocess.CompletedProcess[str],
    limit: int = 2400,
    tr: Translator | None = None,
) -> str:
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    text = combined.strip()
    if text:
        return text[-limit:]
    return _message(tr, "TAIL_NO_OUTPUT", code=completed.returncode)


def _run_subprocess(
    command: Sequence[str],
    *,
    timeout: int,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        creationflags=creationflags,
    )


def _print_tool_output(
    completed: subprocess.CompletedProcess[str], output: Callable[[str], None]
) -> None:
    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if not text:
        return
    for line in text.splitlines()[-60:]:
        output(f"      {line}")


def _execute_install_runtime(
    action: PlannedAction,
    journal: ChangeJournal,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    python_exe = Path(action.payload["python_exe"])
    site_packages = Path(action.payload["site_packages"])
    before = _dist_info_names(site_packages)
    last_completed: subprocess.CompletedProcess[str] | None = None
    for source_name, index_url in PIP_INDEX_CANDIDATES:
        command = _runtime_command_for_python(python_exe, index_url)
        source_label = _pip_source_label(source_name, tr)
        output(_message(tr, "PIP_TRY_SOURCE", source=source_label))
        completed = _run_subprocess(command, timeout=1800)
        if completed.returncode != 0:
            last_completed = completed
            output(
                _message(
                    tr,
                    "PIP_SOURCE_FAILED",
                    source=source_label,
                )
            )
            for line in _tail_output(completed, tr=tr).splitlines():
                output(f"          {line}")
            continue
        after = _dist_info_names(site_packages)
        journal.pip_python = python_exe
        journal.pip_new_dists = sorted(after - before)
        _print_tool_output(completed, output)
        output(
            _message(
                tr,
                "PIP_SOURCE_SUCCEEDED",
                source=source_label,
            )
        )
        if tr is None:
            runtime = probe_openvino_runtime(python_exe)
        else:
            runtime = probe_openvino_runtime(python_exe, tr)
        if not (runtime.installed and runtime.has_tokenizers):
            raise InstallError(
                _message(
                    tr,
                    "ERR_RUNTIME_IMPORT_FAILED",
                    reason=runtime.error or _message(tr, "REASON_UNKNOWN"),
                )
            )
        return
    if last_completed is None:
        raise InstallError(_message(tr, "ERR_PIP_NO_SOURCE"))
    raise InstallError(
        _message(
            tr,
            "ERR_PIP_FAILED",
            detail=_tail_output(last_completed, tr=tr),
        )
    )


def _execute_write_text(
    action: PlannedAction,
    journal: ChangeJournal,
) -> None:
    path = Path(action.payload["path"])
    text = str(action.payload["text"])
    journal.backup(path)
    atomic_write_text(path, text)


def _execute_patch(
    action: PlannedAction,
    journal: ChangeJournal,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    target = Path(action.payload["target"])
    script = Path(action.payload["script"])
    python_exe = Path(action.payload["python_exe"])
    marker = str(action.payload["marker"])
    journal.backup(target)
    completed = _run_subprocess(
        [str(python_exe), "-I", "-B", str(script), "--target", str(target)],
        timeout=600,
    )
    _print_tool_output(completed, output)
    if completed.returncode != 0:
        raise InstallError(
            _message(
                tr,
                "ERR_PATCH_SCRIPT_FAILED",
                detail=_tail_output(completed, tr=tr),
            )
        )
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstallError(
            _message(
                tr,
                "ERR_PATCH_READ_FAILED",
                path=target,
                error=exc,
            )
        ) from exc
    if marker not in text:
        raise InstallError(_message(tr, "ERR_PATCH_MARKER_MISSING", marker=marker))


def _execute_hermes_plugin(
    action: PlannedAction,
    journal: ChangeJournal,
    tr: Translator | None = None,
) -> None:
    source_dir = Path(action.payload["source_dir"])
    target_dir = Path(action.payload["target_dir"])
    journal.ensure_dir(target_dir)
    for name in action.payload["files"]:
        source = source_dir / name
        target = target_dir / name
        if not source.is_file():
            raise InstallError(_message(tr, "ERR_PLUGIN_SOURCE_MISSING", path=source))
        journal.backup(target)
        shutil.copy2(source, target)


def _execute_dsh_plugin(
    action: PlannedAction,
    journal: ChangeJournal,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    profile_dir = Path(action.payload["profile_dir"])
    journal.backup(profile_dir / "package.json")
    journal.backup(profile_dir / "package-lock.json")
    installed_dir = profile_dir / "node_modules" / "dsh-vlm-follow"
    journal.backup(installed_dir)
    completed = _run_subprocess(
        action.payload["command"],
        timeout=900,
        cwd=profile_dir,
    )
    _print_tool_output(completed, output)
    if completed.returncode != 0:
        raise InstallError(
            _message(
                tr,
                "ERR_DSH_INSTALL_FAILED",
                detail=_tail_output(completed, tr=tr),
            )
        )
    deps = _read_dependencies(profile_dir)
    if "dsh-vlm-follow" not in deps and not installed_dir.is_dir():
        raise InstallError(_message(tr, "ERR_DSH_PLUGIN_MISSING"))


VERIFY_SCRIPT = r"""
import json
import sys
from openviking.models.embedder.local_embedders import LocalDenseEmbedder

model_name, cache_dir, dimension_text, model_path = sys.argv[1:5]
dimension = int(dimension_text)
kwargs = {
    "model_name": model_name,
    "cache_dir": cache_dir,
    "dimension": dimension,
    "config": {"provider": "local"},
}
if model_path:
    kwargs["model_path"] = model_path
embedder = LocalDenseEmbedder(**kwargs)
try:
    result = embedder.embed(__VERIFY_TEXT__, is_query=False)
    vector = result.dense_vector
    if vector is None:
        raise RuntimeError(__VERIFY_ERROR_NO_VECTOR__)
    actual = len(vector)
    if actual != dimension:
        raise RuntimeError(
            __VERIFY_ERROR_DIMENSION__ % (dimension, actual)
        )
    print(json.dumps({"dimension": actual}, ensure_ascii=False))
finally:
    close = getattr(embedder, "close", None)
    if callable(close):
        close()
"""


def _build_verify_script(tr: Translator | None = None) -> str:
    replacements = {
        "__VERIFY_TEXT__": _message(tr, "VERIFY_TEXT"),
        "__VERIFY_ERROR_NO_VECTOR__": _message(tr, "VERIFY_ERROR_NO_VECTOR"),
        "__VERIFY_ERROR_DIMENSION__": _message(tr, "VERIFY_ERROR_DIMENSION"),
    }
    script = VERIFY_SCRIPT
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, json.dumps(value, ensure_ascii=False))
    return script


def _execute_verify(
    action: PlannedAction,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    python_exe = Path(action.payload["python_exe"])
    command = [
        str(python_exe),
        "-I",
        "-B",
        "-c",
        _build_verify_script(tr),
        str(action.payload["model_name"]),
        str(action.payload["cache_dir"]),
        str(action.payload["dimension"]),
        str(action.payload["model_path"] or ""),
    ]
    completed = _run_subprocess(command, timeout=900)
    if completed.returncode != 0:
        _print_tool_output(completed, output)
        raise InstallError(
            _message(
                tr,
                "ERR_VERIFY_FAILED",
                detail=_tail_output(completed, tr=tr),
            )
        )
    actual = None
    for line in reversed(completed.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            actual = payload.get("dimension")
            break
    if actual is None:
        _print_tool_output(completed, output)
        raise InstallError(_message(tr, "ERR_VERIFY_NO_DIMENSION"))
    output(_message(tr, "VERIFY_PASSED", actual=actual))


def _action_location(action: PlannedAction) -> str:
    for key in ("target", "path", "target_dir", "profile_dir"):
        value = action.payload.get(key)
        if value:
            return str(value)
    return action.title


def execute_install_plan(
    plan: InstallPlan,
    environment: Environment,
    output: Callable[[str], None] = print,
    tr: Translator | None = None,
) -> None:
    del environment
    journal = ChangeJournal()
    total = len(plan.actions)
    output("")
    output(_message(tr, "STEP6_TITLE"))
    for index, action in enumerate(plan.actions, 1):
        output("")
        output(f"[{index}/{total}] {action.title}")
        try:
            if action.kind == "install_runtime":
                _execute_install_runtime(action, journal, output, tr)
            elif action.kind in ("write_config", "write_custom_params"):
                _execute_write_text(action, journal)
            elif action.kind in ("apply_openvino_patch", "apply_keepalive_patch"):
                _execute_patch(action, journal, output, tr)
            elif action.kind == "install_hermes_plugin":
                _execute_hermes_plugin(action, journal, tr)
            elif action.kind == "install_dsh_plugin":
                _execute_dsh_plugin(action, journal, output, tr)
            elif action.kind == "verify_model":
                _execute_verify(action, output, tr)
            else:
                raise InstallError(_message(tr, "ERR_UNKNOWN_ACTION", kind=action.kind))
        except KeyboardInterrupt:
            output(_message(tr, "EXEC_INTERRUPTED"))
            journal.rollback(output, tr)
            raise
        except Exception as exc:
            output(
                _message(
                    tr,
                    "EXEC_ACTION_FAILED",
                    title=action.title,
                    error=exc,
                )
            )
            output(_message(tr, "EXEC_RESTORING"))
            journal.rollback(output, tr)
            raise InstallError(
                _message(
                    tr,
                    "ERR_STEP_FAILED",
                    index=index,
                    location=_action_location(action),
                    error=exc,
                )
            ) from exc
        output(_message(tr, "EXEC_ACTION_COMPLETED", title=action.title))
    output("")
    output(_message(tr, "EXEC_ALL_COMPLETED"))


def print_completion(
    plan: InstallPlan,
    environment: Environment,
    output: Callable[[str], None],
    tr: Translator | None = None,
) -> None:
    selection = plan.selection
    provider = "local" if plan.mode == "local" else "openai"
    output("")
    output(_message(tr, "STEP7_TITLE"))
    output(
        _message(
            tr,
            "COMPLETION_READY",
            provider=provider,
            model=selection.model_name,
            dimension=selection.dimension,
        )
    )
    if plan.mode == "http":
        output(_message(tr, "COMPLETION_HTTP_ADDRESS", url=HTTP_API_BASE))

    output("")
    output(_message(tr, "COMPLETION_PLUGINS_HEADER"))
    for agent in plan.agents:
        if not agent.connected:
            continue
        if agent.key == "hermes":
            output(_message(tr, "COMPLETION_HERMES_PLUGIN"))
        elif agent.key.startswith("dsh:"):
            output(
                _message(
                    tr,
                    "COMPLETION_DSH_PLUGIN",
                    name=agent.name,
                )
            )

    output("")
    output(_message(tr, "COMPLETION_RESTART_HEADER"))
    output(_message(tr, "COMPLETION_STOP_SERVICE"))
    output(
        '     powershell -NoProfile -Command "Get-CimInstance Win32_Process | '
        "Where-Object { $_.CommandLine -match 'openviking_cli.server_bootstrap' } | "
        'ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"'
    )
    output(_message(tr, "COMPLETION_START_SERVICE"))
    start_script = environment.ov_home / "start-ov.cmd"
    if not start_script.is_file():
        start_script = environment.ov_home / "openviking-server.cmd"
    if start_script.is_file():
        output(f'     "{start_script}"')
    else:
        output(f'     "{environment.python_exe}" -m openviking_cli.server_bootstrap')

    device = "GPU" if "GPU" in environment.runtime.devices else "CPU"
    if selection.preset is not None:
        pooling = selection.preset.pooling
        normalize = selection.preset.normalize
    else:
        pooling = selection.custom.pooling if selection.custom else "unknown"
        normalize = selection.custom.normalize if selection.custom else True
    output("")
    output(_message(tr, "COMPLETION_LOG_HEADER"))
    output(
        "  OpenVINO embedder ready: "
        f"model={selection.model_name} device={device} max_tokens={DEFAULT_MAX_INPUT_TOKENS} "
        f"pooling={pooling} normalize={normalize}"
    )


USAGE = _DEFAULT_TRANSLATOR.t("USAGE")


def _language_argument(argv: Sequence[str] | None) -> str | None:
    args = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(args):
        if arg == "--lang" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def parse_args(
    argv: Sequence[str] | None = None,
    tr: Translator | None = None,
) -> CliOptions:
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = False
    ov_home: Path | None = None
    show_help = False
    show_version = False
    lang = "auto"
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--dry-run":
            dry_run = True
        elif arg in ("-h", "--help"):
            show_help = True
        elif arg in ("-V", "--version"):
            show_version = True
        elif arg == "--ov-home":
            if index + 1 >= len(args):
                raise UsageError(_message(tr, "ERR_OV_HOME_PATH_REQUIRED"))
            ov_home = Path(args[index + 1]).expanduser()
            index += 1
        elif arg.startswith("--ov-home="):
            value = arg.split("=", 1)[1]
            if not value:
                raise UsageError(_message(tr, "ERR_OV_HOME_PATH_REQUIRED"))
            ov_home = Path(value).expanduser()
        elif arg == "--lang":
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise UsageError(_message(tr, "ERR_LANG_VALUE_REQUIRED"))
            lang = args[index + 1]
            index += 1
        elif arg.startswith("--lang="):
            value = arg.split("=", 1)[1]
            if not value:
                raise UsageError(_message(tr, "ERR_LANG_VALUE_REQUIRED"))
            lang = value
        else:
            raise UsageError(_message(tr, "ERR_UNKNOWN_ARGUMENT", arg=arg))
        index += 1
    return CliOptions(
        dry_run=dry_run,
        ov_home=ov_home,
        show_help=show_help,
        show_version=show_version,
        lang=lang,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    input_func: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    tr = Translator(resolve_language(_language_argument(raw_argv)))
    try:
        options = parse_args(raw_argv, tr)
    except UsageError as exc:
        output(_message(tr, "PARAM_ERROR", error=exc))
        output(tr.t("USAGE"))
        return 2
    tr = Translator(resolve_language(options.lang))
    if options.show_version:
        output(
            _message(
                tr,
                "VERSION_LINE_1",
                project_version=VERSION_INFO.project_version,
            )
        )
        output(
            _message(
                tr,
                "VERSION_LINE_2",
                openviking_version=ADAPTED_OPENVIKING_VERSION,
            )
        )
        return 0
    if options.show_help:
        output(tr.t("USAGE").rstrip())
        return 0
    unsupported = platform_support_error(tr)
    if unsupported is not None:
        output(unsupported)
        return 1
    try:
        environment = collect_environment(options.ov_home, output=output, tr=tr)
        if environment is None:
            return 1
        mode = prompt_embedding_mode(input_func, output, tr)
        selection = prompt_model_selection(environment, input_func, output, tr)
        selected_agents = prompt_agent_selection(environment, input_func, output, tr)
        plan = build_install_plan(environment, mode, selection, selected_agents, tr=tr)
        print_install_plan(plan, output, tr)
        if options.dry_run:
            output("")
            output(_message(tr, "DRY_RUN_DONE"))
            return 0
        answer = _read_answer(input_func, _message(tr, "PROMPT_CONFIRM"), tr)
        if answer.lower() in ("n", "no", "否"):
            output(_message(tr, "CANCELLED"))
            return 0
        execute_install_plan(plan, environment, output=output, tr=tr)
        print_completion(plan, environment, output, tr)
        return 0
    except InstallError as exc:
        output(_message(tr, "MAIN_FAILED", error=exc))
        return 1
    except KeyboardInterrupt:
        output("")
        output(_message(tr, "INTERRUPTED"))
        return 130


if __name__ == "__main__":
    sys.exit(main())
