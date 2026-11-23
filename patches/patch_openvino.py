# -*- coding: utf-8 -*-
# ruff: noqa: UP009, UP031
"""可重放补丁：把 OpenVINO 进程内嵌入后端加进 OpenViking 的 local_embedders.py。

背景
  本补丁以 OpenViking 0.5.3 官方原版为适配基础，当前按这个版本适配。
  OpenViking 发布新版本后，本项目会跟着更新适配。官方原版原生的嵌入后端是
  llama-cpp（GGUF）；本补丁加入进程内 OpenVINO Runtime 推理，省掉独立嵌入
  进程的 HTTP 层，并支持五款 int8 预设模型与用户自备的 custom 模型。目标
  文件结构对不上时，脚本会
  停下报错，不会改坏文件。

谁会被覆盖
  安装或重装 OpenViking 会整包替换 venv 的 site-packages（Windows 为 Lib，Linux 为 lib/pythonX.Y），
  本文件打的改动会全部丢失。重装后重跑本脚本即可。

主要改动
  1. LocalModelSpec 新增 OpenVINO 后端、前缀、池化与归一化字段
  2. LOCAL_DENSE_MODEL_SPECS 新增五款 int8 预设模型与 custom 占位表项
  3. LocalDenseEmbedder 新增 OpenVINO 分支：加载、池化、归一化、前缀、close

重复执行
  每处改动先检测标记，已存在则跳过，重复执行安全。

用法
  python patches/patch_openvino.py                     # 自动探测并打补丁
  python patches/patch_openvino.py --check             # 只检查状态，不改
  python patches/patch_openvino.py --target <路径>      # 指定目标文件
"""

import glob
import os
import py_compile
import sys

TARGET_REL = os.path.join("openviking", "models", "embedder", "local_embedders.py")


def site_packages_rel_candidates() -> list[str]:
    """返回可能的 site-packages 相对路径候选，兼容 Windows 与 Linux 布局。"""
    if sys.platform == "win32":
        return [os.path.join("Lib", "site-packages")]
    version = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    return [
        os.path.join("lib", version, "site-packages"),
        os.path.join("lib", "python3.*", "site-packages"),
    ]


INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
MODEL_NAMES = (
    "harrier-ov-int8",
    "Qwen3-Embedding-0.6B-int8-ov",
    "bge-m3-int8-ov",
    "SEA-LION-ModernBERT-Embedding-600M-OV",
    "embeddinggemma-2-openvino-int8",
    "custom",
)

MARK_SPEC_FIELDS = "    ov_doc_prefix: Optional[str] = None"
MARK_IMPL = "def _ov_pool_vector"
OV_END = "    def _format_text(self, text: str, *, is_query: bool) -> str:\n"

TABLE_HEAD = "LOCAL_DENSE_MODEL_SPECS: Dict[str, LocalModelSpec] = {\n"
TABLE_TAIL = "\n}\n\n\ndef get_local_model_spec"

STATE_LABELS = {
    "unpatched": "未打补丁",
    "patched": "已打补丁",
}

# ---------- 改动 1：LocalModelSpec 字段 ----------
SPEC_ANCHOR = """    query_instruction: Optional[str] = None


LOCAL_DENSE_MODEL_SPECS"""
OV_SPEC_FIELDS = """    # 后端类型：gguf 走 llama-cpp（默认），openvino 走 OpenVINO Runtime。
    backend: str = "gguf"
    # OpenVINO 模型目录（相对 cache_dir），内含 IR 与 tokenizer.json。
    ov_model_dir: Optional[str] = None
    # 查询侧指令前缀的任务描述（harrier 与 Qwen3 需要，其余为空）。
    ov_query_task: Optional[str] = None
    # 查询侧字面前缀，优先级高于 ov_query_task（embeddinggemma-2 用）。
    ov_query_prefix: Optional[str] = None
    # 文档侧字面前缀（仅 embeddinggemma-2 需要）。
    ov_doc_prefix: Optional[str] = None
    # 输出向量的池化方式：model=模型自带，last/cls/mean=由本类自行池化。
    ov_pooling: str = "model"
    # 输出是否需要自行做 L2 归一化。
    ov_normalize: bool = True
"""
SPEC_REPLACEMENT = (
    "    query_instruction: Optional[str] = None\n"
    "    pooling_type: Optional[int] = None\n"
    + OV_SPEC_FIELDS
    + "\n\nLOCAL_DENSE_MODEL_SPECS"
)

# ---------- 改动 2：模型表项 ----------
OV_TABLE_ENTRIES = f'''    # harrier-ov-int8：1024 维，输出 1024 维，模型内置 last-token 池化。
    "harrier-ov-int8": LocalModelSpec(
        model_name="harrier-ov-int8",
        dimension=1024,
        filename="harrier-ov-int8/int8/openvino_model.xml",
        download_url="",
        query_instruction=None,
        pooling_type=3,
        backend="openvino",
        ov_model_dir="harrier-ov-int8",
        ov_query_task=(
            "{INSTRUCT}"
        ),
        ov_query_prefix=None,
        ov_doc_prefix=None,
        ov_pooling="model",
        ov_normalize=False,
    ),
    # Qwen3-Embedding-0.6B-int8-ov：1024 维，输出 1024 维，自行 last-token 池化。
    "Qwen3-Embedding-0.6B-int8-ov": LocalModelSpec(
        model_name="Qwen3-Embedding-0.6B-int8-ov",
        dimension=1024,
        filename="Qwen3-Embedding-0.6B-int8-ov/openvino_model.xml",
        download_url="",
        query_instruction=None,
        pooling_type=3,
        backend="openvino",
        ov_model_dir="Qwen3-Embedding-0.6B-int8-ov",
        ov_query_task=(
            "{INSTRUCT}"
        ),
        ov_query_prefix=None,
        ov_doc_prefix=None,
        ov_pooling="last",
        ov_normalize=True,
    ),
    # bge-m3-int8-ov：1024 维，输出 1024 维，自行 CLS 池化。
    "bge-m3-int8-ov": LocalModelSpec(
        model_name="bge-m3-int8-ov",
        dimension=1024,
        filename="bge-m3-int8-ov/openvino_model.xml",
        download_url="",
        query_instruction=None,
        pooling_type=2,
        backend="openvino",
        ov_model_dir="bge-m3-int8-ov",
        ov_query_task=None,
        ov_query_prefix=None,
        ov_doc_prefix=None,
        ov_pooling="cls",
        ov_normalize=True,
    ),
    # SEA-LION-ModernBERT-Embedding-600M-OV：1024 维，输出 1024 维，mask mean 池化。
    "SEA-LION-ModernBERT-Embedding-600M-OV": LocalModelSpec(
        model_name="SEA-LION-ModernBERT-Embedding-600M-OV",
        dimension=1024,
        filename="SEA-LION-ModernBERT-Embedding-600M-OV/openvino_model.xml",
        download_url="",
        query_instruction=None,
        pooling_type=1,
        backend="openvino",
        ov_model_dir="SEA-LION-ModernBERT-Embedding-600M-OV",
        ov_query_task=None,
        ov_query_prefix=None,
        ov_doc_prefix=None,
        ov_pooling="mean",
        ov_normalize=True,
    ),
    # embeddinggemma-2-openvino-int8：768 维，输出 768 维，模型内置池化。
    "embeddinggemma-2-openvino-int8": LocalModelSpec(
        model_name="embeddinggemma-2-openvino-int8",
        dimension=768,
        filename="embeddinggemma-2-openvino-int8/text_int8.xml",
        download_url="",
        query_instruction=None,
        pooling_type=3,
        backend="openvino",
        ov_model_dir="embeddinggemma-2-openvino-int8",
        ov_query_task=None,
        ov_query_prefix="task: search result | query: ",
        ov_doc_prefix="title: none | text: ",
        ov_pooling="model",
        ov_normalize=False,
    ),
    # custom：用户自备的 OpenVINO IR 模型。维度、池化等四项参数不在本表决定，
    # 而是运行时读取模型目录下的 openvino_custom.json，详情见文档。
    "custom": LocalModelSpec(
        model_name="custom",
        dimension=0,
        filename="",
        download_url="",
        query_instruction=None,
        pooling_type=None,
        backend="openvino",
        ov_model_dir=None,
        ov_pooling="last",
        ov_normalize=True,
    ),
'''

# 供 tests/verify_models.py 读取五款预设参数。
NEW_TABLE_TAIL = "        pooling_type=3,\n    ),\n" + OV_TABLE_ENTRIES + "}\n"

# ---------- 改动 2b：维度校验放行 custom ----------
OLD_DIMENSION_GUARD = """        self._dimension = dimension or self.model_spec.dimension
        if self._dimension != self.model_spec.dimension:
            raise ValueError(
"""
NEW_DIMENSION_GUARD = """        self._dimension = dimension or self.model_spec.dimension
        # custom 是用户自备模型的占位表项，维度以 ov.conf 为准，不做固定值比对。
        if model_name != "custom" and self._dimension != self.model_spec.dimension:
            raise ValueError(
"""

# ---------- 改动 3a：__init__ 分流 ----------
OLD_INIT = """        self._resolved_model_path = self._resolve_model_path()
        self._llama = self._load_model()
        self._llama_lock = Lock()
"""
NEW_INIT = """        self._resolved_model_path = self._resolve_model_path()
        # 按 spec.backend 分流：gguf 走 llama-cpp，openvino 走 OpenVINO Runtime。
        self._backend = getattr(self.model_spec, "backend", "gguf")
        self._ov_compiled = None
        self._ov_output = None
        self._ov_tokenizer = None
        if self._backend == "openvino":
            self._load_openvino_model()
            self._llama = None
        else:
            self._llama = self._load_model()
        self._llama_lock = Lock()
"""

# ---------- 改动 3b：OpenVINO 实现 ----------
OV_IMPL = '''    # ---------- OpenVINO 后端 ----------

    def _ov_pad_config(self, model_dir: Path):
        """读取当前模型的 pad 配置；读取失败时返回 (None, None)。"""
        import json

        config_path = model_dir / "tokenizer_config.json"
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, ValueError):
            return None, None
        if not isinstance(config, dict):
            return None, None
        return config.get("pad_token_id"), config.get("pad_token")

    def _ov_build_tokenizer(self, tok_path: Path, model_dir: Path, max_tokens: int):
        """按当前模型表项构建分词器，避免沿用上个模型的 pad 常量。"""
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        tok.enable_truncation(max_length=max_tokens)
        config_pad_id, config_pad_token = self._ov_pad_config(model_dir)
        pad_id = None
        if config_pad_id is not None:
            try:
                pad_id = int(config_pad_id)
            except (TypeError, ValueError):
                pad_id = None
        if pad_id is None and isinstance(config_pad_token, str):
            pad_id = tok.get_vocab().get(config_pad_token)
        if pad_id is None:
            pad_id = 0
        pad_token = tok.id_to_token(pad_id)
        if pad_token is None:
            pad_id = 0
            pad_token = tok.id_to_token(0)
        tok.enable_padding(pad_id=pad_id, pad_token=pad_token or "")
        return tok

    _OV_CUSTOM_DEFAULTS = {
        "pooling": "last",
        "normalize": True,
        "query_task": "",
        "query_prefix": "",
        "doc_prefix": "",
    }

    def _ov_load_custom_config(self, model_dir):
        """读取 custom 模型目录下的 openvino_custom.json，与默认值合并。"""
        import json

        merged = dict(self._OV_CUSTOM_DEFAULTS)
        path = model_dir / "openvino_custom.json"
        if not path.exists():
            logger.warning(
                "openvino_custom.json 不存在，使用默认参数: %s", path
            )
            return merged
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("openvino_custom.json 读取失败，使用默认参数: %s", exc)
            return merged
        if not isinstance(data, dict):
            logger.warning("openvino_custom.json 内容不是对象，使用默认参数")
            return merged
        for key in merged:
            if data.get(key) is not None:
                merged[key] = data[key]
        if merged["pooling"] not in ("model", "last", "cls", "mean"):
            logger.warning("pooling 取值 %r 无效，改用 last", merged["pooling"])
            merged["pooling"] = "last"
        merged["normalize"] = bool(merged["normalize"])
        for key in ("query_task", "query_prefix", "doc_prefix"):
            merged[key] = str(merged[key] or "")
        logger.info("custom 模型参数: %s", merged)
        return merged

    def _ov_param(self, name, default=None):
        """取运行时参数：custom 优先读配置文件，其余读模型表项。"""
        custom = getattr(self, "_ov_custom", None)
        if custom and name in custom:
            return custom[name]
        attr = {
            "pooling": "ov_pooling",
            "normalize": "ov_normalize",
            "query_task": "ov_query_task",
            "query_prefix": "ov_query_prefix",
            "doc_prefix": "ov_doc_prefix",
        }[name]
        return getattr(self.model_spec, attr, default)

    def _load_openvino_model(self) -> None:
        """加载 int8 OpenVINO IR 模型（进程内推理，不经 HTTP 服务）。"""
        try:
            import openvino as ov
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "OpenVINO backend requires 'openvino' and 'tokenizers'. "
                f"Install them into the OpenViking venv. Original error: {exc}"
            ) from exc

        xml_path = self._resolved_model_path
        self._ov_custom = None
        if xml_path.parent.name == "int8":
            model_dir = xml_path.parent.parent
        else:
            model_dir = xml_path.parent
        tok_path = model_dir / "tokenizer.json"
        if self.model_name == "custom":
            if not self.model_path:
                raise EmbeddingConfigurationError(
                    "custom 模型必须在 embedding.dense 中设置 model_path，"
                    "指向 openvino_model.xml 的绝对路径"
                )
            self._ov_custom = self._ov_load_custom_config(model_dir)
        if not xml_path.exists():
            raise EmbeddingConfigurationError(
                f"OpenVINO IR not found: {xml_path}"
                "（检查 embedding.dense.cache_dir 与模型目录）"
            )
        if not tok_path.exists():
            raise EmbeddingConfigurationError(
                f"OpenVINO tokenizer.json not found: {tok_path}"
            )

        # 与 GGUF 分支一致：max_input_tokens 位于 ov.conf 的 embedding 层，
        # 由配置加到实例属性上；取不到时改用 4096。
        max_tokens = int(getattr(self, "max_input_tokens", None) or 4096)
        try:
            tok = self._ov_build_tokenizer(tok_path, model_dir, max_tokens)
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "OpenVINO backend requires 'openvino' and 'tokenizers'. "
                f"Install them into the OpenViking venv. Original error: {exc}"
            ) from exc

        core = ov.Core()
        device = os.environ.get("OPENVIKING_OV_DEVICE", "GPU")
        if device not in core.available_devices:
            logger.warning(
                "OpenVINO device %s unavailable (have %s), falling back to CPU",
                device, core.available_devices,
            )
            device = "CPU"
        model = core.read_model(str(xml_path))
        cache_dir = os.environ.get("OPENVIKING_OV_CACHE") or str(
            Path(self.cache_dir) / ".." / "ov_cache"
        )
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            import time as _t
            _t0 = _t.time()
            compiled = core.compile_model(model, device, {"CACHE_DIR": cache_dir})
            logger.info(
                "OpenVINO embedder ready: model=%s device=%s compile=%.1fs "
                "max_tokens=%d pooling=%s normalize=%s",
                self.model_name,
                device,
                _t.time() - _t0,
                max_tokens,
                self._ov_param("pooling", "model"),
                self._ov_param("normalize", True),
            )
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"OpenVINO compile failed for {xml_path} on {device}: {exc}"
            ) from exc

        self._ov_compiled = compiled
        self._ov_output = compiled.output(0)
        self._ov_tokenizer = tok

    def _ov_pool_vector(self, raw, mask):
        """把二维自带池化或三维逐 token 输出转成单条向量。"""
        import numpy as np

        raw = np.asarray(raw)
        if raw.ndim == 2:
            return raw[0]
        if raw.ndim != 3:
            raise EmbeddingConfigurationError(
                "Unexpected OpenVINO output rank %d, expect 2 or 3" % raw.ndim
            )
        hidden = raw[0]
        n = int(np.asarray(mask)[0].sum())
        if n <= 0:
            n = 1
        pooling = self._ov_param("pooling", "model")
        if pooling == "cls":
            return hidden[0]
        if pooling == "mean":
            return hidden[:n].mean(axis=0)
        return hidden[n - 1]

    def _ov_embed_one(self, text: str) -> List[float]:
        """单条文本前向推理，返回池化并按需归一化后的向量。"""
        import numpy as np

        with self._llama_lock:
            enc = self._ov_tokenizer.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            res = self._ov_compiled({"input_ids": ids, "attention_mask": mask})
            raw = res[self._ov_output]
        vec = np.asarray(self._ov_pool_vector(raw, mask))
        if self._ov_param("normalize", True):
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
        return [float(x) for x in vec]

    def _ov_apply_prefix(self, text: str, *, is_query: bool) -> str:
        """按模型要求给查询或文档加前缀；查询侧字面前缀优先于指令。"""
        if is_query:
            prefix = self._ov_param("query_prefix")
            if prefix:
                return f"{prefix}{text}"
            task = self._ov_param("query_task")
            if task:
                return f"Instruct: {task}\\nQuery: {text}"
            return text
        prefix = self._ov_param("doc_prefix")
        if prefix:
            return f"{prefix}{text}"
        return text

'''

# ---------- 改动 3c：_embed_formatted_text 分流 ----------
OLD_EMB = """    def _embed_formatted_text(self, formatted: str) -> EmbedResult:
        # llama.cpp mutates the shared context, including during native calls that release the GIL.
        with self._llama_lock:
            payload = self._llama.create_embedding(formatted)
            return EmbedResult(dense_vector=self._extract_embedding(payload))
"""
NEW_EMB = """    def _embed_formatted_text(self, formatted: str) -> EmbedResult:
        # llama.cpp mutates the shared context, including during native calls that release the GIL.
        if self._backend == "openvino":
            return EmbedResult(dense_vector=self._ov_embed_one(formatted))
        with self._llama_lock:
            payload = self._llama.create_embedding(formatted)
            return EmbedResult(dense_vector=self._extract_embedding(payload))
"""

# ---------- 改动 3d：embed 加前缀 ----------
OLD_FMT2 = """    def embed(self, text: str, is_query: bool = False) -> EmbedResult:
        formatted = self._format_text(text, is_query=is_query)
"""
NEW_FMT2 = """    def embed(self, text: str, is_query: bool = False) -> EmbedResult:
        if self._backend == "openvino":
            formatted = self._ov_apply_prefix(self._format_text(text, is_query=is_query),
                                              is_query=is_query)
        else:
            formatted = self._format_text(text, is_query=is_query)
"""

# ---------- 改动 3e：close 分流 ----------
OLD_CLOSE = """    def close(self):
        with self._llama_lock:
            close_fn = getattr(self._llama, "close", None)
            if callable(close_fn):
                close_fn()
"""
NEW_CLOSE = """    def close(self):
        if self._backend == "openvino":
            self._ov_compiled = None
            self._ov_tokenizer = None
            return
        with self._llama_lock:
            close_fn = getattr(self._llama, "close", None)
            if callable(close_fn):
                close_fn()
"""


def _target_candidates(base_dir):
    """按站点包布局候选展开目标文件路径，通配符只保留真实匹配项。"""
    candidates = []
    seen = set()
    for site_packages_rel in site_packages_rel_candidates():
        pattern = os.path.join(base_dir, site_packages_rel, TARGET_REL)
        matches = glob.glob(pattern) if glob.has_magic(pattern) else [pattern]
        for candidate in sorted(matches):
            key = os.path.normcase(os.path.abspath(candidate))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _first_existing_target(base_dir):
    for candidate in _target_candidates(base_dir):
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_target(explicit=None):
    """按显式路径、环境变量、已安装包和解释器邻近路径顺序探测目标。"""
    if explicit:
        return explicit

    home = os.environ.get("OPENVIKING_HOME")
    if home:
        return _first_existing_target(os.path.join(home, "venv"))

    try:
        import openviking
    except ImportError:
        openviking = None
    if openviking is not None:
        pkg_file = getattr(openviking, "__file__", None)
        if pkg_file:
            package_dir = os.path.dirname(os.path.abspath(pkg_file))
            return os.path.join(package_dir, "models", "embedder", "local_embedders.py")

    script_dir = os.path.dirname(os.path.abspath(sys.executable))
    for base_dir in (script_dir, os.path.join(script_dir, "..")):
        candidate = _first_existing_target(base_dir)
        if candidate:
            return candidate
    return None


def detect_state(text):
    """识别目标文件是否已打本补丁。"""
    return "patched" if MARK_IMPL in text else "unpatched"


def apply(text, label, old, new, mark):
    """重复执行安全：已改过的地方跳过。"""
    if mark in text:
        print("  [跳过] %s（已存在）" % label)
        return text, False
    n = text.count(old)
    if n != 1:
        print("  [失败] %s：锚点匹配 %d 处，应为 1 处" % (label, n))
        raise SystemExit(1)
    print("  [应用] %s" % label)
    return text.replace(old, new, 1), True


def patch_spec_fields(text):
    """在官方原版 LocalModelSpec 字段末尾插入 OpenVINO 字段。"""
    if MARK_SPEC_FIELDS in text:
        print("  [跳过] spec 字段（已存在）")
        return text, False
    count = text.count(SPEC_ANCHOR)
    if count > 1:
        print("  [失败] spec 字段：锚点匹配多处，无法确定位置，不是官方 0.5.3 原版")
        raise SystemExit(1)
    if count == 0:
        print("  [失败] spec 字段：锚点匹配 0 处，不是官方 0.5.3 原版")
        raise SystemExit(1)
    print("  [应用] spec 字段")
    return text.replace(SPEC_ANCHOR, SPEC_REPLACEMENT, 1), True


def patch_model_table(text):
    """在官方原版模型表末尾插入五款预设与 custom 模型表项。"""
    head_count = text.count(TABLE_HEAD)
    tail_count = text.count(TABLE_TAIL)
    if head_count != 1 or tail_count != 1:
        print(
            "  [失败] 模型表锚点匹配异常（头 %d 处，尾 %d 处）"
            % (head_count, tail_count)
        )
        raise SystemExit(1)
    present = [name for name in MODEL_NAMES if ('"%s": LocalModelSpec(' % name) in text]
    if present:
        print(
            "  [失败] 模型表已含 OpenVINO 表项，目标不是官方 0.5.3 原版: %s"
            % ", ".join(present)
        )
        raise SystemExit(1)
    tail_start = text.index(TABLE_TAIL)
    print("  [应用] 模型表项")
    updated = (
        text[:tail_start] + ",\n" + OV_TABLE_ENTRIES.rstrip("\n") + text[tail_start:]
    )
    return updated, True


def parse_args(argv):
    """解析 --check 与 --target，避免 argparse 输出英文帮助。"""
    check_only = False
    explicit = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--check":
            check_only = True
        elif arg == "--target":
            if index + 1 >= len(argv):
                print("  --target 需要一个文件路径")
                return None, False, 1
            explicit = argv[index + 1]
            index += 1
        elif arg.startswith("--target="):
            explicit = arg.split("=", 1)[1]
        else:
            print("  未知参数: %s" % arg)
            return None, False, 1
        index += 1
    return explicit, check_only, 0


def main(argv):
    explicit, check_only, parse_status = parse_args(argv)
    if parse_status != 0:
        return parse_status

    target = resolve_target(explicit)
    if not target:
        print("  请设置 OPENVIKING_HOME 或用 --target 指定")
        return 1
    if not os.path.isfile(target):
        print("  目标文件不存在: %s" % target)
        print("  请设置 OPENVIKING_HOME 或用 --target 指定")
        return 1

    with open(target, "rb") as fh:
        raw = fh.read()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    state = detect_state(text)

    print("=" * 78)
    print("OpenVINO 进程内后端补丁 — %s 模式" % ("检查" if check_only else "应用"))
    print("=" * 78)
    print("  目标: %s" % target)
    print("  状态: %s" % STATE_LABELS[state])
    print()

    if check_only:
        return 0 if state == "patched" else 2

    changed = False
    if state == "patched":
        print("  [跳过] 补丁已存在")
    else:
        text, c = patch_spec_fields(text)
        changed |= c
        text, c = patch_model_table(text)
        changed |= c
        text, c = apply(
            text,
            "维度校验跳过 custom",
            OLD_DIMENSION_GUARD,
            NEW_DIMENSION_GUARD,
            'model_name != "custom"',
        )
        changed |= c
        text, c = apply(
            text, "__init__ 分流", OLD_INIT, NEW_INIT, "self._backend = getattr"
        )
        changed |= c
        text, c = apply(text, "OpenVINO 实现", OV_END, OV_IMPL + OV_END, MARK_IMPL)
        changed |= c
        text, c = apply(
            text,
            "_embed_formatted_text 分流",
            OLD_EMB,
            NEW_EMB,
            'if self._backend == "openvino":\n            return EmbedResult',
        )
        changed |= c
        text, c = apply(
            text,
            "embed 加前缀",
            OLD_FMT2,
            NEW_FMT2,
            "self._ov_apply_prefix(self._format_text",
        )
        changed |= c
        text, c = apply(
            text,
            "close 分流",
            OLD_CLOSE,
            NEW_CLOSE,
            'if self._backend == "openvino":\n            self._ov_compiled = None',
        )
        changed |= c

    if changed:
        with open(
            target,
            "w",
            encoding="utf-8-sig" if had_bom else "utf-8",
            newline="\r\n",
        ) as fh:
            fh.write(text)
        print()
        print("  已写回")
    else:
        print()
        print("  无需改动")

    try:
        py_compile.compile(target, doraise=True)
        print("  语法校验通过")
    except (py_compile.PyCompileError, OSError) as exc:
        print("  语法校验失败: %s" % exc)
        return 1

    with open(target, "rb") as fh:
        raw = fh.read()
    written = raw.decode("utf-8-sig").replace("\r\n", "\n")
    entry_counts = [
        (name, written.count('"%s": LocalModelSpec(' % name)) for name in MODEL_NAMES
    ]
    bad_entries = [
        "%s=%d" % (name, count) for name, count in entry_counts if count != 1
    ]
    if bad_entries:
        print("  模型表项数量异常（应为 1 处）: %s" % ", ".join(bad_entries))
        return 1
    if written.count("    ov_normalize: bool = True") != 1:
        print("  ov_normalize 字段声明数量异常")
        return 1
    if written.count(MARK_IMPL) != 1:
        print("  OpenVINO 实现数量异常")
        return 1
    print("  六款模型表项齐全")

    print()
    print("  下一步：重启 OpenViking 服务后生效")
    print('    powershell -Command "Stop-Process -Id <PID> -Force"')
    print("    然后重新启动")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
