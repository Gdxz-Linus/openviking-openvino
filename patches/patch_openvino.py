# -*- coding: utf-8 -*-
"""可重放补丁：把 OpenVINO 进程内嵌入后端加进 OpenViking 的 local_embedders.py。

背景
  OpenViking 原生只有 llama-cpp（GGUF）后端。当前生产用微软 harrier-ov-int8 的
  int8 OpenVINO IR，走进程内 OpenVINO Runtime 推理，省掉独立嵌入进程的 HTTP 层。

谁会被覆盖
  OpenViking 升级会整包替换 venv\\Lib\\site-packages\\openviking\\，
  本文件打的三处改动全部丢失。升级后重跑本脚本即可。

三处改动
  1. LocalModelSpec 新增 backend / ov_model_dir / ov_query_task 三个字段
  2. LOCAL_DENSE_MODEL_SPECS 新增 harrier-ov-int8 表项
  3. LocalDenseEmbedder 新增 OpenVINO 分支：加载、推理、查询前缀、close

幂等
  每处改动都先检测标记，已存在则跳过，重复执行安全。

用法
  python tools/patch_openvino.py            # 打补丁
  python tools/patch_openvino.py --check    # 只检查状态，不改
"""
import io
import os
import py_compile
import sys

TARGET = os.path.join(
    os.environ.get("OPENVIKING_HOME", r"D:\OpenViking"),
    r"venv\Lib\site-packages\openviking\models\embedder\local_embedders.py",
)

MARK_SPEC = "ov_query_task"
MARK_ENTRY = '"harrier-ov-int8": LocalModelSpec('
MARK_IMPL = "def _load_openvino_model"

# ---------- 改动 1：LocalModelSpec 字段 ----------
OLD_SPEC = '''    pooling_type: Optional[int] = None


LOCAL_DENSE_MODEL_SPECS'''
NEW_SPEC = '''    pooling_type: Optional[int] = None
    # 后端类型：gguf 走 llama-cpp（默认），openvino 走 OpenVINO Runtime。
    backend: str = "gguf"
    # OpenVINO 模型目录（相对 cache_dir），内含 openvino_model.xml/bin 与 tokenizer.json。
    ov_model_dir: Optional[str] = None
    # 查询侧官方指令前缀任务描述（harrier 必须加，文档侧不加）。
    ov_query_task: Optional[str] = None


LOCAL_DENSE_MODEL_SPECS'''

# ---------- 改动 2：模型表项 ----------
OLD_ENTRY = '''        query_instruction=None,
        pooling_type=3,
    ),
}
'''
NEW_ENTRY = '''        query_instruction=None,
        pooling_type=3,
    ),
    # 微软 harrier-oss-v1-0.6b 的 int8 OpenVINO IR 版：1024 维，32K 上下文，
    # decoder-only + last-token 池化。官方要求查询侧加
    # "Instruct: <任务>" 换行 "Query: <文本>" 前缀，文档侧不加。
    # 走 OpenVINO Runtime 进程内推理，不经 HTTP 服务。
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
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
    ),
}
'''

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
ANCHOR_FMT = '''    def _format_text(self, text: str, *, is_query: bool) -> str:
'''
OV_IMPL = '''    # ---------- OpenVINO 后端 ----------

    def _load_openvino_model(self) -> None:
        """加载 int8 OpenVINO IR 模型（进程内推理，不经 HTTP 服务）。"""
        try:
            import openvino as ov
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "OpenVINO backend requires 'openvino' and 'tokenizers'. "
                f"Install them into the OpenViking venv. Original error: {exc}"
            ) from exc

        xml_path = self._resolved_model_path
        model_dir = xml_path.parent.parent if xml_path.parent.name == "int8" else xml_path.parent
        tok_path = model_dir / "tokenizer.json"
        if not xml_path.exists():
            raise EmbeddingConfigurationError(
                f"OpenVINO IR not found: {xml_path}（检查 embedding.dense.cache_dir 与模型目录）"
            )
        if not tok_path.exists():
            raise EmbeddingConfigurationError(f"OpenVINO tokenizer.json not found: {tok_path}")

        # 与 GGUF 分支一致：max_input_tokens 位于 ov.conf 的 embedding 层，
        # 由配置注入到实例属性上；取不到时回落 4096。
        max_tokens = int(getattr(self, "max_input_tokens", None) or 4096)
        tok = Tokenizer.from_file(str(tok_path))
        tok.enable_truncation(max_length=max_tokens)
        tok.enable_padding(pad_id=151643, pad_token="<|endoftext|>")

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
                "OpenVINO embedder ready: model=%s device=%s compile=%.1fs max_tokens=%d",
                self.model_name, device, _t.time() - _t0, max_tokens,
            )
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"OpenVINO compile failed for {xml_path} on {device}: {exc}"
            ) from exc

        self._ov_compiled = compiled
        self._ov_output = compiled.output(0)
        self._ov_tokenizer = tok

    def _ov_embed_one(self, text: str) -> List[float]:
        """单条文本前向推理，返回原始向量。"""
        import numpy as np

        with self._llama_lock:
            enc = self._ov_tokenizer.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            res = self._ov_compiled({"input_ids": ids, "attention_mask": mask})
            vec = np.asarray(res[self._ov_output]).ravel()
        return [float(x) for x in vec]

    def _ov_apply_prefix(self, text: str, *, is_query: bool) -> str:
        """harrier 官方要求：查询侧加 Instruct/Query 前缀，文档侧不加。"""
        task = getattr(self.model_spec, "ov_query_task", None)
        if not is_query or not task:
            return text
        return f"Instruct: {task}\\nQuery: {text}"

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


def apply(text, label, old, new, mark):
    """幂等替换：mark 已存在则跳过。"""
    if mark in text:
        print("  [跳过] %s（已存在）" % label)
        return text, False
    n = text.count(old)
    if n != 1:
        print("  [失败] %s 锚点匹配 %d 次" % (label, n))
        raise SystemExit(1)
    print("  [应用] %s" % label)
    return text.replace(old, new, 1), True


def main(argv):
    check_only = "--check" in argv

    if not os.path.exists(TARGET):
        print("  目标文件不存在: %s" % TARGET)
        print("  请先确认 OPENVIKING_HOME 环境变量或安装路径")
        return 1

    raw = io.open(TARGET, "rb").read()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")

    print("=" * 78)
    print("OpenVINO 进程内后端补丁 — %s 模式" % ("检查" if check_only else "应用"))
    print("=" * 78)
    print("  目标: %s" % TARGET)
    print()

    state = {
        "spec 字段": MARK_SPEC in text,
        "模型表项": MARK_ENTRY in text,
        "OpenVINO 实现": MARK_IMPL in text,
    }
    for k, v in state.items():
        print("  %-16s %s" % (k, "已存在" if v else "缺失"))

    if check_only:
        print()
        print("  结论: %s" % ("补丁完整" if all(state.values()) else "需要打补丁"))
        return 0 if all(state.values()) else 2

    print()
    changed = False
    text, c = apply(text, "spec 字段", OLD_SPEC, NEW_SPEC, MARK_SPEC)
    changed |= c
    text, c = apply(text, "模型表项", OLD_ENTRY, NEW_ENTRY, MARK_ENTRY)
    changed |= c
    text, c = apply(text, "__init__ 分流", OLD_INIT, NEW_INIT, "self._backend = getattr")
    changed |= c
    text, c = apply(text, "OpenVINO 实现", ANCHOR_FMT, OV_IMPL + ANCHOR_FMT, MARK_IMPL)
    changed |= c
    text, c = apply(text, "_embed_formatted_text 分流", OLD_EMB, NEW_EMB, 'if self._backend == "openvino":\n            return EmbedResult')
    changed |= c
    text, c = apply(text, "embed 加前缀", OLD_FMT2, NEW_FMT2, "self._ov_apply_prefix(self._format_text")
    changed |= c
    text, c = apply(text, "close 分流", OLD_CLOSE, NEW_CLOSE, 'if self._backend == "openvino":\n            self._ov_compiled = None')
    changed |= c

    if changed:
        io.open(TARGET, "w", encoding="utf-8-sig" if had_bom else "utf-8",
                newline="\r\n").write(text)
        print()
        print("  已写回")
    else:
        print()
        print("  无需改动")

    try:
        py_compile.compile(TARGET, doraise=True)
        print("  语法校验通过")
    except Exception as exc:
        print("  语法校验失败: %s" % exc)
        return 1

    print()
    print("  下一步：重启 OpenViking 服务后生效")
    print("    powershell -Command \"Stop-Process -Id <PID> -Force\"")
    print("    然后重新启动")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
