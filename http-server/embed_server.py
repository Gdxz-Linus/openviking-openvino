# -*- coding: utf-8 -*-
# ruff: noqa: UP009, UP031
"""OpenAI 兼容的嵌入服务，后端用 OpenVINO 加载 int8 IR 模型。

为什么需要它：OpenViking 的嵌入 provider 只有 local（llama.cpp）和 openai（HTTP）等，
没有 OpenVINO。把它包成 OpenAI 接口，OpenViking 就能零改源码地用上 OpenVINO。

启动：
  python embed_server.py --model harrier-ov-int8 \
      --model-dir D:/models/harrier-ov-int8 --port 8010
"""

import argparse
import json
import logging
import os
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

DEFAULT_MODEL_NAME = "harrier-ov-int8"
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

# 与 patches/patch_openvino.py 的 LOCAL_DENSE_MODEL_SPECS 保持同口径。
# xml_rel 相对 --model-dir；pooling 只在模型输出为三维张量时生效。
MODEL_PRESETS = {
    "harrier-ov-int8": {
        "dimension": 1024,
        "xml_rel": "int8/openvino_model.xml",
        "pooling": "model",
        "normalize": False,
        "query_task": INSTRUCT,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "Qwen3-Embedding-0.6B-int8-ov": {
        "dimension": 1024,
        "xml_rel": "openvino_model.xml",
        "pooling": "last",
        "normalize": True,
        "query_task": INSTRUCT,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "bge-m3-int8-ov": {
        "dimension": 1024,
        "xml_rel": "openvino_model.xml",
        "pooling": "cls",
        "normalize": True,
        "query_task": None,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "SEA-LION-ModernBERT-Embedding-600M-OV": {
        "dimension": 1024,
        "xml_rel": "openvino_model.xml",
        "pooling": "mean",
        "normalize": True,
        "query_task": None,
        "query_prefix": None,
        "doc_prefix": None,
    },
    "embeddinggemma-2-openvino-int8": {
        "dimension": 768,
        "xml_rel": "text_int8.xml",
        "pooling": "model",
        "normalize": False,
        "query_task": None,
        "query_prefix": "task: search result | query: ",
        "doc_prefix": "title: none | text: ",
    },
}

MODEL_DIR = os.environ.get("OV_EMBED_MODEL_DIR", "")
MODEL_NAME = os.environ.get("OV_EMBED_MODEL_NAME", DEFAULT_MODEL_NAME)
CACHE_DIR = os.environ.get("OV_EMBED_CACHE", "")
MAX_TOKENS = int(os.environ.get("OV_EMBED_MAX_TOKENS", "4096"))
# 保留既有环境变量覆盖能力；默认值与补丁里的 harrier/Qwen3 任务描述一致。
QUERY_TASK_OVERRIDE = os.environ.get("OV_EMBED_QUERY_TASK", "").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-server")

app = FastAPI(title="OpenVINO Embedding Server")

_state = {
    "compiled": None,
    "out": None,
    "tok": None,
    "lock": threading.Lock(),
    "device": "",
    "preset": dict(MODEL_PRESETS[DEFAULT_MODEL_NAME]),
    "model_name": MODEL_NAME,
    "dimension": MODEL_PRESETS[DEFAULT_MODEL_NAME]["dimension"],
}


class EmbeddingRequest(BaseModel):
    input: object
    model: str = MODEL_NAME
    encoding_format: str = "float"
    dimensions: object = None
    # OpenViking 的 openai_embedders 会把 query_param/document_param 交给 openai SDK 的
    # extra_body，而 SDK 的语义是「把 extra_body 的内容展开合并到请求体顶层」，
    # 不会保留 extra_body 这个键（已用本地抓包实测确认：
    # 实际请求体是 {"input":...,"input_type":"query"}）。
    # 所以这里必须按顶层字段接，不能按嵌套字典读。
    input_type: object = None
    extra_body: object = None  # 兼容个别客户端真的写成嵌套时的老形状


def _is_query_request(req: "EmbeddingRequest") -> bool:
    marker = str(req.input_type or "").strip().lower()
    if not marker:
        eb = req.extra_body
        if isinstance(eb, dict):
            marker = str(eb.get("input_type") or "").strip().lower()
    return marker in ("query", "search_query")


def _read_tokenizer_config(model_dir):
    """读取 tokenizer_config.json；缺失或损坏时返回空字典。"""
    path = os.path.join(os.fspath(model_dir), "tokenizer_config.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _resolve_pad(tokenizer, config):
    """解析并验证 pad id，无法使用时回落到 0。"""
    config_pad_id = config.get("pad_token_id")
    pad_id = None
    if config_pad_id is not None:
        try:
            pad_id = int(config_pad_id)
        except (TypeError, ValueError):
            pad_id = None
    if pad_id is None:
        config_pad_token = config.get("pad_token")
        if isinstance(config_pad_token, str):
            pad_id = tokenizer.get_vocab().get(config_pad_token)
    if pad_id is None:
        pad_id = 0

    pad_token = tokenizer.id_to_token(pad_id)
    if pad_token is None:
        pad_id = 0
        pad_token = tokenizer.id_to_token(0)
    return pad_id, pad_token or ""


def configure_tokenizer_padding(tokenizer, model_dir):
    """按当前模型目录的配置启用 padding，避免沿用其他模型的 pad 常量。"""
    pad_id, pad_token = _resolve_pad(tokenizer, _read_tokenizer_config(model_dir))
    tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
    return pad_id, pad_token


def _pool(raw, mask, pooling):
    """把二维自带池化或三维逐 token 输出转成单条向量。"""
    raw = np.asarray(raw)
    if raw.ndim == 2:
        return raw[0]
    if raw.ndim != 3:
        raise ValueError("不支持的模型输出维数: %d（应为 2 或 3）" % raw.ndim)

    hidden = raw[0]
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[0]
    elif mask.ndim != 1:
        raise ValueError("不支持的 attention_mask 维数: %d" % mask.ndim)
    n = int(mask.sum())
    if n <= 0:
        n = 1
    if n > hidden.shape[0]:
        raise ValueError(
            "attention_mask 有效 token 数 %d 超过模型输出长度 %d" % (n, hidden.shape[0])
        )

    if pooling == "cls":
        return hidden[0]
    if pooling == "mean":
        return hidden[:n].mean(axis=0)
    if pooling == "last":
        return hidden[n - 1]
    raise ValueError("不支持的三维池化方式: %s" % pooling)


def _normalize_vector(vector, normalize):
    """按模型预设决定是否做 L2 归一化；零向量保持原样。"""
    vector = np.asarray(vector)
    if not normalize:
        return vector
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        return vector / norm
    return vector


def build_prefix(text, is_query, preset):
    """按模型预设给查询或文档加前缀，查询字面前缀优先于任务指令。"""
    if is_query:
        prefix = preset.get("query_prefix")
        if prefix:
            return f"{prefix}{text}"
        task = preset.get("query_task")
        if task:
            return f"Instruct: {task}\nQuery: {text}"
        return text
    prefix = preset.get("doc_prefix")
    if prefix:
        return f"{prefix}{text}"
    return text


def apply_prefix(text: str, is_query: bool) -> str:
    """使用当前已加载模型的预设加前缀。"""
    return build_prefix(text, is_query, _state["preset"])


def load(device: str, preset=None) -> None:
    import openvino as ov
    from tokenizers import Tokenizer

    preset = dict(preset or _state["preset"])
    xml = os.path.join(MODEL_DIR, preset["xml_rel"])
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(xml):
        raise SystemExit("模型文件不存在: %s" % xml)
    if not os.path.exists(tok_path):
        raise SystemExit("分词器文件不存在: %s" % tok_path)

    tok = Tokenizer.from_file(tok_path)
    tok.enable_truncation(max_length=MAX_TOKENS)
    pad_id, pad_token = configure_tokenizer_padding(tok, MODEL_DIR)
    log.info("分词器就绪 pad_id=%s pad_token=%r", pad_id, pad_token)

    core = ov.Core()
    if device not in core.available_devices:
        log.warning(
            "设备 %s 不可用，可用: %s，改用 CPU", device, core.available_devices
        )
        device = "CPU"
    model = core.read_model(xml)
    t0 = time.time()
    compiled = core.compile_model(model, device, {"CACHE_DIR": CACHE_DIR})
    log.info(
        "模型就绪 model=%s device=%s 编译耗时 %.1f s pooling=%s normalize=%s",
        MODEL_NAME,
        device,
        time.time() - t0,
        preset["pooling"],
        preset["normalize"],
    )

    _state.update(
        compiled=compiled,
        out=compiled.output(0),
        tok=tok,
        device=device,
        preset=preset,
        model_name=MODEL_NAME,
        dimension=int(preset["dimension"]),
    )


def embed_one(text: str):
    tok = _state["tok"]
    compiled = _state["compiled"]
    out = _state["out"]
    preset = _state["preset"]
    with _state["lock"]:
        enc = tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        raw = np.asarray(compiled({"input_ids": ids, "attention_mask": mask})[out])
    vector = _pool(raw, enc.attention_mask, preset["pooling"])
    return _normalize_vector(vector, preset["normalize"])


@app.get("/health")
def health():
    preset = _state["preset"]
    return {
        "status": "ok",
        "healthy": _state["compiled"] is not None,
        "device": _state["device"],
        "model": _state["model_name"],
        "dim": preset["dimension"],
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": _state["model_name"],
                "object": "model",
                "owned_by": "local-openvino",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    raw = req.input
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list):
        texts = [str(x) for x in raw]
    else:
        texts = [str(raw)]

    is_query = _is_query_request(req)
    texts = [apply_prefix(t, is_query) for t in texts]

    t0 = time.time()
    data = []
    total_tokens = 0
    for i, text in enumerate(texts):
        vec = embed_one(text)
        total_tokens += len(_state["tok"].encode(text).ids)
        data.append({"object": "embedding", "index": i, "embedding": vec.tolist()})

    elapsed = time.time() - t0
    log.info(
        "embeddings n=%d chars=%d query=%s %.0f ms",
        len(texts),
        sum(len(t) for t in texts),
        is_query,
        elapsed * 1000,
    )
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


def main():
    global MODEL_DIR, MODEL_NAME, CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default=MODEL_DIR,
        help="模型目录，内含 tokenizer.json 与对应的 OpenVINO IR",
    )
    ap.add_argument(
        "--cache-dir",
        default=CACHE_DIR,
        help="OpenVINO 编译缓存目录，留空则放到模型目录旁边",
    )
    ap.add_argument("--device", default=os.environ.get("OV_EMBED_DEVICE", "GPU"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument(
        "--model",
        default=MODEL_NAME,
        help="模型预设名，默认 harrier-ov-int8；可选: %s" % ", ".join(MODEL_PRESETS),
    )
    ap.add_argument(
        "--model-xml",
        default=None,
        help="覆盖预设的 IR 相对路径（也可传绝对路径）",
    )
    ap.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="覆盖预设的向量维度（仅用于 /health 与检查口径）",
    )
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    CACHE_DIR = args.cache_dir
    if not MODEL_DIR:
        ap.error("必须指定 --model-dir（或设置 OV_EMBED_MODEL_DIR）")
    if not CACHE_DIR:
        CACHE_DIR = os.path.join(
            os.path.dirname(os.path.abspath(MODEL_DIR)), "ov_cache"
        )
    if args.model not in MODEL_PRESETS:
        ap.error("未知模型: %s；可选: %s" % (args.model, ", ".join(MODEL_PRESETS)))
    if args.dimension is not None and args.dimension <= 0:
        ap.error("--dimension 必须是正整数")

    MODEL_NAME = args.model
    preset = dict(MODEL_PRESETS[args.model])
    if args.model_xml:
        preset["xml_rel"] = args.model_xml
    if args.dimension is not None:
        preset["dimension"] = args.dimension
    if QUERY_TASK_OVERRIDE:
        preset["query_task"] = QUERY_TASK_OVERRIDE
    _state.update(
        preset=preset,
        model_name=MODEL_NAME,
        dimension=int(preset["dimension"]),
    )

    load(args.device, preset=preset)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
