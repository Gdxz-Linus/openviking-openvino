# -*- coding: utf-8 -*-
"""OpenAI 兼容的嵌入服务，后端用 OpenVINO 加载 int8 IR 模型。

为什么需要它：OpenViking 的嵌入 provider 只有 local（llama.cpp）和 openai（HTTP）等，
没有 OpenVINO。把它包成 OpenAI 接口，OpenViking 就能零改源码地用上 OpenVINO。

启动：
  python embed_server.py --model-dir D:/models/harrier-ov-int8 --port 8010
"""
import argparse
import json
import logging
import os
import sys
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = os.environ.get("OV_EMBED_MODEL_DIR", "")
MODEL_NAME = os.environ.get("OV_EMBED_MODEL_NAME", "harrier-ov-int8")
CACHE_DIR = os.environ.get("OV_EMBED_CACHE", "")
MAX_TOKENS = int(os.environ.get("OV_EMBED_MAX_TOKENS", "4096"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-server")

app = FastAPI(title="OpenVINO Embedding Server")

_state = {"compiled": None, "out": None, "tok": None, "lock": threading.Lock(), "device": ""}


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


def load(device: str) -> None:
    import openvino as ov
    from tokenizers import Tokenizer

    xml = os.path.join(MODEL_DIR, "int8", "openvino_model.xml")
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(xml):
        raise SystemExit("模型文件不存在: %s" % xml)

    tok = Tokenizer.from_file(tok_path)
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(pad_id=151643, pad_token="<|endoftext|>")

    core = ov.Core()
    if device not in core.available_devices:
        log.warning("设备 %s 不可用，可用: %s，改用 CPU", device, core.available_devices)
        device = "CPU"
    model = core.read_model(xml)
    t0 = time.time()
    compiled = core.compile_model(model, device, {"CACHE_DIR": CACHE_DIR})
    log.info("模型就绪 device=%s 编译耗时 %.1f s", device, time.time() - t0)

    _state.update(compiled=compiled, out=compiled.output(0), tok=tok, device=device)


def embed_one(text: str):
    tok = _state["tok"]
    compiled = _state["compiled"]
    out = _state["out"]
    with _state["lock"]:
        enc = tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        r = compiled({"input_ids": ids, "attention_mask": mask})
        return np.asarray(r[out]).ravel()


# ---------- harrier 官方提示前缀 ----------
# 官方要求：查询侧必须加 "Instruct: <任务>\nQuery: <文本>"，文档侧不加。
# 不加前缀时长文档检索会从 95% 掉到 5%（官方基准实测）。
# OpenViking 在对外 HTTP 层不会替我们拼前缀，所以由本服务按 extra_body.input_type 区分。
QUERY_TASK = os.environ.get(
    "OV_EMBED_QUERY_TASK",
    "Given a web search query, retrieve relevant passages that answer the query",
)


def apply_prefix(text: str, is_query: bool) -> str:
    if not is_query:
        return text
    return "Instruct: %s\nQuery: %s" % (QUERY_TASK, text)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "healthy": _state["compiled"] is not None,
        "device": _state["device"],
        "model": MODEL_NAME,
        "dim": 1024,
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local-openvino"}],
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
    log.info("embeddings n=%d chars=%d query=%s %.0f ms",
             len(texts), sum(len(t) for t in texts), is_query, elapsed * 1000)
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


def main():
    global MODEL_DIR, CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=MODEL_DIR,
                    help="OpenVINO IR 模型目录，内含 int8/openvino_model.xml 与 tokenizer.json")
    ap.add_argument("--cache-dir", default=CACHE_DIR,
                    help="OpenVINO 编译缓存目录，留空则放到模型目录旁边")
    ap.add_argument("--device", default=os.environ.get("OV_EMBED_DEVICE", "GPU"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    CACHE_DIR = args.cache_dir
    if not MODEL_DIR:
        ap.error("必须指定 --model-dir（或设置 OV_EMBED_MODEL_DIR）")
    if not CACHE_DIR:
        CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(MODEL_DIR)), "ov_cache")

    load(args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
