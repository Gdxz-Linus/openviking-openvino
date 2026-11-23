# -*- coding: utf-8 -*-
# ruff: noqa: UP009
"""HTTP 嵌入服务的纯函数单元测试。

覆盖池化、归一化、模型前缀和 tokenizer 补位配置，不加载 OpenVINO 模型。
"""

import importlib
import json
import sys
import sysconfig
import tempfile
import types
import unittest
from pathlib import Path

# 某些宿主会把其他 Python 版本的 site-packages 注入 PYTHONPATH，导致当前
# 解释器误加载不兼容的 NumPy。显式把当前解释器自身的 site-packages 放前面。
_SITE_PACKAGES = sysconfig.get_paths().get("purelib")
if _SITE_PACKAGES:
    sys.path.insert(0, _SITE_PACKAGES)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HTTP_SERVER_DIR = ROOT / "http-server"
sys.path.insert(0, str(HTTP_SERVER_DIR))


def _install_http_stubs():
    """仅在测试环境缺少 Web 依赖时提供最小桩，用于导入纯函数。"""

    class DummyApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return lambda fn: fn

        def post(self, *_args, **_kwargs):
            return lambda fn: fn

    class DummyBaseModel:
        pass

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = DummyApp
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = DummyBaseModel
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *_args, **_kwargs: None
    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("pydantic", pydantic)
    sys.modules.setdefault("uvicorn", uvicorn)


def _import_embed_server():
    try:
        return importlib.import_module("embed_server")
    except ModuleNotFoundError as exc:
        if exc.name not in {"fastapi", "pydantic", "uvicorn"}:
            raise
        _install_http_stubs()
        return importlib.import_module("embed_server")


embed_server = _import_embed_server()


class FakeTokenizer:
    """只实现补位解析需要的最小 tokenizer 接口。"""

    def __init__(self, vocab, zero_token="<zero>"):
        self._vocab = dict(vocab)
        self._id_to_token = {idx: token for token, idx in self._vocab.items()}
        if 0 not in self._id_to_token:
            self._id_to_token[0] = zero_token
        self.padding = None

    def get_vocab(self):
        return dict(self._vocab)

    def id_to_token(self, token_id):
        return self._id_to_token.get(int(token_id))

    def enable_padding(self, **kwargs):
        self.padding = kwargs


class PoolingTests(unittest.TestCase):
    def test_two_dimensional_output_returns_first_row(self):
        raw = np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
        mask = np.ones((1, 2), dtype=np.int64)
        result = embed_server._pool(raw, mask, "model")
        np.testing.assert_array_equal(result, raw[0])

    def test_last_pooling_uses_last_unmasked_token(self):
        raw = np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)
        mask = np.array([[1, 1, 1, 0]], dtype=np.int64)
        result = embed_server._pool(raw, mask, "last")
        np.testing.assert_array_equal(result, raw[0, 2])
        self.assertFalse(np.array_equal(result, raw[0, 3]))

    def test_cls_pooling_uses_first_token(self):
        raw = np.arange(1 * 3 * 2, dtype=np.float32).reshape(1, 3, 2)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        result = embed_server._pool(raw, mask, "cls")
        np.testing.assert_array_equal(result, raw[0, 0])

    def test_mean_pooling_ignores_padding(self):
        raw = np.array([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]])
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        result = embed_server._pool(raw, mask, "mean")
        np.testing.assert_allclose(result, np.array([4.0, 6.0]))

    def test_invalid_rank_raises_value_error(self):
        with self.assertRaises(ValueError):
            embed_server._pool(np.zeros(4, dtype=np.float32), np.ones((1, 4)), "last")
        with self.assertRaises(ValueError):
            embed_server._pool(
                np.zeros((1, 2, 3, 4), dtype=np.float32), np.ones((1, 2)), "last"
            )

    def test_unknown_three_dimensional_pooling_raises_value_error(self):
        with self.assertRaises(ValueError):
            embed_server._pool(
                np.zeros((1, 2, 3), dtype=np.float32),
                np.ones((1, 2), dtype=np.int64),
                "unknown",
            )


class NormalizationTests(unittest.TestCase):
    def test_normalization_enabled_produces_unit_l2_norm(self):
        vector = np.array([3.0, 4.0], dtype=np.float32)
        result = embed_server._normalize_vector(vector, True)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)
        np.testing.assert_allclose(result, np.array([0.6, 0.8], dtype=np.float32))

    def test_normalization_disabled_preserves_values(self):
        vector = np.array([3.0, 4.0], dtype=np.float32)
        result = embed_server._normalize_vector(vector, False)
        np.testing.assert_array_equal(result, vector)


class PrefixTests(unittest.TestCase):
    def test_instruction_prefix_matches_openvino_patch(self):
        preset = embed_server.MODEL_PRESETS["harrier-ov-int8"]
        result = embed_server.build_prefix("文本", True, preset)
        expected = (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: 文本"
        )
        self.assertEqual(result, expected)

    def test_literal_query_prefix_takes_priority(self):
        preset = embed_server.MODEL_PRESETS["embeddinggemma-2-openvino-int8"]
        self.assertEqual(
            embed_server.build_prefix("文本", True, preset),
            "task: search result | query: 文本",
        )
        self.assertEqual(
            embed_server.build_prefix("文本", False, preset),
            "title: none | text: 文本",
        )

    def test_document_side_has_no_instruction_prefix(self):
        preset = embed_server.MODEL_PRESETS["bge-m3-int8-ov"]
        self.assertEqual(embed_server.build_prefix("文本", False, preset), "文本")


class PaddingTests(unittest.TestCase):
    def _configure(self, config, vocab=None):
        tokenizer = FakeTokenizer(vocab or {"<pad>": 7, "<zero>": 0})
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "tokenizer_config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False), encoding="utf-8"
            )
            embed_server.configure_tokenizer_padding(tokenizer, tmp)
        return tokenizer

    def test_pad_token_name_resolves_from_vocabulary(self):
        tokenizer = self._configure({"pad_token": "<pad>"})
        self.assertEqual(tokenizer.padding["pad_id"], 7)
        self.assertEqual(tokenizer.padding["pad_token"], "<pad>")

    def test_missing_pad_fields_fall_back_to_zero(self):
        tokenizer = self._configure({})
        self.assertEqual(tokenizer.padding["pad_id"], 0)
        self.assertEqual(tokenizer.padding["pad_token"], "<zero>")

    def test_out_of_range_pad_id_falls_back_to_zero(self):
        tokenizer = self._configure({"pad_token_id": 999})
        self.assertEqual(tokenizer.padding["pad_id"], 0)
        self.assertEqual(tokenizer.padding["pad_token"], "<zero>")


if __name__ == "__main__":
    unittest.main()
