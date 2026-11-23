# -*- coding: utf-8 -*-
# ruff: noqa: UP009, UP031
"""补丁脚本的 custom 支持单元测试。

覆盖模型表项、维度放行、sidecar 参数读取、参数优先级、前缀与归一化，
不加载真实 OpenVINO 模型。
"""

import contextlib
import importlib.util
import io
import json
import logging
import sys
import sysconfig
import tempfile
import threading
import types
import unittest
from pathlib import Path

_SITE_PACKAGES = sysconfig.get_paths().get("purelib")
if _SITE_PACKAGES:
    sys.path.insert(0, _SITE_PACKAGES)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "patches" / "patch_openvino.py"
sys.path.insert(0, str(ROOT / "tests"))

_LOGGER = logging.getLogger("test.openvino.custom")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载模块: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch_module = _load_module("_patch_openvino_custom_test", PATCH_PATH)
verify_models = _load_module(
    "_verify_models_custom_test", ROOT / "tests" / "verify_models.py"
)


def _build_ov_host_class():
    """把补丁中的 OpenVINO 方法编译成可实例化的测试宿主类。"""
    namespace = {
        "Path": Path,
        "List": list,
        "EmbeddingConfigurationError": type(
            "EmbeddingConfigurationError", (Exception,), {}
        ),
        "logger": _LOGGER,
        "os": __import__("os"),
    }
    source = "class _OvHost:\n" + patch_module.OV_IMPL
    exec(compile(source, "<patch_openvino.OV_IMPL>", "exec"), namespace)  # noqa: S102
    return namespace["_OvHost"]


def _custom_table_block():
    table = patch_module.NEW_TABLE_TAIL
    marker = '"custom": LocalModelSpec('
    start = table.index(marker)
    end = table.index("\n    ),", start)
    return table[start:end]


class CustomTableTests(unittest.TestCase):
    def test_custom_is_registered_after_presets(self):
        self.assertEqual(patch_module.MODEL_NAMES[-1], "custom")
        table = patch_module.OV_TABLE_ENTRIES
        self.assertLess(
            table.index('"embeddinggemma-2-openvino-int8": LocalModelSpec('),
            table.index('"custom": LocalModelSpec('),
        )

    def test_custom_table_entry_uses_placeholder_values(self):
        block = _custom_table_block()
        expected_fields = (
            "dimension=0",
            'filename=""',
            'download_url=""',
            "query_instruction=None",
            "pooling_type=None",
            'backend="openvino"',
            "ov_model_dir=None",
            'ov_pooling="last"',
            "ov_normalize=True",
        )
        for field in expected_fields:
            self.assertIn(field, block)


class DimensionGuardTests(unittest.TestCase):
    def test_custom_skips_fixed_dimension_check(self):
        text = "before\n" + patch_module.OLD_DIMENSION_GUARD + "after\n"
        with contextlib.redirect_stdout(io.StringIO()):
            patched, changed = patch_module.apply(
                text,
                "维度校验跳过 custom",
                patch_module.OLD_DIMENSION_GUARD,
                patch_module.NEW_DIMENSION_GUARD,
                'model_name != "custom"',
            )
        self.assertTrue(changed)
        self.assertIn(
            'if model_name != "custom" and '
            "self._dimension != self.model_spec.dimension:",
            patched,
        )
        self.assertNotIn(
            "        if self._dimension != self.model_spec.dimension:\n"
            "            raise ValueError(",
            patched,
        )

    def test_dimension_guard_replacement_is_idempotent(self):
        text = "before\n" + patch_module.OLD_DIMENSION_GUARD + "after\n"
        with contextlib.redirect_stdout(io.StringIO()):
            patched, _ = patch_module.apply(
                text,
                "维度校验跳过 custom",
                patch_module.OLD_DIMENSION_GUARD,
                patch_module.NEW_DIMENSION_GUARD,
                'model_name != "custom"',
            )
            repeated, changed = patch_module.apply(
                patched,
                "维度校验跳过 custom",
                patch_module.OLD_DIMENSION_GUARD,
                patch_module.NEW_DIMENSION_GUARD,
                'model_name != "custom"',
            )
        self.assertFalse(changed)
        self.assertEqual(repeated, patched)


class CustomConfigTests(unittest.TestCase):
    def setUp(self):
        self.host = _build_ov_host_class()()

    def test_missing_sidecar_warns_and_uses_defaults(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertLogs(_LOGGER, level="WARNING") as logs,
        ):
            config = self.host._ov_load_custom_config(Path(tmp))
        self.assertEqual(
            config,
            {
                "pooling": "last",
                "normalize": True,
                "query_task": "",
                "query_prefix": "",
                "doc_prefix": "",
            },
        )
        self.assertIn("openvino_custom.json 不存在", "\n".join(logs.output))

    def test_partial_sidecar_merges_over_defaults(self):
        payload = {
            "pooling": "model",
            "normalize": False,
            "query_task": "retrieve",
            "query_prefix": "Q: ",
            "doc_prefix": "D: ",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertLogs(_LOGGER, level="INFO") as logs,
        ):
            Path(tmp, "openvino_custom.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            config = self.host._ov_load_custom_config(Path(tmp))
        self.assertEqual(config, payload)
        self.assertIn("custom 模型参数", "\n".join(logs.output))

    def test_invalid_pooling_falls_back_to_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "openvino_custom.json").write_text(
                json.dumps({"pooling": "median", "normalize": False}),
                encoding="utf-8",
            )
            with self.assertLogs(_LOGGER, level="WARNING") as logs:
                config = self.host._ov_load_custom_config(Path(tmp))
        self.assertEqual(config["pooling"], "last")
        self.assertFalse(config["normalize"])
        self.assertIn("pooling 取值", "\n".join(logs.output))

    def test_malformed_json_warns_and_uses_defaults(self):
        for payload in ("{", "[]"):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmp:
                    Path(tmp, "openvino_custom.json").write_text(
                        payload, encoding="utf-8"
                    )
                    with self.assertLogs(_LOGGER, level="WARNING"):
                        config = self.host._ov_load_custom_config(Path(tmp))
                self.assertEqual(config["pooling"], "last")
                self.assertTrue(config["normalize"])


class RuntimeParamTests(unittest.TestCase):
    def setUp(self):
        self.host = _build_ov_host_class()()

    def test_custom_values_take_priority_over_model_spec(self):
        self.host._ov_custom = {"pooling": "last"}
        self.host.model_spec = types.SimpleNamespace(
            ov_pooling="cls",
            ov_normalize=True,
            ov_query_task="preset task",
            ov_query_prefix="P: ",
            ov_doc_prefix="PD: ",
        )
        self.assertEqual(self.host._ov_param("pooling"), "last")
        self.assertTrue(self.host._ov_param("normalize"))

        self.host._ov_custom = None
        self.assertEqual(self.host._ov_param("pooling"), "cls")
        self.assertEqual(self.host._ov_param("query_task"), "preset task")
        self.assertEqual(self.host._ov_param("query_prefix"), "P: ")
        self.assertEqual(self.host._ov_param("doc_prefix"), "PD: ")

    def test_custom_prefixes_and_instruction_are_applied(self):
        self.host._ov_custom = {
            "query_task": "retrieve",
            "query_prefix": "Q: ",
            "doc_prefix": "D: ",
        }
        self.assertEqual(self.host._ov_apply_prefix("文本", is_query=True), "Q: 文本")
        self.assertEqual(self.host._ov_apply_prefix("文本", is_query=False), "D: 文本")

        self.host._ov_custom["query_prefix"] = ""
        self.assertEqual(
            self.host._ov_apply_prefix("文本", is_query=True),
            "Instruct: retrieve\nQuery: 文本",
        )

    def test_custom_normalize_switch_controls_output(self):
        class FakeTokenizer:
            @staticmethod
            def encode(_text):
                return types.SimpleNamespace(ids=[1, 2], attention_mask=[1, 1])

        class FakeCompiled:
            @staticmethod
            def __call__(_inputs):
                return {"out": np.array([[3.0, 4.0]], dtype=np.float32)}

        self.host._ov_tokenizer = FakeTokenizer()
        self.host._ov_compiled = FakeCompiled()
        self.host._ov_output = "out"
        self.host._llama_lock = threading.Lock()

        self.host._ov_custom = {"normalize": False}
        np.testing.assert_allclose(self.host._ov_embed_one("x"), [3.0, 4.0])

        self.host._ov_custom = {"normalize": True}
        np.testing.assert_allclose(self.host._ov_embed_one("x"), [0.6, 0.8], atol=1e-6)


class VerifyModelsTests(unittest.TestCase):
    def test_custom_is_excluded_from_preset_alignment(self):
        table = patch_module.NEW_TABLE_TAIL
        custom = """

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
}
"""
        fake_table = table[: table.rfind("}\n")] + custom
        preset_names = tuple(
            name for name in patch_module.MODEL_NAMES if name != "custom"
        )
        fake_patch = types.SimpleNamespace(
            MODEL_NAMES=preset_names + ("custom",),
            NEW_TABLE_TAIL=fake_table,
        )
        specs = verify_models.parse_patch_specs(fake_patch)
        self.assertEqual(set(specs), set(preset_names))
        self.assertNotIn("custom", specs)


if __name__ == "__main__":
    unittest.main()
