# -*- coding: utf-8 -*-
# ruff: noqa: UP009, UP031
"""实测校验五款 OpenVINO 预设嵌入模型；custom 由 sidecar 参数运行，不参与对齐。

期望参数直接取 patches/patch_openvino.py 的模型表，并与
http-server/embed_server.py 的 MODEL_PRESETS 逐项比对。模型权重只读，
OpenVINO 编译缓存写入系统临时目录，不写入模型目录。

用法：
  python tests/verify_models.py \
      --models harrier-ov-int8,Qwen3-Embedding-0.6B-int8-ov \
      --cache-dir <模型根目录>
"""

import argparse
import ast
import gc
import importlib.util
import os
import re
import sys
import sysconfig
import tempfile
from pathlib import Path

# 外部 PYTHONPATH 可能注入其他 Python 版本的 site-packages，先恢复当前解释器。
_SITE_PACKAGES = sysconfig.get_paths().get("purelib")
if _SITE_PACKAGES:
    sys.path.insert(0, _SITE_PACKAGES)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "patches" / "patch_openvino.py"
SERVER_PATH = ROOT / "http-server" / "embed_server.py"
EXPECTED_FIELDS = (
    "dimension",
    "xml_rel",
    "pooling",
    "normalize",
    "query_task",
    "query_prefix",
    "doc_prefix",
)

TEXTS = (
    ("查询", "OpenVINO embedding server deployment", True),
    (
        "相关文档",
        "The embedding server runs OpenVINO models and exposes an OpenAI-compatible API.",
        False,
    ),
    ("无关文档", "The recipe calls for two eggs, flour, and sugar.", False),
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载模块: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _string_field(block, key):
    match = re.search(re.escape(key) + r'=("[^"]*")', block)
    if not match:
        raise ValueError("模型表缺少字段 %s" % key)
    return ast.literal_eval(match.group(1))


def _optional_string(block, key):
    if re.search(re.escape(key) + r"=None\b", block):
        return None
    return _string_field(block, key)


def _optional_task(block):
    if re.search(r"ov_query_task=None\b", block):
        return None
    match = re.search(r'ov_query_task=\(\s*"([^"]*)"\s*\)', block, re.DOTALL)
    if not match:
        raise ValueError("模型表缺少字段 ov_query_task")
    return match.group(1)


def parse_patch_specs(patch_module):
    """从补丁生成后的模型表字符串中解析五款预设参数。"""
    table = patch_module.NEW_TABLE_TAIL
    specs = {}
    for name in patch_module.MODEL_NAMES:
        if name == "custom":
            continue
        marker = '"%s": LocalModelSpec(' % name
        try:
            start = table.index(marker)
            end = table.index("\n    ),", start)
        except ValueError as exc:
            raise ValueError("补丁模型表缺少表项: %s" % name) from exc
        block = table[start:end]

        dimension_match = re.search(r"\bdimension=(\d+)", block)
        normalize_match = re.search(r"ov_normalize=(True|False)", block)
        if not dimension_match or not normalize_match:
            raise ValueError("模型表项字段不完整: %s" % name)
        filename = _string_field(block, "filename")
        model_dir = _string_field(block, "ov_model_dir")
        prefix = model_dir + "/"
        if not filename.startswith(prefix):
            raise ValueError("filename 不在 ov_model_dir 下: %s" % name)
        specs[name] = {
            "dimension": int(dimension_match.group(1)),
            "xml_rel": filename[len(prefix) :],
            "pooling": _string_field(block, "ov_pooling"),
            "normalize": normalize_match.group(1) == "True",
            "query_task": _optional_task(block),
            "query_prefix": _optional_string(block, "ov_query_prefix"),
            "doc_prefix": _optional_string(block, "ov_doc_prefix"),
        }
    return specs


def _cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("余弦相似度输入含零向量")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _alignment_errors(expected, actual):
    if actual is None:
        return ["embed_server.py 缺少同名模型预设"]
    errors = []
    for key in EXPECTED_FIELDS:
        if expected[key] != actual.get(key):
            errors.append(
                "%s 不一致：补丁=%r，服务=%r" % (key, expected[key], actual.get(key))
            )
    return errors


def verify_model(
    name, expected, server, cache_dir, compile_cache_dir, device, max_tokens
):
    """返回 None 表示跳过，True 表示通过，False 表示失败。"""
    print("\n[%s]" % name, flush=True)
    print(
        "  期望：dimension=%s pooling=%s normalize=%s"
        % (expected["dimension"], expected["pooling"], expected["normalize"])
    )

    model_dir = cache_dir / name
    if not model_dir.is_dir():
        print("  [跳过] 模型目录不存在：%s" % model_dir)
        return None

    xml_path = model_dir / expected["xml_rel"]
    if not xml_path.is_file():
        print("  [失败] IR 文件不存在：%s" % xml_path)
        return False

    alignment_errors = _alignment_errors(expected, server.MODEL_PRESETS.get(name))
    if alignment_errors:
        for message in alignment_errors:
            print("  [失败] %s" % message)
        return False

    server.MODEL_DIR = str(model_dir)
    server.CACHE_DIR = str(compile_cache_dir)
    server.MODEL_NAME = name
    server.MAX_TOKENS = max_tokens
    preset = dict(server.MODEL_PRESETS[name])
    server._state.update(preset=preset, model_name=name, dimension=preset["dimension"])

    try:
        server.load(device, preset=preset)
        vectors = {}
        for label, text, is_query in TEXTS:
            formatted = server.build_prefix(text, is_query, preset)
            vectors[label] = np.asarray(server.embed_one(formatted), dtype=np.float64)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - 单模型失败不能中断整批验证。
        print("  [失败] 加载或推理异常：%s: %s" % (type(exc).__name__, exc))
        return False
    finally:
        server._state["compiled"] = None
        server._state["out"] = None
        server._state["tok"] = None
        gc.collect()

    failures = []
    for label, vector in vectors.items():
        if vector.size != expected["dimension"]:
            failures.append(
                "%s 输出维度 %d，期望 %d" % (label, vector.size, expected["dimension"])
            )
    if failures:
        for message in failures:
            print("  [失败] %s" % message)
        return False
    print("  [通过] 三条文本输出维度均为 %d" % expected["dimension"])

    if expected["normalize"]:
        bad_norms = [
            (label, float(np.linalg.norm(vector)))
            for label, vector in vectors.items()
            if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-3
        ]
        if bad_norms:
            for label, norm in bad_norms:
                print("  [失败] %s 的 L2 范数为 %.9f，未接近 1" % (label, norm))
            return False
        print("  [通过] 三条向量 L2 范数均接近 1")
    else:
        norms = ", ".join(
            "%s=%.4f" % (label, float(np.linalg.norm(vector)))
            for label, vector in vectors.items()
        )
        print("  [跳过] 该模型预设不需要归一化：%s" % norms)

    related = _cosine(vectors["查询"], vectors["相关文档"])
    unrelated = _cosine(vectors["查询"], vectors["无关文档"])
    print("  余弦：查询-相关=%.6f，查询-无关=%.6f" % (related, unrelated))
    if related <= unrelated:
        print("  [失败] 相关文档余弦未大于无关文档")
        return False
    print("  [通过] 相关文档余弦大于无关文档")
    return True


def resolve_model_dir(explicit=None):
    """按显式路径、环境变量和解释器邻近的 venv 顺序探测模型根目录。"""
    if explicit:
        return explicit

    model_dir = os.environ.get("OPENVIKING_MODEL_DIR")
    if model_dir:
        return model_dir

    home = os.environ.get("OPENVIKING_HOME")
    if home:
        return os.path.join(home, "models")

    # 已安装 openviking 包的路径无法可靠映射到模型根目录，按约定跳过此级。

    executable = getattr(sys, "executable", None)
    if executable:
        for parent in Path(os.path.abspath(executable)).parents:
            if parent.name.lower() == "venv":
                return os.path.join(str(parent.parent), "models")
    return None


def parse_args(argv):
    parser = argparse.ArgumentParser(description="实测校验 OpenVINO 嵌入模型")
    parser.add_argument(
        "--models",
        default="",
        help="逗号分隔的模型名；留空时校验补丁表中的全部五款",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="模型根目录（只读）；未指定时按 OPENVIKING_MODEL_DIR、OPENVIKING_HOME/models 顺序探测",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("OV_EMBED_DEVICE", "GPU"),
        help="OpenVINO 设备，默认 GPU，不可用时由服务端回落 CPU",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="分词截断长度，默认 4096",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.max_tokens <= 0:
        print("--max-tokens 必须是正整数", file=sys.stderr)
        return 1

    cache_dir_value = resolve_model_dir(args.cache_dir)
    if not cache_dir_value:
        print(
            "未找到模型目录，请用 --cache-dir 或设置 OPENVIKING_MODEL_DIR",
            file=sys.stderr,
        )
        return 1

    try:
        patch_module = _load_module("_patch_openvino_verify", PATCH_PATH)
        server = _load_module("_embed_server_verify", SERVER_PATH)
        patch_specs = parse_patch_specs(patch_module)
    except Exception as exc:  # noqa: BLE001 - 初始化异常统一转为中文错误报告。
        print("初始化失败：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1

    available = list(patch_specs)
    selected = [
        item.strip()
        for item in (args.models or ",".join(available)).split(",")
        if item.strip()
    ]
    if not selected:
        print("--models 不能为空", file=sys.stderr)
        return 1
    unknown = [name for name in selected if name not in patch_specs]
    if unknown:
        print(
            "未知模型：%s；可选：%s" % (", ".join(unknown), ", ".join(available)),
            file=sys.stderr,
        )
        return 1

    cache_dir = Path(cache_dir_value).expanduser()
    print("=" * 78, flush=True)
    print("OpenVINO 嵌入模型实测校验")
    print("=" * 78)
    print("  模型根目录：%s（只读）" % cache_dir)
    print("  编译缓存：系统临时目录（不写入模型目录）")
    print("  设备：%s" % args.device)
    print()

    passed = 0
    failed = 0
    skipped = 0
    with tempfile.TemporaryDirectory(
        prefix="openviking_verify_ov_cache_", ignore_cleanup_errors=True
    ) as compile_cache:
        for name in selected:
            result = verify_model(
                name,
                patch_specs[name],
                server,
                cache_dir,
                Path(compile_cache),
                args.device,
                args.max_tokens,
            )
            if result is None:
                skipped += 1
            elif result:
                passed += 1
            else:
                failed += 1

    print("\n" + "=" * 78)
    print("汇总：通过 %d，失败 %d，跳过 %d" % (passed, failed, skipped))
    if skipped:
        print("说明：模型目录不存在属于跳过项，不计为失败。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
