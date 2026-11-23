"""install.py 的纯逻辑单元测试。

测试只使用临时目录和 mock，不写真实配置、不安装依赖、不启动子进程。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import string
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import install
from messages import MESSAGES, Translator, verify_keys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loaded = False
    try:
        spec.loader.exec_module(module)
        loaded = True
    finally:
        if not loaded:
            sys.modules.pop(name, None)
    return module


class VersionInfoTests(unittest.TestCase):
    """VERSION.toml 是项目版本与适配目标的单一来源。"""

    def test_load_version_info_returns_expected_fields(self):
        version_path = ROOT / "VERSION.toml"
        with version_path.open("rb") as handle:
            raw = tomllib.load(handle)
        expected = {
            "project_version": raw["project"]["version"],
            "openviking_version": raw["adapt"]["openviking"],
            "file_bytes": raw["adapt"]["openviking_file_bytes"],
            "file_sha256": raw["adapt"]["openviking_file_sha256"],
        }

        self.assertEqual(install.load_version_info(), expected)
        self.assertEqual(install.load_version_info(version_path), expected)

    def test_missing_version_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "VERSION.toml"
            with self.assertRaises(install.InstallError) as caught:
                install.load_version_info(missing)

            message = str(caught.exception)
        self.assertIn("不存在", message)
        self.assertIn(str(missing), message)

    def test_missing_section_or_field_names_the_problem(self):
        project = '[project]\nversion = "1.0.0"\n'
        adapt = (
            "[adapt]\n"
            'openviking = "0.5.3"\n'
            "openviking_file_bytes = 8774\n"
            'openviking_file_sha256 = "abc"\n'
        )
        cases = {
            "missing project section": (adapt, "[project]"),
            "missing adapt section": (project, "[adapt]"),
            "missing project.version": (
                "[project]\n\n" + adapt,
                "[project].version",
            ),
            "missing adapt.openviking": (
                project + "\n[adapt]\nopenviking_file_bytes = 8774\n"
                'openviking_file_sha256 = "abc"\n',
                "[adapt].openviking",
            ),
            "missing adapt.openviking_file_bytes": (
                project + '\n[adapt]\nopenviking = "0.5.3"\n'
                'openviking_file_sha256 = "abc"\n',
                "[adapt].openviking_file_bytes",
            ),
            "missing adapt.openviking_file_sha256": (
                project + '\n[adapt]\nopenviking = "0.5.3"\n'
                "openviking_file_bytes = 8774\n",
                "[adapt].openviking_file_sha256",
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            version_path = Path(tmp) / "VERSION.toml"
            for label, (content, missing_name) in cases.items():
                with self.subTest(case=label):
                    version_path.write_text(content, encoding="utf-8")
                    with self.assertRaises(install.InstallError) as caught:
                        install.load_version_info(version_path)
                    self.assertIn(missing_name, str(caught.exception))

    def test_wrong_field_type_names_the_problem(self):
        content = (
            '[project]\nversion = "1.0.0"\n\n'
            '[adapt]\nopenviking = "0.5.3"\n'
            'openviking_file_bytes = "8774"\n'
            'openviking_file_sha256 = "abc"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            version_path = Path(tmp) / "VERSION.toml"
            version_path.write_text(content, encoding="utf-8")
            with self.assertRaises(install.InstallError) as caught:
                install.load_version_info(version_path)

        message = str(caught.exception)
        self.assertIn("[adapt].openviking_file_bytes", message)
        self.assertIn("必须是整数", message)

    def test_malformed_version_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            version_path = Path(tmp) / "VERSION.toml"
            cases = {
                "toml syntax": b'[project\nversion = "1.0.0"\n',
                "invalid utf-8": b'\xff\xfe[project]\nversion = "1.0.0"\n',
            }
            for label, content in cases.items():
                with self.subTest(case=label):
                    version_path.write_bytes(content)
                    with self.assertRaises(install.InstallError) as caught:
                        install.load_version_info(version_path)
                    self.assertIn("格式错误", str(caught.exception))

    def test_missing_tomllib_raises_clear_error(self):
        with (
            mock.patch.object(install, "_tomllib", None),
            self.assertRaises(install.InstallError) as caught,
        ):
            install.load_version_info(ROOT / "VERSION.toml")

        message = str(caught.exception)
        self.assertIn("tomllib", message)
        self.assertIn("Python 3.11", message)

    def test_version_argument_outputs_two_lines_in_both_languages(self):
        info = install.load_version_info()
        expected_by_language = {
            "zh": [
                f"openviking-openvino {info['project_version']}",
                f"适配 OpenViking {info['openviking_version']} 官方原版",
            ],
            "en": [
                f"openviking-openvino {info['project_version']}",
                (
                    f"Adapted for OpenViking {info['openviking_version']} "
                    "official release"
                ),
            ],
        }

        for option in ("--version", "-V"):
            for language, expected in expected_by_language.items():
                with self.subTest(option=option, language=language):
                    lines: list[str] = []
                    code = install.main(
                        [option, "--lang", language],
                        output=lines.append,
                    )
                    self.assertEqual(code, 0)
                    self.assertEqual(lines, expected)

    def test_help_lists_version_option_in_both_languages(self):
        for language in ("zh", "en"):
            with self.subTest(language=language):
                lines: list[str] = []
                code = install.main(["--help", "--lang", language], output=lines.append)
                self.assertEqual(code, 0)
                self.assertIn("--version", "\n".join(lines))

    def test_compatibility_aliases_match_version_file(self):
        info = install.load_version_info()
        self.assertEqual(install.ADAPTED_OPENVIKING_VERSION, info["openviking_version"])
        self.assertEqual(install.OFFICIAL_LOCAL_EMBEDDERS_SIZE, info["file_bytes"])
        self.assertEqual(install.OFFICIAL_LOCAL_EMBEDDERS_SHA256, info["file_sha256"])
        self.assertEqual(install.VERSION_INFO.project_version, info["project_version"])


class PatchStateTests(unittest.TestCase):
    """补丁目标三态与版本判定。"""

    def test_official_file_is_unpatched(self):
        state = install.classify_patch_signature(
            install.ADAPTED_OPENVIKING_VERSION,
            install.OFFICIAL_LOCAL_EMBEDDERS_SIZE,
            install.OFFICIAL_LOCAL_EMBEDDERS_SHA256,
            False,
        )
        self.assertEqual(state, install.PATCH_STATE_UNPATCHED)

    def test_patch_marker_means_patched(self):
        state = install.classify_patch_signature(
            install.ADAPTED_OPENVIKING_VERSION,
            17495,
            "0" * 64,
            True,
        )
        self.assertEqual(state, install.PATCH_STATE_PATCHED)

    def test_other_content_means_mismatch(self):
        state = install.classify_patch_signature(
            install.ADAPTED_OPENVIKING_VERSION,
            123,
            "a" * 64,
            False,
        )
        self.assertEqual(state, install.PATCH_STATE_MISMATCH)

    def test_non_target_openviking_version_means_mismatch(self):
        state = install.classify_patch_signature(
            "0.5.4",
            install.OFFICIAL_LOCAL_EMBEDDERS_SIZE,
            install.OFFICIAL_LOCAL_EMBEDDERS_SHA256,
            False,
        )
        self.assertEqual(state, install.PATCH_STATE_MISMATCH)


class SitePackagesCandidateTests(unittest.TestCase):
    """两个补丁脚本各自提供 Windows 与 Linux venv 布局候选。"""

    def test_patch_scripts_match_windows_and_linux_venv_layouts(self):
        scripts = {
            "openvino": ROOT / "patches" / "patch_openvino.py",
            "keepalive": ROOT / "patches" / "patch_http_keepalive.py",
        }
        windows = [os.path.join("Lib", "site-packages")]
        current = f"python{sys.version_info[0]}.{sys.version_info[1]}"
        linux = [
            os.path.join("lib", current, "site-packages"),
            os.path.join("lib", "python3.*", "site-packages"),
        ]

        for name, path in scripts.items():
            with self.subTest(script=name):
                module = _load_module(f"patch_{name}_candidate_test", path)
                with mock.patch.object(module.sys, "platform", "win32"):
                    self.assertEqual(
                        module.site_packages_rel_candidates(),
                        windows,
                    )
                with mock.patch.object(module.sys, "platform", "linux"):
                    self.assertEqual(
                        module.site_packages_rel_candidates(),
                        linux,
                    )

    def test_patch_target_resolution_finds_versioned_linux_site_packages(self):
        scripts = {
            "openvino": (
                ROOT / "patches" / "patch_openvino.py",
                "local_embedders.py",
            ),
            "keepalive": (
                ROOT / "patches" / "patch_http_keepalive.py",
                "openai_embedders.py",
            ),
        }
        fake_version = (3, 13, 0, "final", 0)

        for name, (path, target_name) in scripts.items():
            for version_dir in ("python3.13", "python3.12"):
                with (
                    self.subTest(script=name, version_dir=version_dir),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    home = Path(tmp)
                    target = (
                        home
                        / "venv"
                        / "lib"
                        / version_dir
                        / "site-packages"
                        / "openviking"
                        / "models"
                        / "embedder"
                        / target_name
                    )
                    target.parent.mkdir(parents=True)
                    target.write_text("", encoding="utf-8")

                    module = _load_module(
                        f"patch_{name}_{version_dir}_resolve_test", path
                    )
                    with (
                        mock.patch.dict(
                            os.environ,
                            {"OPENVIKING_HOME": str(home)},
                            clear=True,
                        ),
                        mock.patch.object(module.sys, "platform", "linux"),
                        mock.patch.object(module.sys, "version_info", fake_version),
                    ):
                        self.assertEqual(module.resolve_target(), str(target))

    def test_install_finds_versioned_linux_site_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            site_packages = home / "venv" / "lib" / "python3.13" / "site-packages"
            (site_packages / "openviking").mkdir(parents=True)
            with mock.patch.object(install.sys, "version_info", (3, 13, 0, "final", 0)):
                self.assertEqual(install.find_site_packages(home), site_packages)
                self.assertIn(
                    site_packages,
                    install._site_packages_candidates(home),
                )

    def test_patch_resolution_does_not_escape_explicit_openviking_home(self):
        scripts = {
            "openvino": ROOT / "patches" / "patch_openvino.py",
            "keepalive": ROOT / "patches" / "patch_http_keepalive.py",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "empty-home"
            home.mkdir()
            imported_package = root / "other-openviking"
            imported_package.mkdir()
            package_init = imported_package / "__init__.py"
            package_init.write_text("", encoding="utf-8")
            fake_module = type(sys)("openviking")
            fake_module.__file__ = str(package_init)

            for name, path in scripts.items():
                with self.subTest(script=name):
                    module = _load_module(f"patch_{name}_escape_test", path)
                    with (
                        mock.patch.dict(
                            os.environ,
                            {"OPENVIKING_HOME": str(home)},
                            clear=True,
                        ),
                        mock.patch.dict(sys.modules, {"openviking": fake_module}),
                    ):
                        self.assertIsNone(module.resolve_target())


class ModelFileTests(unittest.TestCase):
    """五款预设模型的 IR 与 tokenizer 文件检查。"""

    def test_all_preset_paths_and_metadata(self):
        expected = {
            "harrier-ov-int8": ("int8/openvino_model.xml", 1024, 32768),
            "Qwen3-Embedding-0.6B-int8-ov": ("openvino_model.xml", 1024, 32768),
            "bge-m3-int8-ov": ("openvino_model.xml", 1024, 8192),
            "SEA-LION-ModernBERT-Embedding-600M-OV": (
                "openvino_model.xml",
                1024,
                8192,
            ),
            "embeddinggemma-2-openvino-int8": ("text_int8.xml", 768, 8192),
        }
        for key, (ir_relative, dimension, context) in expected.items():
            with self.subTest(model=key):
                preset = install.PRESET_MODEL_BY_KEY[key]
                self.assertEqual(preset.ir_relative, ir_relative)
                self.assertEqual(preset.dimension, dimension)
                self.assertEqual(preset.context, context)

    def test_ready_requires_ir_and_root_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            for key, preset in install.PRESET_MODEL_BY_KEY.items():
                with self.subTest(model=key):
                    model_dir = cache_dir / preset.directory
                    ir_path = model_dir / preset.ir_relative
                    ir_path.parent.mkdir(parents=True, exist_ok=True)
                    ir_path.write_bytes(b"ir")
                    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

                    check = install.check_preset_model(cache_dir, key)
                    self.assertTrue(check.ready)
                    self.assertEqual(check.missing, ())

                    (model_dir / "tokenizer.json").unlink()
                    check = install.check_preset_model(cache_dir, key)
                    self.assertFalse(check.ready)
                    self.assertIn(model_dir / "tokenizer.json", check.missing)


class ConfigMergeTests(unittest.TestCase):
    """ov.conf 深度合并与 JSON 校验。"""

    def setUp(self):
        self.existing = {
            "server": {"host": "127.0.0.1", "port": 1933},
            "storage": {"workspace": "D:/OpenViking/data"},
            "embedding": {
                "dense": {
                    "provider": "local",
                    "model": "old-model",
                    "dimension": 768,
                    "cache_dir": "D:/OpenViking/models",
                    "timeout": 30,
                },
                "max_input_tokens": 4096,
                "allow_metadata_override": True,
            },
            "vlm": {"provider": "openai", "model": "keep-me"},
        }

    def test_local_merge_changes_only_dense_target_fields(self):
        merged = install.merge_ov_conf(
            self.existing,
            mode="local",
            model="harrier-ov-int8",
            dimension=1024,
            cache_dir="E:/models",
        )
        self.assertIsNot(merged, self.existing)
        self.assertEqual(merged["server"], self.existing["server"])
        self.assertEqual(merged["storage"], self.existing["storage"])
        self.assertEqual(merged["vlm"], self.existing["vlm"])
        self.assertEqual(
            merged["embedding"]["max_input_tokens"],
            self.existing["embedding"]["max_input_tokens"],
        )
        self.assertEqual(
            merged["embedding"]["allow_metadata_override"],
            self.existing["embedding"]["allow_metadata_override"],
        )
        dense = merged["embedding"]["dense"]
        self.assertEqual(dense["provider"], "local")
        self.assertEqual(dense["model"], "harrier-ov-int8")
        self.assertEqual(dense["dimension"], 1024)
        self.assertEqual(dense["cache_dir"], "E:/models")
        self.assertEqual(dense["timeout"], 30)
        self.assertEqual(self.existing["embedding"]["dense"]["model"], "old-model")

    def test_http_merge_adds_connection_fields(self):
        merged = install.merge_ov_conf(
            self.existing,
            mode="http",
            model="bge-m3-int8-ov",
            dimension=1024,
            cache_dir="E:/models",
            api_base="http://127.0.0.1:8010/v1",
            api_key="local-test-key",
            encoding_format="float",
        )
        dense = merged["embedding"]["dense"]
        self.assertEqual(dense["provider"], "openai")
        self.assertEqual(dense["api_base"], "http://127.0.0.1:8010/v1")
        self.assertEqual(dense["api_key"], "local-test-key")
        self.assertEqual(dense["encoding_format"], "float")
        self.assertEqual(dense["cache_dir"], "E:/models")
        self.assertEqual(dense["timeout"], 30)

    def test_serialized_config_is_valid_json(self):
        merged = install.merge_ov_conf(
            self.existing,
            mode="local",
            model="harrier-ov-int8",
            dimension=1024,
            cache_dir="E:/models",
        )
        text = install.serialize_ov_conf(merged)
        self.assertEqual(json.loads(text)["embedding"]["dense"]["provider"], "local")
        with self.assertRaises(install.InstallError):
            install.validate_ov_conf_text("{")


class BackupNameTests(unittest.TestCase):
    """备份文件命名与防覆盖。"""

    def test_backup_name_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ov.conf"
            stamp = datetime(2026, 11, 22, 13, 5, 9)
            backup = install.make_backup_path(path, now=stamp)
            self.assertEqual(backup.name, "ov.conf.bak-20261122-130509")

            backup.write_text("old", encoding="utf-8")
            second = install.make_backup_path(path, now=stamp)
            self.assertNotEqual(second, backup)
            self.assertRegex(second.name, r"^ov\.conf\.bak-\d{8}-\d{6}$")


class CustomModelProbeTests(unittest.TestCase):
    """自定义模型维度、池化与归一化推荐值探测。"""

    def _write_probe_files(self, root, config, pooling):
        (root / "config.json").write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )
        pooling_dir = root / "1_Pooling"
        pooling_dir.mkdir()
        (pooling_dir / "config.json").write_text(
            json.dumps(pooling, ensure_ascii=False), encoding="utf-8"
        )

    def test_hidden_size_and_cls_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_probe_files(
                root,
                {"hidden_size": 768},
                {"pooling_mode_cls_token": True},
            )
            probe = install.probe_custom_parameters(root)
            self.assertEqual(probe.dimension, 768)
            self.assertEqual(probe.pooling, "cls")
            self.assertTrue(probe.normalize)
            self.assertIsNone(probe.query_prefix)

    def test_text_config_and_mean_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_probe_files(
                root,
                {"text_config": {"hidden_size": 1024}},
                {"pooling_mode_mean_tokens": True},
            )
            probe = install.probe_custom_parameters(root)
            self.assertEqual(probe.dimension, 1024)
            self.assertEqual(probe.pooling, "mean")
            self.assertTrue(probe.normalize)

    def test_last_pooling_and_missing_pooling_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_probe_files(
                root,
                {"hidden_size": 1024},
                {"pooling_mode_lasttoken": True},
            )
            probe = install.probe_custom_parameters(root)
            self.assertEqual(probe.pooling, "last")
            self.assertTrue(probe.normalize)

            (root / "1_Pooling" / "config.json").unlink()
            probe = install.probe_custom_parameters(root)
            self.assertIsNone(probe.pooling)
            self.assertIsNone(probe.normalize)


class AgentHomeDiscoveryTests(unittest.TestCase):
    """agent 宿主目录探测：环境变量、命令反查、默认路径。"""

    def test_hermes_environment_variable_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            hermes_home = Path(tmp) / "custom-hermes"
            hermes_home.mkdir()
            with (
                mock.patch.dict(
                    os.environ, {"HERMES_HOME": str(hermes_home)}, clear=True
                ),
                mock.patch.object(install.shutil, "which", return_value=None),
            ):
                self.assertEqual(install.default_hermes_home(), hermes_home)

    def test_hermes_command_path_finds_first_parent_with_plugins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hermes_home = root / "hermes"
            (hermes_home / "plugins").mkdir(parents=True)
            executable = hermes_home / "hermes-agent" / ".hermes" / "bin" / "hermes"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")
            (hermes_home / "hermes-agent" / "plugins").mkdir()

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    install.shutil, "which", return_value=str(executable)
                ),
            ):
                self.assertEqual(install.default_hermes_home(), hermes_home)

    def test_dsh_environment_variable_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            dsh_home = Path(tmp) / "custom-dsh"
            dsh_home.mkdir()
            with (
                mock.patch.dict(os.environ, {"DSH_HOME": str(dsh_home)}, clear=True),
                mock.patch.object(install.shutil, "which", return_value=None),
            ):
                self.assertEqual(install.default_dsh_home(), dsh_home)

    def test_dsh_command_path_finds_first_parent_with_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dsh_home = root / "dsh"
            (dsh_home / "profiles").mkdir(parents=True)
            executable = dsh_home / "runtime" / "node_modules" / ".bin" / "dsh"
            executable.parent.mkdir(parents=True)
            executable.write_text("", encoding="utf-8")

            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch.object(
                    install.shutil, "which", return_value=str(executable)
                ),
            ):
                self.assertEqual(install.default_dsh_home(), dsh_home)


class AgentRecordTests(unittest.TestCase):
    """agent 接入状态与项目插件安装状态。"""

    def test_agent_records_cover_connected_unconnected_and_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hermes_source = root / "plugins" / "hermes-vlm-follow"
            hermes_source.mkdir(parents=True)
            (hermes_source / "__init__.py").write_text("# plugin\n", encoding="utf-8")
            (hermes_source / "plugin.yaml").write_text(
                "name: hermes-vlm-follow\n", encoding="utf-8"
            )
            dsh_source = root / "plugins" / "dsh-vlm-follow"
            dsh_source.mkdir(parents=True)

            hermes_home = root / "hermes"
            (hermes_home / "plugins" / "openviking").mkdir(parents=True)
            hermes_target = hermes_home / "plugins" / "hermes-vlm-follow"
            hermes_target.mkdir()
            for name in ("__init__.py", "plugin.yaml"):
                (hermes_target / name).write_bytes((hermes_source / name).read_bytes())

            dsh_home = root / "dsh"
            web = dsh_home / "profiles" / "web"
            headless = dsh_home / "profiles" / "headless"
            web.mkdir(parents=True)
            headless.mkdir(parents=True)
            (web / "package.json").write_text(
                json.dumps(
                    {
                        "dependencies": {
                            "@openviking/dsh-memory-plugin": "^0.5.33",
                            "dsh-vlm-follow": "link:" + str(dsh_source.resolve()),
                        }
                    }
                ),
                encoding="utf-8",
            )
            (headless / "package.json").write_text(
                json.dumps({"dependencies": {}}), encoding="utf-8"
            )

            records = install.build_agent_records(hermes_home, dsh_home, root)
            by_key = {record.key: record for record in records}

            self.assertTrue(by_key["hermes"].connected)
            self.assertEqual(by_key["hermes"].plugin_state, install.PLUGIN_STATE_SAME)
            self.assertTrue(by_key["dsh:web"].connected)
            self.assertEqual(by_key["dsh:web"].plugin_state, install.PLUGIN_STATE_SAME)
            self.assertFalse(by_key["dsh:headless"].connected)
            self.assertEqual(by_key["dsh:headless"].missing_plugin, "@openviking/*")

            (hermes_target / "plugin.yaml").write_text(
                "name: changed\n", encoding="utf-8"
            )
            changed = install.build_agent_records(hermes_home, dsh_home, root)
            changed_by_key = {record.key: record for record in changed}
            self.assertEqual(
                changed_by_key["hermes"].plugin_state,
                install.PLUGIN_STATE_DIFFERENT,
            )


class DryRunTests(unittest.TestCase):
    """--dry-run 只生成计划，不调用任何修改函数。"""

    def test_dry_run_does_not_execute_plan_or_write_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            preset = install.PRESET_MODEL_BY_KEY["harrier-ov-int8"]
            selection = install.ModelSelection(
                preset=preset,
                model_name=preset.key,
                dimension=preset.dimension,
                cache_dir=home / "models",
                ir_path=home / "models" / preset.directory / preset.ir_relative,
            )
            environment = install.Environment(
                ov_home=home,
                version="0.5.3",
                site_packages=home / "venv" / "Lib" / "site-packages",
                python_exe=home / "venv" / "Scripts" / "python.exe",
                target_path=home / "local_embedders.py",
                patch_state=install.PATCH_STATE_PATCHED,
                runtime=install.RuntimeInfo(
                    installed=True,
                    has_tokenizers=True,
                    version="2026.4.1",
                    devices=("CPU",),
                ),
                agents=(),
                cache_dir=home / "models",
            )
            action = install.PlannedAction(
                kind="write_config",
                title="更新 OpenViking 配置",
                details=(f"目标：{home / 'ov.conf'}",),
                payload={},
            )
            plan = install.InstallPlan(
                actions=(action,),
                mode="local",
                selection=selection,
                agents=(),
            )
            output = []
            input_calls = []

            with (
                mock.patch.object(
                    install, "collect_environment", return_value=environment
                ),
                mock.patch.object(
                    install, "prompt_embedding_mode", return_value="local"
                ),
                mock.patch.object(
                    install, "prompt_model_selection", return_value=selection
                ),
                mock.patch.object(install, "prompt_agent_selection", return_value=[]),
                mock.patch.object(install, "build_install_plan", return_value=plan),
                mock.patch.object(install, "execute_install_plan") as execute,
                mock.patch.object(install, "atomic_write_text") as atomic_write,
                mock.patch.object(install, "subprocess") as subprocess_module,
            ):
                code = install.main(
                    ["--dry-run", "--ov-home", str(home)],
                    input_func=lambda prompt: input_calls.append(prompt) or "",
                    output=output.append,
                )

            self.assertEqual(code, 0)
            execute.assert_not_called()
            atomic_write.assert_not_called()
            subprocess_module.run.assert_not_called()
            self.assertEqual(input_calls, [])
            self.assertFalse((home / "ov.conf").exists())
            self.assertTrue(any("dry-run" in line for line in output))


class RuntimeInstallTests(unittest.TestCase):
    """OpenVINO 安装命令构造与多安装源尝试。"""

    def _environment(self) -> install.Environment:
        root = Path(tempfile.gettempdir()) / "openviking-install-unit"
        return install.Environment(
            ov_home=root,
            version="0.5.3",
            site_packages=root / "venv" / "Lib" / "site-packages",
            python_exe=root / "venv" / "Scripts" / "python.exe",
            target_path=root / "local_embedders.py",
            patch_state=install.PATCH_STATE_PATCHED,
            runtime=install.RuntimeInfo(installed=False, has_tokenizers=False),
            agents=(),
            cache_dir=root / "models",
        )

    @staticmethod
    def _action(environment: install.Environment) -> install.PlannedAction:
        return install.PlannedAction(
            kind="install_runtime",
            title="安装 OpenVINO Runtime 与 tokenizers",
            details=(),
            payload={
                "command": install._runtime_command(environment),
                "python_exe": environment.python_exe,
                "site_packages": environment.site_packages,
            },
        )

    @staticmethod
    def _completed(returncode: int, text: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(["pip"], returncode, stdout=text, stderr="")

    def test_runtime_command_with_and_without_index_url(self):
        environment = self._environment()
        expected = [
            str(environment.python_exe),
            "-I",
            "-B",
            "-m",
            "pip",
            "install",
            "openvino",
            "tokenizers",
        ]
        self.assertEqual(install._runtime_command(environment), expected)

        index_url = install.PIP_INDEX_CANDIDATES[1][1]
        self.assertEqual(
            install._runtime_command(environment, index_url),
            expected + ["--index-url", index_url],
        )

    def test_pip_index_candidates_start_with_official_source(self):
        self.assertEqual(install.PIP_INDEX_CANDIDATES[0], ("官方源", None))
        self.assertGreaterEqual(len(install.PIP_INDEX_CANDIDATES), 2)
        for name, url in install.PIP_INDEX_CANDIDATES[1:]:
            with self.subTest(source=name):
                self.assertTrue(name)
                self.assertIsNotNone(url)
                self.assertTrue(url.startswith("https://"))

    def test_install_runtime_tries_next_source_after_failure(self):
        environment = self._environment()
        journal = install.ChangeJournal()
        output_lines: list[str] = []
        with (
            mock.patch.object(
                install,
                "_run_subprocess",
                side_effect=[
                    self._completed(1, "官方源不可用"),
                    self._completed(0, "安装完成"),
                ],
            ) as run,
            mock.patch.object(
                install,
                "probe_openvino_runtime",
                return_value=install.RuntimeInfo(installed=True, has_tokenizers=True),
            ) as probe,
            mock.patch.object(
                install,
                "_dist_info_names",
                side_effect=[
                    set(),
                    {"openvino-2026.4.1.dist-info"},
                ],
            ) as dist_info,
        ):
            install._execute_install_runtime(
                self._action(environment), journal, output_lines.append
            )

        self.assertEqual(run.call_count, 2)
        first_command = run.call_args_list[0].args[0]
        second_command = run.call_args_list[1].args[0]
        self.assertNotIn("--index-url", first_command)
        self.assertIn("--index-url", second_command)
        self.assertEqual(second_command[-1], install.PIP_INDEX_CANDIDATES[1][1])
        self.assertEqual(run.call_args_list[0].kwargs["timeout"], 1800)
        self.assertEqual(run.call_args_list[1].kwargs["timeout"], 1800)
        self.assertEqual(dist_info.call_count, 2)
        self.assertIn("        尝试 官方源…", output_lines)
        self.assertIn("        尝试 清华镜像…", output_lines)
        self.assertIn("        清华镜像 成功", output_lines)
        probe.assert_called_once_with(environment.python_exe)
        self.assertEqual(journal.pip_python, environment.python_exe)
        self.assertEqual(journal.pip_new_dists, ["openvino-2026.4.1.dist-info"])

    def test_install_runtime_all_sources_fail(self):
        environment = self._environment()
        journal = install.ChangeJournal()
        output_lines: list[str] = []
        with (
            mock.patch.object(
                install,
                "_run_subprocess",
                side_effect=[
                    self._completed(1, "第一轮失败"),
                    self._completed(1, "第二轮失败"),
                    self._completed(1, "第三轮失败"),
                ],
            ) as run,
            mock.patch.object(install, "probe_openvino_runtime") as probe,
            mock.patch.object(
                install, "_dist_info_names", return_value=set()
            ) as dist_info,
            self.assertRaises(install.InstallError) as caught,
        ):
            install._execute_install_runtime(
                self._action(environment), journal, output_lines.append
            )

        self.assertEqual(run.call_count, len(install.PIP_INDEX_CANDIDATES))
        for call, (_, index_url) in zip(
            run.call_args_list, install.PIP_INDEX_CANDIDATES
        ):
            command = call.args[0]
            if index_url is None:
                self.assertNotIn("--index-url", command)
            else:
                self.assertEqual(command[-2:], ["--index-url", index_url])
        self.assertEqual(dist_info.call_count, 1)
        self.assertIn("pip 安装失败", str(caught.exception))
        self.assertIn("第三轮失败", str(caught.exception))
        probe.assert_not_called()
        self.assertIsNone(journal.pip_python)
        self.assertEqual(journal.pip_new_dists, [])

    def test_successful_pip_is_journaled_before_probe_failure(self):
        environment = self._environment()
        journal = install.ChangeJournal()
        output_lines: list[str] = []
        with (
            mock.patch.object(
                install,
                "_run_subprocess",
                return_value=self._completed(0, "安装完成"),
            ) as run,
            mock.patch.object(
                install,
                "probe_openvino_runtime",
                return_value=install.RuntimeInfo(
                    installed=False,
                    has_tokenizers=False,
                    error="导入失败",
                ),
            ),
            mock.patch.object(
                install,
                "_dist_info_names",
                side_effect=[
                    set(),
                    {
                        "openvino-2026.4.1.dist-info",
                        "tokenizers-0.22.0.dist-info",
                    },
                ],
            ),
            self.assertRaises(install.InstallError) as caught,
        ):
            install._execute_install_runtime(
                self._action(environment), journal, output_lines.append
            )

        self.assertEqual(run.call_count, 1)
        self.assertIn("仍无法导入", str(caught.exception))
        self.assertEqual(journal.pip_python, environment.python_exe)
        self.assertEqual(
            journal.pip_new_dists,
            [
                "openvino-2026.4.1.dist-info",
                "tokenizers-0.22.0.dist-info",
            ],
        )
        self.assertIn("        官方源 成功", output_lines)


class ReleasePackageTests(unittest.TestCase):
    """发布包按平台分开组装，Linux 包不带 Windows 批处理入口。"""

    def test_windows_and_linux_archives_have_expected_entrypoints(self):
        builder_path = ROOT / ".github" / "scripts" / "assemble_release.py"
        self.assertTrue(builder_path.is_file(), builder_path)
        builder = _load_module("assemble_release_under_test", builder_path)

        file_names = (
            "install.py",
            "messages.py",
            "install.cmd",
            "install.ps1",
            "README.md",
            "README.en.md",
            "CHANGELOG.md",
            "VERSION.toml",
            "LICENSE",
        )
        dir_names = ("patches", "http-server", "configs", "plugins", "docs", "models")

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dist = Path(tmp) / "dist"
            source.mkdir()
            for name in file_names:
                (source / name).write_text(f"{name}\n", encoding="utf-8")
            for name in dir_names:
                directory = source / name
                directory.mkdir()
                (directory / "marker.txt").write_text(f"{name}\n", encoding="utf-8")

            packages = builder.assemble_release("v0.1.0", source, dist)
            by_platform = {package.platform: package for package in packages}

            self.assertEqual(set(by_platform), {"windows", "linux"})
            self.assertEqual(
                by_platform["windows"].archive,
                dist / "openviking-openvino-windows-v0.1.0.zip",
            )
            self.assertEqual(
                by_platform["linux"].archive,
                dist / "openviking-openvino-linux-v0.1.0.zip",
            )
            self.assertEqual(
                by_platform["windows"].checksum,
                dist / "openviking-openvino-windows-v0.1.0.sha256",
            )
            self.assertEqual(
                by_platform["linux"].checksum,
                dist / "openviking-openvino-linux-v0.1.0.sha256",
            )

            def names_in(archive: Path) -> set[str]:
                with zipfile.ZipFile(archive) as bundle:
                    self.assertIsNone(bundle.testzip())
                    return set(bundle.namelist())

            windows_names = names_in(by_platform["windows"].archive)
            linux_names = names_in(by_platform["linux"].archive)
            windows_prefix = "openviking-openvino-windows-v0.1.0/"
            linux_prefix = "openviking-openvino-linux-v0.1.0/"
            self.assertIn(windows_prefix + "install.cmd", windows_names)
            self.assertNotIn(linux_prefix + "install.cmd", linux_names)
            self.assertIn(linux_prefix + "install.ps1", linux_names)
            self.assertEqual(by_platform["windows"].file_count, len(windows_names))
            self.assertEqual(by_platform["linux"].file_count, len(linux_names))
            self.assertEqual(
                by_platform["windows"].file_count,
                by_platform["linux"].file_count + 1,
            )

            linux_manifest = by_platform["linux"].checksum.read_text(encoding="utf-8")
            self.assertNotIn("install.cmd", linux_manifest)
            self.assertEqual(
                len(linux_manifest.splitlines()),
                by_platform["linux"].file_count,
            )


class MessageCatalogTests(unittest.TestCase):
    """中英文文案表的键、占位符与插值行为。"""

    def test_language_tables_have_identical_keys(self):
        self.assertEqual(verify_keys(), [])
        self.assertEqual(set(MESSAGES["zh"]), set(MESSAGES["en"]))

    def test_placeholders_match_between_languages(self):
        formatter = string.Formatter()
        for key in sorted(MESSAGES["zh"]):
            with self.subTest(key=key):
                zh_fields = {
                    field_name
                    for _, field_name, _, _ in formatter.parse(MESSAGES["zh"][key])
                    if field_name
                }
                en_fields = {
                    field_name
                    for _, field_name, _, _ in formatter.parse(MESSAGES["en"][key])
                    if field_name
                }
                self.assertEqual(zh_fields, en_fields)

    def test_version_messages_have_expected_placeholders(self):
        formatter = string.Formatter()
        expected_fields = {
            "VERSION_LINE_1": {"project_version"},
            "VERSION_LINE_2": {"openviking_version"},
        }
        for key, fields in expected_fields.items():
            with self.subTest(key=key):
                self.assertIn(key, MESSAGES["zh"])
                self.assertIn(key, MESSAGES["en"])
                for language in ("zh", "en"):
                    actual_fields = {
                        field_name
                        for _, field_name, _, _ in formatter.parse(
                            MESSAGES[language][key]
                        )
                        if field_name
                    }
                    self.assertEqual(actual_fields, fields)

    def test_versioned_messages_use_shared_placeholder(self):
        keys = (
            "ERR_NO_OPENVIKING",
            "ERR_NO_OPENVIKING_HINT",
            "PATCH_ADAPTED_VERSION",
            "PATCH_VERSION_WRONG",
            "PATCH_RESTORE_OFFICIAL",
        )
        for key in keys:
            with self.subTest(key=key):
                self.assertIn("{version}", MESSAGES["zh"][key])
                self.assertIn("{version}", MESSAGES["en"][key])

    def test_translator_interpolates_named_values(self):
        translated = Translator("en").t("ENV_INSTALL_TARGET", path="E:/venv")
        self.assertIn("E:/venv", translated)
        self.assertIn("Install target", translated)

    def test_translator_unknown_key_has_clear_error(self):
        with self.assertRaisesRegex(KeyError, "DOES_NOT_EXIST"):
            Translator("zh").t("DOES_NOT_EXIST")

    def test_model_line_interpolates_preset_key(self):
        preset = install.PRESET_MODEL_BY_KEY["harrier-ov-int8"]
        with tempfile.TemporaryDirectory() as tmp:
            line = install._format_model_line(Path(tmp), 1, preset, Translator("en"))
        self.assertIn("harrier-ov-int8", line)
        self.assertIn("1024 dims", line)

    def test_sample_keys_are_non_empty_and_english_has_no_han(self):
        keys = (
            "STEP1_TITLE",
            "ERR_NO_OPENVIKING",
            "PATCH_STATE_UNPATCHED",
            "MODEL_TABLE_TITLE",
        )
        han = re.compile(r"[\u4e00-\u9fff]")
        for key in keys:
            with self.subTest(key=key):
                self.assertTrue(MESSAGES["zh"][key])
                self.assertTrue(MESSAGES["en"][key])
                self.assertIsNone(han.search(MESSAGES["en"][key]))

    def test_all_english_messages_have_no_han_characters(self):
        han = re.compile(r"[\u4e00-\u9fff]")
        for key, value in MESSAGES["en"].items():
            with self.subTest(key=key):
                self.assertIsNone(han.search(value))


class PlatformSupportTests(unittest.TestCase):
    """仅允许 Windows/Linux 的 x86_64 平台进入环境探测。"""

    def test_macos_is_rejected_before_environment_probe(self):
        lines: list[str] = []
        with (
            mock.patch.object(install.sys, "platform", "darwin"),
            mock.patch.object(
                install, "collect_environment", return_value=None
            ) as collect,
        ):
            code = install.main(["--dry-run", "--lang", "zh"], output=lines.append)

        self.assertEqual(code, 1)
        self.assertEqual(
            "\n".join(lines),
            (
                "不支持 macOS。OpenVINO 在 macOS 上只提供 CPU 推理，"
                "拿不到本项目的 GPU 加速收益。请改用 Windows 或 Linux。"
            ),
        )
        collect.assert_not_called()

    def test_arm_architecture_is_rejected(self):
        lines: list[str] = []
        with (
            mock.patch.object(install.sys, "platform", "linux"),
            mock.patch.object(install.platform, "machine", return_value="aarch64"),
            mock.patch.object(install, "collect_environment") as collect,
        ):
            code = install.main(["--dry-run", "--lang", "zh"], output=lines.append)

        self.assertEqual(code, 1)
        self.assertEqual(
            "\n".join(lines),
            (
                "只支持 x86_64 架构。当前是 aarch64，"
                "OpenVINO 在非 x86 平台没有 GPU 支持，与本项目目标不符。"
            ),
        )
        collect.assert_not_called()

    def test_supported_x86_64_architecture_passes_the_check(self):
        for machine in ("AMD64", "x86_64", "x64"):
            with self.subTest(machine=machine):
                with (
                    mock.patch.object(install.sys, "platform", "linux"),
                    mock.patch.object(
                        install.platform, "machine", return_value=machine
                    ),
                    mock.patch.object(
                        install, "collect_environment", return_value=None
                    ) as collect,
                ):
                    code = install.main(
                        ["--dry-run", "--lang", "zh"], output=lambda _: None
                    )

                self.assertEqual(code, 1)
                collect.assert_called_once()


class LanguageResolutionTests(unittest.TestCase):
    """语言选择优先级：参数、专用变量、POSIX locale、Windows UI、默认值。"""

    @staticmethod
    def _windows_api(lcid=None, error=None):
        windll = mock.Mock()
        function = windll.kernel32.GetUserDefaultUILanguage
        if error is not None:
            function.side_effect = error
        else:
            function.return_value = lcid
        return mock.patch.object(install.ctypes, "windll", windll, create=True)

    def test_explicit_zh_overrides_environment(self):
        with mock.patch.dict(
            os.environ,
            {"OPENVIKING_OPENVINO_LANG": "en", "LC_ALL": "en_US.UTF-8"},
            clear=True,
        ):
            self.assertEqual(install.resolve_language("zh"), "zh")

    def test_explicit_en_overrides_environment(self):
        with mock.patch.dict(
            os.environ,
            {"OPENVIKING_OPENVINO_LANG": "zh", "LANG": "zh_CN.UTF-8"},
            clear=True,
        ):
            self.assertEqual(install.resolve_language("en"), "en")

    def test_dedicated_environment_variable_precedes_locale(self):
        with mock.patch.dict(
            os.environ,
            {
                "OPENVIKING_OPENVINO_LANG": "zh",
                "LC_ALL": "en_US.UTF-8",
                "LANG": "en_US.UTF-8",
            },
            clear=True,
        ):
            self.assertEqual(install.resolve_language("auto"), "zh")

    def test_lc_all_precedes_lang(self):
        with mock.patch.dict(
            os.environ,
            {"LC_ALL": "en_US.UTF-8", "LANG": "zh_CN.UTF-8"},
            clear=True,
        ):
            self.assertEqual(install.resolve_language(None), "en")

    def test_chinese_lang_is_detected(self):
        with mock.patch.dict(os.environ, {"LANG": "zh_CN.UTF-8"}, clear=True):
            self.assertEqual(install.resolve_language("auto"), "zh")

    def test_english_lang_is_detected(self):
        with mock.patch.dict(os.environ, {"LANG": "en_US.UTF-8"}, clear=True):
            self.assertEqual(install.resolve_language("auto"), "en")

    def test_lc_messages_is_used_when_others_are_absent(self):
        with mock.patch.dict(os.environ, {"LC_MESSAGES": "zh_CN.UTF-8"}, clear=True):
            self.assertEqual(install.resolve_language("auto"), "zh")

    def test_windows_chinese_ui_is_detected(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(install.os, "name", "nt"),
            self._windows_api(0x0804),
        ):
            self.assertEqual(install.resolve_language("auto"), "zh")

    def test_windows_ai_failure_falls_through_to_english(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(install.os, "name", "nt"),
            self._windows_api(error=OSError("unavailable")),
        ):
            self.assertEqual(install.resolve_language("auto"), "en")

    def test_no_signal_defaults_to_english(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(install.os, "name", "nt"),
            self._windows_api(0x0409),
        ):
            self.assertEqual(install.resolve_language(None), "en")

    def test_unrecognized_explicit_value_behaves_like_auto(self):
        with mock.patch.dict(os.environ, {"LANG": "zh_CN.UTF-8"}, clear=True):
            self.assertEqual(install.resolve_language("fr"), "zh")

    def test_parse_args_records_language(self):
        options = install.parse_args(["--lang", "en"])
        self.assertEqual(options.lang, "en")

    def test_english_help_has_no_han_characters(self):
        lines: list[str] = []
        code = install.main(["--help", "--lang", "en"], output=lines.append)
        text = "\n".join(lines)
        self.assertEqual(code, 0)
        self.assertIn("--lang", text)
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", text))


if __name__ == "__main__":
    unittest.main()
