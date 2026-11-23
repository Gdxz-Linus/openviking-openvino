# -*- coding: utf-8 -*-
# ruff: noqa: UP009, UP031
"""给 OpenViking 的 HTTP 嵌入客户端调高连接保活时间。

背景
----
OpenViking 走 ``provider: openai`` 时，嵌入请求由
``openviking/models/embedder/openai_embedders.py`` 里的 httpx 客户端发出。
官方把 ``keepalive_expiry`` 设为 5 秒。对接自建的 OpenVINO 嵌入服务
（单机回环，响应通常 2-6 ms）时，批量写入或重建索引的相邻请求间隔经常超过
5 秒，连接因此反复过期重建。

实测症状（2026-11-09，8010 自建服务）：服务端日志出现 560 次
``duration_ms≈7800`` 的慢调用，全部落在重连窗口内。

修法
----
把 ``keepalive_expiry`` 从 5.0 提到 60.0，让连接在常规请求间隔内保持复用。
uvicorn 的 keep-alive 默认是 5 秒，客户端设 60 秒不会超出服务端的容忍范围：
服务端会在自己的 5 秒后关闭空闲连接，客户端下次请求时重建一次，不产生错误。

适用前提
--------
只对 ``provider: openai`` 且指向本机或局域网嵌入服务的部署有意义。
对接公有云 API（OpenAI、Jina、Voyage 等）时不需要这个补丁，请勿打。

可重复执行：已打过则直接退出。锚点找不到则报错，不改动文件。

用法
----
  python patches/patch_http_keepalive.py                 # 自动探测并打补丁
  python patches/patch_http_keepalive.py --check         # 只检查状态，不改
  python patches/patch_http_keepalive.py --revert        # 回退到官方值
  python patches/patch_http_keepalive.py --target <路径> # 指定目标文件
"""

import glob
import os
import sys

TARGET_REL = os.path.join("openviking", "models", "embedder", "openai_embedders.py")


def site_packages_rel_candidates() -> list[str]:
    """返回可能的 site-packages 相对路径候选，兼容 Windows 与 Linux 布局。"""
    if sys.platform == "win32":
        return [os.path.join("Lib", "site-packages")]
    version = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    return [
        os.path.join("lib", version, "site-packages"),
        os.path.join("lib", "python3.*", "site-packages"),
    ]


MARK = "keepalive_expiry=60.0"
ANCHOR = "keepalive_expiry=5.0,"
REPLACEMENT = "keepalive_expiry=60.0,"


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
            return os.path.join(
                package_dir, "models", "embedder", "openai_embedders.py"
            )

    script_dir = os.path.dirname(os.path.abspath(sys.executable))
    for base_dir in (script_dir, os.path.join(script_dir, "..")):
        candidate = _first_existing_target(base_dir)
        if candidate:
            return candidate
    return None


def read(path):
    return open(path, encoding="utf-8", newline="").read()


def status(target):
    if not os.path.isfile(target):
        return "missing", "目标文件不存在：%s" % target
    src = read(target)
    if MARK in src:
        return "patched", "保活时间已是 60 秒"
    if src.count(ANCHOR) == 1:
        return "patchable", "未打补丁，锚点齐备"
    return "drift", "锚点匹配 %d 处，上游可能已改动" % src.count(ANCHOR)


def apply_patch(target):
    src = read(target)
    if MARK in src:
        print("  已打过，跳过")
        return 0
    if src.count(ANCHOR) != 1:
        print("  锚点匹配 %d 处，拒绝改动文件" % src.count(ANCHOR))
        return 1
    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write(src.replace(ANCHOR, REPLACEMENT, 1))
    print("  已写入补丁：keepalive_expiry 5.0 → 60.0")
    print()
    print("  下一步：重启 OpenViking 服务让新连接生效")
    return 0


def revert(target):
    if not os.path.isfile(target):
        print("  目标文件不存在")
        return 1
    src = read(target)
    if MARK not in src:
        print("  未打过补丁，无需回退")
        return 0
    with open(target, "w", encoding="utf-8", newline="") as fh:
        fh.write(src.replace(REPLACEMENT, ANCHOR, 1))
    print("  已回退到官方值 5.0")
    print("  记得重启 OpenViking 服务")
    return 0


def parse_args(argv):
    """解析 --check / --revert / --target，避免 argparse 输出英文帮助。"""
    check_only = False
    revert_only = False
    explicit = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--check":
            check_only = True
        elif arg == "--revert":
            revert_only = True
        elif arg == "--target":
            if index + 1 >= len(argv):
                print("  --target 需要一个文件路径")
                return None, False, False, 1
            explicit = argv[index + 1]
            index += 1
        elif arg.startswith("--target="):
            explicit = arg.split("=", 1)[1]
        else:
            print("  未知参数: %s" % arg)
            return None, False, False, 1
        index += 1
    if check_only and revert_only:
        print("  --check 与 --revert 不能同时使用")
        return None, False, False, 1
    return explicit, check_only, revert_only, 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    explicit, check_only, revert_only, parse_status = parse_args(argv)
    if parse_status != 0:
        return parse_status

    target = resolve_target(explicit)
    if not target:
        print("  找不到 openai_embedders.py")
        print("  请设置 OPENVIKING_HOME 或用 --target 指定")
        return 1
    if not os.path.isfile(target):
        print("  目标文件不存在: %s" % target)
        print("  请设置 OPENVIKING_HOME 或用 --target 指定")
        return 1

    if check_only:
        st, msg = status(target)
        print("  http-keepalive: %s —— %s" % (st, msg))
        return 0 if st == "patched" else 2
    if revert_only:
        print("  目标：%s" % target)
        return revert(target)

    st, msg = status(target)
    print("=" * 76)
    print("OpenViking HTTP 嵌入客户端保活补丁")
    print("=" * 76)
    print("  目标：%s" % target)
    print("  状态：%s —— %s" % (st, msg))
    if st == "patched":
        return 0
    if st in ("missing", "drift"):
        print("\n  请核对 openai_embedders.py 的 _http_limits 方法。")
        return 1
    print()
    return apply_patch(target)


if __name__ == "__main__":
    sys.exit(main())
