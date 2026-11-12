# -*- coding: utf-8 -*-
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

可重复执行：已打过则直接退出。锚点找不到则报错不盲改。
"""
import io
import os
import sys

TARGET = os.path.join(
    os.environ.get("OPENVIKING_HOME", r"D:\OpenViking"),
    r"venv\Lib\site-packages\openviking\models\embedder\openai_embedders.py",
)

MARK = "keepalive_expiry=60.0"
ANCHOR = "keepalive_expiry=5.0,"
REPLACEMENT = "keepalive_expiry=60.0,"


def read(path):
    return io.open(path, encoding="utf-8", newline="").read()


def status():
    if not os.path.exists(TARGET):
        return "missing", "目标文件不存在：%s" % TARGET
    src = read(TARGET)
    if MARK in src:
        return "patched", "保活时间已是 60 秒"
    if src.count(ANCHOR) == 1:
        return "patchable", "未打补丁，锚点齐备"
    return "drift", "锚点匹配 %d 次，上游可能已改动" % src.count(ANCHOR)


def apply_patch():
    src = read(TARGET)
    if MARK in src:
        print("  已打过，跳过")
        return 0
    if src.count(ANCHOR) != 1:
        print("  ✗ 锚点匹配 %d 次，拒绝盲改" % src.count(ANCHOR))
        return 1
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src.replace(ANCHOR, REPLACEMENT, 1))
    print("  已写入补丁：keepalive_expiry 5.0 → 60.0")
    print()
    print("  下一步：重启 OpenViking 服务让新连接生效")
    return 0


def revert():
    if not os.path.exists(TARGET):
        print("  ✗ 目标文件不存在")
        return 1
    src = read(TARGET)
    if MARK not in src:
        print("  未打过补丁，无需回退")
        return 0
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src.replace(REPLACEMENT, ANCHOR, 1))
    print("  已回退到官方值 5.0")
    print("  记得重启 OpenViking 服务")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        st, msg = status()
        print("  http-keepalive: %s —— %s" % (st, msg))
        return 0 if st == "patched" else 2
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        return revert()

    st, msg = status()
    print("=" * 76)
    print("OpenViking HTTP 嵌入客户端保活补丁")
    print("=" * 76)
    print("  目标：%s" % TARGET)
    print("  状态：%s —— %s" % (st, msg))
    if st == "patched":
        return 0
    if st in ("missing", "drift"):
        print("\n  请核对 openai_embedders.py 的 _http_limits 方法。")
        return 1
    print()
    return apply_patch()


if __name__ == "__main__":
    sys.exit(main())
