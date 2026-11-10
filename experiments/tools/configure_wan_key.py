"""Save a Beijing DashScope API key with current-user Windows DPAPI, without HTTP.

Run interactively in Windows PowerShell. Input is hidden and never accepted as
a command-line argument. This tool uses a separate key file from Runway.
--self-test only writes and decrypts non-secret fixtures in a temporary folder.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sys
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.synthetic_wan import WAN_KEY_PATH, store_wan_secret
from experiments.common.video.local_credentials import load_runway_secret


def self_test():
    with tempfile.TemporaryDirectory(prefix="wan-dpapi-dummy-") as directory:
        path = Path(directory) / "dummy.dpapi"
        dummy = "non-secret-wan-DPAPI-fixture-测试"
        store_wan_secret(dummy, path)
        encrypted = path.read_bytes()
        if dummy.encode("utf-8") in encrypted or load_runway_secret(path) != dummy:
            raise RuntimeError("DPAPI dummy roundtrip failed")
        try:
            store_wan_secret("replacement-dummy", path)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("Existing credential overwrite protection failed")
        if encrypted != path.read_bytes():
            raise RuntimeError("Existing dummy was changed")
        store_wan_secret("replacement-dummy", path, replace=True)
        if load_runway_secret(path) != "replacement-dummy":
            raise RuntimeError("Explicit dummy replacement failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--replace", action="store_true")
    choices.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if os.name != "nt":
            raise OSError("请在Windows PowerShell中运行此命令。")
        if args.self_test:
            self_test()
            print("DPAPI临时dummy自测通过；未读取真实密钥，未联网。")
            return 0
        if WAN_KEY_PATH.exists() and not args.replace:
            raise FileExistsError("已存在百炼密钥；确认替换时显式追加--replace。")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            secret = getpass.getpass("北京地域 DASHSCOPE_API_KEY（输入不回显）: ")
        saved = store_wan_secret(secret, replace=args.replace)
        del secret
        print("密钥已加密保存：" + str(saved))
        return 0
    except (OSError, ValueError, RuntimeError, getpass.GetPassWarning) as exc:
        print("配置未完成：" + str(exc), file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("配置已取消。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
