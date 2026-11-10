"""Interactively save a Runway key using current-user Windows DPAPI, without HTTP.

Run this yourself in Windows PowerShell. Input is hidden, never echoed or
accepted as a command-line argument. Existing credentials require --replace.
--self-test only uses a system temporary directory and a non-secret dummy.
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
from experiments.common.video.local_credentials import RUNWAY_KEY_PATH, load_runway_secret, store_runway_secret


def self_test() -> None:
    # Never read the default credential path in this test.
    dummy = "non-secret-DPAPI-roundtrip-fixture-测试"
    with tempfile.TemporaryDirectory(prefix="runway-dpapi-selftest-") as directory:
        path = Path(directory) / "dummy.dpapi"
        store_runway_secret(dummy, path)
        encrypted = path.read_bytes()
        if dummy.encode("utf-8") in encrypted or load_runway_secret(path) != dummy:
            raise RuntimeError("DPAPI dummy roundtrip failed")
        try:
            store_runway_secret("different-dummy", path)
        except FileExistsError:
            pass
        else:
            raise RuntimeError("Existing credential overwrite protection failed")
        if path.read_bytes() != encrypted:
            raise RuntimeError("Existing dummy credential was modified")
        store_runway_secret("replacement-dummy", path, replace=True)
        if load_runway_secret(path) != "replacement-dummy":
            raise RuntimeError("Explicit dummy replacement failed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--replace", action="store_true", help="Explicitly replace the existing encrypted key")
    options.add_argument("--self-test", action="store_true", help="Test DPAPI with a temporary dummy only")
    args = parser.parse_args(argv)
    try:
        if os.name != "nt":
            raise OSError("请在 Windows PowerShell 中运行此命令。")
        if args.self_test:
            self_test()
            print("DPAPI 临时 dummy 自测通过。")
            return 0
        if RUNWAY_KEY_PATH.exists() and not args.replace:
            raise FileExistsError("已存在密钥；仅在确认替换时追加 --replace。")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            secret = getpass.getpass("Runway API key（输入不回显）: ")
        saved = store_runway_secret(secret, replace=args.replace)
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
