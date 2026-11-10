"""划分泄漏与 schema 测试。不依赖真实视频、不依赖 pytest。

仓库根目录::

    python experiments/tests/test_split_leakage.py

若已安装 pytest，同一文件也可被收集。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.splits.code.split_checks import load_manifest, validate_manifest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _errors(name: str) -> list[str]:
    return validate_manifest(load_manifest(FIXTURES / name))


def test_ok_fixture_passes() -> None:
    assert _errors("tiny_split_ok.json") == []


def test_leak_fixture_reports_overlap() -> None:
    errors = _errors("tiny_split_leak.json")
    assert any("划分泄漏" in item and "fixture_shared_01" in item for item in errors)


def test_generated_frames_forbidden_in_test() -> None:
    errors = _errors("tiny_split_generated_in_test.json")
    assert any("generated" in item for item in errors)


def test_task_c_and_frame_unit_rejected() -> None:
    errors = _errors("tiny_split_task_c_frame.json")
    assert any("task=C" in item for item in errors)
    assert any("video_id 或 event_id" in item for item in errors)


def test_cli_ok_exit_zero() -> None:
    from experiments.splits.code.split_checks import main

    code = main([str(FIXTURES / "tiny_split_ok.json")])
    assert code == 0


def test_cli_leak_exit_one() -> None:
    from experiments.splits.code.split_checks import main

    code = main([str(FIXTURES / "tiny_split_leak.json")])
    assert code == 1


def _run_all() -> int:
    tests = [
        test_ok_fixture_passes,
        test_leak_fixture_reports_overlap,
        test_generated_frames_forbidden_in_test,
        test_task_c_and_frame_unit_rejected,
        test_cli_ok_exit_zero,
        test_cli_leak_exit_one,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            failed += 1
        else:
            print(f"OK {fn.__name__}")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run_all())
