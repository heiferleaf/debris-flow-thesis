"""指标函数的最小夹具测试：在接入检测框架之前先锁住定义。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tools.metrics import box_iou


def test_identical_boxes_iou_one() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == 1.0


def test_disjoint_boxes_iou_zero() -> None:
    assert box_iou((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)) == 0.0


def test_half_overlap_iou() -> None:
    # 两个 2x2 的框在 x 方向重叠一半，交=2，并=6，IoU=1/3。
    left = (0.0, 0.0, 2.0, 2.0)
    right = (1.0, 0.0, 3.0, 2.0)
    value = box_iou(left, right)
    assert abs(value - (1.0 / 3.0)) < 1e-9


def _run_all() -> int:
    tests = [
        test_identical_boxes_iou_one,
        test_disjoint_boxes_iou_zero,
        test_half_overlap_iou,
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
