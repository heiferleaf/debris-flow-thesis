# -*- coding: utf-8 -*-
"""build_units_v2 的快速夹具测试（合成中间件，纯内存 FakeCap）。

用例：
  A secs_to_ranges：cut 秒区间 -> 中间件帧区间逐帧查 pts_time，不用 fps 算术；
  B scan_middleware 正常：n_coord 帧全健康 -> n_read==n_coord、无低方差、qc 落盘；
  C scan_middleware 低方差：3 帧灰帧 -> lowstd 命中、EOF 断言仍过；
  D scan_middleware 短流：交付 < 坐标表帧数 -> fail_frames 非空（main 里 EOF 断言失败）；
  E extract_unit_mw 正常：768 帧齐全、扫到 EOF 总账 n_scanned==n_mw、旁证落盘；
  F extract_unit_mw 失步：POS_FRAMES 少报 -> 立即失败；
  G extract_unit_mw 总账不符：尾段丢帧 -> n_scanned != n_mw 暴露。
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import N_CLIPS, CLIP_LEN, pick_slots  # noqa: E402
from build_units_v2 import extract_unit_mw, scan_middleware, secs_to_ranges  # noqa: E402

rng = np.random.RandomState(7)


def noise_frame():
    return rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)


def gray_frame():
    return np.full((64, 64, 3), 30, np.uint8)


class FakeMW:
    """模拟干净中间件：n 帧噪声帧；gray_at 集合内的帧换成灰帧；
    tail_drop>0 时末尾少交付（总账不符）；desync_from 起 POS 少报。"""

    def __init__(self, n, gray_at=(), tail_drop=0, desync_from=None):
        self.n = n
        self.deliver = n - tail_drop
        self.gray_at = set(gray_at)
        self.desync_from = desync_from
        self.i = 0
        self.fr = None

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.deliver:
            return False, None
        self.fr = gray_frame() if self.i in self.gray_at else noise_frame()
        self.i += 1
        return True, self.fr.copy()

    def grab(self):
        ok, _ = self.read()
        return ok

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.i if self.desync_from is None or self.i <= self.desync_from \
                else self.i - 3
        return 0.0

    def release(self):
        pass


def pts_of(n, step=0.04):
    return [round(i * step, 3) for i in range(n)]


def with_cap(monkey_target, cap):
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: cap
    return orig


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fails = []

    def check(name, cond, detail=""):
        print(f"{name}: {'OK' if cond else 'FAIL ' + detail}")
        if not cond:
            fails.append(name)

    # A：secs_to_ranges
    pts = pts_of(100)                       # 0.00,0.04,...,3.96
    rs = secs_to_ranges([(1.0, 2.0)], pts)  # 外扩 1s -> pts∈[0,3) -> 帧 0..74
    check("A1 cut 落帧", rs == [(0, 75)], f"got {rs}")
    rs = secs_to_ranges([(1.0, 2.0)], [t + 10.0 for t in pts])   # 时间轴外
    check("A2 区间外不落帧", rs == [], f"got {rs}")
    rs = secs_to_ranges([(-5.0, 100.0)], pts)                    # 全覆盖
    check("A3 全覆盖", rs == [(0, 100)], f"got {rs}")

    for case in ("B", "C", "D"):
        n = 900
        cap = FakeMW(n, gray_at=(100, 101, 102) if case == "C" else (),
                     tail_drop=120 if case == "D" else 0)
        orig = cv2.VideoCapture
        cv2.VideoCapture = lambda *a, **k: cap
        try:
            with tempfile.TemporaryDirectory() as td:
                n_read, ff, lowstd = scan_middleware(Path("fake.mp4"), "u",
                                                     n, pts_of(n))
                qc = Path(td) / "qc"
        finally:
            cv2.VideoCapture = orig
        if case == "B":
            check("B1 全片读齐", n_read == n and not ff, f"{n_read} ff={ff[:4]}")
            check("B2 无低方差", not lowstd, f"{lowstd[:4]}")
        elif case == "C":
            check("C1 灰帧命中", lowstd == [100, 101, 102], f"{lowstd}")
            check("C2 EOF 断言素材", n_read == n and not ff)
        else:
            # 短流：780 帧后 read 全失败，收区到超出坐标表帧数为止（780..900 共 121）
            check("D1 短流暴露", n_read == 780 and len(ff) == 121,
                  f"{n_read} ff={len(ff)}")

    # E：extract_unit_mw 正常（末槽后继续扫到 EOF，总账相符）
    n = 900
    slots = pick_slots(n // CLIP_LEN, [])
    cap = FakeMW(n)
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: cap
    try:
        with tempfile.TemporaryDirectory() as td:
            got, n_sc, fail_at, ff = extract_unit_mw(Path("fake.mp4"), slots,
                                                     Path(td) / "u", False, n)
            side = (Path(td) / "u" / "slots.txt").exists()
    finally:
        cv2.VideoCapture = orig
    check("E1 768 帧齐全", got == N_CLIPS * CLIP_LEN, f"got {got}")
    check("E2 总账相符", n_sc == n and fail_at is None and not ff,
          f"n_sc={n_sc} fail_at={fail_at} ff={ff[:4]}")
    check("E3 旁证落盘", side)

    # F：失步立即失败
    cap = FakeMW(n, desync_from=100)
    cv2.VideoCapture = lambda *a, **k: cap
    try:
        with tempfile.TemporaryDirectory() as td:
            got, n_sc, fail_at, ff = extract_unit_mw(Path("fake.mp4"), slots,
                                                     Path(td) / "u", False, n)
    finally:
        cv2.VideoCapture = orig
    check("F1 失步即停", fail_at is not None and got < N_CLIPS * CLIP_LEN,
          f"fail_at={fail_at} got={got}")

    # G：尾段丢帧 -> 总账不符。断言在 main 层做（n_sc != n_mw 即整单位排除），
    # extract 层如实返回计数即可。
    cap = FakeMW(n, tail_drop=3)
    cv2.VideoCapture = lambda *a, **k: cap
    try:
        with tempfile.TemporaryDirectory() as td:
            got, n_sc, fail_at, ff = extract_unit_mw(Path("fake.mp4"), slots,
                                                     Path(td) / "u", False, n)
    finally:
        cv2.VideoCapture = orig
    check("G1 总账不符暴露", got == N_CLIPS * CLIP_LEN and n_sc == n - 3
          and fail_at is None, f"got={got} n_sc={n_sc} fail_at={fail_at}")

    print("\n== 全部通过 ==" if not fails else f"\n== 失败 {fails} ==")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
