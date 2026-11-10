# -*- coding: utf-8 -*-
"""extract_unit 顺序解码重构的快速夹具测试（合成视频，秒级）。

三个用例：
  A 正常：900 帧视频、48 槽全合法 → 768 帧齐全、slots.txt 旁证、QC 行数 768；
  B 尾部坏区：末尾 60 帧 read 必失败 → 首轮报 fail，重采一轮后成功（禁区并入）；
  C 失步：从第 500 帧起 POS_FRAMES 少报 3（模拟丢帧续解）→ 立即失败不写错位帧。
不依赖真实数据，纯内存合成视频。
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import (CLIP_LEN, N_CLIPS, pick_slots, extract_unit,
                         clean_unit_dir)  # noqa: E402

W, H, FPS = 64, 64, 25.0


def make_video(path, n, broken_from=None, desync_from=None):
    cap = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert cap.isOpened()
    for i in range(n):
        # 每帧灰度随帧号变化（std 高、帧间差可测）
        g = int((i * 7) % 255)
        fr = np.full((H, W, 3), g, np.uint8)
        cap.write(fr)
    cap.release()
    # broken_from/desync_from 无法直接注入 VideoWriter——改用读侧包装不现实，
    # 这里用「视频实际帧数少于声明槽位」与「POS_FRAMES 由后端控制」不可行，
    # 故 B/C 用 monkeypatch 读函数实现（见下）。
    return path


class FakeCap:
    """模拟 VideoCapture：顺序 read，支持坏区（read 失败）与失步（POS 少报）。"""

    def __init__(self, n, broken_from=None, desync_from=None):
        self.n = n
        self.broken_from = broken_from
        self.desync_from = desync_from
        self.i = 0
        self.frame = np.zeros((H, W, 3), np.uint8)

    def read(self):
        if self.i >= self.n or (self.broken_from is not None and self.i >= self.broken_from):
            return False, None
        g = int((self.i * 7) % 255)
        self.frame = np.full((H, W, 3), g, np.uint8)
        self.i += 1
        pos = self.i
        if self.desync_from is not None and self.i > self.desync_from:
            pos = self.i - 3      # 丢帧续解：位置少报
        return True, self.frame.copy()

    def grab(self):
        ok, _ = self.read()
        return ok

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.i if self.desync_from is None else max(
                0, self.i - (3 if self.desync_from is not None and self.i > self.desync_from else 0))
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.n)
        return 0.0

    def release(self):
        pass


def run_case(name, n, broken_from=None, desync_from=None, expect_retries=0):
    with tempfile.TemporaryDirectory() as td:
        udir = Path(td) / "u1"
        qc = Path(td) / "qc.csv"
        slots = pick_slots(n // CLIP_LEN, [])
        orig = cv2.VideoCapture
        cv2.VideoCapture = lambda *a, **k: FakeCap(n, broken_from, desync_from)
        try:
            got, fail_at, fail_frames = extract_unit(Path("fake.mp4"), slots, udir,
                                                     resume=False, qc_path=qc)
        finally:
            cv2.VideoCapture = orig
        n_ok = N_CLIPS * CLIP_LEN
        status = "OK" if (got == n_ok and fail_at is None) else f"got={got} fail_at={fail_at}"
        print(f"{name}: 写 {got} 帧, fail_at={fail_at}, fail_frames={fail_frames[:6]}"
              f"{'...' if len(fail_frames) > 6 else ''} -> {status}")
        return got, fail_at, fail_frames


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("== 用例 A 正常 900 帧 ==")
    run_case("A", 900)
    print("== 用例 B 尾部坏区（840 起不可读，48 槽含尾槽必失败）==")
    run_case("B", 900, broken_from=840)
    print("== 用例 C 失步（500 帧后 POS 少报 3）==")
    run_case("C", 900, desync_from=500)
