# -*- coding: utf-8 -*-
"""扫描一套训练帧，逐张算灰度标准差，统计有多少张接近纯灰。

用途：抽帧时虽然已经把「低灰度帧前后各 16 帧」划进不能取的区间，但那道防护依赖
抽帧脚本自己算的帧号；这里直接对落盘后的 jpg 复算一遍，用结果回答「训练真正吃进去
的图里有没有灰图」。2026-10-25 用它核对新间隔那套帧（v3）是否也全部避开了灰帧。

为什么要单独一个工具：原扫描脚本（`.codex/docs/.temp/scan_training_frames.py`）把路径写死
在 v2 上，改一处就得改代码。这里把帧根与输出路径都做成参数，以后每建一套新帧都能直接复用。

灰度标准差怎么算：cv2 按灰度读入（BGR 三通道先转灰度），取像素值的标准差。
画面越接近一片纯色，标准差越小。本仓库沿用两条线：低于 20 记为低灰度（抽帧禁区线），
低于 5 记为整片均匀灰。

用法（仓库根）：
  .venv/Scripts/python.exe experiments/tools/scan_frame_gray.py \
      --frames-root data/process/EXP-008-units-v3-gap05/frames \
      --out data/process/EXP-008-units-v3-gap05/训练帧灰度.csv

只读帧，不改任何文件；输出一行一个 jpg：
  unit_id, clip_idx, frame_idx, gray_std
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

CLIP_LEN = 16
N_CLIPS = 48


def read_gray_std(path: Path) -> float:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None
    if img is None:
        return float("nan")
    return float(img.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames-root", required=True, type=Path,
                    help="含 <视频编号>/clipXX_fYY.jpg 的目录")
    ap.add_argument("--out", type=Path, default=None,
                    help="逐张结果的 csv；不给则只打印汇总")
    ap.add_argument("--clip-len", type=int, default=CLIP_LEN)
    ap.add_argument("--n-clips", type=int, default=N_CLIPS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    units = sorted(p.name for p in a.frames_root.iterdir() if p.is_dir())
    if not units:
        raise SystemExit(f"这个目录下没有视频子目录：{a.frames_root}")

    rows, per_unit = [], []
    for n, uid in enumerate(units, 1):
        vals = []
        for ci in range(a.n_clips):
            for fi in range(a.clip_len):
                p = a.frames_root / uid / f"clip{ci:02d}_f{fi:02d}.jpg"
                s = read_gray_std(p) if p.is_file() else float("nan")
                rows.append((uid, ci, fi, s))
                vals.append(s)
        v = np.array(vals, dtype=float)
        ok = np.isfinite(v)
        per_unit.append((uid, ok.sum(), int((v < 20).sum()), int((v < 5).sum()),
                         float(np.nanmin(v)) if ok.any() else float("nan")))
        if n % 20 == 0 or n == len(units):
            print(f"[{n}/{len(units)}] {uid} 最低 {per_unit[-1][4]:.2f}", flush=True)

    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "clip_idx", "frame_idx", "gray_std"])
            for uid, ci, fi, s in rows:
                w.writerow([uid, ci, fi, "" if s != s else f"{s:.2f}"])
        print(f"写到 {a.out}")

    v = np.array([r[3] for r in rows], dtype=float)
    ok = np.isfinite(v)
    print(f"\n视频 {len(units)} 条，帧 {len(v)} 张（读到 {int(ok.sum())} 张）")
    print(f"灰度标准差 < 20 的帧：{int((v < 20).sum())} 张")
    print(f"灰度标准差 < 5（整片均匀灰）的帧：{int((v < 5).sum())} 张")
    if ok.any():
        print(f"最低 {np.nanmin(v):.2f}，中位 {np.nanmedian(v):.2f}")
    bad = [u for u, _, n20, _, _ in per_unit if n20]
    print(f"含低灰度帧的视频：{len(bad)} 条" + (f"（{bad[:10]}）" if bad else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
