# -*- coding: utf-8 -*-
"""对中间件 mp4 逐帧算灰度标准差，产出抽帧用的逐帧质检表。

为什么要这个工具：抽帧脚本靠「低灰度帧前后各 16 帧」划禁区，这份判据来自逐帧质检表。
中间件一旦重新转过（2026-10-26 重转 19 条软解中间件），当年的质检表就不再保证与实际
像素对齐——解码器两次跑同一份原件，帧数可能相同而个别帧内容不同。所以重建过的中间件
必须重算一份表，抽帧时用新表。

输出列与 `EXP-008-units-v2/qc/<unit_id>.csv` 一致（frame, pts_time, gray_std），抽帧脚本
可以直接当旧表用。一处差别：本工具的 gray_std 保留 2 位小数，旧表保留 1 位，所以两张表
逐格比较时差值最多 0.05，这是四舍五入，不是内容不同。

两条阈值沿用全仓库同一套：低于 20 记低灰度（抽帧禁区线），低于 5 记整片均匀灰。这两个数
从抽帧脚本导入（`build_units.QC_STD_LOW`），不在这里另写一份，免得哪天改了一处另一处没跟上。

用法（仓库根）：
  .venv/Scripts/python.exe experiments/tools/scan_middleware_gray.py \
      --units v1_19,v1_39 --src data/process/EXP-008-units-src256 \
      --out data/process/EXP-008-units-src256/qc

只读中间件，除 --out 目录外不改任何文件。汇总落 <out>/../中间件灰度汇总.csv。
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/common/video"))

from build_units import QC_STD_LOW  # noqa: E402

STD_LOW = float(QC_STD_LOW)
STD_HARD = 5.0      # 整片均匀灰：抽帧脚本另有一条同值判据，见 build_units 的灰度部分


def scan_one(uid, src, out_dir):
    mp4 = Path(src) / f"{uid}.mp4"
    coord = Path(src) / "coord" / f"{uid}.csv"
    if not mp4.exists() or not coord.exists():
        return {"unit_id": uid, "error": "缺中间件或坐标表"}
    pts = [r["pts_time"] for r in csv.DictReader(open(coord, encoding="utf-8-sig"))]
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return {"unit_id": uid, "error": "中间件打不开"}
    stds = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        stds.append(round(float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).std()), 2))
    cap.release()
    if len(pts) != len(stds):
        return {"unit_id": uid, "error": f"坐标 {len(pts)} 帧 / 解码 {len(stds)} 帧 不等"}

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{uid}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "pts_time", "gray_std"])
        for i, (t, s) in enumerate(zip(pts, stds)):
            w.writerow([i, t, s])

    arr = np.array(stds)
    return {"unit_id": uid, "n_frames": len(stds),
            "low_lt20": int((arr < STD_LOW).sum()),
            "hard_lt5": int((arr < STD_HARD).sum()),
            "low_pct": round(float((arr < STD_LOW).mean()), 4),
            "hard_pct": round(float((arr < STD_HARD).mean()), 4),
            "min_std": round(float(arr.min()), 2),
            "median_std": round(float(np.median(arr)), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--units", required=True, help="逗号分隔的 unit_id")
    ap.add_argument("--src", default="data/process/EXP-008-units-src256")
    ap.add_argument("--out", default="data/process/EXP-008-units-src256/qc")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ids = [s.strip() for s in args.units.split(",") if s.strip()]
    print(f"[gray] 待扫 {len(ids)} 条中间件 -> {args.out}", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_one, u, args.src, args.out) for u in ids]
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            if "error" in r:
                print(f"  ! {r['unit_id']} {r['error']}", flush=True)
            else:
                print(f"  + {r['unit_id']} {r['n_frames']} 帧 低灰度 {r['low_pct']:.2%} "
                      f"整片均匀灰 {r['hard_pct']:.2%} 中位 {r['median_std']}", flush=True)

    rows.sort(key=lambda r: r["unit_id"])
    summary = Path(args.out).parent / "中间件灰度汇总.csv"
    cols = ["unit_id", "n_frames", "low_lt20", "hard_lt5", "low_pct", "hard_pct",
            "min_std", "median_std", "error"]
    with open(summary, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    good = [r for r in rows if "error" not in r]
    print(f"[gray] {len(good)}/{len(ids)} 条算完，用时 {time.time() - t0:.0f} 秒 -> {summary}")


if __name__ == "__main__":
    main()
