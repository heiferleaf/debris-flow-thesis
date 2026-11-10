"""灰屏噪点帧预检:找出"灰色噪点占据大部分画面"的帧,连成候选剔除区间。

背景:2026-09-15 叶海峰确认口径——未标注视频里灰屏噪点占大部分画面的帧剔除
(信号/编码劣化,无判别信息;留作负样本会让模型学"灰屏=负"的捷径)。
**按帧剔除,不按区间**(2026-09-15 叶海峰纠正:区间合并会把散布灰屏并成接近
全片的段、连坐中间的好帧)。--apply 写帧级清单
data/processed/label-review/gray-noise-frames-2026-09-15.csv
(video_id, frame_file, t_sec, edge_ratio),生成训练数据时按帧过滤,
events.csv / events_proposed.csv 均不动。

判据(每帧缩放到宽 320 后算三个量):
  edge_ratio  Sobel 梯度幅度>15 的像素占比——灰屏帧几乎全平坦,此值极低;
  sat_frac    HSV 饱和度>=30 的像素占比——灰屏帧接近无色;
  mud_frac    泥色像素占比(色相 3-40、饱和度>=40、亮度>=40,同 preannotate_events)——
              防止把真实的灰褐泥石流帧误剔(泥流帧此值明显更高)。
灰屏帧 = edge_ratio < --max-edge 且 sat_frac < --max-sat 且 mud_frac < --max-mud。
连续灰屏帧连成区间,间隙 <= --merge-gap 合并,短于 --min-dur 丢弃,首尾外扩一帧。

已标注(含 real 区间)的视频整体跳过——叶海峰口径只针对未标注视频。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\detect_gray_noise_frames.py            # 只统计
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\detect_gray_noise_frames.py --apply    # 写入候选

归档(2026-09-16 data 结构重构):灰屏质控两轮已完结(共删 2,593 帧),一次性不再重跑。
脚本路径引用为旧结构——东川初帧 data/frames → data/process/video_frames/video1_segment/;
帧级清单 data/processed/label-review/ → process/ 下对应位置。若要重跑先按新路径改 --root/--out。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

EVENTS_FIELDS = ["video_id", "start_sec", "end_sec", "confidence", "kind", "note"]
REPO = Path(__file__).resolve().parents[3]   # 仓库根(oneoff 上三级)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="灰屏噪点帧预检(候选写入 events_proposed.csv,人工确认)")
    p.add_argument("--root", default="data/frames", help="frames 根目录(默认 data/frames)")
    p.add_argument("--apply", action="store_true", help="实际写入 events_proposed.csv(缺省只统计)")
    p.add_argument("--max-edge", type=float, default=0.035, help="edge_ratio 阈值(默认 0.035)")
    p.add_argument("--max-sat", type=float, default=0.12, help="sat_frac 阈值(默认 0.12)")
    p.add_argument("--max-mud", type=float, default=0.10, help="mud_frac 阈值(默认 0.10)")
    p.add_argument("--merge-gap", type=float, default=10.0, help="相邻区间间隙合并阈值秒(默认 10)")
    p.add_argument("--min-dur", type=float, default=6.0, help="最短候选区间秒(默认 6)")
    p.add_argument("--samples", type=int, default=10, help="每视频抽取的灰屏帧样例数(默认 10)")
    p.add_argument("--out", default="gray-noise-frames-2026-09-15.csv",
                   help="帧级清单文件名(写在 data/processed/label-review/ 下;第二轮放宽阈值用新文件名,不覆盖前轮存档)")
    return p.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def frame_metrics(img: np.ndarray) -> tuple[float, float, float]:
    """(edge_ratio, sat_frac, mud_frac),img 会原地缩放。"""
    h, w = img.shape[:2]
    if w > 320:
        img = cv2.resize(img, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    mag = np.abs(gx).astype(np.int32) + np.abs(gy).astype(np.int32)   # SAD 近似,免开方
    edge_ratio = float((mag > 60).mean())          # ≈ 梯度幅度>15/255(15*4)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_frac = float((hsv[..., 1] >= 30).mean())
    mud_frac = float(((hsv[..., 0] >= 3) & (hsv[..., 0] <= 40)
                      & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)).mean())
    return edge_ratio, sat_frac, mud_frac


def process_video(task: tuple) -> dict:
    """单视频:逐帧算指标→标灰屏→连区间。返回统计与样例帧路径。"""
    root, name, a = task
    d = Path(root) / name
    frames_dir = d / "frames"
    ts, files = [], []
    with (d / "frames_index.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            files.append(row["frame_file"])
            ts.append(float(row["t_sec"]))
    flags = np.zeros(len(files), dtype=bool)
    em = np.zeros(len(files), dtype=np.float32)
    for i, fn in enumerate(files):
        img = imread_unicode(frames_dir / fn)
        if img is None:
            continue                     # 读不出的帧不当灰屏(已有缺图登记,不在此处理)
        e, s, m = frame_metrics(img)
        em[i] = e
        flags[i] = e < a["max_edge"] and s < a["max_sat"] and m < a["max_mud"]
    gray_frames = [(files[i], float(em[i])) for i in range(len(files)) if flags[i]]
    # 连续灰屏帧 → 区间(首尾外扩一帧),间隙合并,短区间丢弃
    raw: list[list[float]] = []
    i = 0
    step = (ts[1] - ts[0]) if len(ts) > 1 else 2.0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        raw.append([max(ts[0], ts[i] - step), min(ts[-1], ts[j] + step)])
        i = j + 1
    merged: list[list[float]] = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] <= a["merge_gap"]:
            merged[-1][1] = max(merged[-1][1], seg[1])
        else:
            merged.append(seg)
    intervals = [s for s in merged if s[1] - s[0] >= a["min_dur"]]
    # 样例帧:从各区间均匀抽
    samples: list[str] = []
    per = max(1, a["samples"] // max(1, len(intervals))) if intervals else 0
    for s, e in intervals:
        in_seg = [k for k in range(len(ts)) if s <= ts[k] <= e and flags[k]]
        if in_seg:
            pick = in_seg[:: max(1, len(in_seg) // per)][:per]
            samples += [files[k] for k in pick]
    return {"name": name, "n_frames": len(files), "n_gray": int(flags.sum()),
            "gray_ratio": round(float(flags.mean()), 4) if len(flags) else 0.0,
            "intervals": [[round(s, 1), round(e, 1)] for s, e in intervals],
            "edge_median": float(np.median(em)) if len(em) else 0.0,
            "gray_frames": gray_frames,
            "ts_map": {f: round(t, 1) for f, t in zip(files, ts)},
            "samples": samples}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    a = {"max_edge": args.max_edge, "max_sat": args.max_sat, "max_mud": args.max_mud,
         "merge_gap": args.merge_gap, "min_dur": args.min_dur, "samples": args.samples}
    tasks, skipped = [], []
    for d in sorted(root.iterdir()):
        if not (d.is_dir() and (d / "frames_index.csv").is_file()):
            continue
        ev = d / "events.csv"
        has_real = False
        if ev.is_file():
            with ev.open(encoding="utf-8-sig", newline="") as f:
                has_real = any(r.get("confidence") == "real" for r in csv.DictReader(f))
        (skipped if has_real else tasks).append(d.name)
    print(f"待检 {len(tasks)} 个视频;已标注跳过 {len(skipped)} 个", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(process_video, [(str(root), n, a) for n in tasks]):
            results.append(res)
            print(f"  {res['name'][:50]}:灰屏 {res['n_gray']}/{res['n_frames']}"
                  f"({res['gray_ratio']:.0%}),候选区间 {len(res['intervals'])}", flush=True)

    if args.apply:
        # 2026-09-15 叶海峰纠正:按帧剔除,不按区间——区间合并会把散布灰屏并成
        # 接近全片的段、连坐中间的好帧。改为写帧级清单,生成数据时按帧过滤。
        out_csv = REPO / "data/processed/label-review" / args.out
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "frame_file", "t_sec", "edge_ratio"])
            for res in results:
                for fn, e in res["gray_frames"]:
                    w.writerow([res["name"], fn, res["ts_map"][fn], round(e, 4)])
                    n += 1
        print(f"[写入] 灰屏帧清单 {n} 行 -> {out_csv}(生成数据时按帧过滤,不动 events.csv)")
    else:
        print("[只统计] 未写入;加 --apply 落盘")

    out = {"params": {k.replace("_", "-"): v for k, v in a.items()},
           "skipped": skipped,
           "videos": [{k: v for k, v in r.items() if k not in ("samples", "gray_frames", "ts_map")}
                      for r in results],
           "total_gray": sum(r["n_gray"] for r in results),
           "total_intervals": sum(len(r["intervals"]) for r in results),
           "samples": {r["name"]: r["samples"] for r in results if r["samples"]}}
    # 报告落 .temp/(一次性中间件,不脏工作区根;2026-09-16 叶海峰定目录精简)
    out_path = REPO / ".temp" / "检出中间件" / "_gray_detect_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"合计灰屏帧 {out['total_gray']},候选区间 {out['total_intervals']} 条;报告 {out_path}")


if __name__ == "__main__":
    main()
