# -*- coding: utf-8 -*-
"""按 extneg-scenes.csv 的审计分段,把外部视频切成场景帧集(连续帧训练的划分原子)。

口径(叶海峰 2026-09-15):
- 分段 = data/process/video_frames/video2_segment/extneg-scenes.csv(审计台导出,2026-09-16 目录重构
  后归 video2 初帧同目录),标签逐段保持审计结果;
- CSV 里没有的外部视频(小溪/山洪/水库,审计时无已确认区间)= 整段一个隐式负段(未标默认负);
- kind=pos/neg 为训练素材,kind=cut 只切不训(剔除段,目录名带 cut,训练清单排除);
- 帧少的段自动加密抽帧:目标 >= --min-frames 帧/段(时序模型常见片段长 16-32),间隔下限 0.25s;
  帧够的段保持基础间隔(默认 2s,与审计视图一致)。

输出:data/final/video_windows/video2_segment/<video_id>/seg<NN>_<kind>/frame_%06d_t<t>s.jpg +
     frames_index.csv(frame_file,t_sec)(最终场景帧,定稿进 final)
汇总清单:data/process/video_frames/video2_segment/extneg-scene-frames.csv(video_id,seg_id,kind,start_sec,end_sec,interval,n_frames,dir)

用法(仓库根,venv): python experiments/tools/cut_extneg_scenes.py [--min-frames 30] [--interval 2.0] [--overwrite]
依据: 审计分段 extneg-scenes.csv + 数据清单 B 节场景分段口径;原始视频不动,可重跑再生。
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
# 2026-09-16 data 结构重构:分段标记随初帧同放 process/video_frames/video2_segment/;
# 最终场景帧(定稿)进 final/video_windows/video2_segment/。旧 data/frames-scenes/
# 与 data/processed/label-review/extneg-scene*.csv 已并入上述新位置。
SCENES_CSV = REPO / "data/process/video_frames/video2_segment/extneg-scenes.csv"
RAW = Path(r"D:/debris-flow-data/webcam-negatives/raw")
OUT = REPO / "data/final/video_windows/video2_segment"
SUMMARY = REPO / "data/process/video_frames/video2_segment/extneg-scene-frames.csv"
MIN_INTERVAL = 0.25


def read_scenes():
    rows = []
    with open(SCENES_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "video_id": r["video_id"], "seg_id": int(r["seg_id"]),
                "start": float(r["start_sec"]), "end": float(r["end_sec"]),
                "kind": r["kind"].strip(),
            })
    return rows


def seg_interval(dur, base, min_frames):
    """帧数够用保持基础间隔;不够则加密到目标帧数(间隔下限 0.25s)。"""
    if dur / base >= min_frames:
        return base
    return max(MIN_INTERVAL, dur / min_frames)


def extract_segment(cap, vdir, t0, t1, interval, overwrite):
    idx = vdir / "frames_index.csv"
    if idx.is_file() and not overwrite:
        with open(idx, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        return n, True  # 已有,跳过
    vdir.mkdir(parents=True, exist_ok=True)
    ts = np.arange(t0, t1 - 1e-6, interval)
    rows = []
    for i, t in enumerate(ts, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        name = f"frame_{i:06d}_t{t:.1f}s.jpg"
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            break
        buf.tofile(str(vdir / name))  # 中文路径安全写
        rows.append((name, round(float(t), 2)))
    with open(idx, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_file", "t_sec"])
        w.writerows(rows)
    return len(rows), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scenes = read_scenes()
    by_video = {}
    for s in scenes:
        by_video.setdefault(s["video_id"], []).append(s)

    # 未出现在 CSV 的外部视频 = 整段隐式负段(未标默认负口径)
    for mp4 in sorted(RAW.glob("*.mp4")):
        if mp4.stem not in by_video:
            cap = cv2.VideoCapture(str(mp4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
            cap.release()
            by_video[mp4.stem] = [{"video_id": mp4.stem, "seg_id": 1,
                                   "start": 0.0, "end": round(dur, 2), "kind": "neg"}]
            print(f"[隐式负段] {mp4.stem}: 无审计区间,整段 0-{dur:.1f}s 记 neg")

    summary, caps = [], {}
    try:
        for vid in sorted(by_video):
            mp4 = RAW / f"{vid}.mp4"
            if not mp4.is_file():
                print(f"[跳过] {vid}: raw 里找不到视频文件")
                continue
            cap = caps.setdefault(vid, cv2.VideoCapture(str(mp4)))
            for s in sorted(by_video[vid], key=lambda x: x["seg_id"]):
                dur = s["end"] - s["start"]
                itv = seg_interval(dur, args.interval, args.min_frames)
                vdir = OUT / vid / f"seg{s['seg_id']:02d}_{s['kind']}"
                n, skipped = extract_segment(cap, vdir, s["start"], s["end"], itv, args.overwrite)
                summary.append([vid, s["seg_id"], s["kind"], s["start"], s["end"],
                                round(itv, 2), n, str(vdir.relative_to(REPO))])
                mark = "跳过(已有)" if skipped else ""
                print(f"{vid} seg{s['seg_id']:02d} {s['kind']:3s} {s['start']:7.1f}-{s['end']:7.1f}s "
                      f"间隔{s['end'] - s['start']:6.1f}s/{itv:4.2f}s -> {n:4d}帧 {mark}")
    finally:
        for c in caps.values():
            c.release()

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "seg_id", "kind", "start_sec", "end_sec",
                    "interval_s", "n_frames", "dir"])
        w.writerows(summary)

    n_pos = sum(r[6] for r in summary if r[2] == "pos")
    n_neg = sum(r[6] for r in summary if r[2] == "neg")
    n_cut = sum(r[6] for r in summary if r[2] == "cut")
    n_seg = len(summary)
    print(f"\n共 {n_seg} 段 -> pos {n_pos} 帧 / neg {n_neg} 帧 / cut(只切不训) {n_cut} 帧")
    print(f"汇总: {SUMMARY}")


if __name__ == "__main__":
    main()
