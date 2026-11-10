"""外部负样本视频拆帧:把一个装 mp4 的文件夹拆成标注工具可用的 frames 目录。

背景(叶海峰 2026-09-15 定):叶海峰自找的负样本视频(含多场景拼接)统一放一个
文件夹、整批先当负样本;拆帧后在审计台逐场景标 neg 区间、点「导出场景分段」,
段(extneg-scenes.csv)= 外部负样本的划分原子。

输出 <out>/<视频名>/frames/frame_%06d_t{秒}s.jpg + frames_index.csv(frame_file,t_sec),
与东川 frames(data/frames)分根,审计台用 --dir 指过去即可。只读原视频,不改动。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\extract_frames_external.py
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\extract_frames_external.py \\
        --src data/raw/extneg-videos --out data/frames-extneg --interval 2

归档(2026-09-16 data 结构重构):外部视频初帧 data/frames-extneg → data/process/video_frames/
video2_segment/。功能与 batch_extract_frames.py 重叠,后续新视频拆帧直接用长期工具
extract_frames.py/batch_extract_frames.py 并 --out 指新目录即可,本脚本保留作当时外部拆帧存档。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".ts"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="外部负样本视频拆帧(输出标注工具可用的 frames 目录)")
    p.add_argument("--src", default="data/raw/extneg-videos", help="装视频的文件夹(默认 data/raw/extneg-videos)")
    p.add_argument("--out", default="data/frames-extneg", help="拆帧输出根目录(默认 data/frames-extneg)")
    p.add_argument("--interval", type=float, default=2.0, help="抽帧间隔秒(默认 2,与东川拆帧一致)")
    p.add_argument("--overwrite", action="store_true", help="已拆过的视频也重拆(缺省跳过)")
    return p.parse_args()


def extract_one(video: Path, out_root: Path, interval: float) -> tuple[int, float]:
    """拆单个视频,返回 (帧数, 时长秒)。帧名与 frames_index.csv 同东川格式。"""
    vd = out_root / video.stem
    frames_dir = vd / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频:{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n_src / fps if fps else 0.0
    ts, files = [], []
    i, t = 0, 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok:
            break
        fn = f"frame_{i:06d}_t{round(t)}s.jpg"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(frames_dir / fn))
        files.append(fn)
        ts.append(round(t, 2))
        i += 1
        t += interval
    cap.release()
    with (vd / "frames_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_file", "t_sec"])
        for fn, t in zip(files, ts):
            w.writerow([fn, t])
    return len(files), dur


def main() -> None:
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"源文件夹不存在:{src}(把叶海峰找来的负样本视频放这里)")
    vids = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        raise SystemExit(f"{src} 下没有视频文件(支持 {'/'.join(sorted(VIDEO_EXTS))})")
    print(f"待拆 {len(vids)} 个视频 -> {out}(每 {args.interval} 秒一帧)", flush=True)
    for v in vids:
        if not args.overwrite and (out / v.stem / "frames_index.csv").is_file():
            print(f"  [跳过] {v.name} 已拆过(--overwrite 重拆)", flush=True)
            continue
        n, dur = extract_one(v, out, args.interval)
        print(f"  {v.name}: {n} 帧 / {dur:.0f}s", flush=True)
    print(f"[完成] 审计台打开:.venv\\Scripts\\python.exe experiments/tools/event_annotator_server.py --dir {args.out}")


if __name__ == "__main__":
    main()
