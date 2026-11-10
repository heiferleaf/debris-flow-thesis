"""把监控视频拆成等间隔帧,并生成方便人工标注事件区间的材料。

产出(写到 <out>/<视频名>/):
  frames/      frame_%06d.jpg,按 --interval 秒等间隔抽帧,文件名含时间序(可作时序依据)
  sheets/      每 --sheet-rows×--sheet-cols 帧拼一张缩略总览,快速翻看整段视频
  events.csv   事件区间标注模板:用户按总览图填「泥石流出现在画面里的起止秒」,可填多段
  frames_index.csv  每帧一行的索引(frame_file, t_sec, sheet 归属),便于回查

标注约定(与「识别已出现」口径一致):
  正区间 = 泥石流流体(泥浆/石块混合流)在画面中可见的时段,含刚出现到完全退出画面;
  清水/浑水溪流、雨天空镜都不算正。一段视频可以有多段正区间(一场事件多次龙头)。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\extract_frames.py ^
        "data/raw/video_dataset_1/..." --interval 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频拆帧 + 事件区间标注模板生成")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("--interval", type=float, default=2.0, help="抽帧间隔(秒),默认 2")
    parser.add_argument("--out", default="data/process/video_frames/video1_segment",
                        help="输出根目录(默认新 data 结构 data/process/video_frames/video1_segment;"
                             "旧 data/frames 已废弃并入该目录)")
    parser.add_argument("--max-side", type=int, default=960, help="帧图最长边像素上限(省空间)")
    parser.add_argument("--sheet-rows", type=int, default=5, help="总览图行数")
    parser.add_argument("--sheet-cols", type=int, default=6, help="总览图列数")
    return parser.parse_args()


def shrink(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def imwrite_unicode(path: Path, img: np.ndarray, params: list[int]) -> None:
    """cv2.imwrite 在 Windows 中文路径下会静默失败,改用 imencode+tofile。"""
    ok, buf = cv2.imencode(path.suffix.lower(), img, params)
    if not ok:
        raise SystemExit(f"编码失败:{path.name}")
    buf.tofile(str(path))


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"视频不存在:{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频:{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else 0.0
    print(f"视频:{video_path.name}  fps≈{fps:.1f}  总帧 {total}  时长≈{duration / 60:.1f} 分钟")

    out_dir = Path(args.out) / video_path.stem
    frames_dir, sheets_dir = out_dir / "frames", out_dir / "sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # 帧名带抽帧序号与秒数:序号即时序(训练序列模式时按文件名排序即按时间)
    grid, grid_ts, sheet_id = [], [], 0
    index_rows: list[tuple[str, float, int]] = []
    next_t, frame_no = 0.0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_sec + 1e-6 >= next_t:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frame = shrink(frame, args.max_side)
            name = f"frame_{frame_no:06d}_t{int(round(t_sec))}s.jpg"
            imwrite_unicode(frames_dir / name, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            grid.append(cv2.resize(frame, (320, 200)))
            grid_ts.append(t_sec)
            index_rows.append((name, round(t_sec, 2), sheet_id))
            next_t = t_sec + args.interval
            frame_no += 1
            if len(grid) == args.sheet_rows * args.sheet_cols:
                sheet_path = sheets_dir / f"sheet_{sheet_id:03d}.jpg"
                imwrite_unicode(sheet_path, stack_sheet(grid, grid_ts, args), [cv2.IMWRITE_JPEG_QUALITY, 85])
                grid, grid_ts = [], []
                sheet_id += 1
    cap.release()
    if grid:
        imwrite_unicode(sheets_dir / f"sheet_{sheet_id:03d}.jpg", stack_sheet(grid, grid_ts, args), [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheet_id += 1

    with (out_dir / "frames_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_file", "t_sec", "sheet"])
        writer.writerows(index_rows)

    with (out_dir / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "start_sec", "end_sec", "confidence", "kind", "note"])
        writer.writerow([video_path.stem, "", "", "", "pos", "示例行:填实际起止秒后删除本行"])

    print(f"抽帧 {frame_no} 张 → {frames_dir}")
    print(f"总览图 {sheet_id} 张 → {sheets_dir}(看图填 {out_dir / 'events.csv'})")


def stack_sheet(grid: list[np.ndarray], ts: list[float], args: argparse.Namespace) -> np.ndarray:
    """拼总览网格:每格左上角写时间戳,方便对照填 events.csv。"""
    rows, cols = args.sheet_rows, args.sheet_cols
    cell_h, cell_w = grid[0].shape[0], grid[0].shape[1]
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 24, dtype=np.uint8)
    for i, (img, t) in enumerate(zip(grid, ts)):
        r, c = divmod(i, cols)
        y, x = r * cell_h, c * cell_w
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
        cv2.putText(sheet, f"t={t:.0f}s", (x + 6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return sheet


if __name__ == "__main__":
    main()
