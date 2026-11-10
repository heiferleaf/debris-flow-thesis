"""批量拆帧:遍历一个视频数据集目录,逐个调用 extract_frames.py,跳过已拆的视频。

为什么存在:video_dataset_1 有 105 段视频,单段拆帧要几分钟,需要无人值守的顺序批处理;
拆帧产物是 events.csv 空模板 + 总览图,**不做任何预标注**(事件区间由叶海峰人工填,
2026-09-13 定)。

用法(仓库根):
    .venv\\Scripts\\python.exe experiments\\tools\\batch_extract_frames.py data/raw/video_dataset_1

可选:
    --interval 2       抽帧间隔秒(默认 2,与单段工具一致)
    --out data/process/video_frames/video1_segment  输出根目录(新 data 结构;旧 data/frames 已并入)
    --limit N          只处理前 N 段未拆视频(试跑用)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".mpg", ".mpeg", ".ts"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量拆帧(顺序、跳过已拆、失败不中断)")
    p.add_argument("root", help="视频数据集根目录(如 data/raw/video_dataset_1)")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--out", default="data/process/video_frames/video1_segment",
                   help="输出根目录(新 data 结构;旧 data/frames 已并入)")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 段未拆视频,0=全部")
    return p.parse_args()


def already_done(video: Path, out_root: Path) -> bool:
    """frames/ 目录存在且非空即认为已拆。"""
    d = out_root / video.stem / "frames"
    return d.is_dir() and any(d.iterdir())


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[批量] 根目录不存在:{root}", flush=True)
        return 2
    out_root = REPO / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file())
    pending = [v for v in videos if not already_done(v, out_root)]
    print(f"[批量] 共 {len(videos)} 段视频,已拆 {len(videos) - len(pending)} 段,待拆 {len(pending)} 段", flush=True)
    if args.limit:
        pending = pending[: args.limit]
        print(f"[批量] --limit 生效,本轮只拆前 {len(pending)} 段", flush=True)

    extractor = REPO / "experiments" / "tools" / "extract_frames.py"
    python = REPO / ".venv" / "Scripts" / "python.exe"
    ok = fail = 0
    t0 = time.perf_counter()
    for i, video in enumerate(pending, 1):
        print(f"[批量] ({i}/{len(pending)}) {video.name}", flush=True)
        try:
            r = subprocess.run(
                [str(python), str(extractor), str(video), "--interval", str(args.interval), "--out", args.out],
                cwd=str(REPO), timeout=3600,
            )
        except Exception:
            traceback.print_exc()
            r = None
        if r is not None and r.returncode == 0:
            ok += 1
            print(f"[批量] ({i}/{len(pending)}) 完成,累计 {ok} 成功 {fail} 失败,已用 {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        else:
            fail += 1
            print(f"[批量] ({i}/{len(pending)}) ** 失败 (returncode={getattr(r, 'returncode', '?')}),继续下一段 **", flush=True)
    print(f"[批量] 全部结束:{ok} 成功 / {fail} 失败,总耗时 {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
