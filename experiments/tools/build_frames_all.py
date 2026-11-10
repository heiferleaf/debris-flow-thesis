# -*- coding: utf-8 -*-
"""建 frames-all 合并视图:东川(video1_segment) + 网络(video2_segment) 全部视频目录
目录联接,零拷贝——给审计台一个统一入口翻所有视频。

语义(2026-09-16 叶海峰定):**审计台合并视图不再落盘 data/frames-all**,需要时按需
重建,输出目录由 --out 指定,默认 .codex/docs/.temp/frames-all-view(一次性产物不入 data/)。
新的源目录沿 data/process/video_frames/ 下各 video*_segment 根扫描,新增数据源视频
目录自动涵盖,无需改代码。
用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\build_frames_all.py [--out <目录>]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
_FRAMES_ROOT = REPO / "data" / "process" / "video_frames"
SEGMENT_ROOTS = [REPO / "data" / "process" / "video_frames" / name
                 for name in os.listdir(_FRAMES_ROOT)
                 if name.endswith("_segment")] if _FRAMES_ROOT.is_dir() else []


def main() -> None:
    ap = argparse.ArgumentParser(description="审计台帧合并视图(按需重建,默认输出 .temp)")
    ap.add_argument("--out", default=".codex/docs/.temp/frames-all-view",
                    help="合并视图输出目录(默认 .codex/docs/.temp/frames-all-view,审计台按需重建用;"
                         "旧 data/frames-all 不再落盘)")
    args = ap.parse_args()

    all_dir = REPO / args.out
    all_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for root in SEGMENT_ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not (d.is_dir() and (d / "frames_index.csv").is_file()):
                continue
            link = all_dir / d.name
            if link.exists():
                continue
            r = subprocess.run(["cmd", "/c", "mklink", "/j", str(link), str(d)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if link.exists():
                n += 1
            else:
                print("FAIL", d.name, r.stderr[-100:])
    print("linked", n)
    print("total dirs in frames-all:", sum(1 for d in all_dir.iterdir() if (d / "frames_index.csv").is_file()))


if __name__ == "__main__":
    main()