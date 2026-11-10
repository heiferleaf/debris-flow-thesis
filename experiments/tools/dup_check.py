# -*- coding: utf-8 -*-
"""dHash 感知哈希查重:待入集图 vs d4 全部图片 + 外部视频已拆帧。
用法(仓库根目录): python experiments/tools/dup_check.py <待入集图片路径>
判据: 64bit dHash,汉明距 <=8 视为近亲(不可再入集,会训练重复)。"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def dhash(path, size=8):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size))
    return (img[:, 1:] > img[:, :-1])  # 64bit

def dist(a, b):
    return int((a != b).sum())

if len(sys.argv) < 2:
    sys.exit("用法: python dup_check.py <待入集图片路径>")
target = dhash(sys.argv[1])
if target is None:
    sys.exit(f"读不了图: {sys.argv[1]}")
pools = {
    "d4图片": list(Path("data/raw/dataset4-external-negatives").rglob("*.jpg"))
           + list(Path("data/raw/dataset4-external-negatives").rglob("*.png")),
    # 2026-09-16 data 结构重构:外部视频初帧 data/frames-extneg → data/process/video_frames/video2_segment/
    "外部视频帧": list(Path("data/process/video_frames/video2_segment").rglob("*.jpg")),
}
for name, files in pools.items():
    best, bestf = 99, ""
    n_ok = 0
    for f in files:
        h = dhash(f)
        if h is None:
            continue
        n_ok += 1
        d = dist(target, h)
        if d < best:
            best, bestf = d, f.name
    print(f"{name}: 比对 {n_ok} 张,最小汉明距 {best} ({bestf[:50]}) {'<-- 疑似重复!' if best <= 8 else '(无近亲,<=8才算近)'}")
