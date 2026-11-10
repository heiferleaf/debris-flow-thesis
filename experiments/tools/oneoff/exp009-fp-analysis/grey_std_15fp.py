# -*- coding: utf-8 -*-
"""只读：15 条误报视频的全部 768 个采样帧里，有多少张「发灰」。

用途：训练报告 §4.4「这 15 条不是灰帧闯的祸」这句话的复算依据
（research-auditor 2026-10-28 W8：原来那句话用的是灰度标准差 < 5，
比仓库统一判据 < 20 更严，且没有落盘脚本）。

判据与取材管线同源：一张帧转灰度后像素灰度值的标准差低于 20 就算发灰
（`experiments/common/video/build_units.py` 的 QC_STD_LOW = 20）。
脚本同时输出 < 20 与 < 5 两种阈值下的命中数，方便对照更严标准的结论。

误报的认定：标签为负、视频分数（可用片段取最大）≥ 0.5 的视频。

用法（仓库根目录）：
  .venv/Scripts/python.exe experiments/tools/oneoff/exp009-fp-analysis/grey_std_15fp.py
"""
import pathlib

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[4]
FRAMES = REPO / "data/process/EXP-008-units-v2/frames"
RUN = REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug"
STD_PIPE = 20.0   # 取材管线的禁区判据
STD_STRICT = 5.0  # 报告原句用的更严阈值

d = np.load(RUN / "clip-scores.npz", allow_pickle=True)
ids = [str(x) for x in d["unit_ids"]]
clip, mask, label = d["clip_scores"], d["mask"], d["labels"].astype(int)

rows = []
for i, u in enumerate(ids):
    if label[i] != 0:
        continue
    score = float(clip[i][mask[i]].max())
    if score < 0.5:
        continue
    vals, missing = [], 0
    for c in range(clip.shape[1]):
        for f in range(16):
            p = FRAMES / u / f"clip{c:02d}_f{f:02d}.jpg"
            if not p.exists():
                missing += 1
                continue
            g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            vals.append(float(g.std()))
    v = np.array(vals)
    rows.append((u, score, len(v), missing, float(v.min()), int((v < STD_PIPE).sum()),
                 int((v < STD_STRICT).sum())))

print(f"误报视频 {len(rows)} 条（标签为负、视频分数 ≥ 0.5）")
print(f"每条检查 48 个片段 × 16 帧 = 768 张采样帧\n")
print(f"  {'视频':<8}{'分数':>10}{'帧数':>6}{'缺帧':>5}{'最低灰度标准差':>14}"
      f"{'< 20 张数':>10}{'< 5 张数':>9}")
for u, s, n, miss, lo, n20, n5 in sorted(rows, key=lambda r: r[4]):
    print(f"  {u:<8}{s:>10.6f}{n:>6}{miss:>5}{lo:>14.2f}{n20:>10}{n5:>9}")

print()
print(f"汇总：{len(rows)} 条共 {sum(r[2] for r in rows)} 张帧；"
      f"灰度标准差 < {STD_PIPE:.0f} 的 {sum(r[5] for r in rows)} 张；"
      f"< {STD_STRICT:.0f} 的 {sum(r[6] for r in rows)} 张；"
      f"全批最低 {min(r[4] for r in rows):.2f}")
