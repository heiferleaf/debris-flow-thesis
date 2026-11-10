# -*- coding: utf-8 -*-
"""只读：6 条「整段饱和」误报的视频，喂进模型的帧是不是本来就发灰。

线索来自 22 条排除视频的摸底：v1_110 到 v1_113 这 4 条负样本，整条视频有 66%–80% 的
时长落在禁区里（禁区含「灰度标准差低于 20 的帧前后各 16 帧」这一类）。剩下能取样的时
间只有 112–185 秒，48 个片段全部从这段里取。所以要看的是：这些片段本身有多少张是发灰的。

判据：一张帧转灰度后，像素灰度值的标准差低于 20 就算「发灰」（与取材管线判禁区的阈值
同源，见 EXP-014 卡的容量核对）。
"""
import csv
import pathlib

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[4]
FRAMES = REPO / "data/process/EXP-008-units-v2/frames"
MAN = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
D_ = np.load(REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug/clip-scores.npz",
             allow_pickle=True)
GREY_STD = 20.0
SAT6 = ["v1_110", "v1_111", "v1_112", "v1_113", "v1_96", "v1_97"]

ids = [str(x) for x in D_["unit_ids"]]
clip, mask, label = D_["clip_scores"], D_["mask"], D_["labels"].astype(int)
meta = {}
with open(MAN, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        meta[r["unit_id"]] = r

rows = []
for i, u in enumerate(ids):
    ps = [FRAMES / u / f"clip{c:02d}_f00.jpg" for c in range(len(clip[i])) if mask[i][c]]
    vals = []
    for p in ps:
        if not p.exists():
            continue
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        vals.append(float(g.std()))
    if vals:
        rows.append((u, label[i], len(vals), float(np.mean(vals)),
                     float(np.mean([v < GREY_STD for v in vals])),
                     float(clip[i][mask[i]].max())))

print(f"参与统计 {len(rows)} 条视频（每条取每个可用片段的首帧）")
print(f"「发灰」= 灰度标准差 < {GREY_STD}\n")


def block(name, sel):
    if not sel:
        print(f"{name:<22} 0 条")
        return
    std = np.array([r[3] for r in sel])
    frac = np.array([r[4] for r in sel])
    sco = np.array([r[5] for r in sel])
    print(f"{name:<22} {len(sel):>3} 条   片段首帧灰度标准差 中位 {np.median(std):>6.1f}"
          f"   发灰帧占比 中位 {np.median(frac):>5.1%}   视频分数 中位 {np.median(sco):.6f}")


print("按类别看（整条视频层面的中位数）：")
block("6 条整段饱和误报", [r for r in rows if r[0] in SAT6])
block("其余负样本", [r for r in rows if r[1] == 0 and r[0] not in SAT6])
block("全部正样本", [r for r in rows if r[1] == 1])

print("\n逐条看那 6 条：")
print(f"  {'视频':<8}{'可用片段':>8}{'灰度标准差中位':>14}{'发灰帧占比':>12}{'视频分数':>12}")
for r in sorted([r for r in rows if r[0] in SAT6], key=lambda x: -x[5]):
    print(f"  {r[0]:<8}{r[2]:>8}{r[3]:>14.1f}{r[4]:>12.1%}{r[5]:>12.6f}")

print("\n对照：发灰帧占比最高的 10 条（不分正负）：")
for r in sorted(rows, key=lambda x: -x[4])[:10]:
    print(f"  {r[0]:<8}标签 {r[1]}  可用片段 {r[2]:>2}  发灰占比 {r[4]:>5.1%}  "
          f"灰度标准差 {r[3]:>5.1f}  分数 {r[5]:.6f}")
