# -*- coding: utf-8 -*-
"""只读：片段数少于 48 时，「取最高分」这条聚合规则会不会系统性压分。

推理路径允许视频不足 48 组时按实际组数取片段（`build_units.pick_slots` 返回
min(48, 可用槽数)，不补齐、不报错）。但「整条视频的分数 = 可用片段取最大」这条规则
有个几何性质：抽 n 个数的最大值随 n 增大而增大。所以片段少的视频天然吃亏——同样一段
泥石流，只抽到 10 组时拿到的最高分，会比抽到 48 组时低。

这里把这件事量出来：对每条正样本，从它自己的 48 个片段分数里无放回抽 n 个、取最大，
重复多次，看落在 0.5 以下的概率随 n 怎么变。负样本同样做一遍，看误报怎么变。
"""
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[4]
RUN = REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug"
NS = [5, 10, 15, 20, 30, 40, 48]
REPEAT = 2000
TH = 0.5

d = np.load(RUN / "clip-scores.npz", allow_pickle=True)
clip, mask = d["clip_scores"].astype(np.float64), d["mask"]
label = d["labels"].astype(int)
usable = [c[m] for c, m in zip(clip, mask)]

print("一、池子里每条视频实际参与打分的片段数")
for v, nm in ((1, "正样本"), (0, "负样本")):
    n = np.array([len(u) for u, l in zip(usable, label) if l == v])
    print(f"  {nm} {len(n)} 条：最少 {n.min()}，最多 {n.max()}，中位 {int(np.median(n))}，"
          f"不足 48 的 {int((n < 48).sum())} 条")
print("\n  片段数 与 视频分数 的关系（Spearman）：")
from scipy.stats import spearmanr

n_all = np.array([len(u) for u in usable])
s_all = np.array([u.max() for u in usable])
for v, nm in ((1, "正样本"), (0, "负样本")):
    sel = label == v
    rho, p = spearmanr(n_all[sel], s_all[sel])
    print(f"    {nm}：rho {rho:+.3f}（p={p:.2f}，{int(sel.sum())} 条）")

print(f"\n二、只给 n 个片段时，「取最高分」判成有泥石流的比例（阈值 {TH}，每条重复 {REPEAT} 次）")
print(f"  {'n':>4}{'正样本判出率':>14}{'负样本误报率':>14}")
rng = np.random.default_rng(20261027)
rows = []
for n in NS:
    tp = tot_p = fp = tot_n = 0
    for u, l in zip(usable, label):
        if len(u) < n:
            continue
        draws = np.array([rng.choice(u, size=n, replace=False).max() for _ in range(REPEAT)])
        hit = (draws >= TH).mean()
        if l == 1:
            tp += hit
            tot_p += 1
        else:
            fp += hit
            tot_n += 1
    rows.append((n, tp / tot_p, fp / tot_n, tot_p, tot_n))
    print(f"  {n:>4}{tp/tot_p:>14.4f}{fp/tot_n:>14.4f}")

print("\n三、同一件事按「丢掉多少条正样本」说")
base = rows[-1][1]
for n, r, f, npv, nnv in rows:
    print(f"  n={n:>2}：正样本判出率 {r:.4f}（比 n=48 低 {base-r:.4f}，"
          f"相当于 {int(round((base-r)*npv))} 条正样本被判成没有），负样本误报率 {f:.4f}")

print("\n四、固定同一批视频再比一次（消除上一张表里「每个 n 参与的视频不是同一批」的问题）")
print("  只取有 48 个片段的视频，从它自己的 48 个片段里抽 n 个；分母不变。")
full = [u for u in usable if len(u) == 48]
full_l = [l for u, l in zip(usable, label) if len(u) == 48]
n_p = sum(full_l)
n_n = len(full_l) - n_p
print(f"  这批视频：正 {n_p} 条、负 {n_n} 条\n")
print(f"  {'n':>4}{'正样本判出率':>14}{'负样本误报率':>14}")
for n in NS:
    tp = sum((np.array([rng.choice(u, size=n, replace=False).max() for _ in range(REPEAT)]) >= TH).mean()
             for u, l in zip(full, full_l) if l == 1)
    fp = sum((np.array([rng.choice(u, size=n, replace=False).max() for _ in range(REPEAT)]) >= TH).mean()
             for u, l in zip(full, full_l) if l == 0)
    print(f"  {n:>4}{tp/n_p:>14.4f}{(fp/n_n if n_n else float('nan')):>14.4f}")
print("\n  这张表才是同一条视频在不同片段数下的对照：正样本几乎不动，负样本随 n 增大更爱误报"
      "\n  （n 越小，负样本越不容易冒过一个高分），所以「片段少」在正样本这一侧是轻微吃亏，"
      "\n  在负样本这一侧反而更安全。两边都是百分之几以内的量，远小于判定线挪一档带来的变化。")

# 第五节：回答「92 条片段不满 48」与「22 条被排除」是不是同一件事
# 起因：叶海峰 2026-10-27 看到第一节的表，问「那为什么又说只有 22 条视频被排除」。
import csv

print("\n五、第一节那张表（片段数不满 48）与 EXP-014 排除的 22 条不是同一件事")
man = {r["unit_id"]: r
       for r in csv.DictReader(open(REPO / "data/process/EXP-008-units-v2/unit-manifest.csv",
                                    encoding="utf-8"))}
fail = [r["unit_id"] for r in
        csv.DictReader(open(REPO / "data/process/EXP-008-units-v3-gap05/unit-failures.csv",
                            encoding="utf-8"))]
ids = [str(x) for x in d["unit_ids"]]
pos_of = {u: i for i, u in enumerate(ids)}
print(f"  主池 150 条在清单里抽到的位置数：{sorted(set(man[u]['n_clips'] for u in ids))}"
      f"（每条都抽满 48 个位置，所以每条都在池子里）")
print(f"  实际参与打分的片段数不到 48 的有 {sum(1 for u in usable if len(u) < 48)} 条，"
      f"原因是有些位置落在人工标注的剪辑区间里，被掩码剔除、不参与打分。")
not_in = [u for u in fail if u not in pos_of]
print(f"  EXP-014 排除的 22 条：不在主池里的有 {len(not_in)} 条；"
      f"它们在主池里参与打分的片段数是 "
      f"{sorted(len(usable[pos_of[u]]) for u in fail if u in pos_of)}")
print("  也就是说这 22 条在 16 帧连续抽法下是池子里最完整的一批，它们被排除只发生在"
      "\n  EXP-014 那个 0.5 秒间隔的对照里——换成那个间隔，它们凑不满 48 个合法位置。")
