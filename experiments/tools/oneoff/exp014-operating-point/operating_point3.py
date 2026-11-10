# -*- coding: utf-8 -*-
"""判定线折外实验的完整表（第三版）：固定线取舍表 + 两条折外选线规则。

第一版的预注册规则（误报最少、漏报 <= 1）在饱和分数上把线推到网格顶端，
不是内部最优。本文件把它原样保留，另加一条可部署的规则作为对照：
「判定线 = 其余 4 折上正样本分数的第 1 百分位」，即允许在训练折上漏掉 1% 的正样本。

用法：
  .venv/Scripts/python.exe experiments/tools/oneoff/exp014-operating-point/operating_point3.py
"""
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUNS = [
    ("EXP-009 150池 原间隔", REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug/clip-scores.npz"),
    ("EXP-014 128池 原间隔", REPO / "data/results/exp008/EXP-014-strong-v2-128-strict-ft-noaug/clip-scores.npz"),
    ("EXP-014 128池 新间隔", REPO / "data/results/exp008/EXP-014-strong-v3-128-strict-ft-noaug/clip-scores.npz"),
]
CLIP_THR = 0.5
FIXED = [0.5, 0.9, 0.95, 0.99, 0.995]


def load(path):
    z = np.load(path, allow_pickle=True)
    s = z["clip_scores"].astype(np.float64)
    m = z["mask"]
    return (np.where(m, s, -np.inf).max(1),
            np.nanmean(np.where(m, s, np.nan), axis=1),
            ((np.where(m, s, -np.inf) >= CLIP_THR).sum(1) / m.sum(1)),
            z["labels"].astype(int), z["folds"].astype(int))


def conf(y, pred):
    return int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())


def pct(v, folds):
    out = np.empty_like(v)
    for f in np.unique(folds):
        m = folds == f
        r = np.argsort(np.argsort(v[m], kind="mergesort"), kind="mergesort")
        out[m] = r / max(m.sum() - 1, 1)
    return out


def table_fixed(name, v, y):
    P, N = (y == 1).sum(), (y == 0).sum()
    print(f"\n== {name}：固定判定线（描述性，不是折外数） P {P} / N {N}")
    print("  判定线   TP  FP  FN  TN   Precision  Recall   FPR")
    for t in FIXED:
        pred = v >= t
        tp = int((pred & (y == 1)).sum()); fp, fn = conf(y, pred)
        tn = int(N) - fp
        print(f"  {t:<7} {tp:3d} {fp:3d} {fn:3d} {tn:3d}   "
              f"{tp/max(tp+fp,1):9.4f}  {tp/P:6.4f}  {fp/N:5.3f}")


def rule_prereg(y, folds, v, grid):
    """预注册规则：其余 4 折上「误报最少、且漏报 <= 1」、同分取更高。"""
    tot = [0, 0]
    per = []
    for f in sorted(np.unique(folds)):
        hold = folds == f
        best = None
        for t in grid:
            fp, fn = conf(y[~hold], v[~hold] >= t)
            if fn > 1:
                continue
            key = (fp, -t)
            if best is None or key < best[0]:
                best = (key, t)
        if best is None:
            return None, []
        t = best[1]
        fp, fn = conf(y[hold], v[hold] >= t)
        tot[0] += fp
        tot[1] += fn
        per.append((f, t, fp, fn))
    return tot, per


def rule_quantile(y, folds, v, q):
    """可部署规则：判定线 = 其余 4 折正样本分数的第 q 百分位。"""
    tot = [0, 0]
    per = []
    for f in sorted(np.unique(folds)):
        hold = folds == f
        t = float(np.quantile(v[(~hold) & (y == 1)], q))
        fp, fn = conf(y[hold], v[hold] >= t)
        tot[0] += fp
        tot[1] += fn
        per.append((f, t, fp, fn))
    return tot, per


def main():
    out = []
    for name, path in RUNS:
        v1, v2, v3, y, folds = load(path)
        P, N = int((y == 1).sum()), int((y == 0).sum())
        out.append(f"\n{'#'*70}\n{name}（P {P} / N {N}）")
        for line in []:
            pass
        table_fixed(name, v1, y)

        grid = np.round(np.concatenate([np.arange(0.05, 0.9501, 0.01),
                                        np.arange(0.9510, 0.99991, 0.0001)]), 4)
        print("\n  [折外选线 甲] 预注册规则（误报最少、漏报 <= 1、同分取更高）：")
        tot, per = rule_prereg(y, folds, v1, grid)
        if tot is None:
            print("    选不出")
        else:
            print(f"    合计 FP {tot[0]} / FN {tot[1]}｜" +
                  "；".join(f"第{f}折选 {t:.4f}→FP {fp}/FN {fn}" for f, t, fp, fn in per))

        print("  [折外选线 乙] 判定线 = 其余 4 折正样本分数的第 1 百分位（即允许训练折上漏 1%）：")
        tot, per = rule_quantile(y, folds, v1, 0.01)
        print(f"    合计 FP {tot[0]} / FN {tot[1]}｜" +
              "；".join(f"第{f}折线 {t:.4f}→FP {fp}/FN {fn}" for f, t, fp, fn in per))

        print("  [折外选线 丙] 同乙，但第 5 百分位：")
        tot, per = rule_quantile(y, folds, v1, 0.05)
        print(f"    合计 FP {tot[0]} / FN {tot[1]}｜" +
              "；".join(f"第{f}折线 {t:.4f}→FP {fp}/FN {fn}" for f, t, fp, fn in per))

    text = "\n".join(out)
    print(text)
    Path(__file__).with_name("operating_point3.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
