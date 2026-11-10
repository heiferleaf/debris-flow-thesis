# -*- coding: utf-8 -*-
"""判定规则与判定线的折外实验（预注册见同目录 `预注册-判定规则与判定线.md`）。

只读现有折外分数文件，不训练、不改数据。

用法：
  .venv/Scripts/python.exe .codex/docs/.temp/operating_point.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]

RUNS = [
    ("EXP-009 150池 原间隔", REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug/clip-scores.npz"),
    ("EXP-014 128池 原间隔", REPO / "data/results/exp008/EXP-014-strong-v2-128-strict-ft-noaug/clip-scores.npz"),
    ("EXP-014 128池 新间隔", REPO / "data/results/exp008/EXP-014-strong-v3-128-strict-ft-noaug/clip-scores.npz"),
]

SCORE_GRID = np.round(np.arange(0.05, 0.96, 0.01), 4)
FRAC_GRID = np.round(np.arange(0.05, 0.96, 0.05), 4)
CLIP_THR = 0.5
MAX_FN = 1


def rule_scores(z):
    """三条规则的分数（不校准）。返回 dict 规则名 -> (150,) 分数。"""
    s = z["clip_scores"].astype(np.float64)
    m = z["mask"]
    sm = np.where(m, s, -np.inf)
    sav = np.where(m, s, np.nan)
    n_avail = m.sum(1)
    return {
        "R1 取最高分": sm.max(1),
        "R2 全部片段均值": np.nanmean(sav, axis=1),
        "R3 过线片段占比": (sm >= CLIP_THR).sum(1) / n_avail,
    }


def grid_for(rule):
    return FRAC_GRID if rule.startswith("R3") else SCORE_GRID


def conf(y, pred):
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    return fp, fn


def pct_within_fold(v, folds):
    """折内百分位：同折全体视频里，这条视频的分数排在第几百分位。"""
    out = np.empty_like(v, dtype=np.float64)
    for f in np.unique(folds):
        m = folds == f
        x = v[m]
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        out[m] = r / max(len(x) - 1, 1)
    return out


def best_thr(y, v, grid):
    """在给定集合上选判定线：误报最少，且漏报 <= MAX_FN；同分取更高（更保守）的档。"""
    best = None
    for t in grid:
        fp, fn = conf(y, v >= t)
        if fn > MAX_FN:
            continue
        key = (fp, -t)
        if best is None or key < best[0]:
            best = (key, t, fp, fn)
    if best is None:
        return None
    return best[1], best[2], best[3]


def run_one(name, path):
    z = np.load(path, allow_pickle=True)
    y = z["labels"].astype(int)
    folds = z["folds"].astype(int)
    rules = rule_scores(z)
    print(f"\n=== {name} ===")
    print(f"{len(y)} 条（正 {int((y==1).sum())} / 负 {int((y==0).sum())}），"
          f"负样本分布在 {np.unique(folds[y==0]).size} 个折")
    print("\n[一] 直接用默认线（R1/R2 用 0.5，R3 用占比 0.5），不校准：")
    for r, v in rules.items():
        base = 0.5
        fp, fn = conf(y, v >= base)
        print(f"  {r:12s} 线 {base:.2f}：FP {fp} / FN {fn}")

    print("\n[二] 折内百分位校准后，在其余 4 折上选线、留出折上考核（轮转五次）：")
    for r, v in rules.items():
        p = pct_within_fold(v, folds)
        grid = grid_for(r)
        tot_fp = tot_fn = 0
        picks = []
        ok = True
        for f in sorted(np.unique(folds)):
            m_hold = folds == f
            sen = best_thr(y[~m_hold], p[~m_hold], grid)
            if sen is None:
                ok = False
                picks.append((f, None, None, None))
                continue
            t, fp_sel, fn_sel = sen
            pred = p[m_hold] >= t
            fp, fn = conf(y[m_hold], pred)
            tot_fp += fp
            tot_fn += fn
            picks.append((f, t, fp, fn))
        if not ok:
            print(f"  {r:12s} 选不出满足「漏报 <= {MAX_FN}」的线")
            continue
        detail = "；".join(f"第{f}折选 {t:.2f}→FP {fp}/FN {fn}" for f, t, fp, fn in picks)
        print(f"  {r:12s} 合计 FP {tot_fp} / FN {tot_fn}｜{detail}")

    print("\n[三] 参照：不校准、直接在全体上按同一条折外选线规则选（会用到被考核折，只作对照）：")
    for r, v in rules.items():
        grid = grid_for(r)
        sen = best_thr(y, v, grid)
        if sen is None:
            print(f"  {r:12s} 选不出")
            continue
        t, fp_sel, fn_sel = sen
        print(f"  {r:12s} 线 {t:.2f}：FP {fp_sel} / FN {fn_sel}")


def main():
    for name, path in RUNS:
        if not path.is_file():
            print(f"缺文件：{path}")
            continue
        run_one(name, path)


if __name__ == "__main__":
    main()
