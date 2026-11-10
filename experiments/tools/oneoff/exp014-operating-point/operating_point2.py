# -*- coding: utf-8 -*-
"""判定规则与判定线的折外实验（第二版，预注册见同目录 `预注册-判定规则与判定线.md`）。

相对第一版改了两处，都是实现层面的修正，不改预注册的口径：

1. 折内百分位改用「并列值取平均名次」。第一版用 argsort 序号，而这两份分数
   每个折有 45%–83% 的视频并列在 1.0 附近，序号在并列块里任意分配，
   「第 21 百分位」在不同折对应的原始分数完全不可比。
2. 增加「不校准、直接在其余 4 折的原始分数上选线」作为对照，用来判断
   百分位校准到底带来了什么。

只读现有折外分数文件，不训练、不改数据。

用法：
  .venv/Scripts/python.exe experiments/tools/oneoff/exp014-operating-point/operating_point2.py
"""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[4]

RUNS = [
    ("EXP-009 150池 原间隔", REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug/clip-scores.npz"),
    ("EXP-014 128池 原间隔", REPO / "data/results/exp008/EXP-014-strong-v2-128-strict-ft-noaug/clip-scores.npz"),
    ("EXP-014 128池 新间隔", REPO / "data/results/exp008/EXP-014-strong-v3-128-strict-ft-noaug/clip-scores.npz"),
]

# 修订一（见预注册）：0.95 以上用 0.0001 步长补细，因为分数在折内饱和于 1.0。
SCORE_GRID = np.round(np.concatenate([
    np.arange(0.05, 0.9501, 0.01),
    np.arange(0.9510, 0.99991, 0.0001),
]), 4)
FRAC_GRID = np.round(np.arange(0.05, 0.96, 0.05), 4)
CLIP_THR = 0.5
MAX_FN = 1


def rule_scores(z):
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
    return int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())


def pct_tie_aware(v, folds):
    """折内百分位，并列值取平均名次，(150,) 落在 [0,1]。"""
    out = np.empty_like(v, dtype=np.float64)
    for f in np.unique(folds):
        m = folds == f
        x = v[m]
        r = rankdata(x, method="average") - 1.0
        out[m] = r / max(len(x) - 1, 1)
    return out


def best_thr(y, v, grid):
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


def evaluate(y, folds, v, grid, calib):
    """折外选线：留出第 f 折，在其余 4 折上选线，用到第 f 折。轮转五次。"""
    p = pct_tie_aware(v, folds) if calib else v
    tot_fp = tot_fn = 0
    rows = []
    for f in sorted(np.unique(folds)):
        hold = folds == f
        sen = best_thr(y[~hold], p[~hold], grid)
        if sen is None:
            return None, []
        t = sen[0]
        pred_hold = p[hold] >= t
        fp, fn = conf(y[hold], pred_hold)
        # 该折被选中的档位，换算回原始分数是多少（只对百分位校准有意义）
        raw = ""
        if calib:
            held = v[hold][pred_hold]
            raw = f"（该折原始分数最低 {held.min():.4f}）" if held.size else "（该折无报警）"
        rows.append((f, t, fp, fn, raw))
        tot_fp += fp
        tot_fn += fn
    return (tot_fp, tot_fn), rows


def run_one(name, path):
    z = np.load(path, allow_pickle=True)
    y = z["labels"].astype(int)
    folds = z["folds"].astype(int)
    rules = rule_scores(z)
    print(f"\n{'='*72}\n{name}")
    print(f"{len(y)} 条（P {int((y==1).sum())} / N {int((y==0).sum())}）")
    neg_fold = [f"{f}:{int(((y==0)&(folds==f)).sum())}" for f in sorted(np.unique(folds))]
    print("每折负样本数 " + "  ".join(neg_fold))

    for calib, label in ((False, "不校准（直接用原始分数）"), (True, "折内百分位（并列取平均名次）")):
        print(f"\n[{'甲' if not calib else '乙'}] {label}")
        for r, v in rules.items():
            res, rows = evaluate(y, folds, v, grid_for(r), calib)
            if res is None:
                t05 = 0.5
                fp, fn = conf(y, v >= t05)
                print(f"  {r:12s} 折外选不出满足「漏报 <= {MAX_FN}」的线"
                      f"（该规则在 0.5 线上 FP {fp} / FN {fn}）")
                continue
            tot_fp, tot_fn = res
            detail = "；".join(f"第{f}折选 {t:.2f}{raw}→FP {fp}/FN {fn}" for f, t, fp, fn, raw in rows)
            print(f"  {r:12s} 合计 FP {tot_fp} / FN {tot_fn}")
            print(f"               {detail}")

    print("\n[参照] 固定线 0.5 / 占比 0.5（现用的做法，不选线）：")
    for r, v in rules.items():
        fp, fn = conf(y, v >= 0.5)
        print(f"  {r:12s} FP {fp} / FN {fn}")


def main():
    for name, path in RUNS:
        if not path.is_file():
            print(f"缺文件：{path}")
            continue
        run_one(name, path)


if __name__ == "__main__":
    main()
