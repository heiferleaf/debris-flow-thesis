# -*- coding: utf-8 -*-
"""EXP-008 配对重采样区间工具（只读，2026-10-11 重建于创新点执行计划步 3）。

复现 run-log 第 24 条 W4 的口径（原一次性脚本已销毁；2026-10-11 重建并过回归门禁）：
  - 分数直接读各 run 的 predictions.csv（与原脚本一致——存量 CSV 是 W1 整改前的
    6 位舍入版，strong 线 sigmoid 饱和段有并列值，并列按平均秩处理）；
  - 重采样单位 = 场次组（150 池内 31 组，sorted 次序），有放回重抽 20,000 次，
    np.random.default_rng(7)，每对比较各自独立从 seed=7 起；
  - 每次重采样两臂用同一串组抽样结果（配对）；
  - 混算前把每条视频的分数换算成「在它那一折里排第几」的百分位——折划分在重抽样
    **内部**重新算（平均秩 / 折内条数），两臂各自用自己折划分；
  - 混在一起算一次带并列校正的 AUC（Mann-Whitney 中位秩），取两臂差；
  - 某次抽样里没有正或没有负时该次作废（W4 记有效 19,999 次，本工具复现为同一数）；
  - 区间 = 差值分布的 2.5% / 97.5% 分位（numpy 线性插值）。

重建过程五个混算 AUC 点估计与 W4 表逐位一致（0.876585 / 0.939967 / 0.939967 /
0.982301 / 0.982899），第 2 对区间逐位一致 [−0.2191, +0.0377]。

回归门禁（run-log 第 24 条 W4 表，4 位小数容差，任一不过即停下报）：
  mil-strict − strong-strict     = −0.0634 [−0.2224, +0.0092]
  mil-strict − mil-full          = −0.0634 [−0.2191, +0.0377]
  mil-strict-iid − mil-strict    = +0.1057 [+0.0275, +0.3196]
  strong-strict-iid − strong-strict = +0.0429 [+0.0215, +0.1739]

用法（仓库根）：
  python experiments/tools/paired_bootstrap.py            # 全部 4 对 + 门禁
  python experiments/tools/paired_bootstrap.py --pairs 1  # 只跑第 1 对（0 起）
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data/results/exp008"
FEATS = RUNS / "unitfeats-v2-150/unit-features.npz"
MANIFEST = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
IID_FOLDS = REPO / "data/metadata/splits/EXP-008-units-iid-folds.json"

N_BOOT = 20000
SEED = 7

# 回归门禁：比较对 -> (点估计差, 区间下界, 区间上界)，run-log 第 24 条 W4 表
GATE = [
    ("mil-v2-150-strict", "strong-v2-150-strict",
     -0.0634, -0.2224, 0.0092),
    ("mil-v2-150-strict", "mil-v2-150-full",
     -0.0634, -0.2191, 0.0377),
    ("mil-v2-150-strict-iid", "mil-v2-150-strict",
     0.1057, 0.0275, 0.3196),
    ("strong-v2-150-strict-iid", "strong-v2-150-strict",
     0.0429, 0.0215, 0.1739),
]

# 混算 AUC 点估计（W4 表甲/乙列）——重建正确性的第一道检查
POINT = {
    "mil-v2-150-strict": 0.8766,
    "strong-v2-150-strict": 0.9400,
    "mil-v2-150-full": 0.9400,
    "mil-v2-150-strict-iid": 0.9823,
    "strong-v2-150-strict-iid": 0.9829,
}


def load_manifest():
    rows = {}
    with open(MANIFEST, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["unit_id"]] = r
    return rows


def read_predictions(run_dir: Path):
    out = {}
    with open(run_dir / "predictions.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["unit_id"]] = float(r["score"])
    return out


def midrank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """带并列校正的 AUC（Mann-Whitney，并列各计 0.5）。"""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    n = len(s)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j + 2) / 2.0     # 并列取平均秩（1 起）
        i = j + 1
    n_pos = y.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r_sum = ranks[y == 1].sum()
    return (r_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def avg_rank_pct(v: np.ndarray) -> np.ndarray:
    """折内百分位：平均秩（并列取平均）/ 条数。"""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks / len(v)


def pct_within_resample(v: np.ndarray, fold: np.ndarray, idx: np.ndarray):
    """重抽样内部按折重算百分位。"""
    out = np.empty(len(idx), dtype=np.float64)
    for f in np.unique(fold[idx]):
        m = fold[idx] == f
        out[m] = avg_rank_pct(v[idx][m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="0,1,2,3", help="要跑的比较对序号（0 起）")
    args = ap.parse_args()
    pair_ids = [int(x) for x in args.pairs.split(",")]

    man = load_manifest()
    units = sorted(man)                                    # 150 条固定次序
    labels = np.array([1 if man[u]["label"] == "pos" else 0 for u in units])
    groups = np.array([man[u]["group_id"] for u in units])
    uniq_groups = sorted(set(groups))
    print(f"池 {len(units)} 条 / 场次组 {len(uniq_groups)} 组 / "
          f"正 {labels.sum()} 负 {(1 - labels).sum()}")

    d = np.load(FEATS, allow_pickle=True)
    npz_fold = {str(u): int(f) for u, f in zip(d["unit_ids"], d["folds"])}
    with open(IID_FOLDS, encoding="utf-8") as fh:
        iid = json.load(fh)
    iid_fold = {u: int(it["fold"]) for it in iid["folds"] for u in it["val_units"]}
    folds_of = {
        "mil-v2-150-strict": npz_fold,
        "strong-v2-150-strict": npz_fold,
        "mil-v2-150-full": npz_fold,
        "mil-v2-150-strict-iid": iid_fold,
        "strong-v2-150-strict-iid": iid_fold,
    }

    need = sorted({a for a, _, _, _, _ in GATE} | {b for _, b, _, _, _ in GATE})
    vec, fold_of = {}, {}
    for name in need:
        sc = read_predictions(RUNS / name)
        vec[name] = np.array([sc[u] for u in units])
        fold_of[name] = np.array([folds_of[name][u] for u in units])

    # ---- 点估计对账（全 150 池折内百分位后混算 AUC）----
    def pooled(name):
        pct = np.empty(len(units))
        fold, v = fold_of[name], vec[name]
        for f in np.unique(fold):
            m = fold == f
            pct[m] = avg_rank_pct(v[m])
        return midrank_auc(pct, labels)

    for name, exp in POINT.items():
        got = pooled(name)
        tag = "OK" if abs(got - exp) <= 5e-5 else "MISMATCH"
        print(f"[点估计对账] {tag} {name}: 脚本={got:.4f} W4表={exp:.4f}")
        if tag == "MISMATCH":
            raise SystemExit("点估计对账不过——停下，不做重抽样。")

    # ---- 配对重抽样 + 门禁 ----
    g_lists = [np.flatnonzero(groups == g) for g in uniq_groups]
    nG = len(uniq_groups)
    n_fail = 0
    for pi in pair_ids:
        a, b, d_exp, lo_exp, hi_exp = GATE[pi]
        d_point = pooled(a) - pooled(b)
        rng = np.random.default_rng(SEED)          # 每对独立从 seed=7 起
        diffs = []
        for _ in range(N_BOOT):
            draw = rng.integers(0, nG, nG)
            idx = np.concatenate([g_lists[j] for j in draw])
            y = labels[idx]
            if y.sum() == 0 or y.sum() == len(idx):
                continue
            pa = pct_within_resample(vec[a], fold_of[a], idx)
            pb = pct_within_resample(vec[b], fold_of[b], idx)
            diffs.append(midrank_auc(pa, y) - midrank_auc(pb, y))
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        got = (round(d_point, 4), round(lo, 4), round(hi, 4))
        ok = (abs(got[0] - d_exp) <= 5e-5 and abs(got[1] - lo_exp) <= 5e-5
              and abs(got[2] - hi_exp) <= 5e-5)
        if not ok:
            n_fail += 1
        print(f"[门禁] {'OK' if ok else 'FAIL'} 对{pi + 1} {a} − {b}: "
              f"差 {got[0]} [ {got[1]}, {got[2]} ]  "
              f"(目标 {d_exp} [ {lo_exp}, {hi_exp} ], 有效 {len(diffs)} 次)")
    if n_fail:
        raise SystemExit(f"门禁 {n_fail} 对不过——工具不可用，停下报。")
    print("门禁全过：区间工具可用。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
