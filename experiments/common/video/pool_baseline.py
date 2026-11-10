# -*- coding: utf-8 -*-
"""把一份已有的预测分数重新算到另一个池子上：逐折 AUC、TP/FP/FN/TN、FPR。

为什么需要：换池子（如 EXP-014 的 128 条池）之后，旧模型的绝对成绩不能直接沿用，
也不能直接与旧的 150 条池成绩比 —— 池子变了，绝对数字自动变好。凡要做两臂对照，
都必须在同一个池子上重算一遍，本脚本就是这个重算，不重新训练、不改分数。

只做三件事：按池子筛行、核对标签与折归属没被筛错、逐折算指标。

用法（仓库根）：

  .venv/Scripts/python.exe experiments/common/video/pool_baseline.py \
      --predictions data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug/predictions.csv \
      --meta data/process/EXP-008-units-pool128/meta-128.npz

  # 只打印合计
  .venv/Scripts/python.exe experiments/common/video/pool_baseline.py --predictions ... --meta ... --quiet
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
THR = 0.5


def auc(y, s):
    """AUC，用秩公式，并列取平均秩。"""
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    vals, inv = np.unique(allv, return_inverse=True)
    sums = np.bincount(inv, weights=ranks)
    cnt = np.bincount(inv)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2)
                 / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, type=Path, help="run 目录的 predictions.csv")
    ap.add_argument("--meta", required=True, type=Path, help="池子元数据 npz（unit_ids/labels/folds）")
    ap.add_argument("--thr", type=float, default=THR, help=f"判定线，默认 {THR}")
    ap.add_argument("--quiet", action="store_true", help="只打合计行")
    ap.add_argument("--json-out", type=Path, default=None, help="把结果写成 JSON")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    a.meta = resolve_path(a.meta, must_exist=True)
    a.predictions = resolve_path(a.predictions, must_exist=True)
    z = np.load(a.meta, allow_pickle=True)
    pool = [str(u) for u in z["unit_ids"]]
    plab = z["labels"].astype(int)
    pfold = z["folds"].astype(int)

    with open(a.predictions, encoding="utf-8-sig", newline="") as fh:
        rows = {r["unit_id"]: r for r in csv.DictReader(fh)}
    lacking = [u for u in pool if u not in rows]
    if lacking:
        raise SystemExit(f"预测文件里没有这些池内视频：{lacking}")
    # 筛行时最容易犯的错是标签或折归属悄悄错位，逐条核对。
    bad = [u for u in pool
           if int(float(rows[u]["label"])) != int(plab[pool.index(u)])
           or int(float(rows[u]["fold"])) != int(pfold[pool.index(u)])]
    if bad:
        raise SystemExit(f"这些视频的标签或折归属与池子元数据不一致：{bad[:10]}")
    print(f"池子 {len(pool)} 条（正 {int((plab == 1).sum())} / 负 {int((plab == 0).sum())}），"
          f"预测文件 {a.predictions.name}，判定线 {a.thr}")

    score = np.array([float(rows[u]["score"]) for u in pool])
    out = {"pool_size": len(pool), "threshold": a.thr, "predictions": str(a.predictions),
           "meta": str(a.meta), "folds": []}
    tot = dict(tp=0, fp=0, fn=0, tn=0)
    for f in sorted(set(pfold.tolist())):
        m = pfold == f
        p = score[m] >= a.thr
        lab = plab[m]
        tp = int((p & (lab == 1)).sum()); fp = int((p & (lab == 0)).sum())
        fn = int((~p & (lab == 1)).sum()); tn = int((~p & (lab == 0)).sum())
        for k, v in dict(tp=tp, fp=fp, fn=fn, tn=tn).items():
            tot[k] += v
        r = {"fold": int(f), "n": int(m.sum()), "n_pos": int((lab == 1).sum()),
             "n_neg": int((lab == 0).sum()), "auc": auc(lab, score[m]),
             "tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "fpr": fp / (fp + tn) if (fp + tn) else float("nan")}
        out["folds"].append(r)
        if not a.quiet:
            print(f"  第 {f} 折：{r['n']:3d} 条（正 {r['n_pos']:3d} / 负 {r['n_neg']:2d}）"
                  f"  AUC {r['auc']:.4f}  TP {tp:3d} FP {fp:2d} FN {fn:2d} TN {tn:2d}"
                  f"  FPR {r['fpr']:6.1%}")
    aucs = [r["auc"] for r in out["folds"] if np.isfinite(r["auc"])]
    out.update(tp=tot["tp"], fp=tot["fp"], fn=tot["fn"], tn=tot["tn"],
               mean_auc=float(np.mean(aucs)), std_auc=float(np.std(aucs)),
               fpr=tot["fp"] / (tot["fp"] + tot["tn"]))
    print(f"合计：TP {tot['tp']} FP {tot['fp']} FN {tot['fn']} TN {tot['tn']}  "
          f"FPR {out['fpr']:.1%}  五折 AUC 均值 {out['mean_auc']:.4f} ± {out['std_auc']:.4f}  "
          f"逐折 FP {[r['fp'] for r in out['folds']]}")
    if a.json_out:
        a.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写到 {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
