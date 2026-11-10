# -*- coding: utf-8 -*-
"""EXP-008 错误分桶工具（只读，2026-10-11 立于创新点执行计划步 1）。

读九套 run 的 predictions.csv 与取材正本清单，做三件事：
  1. 每套 run 在 θ=0.5 下的工作点（pred_pos_rate / TP/FP/FN/TN / Recall / F1），
     并对 run-log 第 24 条 W5 表逐位对账（七套场次级/IID 设定）。
  2. 按折 / 场次组 / 数据源 / 组内是否有正视频 对 mil-strict 的 150 条视频分桶，
     各桶计数之和必须 = 150。
  3. 对误报视频（负样本被判成有）补逐片段分数（复用 score_units 里每折训练出的那套权重做前向，秒级），
     按「过线片段数」区分「整条持续高分」与「单片段拉高」——创新点候选 1
     （至少 k 个片段过线才报）的动机是否成立的直接证据。

只读：不写任何数据、模型、训练脚本；CSV 产物落 runs/error-analysis-2026-10-12/。

用法（仓库根）：
  python experiments/tools/error_analysis.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from score_units import score_run  # noqa: E402

RUNS = REPO / "data/results/exp008"
OUT = RUNS / "error-analysis-2026-10-12"
FEATS = RUNS / "unitfeats-v2-150/unit-features.npz"
CLIP_LABELS = REPO / "data/process/EXP-008-units-v2/clip-labels.npz"
TH = 0.5

# 九套可引用 run：名称 -> (口径, folds_json 或空=用 npz folds)
NINE = {
    "mil-v2-150-strict": ("strict", ""),
    "strong-v2-150-strict": ("strict", ""),
    "mil-v2-150-full": ("full", ""),
    "strong-v2-150-loose": ("loose", ""),
    "mil-v2-150-strict-iid": ("strict",
                              "data/metadata/splits/EXP-008-units-iid-folds.json"),
    "strong-v2-150-strict-iid": ("strict",
                                 "data/metadata/splits/EXP-008-units-iid-folds.json"),
    "mil-v2-150-full-iid": ("full",
                            "data/metadata/splits/EXP-008-units-iid-folds.json"),
    "mil-v2-150-strict-scene": ("strict",
                                "data/metadata/splits/EXP-008-units-scene-folds.json"),
    "strong-v2-150-strict-scene": ("strict",
                                   "data/metadata/splits/EXP-008-units-scene-folds.json"),
}

# run-log 第 24 条 W5 表（θ=0.5，150 单位混看）——对账标准答案
W5 = {
    "mil-v2-150-strict": (0.833, 4, 16, 0.965, 0.916),
    "strong-v2-150-strict": (0.880, 0, 19, 1.000, 0.922),
    "mil-v2-150-full": (0.793, 17, 23, 0.850, 0.828),
    "strong-v2-150-loose": (0.847, 1, 15, 0.991, 0.933),
    "mil-v2-150-strict-iid": (0.813, 6, 15, 0.947, 0.911),
    "strong-v2-150-strict-iid": (0.787, 2, 7, 0.982, 0.961),
    "mil-v2-150-full-iid": (0.780, 5, 9, 0.956, 0.939),
}


def load_manifest():
    rows = {}
    with open(REPO / "data/process/EXP-008-units-v2/unit-manifest.csv",
              encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["unit_id"]] = r
    return rows


def read_predictions(run_dir: Path):
    out = {}
    with open(run_dir / "predictions.csv", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["unit_id"]] = float(r["score"])
    return out


def workpoint(scores: dict, labels: dict):
    """θ=0.5，判成有泥石流的条件 score >= 0.5（与训练脚本、W5 表一致）。"""
    tp = fp = fn = tn = 0
    for uid, s in scores.items():
        pred = s >= TH
        lab = labels[uid] == 1
        if pred and lab:
            tp += 1
        elif pred and not lab:
            fp += 1
        elif not pred and lab:
            fn += 1
        else:
            tn += 1
    p = tp + fp
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / p if p else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else float("nan"))
    return {"pred_pos_rate": p / len(scores), "TP": tp, "FP": fp, "FN": fn,
            "TN": tn, "Recall": recall, "Precision": precision, "F1": f1}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    labels = {u: (1 if r["label"] == "pos" else 0) for u, r in man.items()}
    groups = {u: r["group_id"] for u, r in man.items()}
    sources = {u: r["source"] for u, r in man.items()}

    # ---- 1. 九套工作点 + W5 对账 ----
    wp_rows, fails = [], []
    for name, (caliber, fj) in NINE.items():
        sc = read_predictions(RUNS / name)
        wp = workpoint(sc, labels)
        wp_rows.append({"run": name, **wp})
        if name in W5:
            exp = W5[name]
            got = (round(wp["pred_pos_rate"], 3), wp["FN"], wp["FP"],
                   round(wp["Recall"], 3), round(wp["F1"], 3))
            ok = (got == exp or
                  (abs(got[0] - exp[0]) <= 0.0005 and got[1:] == exp[1:]
                   and abs(got[3] - exp[3]) <= 0.0005
                   and abs(got[4] - exp[4]) <= 0.0005))
            tag = "OK" if ok else "MISMATCH"
            if not ok:
                fails.append((name, got, exp))
            print(f"[W5对账] {tag} {name}: 脚本={got} W5表={exp}")
    with open(OUT / "run_workpoints.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wp_rows[0].keys()))
        w.writeheader()
        w.writerows(wp_rows)
    if fails:
        raise SystemExit(f"W5 对账不过，停: {fails}")

    # ---- 2. mil-strict 分桶（折 / 场次组 / 数据源 / 组内是否有正视频）----
    sc0 = read_predictions(RUNS / "mil-v2-150-strict")
    group_has_pos = {}
    for u, g in groups.items():
        group_has_pos[g] = group_has_pos.get(g, 0) or (labels[u] == 1)
    fold_of = {}
    d = np.load(FEATS, allow_pickle=True)
    for u, f in zip([str(x) for x in d["unit_ids"]], d["folds"]):
        fold_of[u] = int(f)

    def bucket(u):
        src = u.split("_")[0]
        return {"fold": fold_of[u], "group": groups[u], "source": src,
                "group_has_pos": int(group_has_pos[groups[u]]),
                "label": "pos" if labels[u] else "neg",
                "pred": "pos" if sc0[u] >= TH else "neg"}

    buckets = [bucket(u) for u in sc0]
    assert len(buckets) == 150, f"分桶总数 {len(buckets)} != 150"
    for dim in ["fold", "source", "group_has_pos"]:
        c = {}
        for b in buckets:
            c[b[dim]] = c.get(b[dim], 0) + 1
        s = sum(c.values())
        assert s == 150, f"{dim} 桶计数和 {s} != 150"
        print(f"[分桶对账] 按 {dim}: {dict(sorted(c.items(), key=str))} 合计 {s}")
    with open(OUT / "bucket_units.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["unit_id", *buckets[0].keys(), "score"])
        w.writeheader()
        for u, b in zip(sc0, buckets):
            w.writerow({"unit_id": u, **b, "score": sc0[u]})

    # 负视频所在场次组数（run-log 第 23 条写 10，规划代理数得 9 —— 以本脚本为准）
    neg_groups = sorted({groups[u] for u in sc0 if labels[u] == 0})
    print(f"[对账③] 37 条负视频所在场次组数 = {len(neg_groups)}")
    for g in neg_groups:
        members = [u for u in sc0 if groups[u] == g]
        npos = sum(labels[u] for u in members)
        print(f"    {g}: {len(members)} 条 (正 {npos} / 负 {len(members) - npos})")

    # ---- 3. 误报视频逐片段统计（持续高分 vs 单片段拉高）----
    units, _ = score_run(RUNS / "mil-v2-150-strict", FEATS,
                         CLIP_LABELS, "strict", None)
    fp_rows = []
    for u in units:
        if labels[u["unit_id"]] == 0 and sc0[u["unit_id"]] >= TH:
            usable = [s for s in u["clip_scores"] if s is not None]
            over = [s for s in usable if s >= TH]
            fp_rows.append({
                "unit_id": u["unit_id"], "group_id": u["group_id"],
                "unit_score": u["unit_score"],
                "n_usable": len(usable), "n_over": len(over),
                "over_ratio": round(len(over) / len(usable), 4),
                "clip_median": round(float(np.median(usable)), 4),
                "clip_p25": round(float(np.percentile(usable, 25)), 4),
                "clip_max": round(float(np.max(usable)), 4),
                "type": "持续高分" if len(over) >= 2 else "单片段拉高",
            })
    fp_rows.sort(key=lambda r: (-r["n_over"], r["unit_id"]))
    with open(OUT / "fp_clip_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fp_rows[0].keys()))
        w.writeheader()
        w.writerows(fp_rows)
    n_cont = sum(1 for r in fp_rows if r["type"] == "持续高分")
    print(f"[对账④] 误报 {len(fp_rows)} 条：整条持续高分（≥2 片段过线）{n_cont} 条，"
          f"单片段拉高 {len(fp_rows) - n_cont} 条")
    for r in fp_rows:
        print(f"    {r['unit_id']} {r['type']}: 过线 {r['n_over']}/{r['n_usable']} "
              f"最高 {r['clip_max']:.3f} 中位 {r['clip_median']:.3f}")
    print(f"[产物] {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
