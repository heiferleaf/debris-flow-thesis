"""EXP-004 冒烟:d4 外部负样本并入后的本地全链路跑通(CPU)。

依据(叶海峰 2026-09-14 对话指令"然后跑训练",验证 GPU 租用价值前的流水线检查,非正式实验):
  - 样本 = d1/d2 pass/neg 全部 + d3 pass 全部 + d4 全部 127 张 neg(叶海峰 2026-09-14 定);
  - 划分 = pHash 簇级 85/15(seed=42):d1/d2/d4 联合按簇洗牌,d4 中 0114 归簇 29(最近邻 d=8),
    3 张蒋家沟互为 d=0 共用一簇,其余 d4 自成单张簇;d3 仍按 scene 分侧;
  - 每簇上限 train ≤8 / val ≤4,等间隔抽,对所有来源、所有类别执行(修正 EXP-003 偏差);
  - 模型与超参同 EXP-003:ConvNeXt-T 冻结骨干 + linear probe,AdamW lr1e-3 wd1e-4,
    加权 CE(负权 = n_pos/n_neg),100 epoch,batch 64,取 val F1 最优;
  - 重点检查:d4(val 侧外部域)负样本误报率,以及全链路缺陷(路径/标签/划分)。

用法(仓库根):
    .venv\\Scripts\\python.exe experiments\\common\\convnext\\smoke_d4.py
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "data/results/convnext" / "EXP-004-smoke-d4"
D4 = REPO / "data/raw/dataset4-external-negatives"
SEED = 42
CAP_TRAIN, CAP_VAL = 8, 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def norm(rel: str) -> str:
    return rel.replace("\\", "/")


def load_verdicts(ds: str) -> dict[str, str]:
    out = {}
    with (REPO / "data/processed/label-review" / f"audit-{ds}.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row["verdict"] or "").strip()
            if v:
                out[norm(row["file"])] = v
    return out


def load_clusters() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with (REPO / "data/splits/phash-clusters.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["source"]][norm(row["relpath"])] = int(row["cluster_id"])
    return out


def resolve_path(source: str, rel: str) -> Path:
    cropped = REPO / "data/processed/images-cropped" / source / rel
    if cropped.is_file():
        return cropped
    # dataset4 的目录名与来源名不同(d4 实体在 dataset4-external-negatives)
    base = REPO / "data/raw/dataset4-external-negatives" if source == "dataset4" else REPO / "data/raw" / source
    return base / rel


def cap_per_cluster(items: list[dict], k: int) -> list[dict]:
    """按簇等间隔最多留 k 张(同 EXP-003 d3 抽法),保序。"""
    by_c: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        by_c[it["cluster_id"]].append(it)
    kept: list[dict] = []
    for cid, its in sorted(by_c.items()):
        its = sorted(its, key=lambda x: x["rel"])
        if len(its) <= k:
            kept += its
        else:
            idx = [round(i * (len(its) - 1) / (k - 1)) for i in range(k)]
            kept += [its[i] for i in sorted(set(idx))]
    return kept


def main() -> None:
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log_f = open(run_dir / "run-log.md", "w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log("# EXP-004 冒烟运行日志\n")
    log("依据:叶海峰 2026-09-14 指令(d4 并入后本地跑通,验证 GPU 租用价值);非正式实验。")
    log("生成命令:`.venv\\Scripts\\python.exe experiments\\common\\convnext\\smoke_d4.py`\n")

    verdicts = {ds: load_verdicts(ds) for ds in ("dataset1", "dataset2", "dataset3", "dataset4")}
    clusters = load_clusters()
    rng = random.Random(SEED)

    # ---- d1/d2:pass=正 neg=负;簇来自联合聚类 ----
    d12: list[dict] = []
    for ds in ("dataset1", "dataset2"):
        for rel, v in verdicts[ds].items():
            if v not in ("pass", "neg"):
                continue
            d12.append({"source": ds, "rel": rel, "label": 1 if v == "pass" else 0,
                        "cluster_id": clusters[ds].get(rel)})
    # ---- d4:全负;0114 归簇 29,蒋家沟 3 张共用一簇,其余单张簇 ----
    d4: list[dict] = []
    single = 200_000
    jiang_cid = single + 1  # 蒋家沟 3 张共用
    for rel, v in sorted(verdicts["dataset4"].items()):
        name = rel.split("/")[-1]
        if name.startswith("蒋家沟"):
            cid = jiang_cid
        elif name.startswith("extneg_commons_0114."):
            cid = 29  # 最近邻 d=8/10 均在簇 29(全 drop/7 neg,无正),防近重跨侧
        else:
            single += 1
            cid = single
        d4.append({"source": "dataset4", "rel": rel, "label": 0, "cluster_id": cid})
    # ---- d1/d2/d4 联合簇级 85/15 ----
    joint = d12 + d4
    cid_members: dict[int, list[dict]] = defaultdict(list)
    for it in joint:
        if it["cluster_id"] is not None:
            cid_members[it["cluster_id"]].append(it)
    all_cids = sorted(cid_members)
    rng.shuffle(all_cids)
    n_val = max(1, round(len(all_cids) * 0.15))
    val_cids = set(all_cids[:n_val])
    for it in joint:
        it["split"] = "val" if it["cluster_id"] in val_cids else "train"
    # ---- 每簇上限:train ≤8 / val ≤4,所有来源所有类别(EXP-003 偏差修正) ----
    sampled = (cap_per_cluster([i for i in joint if i["split"] == "train"], CAP_TRAIN)
               + cap_per_cluster([i for i in joint if i["split"] == "val"], CAP_VAL))
    # ---- d3:全正,scene 分侧(全簇 scene3 → val),同 EXP-003 ----
    d3_cid_members: dict[int, list[str]] = defaultdict(list)
    for rel, cid in clusters["dataset3"].items():
        d3_cid_members[cid].append(rel)
    d3_by_cside: dict[tuple[str, int], list[str]] = defaultdict(list)
    for rel in verdicts["dataset3"]:
        cid = clusters["dataset3"].get(rel)
        side = ("val" if all(r.startswith("scene3/") for r in d3_cid_members[cid]) else "train") if cid is not None else ("val" if rel.startswith("scene3/") else "train")
        d3_by_cside[(side, cid if cid is not None else -1)].append(rel)
    for (side, cid), rels in sorted(d3_by_cside.items()):
        its = [{"source": "dataset3", "rel": r, "label": 1, "cluster_id": cid, "split": side}
               for r in sorted(rels)]
        sampled += cap_per_cluster(its, CAP_VAL if side == "val" else CAP_TRAIN)

    for it in sampled:
        it["path"] = resolve_path(it["source"], it["rel"])

    with (run_dir / "sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "source", "rel", "cluster_id", "label", "path"])
        w.writeheader()
        w.writerows({k: it[k] for k in ("split", "source", "rel", "cluster_id", "label", "path")} for it in sampled)

    tr = [s for s in sampled if s["split"] == "train"]
    va = [s for s in sampled if s["split"] == "val"]
    stats = {
        "total": len(sampled), "train": len(tr), "val": len(va),
        "train_pos": sum(s["label"] for s in tr), "train_neg": len(tr) - sum(s["label"] for s in tr),
        "val_pos": sum(s["label"] for s in va), "val_neg": len(va) - sum(s["label"] for s in va),
        "per_source": dict(Counter(s["source"] for s in sampled)),
        "val_per_source": dict(Counter(s["source"] for s in va)),
        "missing_files": [str(s["path"]) for s in sampled if not s["path"].is_file()],
        "seed": SEED,
    }
    log("抽样统计: " + json.dumps(stats, ensure_ascii=False))
    if stats["missing_files"]:
        raise SystemExit(f"缺文件 {len(stats['missing_files'])} 张,中止")

    # ---- 特征(CPU,冻结骨干,直接拉伸 224,同 EXP-003) ----
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(model.classifier[0], model.classifier[1])
    model.eval()
    pre = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def extract(items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        feats = np.zeros((len(items), 768), dtype=np.float32)
        labs = np.zeros(len(items), dtype=np.int64)
        batch, batch_lab = [], []
        with torch.no_grad():
            for i, it in enumerate(items):
                batch.append(pre(Image.open(it["path"]).convert("RGB")))
                batch_lab.append(it["label"])
                if len(batch) == 16 or i == len(items) - 1:
                    out = model(torch.stack(batch)).numpy()
                    feats[i - len(batch) + 1:i + 1] = out
                    labs[i - len(batch) + 1:i + 1] = batch_lab
                    batch, batch_lab = [], []
                if (i + 1) % 400 == 0:
                    log(f"  特征 {i + 1}/{len(items)}  已用 {(time.perf_counter() - t0) / 60:.1f} min")
        return feats, labs

    log(f"[特征] train {len(tr)} + val {len(va)}(CPU)")
    Xtr, ytr = extract(tr)
    Xva, yva = extract(va)
    np.savez_compressed(run_dir / "features.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    log(f"[特征] 完成,累计 {(time.perf_counter() - t0) / 60:.1f} min")

    # ---- linear probe(同 EXP-003 超参) ----
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    w_neg = n_pos / max(n_neg, 1)
    weight = torch.tensor([w_neg, 1.0])
    log(f"[probe] train 正 {n_pos} / 负 {n_neg},负类权重 {w_neg:.2f}")
    head = nn.Linear(768, 2)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=weight)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    gen = torch.Generator().manual_seed(SEED)

    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                "fpr_on_neg": round(fp / (fp + tn), 4) if fp + tn else 0.0}

    best = {"f1": -1.0, "epoch": -1, "state": None}
    for epoch in range(1, 101):
        head.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        for lo in range(0, len(perm), 64):
            idx = perm[lo:lo + 64]
            opt.zero_grad()
            loss = lossf(head(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            m = metrics(yva, head(Xva_t).argmax(1).numpy())
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "epoch": epoch,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}}
        if epoch % 20 == 0:
            log(f"  epoch {epoch:3d}  val F1={m['f1']:.3f} (best {best['f1']:.3f} @ep{best['epoch']})")

    head.load_state_dict(best["state"])
    head.eval()
    torch.save(head.state_dict(), run_dir / "linear_head.pt")
    with torch.no_grad():
        prob_va = torch.softmax(head(Xva_t), 1)[:, 1].numpy()
        pred_va = (prob_va >= 0.5).astype(int)
    val_m = metrics(yva, pred_va)
    p_base = (yva == 1).mean()

    # 重点:d4 外部域 val 侧负样本误报
    d4_val = [i for i, s in enumerate(va) if s["source"] == "dataset4"]
    d4_m = metrics(yva[d4_val], pred_va[d4_val]) if d4_val else {}
    jiang_idx = [i for i in d4_val if "蒋家沟" in va[i]["rel"]]
    result = {
        "spec": "EXP-004-smoke(冒烟,非正式)", "run_dir": str(run_dir),
        "sample": stats, "class_weight_neg": round(w_neg, 3),
        "best_epoch": best["epoch"], "best_val_f1": best["f1"], "val": val_m,
        "baseline_all_positive_f1": round(2 * p_base / (1 + p_base), 4),
        "d4_val_neg": {**d4_m,
                       "prob_mean": round(float(prob_va[d4_val].mean()), 4),
                       "prob_max": round(float(prob_va[d4_val].max()), 4)} if d4_val else None,
        "jiangjiagou3_prob": [round(float(prob_va[i]), 4) for i in jiang_idx],
        "elapsed_min": round((time.perf_counter() - t0) / 60, 1),
        "defects_found": [],
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log("结果: " + json.dumps(result, ensure_ascii=False, indent=2))
    log(f"[完成] 总耗时 {result['elapsed_min']} min,产物在 {run_dir}")
    log_f.close()


if __name__ == "__main__":
    main()
