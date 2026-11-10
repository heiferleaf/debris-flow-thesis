"""EXP-003 小规模可行性试点:抽样 → ConvNeXt-T 抽特征 → linear probe → 指标。

依据 experiments/common/convnext/EXP-003.md(叶海峰 2026-09-13 签字):
  - 正样本 = d1 pass 全部 + d2 pass 全部 + d3 按簇每簇 ≤8 抽(来源分层欠采样);
  - 负样本 = d1 neg + d2 neg 全部;损失按类加权(负类权重 = n_pos/n_neg);
  - 划分 = pHash 簇级 85/15(d1/d2 联合按簇随机,seed=42;d3 按 scene:全簇 scene3 → val,
    否则 train,防跨 scene 泄漏);d3 训练侧每簇 ≤8、val 侧 ≤4,等间隔抽;
  - 填充图用 data/processed/images-cropped/ 裁剪副本(存在即替换),原图只读;
  - 预算 90 分钟;本脚本只做 linear probe,--finetune-stage4 另跑对比。

用法(仓库根):
    .venv\\Scripts\\python.exe experiments\\common\\convnext\\run_pilot.py
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
RUNS = REPO / "data/results/convnext" / "EXP-003-pilot"
SEED = 42
K_D3_TRAIN, K_D3_VAL = 8, 4
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
    """source -> relpath -> cluster_id"""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with (REPO / "data/splits/phash-clusters.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["source"]][norm(row["relpath"])] = int(row["cluster_id"])
    return out


def resolve_path(source: str, rel: str) -> tuple[Path, bool]:
    """优先用裁剪副本(填充图),否则原图。返回 (路径, 是否用裁剪)。"""
    cropped = REPO / "data/processed/images-cropped" / source / rel
    if cropped.is_file():
        return cropped, True
    return REPO / "data/raw" / source / rel, False


def scene3_of(rel: str) -> bool:
    return rel.startswith("scene3/")


def main() -> None:
    run_dir = RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    rng = random.Random(SEED)

    verdicts = {ds: load_verdicts(ds) for ds in ("dataset1", "dataset2", "dataset3")}
    clusters = load_clusters()

    # ---- d1/d2:pass=正 neg=负,簇级 85/15(跨库簇天然同侧) ----
    d12: list[dict] = []
    for ds in ("dataset1", "dataset2"):
        for rel, v in verdicts[ds].items():
            if v not in ("pass", "neg"):
                continue
            d12.append({"source": ds, "rel": rel, "label": 1 if v == "pass" else 0,
                        "cluster_id": clusters[ds].get(rel)})
    cid_members: dict[int, list[dict]] = defaultdict(list)
    for it in d12:
        if it["cluster_id"] is not None:
            cid_members[it["cluster_id"]].append(it)
    all_cids = sorted(cid_members)
    rng.shuffle(all_cids)
    n_val = max(1, round(len(all_cids) * 0.15))
    val_cids = set(all_cids[:n_val])
    for it in d12:
        cid = it["cluster_id"]
        it["split"] = "val" if (cid is not None and cid in val_cids) else "train"

    # ---- d3:全正;按 scene 分侧(全簇 scene3 → val,否则 train),每簇限量等间隔抽 ----
    d3_cid_members: dict[int, list[str]] = defaultdict(list)
    for rel, cid in clusters["dataset3"].items():
        d3_cid_members[cid].append(rel)
    d3_by_cside: dict[tuple[str, int], list[str]] = defaultdict(list)
    for rel in verdicts["dataset3"]:
        cid = clusters["dataset3"].get(rel)
        if cid is None:  # 无簇信息按 scene 兜底
            side = "val" if scene3_of(rel) else "train"
        else:
            # 簇 side 由成员 scene 决定:仅当该簇在 d3 内的全部成员都是 scene3 才 val
            side = "val" if all(scene3_of(r) for r in d3_cid_members[cid]) else "train"
        d3_by_cside[(side, cid if cid is not None else -1)].append(rel)
    d3: list[dict] = []
    for (side, cid), rels in sorted(d3_by_cside.items()):
        rels = sorted(rels)
        k = K_D3_TRAIN if side == "train" else K_D3_VAL
        if len(rels) <= k:
            picked = rels
        else:  # 等间隔抽 k 张
            idx = [round(i * (len(rels) - 1) / (k - 1)) for i in range(k)]
            picked = [rels[i] for i in sorted(set(idx))]
        d3 += [{"source": "dataset3", "rel": r, "label": 1, "cluster_id": cid, "split": side}
               for r in picked]

    samples = d12 + d3
    for it in samples:
        it["path"], it["used_crop"] = resolve_path(it["source"], it["rel"])

    with (run_dir / "sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "source", "rel", "cluster_id", "label", "used_crop", "path"])
        w.writeheader()
        w.writerows({"split": it["split"], "source": it["source"], "rel": it["rel"],
                     "cluster_id": it["cluster_id"], "label": it["label"],
                     "used_crop": it["used_crop"], "path": str(it["path"])} for it in samples)

    tr = [s for s in samples if s["split"] == "train"]
    va = [s for s in samples if s["split"] == "val"]
    stats = {
        "total": len(samples), "train": len(tr), "val": len(va),
        "train_pos": sum(s["label"] for s in tr), "train_neg": len(tr) - sum(s["label"] for s in tr),
        "val_pos": sum(s["label"] for s in va), "val_neg": len(va) - sum(s["label"] for s in va),
        "used_crop": sum(s["used_crop"] for s in samples),
        "missing_files": [str(s["path"]) for s in samples if not s["path"].is_file()],
        "seed": SEED,
    }
    print("抽样统计:", json.dumps(stats, ensure_ascii=False), flush=True)
    if stats["missing_files"]:
        raise SystemExit(f"缺文件 {len(stats['missing_files'])} 张,中止(明细见 sample.csv)")
    with (run_dir / "sample_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- 抽特征(冻结 backbone,直接拉伸 224) ----
    dev = "cpu"
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(model.classifier[0], model.classifier[1])  # 去掉 1000 类头,留 LayerNorm+Flatten
    model.eval().to(dev)
    pre = transforms.Compose([
        transforms.Resize((224, 224)),  # 直接拉伸,不用 letterbox(EXP-003 卡)
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def extract(split_items: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        feats = np.zeros((len(split_items), 768), dtype=np.float32)
        labs = np.zeros(len(split_items), dtype=np.int64)
        batch, batch_lab = [], []
        with torch.no_grad():
            for i, it in enumerate(split_items):
                batch.append(pre(Image.open(it["path"]).convert("RGB")))
                batch_lab.append(it["label"])
                if len(batch) == 16 or i == len(split_items) - 1:
                    x = torch.stack(batch)
                    out = model(x).numpy()
                    feats[i - len(batch) + 1:i + 1] = out
                    labs[i - len(batch) + 1:i + 1] = batch_lab
                    batch, batch_lab = [], []
                if (i + 1) % 200 == 0:
                    el = time.perf_counter() - t_start
                    print(f"  特征 {i + 1}/{len(split_items)}  已用 {el / 60:.1f} min", flush=True)
        return feats, labs

    print(f"[特征] train {len(tr)} 张 + val {len(va)} 张(CPU)…", flush=True)
    Xtr, ytr = extract(tr)
    Xva, yva = extract(va)
    np.savez_compressed(run_dir / "features.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    print(f"[特征] 完成,累计 {((time.perf_counter() - t_start) / 60):.1f} min", flush=True)

    # ---- linear probe(类加权 CE) ----
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    w_neg = n_pos / max(n_neg, 1)
    weight = torch.tensor([w_neg, 1.0])
    print(f"[probe] train 正 {n_pos} / 负 {n_neg},负类权重 {w_neg:.2f}", flush=True)
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
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                "fpr_on_neg": round(fp / (fp + tn), 4) if fp + tn else 0.0}

    best = {"f1": -1.0, "epoch": -1, "state": None, "val": None}
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
            pred = head(Xva_t).argmax(1).numpy()
        m = metrics(yva, pred)
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "epoch": epoch,
                    "state": {k: v.clone() for k, v in head.state_dict().items()}, "val": m}
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val F1={m['f1']:.3f} (best {best['f1']:.3f} @ep{best['epoch']})", flush=True)

    head.load_state_dict(best["state"])
    head.eval()
    torch.save(head.state_dict(), run_dir / "linear_head.pt")
    with torch.no_grad():
        pred_va = head(Xva_t).argmax(1).numpy()
        pred_tr = head(Xtr_t).argmax(1).numpy()
    val_m = metrics(yva, pred_va)
    # 卡片要求:误报率在全部 799 负样本上单独报。
    # 注意 train 侧负样本(614)参与过训练,数字偏乐观;干净的口径是 val 负样本(185)那一份。
    all_y = np.concatenate([ytr, yva])
    all_p = np.concatenate([pred_tr, pred_va])
    all_neg_m = metrics(all_y, all_p)
    neg_m = {
        "val_neg_total": int((yva == 0).sum()),
        "false_positives": val_m["fp"],
        "fpr": val_m["fpr_on_neg"],
        "all_neg_total": int((all_y == 0).sum()),
        "all_neg_false_positives": all_neg_m["fp"],
        "all_neg_fpr_incl_train_negs": all_neg_m["fpr_on_neg"],
    }

    p_base = (yva == 1).mean()
    base_f1 = 2 * p_base / (1 + p_base)  # 全猜正基线
    result = {
        "spec": "EXP-003", "run_dir": str(run_dir),
        "sample": stats, "class_weight_neg": round(w_neg, 3),
        "best_epoch": best["epoch"], "val": val_m, "val_neg_report": neg_m,
        "baseline_all_positive_f1": round(base_f1, 4),
        "f1_minus_baseline": round(val_m["f1"] - base_f1, 4),
        "elapsed_min": round((time.perf_counter() - t_start) / 60, 1),
        "decision_rule": "F1 - 全猜正基线 >= 0.15 且跨来源有效 -> 复现线可参考",
    }
    # 按来源拆 val 指标(卡上要求"跨来源有效")
    src_of_val = [s["source"] for s in va]
    for src in sorted(set(src_of_val)):
        idx = [i for i, s in enumerate(src_of_val) if s == src]
        if idx:
            result[f"val_only_{src}"] = metrics(yva[idx], pred_va[idx])
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"[完成] 总耗时 {result['elapsed_min']} min,产物在 {run_dir}", flush=True)


if __name__ == "__main__":
    main()
