"""EXP-005 单帧正式基线:全量数据 + 簇级 70/15/15 + 学习曲线(依据 EXP-005.md,叶海峰 2026-09-14 签字)。

用法(仓库根):
    .venv\\Scripts\\python.exe experiments\\common\\convnext\\run_full.py
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
RUNS = REPO / "data" / "results" / "convnext"
SEED = 42
CAPS = {"train": 16, "val": 8, "test": 8}
FRACS = [0.25, 0.5, 0.75, 1.0]
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
    base = REPO / "data/raw/dataset4-external-negatives" if source == "dataset4" else REPO / "data/raw" / source
    return base / rel


def cap_per_cluster(items: list[dict], k: int) -> list[dict]:
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

    log("# EXP-005 正式基线运行日志\n")
    log("依据: EXP-005.md(叶海峰 2026-09-14 签字'可以开始训练')")
    log("命令: `.venv\\Scripts\\python.exe experiments\\common\\convnext\\run_full.py`\n")

    verdicts = {ds: load_verdicts(ds) for ds in ("dataset1", "dataset2", "dataset3", "dataset4")}
    clusters = load_clusters()
    rng = random.Random(SEED)

    samples: list[dict] = []
    for ds in ("dataset1", "dataset2"):
        for rel, v in verdicts[ds].items():
            if v in ("pass", "neg"):
                samples.append({"source": ds, "rel": rel, "label": 1 if v == "pass" else 0,
                                "cluster_id": clusters[ds].get(rel)})
    for rel, v in sorted(verdicts["dataset4"].items()):
        name = rel.split("/")[-1]
        if name.startswith("蒋家沟"):
            cid = 200_001
        elif name.startswith("extneg_commons_0114."):
            cid = 29
        else:
            cid = None  # 非特殊单张簇,后面按文件名序统一编号
        samples.append({"source": "dataset4", "rel": rel, "label": 0, "cluster_id": cid})
    for rel, v in verdicts["dataset3"].items():
        if v == "pass":
            samples.append({"source": "dataset3", "rel": rel, "label": 1,
                            "cluster_id": clusters["dataset3"].get(rel)})
    # d4 非特殊单张簇统一编号(与顺序无关的稳定编号:按文件名序)
    d4_others = sorted(s["rel"] for s in samples
                       if s["source"] == "dataset4" and s["cluster_id"] is None)
    assign = {rel: 300_000 + i for i, rel in enumerate(d4_others)}
    for s in samples:
        if s["source"] == "dataset4" and s["cluster_id"] is None:
            s["cluster_id"] = assign[s["rel"]]

    # 联合簇级 70/15/15
    cid_members: dict[int, list[dict]] = defaultdict(list)
    for it in samples:
        if it["cluster_id"] is not None:
            cid_members[it["cluster_id"]].append(it)
    cids = sorted(cid_members)
    rng.shuffle(cids)
    n_val, n_test = round(len(cids) * 0.15), round(len(cids) * 0.15)
    val_cids = set(cids[:n_val])
    test_cids = set(cids[n_val:n_val + n_test])
    for it in samples:
        cid = it["cluster_id"]
        it["split"] = "val" if cid in val_cids else ("test" if cid in test_cids else "train")
    sampled: list[dict] = []
    for split in ("train", "val", "test"):
        sampled += cap_per_cluster([s for s in samples if s["split"] == split], CAPS[split])
    for s in sampled:
        s["path"] = resolve_path(s["source"], s["rel"])

    with (run_dir / "sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "source", "rel", "cluster_id", "label", "path"])
        w.writeheader()
        w.writerows({k: s[k] for k in ("split", "source", "rel", "cluster_id", "label", "path")} for s in sampled)

    tr = [s for s in sampled if s["split"] == "train"]
    va = [s for s in sampled if s["split"] == "val"]
    te = [s for s in sampled if s["split"] == "test"]
    stats = {
        "total": len(sampled),
        "train": len(tr), "val": len(va), "test": len(te),
        "train_pos": sum(s["label"] for s in tr), "train_neg": len(tr) - sum(s["label"] for s in tr),
        "val_pos": sum(s["label"] for s in va), "val_neg": len(va) - sum(s["label"] for s in va),
        "test_pos": sum(s["label"] for s in te), "test_neg": len(te) - sum(s["label"] for s in te),
        "clusters": {"train": len({s["cluster_id"] for s in tr}), "val": len(val_cids), "test": len(test_cids)},
        "missing_files": [str(s["path"]) for s in sampled if not s["path"].is_file()],
        "seed": SEED, "caps": CAPS,
    }
    log("抽样统计: " + json.dumps(stats, ensure_ascii=False))
    if stats["missing_files"]:
        raise SystemExit(f"缺文件 {len(stats['missing_files'])} 张,中止")

    # 特征一次性抽全(按 split 存),学习曲线复用
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
        batch, blab = [], []
        with torch.no_grad():
            for i, it in enumerate(items):
                batch.append(pre(Image.open(it["path"]).convert("RGB")))
                blab.append(it["label"])
                if len(batch) == 16 or i == len(items) - 1:
                    out = model(torch.stack(batch)).numpy()
                    feats[i - len(batch) + 1:i + 1] = out
                    labs[i - len(batch) + 1:i + 1] = blab
                    batch, blab = [], []
                if (i + 1) % 1000 == 0:
                    log(f"  特征 {i + 1}/{len(items)}  已用 {(time.perf_counter() - t0) / 60:.1f} min")
        return feats, labs

    log(f"[特征] train {len(tr)} + val {len(va)} + test {len(te)}(CPU)")
    splits_arr = {}
    for name, items in (("train", tr), ("val", va), ("test", te)):
        X, y = extract(items)
        splits_arr[name] = (X, y)
        np.savez_compressed(run_dir / f"features-{name}.npz", X=X, y=y)
    log(f"[特征] 完成,累计 {(time.perf_counter() - t0) / 60:.1f} min")

    Xtr, ytr = splits_arr["train"]
    Xva, yva = splits_arr["val"]
    Xte, yte = splits_arr["test"]
    tr_cids = np.array([s["cluster_id"] for s in tr])

    def train_probe(X: np.ndarray, y: np.ndarray, epochs: int = 200) -> nn.Linear:
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        weight = torch.tensor([n_pos / max(n_neg, 1), 1.0])
        head = nn.Linear(768, 2)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=weight)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        gen = torch.Generator().manual_seed(SEED)
        best = {"f1": -1.0, "state": None}
        for _ in range(epochs):
            head.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for lo in range(0, len(perm), 64):
                idx = perm[lo:lo + 64]
                opt.zero_grad()
                lossf(head(Xt[idx]), yt[idx]).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                m = metrics(yva, head(Xva_t := torch.from_numpy(Xva)).argmax(1).numpy())
            if m["f1"] > best["f1"]:
                best = {"f1": m["f1"], "state": {k: v.clone() for k, v in head.state_dict().items()}}
        head.load_state_dict(best["state"])
        head.eval()
        return head

    # 学习曲线(训练簇 25/50/75/100%)
    curve = []
    unique_cids = sorted(set(tr_cids.tolist()))
    rng2 = random.Random(SEED)
    for frac in FRACS:
        picked = set(unique_cids)
        if frac < 1.0:
            k = max(1, round(len(unique_cids) * frac))
            picked = set(rng2.sample(unique_cids, k))
        mask = np.array([c in picked for c in tr_cids])
        log(f"[曲线 {int(frac*100)}%] train 簇 {len(picked)}/{len(unique_cids)},样本 {int(mask.sum())}")
        head = train_probe(Xtr[mask], ytr[mask])
        with torch.no_grad():
            m = metrics(yva, head(torch.from_numpy(Xva)).argmax(1).numpy())
        curve.append({"frac": frac, "train_clusters": len(picked), "train_samples": int(mask.sum()),
                      "val": m})
        log(f"[曲线 {int(frac*100)}%] val F1={m['f1']:.4f}")

    # 全量最终模型 + test
    head_full = train_probe(Xtr, ytr)
    torch.save(head_full.state_dict(), run_dir / "linear_head.pt")
    with torch.no_grad():
        prob_te = torch.softmax(head_full(torch.from_numpy(Xte)), 1)[:, 1].numpy()
        pred_te = (prob_te >= 0.5).astype(int)
    test_m = metrics(yte, pred_te)
    per_src = {}
    srcs = [s["source"] for s in te]
    for src in sorted(set(srcs)):
        idx = [i for i, s in enumerate(srcs) if s == src]
        per_src[src] = metrics(yte[idx], pred_te[idx])
        if src == "dataset4":
            per_src[src]["prob_mean"] = round(float(prob_te[idx].mean()), 4)
            per_src[src]["prob_max"] = round(float(prob_te[idx].max()), 4)
    p_base = (yte == 1).mean()

    # 学习曲线收敛判据:相邻档 F1 增益
    gains = [round(curve[i + 1]["val"]["f1"] - curve[i]["val"]["f1"], 4) for i in range(len(curve) - 1)]

    result = {
        "spec": "EXP-005", "run_dir": str(run_dir),
        "sample": stats,
        "learning_curve": curve, "curve_gains": gains,
        "saturated_hint": "翻倍增益<0.005 视为接近饱和(学习手册 2.6)" ,
        "test": test_m, "test_per_source": per_src,
        "baseline_all_positive_f1": round(2 * p_base / (1 + p_base), 4),
        "f1_minus_baseline": round(test_m["f1"] - 2 * p_base / (1 + p_base), 4),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 1),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log("结果: " + json.dumps(result, ensure_ascii=False, indent=2))
    log(f"[完成] 总耗时 {result['elapsed_min']} min")
    log_f.close()


if __name__ == "__main__":
    main()
