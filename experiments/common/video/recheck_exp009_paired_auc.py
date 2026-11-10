# -*- coding: utf-8 -*-
"""复核 EXP-009 与冻结强标签线的配对 AUC，保留冻结线的完整精度。

运行：.venv/Scripts/python.exe experiments/common/video/recheck_exp009_paired_auc.py
只读已保存权重、特征、标签与 EXP-009 预测，不训练、不改原产物。
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_strong import StrongHead


REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "data/results/exp008"
FT = RUNS / "EXP-009-strong-v2-150-strict-ft-noaug"
FROZEN = RUNS / "strong-v2-150-strict"
FEATURES = RUNS / "unitfeats-v2-150/unit-features.npz"
CLIPS = REPO / "data/process/EXP-008-units-v2/clip-labels.npz"
SEED = 20261025
N_BOOT = 20000


def auc(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    values = np.concatenate([pos, neg])
    order = values.argsort()
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse = np.unique(values, return_inverse=True)
    sums = np.zeros(inverse.max() + 1)
    for rank, index in zip(ranks, inverse):
        sums[index] += rank
    ranks = (sums / np.bincount(inverse))[inverse]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    ft_rows = rows(FT / "predictions.csv")
    frozen_rows = rows(FROZEN / "predictions.csv")
    ids = [row["unit_id"] for row in ft_rows]
    assert len(ids) == len(set(ids)) == 150
    frozen_meta = {row["unit_id"]: row for row in frozen_rows}
    assert set(ids) == set(frozen_meta)
    assert all(row["label"] == frozen_meta[row["unit_id"]]["label"]
               and row["fold"] == frozen_meta[row["unit_id"]]["fold"]
               for row in ft_rows)

    feature_data = np.load(FEATURES, allow_pickle=True)
    clip_data = np.load(CLIPS, allow_pickle=True)
    feature_ids = [str(value) for value in feature_data["unit_ids"]]
    assert feature_ids == [str(value) for value in clip_data["unit_ids"]]
    feature_index = {unit_id: i for i, unit_id in enumerate(feature_ids)}
    assert set(ids) == set(feature_index)
    features = np.asarray(feature_data["feats"], dtype=np.float32)
    mask = np.asarray(clip_data["strict"] >= 0, dtype=bool)
    assert features.shape[:2] == mask.shape == (150, 48)
    assert mask.any(axis=1).all()

    labels = np.array([int(row["label"]) for row in ft_rows])
    folds = np.array([int(row["fold"]) for row in ft_rows])
    ft_scores = np.array([float(row["score"]) for row in ft_rows])
    frozen_scores = np.empty(len(ids), dtype=np.float64)
    with torch.no_grad():
        for fold in sorted(set(folds)):
            model = StrongHead()
            state = torch.load(FROZEN / f"strong_f{fold}.pt", map_location="cpu",
                               weights_only=True)
            model.load_state_dict(state)
            model.eval()
            for output_index in np.flatnonzero(folds == fold):
                input_index = feature_index[ids[output_index]]
                logits = model(torch.from_numpy(features[input_index])).squeeze(-1)
                scores = torch.sigmoid(logits).numpy()
                frozen_scores[output_index] = float(scores[mask[input_index]].max())

    unique_folds = sorted(set(folds))
    ft_auc = np.array([auc(labels[folds == f], ft_scores[folds == f]) for f in unique_folds])
    frozen_auc = np.array([auc(labels[folds == f], frozen_scores[folds == f]) for f in unique_folds])
    with (FROZEN / "metrics.json").open(encoding="utf-8") as stream:
        recorded = json.load(stream)
    recorded_auc = np.array([row["auc"] for row in recorded["folds"]])
    assert np.allclose(frozen_auc, recorded_auc, atol=1e-4), (frozen_auc, recorded_auc)

    generator = np.random.default_rng(SEED)
    bootstrap = np.full(N_BOOT, np.nan)
    for iteration in range(N_BOOT):
        differences = []
        for fold in unique_folds:
            members = np.flatnonzero(folds == fold)
            selected = generator.choice(members, size=len(members), replace=True)
            sampled_labels = labels[selected]
            if sampled_labels.sum() in (0, len(sampled_labels)):
                break
            differences.append(auc(sampled_labels, ft_scores[selected])
                               - auc(sampled_labels, frozen_scores[selected]))
        else:
            bootstrap[iteration] = np.mean(differences)
    valid = bootstrap[np.isfinite(bootstrap)]
    interval = np.percentile(valid, [2.5, 97.5])
    print("冻结线逐折 AUC:", frozen_auc)
    print("微调线逐折 AUC:", ft_auc)
    print("逐折均值的配对差:", float(np.mean(ft_auc - frozen_auc)))
    print(f"重抽 {N_BOOT} 次，种子 {SEED}，有效 {len(valid)} 次")
    print("95% 区间:", interval)


if __name__ == "__main__":
    main()
