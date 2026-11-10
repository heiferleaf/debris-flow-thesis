# -*- coding: utf-8 -*-
"""EXP-008 R2：冻结 r2plus1d_18 提单位特征（单位 = 48 片段 × 16 帧 → 48 × 512 维）。

骨干冻结（不参与训练参数更新，偏差 D5：以 r2plus1d_18 替代 C3D）；每片段一
16×112×112 张量经骨干得 512 维特征（去掉 fc 分类头）。

输入：data/process/EXP-008-units/unit-manifest.csv + frames/<unit_id>/clip*.jpg
输出：data/results/exp008/<tag>/unit-features.npz
  keys: unit_ids (str), labels (int 0/1), groups (str), folds (int),
        feats ([n_unit, 48, 512])

用法（仓库根，venv）：
  python experiments/common/video/extract_feats.py --tag unitfeats-smoke --units 4
  python experiments/common/video/extract_feats.py --tag unitfeats-v2-150
环境变量：DF_RUNS_ROOT 覆盖 runs 根。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.common.video.artifact_paths import recorded_path
from experiments.paths import resolve_path  # noqa: E402
RUNS_ROOT = resolve_path(os.environ.get("DF_RUNS_ROOT") or recorded_path(REPO / "data/results/exp001", repo=REPO))
UNITS_ROOT = Path(os.environ.get("DF_UNITS_ROOT") or recorded_path(REPO / "data/process/EXP-008-units", repo=REPO))
MANIFEST = UNITS_ROOT / "unit-manifest.csv"

# r2plus1d_18 输入变换（Kinetics400 口径）
TF = transforms.Compose([
    transforms.ToTensor(),            # HWC uint8 -> CHW float [0,1]
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989]),
])


def read_manifest():
    return list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig", newline="")))


def load_clip(unit_dir: Path, ci: int, clip_len: int) -> torch.Tensor:
    """读一片段 16 帧 jpg -> [3,T,H,W]（112×112）。"""
    frames = []
    for fi in range(clip_len):
        p = unit_dir / f"clip{ci:02d}_f{fi:02d}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"缺失帧 {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        frames.append(TF(img))
    return torch.stack(frames, dim=1)          # [3, T, H, W]


@torch.inference_mode()
def extract_unit_features(model, row: dict) -> np.ndarray:
    """整单位：48 片段逐个前向 -> [48, 512]。"""
    unit_dir = UNITS_ROOT / "frames" / row["unit_id"]
    n_clips = int(row["n_clips"])
    clip_len = int(row["clip_len"])
    out = np.zeros((n_clips, 512), dtype=np.float32)
    for ci in range(n_clips):
        x = load_clip(unit_dir, ci, clip_len).unsqueeze(0)   # [1,3,16,112,112]
        out[ci] = model(x).squeeze(0).cpu().numpy()
    return out


def make_backbone() -> torch.nn.Module:
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()             # 取池化后 512 维，去分类头
    model.eval()
    model = model.to("cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser(description="EXP-008 R2 冻结骨干提单位特征")
    ap.add_argument("--tag", default="unitfeats-smoke", help="输出目录标签")
    ap.add_argument("--units", type=int, default=0, help="只处理前 N 个单位（冒烟）")
    args = ap.parse_args()

    rows = read_manifest()
    if args.units:
        rows = rows[:args.units]
    print(f"[R2] 将提取 {len(rows)} 单位特征")

    model = make_backbone()
    unit_ids, feats, labels, groups, folds = [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        t = time.time()
        f = extract_unit_features(model, r)
        unit_ids.append(r["unit_id"])
        feats.append(f)
        labels.append(1 if r["label"] == "pos" else 0)
        groups.append(r["group_id"])
        folds.append(int(r["fold"]))
        print(f"  [{i}/{len(rows)}] {r['unit_id']} {r['label']} 特征{f.shape} "
              f"{time.time()-t:.1f}s（累计 {time.time()-t0:.1f}s）")

    out_dir = RUNS_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "unit-features.npz",
        unit_ids=np.array(unit_ids), labels=np.array(labels),
        groups=np.array(groups), folds=np.array(folds),
        feats=np.stack(feats, axis=0))       # [n_unit, 48, 512]
    print(f"[R2] 完成，耗时 {time.time()-t0:.1f}s")
    print(f"[R2] -> {out_dir / 'unit-features.npz'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
