# -*- coding: utf-8 -*-
"""EXP-006 特征缓存:冻结 ConvNeXt-T(与 EXP-005 同权重)对试点段清单的帧提 768 维特征,
顺带用 EXP-005 linear_head 打每帧正类概率,一次存盘。CPU 即可。

用法(仓库根,venv): python experiments/common/temporal/cache_features.py [--batch 32]
输入:adapters.manifest_path()(段清单)
输出:adapters.out_cache_path()(cache.npz: feats/probs/row_id/labels,权重/缓存不进 data/) 依据: EXP-006 卡。
"""
import argparse
import csv
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:  # 包方式（python -m experiments.common.temporal.cache_features）
    from experiments.common.temporal import adapters
except ImportError:  # 文件方式（python experiments/common/temporal/cache_features.py）
    import adapters

MANIFEST = adapters.manifest_path()
RUN = adapters.out_cache_path().parent      # runs/EXP-006-pilot/（权重不进 data/）
CACHE_OUT = adapters.out_cache_path()
RUN_EXP5 = adapters.exp5_weights_dir()      # EXP-005 冻结权重（旧位置 fallback）

pre = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(42)

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    print(f"清单段数 {len(rows)}")

    backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(backbone.classifier[0], backbone.classifier[1])  # 768 维
    backbone.eval()
    head = nn.Linear(768, 2)
    head.load_state_dict(torch.load(RUN_EXP5 / "linear_head.pt", map_location="cpu"))
    head.eval()

    feats, probs, row_id, labels = [], [], [], []
    t0, n_done, n_fail = time.time(), 0, 0
    batch_imgs, batch_meta = [], []

    def flush():
        nonlocal n_done, n_fail
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs)
        with torch.no_grad():
            f = backbone(x)
            p = torch.softmax(head(f), 1)[:, 1]
        feats.append(f.numpy())
        probs.append(p.numpy())
        row_id.extend(batch_meta)
        n_done += len(batch_imgs)
        batch_imgs.clear(), batch_meta.clear()

    for ri, r in enumerate(rows):
        labels.append(1 if r["kind"] == "pos" else 0)
        for fn in r["frames"].split(";"):
            fp = adapters.resolve_frame(r["source"], fn)  # source→videoN 映射 + 帧根在适配器内
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                n_fail += 1
                if n_fail == 1:
                    print(f"[读图失败样例] {fp}")
                continue
            batch_imgs.append(pre(img))
            batch_meta.append(ri)
            if len(batch_imgs) >= args.batch:
                flush()
    flush()

    F = np.concatenate(feats).astype(np.float32)
    P = np.concatenate(probs).astype(np.float32)
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_OUT, feats=F, probs=P,
                        row_id=np.array(row_id), labels=np.array(labels),
                        n_rows=len(rows))
    dt = time.time() - t0
    print(f"缓存完成: {F.shape[0]} 帧 x {F.shape[1]} 维, 读图失败 {n_fail}, 耗时 {dt/60:.1f} min -> {CACHE_OUT}")
    print(f"正帧 {int(np.array(labels)[row_id].sum())} (按帧归属)")


if __name__ == "__main__":
    main()
