# -*- coding: utf-8 -*-
"""把「加长间隔」实验的两套设定对齐到同一个池子，产出两边共用的元数据。

背景：0.5 秒间隔下部分视频凑不满 48 个片段，必须整条排除（research-auditor
2026-10-25 方案甲）。这样一来两套设定要比，就得在同一批视频、同一套冻结的折
划分上各训一次，否则「池子变了」会和「间隔变了」混在一起。

本脚本以 v3 清单为池子的唯一依据，产出四样东西：
  1. meta-<N>.npz        —— unit_ids / labels / folds，折划分原样抄自冻结版
                             特征 npz，不重新划分（只按池子筛行）。
  2. clip-labels-v2-<N>.npz —— 原间隔那套设定的片段标签，同样只筛行。
  3. clip-labels-v3-<N>.npz —— 新间隔那套设定的片段标签，同样只筛行。
  4. EXP-008-units-folds-<N>.json —— 池内的逐折条数登记，供开跑前核对表对照。
     它不改变任何一条视频的折归属，只是把 150 条的登记表按池子裁一遍。

生成命令：
  python experiments/common/video/make_pool_npz.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
V2 = REPO / "data/process/EXP-008-units-v2"
V3 = REPO / "data/process/EXP-008-units-v3-gap05"
META150 = REPO / "data/results/exp008/unitfeats-v2-150/unit-features.npz"
OUT = REPO / "data/process/EXP-008-units-pool128"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v3_rows = list(csv.DictReader(open(V3 / "unit-manifest.csv", encoding="utf-8-sig")))
    pool = [r["unit_id"] for r in v3_rows]
    meta = np.load(META150, allow_pickle=True)
    frozen_ids = [str(x) for x in meta["unit_ids"]]
    assert set(pool) <= set(frozen_ids), "池里有冻结版没有的视频"
    idx = [frozen_ids.index(u) for u in pool]          # 保持冻结版的相对行序

    OUT.mkdir(parents=True, exist_ok=True)
    n = len(pool)
    labels = meta["labels"].astype(np.int8)[idx]
    folds = meta["folds"].astype(np.int8)[idx]

    # 折归属与标签的第二来源核对（research-auditor 2026-10-25 W5）。
    # 下面写出的 folds 登记若直接取自特征 npz，而 train_ft.py 又拿训练输入的
    # folds 去和这份登记比，两边同源——行序一旦错位会一起错、核对照样全绿，是
    # 自证循环。所以先与 v2 的 unit-manifest.csv 逐条比一次（那份清单是独立生成的）。
    _LAB = {"pos": 1, "neg": 0}
    with open(V2 / "unit-manifest.csv", encoding="utf-8-sig", newline="") as fh:
        v2_manifest = {r["unit_id"]: r for r in csv.DictReader(fh)}
    bad = []
    for u, lab, fo in zip(pool, labels, folds):
        r = v2_manifest.get(u)
        if r is None:
            bad.append(f"{u}：v2 清单里没有这条")
        elif int(r["fold"]) != int(fo) or _LAB[r["label"]] != int(lab):
            bad.append(f"{u}：清单 fold={r['fold']} label={r['label']}，"
                       f"特征 npz fold={fo} label={lab}")
    if bad:
        raise SystemExit("池内视频的折归属或标签与 unit-manifest.csv 不一致，"
                         "拒绝写登记：\n  " + "\n  ".join(bad[:10]))
    print(f"[pool] 折归属与标签已与 unit-manifest.csv 逐条核对：{n} 条全对")

    np.savez_compressed(OUT / f"meta-{n}.npz", unit_ids=np.array(pool),
                        labels=labels, folds=folds)

    for tag, src in [("v2", V2 / "clip-labels.npz"), ("v3", V3 / "clip-labels.npz")]:
        cl = np.load(src, allow_pickle=True)
        ids = [str(x) for x in cl["unit_ids"]]
        keep = [ids.index(u) for u in pool]            # 按池子顺序重排
        np.savez_compressed(OUT / f"clip-labels-{tag}-{n}.npz",
                            unit_ids=np.array(pool),
                            strict=cl["strict"][keep], loose=cl["loose"][keep])
        m = cl["strict"][keep]
        print(f"[pool] {tag} strict：pos {int((m==1).sum())} / neg {int((m==0).sum())} "
              f"/ 不可用 {int((m==-1).sum())}（可用 {100*(m>=0).mean():.1f}%）")

    cnt = Counter(int(f) for f in folds)
    reg = {"note": f"EXP-008 池 {n} 条的逐折登记，折归属原样抄自 "
                   f"EXP-008-units-folds.json，未重新划分",
           "folds": [{"fold": f, "val_units": int(cnt[f]),
                      "val_pos": int(((folds == f) & (labels == 1)).sum()),
                      "val_neg": int(((folds == f) & (labels == 0)).sum())}
                     for f in sorted(cnt)]}
    (REPO / f"data/metadata/splits/EXP-008-units-folds-{n}.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[pool] 池 {n} 条（正 {int((labels==1).sum())} / 负 {int((labels==0).sum())}）")
    for d in reg["folds"]:
        print(f"[pool]   第 {d['fold']} 折：{d['val_units']} 条"
              f"（正 {d['val_pos']} / 负 {d['val_neg']}）")
    print(f"[pool] -> {OUT}")


if __name__ == "__main__":
    main()
