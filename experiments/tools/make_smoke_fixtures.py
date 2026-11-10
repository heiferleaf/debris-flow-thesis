# -*- coding: utf-8 -*-
"""tiny 冒烟夹具派生工具（2026-10-05 立，依据 research-auditor 复核 W2）。

背景：`train_mil.py` 的 `--clip-labels` 掩码分支在 2026-10-05 被查出 min 项
填充方向写反，错误一路跑到全量 5 折才被发现——根因是该分支从未单独跑过 tiny
冒烟（旧 `runs/mil-smoke` 由引入 `--clip-labels` 之前的版本产出）。本工具把全量
片段标签按冒烟特征的单位行序切片，产出可直接喂给冒烟命令的夹具。

输入：
  - data/results/exp008/unitfeats-smoke/unit-features.npz（4 单位，定行序）
  - data/process/EXP-008-units-v2/clip-labels.npz（150 单位全量标签，数据产物）
输出：
  - experiments/tests/fixtures/clip-labels-smoke.npz（4 单位，行序与冒烟特征一致）

生成命令（仓库根）：
  python experiments/tools/make_smoke_fixtures.py

用法（掩码分支冒烟门禁，仓库根）：
  python experiments/common/video/train_mil.py \
      --feats data/results/exp008/unitfeats-smoke/unit-features.npz \
      --clip-labels experiments/tests/fixtures/clip-labels-smoke.npz \
      --caliber strict --smoke --epochs 8 --tag mil-strict-smoke
  断言：跑完不抛 RuntimeError（损失有限性守卫），且 stdout 打印逐 epoch 损失。

夹具随全量标签变化而失效：clip-labels.npz 重新生成后须重跑本工具。
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SMOKE_FEATS = REPO / "data/results/exp008/unitfeats-smoke/unit-features.npz"
FULL_LABELS = REPO / "data/process/EXP-008-units-v2/clip-labels.npz"
OUT = REPO / "experiments/tests/fixtures/clip-labels-smoke.npz"


def main():
    feats = np.load(SMOKE_FEATS, allow_pickle=True)
    labels = np.load(FULL_LABELS, allow_pickle=True)
    order = [str(u) for u in feats["unit_ids"]]
    full_ids = [str(u) for u in labels["unit_ids"]]

    missing = [u for u in order if u not in full_ids]
    assert not missing, f"全量标签缺这些冒烟单位：{missing}"
    idx = [full_ids.index(u) for u in order]

    strict = labels["strict"][idx]
    loose = labels["loose"][idx]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, unit_ids=np.array(order), strict=strict, loose=loose)

    print(f"[fixture] 行序 = {order}")
    for i, u in enumerate(order):
        s, lo = strict[i], loose[i]
        print(f"  {u}: strict 可用 {int((s >= 0).sum())}/48 "
              f"(pos {int((s == 1).sum())} / neg {int((s == 0).sum())}) | "
              f"loose 可用 {int((lo >= 0).sum())}/48")
    print(f"[fixture] -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
