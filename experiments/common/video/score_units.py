# -*- coding: utf-8 -*-
"""EXP-008 逐片段打分：把每一折训练出的那套模型对 48 个片段的分数还原出来（2026-10-05 立）。

用途：训练时只落盘视频级分数（`predictions.csv` 的 score = 该视频 48 个片段分数的
最大值），但要查看「模型是被哪个片段拉高的、48 个片段各自多少分」，就得把逐片段
分数补齐。本模块用已冻结的每折权重对已缓存特征做一次前向即可得到；特征已缓存，
一次全量前向是秒级，不需要重训也不需要重抽帧。

打分口径与训练完全一致：
  - 每个单位用**它所在验证折的头**打分（该头训练时没见过这个单位），
    即 `mil_f{fold}.pt`；这正是 `predictions.csv` 的口径，可用它逐位校验。
  - 模型置于 `eval()`（关 Dropout），分数确定性。
  - 单位分数 = max(可用片段分数)（MIL 口径）；掩码口径与训练时相同。

两条训练线共用本模块，用 `--head` 选：

  --head mil（默认）  读 `mil_f{折}.pt`，`MilHead` 输出直接就是分数
  --head strong       读 `strong_f{折}.pt`，`StrongHead` 输出是 logits，
                      要过一次 Sigmoid 才是分数（与 train_strong.unit_scores 同口径）。
                      因为 Sigmoid 单调，取最大分的片段号两种口径下相同。

用法（仓库根）：
  python experiments/common/video/score_units.py \
      --run data/results/exp008/mil-v2-150-strict
  python experiments/common/video/score_units.py --head strong \
      --run data/results/exp008/strong-v2-150-strict
  # 默认按 run 目录名推特征与口径，并打印与 predictions.csv 的最大绝对差
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
sys.path.insert(0, str(REPO / "experiments/common/video"))
from train_mil import MilHead  # noqa: E402
from train_strong import StrongHead  # noqa: E402

# 两条训练线的折权重文件名与判定头：`--head` 选哪一行。
HEADS = {
    "mil": (MilHead, "mil_f{}.pt", False),
    "strong": (StrongHead, "strong_f{}.pt", True),   # True = 输出是 logits，要过 Sigmoid
}

# 取材正本根：2026-10-05 坏块事故重建后为 EXP-008-units-v2（旧根 EXP-008-units
# 已作废）。与 build_units_v2.py 一致，允许 DF_UNITS_ROOT 覆盖。
UNITS_ROOT = Path(os.environ.get("DF_UNITS_ROOT")
                  or REPO / "data/process/EXP-008-units-v2")
MANIFEST = UNITS_ROOT / "unit-manifest.csv"
CLIP_LABELS = UNITS_ROOT / "clip-labels.npz"


def load_manifest():
    """unit_id -> 清单行（含 group_id/source/video_id/clip_start_frames）。"""
    rows = {}
    with open(MANIFEST, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["unit_id"]] = r
    return rows


def score_run(run_dir: Path, feats_path: Path, clip_labels=None,
              caliber: str = "strict", folds_json=None, kind: str = "mil"):
    """返回 (units, meta)。

    units = 单位列表，每项含：
      unit_id, label, fold, group_id, source, video_id,
      clip_starts [48], clip_scores [48]（不可用片段为 None）,
      unit_score（可用片段的最大分）, argmax_clip（拉高该单位的片段号）,
      frames_dir（帧目录，供前端取图）
    meta = {"caliber":..., "n_clip":..., "run":...}

    folds_json：协议三（iid）的 run 必须传 `data/metadata/splits/EXP-008-units-iid-folds.json`；
      不传则用特征 npz 的 folds 字段（协议一正本）。用错折号会取到训练时见过这条视频的
      那一折权重，分数偏高且与 predictions.csv 对不上（实测最大绝对差 7.9e-02）。
    """
    run_dir = resolve_path(run_dir, must_exist=True)
    feats_path = resolve_path(feats_path, must_exist=True)
    if clip_labels is not None:
        clip_labels = resolve_path(clip_labels, must_exist=True)
    if folds_json:
        folds_json = resolve_path(folds_json, must_exist=True)
    d = np.load(feats_path, allow_pickle=True)
    feats = d["feats"]
    labels = d["labels"].astype(int)
    unit_ids = [str(u) for u in d["unit_ids"]]
    groups = [str(g) for g in d["groups"]]
    if folds_json:
        fj = json.loads(Path(folds_json).read_text(encoding="utf-8"))
        u2f = {u: int(k["fold"]) for k in fj["folds"] for u in k["val_units"]}
        missing = [u for u in unit_ids if u not in u2f]
        assert not missing, f"折 JSON 缺单位映射: {missing[:5]}"
        folds = np.array([u2f[u] for u in unit_ids])
    else:
        folds = d["folds"].astype(int)
    n, n_clip = feats.shape[0], feats.shape[1]

    # 掩码口径：与训练一致（strict/loose）；没给标签文件就全可用
    if clip_labels is not None and Path(clip_labels).is_file():
        cl = np.load(clip_labels, allow_pickle=True)
        assert list(cl["unit_ids"]) == unit_ids, "clip-labels 与特征行序不一致"
        mask = cl[caliber].astype(int) >= 0
    else:
        mask = np.ones((n, n_clip), bool)

    head_cls, wname, is_logits = HEADS[kind]
    X = torch.tensor(feats, dtype=torch.float32)
    all_scores = np.full((n, n_clip), np.nan, np.float32)
    for f in sorted(set(int(x) for x in folds)):
        w = run_dir / wname.format(f)
        if not w.is_file():
            raise SystemExit(
                f"缺折权重 {w}——该 run 不是 "
                f"{'train_mil.py' if kind == 'mil' else 'train_strong.py'} 的产物？"
                f"（当前 --head {kind}；换另一条线请改 --head）")
        head = head_cls()
        head.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        head.eval()
        with torch.no_grad():
            s = head(X).squeeze(-1)                   # [n, n_clip]
            if is_logits:
                s = torch.sigmoid(s)                  # logits -> [0,1]，同 train_strong
            s = s.numpy()
        rows = np.where(folds == f)[0]
        all_scores[rows] = s[rows]

    man = load_manifest()
    units = []
    for i, uid in enumerate(unit_ids):
        m = man.get(uid, {})
        starts = [int(x) for x in m.get("clip_start_frames", "").split(";") if x != ""]
        ok = mask[i]
        scores = [None if not ok[j] else round(float(all_scores[i, j]), 6)
                  for j in range(n_clip)]
        usable = np.where(ok)[0]
        if len(usable) == 0:
            us, am = None, None
        else:
            j = int(usable[np.argmax(all_scores[i, usable])])
            us, am = round(float(all_scores[i, j]), 6), j
        units.append({
            "unit_id": uid, "label": int(labels[i]), "fold": int(folds[i]),
            "group_id": groups[i], "source": m.get("source", ""),
            "video_id": m.get("video_id", ""),
            "clip_starts": starts, "clip_scores": scores,
            "unit_score": us, "argmax_clip": am,
            "n_usable": int(ok.sum()),
        })
    meta = {"run": run_dir.name, "head": kind,
            "caliber": caliber if clip_labels else "full",
            "n_clip": n_clip, "n_unit": n}
    return units, meta


def verify_against_predictions(run_dir: Path, units: list):
    """与 predictions.csv 对照：返回 (n_checked, max_abs_diff)。"""
    p = resolve_path(run_dir) / "predictions.csv"
    if not p.is_file():
        return 0, None
    by = {u["unit_id"]: u for u in units}
    n, worst = 0, 0.0
    with open(p, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            u = by.get(r["unit_id"])
            if u is None or u["unit_score"] is None:
                continue
            worst = max(worst, abs(float(r["score"]) - u["unit_score"]))
            n += 1
    return n, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="训练 run 目录（mil 或 strong 线）")
    ap.add_argument("--head", choices=["mil", "strong"], default="mil",
                    help="哪条训练线的折权重与判定头：mil=弱标签线（mil_f*.pt，"
                         "输出即分数）；strong=强标签线（strong_f*.pt，输出是 "
                         "logits，过一次 Sigmoid）。默认 mil，行为与加本参数前逐位相同")
    ap.add_argument("--feats", default="", help="特征 npz；默认按 run 名推断")
    ap.add_argument("--clip-labels", default=str(CLIP_LABELS))
    ap.add_argument("--caliber", choices=["strict", "loose", "full"], default="strict",
                    help="strict/loose 用片段掩码（与 --clip-labels 训练的 run 对应）；"
                         "full = 48 片段全用（与不传 --clip-labels 的 run 对应）")
    ap.add_argument("--folds-json", default="",
                    help="协议三 run 必传 EXP-008-units-iid-folds.json；默认用 npz folds")
    args = ap.parse_args()

    run_dir = resolve_path(args.run)
    feats = Path(args.feats) if args.feats else _default_feats(run_dir)
    cl = None if args.caliber == "full" else Path(args.clip_labels)
    units, meta = score_run(run_dir, feats, cl, args.caliber,
                            args.folds_json or None, args.head)
    print(f"[score] run={meta['run']} 头 {meta['head']} "
          f"单位 {meta['n_unit']} × 片段 {meta['n_clip']} "
          f"口径 {meta['caliber']} 特征 {feats.name}")
    n, worst = verify_against_predictions(run_dir, units)
    if n:
        print(f"[score] 与 predictions.csv 对照 {n} 单位，最大绝对差 {worst:.2e}"
              f"（{'>=' if worst > 1e-5 else '<'} 1e-5，应为 0——同权重同特征同口径）")
    else:
        print("[score] 该 run 无 predictions.csv，跳过对照")


def _default_feats(run_dir: Path) -> Path:
    """run 目录名 -> 特征 npz。iid 折的 run 用同一份全量特征。"""
    return REPO / "data/results/exp008/unitfeats-v2-150/unit-features.npz"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
