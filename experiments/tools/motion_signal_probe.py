# -*- coding: utf-8 -*-
"""运动信号预检（第二版）：按 research-auditor 2026-10-20 复核意见扩写。

相对第一版增加四件事：
1. 掩掉画面边缘 12%（右下角地址水印、角落时间戳会在帧差里造成固定位置脉冲）；
2. 按原生帧率把帧差归一成「每秒位移」，消除东川（约 20 fps）与外部素材（约 10 fps）的尺度差；
3. 加「视频内对比」：只在同一条视频内部比较正片段与负片段的帧差，这条读数不受机位与标签同向的混淆；
4. 负样本拆成两类分别报数：纯负视频里的空沟画面，与正视频里的非事件画面（后者常是常规水流）。

用法：
    .venv/Scripts/python.exe .codex/docs/.temp/motion-signal/motion_probe2.py \
        --out .codex/docs/.temp/motion-signal/clip-motion-features-v2.csv
"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

FRAMES_ROOT = os.path.join("data", "process", "EXP-008-units-v2", "frames")
LABELS_CSV = os.path.join("data", "process", "EXP-008-units-v2", "clip-labels.csv")
MANIFEST_CSV = os.path.join("data", "process", "EXP-008-units-v2", "unit-manifest.csv")

BORDER = 0.12  # 左右上下各裁掉 12% 的边


def load_clip_frames(unit_id, clip_idx, n_frames=16):
    arrs = []
    for f in range(n_frames):
        p = os.path.join(FRAMES_ROOT, unit_id, f"clip{clip_idx:02d}_f{f:02d}.jpg")
        with Image.open(p) as im:
            g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        arrs.append(g)
    return np.stack(arrs, axis=0)


def clip_features(stack):
    h, w = stack.shape[1:]
    dy, dx = int(h * BORDER), int(w * BORDER)
    inner = stack[:, dy:h - dy, dx:w - dx]
    diffs = inner[1:] - inner[:-1]
    ad = np.abs(diffs)

    mad = float(ad.mean())
    denergy = float((diffs ** 2).mean())
    tstd = float(inner.std(axis=0).mean())

    flat = diffs.reshape(diffs.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1) + 1e-8
    unit = flat / norms[:, None]
    corr = float((unit[1:] * unit[:-1]).sum(axis=1).mean())

    d = ad.reshape(ad.shape[0], -1)
    k = max(1, d.shape[1] // 10)
    top = np.sort(d, axis=1)[:, -k:]
    concentration = float((top.sum(axis=1) / (d.sum(axis=1) + 1e-8)).mean())

    # 运动的时间稳定性：泥石流的湍动是连续起伏，云台抖动是脉冲。
    # 用相邻帧差序列的变异系数（标准差/均值），越大越像脉冲。
    m = ad.mean(axis=(1, 2))
    cv = float(m.std() / (m.mean() + 1e-8))

    return {"mad": mad, "denergy": denergy, "tstd": tstd, "diff_corr": corr,
            "concentration": concentration, "temporal_cv": cv}


def auc(y, s):
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


NAMES = ["mad", "mad_per_sec", "denergy", "tstd", "diff_corr",
         "concentration", "temporal_cv"]

CENTER_FRAME = 8  # 16 帧里的第 9 帧，即中心帧；取哪一帧不影响结论，取中心帧是为了确定性


def inner_frame(path):
    """读一张帧图，按与 clip_features 相同的口径裁边并归一到 0–1。"""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    h, w = g.shape
    dy, dx = int(h * BORDER), int(w * BORDER)
    return g[dy:h - dy, dx:w - dx]


def redundancy_summary(clips, feats):
    """片段内 16 帧有多像、相邻片段之间有多不像。

    这是描述性统计，不进任何判别结论；它的用途是回答「视频侧 11 万帧里
    有多少是同一张图重复」，为「每个片段只取 1 帧」的数据方案提供依据。

    口径与运动特征完全一致（裁掉四边各 12%、灰度归一到 0–1）。
    「相邻片段」的定义写死：同一条视频内、strict 可用片段序列里编号相差 1 的两段；
    中间夹着掩码剔除片段的配对不算相邻（那两段不是时间上紧挨着的画面）。

    返回：片段内相邻帧差的全部值，与相邻片段中心帧差的全部值。
    """
    within = np.array([feats[(u, c)]["mad"] for u, c, _ in clips], dtype=np.float64)

    by_unit = {}
    for u, c, _ in clips:
        by_unit.setdefault(u, []).append(c)

    pairs = []
    for u, idxs in by_unit.items():
        idxs.sort()
        for a, b in zip(idxs, idxs[1:]):
            if b - a == 1:
                pairs.append((u, a, b))

    def one_pair(p):
        u, a, b = p
        fa = inner_frame(os.path.join(FRAMES_ROOT, u, f"clip{a:02d}_f{CENTER_FRAME:02d}.jpg"))
        fb = inner_frame(os.path.join(FRAMES_ROOT, u, f"clip{b:02d}_f{CENTER_FRAME:02d}.jpg"))
        return float(np.abs(fa - fb).mean())

    with ThreadPoolExecutor(max_workers=8) as pool:
        between = np.array(list(pool.map(one_pair, pairs)), dtype=np.float64)

    return within, between


def fmt_row(label, vals, width=9):
    return label.ljust(16) + "".join(
        ("   nan" if np.isnan(v) else f"{v:{width}.3f}") for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".codex/docs/.temp/motion-signal/clip-motion-features-v2.csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    meta = {}
    with open(MANIFEST_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            duration = float(row["duration_sec"]) if row["duration_sec"] else 0.0
            native = float(row["n_frames_mw"]) if row["n_frames_mw"] else 0.0
            fps = native / duration if duration > 0 else float("nan")
            meta[row["unit_id"]] = {
                "fold": int(row["fold"]),
                "unit_label": row["label"],
                "source": row["source"],
                "fps": fps,
            }

    clips = []
    with open(LABELS_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["strict"] not in ("0", "1"):
                continue
            clips.append((row["unit_id"], int(row["clip_idx"]), int(row["strict"])))
    print(f"片段数：{len(clips)}", flush=True)

    feats = {}

    def work(item):
        unit_id, clip_idx, _ = item
        return (unit_id, clip_idx), clip_features(load_clip_frames(unit_id, clip_idx))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (key, feat) in enumerate(pool.map(work, clips), 1):
            feats[key] = feat
            if n % 1000 == 0:
                print(f"  已算 {n}/{len(clips)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "clip_idx", "clip_label", "unit_label", "source",
                    "fold", "fps", "mad", "mad_per_sec", "denergy", "tstd",
                    "diff_corr", "concentration", "temporal_cv"])
        for unit_id, clip_idx, lab in clips:
            ft = feats[(unit_id, clip_idx)]
            m = meta[unit_id]
            w.writerow([unit_id, clip_idx, lab, m["unit_label"], m["source"],
                        m["fold"], f"{m['fps']:.3f}", f"{ft['mad']:.6f}",
                        f"{ft['mad'] * m['fps']:.4f}", f"{ft['denergy']:.6f}",
                        f"{ft['tstd']:.6f}", f"{ft['diff_corr']:.6f}",
                        f"{ft['concentration']:.6f}", f"{ft['temporal_cv']:.6f}"])

    # 组装带派生列的记录
    recs = []
    for unit_id, clip_idx, lab in clips:
        ft = dict(feats[(unit_id, clip_idx)])
        ft["mad_per_sec"] = ft["mad"] * meta[unit_id]["fps"]
        recs.append({"unit_id": unit_id, "clip_idx": clip_idx, "y": lab,
                     "unit_label": meta[unit_id]["unit_label"],
                     "source": meta[unit_id]["source"],
                     "fold": meta[unit_id]["fold"], **ft})

    folds = sorted({r["fold"] for r in recs})
    print("\n【全池】逐折单特征 AUC（片段级；含机位混淆，只能当上界看）")
    print("特征".ljust(16) + "".join(f"fold{f}".rjust(9) for f in folds) + "平均".rjust(10))
    for k in NAMES:
        vals = [auc([r["y"] for r in recs if r["fold"] == f],
                    [r[k] for r in recs if r["fold"] == f]) for f in folds]
        print(fmt_row(k, vals + [float(np.nanmean(vals))]))

    print("\n【视频内】只在同时含正负片段的视频内部比（不受机位与标签同向的混淆）")
    mixed = {}
    for r in recs:
        mixed.setdefault(r["unit_id"], []).append(r)
    mixed = {u: rs for u, rs in mixed.items() if len({r["y"] for r in rs}) == 2}
    print(f"可用于视频内对比的视频：{len(mixed)} 条")
    for k in NAMES:
        per_video = [(u, auc([r["y"] for r in rs], [r[k] for r in rs]))
                     for u, rs in sorted(mixed.items())]
        vals = [v for _, v in per_video if not np.isnan(v)]
        above = sum(1 for v in vals if v > 0.5)
        print(f"  {k:<16} 中位 AUC {np.median(vals):.3f}   大于 0.5 的视频 "
              f"{above}/{len(vals)}   区间 [{min(vals):.3f}, {max(vals):.3f}]")
    for u, rs in sorted(mixed.items()):
        n_pos = sum(1 for r in rs if r["y"] == 1)
        print(f"    {u}: 正片段 {n_pos} / 负片段 {len(rs) - n_pos}，"
              f"mad 均值 正 {np.mean([r['mad'] for r in rs if r['y'] == 1]):.4f} "
              f"负 {np.mean([r['mad'] for r in rs if r['y'] == 0]):.4f}")

    print("\n【负样本拆类】帧差均值（mad）分布")
    pure_neg = [r for r in recs if r["unit_label"] == "neg"]
    in_pos_neg = [r for r in recs if r["unit_label"] == "pos" and r["y"] == 0]
    pos = [r for r in recs if r["y"] == 1]
    for name, group in [("空沟/无水（纯负视频的片段）", pure_neg),
                        ("正视频里的非事件片段", in_pos_neg),
                        ("泥石流片段", pos)]:
        if not group:
            continue
        v = np.array([r["mad"] for r in group])
        print(f"  {name:<24} n={len(v):5d}  中位 {np.median(v):.4f}  "
              f"P10 {np.percentile(v, 10):.4f}  P90 {np.percentile(v, 90):.4f}")

    print("\n【按数据源】" + " / ".join(sorted({r["source"] for r in recs})))
    for src in sorted({r["source"] for r in recs}):
        rs = [r for r in recs if r["source"] == src]
        pos_s = [r for r in rs if r["y"] == 1]
        neg_s = [r for r in rs if r["y"] == 0]
        a = auc([r["y"] for r in rs], [r["mad"] for r in rs])
        print(f"  {src:<12} n={len(rs):5d} 正 {len(pos_s):5d} 负 {len(neg_s):5d}  "
              f"mad 正 {np.mean([r['mad'] for r in pos_s]):.4f}  "
              f"负 {np.mean([r['mad'] for r in neg_s]):.4f}  AUC {a:.3f}")

    # ---- 帧冗余汇总（描述性统计，与上面的判别结论分开看）----
    within, between = redundancy_summary(clips, feats)
    w_med = float(np.median(within))
    b_med = float(np.median(between))
    summary = {
        "口径": "裁掉画面四边各 12%，灰度归一到 0–1；与运动特征同一套读图代码",
        "片段内相邻帧差": {"样本量": int(len(within)), "中位数": w_med,
                     "均值": float(within.mean())},
        "相邻片段中心帧差": {"样本量": int(len(between)), "中位数": b_med,
                       "均值": float(between.mean()),
                       "相邻的定义": "同一视频内 strict 可用片段编号相差 1 的两段，"
                                    "中间夹掩码剔除段的不算"},
        "比值": b_med / w_med if w_med > 0 else float("nan"),
        "结论": "片段内 16 帧高度重复（近似同一张图）；片段之间才是新画面。"
                "因此视频侧的独立样本单位是片段，不是帧。",
    }
    print("\n【帧冗余汇总】")
    print(f"  片段内相邻帧差中位 {w_med:.4f}（n={len(within)}）")
    print(f"  相邻片段中心帧差中位 {b_med:.4f}（n={len(between)}）")
    print(f"  两者比值 {b_med / w_med:.2f} 倍")
    side = os.path.splitext(args.out)[0] + ".redundancy.json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  汇总已写 {side}")


if __name__ == "__main__":
    sys.exit(main())
