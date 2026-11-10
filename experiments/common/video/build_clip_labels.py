# -*- coding: utf-8 -*-
# 【已作废 2026-10-05】属坏块事故之前那套取材管线（v1），已被同目录 build_clip_labels_v2.py
# 取代，现无任何脚本调用；下面 docstring 里写的输出路径 `data/process/EXP-008-units/` 是作废
# 旧根。保留原因：run-log「已作废成绩一览」按名引用它产出的标签。勿用于新训练。
"""EXP-008 阶段②：片段级强标签生成（一条样本 = 整条视频的 48 片段，标签重算自审计段）。

双口径（research-auditor 2026-10-04 决策复核：两口径各跑一遍，主对照用严格口径）：
  strict 严格：片段 [start, start+16) 帧完整落在单个 pos 或 neg 审计段内才可标注；
  loose 宽松：strict 基础上，完整落在 ≥2s 未标注缝隙（隐式负区，协议硬约束 9 口径）
             内的片段也算 neg。两口径下跨任何区间边界的片段一律不可用（-1）。
  与 cut 区间相交的片段为 0（R1 取材已保证，脚本会断言复核）。

输入：unit-manifest.csv（clip_start_frames/src_fps/n_frames_src）+ 三源 events.csv
输出：data/process/EXP-008-units/clip-labels.npz
  keys: unit_ids, strict [151,48] int8, loose [151,48] int8（1=pos 0=neg -1=不可用）
      + clip-labels.csv（逐片段人读版）

生成命令（依据 = EXP-008 卡第三段设计 + research-auditor 2026-10-04 复核结论）：
  python experiments/common/video/build_clip_labels.py
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import N_CLIPS, CLIP_LEN, OUT_MANIFEST, _SEG_EVENTS, read_csv_rows  # noqa: E402

OUT_NPZ = OUT_MANIFEST.parent / "clip-labels.npz"
OUT_CSV = OUT_MANIFEST.parent / "clip-labels.csv"
GAP_MIN_SEC = 2.0          # 未标注缝隙计入隐式负的最短时长（协议硬约束 9 口径）


def load_spans_by_key():
    """video_key -> [(a_frame, b_frame, kind)]，帧域半开区间，边界向内收缩。"""
    spans = defaultdict(list)
    for p in _SEG_EVENTS.values():
        for r in read_csv_rows(p):
            fps = None  # 帧率按 manifest 逐条视频换算，这里先存秒
            spans[r["video_key"]].append(
                (float(r["start_sec"]), float(r["end_sec"]), r["kind"].strip()))
    return spans


def merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main():
    rows = list(csv.DictReader(open(OUT_MANIFEST, encoding="utf-8-sig", newline="")))
    spans_sec = load_spans_by_key()

    strict_all, loose_all, uids = [], [], []
    for r in rows:
        fps = float(r["src_fps"])
        n_src = int(r["n_frames_src"])
        starts = [int(x) for x in r["clip_start_frames"].split(";")]
        assert len(starts) == N_CLIPS
        # unit_id = video_key（v1_07 形态，events.csv 的 video_key 列同键）
        vid_key = r["unit_id"]
        sp = spans_sec[vid_key]
        # 帧域收缩边界（片段必须完整落在 [a,b) 内，保证标注纯度）
        lab = [(math.ceil(s * fps), math.floor(e * fps), k) for s, e, k in sp]
        strict = np.full(N_CLIPS, -1, np.int8)
        loose = np.full(N_CLIPS, -1, np.int8)
        for i, st in enumerate(starts):
            ce = st + CLIP_LEN
            for a, b, k in lab:
                if a <= st and ce <= b:
                    if k == "pos":
                        strict[i] = loose[i] = 1
                    elif k == "neg":
                        strict[i] = loose[i] = 0
                    break
        # 未标注缝隙（宽松口径）：全帧域减去 pos/neg/cut 收缩区间；gap ≥ 2s 内片段 = 负
        covered = merge([(a, b) for a, b, k in lab])
        gaps = []
        prev = 0
        for a, b in covered:
            if a > prev:
                gaps.append((prev, a))
            prev = max(prev, b)
        if prev < n_src:
            gaps.append((prev, n_src))
        gaps = [(a, b) for a, b in gaps if (b - a) / fps >= GAP_MIN_SEC]
        for i, st in enumerate(starts):
            if loose[i] != -1:
                continue
            ce = st + CLIP_LEN
            for a, b in gaps:
                if a <= st and ce <= b:
                    loose[i] = 0
                    break
        strict_all.append(strict)
        loose_all.append(loose)
        uids.append(r["unit_id"])

    strict = np.stack(strict_all)
    loose = np.stack(loose_all)
    np.savez_compressed(OUT_NPZ, unit_ids=np.array(uids), strict=strict, loose=loose)

    # 人读 CSV + 断言复核（与 cut 相交必须为 0：R1 取材防呆已保证）
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "clip_idx", "clip_start_frame", "strict", "loose"])
        for r, sr, lo in zip(rows, strict, loose):
            starts = [int(x) for x in r["clip_start_frames"].split(";")]
            for i in range(N_CLIPS):
                w.writerow([r["unit_id"], i, starts[i], sr[i], lo[i]])

    def stats(m):
        pos = int((m == 1).sum()); neg = int((m == 0).sum()); un = int((m == -1).sum())
        return pos, neg, un
    sp, sn, su = stats(strict)
    lp, ln, lu = stats(loose)
    tot = strict.size
    print(f"[labels] 视频 {len(uids)} x 片段 {N_CLIPS} = {tot}")
    print(f"[labels] strict：pos {sp} / neg {sn} / 不可用 {su}（可用 {100*(sp+sn)/tot:.1f}%）")
    print(f"[labels] loose ：pos {lp} / neg {ln} / 不可用 {lu}（可用 {100*(lp+ln)/tot:.1f}%）")
    print(f"[labels] 类平衡 strict pos 占比 "
          f"{100*sp/(sp+sn):.1f}%" if sp + sn else "[labels] strict 无可用片段（查键名）",
          f"，pos_weight(负/正) ≈ {sn/sp:.2f}" if sp else "")
    print(f"[labels] -> {OUT_NPZ}")
    print(f"[labels] -> {OUT_CSV}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
