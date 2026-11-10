# -*- coding: utf-8 -*-
"""EXP-008 v3：片段级强标签生成（采样帧间隔可变的版本）。

与 build_clip_labels_v2.py 的唯一区别在「片段覆盖哪段时间」。v2 的片段是连续
16 帧，所以覆盖时间 = pts[首帧 .. 首帧+15]。v3 的 16 帧每隔 frame_stride 帧取
一个，覆盖时间 = pts[首帧 .. 首帧+15×frame_stride]。若沿用 v2 的写法，7.5 秒
跨度的片段会被当成 0.75 秒来判，跨越审计段边界的片段会被错标成可用。

判定规则与 v2 完全一致（未改口径）：
  - strict：片段首帧 pts >= 段起点 且 末帧 pts < 段终点，且整段落在单个 pos/neg
    审计段内；
  - loose：strict 基础上，完整落在 >=2 秒未标注缝隙内的片段也计负（协议硬约束 9）；
  - 片段时间跨度与 PTS 倒步邻域相交 = 不可用。

输入：指定数据根的 unit-manifest.csv（含 clip_start_frames 与 frame_stride）
      + EXP-008-units-src256/coord/ + 三源 events.csv
输出：<根>/clip-labels.npz（keys: unit_ids, strict, loose）
      + <根>/clip-labels.csv（逐片段人读版）

生成命令（依据 = research-auditor 2026-10-25 阻断 B2–B4 的第 3 条）：
  python experiments/common/video/build_clip_labels_v3.py --root EXP-008-units-v3-gap05
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import CLIP_LEN, N_CLIPS, _SEG_EVENTS, read_csv_rows  # noqa: E402
from build_units_v2 import SRC256_ROOT, load_coord, pts_jump_zones  # noqa: E402

GAP_MIN_SEC = 2.0


def load_spans_by_key():
    spans = defaultdict(list)
    for p in _SEG_EVENTS.values():
        for r in read_csv_rows(p):
            spans[r["video_key"]].append(
                (float(r["start_sec"]), float(r["end_sec"]), r["kind"].strip()))
    return spans


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="EXP-008-units-v3-gap05")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = REPO / "data/process" / args.root
    man = root / "unit-manifest.csv"
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig", newline="")))
    if not rows:
        sys.exit(f"[labels-v3] 清单为空：{man}")
    spans_sec = load_spans_by_key()

    strict_all, loose_all, uids = [], [], []
    for r in rows:
        uid = r["unit_id"]
        stride = int(r["frame_stride"])
        starts = [int(x) for x in r["clip_start_frames"].split(";")]
        assert len(starts) == N_CLIPS, f"{uid} 片段数 {len(starts)}"
        n_mw, pts, _ = load_coord(uid)
        jump = pts_jump_zones(pts)
        sp = spans_sec[uid]
        strict = np.full(N_CLIPS, -1, np.int8)
        loose = np.full(N_CLIPS, -1, np.int8)
        # 片段实际覆盖：首帧 pts .. 首帧+15×stride 的 pts
        clip_t = [(pts[st], pts[st + (CLIP_LEN - 1) * stride],
                   bool(jump and any(a <= st + (CLIP_LEN - 1) * stride and st < b
                                     for a, b in jump)))
                  for st in starts]
        for i, (t0, t1, jhit) in enumerate(clip_t):
            if jhit:
                continue
            for s, e, k in sp:
                if t0 >= s and t1 < e:
                    if k == "pos":
                        strict[i] = loose[i] = 1
                    elif k == "neg":
                        strict[i] = loose[i] = 0
                    break

        covered = merge([(s, e) for s, e, _ in sp])
        gaps, prev = [], 0.0
        for a, b in covered:
            if a > prev:
                gaps.append((prev, a))
            prev = max(prev, b)
        if prev < pts[-1]:
            gaps.append((prev, pts[-1]))
        gaps = [(a, b) for a, b in gaps if b - a >= GAP_MIN_SEC]
        for i, (t0, t1, jhit) in enumerate(clip_t):
            if loose[i] != -1 or jhit:
                continue
            for a, b in gaps:
                if a <= t0 and t1 < b:
                    loose[i] = 0
                    break
        strict_all.append(strict)
        loose_all.append(loose)
        uids.append(uid)

    strict = np.stack(strict_all)
    loose = np.stack(loose_all)
    np.savez_compressed(root / "clip-labels.npz",
                        unit_ids=np.array(uids), strict=strict, loose=loose)
    with open(root / "clip-labels.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "clip_idx", "clip_start_frame", "strict", "loose"])
        for r, sr, lo in zip(rows, strict, loose):
            starts = [int(x) for x in r["clip_start_frames"].split(";")]
            for i in range(N_CLIPS):
                w.writerow([r["unit_id"], i, starts[i], sr[i], lo[i]])

    def stats(m):
        return int((m == 1).sum()), int((m == 0).sum()), int((m == -1).sum())
    sp_, sn_, su_ = stats(strict)
    lp_, ln_, lu_ = stats(loose)
    tot = strict.size
    print(f"[labels-v3] 视频 {len(uids)} 条 × 片段 {N_CLIPS} = {tot}")
    print(f"[labels-v3] strict：pos {sp_} / neg {sn_} / 不可用 {su_}"
          f"（可用 {100*(sp_+sn_)/tot:.1f}%）")
    print(f"[labels-v3] loose ：pos {lp_} / neg {ln_} / 不可用 {lu_}"
          f"（可用 {100*(lp_+ln_)/tot:.1f}%）")
    zero = [u for u, sr in zip(uids, strict) if (sr != -1).sum() == 0]
    if zero:
        print(f"[labels-v3] strict 下一个可用片段都没有的视频 {len(zero)} 条："
              f"{', '.join(sorted(zero))}")
    print(f"[labels-v3] -> {root / 'clip-labels.npz'}")


if __name__ == "__main__":
    main()
