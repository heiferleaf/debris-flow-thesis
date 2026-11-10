# -*- coding: utf-8 -*-
"""EXP-008 阶段② v2：片段级强标签生成（B5 映射口径改版，2026-10-05）。

与旧版 build_clip_labels.py 的唯一本质区别：**废弃 秒×fps 算术**。v1_07 实证容器
元数据 fps（25）与 PTS 实测节奏（约 20）矛盾，旧映射把段边界画到不可解码区；
人工审计背书的时间体系 = POS_MSEC。v2 全程在时间域判断：

  - 片段覆盖时间 = 中间件坐标表 pts_time[first_frame .. last_frame]；
  - strict：片段首帧 pts >= 段起点 且 末帧 pts < 段终点（与旧版帧域收缩边界同义），
    且整段落在单个 pos/neg 审计段内；
  - loose：strict 基础上，完整落在 ≥2s 未标注时间缝隙内的片段也计负（协议硬约束 9）；
  - 与 cut 相交 = 0 的断言保留（R1 v2 取材禁区已保证，此处复核）。

时间轴依据：events.csv 秒数、中间件 pts_time、审计 POS_MSEC 同轴（v1_07 锚点核对
518 帧 miss=0 实证）；±40ms 内的边界抖动由「首帧 >= s / 末帧 < e」的半开区间语义
吸收，不做额外容差（与旧版帧域口径对齐，避免两版数字不可比）。

输入：EXP-008-units-v2/unit-manifest.csv + EXP-008-units-src256/coord/ + 三源 events.csv
输出：data/process/EXP-008-units-v2/clip-labels.npz（keys: unit_ids, strict, loose）
  + clip-labels.csv（逐片段人读版）

生成命令（依据 = research-auditor 2026-10-05 复核阻断 B5 + 2026-10-04 双口径决策）：
  python experiments/common/video/build_clip_labels_v2.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import CLIP_LEN, N_CLIPS, _SEG_EVENTS, read_csv_rows  # noqa: E402
from build_units_v2 import (SRC256_ROOT, UNITS_ROOT, load_coord,  # noqa: E402
                            pts_jump_zones)

OUT_NPZ = UNITS_ROOT / "clip-labels.npz"
OUT_CSV = UNITS_ROOT / "clip-labels.csv"
GAP_MIN_SEC = 2.0          # 未标注缝隙计入隐式负的最短时长（协议硬约束 9 口径）


def load_spans_by_key():
    """video_key -> [(start_sec, end_sec, kind)]（审计段，秒域，原样不收缩）。"""
    spans = defaultdict(list)
    for p in _SEG_EVENTS.values():
        for r in read_csv_rows(p):
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
    man = UNITS_ROOT / "unit-manifest.csv"
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig", newline="")))
    if not rows:
        sys.exit(f"[labels-v2] 清单为空：{man}")
    spans_sec = load_spans_by_key()

    strict_all, loose_all, uids = [], [], []
    for r in rows:
        uid = r["unit_id"]
        starts = [int(x) for x in r["clip_start_frames"].split(";")]
        assert len(starts) == N_CLIPS
        n_mw, pts, mono = load_coord(uid)          # B2：一切时间换算只查坐标表
        last_pts = pts[-1]
        jump = pts_jump_zones(pts)                 # W4 倒步邻域不参与任何标注
        if jump:
            jframes = {f for a, b in jump for f in range(a, b)}
        else:
            jframes = set()
        sp = spans_sec[uid]
        strict = np.full(N_CLIPS, -1, np.int8)
        loose = np.full(N_CLIPS, -1, np.int8)
        clip_t = [(pts[st], pts[st + CLIP_LEN - 1],
                   jframes.intersection(range(st, st + CLIP_LEN)))
                  for st in starts]
        for i, (t0, t1, jhit) in enumerate(clip_t):
            if jhit:
                continue                     # 片段落在 PTS 倒步邻域：不可用
            for s, e, k in sp:
                if t0 >= s and t1 < e:       # 时间域「完整落在收缩段内」
                    if k == "pos":
                        strict[i] = loose[i] = 1
                    elif k == "neg":
                        strict[i] = loose[i] = 0
                    break
        # 未标注缝隙（宽松口径）：[0, last_pts] 减去 pos/neg/cut 覆盖；≥2s 计负
        covered = merge([(s, e) for s, e, _ in sp])
        gaps = []
        prev = 0.0
        for a, b in covered:
            if a > prev:
                gaps.append((prev, a))
            prev = max(prev, b)
        if prev < last_pts:
            gaps.append((prev, last_pts))
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
    UNITS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, unit_ids=np.array(uids), strict=strict, loose=loose)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit_id", "clip_idx", "clip_start_frame", "strict", "loose"])
        for r, sr, lo in zip(rows, strict, loose):
            starts = [int(x) for x in r["clip_start_frames"].split(";")]
            for i in range(N_CLIPS):
                w.writerow([r["unit_id"], i, starts[i], sr[i], lo[i]])

    def stats(m):
        return int((m == 1).sum()), int((m == 0).sum()), int((m == -1).sum())
    sp, sn, su = stats(strict)
    lp, ln, lu = stats(loose)
    tot = strict.size
    print(f"[labels-v2] 视频 {len(uids)} x 片段 {N_CLIPS} = {tot}")
    print(f"[labels-v2] strict：pos {sp} / neg {sn} / 不可用 {su}（可用 {100*(sp+sn)/tot:.1f}%）")
    print(f"[labels-v2] loose ：pos {lp} / neg {ln} / 不可用 {lu}（可用 {100*(lp+ln)/tot:.1f}%）")
    if sp:
        print(f"[labels-v2] 类平衡 strict pos 占比 {100*sp/(sp+sn):.1f}%，"
              f"pos_weight(负/正) ≈ {sn/sp:.2f}")
    print(f"[labels-v2] -> {OUT_NPZ}")
    print(f"[labels-v2] -> {OUT_CSV}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
