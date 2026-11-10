# -*- coding: utf-8 -*-
"""EXP-008 R1 v2 取材不变式校验（对 build_units_v2 产物的独立复核，2026-10-05）。

五组检查（全绿输出 PASS，任一失败列出明细并退出码 2）：
  C1 组折一致性：同 group_id 的单位 fold 全同；fold 值在 folds.json 折号集合内。
  C2 槽位形态：48 个槽位严格递增、间隔 ≥16 帧、首槽 ≥0、末槽+16 ≤ 中间件帧数。
  C3 落盘完整：frames/<uid>/ 恰 768 张 jpg；slots.txt 旁证与清单逐位一致。
  C4 禁区复穿：任一片段帧区间不得命中 qc/ 全片质检的低方差帧、events.csv cut
     外扩区间（坐标表逐帧落帧口径）、v1_109 坏区；qc 行数必须等于坐标表帧数
     （EOF 总账旁证）。
  C5 标签一致：clip-labels.npz 行序与清单一致；strict 不可用的片段在 C4 口径下
     确实无段可落（抽查性复核，防标签与禁区两套口径漂移）。

用法：python experiments/common/video/check_units_v2.py [--only v1_07,v1_08]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import (CLIP_LEN, CUT_MARGIN_S, N_CLIPS,  # noqa: E402
                         VIDEO_KEY_OF_V1_109, V1_109_BAD_FROM_S, load_events,
                         read_csv_rows)
from build_units_v2 import (FRAMES_ROOT, QC_DIR, QC_STD_LOW, UNITS_ROOT,  # noqa: E402
                            load_coord, pts_jump_zones, secs_to_ranges)

FOLDS_JSON = REPO / "data/metadata/splits/EXP-008-units-folds.json"

from build_units import _SEG_EVENTS  # noqa: E402


def _events_csv(source):
    return _SEG_EVENTS[source]


def main():
    ap = argparse.ArgumentParser(description="EXP-008 v2 取材不变式校验")
    ap.add_argument("--only", default="", help="只查指定 unit_id（逗号分隔）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    man = UNITS_ROOT / "unit-manifest.csv"
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig", newline="")))
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    if only:
        rows = [r for r in rows if r["unit_id"] in only]
    print(f"[check-v2] 清单 {len(rows)} 单位（{man}）")

    folds = json.load(open(FOLDS_JSON, encoding="utf-8"))
    valid_folds = {k["fold"] for k in folds["folds"]}
    vid2key, vid2cuts, _ = load_events()
    problems = defaultdict(list)

    # C1 组折一致性
    g2f = defaultdict(set)
    for r in rows:
        g2f[r["group_id"]].add(r["fold"])
    for g, fs in g2f.items():
        if len(fs) != 1:
            problems["C1"].append(f"组 {g} 出现多个折 {sorted(fs)}")
    for r in rows:
        if int(r["fold"]) not in valid_folds:
            problems["C1"].append(f"{r['unit_id']} fold={r['fold']} 不在折清单")

    for i, r in enumerate(rows, 1):
        uid = r["unit_id"]
        n_mw, pts, mono = load_coord(uid)
        starts = [int(x) for x in r["clip_start_frames"].split(";")]

        # C2 槽位形态
        if len(starts) != N_CLIPS:
            problems["C2"].append(f"{uid} 槽数 {len(starts)} != {N_CLIPS}")
        elif (any(b - a < CLIP_LEN for a, b in zip(starts, starts[1:]))
                or starts[0] < 0 or starts[-1] + CLIP_LEN > n_mw):
            problems["C2"].append(f"{uid} 槽位形态非法（间隔/越界）")

        # C3 落盘完整
        udir = FRAMES_ROOT / uid
        n_jpg = len(list(udir.glob("clip*.jpg")))
        if n_jpg != N_CLIPS * CLIP_LEN:
            problems["C3"].append(f"{uid} jpg {n_jpg} != {N_CLIPS*CLIP_LEN}")
        side = (udir / "slots.txt")
        # 清单里的 clip_start_frames 已是帧号（槽号×16），与旁证直接逐位比
        if not side.exists() or side.read_text(encoding="utf-8").strip() \
                != ";".join(str(s) for s in starts):
            problems["C3"].append(f"{uid} slots.txt 旁证失配")

        # C4 禁区复穿（低方差 + cut 逐帧落帧 + v1_109）
        qc_csv = QC_DIR / f"{uid}.csv"
        qrows = read_csv_rows(qc_csv) if qc_csv.exists() else []
        if len(qrows) != n_mw:
            problems["C4"].append(f"{uid} qc 行数 {len(qrows)} != 坐标表 {n_mw}")
        lowstd = {int(q["frame"]) for q in qrows
                  if q.get("gray_std") and float(q["gray_std"]) < QC_STD_LOW}
        forb = secs_to_ranges(vid2cuts.get(
            (r["source"], r["video_id"]), []), pts)
        if uid == VIDEO_KEY_OF_V1_109:
            fb = next((j for j, t in enumerate(pts) if t >= V1_109_BAD_FROM_S), n_mw)
            if fb < n_mw:
                forb.append((fb, n_mw))
        jump = pts_jump_zones(pts)
        for si, st in enumerate(starts):
            span = range(st, st + CLIP_LEN)
            hit_low = sorted(lowstd.intersection(span))
            if hit_low:
                problems["C4"].append(f"{uid} 片段{si} 命中低方差帧 {hit_low[:4]}")
            for fa, fb2 in forb:
                if not (st + CLIP_LEN <= fa or st >= fb2):
                    problems["C4"].append(
                        f"{uid} 片段{si} 命中禁区 [{fa},{fb2})")
            for ja, jb in jump:               # W4：倒步邻域复穿检查
                if not (st + CLIP_LEN <= ja or st >= jb):
                    problems["C4"].append(
                        f"{uid} 片段{si} 命中 PTS 倒步邻域 [{ja},{jb})")

        # C5 标签一致：npz 行序兜底 + 用 events 段重算 strict 标签逐位对比
        npz_path = UNITS_ROOT / "clip-labels.npz"
        if npz_path.exists():
            z = np.load(npz_path, allow_pickle=False)
            uids = [str(x) for x in z["unit_ids"]]
            if uid not in uids:
                problems["C5"].append(f"{uid} 不在 clip-labels.npz")
            else:
                j = uids.index(uid)
                sr = z["strict"][j]
                spans = [(float(x["start_sec"]), float(x["end_sec"]),
                          x["kind"].strip())
                         for x in read_csv_rows(_events_csv(r["source"]))
                         if x["video_key"] == uid]
                for si, st in enumerate(starts):
                    t0, t1 = pts[st], pts[st + CLIP_LEN - 1]
                    want = -1
                    for s, e, k in spans:
                        if t0 >= s and t1 < e:
                            want = 1 if k == "pos" else 0 if k == "neg" else -1
                            break
                    if int(sr[si]) != want:
                        problems["C5"].append(
                            f"{uid} 片段{si} npz={int(sr[si])} 重算={want}")
        if (i % 20 == 0) or i == len(rows):
            print(f"  … {i}/{len(rows)}")

    n_bad = sum(len(v) for v in problems.values())
    for k in sorted(problems):
        print(f"[check-v2] {k}：{len(problems[k])} 条")
        for m in problems[k][:20]:
            print(f"    {m}")
        if len(problems[k]) > 20:
            print(f"    …（其余 {len(problems[k])-20} 条略）")
    print(f"[check-v2] {'全绿 PASS' if n_bad == 0 else f'FAIL 共 {n_bad} 条'}")
    sys.exit(2 if n_bad else 0)


if __name__ == "__main__":
    main()
