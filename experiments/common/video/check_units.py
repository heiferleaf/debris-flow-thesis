# -*- coding: utf-8 -*-
# 【已作废 2026-10-05】属坏块事故之前那套取材管线（v1），已被同目录 check_units_v2.py 取代，
# 现无任何脚本调用。保留原因：run-log「已作废成绩一览」按名引用它校验过的数据。勿用于新训练。
"""EXP-008 R1 校验：单位清单 + 落盘帧逐条对照设计（任何一条不过即退出码 1）。

校验项（对应 EXP-008.md 数据契约，独立重算，不复用 build_units 的中间结果）：
  C1 单位池：清单单位数 = 段清单东川三源视频数；逐单位 label 与 events.csv 重算一致
     （含任一 pos 段 = pos，否则 neg）；events 里含 pos/neg 的视频与清单集合相等
     （防漏防多）；source 白名单 = {dongchuan, video4, video5}。
  C2 组不变式：每单位恰一组；同组全部单位同折；折号与 EXP-006-session-folds.json
     组到折映射一致。
  C3 片段形态：每单位 n_clips=48、clip_len=16；clip_start_frames 共 48 个、严格递增、
     相邻间隔 >= 16 帧（不重叠）；首槽 >= 0；末槽+16 <= n_frames_src。
  C4 禁区回避：任一片段与剔除区间（events.csv cut 行，前后外扩 1s）或 v1_109 坏区
     （>= 1446.6s）在帧域相交即失败（与 build_units 同一规则重算）。
  C5 帧落盘：每单位目录恰好 768 张 clip{cc}_f{ff}.jpg 且文件名全集无缺无多；清单
     n_frames = 768。
  C6 折构成：打印各折验证单位数与正负构成（不强校验，供 run-log 如实报告）。

用法（仓库根，venv）：
  python experiments/common/video/check_units.py
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import (CLIP_LEN, FRAMES_ROOT, FOLDS_JSON, N_CLIPS, OUT_MANIFEST,
                         SEG_MANIFEST, SOURCES, VIDEO_KEY_OF_V1_109,
                         V1_109_BAD_FROM_S, load_events, merge_ranges)  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"：{detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser(description="EXP-008 R1 校验")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny 冒烟模式：跳过 C1a/C1b 池规模校验，其余照跑")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(OUT_MANIFEST, encoding="utf-8-sig", newline="")))
    # 有据排除清单（build_units 落盘：无法支撑 48×16 单位形态的视频）
    fail_csv = FRAMES_ROOT.parent / "unit-failures.csv"
    excluded = {}
    if fail_csv.exists():
        for r in csv.DictReader(open(fail_csv, encoding="utf-8-sig", newline="")):
            excluded[r["video_id"]] = r["reason"]
        print(f"  [INFO] 有据排除 {len(excluded)} 条："
              + "；".join(f"{k}（{v}）" for k, v in excluded.items()))
    vid2key, vid2cuts, _ = load_events()
    key2vid = {v: k for k, v in vid2key.items()}

    # C1 单位池
    seg_v = set()
    seg_label = {}
    for r in csv.DictReader(open(SEG_MANIFEST, encoding="utf-8-sig", newline="")):
        if r["source"] in SOURCES:
            seg_v.add(r["video_id"])
            if r["kind"] == "pos":
                seg_label[r["video_id"]] = "pos"
            elif r["kind"] == "neg":
                seg_label.setdefault(r["video_id"], "neg")
    _, _, vid2kinds = load_events()
    ev_v = {vid for (s, vid), kinds in vid2kinds.items()
            if s in SOURCES and kinds & {"pos", "neg"}}
    check("C1a 清单单位数 = 段清单东川视频数 − 有据排除",
          args.tiny or len(rows) == len(seg_v) - len(excluded),
          f"清单 {len(rows)} vs 段清单 {len(seg_v)} − 排除 {len(excluded)}")
    check("C0 排除与清单互斥（被排除视频不得出现在清单）",
          not (excluded.keys() & {r["video_id"] for r in rows}),
          str(excluded.keys() & {r["video_id"] for r in rows}) or "")
    check("C1b 集合相等（清单 vs events 含 pos/neg 视频减排除）",
          args.tiny or {r["video_id"] for r in rows} == ev_v - set(excluded),
          f"差集 {({r['video_id'] for r in rows} ^ (ev_v - set(excluded))) or '无'}")
    bad_label = [r["unit_id"] for r in rows if r["label"] != seg_label[r["video_id"]]]
    check("C1c 逐单位 label 与重算一致", not bad_label, f"不一致 {bad_label[:5]}")
    check("C1d source 白名单", {r["source"] for r in rows} <= set(SOURCES),
          str(sorted({r["source"] for r in rows})))
    n_pos = sum(1 for r in rows if r["label"] == "pos")
    print(f"  [INFO] 单位 {len(rows)} = 正 {n_pos} / 负 {len(rows)-n_pos}")

    # C2 组不变式
    folds = json.load(open(FOLDS_JSON, encoding="utf-8"))
    g2f = {g: k["fold"] for k in folds["folds"] for g in k["val_groups"]}
    g_of = defaultdict(set)
    for r in rows:
        g_of[r["group_id"]].add(r["fold"])
    check("C2a 每组恰一折", all(len(v) == 1 for v in g_of.values()),
          str({g: v for g, v in g_of.items() if len(v) > 1}) or "")
    bad = [r["unit_id"] for r in rows
           if r["group_id"] not in g2f or g2f[r["group_id"]] != int(r["fold"])]
    check("C2b 折号与组折映射一致（组不在映射中即失败）", not bad, f"不一致/缺映射 {bad[:5]}")

    # C4 禁区（帧域）
    def forb_frames(r):
        fps = float(r["src_fps"])
        n_src = int(r["n_frames_src"])
        rs = []
        for s, e in vid2cuts.get(key2vid[r["unit_id"]], []):
            a = max(0, math.ceil((s - 1.0) * fps))
            b = min(n_src, math.ceil((e + 1.0) * fps))
            if b > a:
                rs.append((a, b))
        if r["unit_id"] == VIDEO_KEY_OF_V1_109:
            rs.append((math.ceil(V1_109_BAD_FROM_S * fps), n_src))
        return merge_ranges(rs)

    # C3+C5 逐单位
    pat = re.compile(r"^clip(\d{2})_f(\d{2})\.jpg$")
    for r in rows:
        uid = r["unit_id"]
        starts = [int(x) for x in r["clip_start_frames"].split(";")]
        ok3 = (int(r["n_clips"]) == N_CLIPS and int(r["clip_len"]) == CLIP_LEN
               and len(starts) == N_CLIPS
               and all(b - a >= CLIP_LEN for a, b in zip(starts, starts[1:]))
               and starts[0] >= 0 and starts[-1] + CLIP_LEN <= int(r["n_frames_src"]))
        if not ok3:
            check(f"C3 {uid} 片段形态", False)
            continue
        inter = [(fa, fb) for s in starts for fa, fb in forb_frames(r)
                 if s < fb and s + CLIP_LEN > fa]
        if inter:
            check(f"C4 {uid} 禁区回避", False, f"相交区间 {inter[:3]}")
        d = FRAMES_ROOT / uid
        files = [p.name for p in d.glob("clip*.jpg")] if d.is_dir() else []
        want = {f"clip{c:02d}_f{f:02d}.jpg" for c in range(N_CLIPS) for f in range(CLIP_LEN)}
        got = set(files)
        sidecar = d / "slots.txt"
        side = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""
        if (got != want or int(r["n_frames"]) != N_CLIPS * CLIP_LEN
                or side != r["clip_start_frames"]):
            check(f"C5 {uid} 帧落盘（含槽位旁证一致）", False,
                  f"缺 {len(want-got)} 多 {len(got-want)} n_frames={r['n_frames']} "
                  f"旁证{'失配' if side != r['clip_start_frames'] else '一致'}")
    if not FAILS or all(not f.startswith(("C3", "C4", "C5")) for f in FAILS):
        check("C3/C4/C5 全部单位", True, f"{len(rows)} 单位逐条通过")

    # C6 折构成（报告用）
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[int(r["fold"])][0 if r["label"] == "pos" else 1] += 1
    print("  [INFO] 各折全体单位构成（fold: 正/负）：")
    for f in sorted(per):
        print(f"    fold{f}: {per[f][0]} 正 / {per[f][1]} 负")

    print(f"\n{'全部校验通过' if not FAILS else '校验失败：' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
