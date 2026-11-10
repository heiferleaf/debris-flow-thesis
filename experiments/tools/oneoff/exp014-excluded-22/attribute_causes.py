# -*- coding: utf-8 -*-
"""EXP-014 在 0.5s 间隔下排除的那 22 条：逐条归因排除原因。

复用 build_units_stride 的同一套禁区构造（剪切区间 / 低灰度帧 / PTS 倒步 /
时间空洞），对每条视频分别计算「只留某几类禁区」时的合法片段数，从而把
「凑不满 48」归因到哪几类禁区。只读，不写任何数据文件。

用法（仓库根目录）：
  .venv/Scripts/python.exe experiments/tools/oneoff/exp014-excluded-22/attribute_causes.py
"""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import (CLIP_LEN, N_CLIPS, QC_STD_LOW,  # noqa: E402
                         VIDEO_KEY_OF_V1_109, V1_109_BAD_FROM_S,
                         load_events, merge_ranges, read_csv_rows)
from build_units_v2 import (pts_gap_zones, pts_jump_zones,  # noqa: E402
                            zones_from_frames, secs_to_ranges)
from build_units_stride import blocks_for, slot_ok  # noqa: E402

SRC256 = REPO / "data/process/EXP-008-units-src256"
V2 = REPO / "data/process/EXP-008-units-v2"
V3 = REPO / "data/process/EXP-008-units-v3-gap05"
GAP = 0.5


def zones_by_type(uid, cuts, pts, lowstd):
    """返回 {类型名: 帧号区间列表}。剪切类再拆出 v1_109 的坏区块。"""
    n = len(pts)
    z = {
        "剪切剔除区（±1 秒外扩）": secs_to_ranges(cuts, pts),
        "低灰度帧（前后各 16 帧）": zones_from_frames(lowstd, n),
        "PTS 倒步邻域": pts_jump_zones(pts),
        "时间空洞（前后各 1 秒）": pts_gap_zones(pts),
    }
    if uid == VIDEO_KEY_OF_V1_109:
        fb = next((j for j, t in enumerate(pts) if t >= V1_109_BAD_FROM_S), n)
        if fb < n:
            z.setdefault("剪切剔除区（±1 秒外扩）", []).append((fb, n))
    return z


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fails = {r["unit_id"] for r in csv.DictReader(open(V3 / "unit-failures.csv", encoding="utf-8-sig"))}
    man = {r["unit_id"]: r for r in csv.DictReader(open(V2 / "unit-manifest.csv", encoding="utf-8-sig"))}
    _, vid2cuts, _ = load_events()

    rows = []
    for uid in sorted(fails):
        row = man[uid]
        pts = [float(r["pts_time"]) for r in read_csv_rows(SRC256 / "coord" / f"{uid}.csv")]
        n = len(pts)
        lowstd = [int(r["frame"]) for r in read_csv_rows(V2 / "qc" / f"{uid}.csv")
                  if r.get("gray_std") and float(r["gray_std"]) < QC_STD_LOW]
        fps = (n - 1) / pts[-1] if pts[-1] > 0 else 10.0
        stride = max(1, int(round(GAP * fps)))
        block = blocks_for(stride)
        n_slots = n // block
        zt = zones_by_type(uid, vid2cuts.get((row["source"], row["video_id"]), []), pts, lowstd)

        def legal(keep):
            forb = merge_ranges([r for k in keep for r in zt[k]])
            return sum(1 for s in range(n_slots) if slot_ok(s, block, forb))

        all_types = list(zt)
        n_all = legal(all_types)
        assert n_all < N_CLIPS, f"{uid}: 全禁区下已有 {n_all} 个合法片段，不在排除名单里"
        without = {k: legal([t for t in all_types if t != k]) for k in all_types}
        cut_types = [k for k in all_types if "剪切" in k]
        grey_types = [k for k in all_types if "剪切" not in k]
        n_only_cut = legal(cut_types)        # 只留剪切区 → 这条是不是被灰度/倒步/空洞卡死
        n_only_grey = legal(grey_types)      # 只留灰度族 → 这条是不是被剪切区卡死
        n_none = legal([])                   # 全拿掉还凑不满就是容量/时长问题
        # 归因：拿掉某类禁区后能凑满 48 的，就是「有它才凑不满」的原因类
        causes = [k for k, v in without.items() if v >= N_CLIPS]
        if n_none < N_CLIPS:
            grp = f"全拿掉也只有 {n_none} 个合法片段（容量/时长不足，与禁区无关）"
        elif n_only_cut >= N_CLIPS and n_only_grey >= N_CLIPS:
            grp = "两族单独留都凑不满，两族合起来才卡死"
        elif n_only_grey >= N_CLIPS:
            grp = "剪切区一族卡死（只留灰度/倒步/空洞能凑满）"
        elif n_only_cut >= N_CLIPS:
            grp = "灰度/倒步/空洞一族卡死（只留剪切区能凑满）"
        else:
            grp = "两族合起来卡死，且各自都缺一截"
        rows.append((uid, n_slots, n_all, without, causes, n_only_cut, n_only_grey, n_none, grp,
                     {k: len(v) for k, v in zt.items()}))

    print(f"0.5s 间隔下排除的 {len(rows)} 条，逐条归因（槽位总数 = 按该条帧率步长切出的"
          f"候选位置个数；「只留剪切区」「只留灰度/倒步/空洞」= 两族禁区分别单独保留时的"
          f"合法片段数）：\n")
    for uid, n_slots, n_all, without, causes, n_only_cut, n_only_grey, n_none, grp, zlen in rows:
        print(f"  {uid}  槽位总数 {n_slots}，全禁区合法 {n_all}，"
              f"只留剪切区 {n_only_cut}，只留灰度/倒步/空洞 {n_only_grey}，全拿掉 {n_none}")
        print(f"      → 归因：{grp}")
        print(f"      禁区规模：{'；'.join(f'{k} {v} 段' for k, v in zlen.items() if v)}\n")

    cut_only = [r[0] for r in rows if r[8].startswith("剪切区一族")]
    grey_only = [r[0] for r in rows if r[8].startswith("灰度/倒步/空洞一族")]
    joint = [r[0] for r in rows if r[8].startswith("两族")]
    cap = [r[0] for r in rows if r[8].startswith("全拿掉")]
    print(f"汇总（{len(rows)} 条）：")
    print(f"  剪切剔除区卡死 {len(cut_only)} 条：{'、'.join(cut_only)}")
    print(f"  灰度帧/倒步/空洞卡死 {len(grey_only)} 条：{'、'.join(grey_only)}")
    print(f"  两族共同卡死 {len(joint)} 条：{'、'.join(joint)}")
    print(f"  容量/时长不足 {len(cap)} 条：{'、'.join(cap)}")


if __name__ == "__main__":
    main()
