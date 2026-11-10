# -*- coding: utf-8 -*-
"""离线一致性扫描：推理侧现场重算的禁区，会不会把冻结清单认可的片段判成不可用（只读）。

为什么需要这个脚本（2026-10-28，research-auditor 决策复核前提五的配套证据）：

`infer_video.py` 从 2026-10-28 起按训练同款规则剔除落在禁区里的片段（人工标的剪辑区间、
低灰度帧、时间戳倒步、时间空洞），往返自检 `check_infer_roundtrip.py` 新增一项「禁区不得
误杀清单槽位」。但往返自检每条视频要重抽 768 帧跑一次骨干前向，只能抽查几条；本脚本不跑
模型，只解一遍时间戳、算一遍禁区，可以把 150 条全部扫一遍。

它回答两件事：
  1. 哪些视频按今天的规则会有倒步/空洞禁区（这两类禁区是 2026-10-25 才加进建库代码的，
     而清单是 2026-10-05/06 建的，所以清单槽位里可能有位置按今天的规则不合法）。
  2. 每条视频的 48 个清单槽位里，有几个落在今天的禁区里（0 = 往返自检可以照常做）。

不跑模型、不改数据，只读 `unit-manifest.csv`、`EXP-008-units-src256/*.mp4` 和
`EXP-008-units-v2/qc-*.csv`。

用法（仓库根）：
    .venv/Scripts/python.exe experiments/tools/oneoff/exp008-zone-audit/zone_consistency.py
    .venv/Scripts/python.exe experiments/tools/oneoff/exp008-zone-audit/zone_consistency.py v1_09 v1_109
    .venv/Scripts/python.exe experiments/tools/oneoff/exp008-zone-audit/zone_consistency.py --csv out.csv

退出码：0 = 全部扫通（有被误杀的也算扫通，误杀是结论不是故障）；1 = 有条目没扫成
（取不到时间戳、帧数对不上、视频或清单缺失）。
"""
import argparse
import csv
import sys
from pathlib import Path


def find_repo() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "experiments/tools/infer_video.py").is_file():
            return p
    raise SystemExit("找不到仓库根：从本文件往上没有 experiments/tools/infer_video.py")


REPO = find_repo()
sys.path.insert(0, str(REPO / "experiments/tools"))
sys.path.insert(0, str(REPO / "experiments/common/video"))

import infer_video as iv                                   # noqa: E402
from build_units import CLIP_LEN, slot_ok                   # noqa: E402

MAN = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
SRC = REPO / "data/process/EXP-008-units-src256"


def scan_one(uid, row):
    """返回一行结果字典；fatal 非空表示这条没扫成。"""
    video = SRC / f"{uid}.mp4"
    res = {"unit_id": uid, "n_slots": "", "n_jump": "", "n_gap": "",
           "n_zone_ranges": "", "n_killed": "", "killed_frames": "", "fatal": ""}
    if not video.is_file():
        res["fatal"] = "缺 src256 中间件"
        return res
    pts = iv.probe_pts(video)
    if pts is None:
        res["fatal"] = "取不到逐帧时间戳"
        return res
    n_mw = int(row["n_frames_mw"])
    # 与 infer_video 同一条容差：容器帧数与解码器实际吐出的帧数允许差一帧（v1_09 实测
    # 20550 vs 20551）；差得更多说明两条解码路径对不上，时间轴不可比。
    if abs(len(pts) - n_mw) > 1:
        res["fatal"] = f"时间戳 {len(pts)} 条 vs 清单 {n_mw} 帧"
        return res
    slots = [int(x) // CLIP_LEN for x in row["clip_start_frames"].split(";") if x]
    forb, n_jump, n_gap = iv.training_zones(video, pts, len(pts))
    killed = [s * CLIP_LEN for s in slots if not slot_ok(s, forb)]
    res.update(n_slots=len(slots), n_jump=n_jump, n_gap=n_gap,
               n_zone_ranges=len(forb), n_killed=len(killed),
               killed_frames=";".join(str(x) for x in killed))
    return res


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("units", nargs="*", help="只扫这些 unit_id（默认全部）")
    ap.add_argument("--csv", help="把逐条结果写成 CSV（便于复算与引用）")
    args = ap.parse_args(argv)

    with open(MAN, encoding="utf-8-sig", newline="") as fh:
        man = {r["unit_id"]: r for r in csv.DictReader(fh)}
    todo = args.units or sorted(man)
    rows, n_fatal = [], 0
    for i, uid in enumerate(todo, 1):
        if uid not in man:
            print(f"[{i}/{len(todo)}] {uid}: 不在 unit-manifest.csv 里")
            n_fatal += 1
            continue
        r = scan_one(uid, man[uid])
        rows.append(r)
        if r["fatal"]:
            n_fatal += 1
            print(f"[{i}/{len(todo)}] 未扫成 {uid}: {r['fatal']}", flush=True)
        else:
            tag = "OK  " if not r["n_killed"] else "剔除"
            print(f"[{i}/{len(todo)}] {tag} {uid} 槽位{r['n_slots']} 倒步{r['n_jump']} "
                  f"空洞{r['n_gap']} 禁区{r['n_zone_ranges']}段 "
                  f"落在禁区里的清单槽位{r['n_killed']}个"
                  + (f" {r['killed_frames']}" if r["n_killed"] else ""), flush=True)

    ok = [r for r in rows if not r["fatal"]]
    with_zone = [r for r in ok if r["n_jump"] or r["n_gap"]]
    killed = [r for r in ok if r["n_killed"]]
    print(f"\n共 {len(todo)} 条：扫通 {len(ok)} 条，未扫成 {n_fatal} 条。")
    print(f"其中 {len(with_zone)} 条按今天的规则有倒步或空洞禁区；"
          f"{len(killed)} 条有清单槽位落在禁区里（这些不能拿去做往返自检）。")
    if killed:
        print("有清单槽位落在禁区里的：" + "、".join(
            f"{r['unit_id']}({r['n_killed']})" for r in killed))
    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"逐条结果已写入 {args.csv}")
    return 1 if n_fatal else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.exit(main(sys.argv[1:]))
