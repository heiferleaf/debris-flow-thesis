# -*- coding: utf-8 -*-
"""只算容量、不写帧：某批视频在指定采样间隔下能放下多少个合法片段。

回答的问题：换一套中间件（换解码基座）或换一个采样间隔，某些视频还凑不凑得满
48 个片段。判据与 `experiments/common/video/build_units_stride.py` 完全同一套
（整块不入禁区；禁区 = 剪辑区间外扩 1 秒 + 灰度低于 20 的帧前后各 16 帧 +
时间戳倒退邻域 + 时间空洞前后 1 秒）。这里算出的是**容量上界**，不是那边最终落进
池子的条数：抽帧脚本在选槽之后还要过严格标签等筛选，落进池子的片段不会多于这里
算出的数。写进文档时按上界表述，不要写成「等于」。

只读中间件、坐标表、灰度表；除 `--out` 指定的表以外不写任何文件。

用法（仓库根，逐条核对 2026-10-26 的 19 条排除理由）：

  .venv/Scripts/python.exe experiments/tools/check_clip_capacity.py \
      --base 软解 --src data/process/EXP-008-units-src256 \
      --qc-dir data/process/EXP-008-units-src256/qc --gap 0.5 \
      --units v1_19,v1_39 --out data/process/EXP-008-units-v3-gap05/容量核对.csv

  # 第二个基座追加到同一张表
  ... --base 显卡 --src data/process/EXP-008-units-src256-hw \
      --qc-dir data/process/EXP-008-units-src256-hw/qc --units ... --append
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/common/video"))

from build_units import N_CLIPS, QC_STD_LOW, load_events, read_csv_rows  # noqa: E402
from build_units_stride import (blocks_for, build_forbidden,  # noqa: E402
                                pick_slots)

V2 = REPO / "data/process/EXP-008-units-v2"
COLS = ["base", "unit_id", "n_frames", "fps", "stride", "block", "n_slots",
        "slots", "reach48", "n_forbidden", "forbidden_cover", "low_grey_frames",
        "error"]


def _repo_path(p):
    """三个路径参数都按仓库根解析，不按当前目录。

    原来 --out 按仓库根、--src 与 --qc-dir 按当前目录，换一个目录执行就会读到
    「找不到坐标表」而 --out 仍写对地方，报警与产物对不上（2026-10-26 审查 S3）。
    """
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gap", type=float, default=0.5, help="相邻采样帧的间隔（秒）")
    ap.add_argument("--src", required=True, help="中间件目录（含 coord/）")
    ap.add_argument("--qc-dir", required=True, help="逐帧灰度表目录")
    ap.add_argument("--units", required=True, help="逗号分隔的 unit_id")
    ap.add_argument("--base", default="", help="写进表里的基座名，如 软解 / 显卡")
    ap.add_argument("--out", required=True, help="结果表路径")
    ap.add_argument("--append", action="store_true", help="追加到已有的表")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    src, qcd = _repo_path(args.src), _repo_path(args.qc_dir)
    want = [s.strip() for s in args.units.split(",") if s.strip()]
    _, vid2cuts, _ = load_events()
    man = {r["unit_id"]: r for r in
           csv.DictReader(open(V2 / "unit-manifest.csv", encoding="utf-8-sig"))}

    print(f"间隔 {args.gap}s，基座 {args.base or src.name}", flush=True)
    rows = []
    for uid in want:
        row = man[uid]
        coord, qc = src / "coord" / f"{uid}.csv", qcd / f"{uid}.csv"
        if not coord.exists() or not qc.exists():
            print(f"  ! {uid} 缺坐标表或灰度表")
            rows.append({"base": args.base, "unit_id": uid, "error": "缺文件"})
            continue
        pts = [float(r["pts_time"]) for r in read_csv_rows(coord)]
        n = len(pts)
        qc_rows = read_csv_rows(qc)
        align_err = None
        if len(qc_rows) != n:
            # 灰度表与坐标表必须指同一套帧。行数不等有两种情况，必须分开处理：
            #   ① 仅仅尾部少了若干行（实测 v1_09 少最后一帧），帧号与时间戳一一对得上，
            #      不影响禁区位置，可以继续算，但要打印出来；
            #   ② 整表来自另一次转码或另一条视频，行数可能相等、也可能不等，靠帧号
            #      同一位置的时间戳探测能露出来——这时帧号错位，算出的禁区是错的，必须停下。
            # 探测容差取 0.5 秒：同一套转码的行尾会有一帧级的小漂移（实测 v1_09 尾部差
            # 0.04 秒，起因是灰度表把末尾一帧的 pts 记到了前一帧上），错位一次转码则至少
            # 差几百毫秒到几秒，两者能分开。行数不等时只打印不阻断，等号时连打印也省掉。
            # 只比行数不比时间戳，会把 ① 误判成 ②；只比时间戳不比行数，查不出 ①。
            probes = sorted({0, len(qc_rows) // 2, len(qc_rows) - 1})
            off = []
            for i in probes:
                try:
                    qp = float(qc_rows[i]["pts_time"])
                    cp = pts[int(qc_rows[i]["frame"])]
                except (KeyError, ValueError, IndexError):
                    off.append(f"第 {i} 行读不出帧号或时间戳")
                    continue
                if abs(qp - cp) > 0.5:
                    off.append(f"第 {i} 行 {qp:.2f}s 对坐标表 {cp:.2f}s")
            if off:
                align_err = f"灰度表与坐标表不是同一套帧（{'; '.join(off[:2])}）"
            else:
                print(f"  ! {uid} 灰度表 {len(qc_rows)} 行比坐标表 {n} 行少 "
                      f"{n - len(qc_rows)} 行，抽查帧号与时间戳吻合，按现有行继续")
        if align_err:
            print(f"  ! {uid} {align_err}")
            rows.append({"base": args.base, "unit_id": uid, "error": align_err})
            continue
        lowstd = [int(r["frame"]) for r in qc_rows
                  if r.get("gray_std") and float(r["gray_std"]) < QC_STD_LOW]
        fps = (n - 1) / pts[-1] if pts[-1] > 0 else 10.0
        stride = max(1, int(round(args.gap * fps)))
        block = blocks_for(stride)
        n_slots = n // block
        forb = build_forbidden(uid, vid2cuts.get((row["source"], row["video_id"]), []),
                               pts, lowstd)
        slots = pick_slots(n_slots, block, forb)
        cover = sum(min(b, n) - min(a, n) for a, b in forb if b > a) / n
        rows.append({"base": args.base, "unit_id": uid, "n_frames": n,
                     "fps": round(fps, 2), "stride": stride, "block": block,
                     "n_slots": n_slots, "slots": len(slots),
                     "reach48": "是" if len(slots) >= N_CLIPS else "否",
                     "n_forbidden": len(forb), "forbidden_cover": round(cover, 4),
                     "low_grey_frames": len(lowstd)})
        print(f"  {uid} 帧 {n} 步长 {stride} 槽 {n_slots} 合法片段 {len(slots)} "
              f"{'够' if len(slots) >= N_CLIPS else '不够'} 禁区占比 {cover:.1%}")

    out = _repo_path(args.out)
    old = []
    if args.append and out.exists():
        old = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(old + rows)
    ok = [r for r in rows if "error" not in r]
    print(f"\n够 48 组的 {sum(r['reach48'] == '是' for r in ok)}/{len(ok)} 条 -> {out}")


if __name__ == "__main__":
    main()
