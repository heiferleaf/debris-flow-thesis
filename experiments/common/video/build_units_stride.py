# -*- coding: utf-8 -*-
"""EXP-008 v3：按「加长采样间隔」重建片段帧（research-auditor 2026-10-25 方案甲）。

与 v2 的唯一差别：一个片段里的 16 帧不再是相邻帧，而是每隔 STRIDE 帧取一帧。
STRIDE = 目标间隔（秒）× 源视频帧率，四舍五入到整帧。源视频帧率不同，实际
间隔就略有差异；跨度（15×STRIDE÷帧率）约等于目标秒数。每条视频的实际步长、
跨度、帧率逐条写进清单，不把「0.5 秒」当成事实断言。

四条硬约束（research-auditor 2026-10-25 阻断 B2–B4 的整改）：
  1. 整个取帧块 [s*BLOCK, s*BLOCK+BLOCK) 不得与任何禁区相交。禁区 = 剔除区间
     （外扩 1 秒）+ 低灰度帧前后各 16 帧 + PTS 倒步邻域 + 时间空洞前后各 1 秒。
     按整块判定而非逐帧判定，是为了和容量估算脚本同一口径（整块干净 ⇒
     16 个采样帧干净）。
  2. 抽帧按显式帧号列表取，不再用 slot*CLIP_LEN+fi 的连续式。
  3. 一个片段的首末帧时间戳由清单里的 frame_stride 反推，标签脚本据此判定。
  4. 凑不满 48 个片段的视频整条排除，落 unit-failures.csv，不静默少给。

v1_109 坏区块、解码失败后加宽禁区重采、读失败后继续 grab，三处与 v2 保持一致。

生成命令（依据 = research-auditor 2026-10-25 决策复核「方案甲」）：
  python experiments/common/video/build_units_stride.py --gap 0.5
  python experiments/common/video/build_units_stride.py --gap 0.5 --only v1_07
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import (CLIP_LEN, N_CLIPS, QC_STD_LOW,  # noqa: E402
                         VIDEO_KEY_OF_V1_109, V1_109_BAD_FROM_S,
                         load_events, merge_ranges, read_csv_rows)
from build_units_v2 import (pts_gap_zones, pts_jump_zones,  # noqa: E402
                            zones_from_frames, secs_to_ranges)

SRC256 = REPO / "data/process/EXP-008-units-src256"
V2 = REPO / "data/process/EXP-008-units-v2"
MAX_PICK_RETRY = 6


def blocks_for(stride):
    """步长（帧）-> 一个片段占用的帧数。16 帧之间有 15 个间隔。"""
    return (CLIP_LEN - 1) * stride + 1


def slot_ok(sid, block, forb):
    """整块 [sid*block, sid*block+block) 与全部禁区不相交。"""
    a, b = sid * block, (sid + 1) * block
    return all(b <= fa or a >= fb for fa, fb in forb)


def pick_slots(n_slots, block, forb, k=N_CLIPS):
    """等间隔取目标槽，不合法就往两边就近挪。返回升序槽号，可能少于 k 个。"""
    k = min(k, n_slots)
    targets = [int(round(x)) for x in np.linspace(0, n_slots - 1, k)]
    chosen = []
    for t in targets:
        if slot_ok(t, block, forb) and t not in chosen:
            chosen.append(t)
            continue
        for d in range(1, n_slots):
            hit = next((s for s in (t - d, t + d)
                        if 0 <= s < n_slots and s not in chosen
                        and slot_ok(s, block, forb)), None)
            if hit is not None:
                chosen.append(hit)
                break
        else:
            break          # 一个目标挪不到合法位置，停止，由调用方判容量不足
    return sorted(chosen)


def frame_list(slots, block, stride):
    """槽号列表 -> [(片段序号, 中间件帧号, 片段内帧序号)]。"""
    return [(ci, s * block + fi * stride, fi)
            for ci, s in enumerate(slots) for fi in range(CLIP_LEN)]


def build_forbidden(uid, cuts, pts, lowstd):
    n = len(pts)
    forb = list(secs_to_ranges(cuts, pts))
    if uid == VIDEO_KEY_OF_V1_109:
        fb = next((j for j, t in enumerate(pts) if t >= V1_109_BAD_FROM_S), n)
        if fb < n:
            forb.append((fb, n))
    forb += zones_from_frames(lowstd, n)
    forb += pts_jump_zones(pts)
    forb += pts_gap_zones(pts)
    return merge_ranges(forb)


def extract(mw, frames, unit_dir, resume, expect_key):
    """顺序解码一遍，只写列出的帧。返回 (写出数, 扫描数, 首个失败帧, 失败帧列表)。

    扫描数 -1 表示断点续传直接跳过、本轮没扫。"""
    unit_dir.mkdir(parents=True, exist_ok=True)
    sidecar = unit_dir / "slots.txt"
    if (resume and sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() == expect_key
            and len(list(unit_dir.glob("clip*.jpg"))) == N_CLIPS * CLIP_LEN):
        return N_CLIPS * CLIP_LEN, -1, None, []
    need = {fno: (ci, fi) for ci, fno, fi in frames}
    last_need = max(need)
    cap = cv2.VideoCapture(str(mw))
    if not cap.isOpened():
        raise RuntimeError(f"中间件打不开：{mw}")
    n_write = n_scan = 0
    fail_at, fail_frames = None, []
    fi = 0
    while True:
        if fail_at is None and fi > last_need:
            break
        ok, fr = (cap.read() if fail_at is None else (cap.grab(), None))
        if not ok or (fail_at is None and fr is None):
            fail_frames.append(fi)
            if fail_at is None:
                fail_at = fi
                fi += 1
                continue
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != fi + 1:
            fail_frames.append(fi)
            if fail_at is None:
                fail_at = fi
            break
        n_scan += 1
        if fail_at is None and fi in need:
            ci, k = need[fi]
            cv2.imwrite(str(unit_dir / f"clip{ci:02d}_f{k:02d}.jpg"), fr)
            n_write += 1
        fi += 1
    if fail_at is None:
        while cap.grab():
            n_scan += 1
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != fi + 1:
                fail_frames.append(fi)
                break
            fi += 1
    cap.release()
    return n_write, n_scan, fail_at, fail_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.5, help="相邻采样帧的间隔（秒）")
    ap.add_argument("--out", default="EXP-008-units-v3-gap05")
    ap.add_argument("--only", default="")
    ap.add_argument("--qc-dir", default="",
                    help="逐帧灰度质检表目录，默认 <v2>/qc。中间件重新转过的视频必须"
                         "指向用新中间件重算的表，否则禁区与实际像素对不上")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    out_root = REPO / "data/process" / args.out
    frames_root = out_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None

    _, vid2cuts, _ = load_events()
    man = list(csv.DictReader(open(V2 / "unit-manifest.csv", encoding="utf-8-sig")))
    print(f"[stride] 目标间隔 {args.gap}s，候选 {len(man)} 条 -> {out_root}", flush=True)

    ok_rows, fail_rows = [], []
    t0 = time.time()
    for i, row in enumerate(man, 1):
        uid = row["unit_id"]
        if only and uid not in only:
            continue
        mw = SRC256 / f"{uid}.mp4"
        coord = SRC256 / "coord" / f"{uid}.csv"
        qc = (Path(args.qc_dir) / f"{uid}.csv" if args.qc_dir
              else V2 / "qc" / f"{uid}.csv")
        if not mw.exists() or not coord.exists() or not qc.exists():
            # 循环里当场打印（research-auditor 2026-10-25 S5）：只在末尾汇总时，
            # 跑的过程中看不出少了哪几条，日志本身说明不了这 19 条的来源。
            print(f"  [{i}/{len(man)}] - {uid} 排除：无软解中间件、坐标表或质检表",
                  flush=True)
            fail_rows.append({"unit_id": uid, "reason": "无软解中间件、坐标表或质检表"})
            continue
        pts = [float(r["pts_time"]) for r in read_csv_rows(coord)]
        n = len(pts)
        qc_rows = read_csv_rows(qc)
        lowstd = [int(r["frame"]) for r in qc_rows
                  if r.get("gray_std") and float(r["gray_std"]) < QC_STD_LOW]
        if len(qc_rows) != n:
            # v1_09 的质检表少最后一帧（20550 行对 20551 帧）。缺记录的帧无法
            # 证明可用，按低灰度帧处理并入禁区；只接受「恰好缺末尾若干帧」这一
            # 种情况，其它行数不符一律视为跨解码基座错位，直接报错。
            missing = list(range(len(qc_rows), n))
            if missing != list(range(n - len(missing), n)) or len(missing) > 4:
                raise RuntimeError(f"{uid}: 质检 {len(qc_rows)} 行与坐标 {n} 帧不等，"
                                   f"且不是缺末尾几帧，跨解码基座复用会错位")
            lowstd += [f for f in missing if f not in lowstd]
            print(f"  [{i}/{len(man)}] ! {uid} 质检缺末尾 {len(missing)} 帧"
                  f"（{missing}），已按不可用处理", flush=True)

        fps = (n - 1) / pts[-1] if pts[-1] > 0 else 10.0
        stride = max(1, int(round(args.gap * fps)))
        block = blocks_for(stride)
        n_slots = n // block
        forb0 = build_forbidden(uid, vid2cuts.get((row["source"], row["video_id"]), []),
                                pts, lowstd)
        slots = pick_slots(n_slots, block, forb0)
        if len(slots) < N_CLIPS:
            fail_rows.append({"unit_id": uid,
                              "reason": f"合法片段 {len(slots)}<{N_CLIPS}"
                                        f"（{args.gap}s 跨度需连续 {block} 帧，全片 {n} 帧）"})
            print(f"  [{i}/{len(man)}] - {uid} 排除：{len(slots)} 个合法片段", flush=True)
            continue

        unit_dir = frames_root / uid
        forb, retries = forb0, 0
        nw, nsc, fail_at, bad = 0, -1, None, []
        while True:
            key = ";".join(str(s * block) for s in slots)
            nw, nsc, fail_at, bad = extract(mw, frame_list(slots, block, stride),
                                            unit_dir, resume=True, expect_key=key)
            if fail_at is None and nw == N_CLIPS * CLIP_LEN:
                break
            if retries >= MAX_PICK_RETRY:
                break
            forb = merge_ranges(forb + zones_from_frames(bad or [fail_at], n, pad=16 * 2 ** retries))
            slots = pick_slots(n_slots, block, forb)
            for f in unit_dir.glob("clip*.jpg"):
                f.unlink()
            if len(slots) < N_CLIPS:
                break
            retries += 1
        if fail_at is not None or nw != N_CLIPS * CLIP_LEN or len(slots) < N_CLIPS:
            fail_rows.append({"unit_id": uid,
                              "reason": f"重采 {retries} 次仍不满 768 帧"
                                        f"（写出 {nw}，失败帧位 {fail_at}，合法片段 {len(slots)}）"})
            print(f"  [{i}/{len(man)}] - {uid} 排除：{fail_rows[-1]['reason']}", flush=True)
            continue
        if nsc != -1 and nsc != n:
            fail_rows.append({"unit_id": uid, "reason": f"扫描 {nsc}/{n} 帧，帧数不符"})
            continue

        starts = [s * block for s in slots]
        (unit_dir / "slots.txt").write_text(";".join(str(x) for x in starts), encoding="utf-8")
        spans = [pts[a + (CLIP_LEN - 1) * stride] - pts[a] for a in starts]
        ok_rows.append({
            "unit_id": uid, "source": row["source"], "video_id": row["video_id"],
            "group_id": row["group_id"], "fold": row["fold"], "label": row["label"],
            "n_clips": N_CLIPS, "clip_len": CLIP_LEN,
            "target_gap_sec": args.gap, "frame_stride": stride,
            "span_sec": round(float(np.median(spans)), 3),
            "src_fps": round(fps, 2), "n_frames_mw": n,
            "clip_start_frames": ";".join(str(x) for x in starts),
        })
        print(f"  [{i}/{len(man)}] + {uid} 步长 {stride} 帧 跨度 {np.median(spans):.2f}s "
              f"重采 {retries} 次", flush=True)

    cols = ["unit_id", "source", "video_id", "group_id", "fold", "label", "n_clips",
            "clip_len", "target_gap_sec", "frame_stride", "span_sec", "src_fps",
            "n_frames_mw", "clip_start_frames"]
    with open(out_root / "unit-manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(ok_rows)
    with open(out_root / "unit-failures.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["unit_id", "reason"])
        w.writeheader()
        w.writerows(fail_rows)
    print(f"[stride] 完成 {len(ok_rows)} 条入池，{len(fail_rows)} 条排除，"
          f"用时 {time.time() - t0:.0f} 秒", flush=True)
    if ok_rows:
        sp = sorted(r["span_sec"] for r in ok_rows)
        print(f"[stride] 跨度分布：最小 {sp[0]:.2f}s，中位 {sp[len(sp)//2]:.2f}s，"
              f"最大 {sp[-1]:.2f}s；跨度 <2 秒（等于没拉长）的 "
              f"{sum(1 for x in sp if x < 2)} 条", flush=True)
    for r in fail_rows:
        print(f"    排除 {r['unit_id']}：{r['reason']}", flush=True)


if __name__ == "__main__":
    main()
