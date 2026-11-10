# -*- coding: utf-8 -*-
"""核对 v3（加长采样间隔）产物，并生成肉眼可验的对比页。

查五件事：
  1. 每条视频 768 张 jpg 齐全、命名规范。
  2. 清单里 frame_stride / clip_start_frames / span_sec 三者自洽（跨度 = 首末帧
     时间戳之差，误差 < 0.05 秒）。
  3. 每个采样帧都不落在任何禁区里（剔除区间、低灰度帧、PTS 倒步、时间空洞）。
  4. 抽样回源：按记录的首帧号回到中间件解码同一帧，与落盘 jpg 比像素——确认
     「第 ci 个片段的第 fi 帧」确实来自它声称的那个中间件帧，而不是错位一帧。
     先按帧号跳到该帧取图；像素差超过 3 时，改用与抽帧脚本相同的顺序解码复核
     同一批帧号（跳帧取图在时间戳不单调的中间件上会落到相邻帧，如 v1_09 有
     一处 0.20 秒倒步）。顺序解码通过则记为备注，不计为问题。
  5. 逐片段统计新旧两版的帧间像素差，并把 16 帧排成一行放进网页，供人眼比对。

输出：
  data/process/EXP-008-units-v3-gap05/核对结果.csv
  data/process/EXP-008-units-v3-gap05/核对.html
  加 --units 时只写 核对结果-复跑.csv，不重画网页（复跑是补核个别视频，
  不该覆盖全量报告）。

生成命令：
  python experiments/common/video/verify_units_v3.py
  python experiments/common/video/verify_units_v3.py --units v1_09
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import CLIP_LEN, N_CLIPS  # noqa: E402
from build_units_stride import build_forbidden  # noqa: E402
from build_units_v2 import load_coord  # noqa: E402

V2_FRAMES = REPO / "data/process/EXP-008-units-v2/frames"
SRC256 = REPO / "data/process/EXP-008-units-src256"


def frames_of(starts, stride):
    return [(ci, st + fi * stride, fi)
            for ci, st in enumerate(starts) for fi in range(CLIP_LEN)]


def read_mw_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def sequential_frames(mw, idxs):
    """顺序解码一次，取出指定帧号对应的画面。

    与 `read_mw_frame` 的区别：不 seek，帧号按解码顺序数，因此不受 seek 精度影响。
    抽帧脚本本身就是顺序解码写的，所以这条路径给出的才是「第几个解码帧」的真值。
    只解到最大帧号为止。"""
    want = set(idxs)
    out = {}
    cap = cv2.VideoCapture(str(mw))
    fi = 0
    last = max(want)
    while fi <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in want:
            out[fi] = fr.copy()
        fi += 1
    cap.release()
    return out


def frame_diff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def clip_diff(d, ci):
    """一个片段 16 帧相邻两帧的平均像素差的中位数；缺帧返回 -1。"""
    prev, diffs = None, []
    for fi in range(CLIP_LEN):
        p = d / f"clip{ci:02d}_f{fi:02d}.jpg"
        if not p.exists():
            return -1.0
        img = cv2.imread(str(p))
        if img is None:
            return -1.0
        if prev is not None:
            diffs.append(float(np.abs(img.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = img
    return float(np.median(diffs)) if diffs else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="EXP-008-units-v3-gap05")
    ap.add_argument("--check-n", type=int, default=2, help="每条视频抽样回源的片段数")
    ap.add_argument("--units", nargs="*", default=None,
                    help="只核对这些视频（不给则全部）。给了就只写 核对结果-复跑.csv，"
                         "不重画整页 HTML——复跑是补核个别视频，不该覆盖全量报告")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root = REPO / "data/process" / args.root
    frames_root = root / "frames"
    rows = list(csv.DictReader(open(root / "unit-manifest.csv", encoding="utf-8-sig")))
    only = set(args.units or [])
    if only:
        rows = [r for r in rows if r["unit_id"] in only]
        lacking = only - {r["unit_id"] for r in rows}
        if lacking:
            raise SystemExit(f"清单里没有这些视频：{sorted(lacking)}")

    out, per_unit = [], {}
    bad_total = 0
    _, vid2cuts, _ = load_events_cuts()
    for k, r in enumerate(rows, 1):
        uid = r["unit_id"]
        stride = int(r["frame_stride"])
        starts = [int(x) for x in r["clip_start_frames"].split(";")]
        d = frames_root / uid
        problems = []
        note = ""

        # 1. 文件齐全
        n_jpg = len(list(d.glob("clip*.jpg")))
        if n_jpg != N_CLIPS * CLIP_LEN:
            problems.append(f"jpg {n_jpg}/{N_CLIPS*CLIP_LEN}")
        missing = [x for x in frames_of(starts, stride)
                   if not (d / f"clip{x[0]:02d}_f{x[2]:02d}.jpg").exists()]
        if missing:
            problems.append(f"缺 {len(missing)} 个命名位")

        # 2. 清单自洽
        n_mw, pts, _ = load_coord(uid)
        spans = [pts[st + (CLIP_LEN - 1) * stride] - pts[st] for st in starts]
        rec = float(r["span_sec"])
        if abs(float(np.median(spans)) - rec) > 0.05:
            problems.append(f"跨度记录 {rec} 与实算 {np.median(spans):.2f} 不符")
        if max(starts) + (CLIP_LEN - 1) * stride >= n_mw:
            problems.append("末帧超出中间件帧数")

        # 3. 采样帧不落禁区
        qc = list(csv.DictReader(open(REPO / "data/process/EXP-008-units-v2/qc" / f"{uid}.csv",
                                      encoding="utf-8-sig")))
        lowstd = [int(x["frame"]) for x in qc
                  if x.get("gray_std") and float(x["gray_std"]) < 20.0]
        forb = build_forbidden(uid, vid2cuts.get((r["source"], r["video_id"]), []), pts, lowstd)
        hit = [f for ci, f, fi in frames_of(starts, stride)
               if any(a <= f < b for a, b in forb)]
        if hit:
            problems.append(f"{len(hit)} 个采样帧落在禁区（首 {hit[0]}）")

        # 4. 抽样回源
        cap = cv2.VideoCapture(str(SRC256 / f"{uid}.mp4"))
        sample_ci = sorted({0, N_CLIPS // 2, N_CLIPS - 1})[:args.check_n]
        worst = 0.0
        wanted = []
        for ci in sample_ci:
            st = starts[ci]
            for fi in (0, CLIP_LEN // 2, CLIP_LEN - 1):
                idx = st + fi * stride
                wanted.append((ci, fi, idx))
                src = read_mw_frame(cap, idx)
                got = cv2.imread(str(d / f"clip{ci:02d}_f{fi:02d}.jpg"))
                if src is None or got is None:
                    problems.append(f"回源读不到 片段{ci} 帧{fi}")
                    continue
                worst = max(worst, frame_diff(src, got))
        cap.release()
        if worst > 3.0:
            # seek 可能落在相邻帧上：PTS 不单调的中间件（如 v1_09 有一处 0.20 秒倒步）
            # 会让 CAP_PROP_POS_FRAMES 跳错位置。这一步改用顺序解码复核同一批帧号——
            # 顺序解码的帧号是确定的，不受 seek 精度影响，也是抽帧脚本实际用的方式。
            # 复核通过就说明数据没问题、是 seek 的毛病；复核仍不过才是数据缺陷。
            seq = sequential_frames(SRC256 / f"{uid}.mp4", [x[2] for x in wanted])
            seq_worst = 0.0
            for ci, fi, idx in wanted:
                src = seq.get(idx)
                got = cv2.imread(str(d / f"clip{ci:02d}_f{fi:02d}.jpg"))
                if src is None or got is None:
                    seq_worst = float("inf")
                    break
                seq_worst = max(seq_worst, frame_diff(src, got))
            if seq_worst <= 3.0:
                note = (f"seek 回源差 {worst:.2f}，顺序解码复核 {seq_worst:.2f} 通过"
                        f"（该中间件 seek 会落到相邻帧，不是数据问题）")
            else:
                problems.append(f"回源像素差 seek {worst:.2f} / 顺序解码 {seq_worst:.2f}"
                                f"（应≈JPEG 噪声，<1）")
            worst = seq_worst

        # 5. 帧间变化
        v3d = [clip_diff(d, ci) for ci in range(N_CLIPS)]
        v2d = [clip_diff(V2_FRAMES / uid, ci) for ci in range(N_CLIPS)]
        v3m, v2m = float(np.median(v3d)), float(np.median(v2d))
        per_unit[uid] = {
            "stride": stride, "span": float(np.median(spans)),
            "v3": v3d, "v2": v2d, "label": r["label"], "fold": r["fold"],
            "src_fps": float(r["src_fps"]),
        }
        bad_total += bool(problems)
        out.append([uid, stride, round(float(np.median(spans)), 2), round(v3m, 2),
                    round(v2m, 2), round(worst, 2),
                    ("通过" if not problems else "；".join(problems))
                    + (f"（{note}）" if note else "")])
        print(f"  [{k}/{len(rows)}] {uid} 通过" if not problems
              else f"  [{k}/{len(rows)}] {uid} 问题：{'；'.join(problems)}", flush=True)

    out_csv = root / ("核对结果-复跑.csv" if only else "核对结果.csv")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit_id", "frame_stride", "span_sec", "v3帧间差中位",
                    "v2帧间差中位", "回源最大像素差", "结论"])
        w.writerows(out)
    if only:
        print(f"\n[核对] 复跑 {len(rows)} 条，有问题 {bad_total} 条 -> {out_csv}")
        return 0

    # ---- 网页 ----
    v3all = [x for u in per_unit.values() for x in u["v3"] if x >= 0]
    v2all = [x for u in per_unit.values() for x in u["v2"] if x >= 0]
    h = ["""<meta charset="utf-8"><title>加长间隔核对</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei";margin:0;padding:16px;background:#fafafa;color:#222}
 h1{font-size:20px;margin:0 0 8px}
 .sum{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:16px;font-size:14px;line-height:1.9}
 .vid{border:1px solid #ddd;border-radius:8px;margin:8px 0;background:#fff}
 .vid>summary{padding:10px 12px;cursor:pointer;font-size:14px}
 .row{display:grid;grid-template-columns:repeat(16,1fr);gap:2px;padding:4px 8px}
 .row img{width:100%;display:block;border-radius:2px}
 .lab{font-size:11px;color:#666;padding:0 8px 6px}
 .mono{font-variant-numeric:tabular-nums}
 .bad{color:#c00;font-weight:600}
</style>
<h1>加长采样间隔后的训练帧核对</h1>"""]
    h.append(
        f'<div class="sum"><b>{len(rows)} 条视频，每条 48 个片段 × 16 帧。</b>'
        f'有问题的视频：<b class="{"bad" if bad_total else ""}">{bad_total}</b> 条。<br>'
        f'片段内相邻采样帧的平均像素差（中位数）：<br>'
        f'　新间隔（每 0.5 秒取一帧）：全体中位 <b>{np.median(v3all):.2f}</b>，'
        f'最小 {min(v3all):.2f}，最大 {max(v3all):.2f}<br>'
        f'　原间隔（连续帧）：全体中位 <b>{np.median(v2all):.2f}</b>，'
        f'最小 {min(v2all):.2f}，最大 {max(v2all):.2f}<br>'
        f'数字越大表示这 16 帧之间画面变化越多。灰色的是没拉长的片段。<br>'
        f'点开每条视频，上排是原间隔的片段 0，下排是新间隔的片段 0——'
        f'同一条视频、同一个开头，直接看 16 帧跨过了多长时间。</div>')
    for uid in sorted(per_unit):
        u = per_unit[uid]
        h.append(f'<details class="vid"><summary><b>{uid}</b>（'
                 f'{"有泥石流" if u["label"]=="pos" else "没有泥石流"}，第 {u["fold"]} 折）：'
                 f'源 {u["src_fps"]:.1f} 帧/秒，每 {u["stride"]} 帧取一帧，'
                 f'16 帧跨 <b>{u["span"]:.2f} 秒</b>；'
                 f'片段内帧间差 新 {np.median(u["v3"]):.2f} / 原 {np.median(u["v2"]):.2f}'
                 f'</summary>'
                 f'<div class="lab">原间隔（连续 16 帧，跨 {15/u["src_fps"]:.2f} 秒）</div><div class="row">')
        for fi in range(CLIP_LEN):
            h.append(f'<img loading="lazy" src="../EXP-008-units-v2/frames/{uid}/clip00_f{fi:02d}.jpg" alt="">')
        h.append(f'</div><div class="lab">新间隔（每 {u["stride"]} 帧取一帧，跨 {u["span"]:.2f} 秒）</div><div class="row">')
        for fi in range(CLIP_LEN):
            h.append(f'<img loading="lazy" src="frames/{uid}/clip00_f{fi:02d}.jpg" alt="">')
        h.append('</div></details>')
    (root / "核对.html").write_text("\n".join(h), encoding="utf-8")
    print(f"\n[核对] 通过 {len(rows)-bad_total}/{len(rows)} 条")
    print(f"[核对] -> {root / '核对结果.csv'}")
    print(f"[核对] -> {root / '核对.html'}")


_CUTS = None


def load_events_cuts():
    global _CUTS
    if _CUTS is None:
        from build_units import load_events
        _CUTS = load_events()
    return _CUTS


if __name__ == "__main__":
    main()
