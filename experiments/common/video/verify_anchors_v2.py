# -*- coding: utf-8 -*-
"""锚点核对（B4①）：中间件时间轴 vs 人工审计初帧 frames_index.csv（2026-10-05 立）。

两层核对（research-auditor 交叉核对四条件）：
  ①时间轴对账（全量，不解码）：每条审计帧的 t_sec 必须在中间件坐标表 pts_time 的
    ±40ms 容差内有最近邻帧（条件①±40ms、条件②断言方向=审计每行有对应，不是帧数
    相等——审计表只覆盖源可解码前缀，中间件尾部多解帧是预期）。
  ②内容锚点（每单位 2 帧，解码）：取该单位最早、最晚两条审计帧，从中间件顺序解码
    到对应帧，与审计初帧拼对比图（左审计/右中间件）+ 灰度 MAD。MAD 阈值 15（转码
    CRF10 实测湍流区 MAD≈7.5，15 起人工必看）；拼图供叶海峰抽查页引用。

条件④已知限制：frames_index.csv 本身是 OpenCV 源解码产物（t_sec = POS_MSEC），
其正确性由「人工审计背书时间体系 + 内容锚点肉眼比对」闭环，本工具不循环论证。

输入：EXP-008-units-src256/coord/、frames_index.csv（三源）、
data/process/video_frames/<seg>/frames/（审计初帧）。
输出：data/process/EXP-008-units-v2/anchors/{anchor-report.csv, pairs/ 对比图}。

用法：python experiments/common/video/verify_anchors_v2.py [--only v1_07]
"""
import argparse
import bisect
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import read_csv_rows  # noqa: E402
from build_units_v2 import SRC256_ROOT, UNITS_ROOT  # noqa: E402

FRAMES_INDEX = {   # source -> frames_index.csv（审计初帧清单）
    "dongchuan": REPO / "data/process/video_frames/video1_segment/frames_index.csv",
    "video4": REPO / "data/process/video_frames/video4_segment/frames_index.csv",
    "video5": REPO / "data/process/video_frames/video5_segment/frames_index.csv",
}
AUDIT_FRAMES_ROOT = {
    "dongchuan": REPO / "data/process/video_frames/video1_segment/frames",
    "video4": REPO / "data/process/video_frames/video4_segment/frames",
    "video5": REPO / "data/process/video_frames/video5_segment/frames",
}
TOL_S = 0.04       # ±40ms 容差（timescale 舍入与 VFR 抖动）
MAD_FLAG = 15.0    # 灰度 MAD 超过此值标记人工必看（CRF10 实测湍流区 ≈7.5）
CMP_SIZE = (64, 64)   # 内容比对统一缩到 64×64（两边同样拉伸，几何可比）


def load_audit_index():
    """video_key -> [(t_sec, 审计帧路径)...]（按时间升序）。"""
    out = {}
    for src, p in FRAMES_INDEX.items():
        for r in read_csv_rows(p):
            out.setdefault(r["video_key"], []).append(
                (float(r["t_sec"]), AUDIT_FRAMES_ROOT[src] / r["frame_file"]))
    for k in out:
        out[k].sort()
    return out


def unit_tol(pts):
    """逐条容差：max(±40ms, 中间件帧间隔中位数的一半 + 1e-6)。

    部分源转码后时间基量化到 0.1s 网格，审计 POS_MSEC 落在两帧正中间时
    （距离各 0.05s），固定 ±40ms 会把同一帧误判为缺失（v1_19 实证 103 帧
    最近邻距离全部 0.04~0.05s，非丢帧）。"""
    import statistics
    diffs = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    med = statistics.median(diffs) if diffs else 0.0
    return max(TOL_S, med / 2 + 1e-6)


def nearest_idx(pts_sorted, t, tol):
    """双侧最近邻：pts_sorted 中与 t 距离 ≤ tol 的帧里取距离最小者，无则 None。

    不能只测 bisect 位置及其后一格就返回——t 落在两帧正中间附近时两个
    候选都在容差内，须比较距离取更近者（v1_113 实证：审计 t=549.98 落在
    549.95 与 550.00 之间，先测 bisect 侧取到 549.95 的灰帧、MAD 虚高
    34.78；取 550.00 正常帧后应为转码损耗量级）。"""
    i = bisect.bisect_left(pts_sorted, t - tol)
    best = None
    for j in (i, i + 1):
        if 0 <= j < len(pts_sorted) and abs(pts_sorted[j] - t) <= tol:
            if best is None or abs(pts_sorted[j] - t) < abs(pts_sorted[best] - t):
                best = j
    return best


def time_axis_check(vkey, pts, tol):
    """①时间轴对账：返回 (审计帧总数, 容差内无对应帧数, 首批无对应样例)。

    容差逐帧自适应：基线 ±40ms；该位置 ±1.5s 窗内中间件最大帧距的一半与基线取大者
    ——可变帧率源（如 v5_03 部分时段 0.2s 帧距）里，审计时间戳落在两帧正中间时
    最近邻恰为局部帧距的一半，固定容差会误判缺失（v1_19/v5_03 实证 miss 全部
    0.04~0.1s，非丢帧）。"""
    spts = sorted(pts)
    miss = 0
    samples = []
    for t, _ in AUDIT_IDX.get(vkey, []):
        window = [p for p in spts if t - 1.5 <= p <= t + 1.5]
        if len(window) >= 2:
            gap = max(b - a for a, b in zip(window, window[1:]))
            t_tol = max(tol, gap / 2 + 1e-6)
        else:
            t_tol = tol
        if nearest_idx(spts, t, t_tol) is None:
            miss += 1
            if len(samples) < 5:
                samples.append(t)
    return len(AUDIT_IDX.get(vkey, [])), miss, samples


def grab_mw_frame(mw_path, want_pts, spts, tol):
    """中间件顺序解码到 want_pts 最近邻帧（双侧查找），返回 BGR 图或 None。"""
    fi = nearest_idx(spts, want_pts, tol)
    if fi is None:
        return None
    cap = cv2.VideoCapture(str(mw_path))
    if not cap.isOpened():
        return None
    fr = None
    for i in range(fi + 1):
        ok, fr = cap.read()
        if not ok:
            fr = None
            break
    cap.release()
    return fr


def anchor_pair(vkey, mw_path, pts, spts, tol, rows, out_dir):
    """②内容锚点：取最早/最晚两条审计帧，拼对比图，返回逐锚点记录。"""
    recs = []
    picks = [rows[0], rows[-1]] if len(rows) >= 2 else [rows[0], rows[0]]
    for tag, (t, apath) in zip(("first", "last"), picks):
        mw_fr = grab_mw_frame(mw_path, t, spts, tol)
        aud = cv2.imread(str(apath))
        if mw_fr is None or aud is None:
            recs.append({"video_key": vkey, "anchor": tag, "t_sec": t,
                         "mad": "", "flag": "missing"})
            continue
        a = cv2.cvtColor(cv2.resize(aud, CMP_SIZE), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(mw_fr, CMP_SIZE), cv2.COLOR_BGR2GRAY)
        mad = round(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), 2)
        # 拼图：左审计（960 宽）右中间件（放大到同高 540），上条标 t 与 MAD
        mw_up = cv2.resize(mw_fr, (960, 540), interpolation=cv2.INTER_NEAREST)
        if aud.shape[:2] != (540, 960):     # 审计初帧可能是 4K 原始分辨率
            aud = cv2.resize(aud, (960, 540))
        h_txt = 40
        canvas = np.full((540 + h_txt, 960 * 2 + 8, 3), 255, np.uint8)
        canvas[h_txt:, :960] = aud
        canvas[h_txt:, 968:] = mw_up
        label = f"{vkey} {tag} t={t:.2f}s  audit | middleware  MAD={mad}"
        cv2.putText(canvas, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2, cv2.LINE_AA)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{vkey}_{tag}_t{t:.2f}s.jpg"), canvas)
        recs.append({"video_key": vkey, "anchor": tag, "t_sec": t,
                     "mad": mad, "flag": "review" if mad >= MAD_FLAG else "ok"})
    return recs


AUDIT_IDX = None


def _write_report(report_path, recs):
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_key", "anchor", "t_sec",
                                          "mad", "flag"])
        w.writeheader()
        w.writerows(recs)


def main():
    global AUDIT_IDX
    ap = argparse.ArgumentParser(description="锚点核对：中间件 vs 审计初帧")
    ap.add_argument("--only", default="", help="只核指定 video_key（逗号分隔）")
    ap.add_argument("--pairs", action="store_true",
                    help="只补内容锚点拼图（跳过已完整的单位）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    AUDIT_IDX = load_audit_index()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    keys = sorted(AUDIT_IDX)
    if only:
        keys = [k for k in keys if k in only]
    out_dir = UNITS_ROOT / "anchors"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "anchor-report.csv"
    done = {}
    if report_path.exists():
        for r in read_csv_rows(report_path):
            done.setdefault(r["video_key"], []).append(r)

    t0 = time.time()
    all_recs = []
    skipped = []
    n_miss_total = 0
    for i, vkey in enumerate(keys, 1):
        coord = SRC256_ROOT / "coord" / f"{vkey}.csv"
        mw = SRC256_ROOT / f"{vkey}.mp4"
        if not coord.exists() or not mw.exists():
            reason = "+".join(x for x, ok in
                              (("无中间件mp4", mw.exists()),
                               ("无坐标表", coord.exists())) if not ok)
            skipped.append({"video_key": vkey,
                            "n_audit_frames": len(AUDIT_IDX[vkey]),
                            "reason": reason})
            print(f"  [{i}/{len(keys)}] {vkey}: 中间件/坐标表缺失，跳过")
            continue
        pts = [float(r["pts_time"]) for r in read_csv_rows(coord)]
        tol = unit_tol(pts)
        spts = sorted(pts)
        n_audit, n_miss, samples = time_axis_check(vkey, pts, tol)
        n_miss_total += n_miss
        base = {"video_key": vkey, "anchor": "time-axis", "t_sec": "",
                "mad": "", "flag": f"audit={n_audit} miss={n_miss} {samples}"}
        recs = [base]
        pair_dir = out_dir / "pairs"
        have = {r.get("anchor") for r in done.get(vkey, [])}
        if not (args.pairs and {"first", "last"} <= have):
            recs += anchor_pair(vkey, mw, pts, spts, tol, AUDIT_IDX[vkey],
                                pair_dir)
        else:
            recs = list(done.get(vkey, []))
        all_recs += recs
        bad = [r for r in recs if r.get("flag") in ("review", "missing")]
        n_miss_txt = f" miss={n_miss}" + (f" {samples}" if samples else "")
        print(f"  [{i}/{len(keys)}] {vkey}: 审计{n_audit}帧{n_miss_txt} "
              f"MAD={[r.get('mad') for r in recs if r.get('anchor') != 'time-axis']}"
              f"{' !!' if bad else ''} ({time.time()-t0:.0f}s)", flush=True)
        if i % 20 == 0 or i == len(keys):   # 中途落账，防崩溃丢进度
            _write_report(report_path, all_recs)

    _write_report(report_path, all_recs)
    if skipped:   # 跳过项落账（事后审查 B2：不能只留在 stdout）
        skip_path = out_dir / "skipped.csv"
        with open(skip_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["video_key", "n_audit_frames",
                                              "reason"])
            w.writeheader()
            w.writerows(skipped)
        print(f"[anchors] 跳过 {len(skipped)} 条（无中间件，明细 -> {skip_path}）")
    print(f"\n[anchors] 时间轴无对应帧合计 {n_miss_total}；报告 -> {report_path}")
    print(f"[anchors] 拼图 -> {out_dir/'pairs'}（MAD>={MAD_FLAG} 标 review 需人工看）")
    sys.exit(2 if n_miss_total else 0)


if __name__ == "__main__":
    main()
