# -*- coding: utf-8 -*-
"""EXP-008 显卡解码版中间件生成（2026-10-25 立）。

与 build_units_src256.py 的唯一差别：在 ffmpeg 命令前置 -hwaccel d3d11va。
输出到独立新目录 EXP-008-units-src256-hw/，不覆盖旧软件解码中间件。

实测依据（2026-10-25 batch_hw_probe.py）：
  - 18 条鸡冠山上游同机位视频：显卡解码低灰度帧 0%、PTS 单调、空洞 5~26%
  - v1_39：显卡解码后仍 25.6% 低灰度帧（真实暗画面，不转）
  - v1_09：PTS 有 107 处非单调（单独排查，先不转）

用法：
  python build_units_src256_hw.py --only v1_45,v1_46,...  # 转指定清单
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/common/video"))

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SRC256_HW = REPO / "data/process/EXP-008-units-src256-hw"


def unit_video_rows():
    from build_units import build_video_index, load_events, locate_raw_video
    vid2key, _, _ = load_events()
    out = []
    for (src, video_id), info in sorted(build_video_index().items()):
        vkey = vid2key.get((src, video_id))
        if vkey is None:
            continue
        vp = locate_raw_video(video_id)
        if vp is None:
            continue
        out.append((vkey, Path(vp)))
    return out


def run_ffmpeg_hw(unit_id, src, dst, log_path):
    cmd = [FFMPEG, "-y", "-hide_banner",
           "-hwaccel", "d3d11va",
           "-i", str(src),
           "-vf", "scale=256:256", "-vsync", "0",
           "-c:v", "libx264", "-crf", "10", "-g", "30", "-an",
           str(dst)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", encoding="utf-8")
    log_path.write_text(proc.stderr, encoding="utf-8")
    warn_lines = [ln for ln in proc.stderr.splitlines()
                  if re.search(r"Error|error|Could not|corrupt|damaged|Invalid|missing", ln)]
    return proc.returncode, warn_lines, time.time() - t0


def probe_frames(unit_id, dst, coord_path):
    cmd = [FFMPEG, "-hide_banner", "-v", "info", "-i", str(dst),
           "-vf", "showinfo", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"showinfo 失败 {unit_id}: {proc.stderr[:300]}")
    pat = re.compile(r"n:\s*(\d+) pts:\s*\S+ pts_time:\s*([0-9.]+|N/A)")
    n = 0
    pts = []
    with open(coord_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "pts_time"])
        for ln in proc.stderr.splitlines():
            m = pat.search(ln)
            if not m:
                continue
            if int(m.group(1)) != n:
                raise RuntimeError(f"{unit_id} showinfo 帧序跳变: 期望 {n} 得 {m.group(1)}")
            t = m.group(2)
            w.writerow([n, t])
            pts.append(float(t) if t != "N/A" else -1.0)
            n += 1
    mono = all(b >= a - 1e-6 for a, b in zip(pts, pts[1:])) and -1.0 not in pts
    # 同时算空洞统计
    gaps = []
    for i in range(1, len(pts)):
        d = pts[i] - pts[i-1]
        if d > 0.2:
            gaps.append(d)
    return n, (pts[-1] if pts else 0.0), mono, len(gaps), sum(gaps)


def write_report(report_path, all_rows):
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["unit_id", "ok", "rc", "frames",
                                          "last_pts", "pts_monotonic",
                                          "gaps_gt02", "gap_total_s",
                                          "warn_lines", "sec", "err"])
        w.writeheader()
        for uid in sorted(all_rows):
            w.writerow(all_rows[uid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    (SRC256_HW / "coord").mkdir(parents=True, exist_ok=True)
    (SRC256_HW / "log").mkdir(parents=True, exist_ok=True)
    report_path = SRC256_HW / "report.csv"
    done = {}
    if report_path.exists() and not args.force:
        with open(report_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done[r["unit_id"]] = r

    rows = [(uid, vp) for uid, vp in unit_video_rows() if only is None or uid in only]
    print(f"[src256-hw] 待转 {len(rows)} 条 -> {SRC256_HW}", flush=True)
    all_rows = dict(done)
    for i, (uid, src) in enumerate(rows, 1):
        dst = SRC256_HW / f"{uid}.mp4"
        if dst.exists() and dst.stat().st_size > 0 and uid in done and not args.force:
            continue
        if not src.exists():
            print(f"[{i}/{len(rows)}] {uid} 源不存在", flush=True)
            all_rows[uid] = {"unit_id": uid, "ok": 0, "err": "src-missing"}
            write_report(report_path, all_rows)
            continue
        rc, warns, dt = run_ffmpeg_hw(uid, src, dst, SRC256_HW / "log" / f"{uid}.txt")
        if not dst.exists() or dst.stat().st_size == 0:
            print(f"[{i}/{len(rows)}] {uid} 转码无输出 rc={rc}", flush=True)
            all_rows[uid] = {"unit_id": uid, "ok": 0, "err": f"rc={rc}"}
            write_report(report_path, all_rows)
            continue
        try:
            n, last_pts, mono, gaps, gap_total = probe_frames(
                uid, dst, SRC256_HW / "coord" / f"{uid}.csv")
        except RuntimeError as e:
            all_rows[uid] = {"unit_id": uid, "ok": 0, "err": str(e)[:120]}
            write_report(report_path, all_rows)
            continue
        all_rows[uid] = {"unit_id": uid, "ok": 1, "rc": rc,
                         "frames": n, "last_pts": round(last_pts, 2),
                         "pts_monotonic": int(mono),
                         "gaps_gt02": gaps, "gap_total_s": round(gap_total, 1),
                         "warn_lines": len(warns), "sec": round(dt, 1)}
        write_report(report_path, all_rows)
        flag = "" if mono else " [PTS非单调!]"
        print(f"[{i}/{len(rows)}] {uid}: {n} 帧 {last_pts:.1f}s "
              f"空洞{gaps}/{gap_total:.1f}s 告警{len(warns)}行 {dt:.0f}s{flag}", flush=True)

    n_ok = sum(1 for r in all_rows.values() if str(r.get("ok")) == "1")
    n_mono = sum(1 for r in all_rows.values() if str(r.get("pts_monotonic")) == "1")
    print(f"[src256-hw] 完成 {n_ok}/{len(all_rows)}，PTS 单调 {n_mono}", flush=True)


if __name__ == "__main__":
    main()
