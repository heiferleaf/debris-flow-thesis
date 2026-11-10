# -*- coding: utf-8 -*-
"""EXP-008 取材中间件生成：源视频 → 256×256 CRF10 干净中间件（2026-10-05 立）。

背景与依据：训练抽帧的逐槽 seek 在 HEVC 上静默解出灰底坏块（run log 2026-10-05）；
顺序解码在受损流上又存在负载相关静默丢帧与尾部不可读。research-auditor 复核通过
「转码中间件」架构（复核结论 5 阻断，本文实现 B1/B2/B3/W1-W4/S1/S2）：

  对每条源视频执行：
    ffmpeg -i 源 -vf scale=256:256 -vsync 0
           -c:v libx264 -crf 10 -g 30 -an 中间件.mp4
  - `-vsync 0`（passthrough）：不按 PTS 复制/丢帧，中间件帧序 = 解码交付序；
  - 禁 `-discardcorrupt`（丢帧会让帧号静默移位，比留灰帧更糟）；
  - stderr 逐行收集解码告警（辅助对账，非主防线——告警不带帧号）；
  - ffprobe -count_frames 复核输出帧数；-show_frames 落逐帧坐标表
    （帧号↔pts_time，S2），后续所有秒↔帧换算查表，不用元数据 fps；
  - 中间件 POS_MSEC 单调性断言（W4）。

主防线是 build_units 侧对中间件的全片顺序质检（B3），本工具只负责把
「解码不确定性」变成「可见的告警清单与帧数账目」。

用法（仓库根，venv）：
  python experiments/tools/build_units_src256.py --only v1_07[,v1_08]   # 试转/补转
  python experiments/tools/build_units_src256.py                        # 全量 151 条
产物：data/process/EXP-008-units-src256/{<unit_id>.mp4, coord/<unit_id>.csv,
log/<unit_id>.txt}，汇总 report.csv。幂等：已存在且帧数>0 的跳过（--force 重转）。
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

import imageio_ffmpeg  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()   # 独立 ffmpeg 二进制（venv 内，无系统安装）
# 注：imageio-ffmpeg 不带 ffprobe，且本机网络到 GitHub 不通（static-ffmpeg 二进制
# 下载超时）。逐帧坐标表改用 showinfo 滤镜（ffmpeg -vf showinfo -f null -，stderr
# 每帧一行 n:/pts_time:），能力实测 4.2.2 具备（hevc 解码/libx264/showinfo）。
SRC256 = REPO / "data/process/EXP-008-units-src256"


def unit_video_rows():
    """unit_id -> 原始视频路径。集合 = build_units 主流程的视频池（events 索引
    全集），与训练取材同一套寻径，不另写一份。"""
    from build_units import build_video_index, load_events, locate_raw_video
    vid2key, _, _ = load_events()
    out = []
    for (src, video_id), info in sorted(build_video_index().items()):
        vkey = vid2key.get((src, video_id))
        if vkey is None:
            raise RuntimeError(f"events.csv 找不到 video_id：{(src, video_id)}")
        vp = locate_raw_video(video_id)
        if vp is None:
            raise RuntimeError(f"raw 未定位：{video_id} ({vkey})")
        out.append((vkey, Path(vp)))
    return out


def run_ffmpeg(unit_id, src, dst, log_path):
    # 容错取向：不设 err_detect（默认 careful 即不因个别错误中止解码；
    # aggressive/explode 反而会更早弃帧）。禁 -discardcorrupt（丢帧移位比留灰帧糟）。
    cmd = [FFMPEG, "-y", "-hide_banner",
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
    """showinfo 逐帧坐标表（帧号, pts_time 秒）+ 总帧数。

    `ffmpeg -vf showinfo -f null -` 在 stderr 每帧打一行
    `[...showinfo...] n:  0 pts:... pts_time:0.000000`，解析之。
    """
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
            if int(m.group(1)) != n:      # showinfo 帧序必须连续
                raise RuntimeError(f"{unit_id} showinfo 帧序跳变: 期望 {n} 得 {m.group(1)}")
            t = m.group(2)
            w.writerow([n, t])
            pts.append(float(t) if t != "N/A" else -1.0)
            n += 1
    # W4：pts 单调不减断言（N/A 记 -1，计入不单调）
    mono = all(b >= a - 1e-6 for a, b in zip(pts, pts[1:])) and -1.0 not in pts
    return n, (pts[-1] if pts else 0.0), mono


def write_report(report_path, all_rows):
    """汇总落盘（覆盖写，含历史行）。逐单位调用：中途崩溃不丢进度账，
    重启按已 ok 的行幂等跳过。"""
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["unit_id", "ok", "rc", "frames",
                                          "last_pts", "pts_monotonic",
                                          "warn_lines", "sec", "err"])
        w.writeheader()
        for uid in sorted(all_rows):
            w.writerow(all_rows[uid])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="只转指定 unit_id（逗号分隔）")
    ap.add_argument("--force", action="store_true", help="已存在也重转")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    (SRC256 / "coord").mkdir(parents=True, exist_ok=True)
    (SRC256 / "log").mkdir(parents=True, exist_ok=True)
    report_path = SRC256 / "report.csv"
    done = {}
    if report_path.exists() and not args.force:
        with open(report_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done[r["unit_id"]] = r

    rows = [(uid, vp) for uid, vp in unit_video_rows() if only is None or uid in only]
    print(f"[src256] 待转 {len(rows)} 条 -> {SRC256}")
    report_rows = []
    all_rows = dict(done)
    for i, (uid, src) in enumerate(rows, 1):
        dst = SRC256 / f"{uid}.mp4"
        if dst.exists() and dst.stat().st_size > 0 and uid in done and not args.force:
            continue
        if not src.exists():
            print(f"[{i}/{len(rows)}] {uid} 源不存在: {src}")
            report_rows.append({"unit_id": uid, "ok": 0, "err": "src-missing"})
            all_rows[uid] = report_rows[-1]
            write_report(report_path, all_rows)
            continue
        rc, warns, dt = run_ffmpeg(uid, src, dst, SRC256 / "log" / f"{uid}.txt")
        if not dst.exists() or dst.stat().st_size == 0:
            print(f"[{i}/{len(rows)}] {uid} 转码无输出 rc={rc}")
            report_rows.append({"unit_id": uid, "ok": 0, "err": f"rc={rc}"})
            all_rows[uid] = report_rows[-1]
            write_report(report_path, all_rows)
            continue
        try:
            n, last_pts, mono = probe_frames(uid, dst, SRC256 / "coord" / f"{uid}.csv")
        except RuntimeError as e:
            report_rows.append({"unit_id": uid, "ok": 0, "err": str(e)[:120]})
            all_rows[uid] = report_rows[-1]
            write_report(report_path, all_rows)
            continue
        report_rows.append({"unit_id": uid, "ok": 1, "rc": rc,
                            "frames": n, "last_pts": round(last_pts, 2),
                            "pts_monotonic": int(mono),
                            "warn_lines": len(warns), "sec": round(dt, 1)})
        all_rows[uid] = report_rows[-1]
        write_report(report_path, all_rows)
        flag = "" if mono else " [PTS非单调!]"
        print(f"[{i}/{len(rows)}] {uid}: {n} 帧 last_pts={last_pts:.1f}s "
              f"告警{len(warns)}行 rc={rc} {dt:.0f}s{flag}", flush=True)

    n_ok = sum(1 for r in all_rows.values() if str(r.get("ok")) == "1")
    n_mono = sum(1 for r in all_rows.values() if str(r.get("pts_monotonic")) == "1")
    print(f"[src256] 完成 {n_ok}/{len(all_rows)}，PTS 单调 {n_mono}，报告 -> {report_path}")


if __name__ == "__main__":
    main()
