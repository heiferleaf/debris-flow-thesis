# -*- coding: utf-8 -*-
"""EXP-008 R1 v2：从 256×256 转码中间件取材（2026-10-05 坏块事故重建，复核架构 C）。

背景：旧取材（build_units.py，逐槽 seek + OpenCV 直读源视频）在 HEVC 上静默产出
灰底坏块且存在负载相关丢帧，旧产物整体作废（run log 2026-10-05）。新链路分两层：

  第一层（build_units_src256.py，已另行执行）：源视频 → ffmpeg 转码中间件
    data/process/EXP-008-units-src256/{unit_id}.mp4 + coord/逐帧坐标表(frame,pts_time)。
  第二层（本脚本）：从中间件取材。中间件是干净 H.264 流，OpenCV 顺序解码可靠；
    256×256 解码成本约为 4K 源的 1/50，重采不再昂贵。

本脚本逐单位执行（阻断 B1–B4 对应落点）：
  1. B3 全片质检（主防线）：中间件整条顺序解码到 EOF，逐帧记灰度 std 落
     qc/{unit_id}.csv；低方差帧（< QC_STD_LOW，转码无法修复的灰底坏帧）、读失败帧
     各自成区，全部并入禁取。转码告警日志只作辅助对账（告警不带帧号，不参与控制流）。
  2. B2 中间件自成时间轴：cut 剔除区间（events.csv 秒）→ 用 coord 坐标表逐帧查
     pts_time 落帧，禁用任何 秒×fps 算术。帧号一律是中间件帧号，与源帧号无关
     （转码丢帧后源帧号整体前移，绝不能混用）。
  3. B4 EOF 帧数断言：全片扫描交付帧数必须等于 coord 表帧数，不符整单位失败；
     每帧保留 POS_FRAMES 失步核对（解码器丢帧续解则立即失败）。
  4. B1 新落盘根：产物落 data/process/EXP-008-units-v2/，与旧根
     EXP-008-units（已改名 EXP-008-units-voided-seekbadblocks-20261005）零共享。

时间语义：events.csv 的秒数与审计初帧 frames_index.csv 的 t_sec 同属源视频
POS_MSEC 时间体系；中间件 -vsync 0 直通源 PTS，坐标表 pts_time 与之同轴。
锚点核对（每条视频 ≥2 帧 vs frames_index.csv，±40ms 容差）由
verify_anchors_v2.py 单独执行，属重建验收必过项，不在本脚本内。

输入：EXP-008-units-src256/{unit_id}.mp4 + coord/{unit_id}.csv、三源 events.csv、
段清单、folds.json（均沿用 build_units.py 同一套索引函数，不另写寻径）。
输出：data/process/EXP-008-units-v2/{unit-manifest.csv, frames/, qc/,
unit-failures.csv}。写盘帧不另记质检表：全片 qc/ 已覆盖每个中间件帧（含写盘帧），
是旧版 qc-frames.csv 的超集。

清单列说明（列名与旧版一致以兼容读方，语义两处变化）：
  - clip_start_frames = 中间件帧号（旧版为源帧号）；
  - src_fps = 源元数据 fps，仅存档参考，不参与任何换算（v1_07 元数据 25fps 与
    PTS 实测约 20fps 矛盾，故全面弃用 fps 算术）；
  - duration_sec = 中间件坐标表末帧 pts_time。

用法（仓库根，venv）：
  python experiments/common/video/build_units_v2.py --tiny 2   # tiny 冒烟
  python experiments/common/video/build_units_v2.py --only v1_07
  python experiments/common/video/build_units_v2.py            # 全量（断点续传）
环境变量：DF_UNITS_ROOT（默认 v2 根）、DF_SRC256_ROOT（中间件根）、DF_QC_STD_LOW。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/common/video"))
from experiments.common.video.artifact_paths import recorded_path
from build_units import (CUT_MARGIN_S, N_CLIPS, CLIP_LEN, QC_STD_LOW,  # noqa: E402
                         RESIZE, VIDEO_KEY_OF_V1_109, V1_109_BAD_FROM_S,
                         build_video_index, clean_unit_dir, load_events,
                         merge_ranges, pick_slots, read_csv_rows)

SRC256_ROOT = Path(os.environ.get("DF_SRC256_ROOT")
                   or recorded_path(REPO / "data/process/EXP-008-units-src256", repo=REPO))
UNITS_ROOT = Path(os.environ.get("DF_UNITS_ROOT")
                  or recorded_path(REPO / "data/process/EXP-008-units-v2", repo=REPO))
OUT_MANIFEST = UNITS_ROOT / "unit-manifest.csv"
FRAMES_ROOT = UNITS_ROOT / "frames"
QC_DIR = UNITS_ROOT / "qc"              # B3 全片逐帧 std（每单位一份 qc/{unit_id}.csv）
LOWSTD_ZONE_PAD = 16    # 低方差帧两侧并入禁区的帧数（灰底坏区成片出现，不单帧零割）
MAX_PICK_RETRY = 6      # 中间件重采上限（干净流上不应触发，触发即响亮报告）


def load_coord(unit_id):
    """中间件坐标表 -> (帧数, pts_time 列表, 是否单调)。B2：一切秒↔帧换算只查此表。

    PTS 非单调不在此处排除（W4 口径=单独标记）：调用方把倒步邻域并入禁区后
    仍可用（跳步处 ±2s 内不落任何片段，映射在其余区间单调可信）。"""
    p = SRC256_ROOT / "coord" / f"{unit_id}.csv"
    if not p.exists():
        raise RuntimeError(f"坐标表缺失：{p}")
    rows = read_csv_rows(p)
    pts = [float(r["pts_time"]) for r in rows]
    frames = [int(r["frame"]) for r in rows]
    if frames != list(range(len(rows))):
        raise RuntimeError(f"{unit_id} 坐标表帧序不连续")
    mono = all(b >= a - 1e-6 for a, b in zip(pts, pts[1:]))
    return len(rows), pts, mono


def pts_jump_zones(pts, pad=64):
    """PTS 倒步位置 -> 帧禁区 [i-pad, i+pad]（≈±3s@20fps，W4 受影响邻域禁入）。"""
    out = []
    for i in range(1, len(pts)):
        if pts[i] < pts[i - 1] - 1e-6:
            out.append((max(0, i - pad), min(len(pts), i + pad + 1)))
    return out


def pts_gap_zones(pts, gap_thresh=0.2, pad_sec=1.0):
    """PTS 前跳（空洞）位置 -> 帧禁区。显卡解码丢帧产生时间空洞，
    空洞前后 pad_sec 秒内不落片段槽（W1：按秒外扩不按帧数）。

    阈值按源视频帧率自适应（2026-10-25 修）：v1_34 是 2.5fps 低帧率源，
    正常帧间隔就有 0.4 秒，固定 0.2 秒阈值会把整条视频判成空洞、抽不出槽。
    取 max(固定下限, 相邻间隔中位数 × 3)：10fps 源阈值 0.3 秒，2.5fps 源 1.2 秒。
    """
    diffs = sorted(b - a for a, b in zip(pts, pts[1:]) if b > a)
    if diffs:
        med = diffs[len(diffs) // 2]
        gap_thresh = max(gap_thresh, 3.0 * med)
    out = []
    for i in range(1, len(pts)):
        d = pts[i] - pts[i - 1]
        if d > gap_thresh:
            # 找空洞前后 pad_sec 秒对应的帧号
            left = next((j for j in range(i - 1, -1, -1)
                         if pts[i - 1] - pts[j] >= pad_sec), 0)
            right = next((j for j in range(i, len(pts))
                          if pts[j] - pts[i] >= pad_sec), len(pts) - 1)
            out.append((left, right + 1))
    return out


def secs_to_ranges(secs, pts):
    """秒区间 [(a,b)...] -> 中间件帧区间 [fa,fb)（闭开）。B2：逐帧查 pts_time，
    禁用秒×fps。区间外扩 CUT_MARGIN_S 秒防边界标注误差。"""
    rs = []
    for s, e in secs:
        fa = next((i for i, t in enumerate(pts) if t >= s - CUT_MARGIN_S), len(pts))
        fb = next((i for i, t in enumerate(pts) if t >= e + CUT_MARGIN_S), len(pts))
        if fb > fa:
            rs.append((fa, fb))
    return rs


def scan_middleware(path, unit_id, n_coord, pts):
    """B3 主防线 + B4 EOF 断言：中间件整条顺序解码到 EOF，逐帧灰度 std。

    返回 (n_read, fail_frames, lowstd_frames)。n_read != n_coord 即 EOF 断言失败。
    每帧保留 POS_FRAMES 失步核对：解码器丢帧续解时其后帧位不可信，立即失败。
    逐帧 std 落 qc/{unit_id}.csv 供 W1 阈值标定与事后审计。
    """
    QC_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"中间件打不开：{path}")
    n_read = 0
    fail_frames = []
    lowstd_frames = []
    desync_at = None
    fi = 0
    with open(QC_DIR / f"{unit_id}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "pts_time", "gray_std"])
        while True:
            ok, fr = cap.read()
            if not ok:
                fail_frames.append(fi)
                fi += 1
                if fi > n_coord:      # 中间件是干净流：到 coord 帧数还没到尾即异常
                    break
                continue
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos != fi + 1:
                desync_at = fi        # 失步：帧位不可信，立即停
                break
            # std 在 256×256 全帧上算（中间件原生尺寸），与 QC_STD_LOW 的标定
            # 依据（源视频双路径实测，256 尺度）同一把尺子，不做降采样
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            gstd = round(float(gray.std()), 1)
            w.writerow([fi, round(pts[fi], 3) if fi < len(pts) else "", gstd])
            if gstd < QC_STD_LOW:
                lowstd_frames.append(fi)
            n_read += 1
            fi += 1
            if n_read >= n_coord:     # 扫满 coord 帧数后应恰是 EOF
                if cap.grab():
                    fail_frames.append(fi)   # coord 之外还有帧：账目不符
                break
    cap.release()
    if desync_at is not None:
        raise RuntimeError(f"{unit_id} 中间件 POS_FRAMES 失步于帧 {desync_at}")
    return n_read, fail_frames, lowstd_frames


def zones_from_frames(frames, n_total, pad=LOWSTD_ZONE_PAD):
    """帧位清单 -> 并入禁区的帧区间 [a,b)（各自外扩 pad 帧）。"""
    return [(max(0, f - pad), min(n_total, f + pad + 1)) for f in frames]


def extract_unit_mw(path, slots, unit_dir, resume, n_mw):
    """从中间件顺序解码，路过各片段槽写 16 帧；末槽写完后继续 grab 到 EOF
    （B4 EOF 断言：交付总数必须等于坐标表帧数，不符=解码漂移，整单位失败——
    -560 事故的教训：POS_FRAMES 核对对负载丢帧失明，总账才是硬防线）。

    返回 (n_write, n_scanned, fail_at, fail_frames)。n_scanned = 扫到 EOF 的交付
    帧数；resume 命中续传时返回 -1（本轮无解码发生）。"""
    unit_dir.mkdir(parents=True, exist_ok=True)
    sidecar = unit_dir / "slots.txt"
    n_files = len(list(unit_dir.glob("clip*.jpg")))
    if (resume and n_files == N_CLIPS * CLIP_LEN and sidecar.exists()
            and sidecar.read_text(encoding="utf-8").strip()
            == ";".join(str(s * CLIP_LEN) for s in slots)):
        return N_CLIPS * CLIP_LEN, -1, None, []   # -1 = 未扫描（续传跳过）
    need = {}
    for ci, slot in enumerate(slots):
        for fi in range(CLIP_LEN):
            need[slot * CLIP_LEN + fi] = (ci, fi)
    last_need = max(need)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"中间件打不开：{path}")
    n_write = 0
    n_scanned = 0
    fail_at = None
    fail_frames = []
    fi = 0
    while True:
        if fail_at is None and fi > last_need:
            break
        if fail_at is None:
            ok, fr = cap.read()
        else:
            ok, fr = cap.grab(), None
        if not ok or (fail_at is None and fr is None):
            fail_frames.append(fi)
            if fail_at is None:
                fail_at = fi
                fi += 1
                continue
            break     # 中间件干净流上 read 不应失败；失败即异常，调用方整单位处理
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos != fi + 1:
            fail_frames.append(fi)
            if fail_at is None:
                fail_at = fi
            break
        n_scanned += 1
        if fail_at is None and fi in need:
            ci, k = need[fi]
            name = f"clip{ci:02d}_f{k:02d}.jpg"
            cv2.imwrite(str(unit_dir / name), fr)
            n_write += 1
        fi += 1
    # 尾段：末槽之后继续 grab 到 EOF，只对总账（不写帧）。中间件干净流上这里
    # 应无失败；提前失败或超过坐标表帧数都算账目不符。
    if fail_at is None:
        while True:
            if not cap.grab():
                break                      # 正常 EOF
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos != fi + 1:
                fail_frames.append(fi)
                fail_at = fi
                break
            n_scanned += 1
            fi += 1
            if fi > n_mw:
                fail_frames.append(fi)     # 超出坐标表帧数：账目不符
                fail_at = fi
                break
    cap.release()
    if fail_at is None:
        sidecar.write_text(";".join(str(s * CLIP_LEN) for s in slots),
                           encoding="utf-8")
    return n_write, n_scanned, fail_at, fail_frames


def main():
    ap = argparse.ArgumentParser(description="EXP-008 R1 v2：从转码中间件取材")
    ap.add_argument("--tiny", type=int, default=0, metavar="K", help="每类只出 K 个单位")
    ap.add_argument("--only", default="", help="只处理指定 unit_id（逗号分隔）")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    vid2key, vid2cuts, _ = load_events()
    vindex = build_video_index()
    print(f"[R1v2] 段清单东川三源视频 {len(vindex)} 条；中间件根 {SRC256_ROOT}")
    print(f"[R1v2] 落盘根 {UNITS_ROOT}（B1 新根，与作废旧根零共享）")

    units = []
    for (src, video_id), info in sorted(vindex.items()):
        vkey = vid2key.get((src, video_id))
        if vkey is None:
            raise RuntimeError(f"events.csv 找不到 video_id：{(src, video_id)}")
        mw = SRC256_ROOT / f"{vkey}.mp4"
        units.append({"unit_id": vkey, "source": src, "video_id": video_id,
                      "mw_path": mw, **info})
    n_pos = sum(1 for u in units if u["label"] == 1)
    print(f"[R1v2] 单位池 {len(units)}（正 {n_pos}/负 {len(units)-n_pos}）")
    if args.tiny:
        units = pick_tiny(units, args.tiny)
        print(f"[R1v2] tiny 模式：取 {len(units)} 个")
    if args.only:
        keep = set(args.only.split(","))
        units = [u for u in units if u["unit_id"] in keep]

    out_rows, t_all = [], time.time()
    failures = []   # (unit_id, reason)：单单位失败不中止整场，结尾落 unit-failures.csv
    for i, u in enumerate(units, 1):
        uid = u["unit_id"]
        try:
            mw = u["mw_path"]
            if not mw.exists() or mw.stat().st_size == 0:
                raise RuntimeError("中间件缺失（先跑 build_units_src256.py）")
            n_mw, pts, mono = load_coord(uid)
            if n_mw < N_CLIPS * CLIP_LEN:
                raise RuntimeError(f"中间件仅 {n_mw} 帧 < {N_CLIPS*CLIP_LEN}")
            cuts = vid2cuts.get((u["source"], u["video_id"]), [])
            # B3 主防线：全片质检（每单位只做一次，结果落盘；断点续传时复用）
            qc_csv = QC_DIR / f"{uid}.csv"
            if qc_csv.exists() and sum(1 for _ in open(qc_csv, encoding="utf-8")) - 1 == n_mw:
                rows = read_csv_rows(qc_csv)
                fail_frames = [int(r["frame"]) for r in rows if not r.get("gray_std")]
                lowstd = [int(r["frame"]) for r in rows
                          if r.get("gray_std") and float(r["gray_std"]) < QC_STD_LOW]
                n_scan = n_mw
            else:
                n_scan, fail_frames, lowstd = scan_middleware(mw, uid, n_mw, pts)
            if n_scan != n_mw or fail_frames:
                raise RuntimeError(f"EOF 断言失败：扫描 {n_scan}/{n_mw} 帧，"
                                   f"读失败 {len(fail_frames)} 帧（B4）")
            # B2：cut 秒 -> coord 逐帧查表；v1_109 坏区按 pts_time 落帧
            forb = secs_to_ranges(cuts, pts)
            if uid == VIDEO_KEY_OF_V1_109:
                fb = next((j for j, t in enumerate(pts) if t >= V1_109_BAD_FROM_S),
                          n_mw)
                if fb < n_mw:
                    forb.append((fb, n_mw))
            forb += zones_from_frames(lowstd, n_mw)
            jump = pts_jump_zones(pts)
            if jump:
                # W4：PTS 倒步邻域禁入（单独标记不整条排除），清单带 pts_monotonic=0
                print(f"  [{i}/{len(units)}] !! {uid} PTS 倒步 {len(jump)} 处，"
                      f"邻域并入禁区（W4 单独标记）", flush=True)
            forb += jump
            # 显卡解码空洞：前跳 >0.2s 的位置前后 1s 禁入
            gaps = pts_gap_zones(pts)
            if gaps:
                print(f"  [{i}/{len(units)}] {uid} PTS 空洞 {len(gaps)} 处，"
                      f"±1s 并入禁区", flush=True)
            forb += gaps
            forb = merge_ranges(forb)
            n_slots = n_mw // CLIP_LEN
            slots = pick_slots(n_slots, forb)
            unit_dir = FRAMES_ROOT / uid
            n, n_sc, fail_at, ff = extract_unit_mw(mw, slots, unit_dir,
                                                   resume=not args.tiny, n_mw=n_mw)
            retries = 0
            while fail_at is not None and retries < MAX_PICK_RETRY:
                forb = merge_ranges(forb + zones_from_frames(ff or [fail_at], n_mw,
                                                             pad=16 * 2 ** retries))
                clean_unit_dir(unit_dir)
                slots = pick_slots(n_slots, forb)
                n, n_sc, fail_at, ff = extract_unit_mw(mw, slots, unit_dir,
                                                       resume=False, n_mw=n_mw)
                retries += 1
            if fail_at is not None or n != N_CLIPS * CLIP_LEN:
                raise RuntimeError(f"重采 {retries} 次仍不满 768 帧（失败帧位 {fail_at}）")
            if n_sc not in (-1, n_mw):
                raise RuntimeError(f"EOF 断言失败：抽帧扫描 {n_sc}/{n_mw} 帧（B4）")
        except RuntimeError as e:
            failures.append((uid, str(e)))
            print(f"  [{i}/{len(units)}] !! {uid} 排除：{e}", flush=True)
            continue
        out_rows.append({
            "unit_id": uid, "source": u["source"], "video_id": u["video_id"],
            "group_id": u["group_id"], "fold": u["fold"],
            "label": "pos" if u["label"] == 1 else "neg",
            "n_clips": N_CLIPS, "clip_len": CLIP_LEN,
            "n_frames_mw": n_mw,
            "pts_monotonic": int(mono),
            "duration_sec": round(pts[-1], 1),
            "clip_start_frames": ";".join(str(s * CLIP_LEN) for s in slots),
            "n_frames": n,
        })
        print(f"  [{i}/{len(units)}] + {uid} {'pos' if u['label'] else 'neg'} "
              f"mw={n_mw}帧 {pts[-1]/60:.1f}min 低方差{len(lowstd)}帧 "
              f"slots[0]={slots[0]} slots[-1]={slots[-1]} fold={u['fold']} "
              f"重采{retries} ({time.time()-t_all:.0f}s)", flush=True)

    UNITS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["unit_id", "source", "video_id", "group_id", "fold", "label",
              "n_clips", "clip_len", "n_frames_mw", "pts_monotonic",
              "duration_sec", "clip_start_frames", "n_frames"]
    # 增量重跑（--only/--tiny）不丢历史行：合并旧清单，本轮行覆盖同 unit_id
    old = {}
    if OUT_MANIFEST.exists():
        for r in read_csv_rows(OUT_MANIFEST):
            old[r["unit_id"]] = r
    for r in out_rows:
        old[r["unit_id"]] = r
    merged = [old[k] for k in sorted(old)]
    with open(OUT_MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    print(f"\n[R1v2] 完成：{len(out_rows)} 单位（"
          f"正 {sum(1 for r in out_rows if r['label']=='pos')}/"
          f"负 {sum(1 for r in out_rows if r['label']=='neg')}），"
          f"耗时 {time.time()-t_all:.0f}s -> {OUT_MANIFEST}")
    if failures:
        fail_csv = UNITS_ROOT / "unit-failures.csv"
        with open(fail_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "reason"])
            w.writerows(failures)
        print(f"[R1v2] 排除 {len(failures)} 单位（响亮报告；理由落 {fail_csv}）：")
        for uid, reason in failures:
            print(f"  !! {uid}: {reason}")
        sys.exit(2)


def pick_tiny(units, k):
    """tiny 冒烟：中间件文件大的优先（时长长），每类 k 个、组不重复。"""
    pos = sorted([u for u in units if u["label"] == 1
                  and u["mw_path"].exists()],
                 key=lambda u: -u["mw_path"].stat().st_size)
    neg = sorted([u for u in units if u["label"] == 0
                  and u["mw_path"].exists()],
                 key=lambda u: -u["mw_path"].stat().st_size)
    out, seen = [], set()
    for pool in (pos, neg):
        got = 0
        for u in pool:
            if got >= k:
                break
            if u["group_id"] in seen:
                continue
            seen.add(u["group_id"])
            out.append(u)
            got += 1
    return out


if __name__ == "__main__":
    main()
