# -*- coding: utf-8 -*-
# 【已作废 2026-10-05】属坏块事故（训练抽帧跳转解码解出灰底坏块）之前那套取材管线（v1），
# 取材main已被build_units_v2.py取代，勿执行本文件的旧取材入口。
# v2仍导入本文件的常量、索引和区间辅助函数；它们不等于调用旧取材main。
"""EXP-008 R1：以一条视频为一条样本的弱标签取材（一条样本 = 一条视频 = 48 片段 × 16 帧）。

设计依据：docs/process/experiments/EXP-008.md（叶海峰 2026-10-03 签字重开版）。设计逐条：
  - 一条样本 = 一条完整源视频；标签 = 视频级弱标签：events.csv 含任一 pos 审计段 = 正，
    只有 neg 段 = 负；整段纯 cut 的视频不入池（叶海峰 2026-10-03：整段噪点无法使用）。
  - 16 帧片段 = 源视频**原生帧率**下连续 16 帧、零重复（作废旧版 25fps 网格重复帧方案）。
  - 48 片段：整条视频顺序连续切 N = n_frames // 16 个片段（不重叠），按时间等间隔
    （linspace）抽 48 个，保持时间顺序（偏差 D3：原文未交代 N>48 选法）。
  - 落点防呆：片段与剔除区间（cut，前后各扩 1s 防边界标注误差）或坏帧区间相交时，
    向最近合法片段槽平移重采；抽帧读失败同样把该区间列入禁取再重采。
  - 坏帧区间：v1_109（门前沟中游 19：57）自 1446.6s 起 HEVC 参考帧链断裂（2026-10-02
    检出，实测 1400 帧仅 432 可解），该时刻起整段禁取。
  - 组与折：组 = 机位×场次，沿用 EXP-006-pilot 段清单的 group_id；折 =
    data/metadata/splits/EXP-008-units-folds.json（2026-10-04 视频级分层重划，
    make_unit_folds.py 生成，一次落盘后冻结），整组同侧。
  - 边界不内缩 4 秒（偏差 D2）：标签在视频级，片段落点不影响标签正确性。

抽帧方式（2026-10-05 顺序解码重构，research-auditor 复核通过）：整条视频从头逐帧
顺序解码一遍，路过各片段槽时写 16 帧——与审计初帧（extract_frames.py）同一条解码
路径。此前按槽 seek（CAP_PROP_POS_FRAMES）取帧在 HEVC 视频上会因缺参考帧静默解出
灰底碎块坏块（2026-10-05 事故，v1_07/v1_05 实证，人工审计对该类坏块结构性失明），
顺序解码从根上消除。两道配套防护：
  - 帧计数失步防护：每帧 read 后核对 CAP_PROP_POS_FRAMES 与本地计数，不一致即整条
    视频报失败走重采——解码器在坏区会丢帧续解（v1_109 实测 1400 帧仅 432 可解），
    不防则其后所有槽位整体错位且不报错。
  - 逐帧质检常驻：每写一帧记灰度 std 与同片段邻帧灰度差，落 qc-frames.csv；判坏
    阈值按双路径实测标定（跳转坏块 std≈11 / 顺序正常帧 std≈39，取 20 为警戒线，
    只标记不剔除，全量摸底后随摸底分布复核）。

输入：
  - 段清单 data/results/temporal/EXP-006-pilot/segment-manifest-20260924-addrmap.csv
    （只取 source ∈ {dongchuan, video4, video5}，提供 video_id -> group_id）
  - 各源 events.csv（data/process/video_frames/<seg>/events.csv，提供 video_id -> video_key、
    cut 区间）
  - 原始视频 data/raw 下 rglob 反查（video_id = 文件 stem，v1/v4/v5 均适用，沿用旧版逻辑）

输出：
  - data/process/EXP-008-units/unit-manifest.csv      单位清单（每单位一行）
  - data/process/EXP-008-units/frames/<unit_id>/      clip{cc:02d}_f{ff:02d}.jpg 共 768 帧
  - data/process/EXP-008-units/qc-frames.csv          逐帧质检（unit_id/clip/frame/
    gray_std/prev_diff/flag；flag=lowstd 即灰度 std<QC_STD_LOW，只标记不剔除；
    重采单位的失败轮次行会残留，分析按 (unit_id, frame) 取最后一次）

用法（仓库根，venv）：
  python experiments/common/video/build_units.py --tiny 2   # tiny 冒烟：2 正 + 2 负
  python experiments/common/video/build_units.py            # 全量（断点续传：满帧单位跳过）
环境变量：DF_UNITS_ROOT 覆盖落盘根。
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
UNITS_ROOT = Path(os.environ.get("DF_UNITS_ROOT") or REPO / "data/process/EXP-008-units")
OUT_MANIFEST = UNITS_ROOT / "unit-manifest.csv"
QC_FRAMES_CSV = UNITS_ROOT / "qc-frames.csv"   # 逐帧质检统计（2026-10-05 阻断 3）
FRAMES_ROOT = UNITS_ROOT / "frames"
SEG_MANIFEST = (REPO / "data/results/temporal/EXP-006-pilot"
                / "segment-manifest-20260924-addrmap.csv")
FOLDS_JSON = REPO / "data/metadata/splits/EXP-008-units-folds.json"

SOURCES = ("dongchuan", "video4", "video5")   # 东川三源白名单（叶海峰 2026-10-02 拍板袋域）
N_CLIPS = 48        # 每单位片段数（Wu 形态）
CLIP_LEN = 16       # 每片段帧数（Wu 形态）
CUT_MARGIN_S = 1.0  # 剔除区间边界外扩（防边界秒级标注误差把坏帧缝进片段）
RESIZE = 256        # 帧落盘边长（与旧版一致，R2 再缩 112）
MAX_PICK_RETRY = 12 # 单单位抽帧读失败重采上限，超了整单位报错停
QC_STD_LOW = 20.0   # 逐帧质检警戒线（灰度 std）：跳转坏块实测 ≈11、正常帧 ≈39，
                    # 取 20 只标记不剔除；阈值依据 .codex/docs/.temp/v107-decode-check/
                    # 双路径对比（2026-10-05），全量摸底分布出来后复核
VIDEO_KEY_OF_V1_109 = "v1_109"   # 坏帧视频（门前沟中游 19：57）
V1_109_BAD_FROM_S = 1446.6       # 坏区起点（实测依据见 run-log 2026-10-02）
_VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

_SEG_EVENTS = {  # source -> events.csv（video_id/video_key/cut 区间来源）
    "dongchuan": REPO / "data/process/video_frames/video1_segment/events.csv",
    "video4": REPO / "data/process/video_frames/video4_segment/events.csv",
    "video5": REPO / "data/process/video_frames/video5_segment/events.csv",
}


def read_csv_rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))


_RAW_CACHE = None


def raw_video_table() -> dict:
    """一次 rglob 建 stem->Path 全表（v1 长中文名 / v4/v5 D 号文件名，沿用旧版逻辑）。"""
    global _RAW_CACHE
    if _RAW_CACHE is None:
        _RAW_CACHE = {}
        raw = REPO / "data/raw"
        for f in raw.rglob("*"):
            if f.is_file() and f.suffix.lower() in _VIDEO_EXT:
                _RAW_CACHE.setdefault(f.stem, f)
    return _RAW_CACHE


def locate_raw_video(video_id: str):
    tbl = raw_video_table()
    if video_id in tbl:
        return tbl[video_id]
    for stem, f in tbl.items():
        if video_id and len(video_id) > 4 and (video_id in stem or stem in video_id):
            return f
    return None


def load_events():
    """返回 (vid2key, vid2cuts, vid2kinds)。vid = (source, video_id)。"""
    vid2key, vid2cuts, vid2kinds = {}, {}, defaultdict(set)
    for src, p in _SEG_EVENTS.items():
        for r in read_csv_rows(p):
            vid = (src, r["video_id"])
            key = vid2key.setdefault(vid, r["video_key"])
            if key != r["video_key"]:
                raise RuntimeError(f"events.csv 中 video_id 对应多个 video_key：{vid}")
            kind = r["kind"].strip()
            vid2kinds[vid].add(kind)
            if kind == "cut":
                vid2cuts.setdefault(vid, []).append(
                    (float(r["start_sec"]), float(r["end_sec"])))
    return vid2key, vid2cuts, vid2kinds


def build_video_index():
    """段清单 -> 每视频 (group_id, fold, label)。组冲突即报错（组不变式）。"""
    folds = json.load(open(FOLDS_JSON, encoding="utf-8"))
    g2f = {}
    for k in folds["folds"]:
        for g in k["val_groups"]:
            g2f[g] = k["fold"]
    v2g = defaultdict(set)
    v2label = {}
    for r in read_csv_rows(SEG_MANIFEST):
        if r["source"] not in SOURCES:
            continue
        vid = (r["source"], r["video_id"])
        v2g[vid].add(r["group_id"])
        if r["kind"] == "pos":
            v2label[vid] = 1
        elif r["kind"] == "neg":
            v2label.setdefault(vid, 0)
    out = {}
    for vid, gs in v2g.items():
        if len(gs) != 1:
            raise RuntimeError(f"组不变式破坏：{vid} 属多组 {sorted(gs)}")
        g = next(iter(gs))
        out[vid] = {"group_id": g, "fold": g2f.get(g, 0), "label": v2label[vid]}
    return out


def merge_ranges(ranges):
    """合并相交/相邻区间。输入 [(a,b)...] 帧/秒皆可，要求同单位。"""
    rs = sorted(ranges)
    out = []
    for a, b in rs:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def forbidden_frames(vid, vkey, cuts, src_fps, n_src):
    """禁取帧区间列表（帧号，闭开区间 [a,b)）：cut 外扩 1s + v1_109 坏区。"""
    rs = []
    for s, e in cuts:
        a = max(0, math.ceil((s - CUT_MARGIN_S) * src_fps))
        b = min(n_src, math.ceil((e + CUT_MARGIN_S) * src_fps))
        if b > a:
            rs.append((a, b))
    if vkey == VIDEO_KEY_OF_V1_109:
        rs.append((math.ceil(V1_109_BAD_FROM_S * src_fps), n_src))
    return merge_ranges(rs)


def slot_ok(slot, forb):
    """片段槽 slot（覆盖帧 [slot*16, slot*16+16)）是否与全部禁取区间不相交。"""
    a, b = slot * CLIP_LEN, (slot + 1) * CLIP_LEN
    return all(b <= fa or a >= fb for fa, fb in forb)


def pick_slots(n_slots, forb, taken=frozenset(), k=N_CLIPS):
    """等间隔目标 + 最近合法槽重采。返回长度 min(k, n_slots) 的升序槽号列表；不够则 RuntimeError。

    k 参数（2026-10-14 叶海峰拍板推理放宽后新增）：推理工具允许视频不足 48 槽时
    按实际槽数取片段；不给 k 时行为与旧版完全一致（取 N_CLIPS 个，不足即报错），
    取材管线调用方零改动。"""
    k = min(k, n_slots)
    targets = [int(round(x)) for x in np.linspace(0, n_slots - 1, k)]
    chosen = []
    for t in targets:
        if slot_ok(t, forb) and t not in chosen and t not in taken:
            chosen.append(t)
            continue
        for d in range(1, n_slots):
            for s in (t - d, t + d):
                if 0 <= s < n_slots and s not in chosen and s not in taken and slot_ok(s, forb):
                    chosen.append(s)
                    break
            else:
                continue
            break
        else:
            raise RuntimeError(f"片段槽不足：目标 {t} 附近无合法槽")
    return sorted(chosen)


def _slots_key(slots):
    return ";".join(str(s * CLIP_LEN) for s in slots)


def extract_unit(video: Path, slots, unit_dir: Path, resume: bool, progress=None,
                 qc_path=None):
    """整条视频顺序解码一遍、路过各片段槽写 16 帧落盘。

    返回 (成功写的帧数, 首个失败帧的全局帧位或 None, 读失败帧位列表)。
    读失败不再立即返回：首个失败后改用 grab（只解不转）把余下帧扫完，收集
    全部读失败帧位一次性交回——调用方把所有区间并进禁区重采一次即可，避免
    逐轮重采每轮整片重解（research-auditor 2026-10-05 警告 5）。

    resume 时须三证齐全才跳过：768 帧齐 + slots.txt 旁证与本轮槽位逐位一致
    （防旧版重采产物与新清单错位——审计 W1）。成功后写 slots.txt 旁证。

    顺序解码两道防护（2026-10-05，research-auditor 阻断 2/3）：
      - 帧计数失步防护：每帧 read/grab 后核对 CAP_PROP_POS_FRAMES 与本地计数，
        不一致立即停止并报失败——解码器在坏区丢帧续解时其后所有帧位不可信，
        继续写会把错位帧当好帧。
      - 逐帧质检：每写一帧记灰度 std 与同片段邻帧灰度差，qc_path 给了就追加
        qc-frames.csv；灰度 std < QC_STD_LOW 记 lowstd 标记（只标记不剔除）。

    progress(done, total) 可选，每写完一个片段调一次，只用于报进度，不参与任何
    控制流。"""
    unit_dir.mkdir(parents=True, exist_ok=True)
    sidecar = unit_dir / "slots.txt"
    n_files = len(list(unit_dir.glob("clip*.jpg")))
    if (resume and n_files == N_CLIPS * CLIP_LEN and sidecar.exists()
            and sidecar.read_text(encoding="utf-8").strip() == _slots_key(slots)):
        return N_CLIPS * CLIP_LEN, None, []
    need = {}                       # 全局帧位 -> (片段序号, 片段内帧序号)
    for ci, slot in enumerate(slots):
        for fi in range(CLIP_LEN):
            need[slot * CLIP_LEN + fi] = (ci, fi)
    last_need = max(need)
    qc_f = None
    if qc_path is not None:
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not qc_path.exists()
        qc_f = open(qc_path, "a", encoding="utf-8", newline="")
        qc_w = csv.writer(qc_f)
        if new_file:
            qc_w.writerow(["unit_id", "clip", "frame", "file",
                           "gray_std", "prev_diff", "flag"])
    cap = cv2.VideoCapture(str(video))
    n_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 声明帧数：区分坏帧与视频尾
    n_write = 0
    prev_gray = {}                  # 片段序号 -> 上一帧灰度（算邻帧差分）
    fail_at = None
    fail_frames = []                # 全部读失败/失步帧位（供调用方并入禁区）

    def _close():
        cap.release()
        if qc_f:
            qc_f.close()

    fi = 0
    while True:
        if fail_at is None and fi > last_need:
            break                    # 主体扫描到最后一槽末帧：成功收尾
        # 主体阶段读帧；失败后续扫阶段改用 grab（只解不转），扫到视频尾收全坏区
        if fail_at is None:
            ok, fr = cap.read()
        else:
            ok, fr = cap.grab(), None
        if not ok or (fail_at is None and fr is None):
            fail_frames.append(fi)
            if fail_at is None:
                fail_at = fi         # 首个失败：转入续扫，把余下坏区一次收全
                fi += 1
                continue
            else:
                # 续扫阶段 grab 失败有两种：中途坏帧（继续收）与视频尾（停）。
                # 未到声明帧数视为坏帧继续收；声明值不可信（≤0）则到首个失败即停。
                if fi >= n_declared or n_declared <= 0:
                    break
                fi += 1
                continue
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if pos != fi + 1:
            fail_frames.append(fi)   # 帧计数失步：解码器丢帧续解，此处起不可信
            if fail_at is None:
                fail_at = fi
            break                    # 失步后帧位全不可信，续扫无意义，到此为止
        if fail_at is None and fi in need:
            ci, k = need[fi]
            if fr.shape[0] != RESIZE or fr.shape[1] != RESIZE:
                up = fr.shape[0] < RESIZE or fr.shape[1] < RESIZE
                fr = cv2.resize(fr, (RESIZE, RESIZE),
                                interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
            name = f"clip{ci:02d}_f{k:02d}.jpg"
            cv2.imwrite(str(unit_dir / name), fr)
            n_write += 1
            if qc_f is not None:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                gstd = round(float(gray.std()), 1)
                pdiff = (round(float(np.abs(gray.astype(np.int16)
                                            - prev_gray[ci]).mean()), 1)
                         if ci in prev_gray else "")
                prev_gray[ci] = gray
                qc_w.writerow([unit_dir.name, ci, fi, name, gstd, pdiff,
                               "lowstd" if gstd < QC_STD_LOW else ""])
            if k == CLIP_LEN - 1 and progress:
                progress(ci + 1, len(slots))
        fi += 1
    _close()
    if fail_at is None:
        sidecar.write_text(_slots_key(slots), encoding="utf-8")   # 成功后落槽位旁证
    return n_write, fail_at, fail_frames


def clean_unit_dir(unit_dir: Path):
    """重采前清掉半成品帧目录与槽位旁证（防新旧片段混装、旁证失配）。"""
    if unit_dir.is_dir():
        for p in unit_dir.glob("clip*.jpg"):
            p.unlink()
        (unit_dir / "slots.txt").unlink(missing_ok=True)


def pick_tiny(units, k):
    """tiny 冒烟：按时长降序挑 k 正 + k 负，组不重复。"""
    pos = sorted([u for u in units if u["label"] == 1], key=lambda u: -u["duration_hint"])
    neg = sorted([u for u in units if u["label"] == 0], key=lambda u: -u["duration_hint"])
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


def main():
    ap = argparse.ArgumentParser(description="EXP-008 R1 整条视频取材")
    ap.add_argument("--tiny", type=int, default=0, metavar="K", help="每类只出 K 个单位")
    ap.add_argument("--only", default="", help="只处理指定 unit_id（逗号分隔，调试用）")
    args = ap.parse_args()

    vid2key, vid2cuts, _ = load_events()
    vindex = build_video_index()
    print(f"[R1] 段清单东川三源视频 {len(vindex)} 条（整段纯 cut 视频已天然排除）")

    units = []
    for (src, video_id), info in sorted(vindex.items()):
        vkey = vid2key.get((src, video_id))
        if vkey is None:
            raise RuntimeError(f"events.csv 找不到 video_id：{(src, video_id)}")
        video = locate_raw_video(video_id)
        if video is None:
            raise RuntimeError(f"raw 未定位：{video_id}")
        cap = cv2.VideoCapture(str(video))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if src_fps <= 0 or n_src < N_CLIPS * CLIP_LEN:
            raise RuntimeError(f"{vkey} 元数据异常 fps={src_fps} frames={n_src}")
        units.append({
            "unit_id": vkey, "source": src, "video_id": video_id,
            "video_path": video, "src_fps": src_fps, "n_src": n_src,
            "duration_hint": n_src / src_fps, **info,
        })

    n_pos = sum(1 for u in units if u["label"] == 1)
    print(f"[R1] 单位池 {len(units)}（正 {n_pos}/负 {len(units)-n_pos}）")
    if args.tiny:
        units = pick_tiny(units, args.tiny)
        print(f"[R1] tiny 模式：取 {len(units)} 个")
    if args.only:
        keep = set(args.only.split(","))
        units = [u for u in units if u["unit_id"] in keep]
        print(f"[R1] only 模式：取 {len(units)} 个")

    out_rows, t_all = [], time.time()
    failures = []   # (unit_id, video_id, reason)。单单位失败不中止整场；结尾写
                    # failures.csv 排除清单并响亮报告（与「整段纯 cut 不入池」同原则）。
    for i, u in enumerate(units, 1):
        n_slots = u["n_src"] // CLIP_LEN
        if n_slots < N_CLIPS:
            failures.append((u["unit_id"], u["video_id"],
                             f"只能切 {n_slots} 片段 < {N_CLIPS}"))
            print(f"  [{i}/{len(units)}] !! {u['unit_id']} 排除：{failures[-1][2]}")
            continue
        forb = forbidden_frames((u["source"], u["video_id"]), u["unit_id"],
                                vid2cuts.get((u["source"], u["video_id"]), []),
                                u["src_fps"], u["n_src"])
        unit_dir = FRAMES_ROOT / u["unit_id"]
        try:
            slots = pick_slots(n_slots, forb)
            n, fail_at, fail_frames = extract_unit(
                u["video_path"], slots, unit_dir,
                resume=not args.tiny, qc_path=QC_FRAMES_CSV)
            retries = 0
            while fail_at is not None and retries < MAX_PICK_RETRY:
                # 外扩按指数增长（0.5s→1s→2s→…）：HEVC 坏区可达几十秒（v1_07 尾部实测
                # 17s、v1_109 实测数分钟），固定 0.5s 累计 12 次只有 6s，爬不出坏区。
                # 首轮就把全部读失败区间并入禁区（审计警告 5）：一次重采同时避开
                # 所有坏区，不再逐轮撞一个坏区就整片重解一遍。
                margin = 0.5 * (2 ** retries)
                zones = sorted(set(fail_frames)) or [fail_at]
                forb = merge_ranges(forb + [
                    (max(0, f - int(margin * u["src_fps"])),
                     min(u["n_src"], f + CLIP_LEN + int(margin * u["src_fps"])))
                    for f in zones])
                clean_unit_dir(unit_dir)
                slots = pick_slots(n_slots, forb)        # 禁区扩大后整体重挑，保证不重叠
                n, fail_at, fail_frames = extract_unit(
                    u["video_path"], slots, unit_dir,
                    resume=False, qc_path=QC_FRAMES_CSV)
                retries += 1
        except RuntimeError as e:
            failures.append((u["unit_id"], u["video_id"], f"选槽/取材失败：{e}"))
            print(f"  [{i}/{len(units)}] !! {u['unit_id']} 排除：{failures[-1][2]}")
            continue
        if n != N_CLIPS * CLIP_LEN or fail_at is not None:
            failures.append((u["unit_id"], u["video_id"],
                             f"写 {n} 帧（重采 {retries} 次仍不满，末次失败帧位 {fail_at}）"))
            print(f"  [{i}/{len(units)}] !! {u['unit_id']} 排除：{failures[-1][2]}")
            continue
        out_rows.append({
            "unit_id": u["unit_id"], "source": u["source"], "video_id": u["video_id"],
            "group_id": u["group_id"], "fold": u["fold"],
            "label": "pos" if u["label"] == 1 else "neg",
            "n_clips": N_CLIPS, "clip_len": CLIP_LEN,
            "src_fps": round(u["src_fps"], 3), "n_frames_src": u["n_src"],
            "duration_sec": round(u["n_src"] / u["src_fps"], 1),
            "clip_start_frames": ";".join(str(s * CLIP_LEN) for s in slots),
            "n_frames": n,
        })
        print(f"  [{i}/{len(units)}] + {u['unit_id']} "
              f"{'pos' if u['label'] else 'neg'} src={u['src_fps']:.1f}fps "
              f"dur={u['n_src']/u['src_fps']/60:.1f}min slots[0]={slots[0]} "
              f"slots[-1]={slots[-1]} fold={u['fold']} 重采{retries} ({time.time()-t_all:.0f}s)")

    UNITS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = ["unit_id", "source", "video_id", "group_id", "fold", "label",
              "n_clips", "clip_len", "src_fps", "n_frames_src", "duration_sec",
              "clip_start_frames", "n_frames"]
    with open(OUT_MANIFEST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[R1] 完成：{len(out_rows)} 单位（"
          f"正 {sum(1 for r in out_rows if r['label']=='pos')}/"
          f"负 {sum(1 for r in out_rows if r['label']=='neg')}），"
          f"耗时 {time.time()-t_all:.0f}s")
    print(f"[R1] 清单 -> {OUT_MANIFEST}")
    if failures:
        fail_csv = UNITS_ROOT / "unit-failures.csv"
        with open(fail_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "video_id", "reason"])
            w.writerows(failures)
        print(f"[R1] 排除 {len(failures)} 单位（响亮报告，不静默；理由落 {fail_csv}）：")
        for uid, vid, reason in failures:
            print(f"  !! {uid}: {reason}")
        print("[R1] 排除原则与「整段纯 cut 不入池」同（叶海峰 2026-10-03）："
              "画面无法支撑 48×16 单位形态的视频不进弱标签池；"
              "排除清单同步 run-log 与数据清单，待叶海峰过目。")
        sys.exit(2)   # 有排除：清单已写但退出码 2 区别于全绿 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
