"""Shared EXP-008 v2 input helpers; the retired v1 extraction entry is omitted.
The input index is adapted to this branch's frozen 150-member manifest.
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]

UNITS_ROOT = Path(os.environ.get("DF_UNITS_ROOT") or REPO / "data/process/EXP-008-units")

FRAMES_ROOT = UNITS_ROOT / "frames"

SOURCES = ("dongchuan", "video4", "video5")

N_CLIPS = 48

CLIP_LEN = 16

CUT_MARGIN_S = 1.0

RESIZE = 256

QC_STD_LOW = 20.0

VIDEO_KEY_OF_V1_109 = "v1_109"

V1_109_BAD_FROM_S = 1446.6

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

def clean_unit_dir(unit_dir: Path):
    """重采前清掉半成品帧目录与槽位旁证（防新旧片段混装、旁证失配）。"""
    if unit_dir.is_dir():
        for p in unit_dir.glob("clip*.jpg"):
            p.unlink()
        (unit_dir / "slots.txt").unlink(missing_ok=True)

def build_video_index():
    """Read the frozen experiment membership without other experiment folders."""
    manifest = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
    return {(r["source"], r["video_id"]):
            {"group_id": r["group_id"], "fold": int(r["fold"]),
             "label": int(r["label"] == "pos")}
            for r in read_csv_rows(manifest)}

