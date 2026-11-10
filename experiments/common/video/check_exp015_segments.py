# -*- coding: utf-8 -*-
"""EXP-015 签卡前只读核对片段的标注段归属与每段 64 帧上限。

运行：.venv/Scripts/python.exe experiments/common/video/check_exp015_segments.py
仅输出计数；不写数据、不选择或保存训练片段。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
LABELS = REPO / "data/process/EXP-008-units-v2/clip-labels.npz"
COORD = REPO / "data/process/EXP-008-units-src256/coord"
ZONES = REPO / "experiments/tools/oneoff/exp008-zone-audit/result.csv"
EVENTS = [REPO / f"data/process/video_frames/video{i}_segment/events.csv"
          for i in (1, 4, 5)]
CLIP_LEN = 16
MAX_CLIPS_PER_SEGMENT = 4


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = csv_rows(MANIFEST)
    labels_npz = np.load(LABELS, allow_pickle=True)
    ids = [str(value) for value in labels_npz["unit_ids"]]
    strict = np.asarray(labels_npz["strict"], dtype=int)
    assert ids == [row["unit_id"] for row in manifest]
    assert strict.shape == (150, 48)

    events = defaultdict(list)
    for source in EVENTS:
        for position, row in enumerate(csv_rows(source)):
            key = row["video_key"]
            events[key].append((str(source), position, row))
    newly_forbidden = {}
    for row in csv_rows(ZONES):
        newly_forbidden[row["unit_id"]] = {
            int(value) for value in row["killed_frames"].split(";") if value
        }

    by_segment_old = defaultdict(list)
    by_segment_new = defaultdict(list)
    n_strict = n_newly_forbidden = 0
    forbidden_labels = [0, 0]
    for row_index, row in enumerate(manifest):
        uid = row["unit_id"]
        fold = int(row["fold"])
        starts = [int(value) for value in row["clip_start_frames"].split(";") if value]
        assert len(starts) == 48
        coordinates = [float(coord["pts_time"])
                       for coord in csv_rows(COORD / f"{uid}.csv")]
        for clip_index, (start, label) in enumerate(zip(starts, strict[row_index])):
            if label < 0:
                continue
            n_strict += 1
            begin = coordinates[start]
            end = coordinates[start + CLIP_LEN - 1]
            matches = [
                (source, position) for source, position, event in events[uid]
                if event["kind"] == ("pos" if label == 1 else "neg")
                and float(event["start_sec"]) <= begin
                and end < float(event["end_sec"])
            ]
            if len(matches) != 1:
                raise AssertionError(f"{uid} 片段 {clip_index} 标注段匹配数 {len(matches)}")
            segment = (uid, *matches[0], fold)
            by_segment_old[segment].append(clip_index)
            if start in newly_forbidden.get(uid, set()):
                n_newly_forbidden += 1
                forbidden_labels[label] += 1
            else:
                by_segment_new[segment].append(clip_index)

    print(f"严格片段 {n_strict}，新禁区剔除 {n_newly_forbidden}"
          f"（正 {forbidden_labels[1]}、负 {forbidden_labels[0]}），"
          f"剩余 {n_strict - n_newly_forbidden}")
    for name, segments in (("旧", by_segment_old), ("加现行禁区", by_segment_new)):
        selected = sum(min(MAX_CLIPS_PER_SEGMENT, len(clips))
                       for clips in segments.values())
        print(f"{name}：含片段标注段 {len(segments)}，每段最多 4 片段后 {selected}")
        for held_out in range(1, 6):
            old_count = sum(len(clips) for segment, clips in segments.items()
                            if segment[-1] != held_out)
            new_count = sum(min(MAX_CLIPS_PER_SEGMENT, len(clips))
                            for segment, clips in segments.items()
                            if segment[-1] != held_out)
            print(f"  留出折 {held_out}：训练侧 {old_count} → {new_count}"
                  f"；目标批大小 24 时每轮 {int(np.ceil(new_count / 24))} 批")


if __name__ == "__main__":
    main()
