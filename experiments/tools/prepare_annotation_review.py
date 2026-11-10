"""Freeze metadata-only, camera-stratified annotation review targets.

The source inventory contains provisional saved labels, used ONLY as sampling
strata. Outputs are review assignments, never training labels or video clips.
Cached images supply context; they do not certify full-motion observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_UNITS = {"v1_16", "v1_74", "v1_95", "v1_96", "v1_112"}
LABELS = ("N", "P", "T")


def candidates(inventory: dict, duration: float) -> dict:
    result = {label: defaultdict(list) for label in LABELS}
    for unit in inventory["units"]:
        if unit["unit_id"] in CALIBRATION_UNITS:
            continue
        source = unit.get("annotation_source") or {}
        for interval in unit["intervals"]:
            label = interval["label"]
            length = interval["end_sec"] - interval["start_sec"]
            if label not in result or length < duration:
                continue
            # Bound proposal density without treating a long interval as more
            # independent evidence. Adjacent review targets never overlap.
            count = min(20, int(math.floor(length / duration)))
            for i in range(count):
                start = interval["start_sec"] + (length - duration) * (i + .5) / count
                item = {"unit_id": unit["unit_id"], "camera": unit["source_camera"],
                        "group_id": unit["group_id"], "start_sec": round(start, 6),
                        "end_sec": round(start + duration, 6), "sampling_stratum": label,
                        "annotation_source": source,
                        "annotation_line": interval["annotation_line"]}
                result[label][unit["source_camera"]].append(item)
    return result


def select_targets(by_label: dict, per_class: int, seed: int) -> list[dict]:
    rng, selected = random.Random(seed), []
    for label in LABELS:
        buckets = by_label[label]
        cameras = sorted(buckets)
        rng.shuffle(cameras)
        for camera in cameras:
            rng.shuffle(buckets[camera])
        chosen, usage = [], defaultdict(int)
        # Give cameras turns. Prefer fresh videos before taking further
        # windows from a video, rather than allocating by number of files.
        for max_per_video in range(1, per_class + 1):
            progress = True
            while progress and len(chosen) < per_class:
                progress = False
                for camera in cameras:
                    for index, item in enumerate(buckets[camera]):
                        if usage[item["unit_id"]] >= max_per_video:
                            continue
                        conflict = any(old["unit_id"] == item["unit_id"] and
                                       max(old["start_sec"], item["start_sec"]) <
                                       min(old["end_sec"], item["end_sec"])
                                       for old in selected + chosen)
                        if conflict:
                            continue
                        chosen.append(buckets[camera].pop(index))
                        usage[item["unit_id"]] += 1
                        progress = True
                        break
                    if len(chosen) == per_class:
                        break
            if len(chosen) == per_class:
                break
        if len(chosen) != per_class:
            raise ValueError(f"Insufficient non-overlapping {label} candidates: {len(chosen)}/{per_class}; do not fill by duplication")
        selected.extend(chosen)
    rng.shuffle(selected)
    for index, item in enumerate(selected, 1):
        item["review_id"] = f"R{index:03d}"
    return selected


def add_context(selected: list[dict], root: Path, context: float) -> None:
    needed = defaultdict(set)
    for item in selected:
        index = item["annotation_source"].get("frames_index_candidate_path")
        if not index:
            raise ValueError("Missing cached index for " + item["unit_id"])
        needed[index].add(item["unit_id"])
    cached = {}
    for index, ids in needed.items():
        with (root / index).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                uid = row.get("video_key", "")
                if uid in ids:
                    cached.setdefault((index, uid), []).append(row)
    for item in selected:
        index = item["annotation_source"]["frames_index_candidate_path"]
        rows = cached.get((index, item["unit_id"]), [])
        frames = []
        for row in rows:
            t = float(row["t_sec"])
            if item["start_sec"] - context <= t <= item["end_sec"] + context:
                frames.append({"t_sec": t, "image_path":
                               (Path(index).parent / "frames" / row["frame_file"]).as_posix(),
                               "inside_target": item["start_sec"] <= t < item["end_sec"]})
        frames.sort(key=lambda row: row["t_sec"])
        item["cached_context_frames"] = frames
        item["evidence_status"] = "cached_images_only_not_full_motion"
        item["full_motion_seen"] = False
        item["needs_targeted_video_check"] = (len(frames) < 3 or not any(row["inside_target"] for row in frames))


def prepare(inventory_path: Path, out: Path, root: Path, seed: int = 20261107,
            per_class: int = 10, duration: float = 3, context: float = 5) -> dict:
    raw = inventory_path.read_bytes()
    inventory = json.loads(raw.decode("utf-8-sig"))
    selected = select_targets(candidates(inventory, duration), per_class, seed)
    add_context(selected, root, context)
    scope = {"purpose": "Fixed review denominator; not training supervision or geological truth",
             "inventory": inventory_path.resolve().relative_to(root.resolve()).as_posix(),
             "inventory_sha256": hashlib.sha256(raw).hexdigest(),
             "seed": seed, "per_provisional_stratum": per_class, "target_duration_sec": duration,
             "context_either_side_sec": context, "excluded_calibration_units": sorted(CALIBRATION_UNITS),
             "selection": "Camera round-robin with shuffled candidates; prefer fresh videos; no overlapping targets or score selection",
             "training_ready": False, "review_complete": False,
             "uncertain_counts_in_denominator": True, "post_review_replacement_allowed": False,
             "same_model_review_is_not_independent_expert_accuracy": True,
             "items": selected}
    blind = {"purpose": "Review actual visible content; do not consult selection-key labels or model scores",
             "instructions": "Judge the target with context, allow unresolved, record visible evidence and observation limits. Sparse cached images are not full-motion viewing.",
             "items": [{"review_id": item["review_id"], "target_start_sec": item["start_sec"],
                        "target_end_sec": item["end_sec"], "context": item["cached_context_frames"],
                        "needs_targeted_video_check": item["needs_targeted_video_check"]} for item in selected]}
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("selection-key.json", "blind-items.json", "review-round1.json", "review-round2.json"):
        if (out / filename).exists():
            raise ValueError("Refusing to overwrite a frozen assignment or review: " + str(out / filename))
    (out / "selection-key.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "blind-items.json").write_text(json.dumps(blind, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for round_number in (1, 2):
        form = {"round": round_number, "status": "not_reviewed", "items":
                [{"review_id": item["review_id"], "label": None, "visible_evidence": None,
                  "observation_limits": None, "full_motion_seen": False} for item in selected]}
        (out / f"review-round{round_number}.json").write_text(json.dumps(form, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"review_targets": len(selected), "cameras": len({item["camera"] for item in selected}),
            "strata_counts": {label: sum(item["sampling_stratum"] == label for item in selected) for label in LABELS},
            "T_cameras": len({item["camera"] for item in selected if item["sampling_stratum"] == "T"}),
            "needs_targeted_video_check": sum(item["needs_targeted_video_check"] for item in selected),
            "review_complete": False, "training_ready": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inventory, args.out, ROOT), ensure_ascii=False))


if __name__ == "__main__":
    main()
