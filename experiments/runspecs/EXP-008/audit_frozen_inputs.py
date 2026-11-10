"""Audit the frozen EXP-008 inputs without training or extracting features.

Reads every selected JPEG and the historical QC/coordinate tables. Findings are
written separately; original frames, features, labels and metrics are untouched.
"""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
UNITS = ROOT / "data/process/EXP-008-units-v2"
COORD = ROOT / "data/process/EXP-008-units-src256/coord"
FEATURES = ROOT / "data/results/exp008/unitfeats-v2-150/unit-features.npz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def audit_unit(row):
    uid = row["unit_id"]
    starts = [int(x) for x in row["clip_start_frames"].split(";")]
    result = {"unit_id": uid, "clips": len(starts), "errors": [],
              "temporal_findings": [], "jpeg_warnings": []}
    errors = result["errors"]
    if len(starts) != 48 or len(set(starts)) != 48:
        errors.append("clip_count_or_unique_starts")
    if starts != sorted(starts) or any(s % 16 for s in starts):
        errors.append("slot_order_or_alignment")
    if any(b < a + 16 for a, b in zip(starts, starts[1:])):
        errors.append("overlapping_clips")
    folder = UNITS / "frames" / uid
    sidecar = folder / "slots.txt"
    if not sidecar.exists() or sidecar.read_text().strip() != row["clip_start_frames"]:
        errors.append("slots_sidecar_mismatch")
    qc_path = UNITS / "qc" / (uid + ".csv")
    qc = np.genfromtxt(str(qc_path), delimiter=",", skip_header=1)
    coord_path = COORD / (uid + ".csv")
    coord = np.genfromtxt(str(coord_path), delimiter=",", skip_header=1)
    if len(qc) != int(row["n_frames_mw"]) or not np.array_equal(qc[:, 0], np.arange(len(qc))):
        errors.append("qc_frame_coverage")
    # The QC snapshot records the exact per-frame timestamps used for extraction.
    # Coordinate tables can include extra terminal frames; report, do not silently
    # truncate the QC history or replace its coordinates with a later table.
    result["coord_rows"] = len(coord)
    result["qc_coord_frame_count_difference"] = len(coord) - len(qc)
    if any(s < 0 or s + 16 > len(qc) for s in starts):
        errors.append("out_of_bounds")
        return result
    pts = qc[:, 1]
    differences = np.diff(pts)
    positive_steps = differences[differences > 0]
    median_step = float(np.median(positive_steps))
    hole_limit = max(0.2, 3 * median_step)
    selected = np.array([s + k for s in starts for k in range(16)])
    result.update({"qc_rows": len(qc),
                   "whole_video_low_std_frames": int((qc[:, 2] < 20).sum()),
                   "selected_qc_min_std": float(qc[selected, 2].min()),
                   "selected_qc_low_std_frames": int((qc[selected, 2] < 20).sum()),
                   "qc_coord_max_seconds_difference": float(np.abs(qc[:min(len(qc), len(coord)), 1] - coord[:min(len(qc), len(coord)), 1]).max()),
                   "positive_step_median_seconds": median_step,
                   "large_gap_threshold_seconds": hole_limit,
                   "qc_sha256": digest(qc_path), "coord_sha256": digest(coord_path)})
    if result["selected_qc_low_std_frames"]:
        errors.append("selected_historical_low_std_frame")
    padding_hits = [i for i, s in enumerate(starts)
                    if np.any(qc[max(0, s - 16):min(len(qc), s + 32), 2] < 20)]
    if padding_hits:
        errors.append("historical_low_std_padding_overlap")
    result["selected_clips_with_rounded_equal_timestamps"] = 0
    result["clip_first_last_spans_seconds"] = []
    result["selected_jpegs"] = 0
    result["jpeg_min_std"] = None
    result["jpeg_max_std_difference_from_qc"] = 0.0
    for ci, start in enumerate(starts):
        clip_pts = pts[start:start + 16]
        dt = np.diff(clip_pts)
        local_positive = dt[dt > 1e-6]
        local_step = float(np.median(local_positive)) if len(local_positive) else 0.0
        local_hole_limit = max(hole_limit, 3 * local_step)
        result["clip_first_last_spans_seconds"].append(float(clip_pts[-1] - clip_pts[0]))
        if np.any(np.abs(dt) < 1e-9):
            result["selected_clips_with_rounded_equal_timestamps"] += 1
        # Equality of a rounded time string does not establish repeated images.
        if np.any(dt < -1e-6) or np.any(dt > local_hole_limit + 1e-6):
            result["temporal_findings"].append({
                "clip_index_zero_based": ci,
                "middleware_frame_start": start,
                "first_last_seconds": [float(clip_pts[0]), float(clip_pts[-1])],
                "local_positive_step_median_seconds": local_step,
                "minimum_step_seconds": float(dt.min()),
                "maximum_step_seconds": float(dt.max())})
        if uid == "v1_109" and np.any(clip_pts >= 1446.6):
            errors.append("known_v1_109_bad_tail")
        for fi in range(16):
            path = folder / ("clip%02d_f%02d.jpg" % (ci, fi))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                errors.append("missing_or_unreadable:" + path.name)
                continue
            result["selected_jpegs"] += 1
            if image.shape != (256, 256, 3):
                errors.append("jpeg_shape:" + path.name + ":" + str(image.shape))
            std = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std())
            result["jpeg_min_std"] = std if result["jpeg_min_std"] is None else min(std, result["jpeg_min_std"])
            diff = abs(std - float(qc[start + fi, 2]))
            result["jpeg_max_std_difference_from_qc"] = max(diff, result["jpeg_max_std_difference_from_qc"])
            if std < 20:
                result["jpeg_warnings"].append({"clip": ci, "frame": fi, "std": std,
                                                "qc_std": float(qc[start + fi, 2])})
    result["jpeg_files_on_disk"] = sum(1 for _ in folder.glob("clip*.jpg"))
    if result["jpeg_files_on_disk"] != 768 or result["selected_jpegs"] != 768:
        errors.append("jpeg_count")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "data/metadata/experiment-archive/exp008-input-quality-20261110.json")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    started = time.monotonic()
    rows = csv_rows(UNITS / "unit-manifest.csv")
    data = np.load(str(FEATURES), allow_pickle=False)
    alignment = {
        "ids": data["unit_ids"].tolist() == [r["unit_id"] for r in rows],
        "labels": data["labels"].tolist() == [int(r["label"] == "pos") for r in rows],
        "groups": data["groups"].tolist() == [r["group_id"] for r in rows],
        "folds": data["folds"].tolist() == [int(r["fold"]) for r in rows],
        "shape": list(data["feats"].shape) == [150, 48, 512],
        "finite_features": bool(np.isfinite(data["feats"]).all())}
    group_folds = defaultdict(set)
    for row in rows:
        group_folds[row["group_id"]].add(int(row["fold"]))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit_unit, row): row["unit_id"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 15 == 0:
                print("Audited %d/%d videos, %.1fs" % (len(results), len(rows), time.monotonic() - started), flush=True)
    results.sort(key=lambda r: r["unit_id"])
    summary = {
        "units": len(rows), "clips": sum(r["clips"] for r in results),
        "selected_jpegs_read": sum(r.get("selected_jpegs", 0) for r in results),
        "historical_qc_rows": sum(r.get("qc_rows", 0) for r in results),
        "historical_whole_video_low_std_frames": sum(r.get("whole_video_low_std_frames", 0) for r in results),
        "selected_historical_low_std_frames": sum(r.get("selected_qc_low_std_frames", 0) for r in results),
        "units_with_integrity_errors": [r["unit_id"] for r in results if r["errors"]],
        "units_with_large_internal_time_gaps": [r["unit_id"] for r in results if r["temporal_findings"]],
        "clips_with_large_internal_time_gaps": sum(len(r["temporal_findings"]) for r in results),
        "jpeg_std_below_20": sum(len(r["jpeg_warnings"]) for r in results),
        "cache_alignment": alignment,
        "groups_spanning_folds": [g for g, folds in group_folds.items() if len(folds) > 1],
        "class_counts": dict(Counter(r["label"] for r in rows)),
        "elapsed_seconds": round(time.monotonic() - started, 2)}
    report = {
        "scope": "All 150 frozen units, historical QC/coordinates and every selected JPEG; no training, video regeneration or feature extraction.",
        "manifest_sha256": digest(UNITS / "unit-manifest.csv"),
        "features_sha256": digest(FEATURES),
        "script_sha256": digest(Path(__file__)),
        "historical_cut_snapshot": "Original events.csv snapshot unavailable; present-day labels are not substituted for historical labels.",
        "summary": summary, "units": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    print("Report:", args.output, flush=True)


if __name__ == "__main__":
    main()
