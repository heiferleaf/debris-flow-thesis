"""Enumerate the 150-video first-PTS-anchored grid without media decoding."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.video.full_coverage_sampling import (  # noqa: E402
    VERSION, deduplicate_physical, diagnostic_loss_coefficients, epoch_order,
    full_grid, integrate_release, physical_window_key, pts_runs, update_scenarios, validate_release,
)

DEFAULT_SOURCES = "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
DEFAULT_RELEASE = "data/metadata/annotation-revisions/npt-20261108-evidence-working-v2/released-intervals.json"
DEFAULT_CACHE = "data/process/EXP-016/metadata-v4/verified-cache-files.csv"
DEFAULT_OUTPUT = "data/metadata/EXP-028/full-coverage-plan-20261108-v1"


def rooted(path, root=ROOT):
    path = Path(path)
    return path if path.is_absolute() else root / path


def ref(path, root=ROOT):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def fingerprint(path, root=ROOT):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return {"path": ref(path, root), "sha256": h.hexdigest(), "size_bytes": path.stat().st_size,
            "verification": "hashed_in_this_metadata_run"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_coordinates(source, root):
    """Read and validate all frozen rows; reuse their previously checked digests."""
    n = int(source["n_frames"])
    pts, stds = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
    coord_path, qc_path = rooted(source["coord_path"], root), rooted(source["qc_path"], root)
    with coord_path.open(encoding="utf-8-sig", newline="") as cf, qc_path.open(encoding="utf-8-sig", newline="") as qf:
        coord, qc = csv.DictReader(cf), csv.DictReader(qf)
        count = 0
        while True:
            frame, quality = next(coord, None), next(qc, None)
            if frame is None or quality is None:
                if frame is not None or quality is not None:
                    raise ValueError("Coordinate/QC frame count disagreement: " + source["unit_id"])
                break
            if count >= n or int(frame["frame"]) != count or int(quality["frame"]) != count:
                raise ValueError("Coordinate/QC noncontiguous frame identity")
            timestamp, qt, std = float(frame["pts_time"]), float(quality["pts_time"]), float(quality["gray_std"])
            if not all(math.isfinite(v) for v in (timestamp, qt, std)) or abs(timestamp - qt) > .001001:
                raise ValueError("Nonfinite or disagreeing coordinate/QC PTS")
            pts[count], stds[count] = timestamp, std
            count += 1
    if count != n:
        raise ValueError("Frozen frame-count disagreement: " + source["unit_id"])
    forbidden, clean = json.loads(source["forbidden_ranges_json"]), json.loads(source["clean_runs_json"])
    for ranges in (forbidden, clean):
        for left, right in ranges:
            if type(left) is not int or type(right) is not int or not 0 <= left < right <= n:
                raise ValueError("Invalid frozen source-frame range")
    gray_prefix = np.concatenate(([0], np.cumsum(stds < 20)))
    backward_prefix = np.concatenate(([0], np.cumsum(np.concatenate(([False], np.diff(pts) < 0)))))
    return pts, stds, forbidden, clean, gray_prefix, backward_prefix


def cache_lookup(path, sources, root):
    """Reuse verified JPEG index; stat relevant files, never open/decode/hash JPEGs."""
    result, aliases = {}, {s["unit_id"]: s["middleware_sha256"] for s in sources}
    duplicate_count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["unit_id"] not in aliases:
                continue
            index = int(row["frame_index"])
            key = aliases[row["unit_id"]], index
            for field in ("jpeg_sha256", "source_bgr_sha256", "rgb112_sha256"):
                if len(row[field]) != 64 or any(c not in "0123456789abcdef" for c in row[field]):
                    raise ValueError("Invalid cached frame fingerprint")
            if key in result:
                previous = result[key]
                if any(row[field] != previous[field] for field in ("jpeg_sha256", "source_bgr_sha256", "rgb112_sha256", "source_pts_seconds")):
                    raise ValueError("Physical cache alias conflict")
                duplicate_count += 1
            else:
                result[key] = row
    return result, duplicate_count


def overlap_report(rows):
    matched = sorted((r for r in rows if r["frame_indices"] is not None),
                     key=lambda r: (r["actual_start_seconds"], r["actual_end_seconds"]))
    overlapping, max_seconds = 0, 0.0
    previous_end = None
    frame_use = Counter()
    for row in matched:
        if previous_end is not None:
            intersection = max(0.0, min(previous_end, row["actual_end_seconds"]) - row["actual_start_seconds"])
            overlapping += int(intersection > 1e-9)
            max_seconds = max(max_seconds, intersection)
        previous_end = max(previous_end, row["actual_end_seconds"]) if previous_end is not None else row["actual_end_seconds"]
        frame_use.update(row["frame_indices"])
    return {"actual_PTS_overlapping_window_count": overlapping,
            "max_actual_PTS_overlap_seconds": max_seconds,
            "sampled_frame_exposures": sum(frame_use.values()), "unique_sampled_source_frames": len(frame_use),
            "sampled_source_frame_reuse_exposures": sum(v - 1 for v in frame_use.values())}


def count_rows(rows):
    matched = [r for r in rows if r["frame_indices"] is not None]
    return {"full_grid_count": len(rows), "PTS_matched_count": len(matched),
            "semantic_complete_count": sum(r.get("semantic_three_class_complete", False) for r in rows),
            "semantic_existence_complete_count": sum(r.get("semantic_existence_complete", False) for r in rows),
            "legacy_guard_pass_count": sum(r.get("legacy_guard_pass", False) for r in rows),
            "gray_QC_pass_count": sum(r.get("gray_QC_pass", False) for r in rows),
            "PTS_pass_count": sum(r.get("PTS_pass", False) for r in rows),
            "conservative_legal_K": sum(r.get("conservative_legal", False) for r in rows),
            "cache_complete_K": sum(r.get("cache_complete_legal", False) for r in rows),
            "cache_missing_legal_K": sum(r.get("conservative_legal", False) and not r.get("cache_complete_legal", False) for r in rows),
            "specific_scope_gap_count": sum(r.get("specific_scope_gap", False) for r in rows),
            "actual_span_min_seconds": min((r["actual_span_seconds"] for r in matched), default=None),
            "actual_span_max_seconds": max((r["actual_span_seconds"] for r in matched), default=None)}


def prepare(sources_path, release_path, cache_path, output, *, root=ROOT, seed=7, command=None):
    start = time.perf_counter()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing plan: " + str(output))
    sources = read_csv(sources_path)
    if len({s["unit_id"] for s in sources}) != len(sources):
        raise ValueError("Repeated source unit")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") != 1 or release.get("training_ready") is not False:
        raise ValueError("Expected bounded working-label release schema1, training_ready=false")
    grouped = defaultdict(list)
    for row in release["rows"]:
        grouped[row["unit_id"]].append(row)
    if set(grouped) != {s["unit_id"] for s in sources}:
        raise ValueError("Source/release units disagree; no implicit fallback")
    grouped = {uid: validate_release(rows) for uid, rows in grouped.items()}
    inputs = [fingerprint(p, root) for p in (sources_path, release_path, cache_path)]
    for name in ("experiments/common/video/full_coverage_sampling.py", "experiments/runspecs/EXP-028/prepare_full_coverage_plan.py",
                 "experiments/common/video/pts_clip_sampling.py", "experiments/tests/test_full_coverage_sampling.py"):
        inputs.append(fingerprint(root / name, root))
    cache, cache_alias_count = cache_lookup(cache_path, sources, root)
    cache_stat, all_rows, mappings, unit_reports, gap_rows = {}, [], [], [], []
    camera_for_media = {}
    for ordinal, source in enumerate(sources, 1):
        uid, media = source["unit_id"], source["middleware_sha256"]
        parts = source["group_id"].split("|")
        if len(parts) != 3 or parts[0] != "d" or not parts[1]:
            raise ValueError("Expected explicit registered camera in group_id")
        camera = parts[1]
        if any(r["source_camera"] != camera or r["group_id"] != source["group_id"] for r in grouped[uid]):
            raise ValueError("Release/source camera or group disagreement: " + uid)
        if media in camera_for_media and camera_for_media[media] != camera:
            raise ValueError("Same physical media assigned different cameras")
        camera_for_media[media] = camera
        pts, stds, forbidden, clean, gray, back = load_coordinates(source, root)
        local = []
        for row in full_grid(pts):
            row.update(unit_id=uid, source_camera=camera, group_id=source["group_id"],
                       physical_video_id=media, frozen_middleware_sha256=media,
                       motion_state="unknown", training_ready=False,
                       physical_window_key=None, legacy_guard_pass=False, gray_QC_pass=False,
                       PTS_pass=False, conservative_legal=False, cache_complete_legal=False,
                       specific_scope_gap=False, rejected_reasons=list(row["sampling_reasons"]))
            indices = row["frame_indices"]
            if indices is not None:
                first, last = indices[0], indices[-1]
                row["physical_window_key"] = physical_window_key(media, indices)
                row["legacy_guard_pass"] = not any(first < right and last >= left for left, right in forbidden)
                row["gray_QC_pass"] = bool(gray[last + 1] == gray[first])
                row["PTS_pass"] = bool(back[last + 1] == back[first + 1])
                row["composition"] = integrate_release(row["actual_start_seconds"], row["actual_end_seconds"], grouped[uid])
                row["semantic_three_class_complete"] = row["composition"]["semantic_three_class_complete"]
                row["semantic_existence_complete"] = row["composition"]["semantic_existence_complete"]
                row["rejected_reasons"].extend(row["composition"]["semantic_reasons"])
                for field, reason in (("legacy_guard_pass", "legacy_frozen_forbidden_frame_range_including_semantic_cut"),
                                      ("gray_QC_pass", "source_gray_std_below20"), ("PTS_pass", "source_PTS_backstep")):
                    if not row[field]:
                        row["rejected_reasons"].append(reason)
                row["conservative_legal"] = bool(row["semantic_three_class_complete"] and row["legacy_guard_pass"] and row["gray_QC_pass"] and row["PTS_pass"])
                positive_duration = math.fsum(row["composition"]["state_seconds"].get(s, 0) for s in ("T", "P"))
                row["specific_scope_gap"] = bool(positive_duration > 0 and not row["legacy_guard_pass"])
                if row["specific_scope_gap"]:
                    gap_rows.append({"unit_id": uid, "source_camera": camera, "grid_ordinal": row["grid_ordinal"],
                                     "actual_start_seconds": row["actual_start_seconds"], "actual_end_seconds": row["actual_end_seconds"],
                                     "physical_window_key": row["physical_window_key"], "released_T_P_seconds": positive_duration,
                                     "semantic_complete": row["semantic_three_class_complete"],
                                     "reason": "new_released_T_or_P_intersects_legacy_frozen_forbidden_guard; no automatic guard removal"})
                missing = []
                for index, pt in zip(indices, row["source_pts_seconds"]):
                    ck = media, index
                    cached = cache.get(ck)
                    if cached is None:
                        missing.append(index)
                        continue
                    if abs(float(cached["source_pts_seconds"]) - pt) > 1e-8 or abs(float(cached["physical_pts_seconds"]) - pt) > .001001:
                        raise ValueError("Cached source/physical PTS conflict")
                    if ck not in cache_stat:
                        cp = rooted(cached["cache_path"], root)
                        cache_stat[ck] = cp.is_file() and cp.stat().st_size == int(cached["size_bytes"])
                    if not cache_stat[ck]:
                        missing.append(index)
                row["cache_missing_frame_indices"] = missing
                row["cache_complete_for_matched_window"] = not missing
                row["cache_complete_legal"] = bool(row["conservative_legal"] and not missing)
                if row["conservative_legal"] and missing:
                    row["cache_gap_reason"] = "missing_from_verified_index_or_file_size_presence_check"
            local.append(row)
        all_rows.extend(local)
        tail_start = float(pts[0]) + len(local) * 3.0
        unit_reports.append({"unit_id": uid, "source_camera": camera, "physical_video_id": media,
                             "source_video_id": source["video_id"], "n_frames": len(pts),
                             "first_PTS_seconds": float(pts[0]), "last_PTS_seconds": float(pts[-1]),
                             "max_PTS_seconds": float(np.max(pts)), "PTS_monotone_run_count": len(pts_runs(pts)),
                             "nominal_tail_seconds": max(0.0, float(np.max(pts)) - tail_start),
                             "legacy_frozen_clean_run_count": len(clean),
                             **count_rows(local), **overlap_report(local)})
        mappings.append({"unit_id": uid, "source_camera": camera, "group_id": source["group_id"],
                         "source_video_id": source["video_id"], "physical_video_id": media,
                         "middleware_path": source["middleware_path"], "frozen_middleware_sha256": media,
                         "media_fingerprint_scope": "frozen_middleware_reference_reused_not_original_video_new_hash",
                         "coord_path": source["coord_path"], "frozen_coord_sha256": source["coord_sha256"],
                         "qc_path": source["qc_path"], "frozen_qc_sha256": source["qc_sha256"],
                         "metadata_hash_scope": "prior_verified_fingerprints_reused; all_actual_CSV_rows_read_and_validated_not_rehashed",
                         "forbidden_ranges_json": source["forbidden_ranges_json"], "clean_runs_json": source["clean_runs_json"],
                         "supervision_timebase_status": source["supervision_timebase_status"], "motion_state": "unknown"})
        if ordinal % 10 == 0 or ordinal == len(sources):
            print(json.dumps({"completed_units": ordinal, "total_units": len(sources), "grid_rows": len(all_rows),
                              "seconds": round(time.perf_counter() - start, 2)}), flush=True)
    unique, duplicate_rows = deduplicate_physical(all_rows)
    retained = {id(r) for r in unique}
    for row in all_rows:
        row["retained_unique_physical_window"] = id(row) in retained and row["physical_window_key"] is not None
    legal = [r for r in unique if r["conservative_legal"]]
    cached_legal = [r for r in legal if r["cache_complete_legal"]]
    coefficients, coefficient_cameras = diagnostic_loss_coefficients(legal)
    cached_coefficients, _ = diagnostic_loss_coefficients(cached_legal)
    coeff_by_key = {r["physical_window_key"]: r["loss_coefficient_w"] for r in coefficients}
    for row in unique:
        if row["physical_window_key"] in coeff_by_key:
            row["diagnostic_full_pool_loss_coefficient"] = coeff_by_key[row["physical_window_key"]]
    by_camera = []
    for camera in sorted({r["source_camera"] for r in all_rows}):
        members = [r for r in unique if r["source_camera"] == camera]
        local_units = [u for u in unit_reports if u["source_camera"] == camera]
        by_camera.append({"source_camera": camera, "source_unit_count": len(local_units),
                          "physical_video_count": len({u["physical_video_id"] for u in local_units}),
                          **count_rows(members),
                          "actual_PTS_overlapping_window_count": sum(u["actual_PTS_overlapping_window_count"] for u in local_units),
                          "unique_sampled_source_frames": sum(u["unique_sampled_source_frames"] for u in local_units)})
    q_counts = Counter()
    for row in legal:
        q = row["composition"]["q_N_T_P"]
        q_counts["pure_" + ("N", "T", "P")[q.index(1.0)] if 1.0 in q else "mixed_N_T_P"] += 1
    summary = {"version": VERSION, "status": "metadata_only_full_pool_diagnostic_not_training_plan",
               "created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
               "training_ready": False, "no_training": True, "no_media_decode": True, "paid_operations": 0,
               "unit_count": len(sources), "registered_camera_count": len(by_camera),
               "source_frame_rows_read": sum(int(s["n_frames"]) for s in sources),
               "grid": {"anchor": "each_media_first_PTS", "nominal_span_seconds": 3, "step_seconds": 3,
                        "sample_count": 16, "tail_end_aligned_window_added": False,
                        "reset_at_labels_QC_or_clean_runs": False, "horizon": "max_PTS; backstep ambiguity explicitly rejected",
                        "nominal_supports_nonoverlapping": True,
                        "actual_PTS_supports_may_overlap": True, "time_integration": "[actual_first_PTS,actual_last_PTS)",
                        "matching_helper": "pts_clip_sampling.nearest_window", "point_tolerance_seconds": .100001,
                        "actual_span_allowed_seconds": [2.9, 3.1]},
               "counts_all_grid_rows": count_rows(all_rows), "counts_unique_physical": count_rows(unique),
               "exact_physical_duplicate_rows": len(duplicate_rows), "physical_conflicts": 0,
               "physical_window_identity": "SHA256(canonical JSON of frozen middleware SHA256 and ordered frame indices); not a new source-video hash",
               "physical_original_video_identity": "frozen source middleware SHA256 deduplicates exact aliases; no inferred near-duplicate/original-to-derivative index mapping",
               "semantic_complete_definition": "full actual duration covered by current release mask_three_class=true N/T/P; maskfalse P, U, pending cuts and gaps remain unknown",
               "legacy_guard_scope": "Frozen forbidden frame ranges include semantic cuts and technical exclusions; retained conservatively and not described as purely technical",
               "cache_check_scope": "original verified index plus current relevant-file existence and byte size; no new JPEG hash or pixel decoding",
               "cache_index_rows_in_pool": len(cache), "cache_index_physical_alias_rows": cache_alias_count,
               "relevant_cached_frame_files_stat_checked": len(cache_stat),
               "legal_composition_counts": dict(q_counts),
               "zero_K_units": [r["unit_id"] for r in unit_reports if r["conservative_legal_K"] == 0],
               "zero_K_cameras": [r["source_camera"] for r in by_camera if r["conservative_legal_K"] == 0],
               "diagnostic_coefficients": {"pool": "all150, no cross-camera train roles frozen; recompute after role freeze",
                                           "K": len(legal), "C_nonempty_cameras": len(coefficient_cameras),
                                           "base": "b_i=1/(V_c*n_v), physical-video aliases merged within camera",
                                           "static": "Only manually confirmed pure-static N can be discounted; S=confirmed-static base mass, D=all other undiscounted base mass including unknown, factor=min(1,D/S); D=0 keeps1. Unknown does not assert motion. Static mass<=50% applies only within D>0 cameras after adjustment, not globally.",
                                           "motion_metadata_present": False, "motion_state_all": "unknown",
                                           "static_discount_applied": False,
                                           "normalization": "w_i=(K/C)*a_i/sum_camera(a); mean1, each camera total K/C",
                                           "interpretation": "loss coefficients, no assertion of equal actual gradients or independent samples"},
               "traversal": {"scope": "diagnostic全池, no train roles frozen", "seed": seed,
                             "each_legal_unique_physical_window_once": True, "budget_window_caps": None,
                             "one_and_twenty_traversal_scenarios": update_scenarios(len(legal)),
                             "cache_complete_subset_scenarios": update_scenarios(len(cached_legal)),
                             "numerical_recipe_only": "FP32; freeze BN running statistics, train BN affine; effective batch24, final tail kept, sum(wL)/24",
                             "GPU_timing_available": False, "estimated_hours": None, "estimated_cost": None},
               "actual_PTS_overlap": {"window_count": sum(u["actual_PTS_overlapping_window_count"] for u in unit_reports),
                                      "max_seconds": max((u["max_actual_PTS_overlap_seconds"] for u in unit_reports), default=0)},
               "source_frame_reuse": {"sampled_exposures": sum(u["sampled_frame_exposures"] for u in unit_reports),
                                      "unique_frames": sum(u["unique_sampled_source_frames"] for u in unit_reports),
                                      "reuse_exposures": sum(u["sampled_source_frame_reuse_exposures"] for u in unit_reports)},
               "specific_scope_gap_count": len(gap_rows),
               "remaining_gates": ["cross-camera roles and training-only coefficients not frozen", "training card unsigned/unexecuted",
                                   "frozen guard technical/semantic reason review incomplete", "missing cache positions not decoded",
                                   "bounded working labels and prior timebase evidence do not establish new full-original-pixel mapping certification"],
               "elapsed_seconds": time.perf_counter() - start}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "windows.jsonl").open("x", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    write_csv(output / "per-unit.csv", unit_reports)
    write_csv(output / "per-camera.csv", by_camera)
    write_csv(output / "source-mapping.csv", mappings)
    write_csv(output / "diagnostic-loss-coefficients.csv", coefficients)
    write_csv(output / "cache-subset-diagnostic-loss-coefficients.csv", cached_coefficients)
    write_csv(output / "specific-scope-gaps.csv", gap_rows, ["unit_id", "source_camera", "grid_ordinal", "actual_start_seconds", "actual_end_seconds", "physical_window_key", "released_T_P_seconds", "semantic_complete", "reason"])
    write_json(output / "physical-duplicate-audit.json", {"duplicates": duplicate_rows, "conflicts": [], "cache_alias_rows": cache_alias_count})
    write_json(output / "diagnostic-coefficient-cameras.json", coefficient_cameras)
    write_json(output / "diagnostic-epoch-seed7.json", {"scope": "all150 diagnostic only, not frozen train roles", "seed": seed,
                                                      "training_ready": False, "physical_window_keys": epoch_order(legal, seed)})
    write_json(output / "input-references.json", {"current_run_hashed_inputs": inputs,
                                                "reused_coordinate_QC_media_fingerprints": "source-mapping.csv",
                                                "hash_scope": "No whole media or JPEG hash. Actual coordinate/QC rows read, prior verified SHA256 references reused."})
    write_json(output / "summary.json", summary)
    print(json.dumps({"status": "complete", "output": ref(output, root), "K": len(legal), "cache_complete_K": len(cached_legal),
                      "grid_count": len(all_rows), "seconds": round(time.perf_counter() - start, 2)}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--cache-index", default=DEFAULT_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    prepare(rooted(args.sources), rooted(args.release), rooted(args.cache_index), rooted(args.output), seed=args.seed, command=sys.argv)


if __name__ == "__main__":
    main()
