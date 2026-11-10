"""Metadata-only full-grid sampling and loss-coefficient diagnostics.

No media I/O, label release, role selection, or training lives in this module.
The existing nearest_window is the sole 16-real-frame matching rule.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import re

import numpy as np

from experiments.common.video.pts_clip_sampling import (
    CLIP_LEN, MAX_POINT_ERROR, SPAN_SECONDS, nearest_window,
)

VERSION = "full-coverage-grid-20261108-v1"
EPS = 1e-9


def physical_window_key(media_sha256, frame_indices):
    """Frozen middleware identity + ordered physical frame indices; no new media hash."""
    if not re.fullmatch(r"[0-9a-f]{64}", media_sha256):
        raise ValueError("Expected frozen lowercase middleware SHA256")
    indices = list(frame_indices)
    if (len(indices) != CLIP_LEN or any(type(i) is not int or i < 0 for i in indices)
            or any(b <= a for a, b in zip(indices, indices[1:]))):
        raise ValueError("Expected sixteen distinct ordered physical source-frame indices")
    identity = {"frozen_middleware_sha256": media_sha256, "ordered_frame_indices": indices}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def pts_runs(pts):
    """Split only at actual PTS backsteps, never at QC or semantic boundaries."""
    pts = np.asarray(pts, dtype=np.float64)
    if not len(pts) or not np.isfinite(pts).all():
        raise ValueError("Expected nonempty finite source PTS")
    cuts = [0] + (np.flatnonzero(np.diff(pts) < 0) + 1).tolist() + [len(pts)]
    return list(zip(cuts, cuts[1:]))


def full_grid(pts):
    """One first-PTS-anchored 3s grid; no label/QC phase reset or tail extra window.

    Enumerate complete nominal supports through maxPTS. More than one matching
    PTS run is an explicit ambiguity, rather than a hidden phase/run fallback.
    """
    pts = np.asarray(pts, dtype=np.float64)
    runs = pts_runs(pts)
    anchor, horizon = float(pts[0]), float(np.max(pts))
    count = max(0, math.floor((horizon - anchor - SPAN_SECONDS) / SPAN_SECONDS + EPS) + 1)
    for ordinal in range(count):
        target = anchor + SPAN_SECONDS * ordinal
        matches = []
        for a, b in runs:
            if pts[a] > target + MAX_POINT_ERROR or pts[b - 1] < target + SPAN_SECONDS - MAX_POINT_ERROR:
                continue
            indices = nearest_window(pts, a, b, target)
            if indices is not None:
                matches.append(indices)
        record = {"grid_ordinal": ordinal, "target_start_seconds": target,
                  "target_end_seconds": target + SPAN_SECONDS,
                  "grid_anchor_seconds": anchor, "grid_step_seconds": SPAN_SECONDS,
                  "frame_indices": None, "source_pts_seconds": None,
                  "sampling_reasons": []}
        if len(matches) != 1:
            record["sampling_reasons"] = ["no_valid_16_real_frame_PTS_match" if not matches else "ambiguous_PTS_run_match"]
        else:
            indices = matches[0]
            actual = pts[indices]
            record.update(frame_indices=list(map(int, indices)), source_pts_seconds=list(map(float, actual)),
                          actual_start_seconds=float(actual[0]), actual_end_seconds=float(actual[-1]),
                          actual_span_seconds=float(actual[-1] - actual[0]),
                          first_PTS_offset_seconds=float(actual[0] - target),
                          last_PTS_offset_seconds=float(actual[-1] - target - SPAN_SECONDS),
                          max_sampling_error_seconds=float(np.max(abs(actual - np.linspace(target, target + SPAN_SECONDS, CLIP_LEN)))))
        yield record


def validate_release(intervals):
    """Require explicit flags, ordered disjoint durations, and current state semantics."""
    result = sorted(intervals, key=lambda r: (float(r["start_sec"]), float(r["end_sec"])))
    previous = None
    for row in result:
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("Invalid released interval")
        if previous is not None and start < previous:
            raise ValueError("Overlapping released intervals: " + str(row.get("unit_id")))
        previous = end
        if row["released_state"] not in {"N", "T", "P", "U", "QC_or_semantic_pending"}:
            raise ValueError("Unsupported released state")
        if any(type(row[field]) is not bool for field in ("mask_three_class", "mask_existence")):
            raise ValueError("Release masks must be explicit booleans")
        if row["mask_three_class"] and row["released_state"] not in {"N", "T", "P"}:
            raise ValueError("Three-class mask has an unknown state")
        if row["mask_existence"] and row["existence_value"] not in (0, 1):
            raise ValueError("Existence mask requires a binary existence value")
        if row["mask_three_class"] and row["mask_existence"]:
            if int(row["released_state"] != "N") != row["existence_value"]:
                raise ValueError("State/existence release conflict")
    return result


def integrate_release(start, end, intervals):
    """Integrate on actual first/last PTS; unknown portions never renormalize q/r."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid actual window support")
    length = end - start
    state_parts = defaultdict(list)
    observed = {s: [] for s in ("N", "T", "P")}
    existence, covered, class_unknown, existence_unknown = [], [], [], []
    masked_states, overlap_ids = Counter(), []
    for row in intervals:
        duration = min(end, float(row["end_sec"])) - max(start, float(row["start_sec"]))
        if duration <= 0:
            continue
        state = row["released_state"]
        state_parts[state].append(duration)
        covered.append(duration)
        overlap_ids.append(row.get("release_id"))
        if row["mask_three_class"]:
            observed[state].append(duration)
        else:
            class_unknown.append(duration)
            masked_states[state] += duration
        if row["mask_existence"]:
            existence.append(duration * row["existence_value"])
        else:
            existence_unknown.append(duration)
    gap = max(0.0, length - math.fsum(covered))
    class_u = math.fsum(class_unknown) + gap
    exist_u = math.fsum(existence_unknown) + gap
    class_complete, exist_complete = class_u <= EPS, exist_u <= EPS
    observed_fraction = [math.fsum(observed[s]) / length for s in ("N", "T", "P")]
    r_min = math.fsum(existence) / length
    reasons = []
    if gap > EPS:
        reasons.append("implicit_unannotated_gap")
    for state in sorted(masked_states):
        reasons.append("mask_three_class_false:" + state)
    return {"integration_support": "[actual_first_PTS,actual_last_PTS)", "duration_seconds": length,
            "state_seconds": {s: math.fsum(v) for s, v in state_parts.items()},
            "implicit_gap_seconds": gap, "three_class_unknown_seconds": class_u,
            "existence_unknown_seconds": exist_u,
            "observed_N_T_P_fraction": observed_fraction,
            "q_N_T_P": observed_fraction if class_complete else None,
            "r_existence": r_min if exist_complete else None,
            "r_existence_bounds": [r_min, min(1.0, r_min + exist_u / length)],
            "semantic_three_class_complete": class_complete, "semantic_existence_complete": exist_complete,
            "release_ids": overlap_ids, "semantic_reasons": reasons,
            "masked_released_state_seconds": dict(masked_states)}


def deduplicate_physical(rows):
    """Exact physical duplicates collapse; conflicting labels or cameras fail closed."""
    unique, duplicates, seen = [], [], {}
    fields = ("source_camera", "composition", "legacy_guard_pass", "gray_QC_pass", "PTS_pass")
    for row in rows:
        key = row.get("physical_window_key")
        if key is None:
            unique.append(row)
            continue
        if key not in seen:
            seen[key] = row
            unique.append(row)
            continue
        original = seen[key]
        # Release ids and interval boundaries can differ across aliases; compare
        # actual integrated semantic truth, never row identity or prose.
        def semantic(r):
            c = r.get("composition", {})
            return {k: c.get(k) for k in ("q_N_T_P", "r_existence", "r_existence_bounds",
                                         "observed_N_T_P_fraction", "semantic_three_class_complete",
                                         "semantic_existence_complete", "masked_released_state_seconds")}
        if semantic(row) != semantic(original) or any(row.get(f) != original.get(f) for f in fields if f != "composition"):
            raise ValueError("Conflicting physical-window alias: " + key)
        duplicates.append({"physical_window_key": key, "retained_unit_id": original["unit_id"],
                           "duplicate_unit_id": row["unit_id"], "duplicate_grid_ordinal": row.get("grid_ordinal")})
    return unique, duplicates


def diagnostic_loss_coefficients(rows):
    """Video-equal base then camera-equal loss coefficients; no gradient claim.

    motion_state may only be confirmed_static_N, confirmed_nonstatic, unknown.
    Unknown is never treated as static or as confirmed motion. It remains
    undiscounted and contributes to the complementary base mass D.
    """
    result = []
    cameras = defaultdict(list)
    physical = set()
    for row in rows:
        if row["physical_window_key"] in physical:
            raise ValueError("Duplicate physical window in coefficient pool")
        physical.add(row["physical_window_key"])
        cameras[row["source_camera"]].append(row)
    k, c = len(rows), len(cameras)
    camera_report = []
    for camera, members in sorted(cameras.items()):
        videos = Counter(r["physical_video_id"] for r in members)
        v = len(videos)
        bases = [1.0 / (v * videos[r["physical_video_id"]]) for r in members]
        states = [r.get("motion_state", "unknown") for r in members]
        if any(s not in {"confirmed_static_N", "confirmed_nonstatic", "unknown"} for s in states):
            raise ValueError("Unsupported motion evidence state")
        for r, s in zip(members, states):
            if s == "confirmed_static_N" and r.get("composition", {}).get("q_N_T_P") != [1.0, 0.0, 0.0]:
                raise ValueError("Static-N evidence requires complete pure N")
        static = math.fsum(b for b, s in zip(bases, states) if s == "confirmed_static_N")
        other = math.fsum(b for b, s in zip(bases, states) if s != "confirmed_static_N")
        factor = min(1.0, other / static) if static > 0 and other > 0 else 1.0
        adjusted = [b * (factor if s == "confirmed_static_N" else 1.0) for b, s in zip(bases, states)]
        total = math.fsum(adjusted)
        for row, base, a, state in zip(members, bases, adjusted, states):
            result.append({"physical_window_key": row["physical_window_key"], "unit_id": row["unit_id"],
                           "source_camera": camera, "physical_video_id": row["physical_video_id"],
                           "V_c": v, "n_v": videos[row["physical_video_id"]], "motion_state": state,
                           "base_coefficient_b": base, "static_N_factor": factor if state == "confirmed_static_N" else 1.0,
                           "adjusted_coefficient_a": a, "loss_coefficient_w": (k / c) * a / total})
        camera_report.append({"source_camera": camera, "K_camera": len(members), "V_c": v,
                              "S_confirmed_static_N_base_mass": static,
                              "D_other_not_discounted_base_mass": other,
                              "unknown_motion_windows": states.count("unknown"),
                              "static_N_factor": factor, "loss_coefficient_sum": k / c})
    return result, camera_report


def epoch_order(rows, seed=7):
    """One traversal, seeded shuffle; callers must freeze training roles first."""
    keys = [r["physical_window_key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("An epoch must visit each physical window once")
    random.Random(seed).shuffle(keys)
    return keys


def update_scenarios(k, effective_batch=24):
    if type(k) is not int or k < 0 or type(effective_batch) is not int or effective_batch < 1:
        raise ValueError("Expected nonnegative K and positive effective batch")
    updates = (k + effective_batch - 1) // effective_batch
    return {"K": k, "effective_batch_windows": effective_batch,
            "one_traversal_updates": updates, "twenty_traversal_scenario_updates": updates * 20,
            "twenty_is_scenario_not_default_epochs": True,
            "tail_windows": k % effective_batch,
            "tail_policy": "keep_without_drop_or_padding; fixed denominator24, sum(w_i*L_i)/24",
            "loss_denominator": effective_batch}
