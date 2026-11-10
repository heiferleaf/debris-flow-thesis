"""Metadata coverage of two inherited candidate grids under a semantic release.

No decoding, media hashing, frame-table scan, QC rerun, sampling, or training.
The first/last explicit source PTS define the half-open window support. This
reports the old grids, whose original semantic cut/core restrictions remain.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
DEFAULT_BANK = ROOT / "data/metadata/EXP-017/role-free-plan-v3/windows.jsonl"
DEFAULT_DENSE = ROOT / "data/process/EXP-017/dense-pixel-reference-v1/dense-windows.jsonl"
STATES = ("N", "T", "P", "A", "U", "pending_cut", "other_unknown")
COUNT_FIELDS = (
    "candidate_windows", "duplicate_support_rows", "complete_N", "complete_T",
    "complete_P", "complete_MIX", "complete_three_class", "incomplete_three_class",
    "pending_cut_windows", "explicit_U_windows", "unannotated_gap_windows",
    "ineligible_NTP_windows", "candidate_or_pending_release_windows",
    "A_existence_only_windows", "complete_existence_windows",
    "partial_existence_windows", "no_existence_supervision_windows",
    "overlap_N_windows", "overlap_T_windows", "overlap_P_windows",
    "eligible_T_overlap_windows", "eligible_P_overlap_windows",
    "complete_with_T_windows", "complete_with_P_windows",
    "max_nonoverlap_complete_total", "max_nonoverlap_complete_N",
    "max_nonoverlap_complete_T", "max_nonoverlap_complete_P", "max_nonoverlap_complete_MIX",
    "total_optimal_schedule_N", "total_optimal_schedule_T", "total_optimal_schedule_P", "total_optimal_schedule_MIX",
)
CAVEAT = (
    "Only inherited candidate-grid coverage: bank was selected from original N/P "
    "cores; dense was restricted by historical cut/QC eligibility. Zero T/MIX "
    "candidates does not mean the original video has no T or mixed windows. "
    "This audit neither enumerates all possible new windows nor completes "
    "training/input readiness or original-media time alignment."
)


def ref(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_fingerprinted_json(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), {
        "path": ref(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "hash_scope": "this JSON only; no media bytes read",
    }


def boolean(row: dict, key: str) -> bool:
    if type(row.get(key)) is not bool:
        raise ValueError(f"Release row needs explicit boolean {key}: {row.get('release_id')}")
    return row[key]


def is_pending(row: dict) -> bool:
    status = str(row.get("release_status", "")).lower()
    return any(word in status for word in ("candidate", "pending", "unresolved", "unreviewed"))


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    state: str
    three: bool
    existence: bool
    existence_value: int | None
    pending_release: bool
    release_id: str


def release_interval(row: dict) -> Interval:
    start, end = float(row["start_sec"]), float(row["end_sec"])
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError(f"Invalid released interval: {row.get('release_id')}")
    state = row["released_state"]
    # A historical cut remains pending, not a newly asserted technical fault.
    if state not in {"N", "T", "P", "A", "U"}:
        state = "pending_cut" if row.get("original_kind") == "cut" or state in {
            "cut", "QC_or_semantic_pending", "pending_cut"
        } else "other_unknown"
    eligible = boolean(row, "eligible_for_label_supervision")
    mask_three, mask_existence = boolean(row, "mask_three_class"), boolean(row, "mask_existence")
    pending = is_pending(row)
    three = state in {"N", "T", "P"} and eligible and mask_three and not pending
    # Existence is a separate explicit mask. A may supply only this mask even
    # when the overall three-state label is not eligible. No inferred A target.
    value = row.get("existence_value")
    existence = state in {"N", "T", "P", "A"} and mask_existence and not pending
    if state != "A":
        existence = existence and eligible
    if existence:
        expected = 0 if state == "N" else 1
        if type(value) not in {int, float} or value != expected:
            raise ValueError(f"Existence value contradicts state/mask: {row.get('release_id')}")
        value = expected
    else:
        value = None
    return Interval(start, end, state, three, existence, value, pending,
                    str(row.get("release_id", row.get("current_id", "unidentified"))))


class Timeline:
    def __init__(self, intervals: list[Interval]):
        self.intervals = sorted(intervals, key=lambda x: (x.start, x.end))
        # Fail on overlap instead of double counting duration or choosing a
        # state by precedence. Adjacent half-open intervals are valid.
        for left, right in zip(self.intervals, self.intervals[1:]):
            if right.start < left.end:
                raise ValueError(f"Overlapping released intervals: {left.release_id}, {right.release_id}")
        self.ends = [item.end for item in self.intervals]

    def intersect(self, start: float, end: float) -> list[tuple[Interval, float, float]]:
        index = bisect_right(self.ends, start)
        result = []
        while index < len(self.intervals) and self.intervals[index].start < end:
            item = self.intervals[index]
            a, b = max(start, item.start), min(end, item.end)
            if a < b:
                result.append((item, a, b))
            index += 1
        return result


def compose(start: float, end: float, timeline: Timeline) -> dict:
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("Window needs finite increasing first/last actual PTS")
    length = end - start
    tolerance = max(1e-9, length * 1e-9)
    state_seconds = {state: 0.0 for state in STATES}
    eligible_seconds = {state: 0.0 for state in ("N", "T", "P")}
    existence_seconds = [0.0, 0.0]
    existence_A_seconds = 0.0
    candidate_seconds = 0.0
    witnesses = []
    for item, a, b in timeline.intersect(start, end):
        duration = b - a
        state_seconds[item.state] += duration
        if item.three:
            eligible_seconds[item.state] += duration
        if item.existence:
            existence_seconds[item.existence_value] += duration
            if item.state == "A":
                existence_A_seconds += duration
        if item.pending_release:
            candidate_seconds += duration
        witnesses.append({"release_id": item.release_id, "state": item.state,
                          "intersection_sec": [a, b], "mask_three_effective": item.three,
                          "mask_existence_effective": item.existence})
    covered = math.fsum(state_seconds.values())
    if covered > length + tolerance:
        raise ValueError("Release intersections double counted time")
    gap = max(0.0, length - covered)
    three_known = math.fsum(eligible_seconds.values())
    existence_known = math.fsum(existence_seconds)
    complete_three = abs(three_known - length) <= tolerance
    complete_existence = abs(existence_known - length) <= tolerance
    q = [eligible_seconds[x] / length for x in ("N", "T", "P")] if complete_three else None
    nonzero = [x for x in ("N", "T", "P") if eligible_seconds[x] > tolerance]
    status = (nonzero[0] if len(nonzero) == 1 else "MIX") if complete_three else "incomplete"
    return {
        "support_sec": [start, end], "duration_sec": length, "status": status,
        "complete_three_class": complete_three, "q_N_T_P": q,
        "r_T_plus_P": q[1] + q[2] if q is not None else None,
        "released_state_seconds": state_seconds,
        "eligible_N_T_P_seconds": eligible_seconds,
        "unannotated_seconds": gap,
        "ineligible_NTP_seconds": math.fsum(state_seconds[x] for x in ("N", "T", "P")) - three_known,
        "candidate_or_pending_release_seconds": candidate_seconds,
        "existence_known_negative_seconds": existence_seconds[0],
        "existence_known_positive_seconds": existence_seconds[1],
        "existence_A_seconds": existence_A_seconds,
        "existence_known_fraction": existence_known / length,
        "complete_existence": complete_existence,
        "existence_target_if_complete": existence_seconds[1] / length if complete_existence else None,
        # These are fractions of the whole support, never of the known subset.
        "existence_positive_fraction_bounds": [existence_seconds[1] / length,
            (existence_seconds[1] + max(0.0, length - existence_known)) / length],
        "release_intersections": witnesses,
    }


def load_release(path: Path, pool_ids: set[str]) -> tuple[dict[str, Timeline], dict, list[str]]:
    document, identity = read_fingerprinted_json(path)
    if document.get("schema_version") != 1 or not isinstance(document.get("rows"), list):
        raise ValueError("Expected released-intervals.json schema_version=1 with rows")
    grouped = defaultdict(list)
    outside = []
    for row in document["rows"]:
        uid = row["unit_id"]
        if uid not in pool_ids:
            outside.append(uid)
            continue
        grouped[uid].append(release_interval(row))
    return {uid: Timeline(items) for uid, items in grouped.items()}, identity, sorted(set(outside))


def read_sources(path: Path) -> tuple[dict[str, dict], dict]:
    raw = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sources = {}
        for row in csv.DictReader(stream):
            uid = row["unit_id"]
            if uid in sources:
                raise ValueError(f"Duplicate source unit: {uid}")
            parts = row.get("group_id", "").split("|")
            camera = row.get("source_camera") or (parts[1] if len(parts) >= 3 else "UNREGISTERED")
            sources[uid] = {**row, "source_camera": camera}
    return sources, {"path": ref(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                     "use": "unit/camera/media-reference provenance only; old video labels are ignored"}


def coordinate_endpoints(source: dict) -> dict:
    """Read two existing coordinate records, not the whole frame table/hash."""
    path = resolve(source["coord_path"])
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8-sig").strip()
        first = stream.readline().decode("utf-8-sig").strip()
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 8192))
        tail = stream.read().splitlines()
    last = next(line.decode("utf-8-sig").strip() for line in reversed(tail) if line.strip())
    first_row = next(csv.DictReader([header, first]))
    last_row = next(csv.DictReader([header, last]))
    a, b = float(first_row["pts_time"]), float(last_row["pts_time"])
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        raise ValueError(f"Invalid coordinate endpoint span: {path}")
    return {"coord_path": ref(path), "middleware_first_PTS_sec": a, "middleware_last_PTS_sec": b,
            "middleware_PTS_span_seconds": b - a,
            "coord_first_frame_record": int(first_row["frame"]), "coord_last_frame_record": int(last_row["frame"])}


def window_support(row: dict) -> tuple[float, float]:
    pts, indices = row.get("source_pts_seconds"), row.get("frame_indices")
    if not isinstance(pts, list) or not isinstance(indices, list) or len(pts) != 16 or len(indices) != 16:
        raise ValueError("Expected explicit 16 source_pts_seconds and frame_indices")
    times = [float(value) for value in pts]
    if any(not math.isfinite(x) for x in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Candidate PTS must be finite and strictly increasing")
    if any(type(x) is not int or x < 0 for x in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Candidate frame indices must be nonnegative and increasing")
    return times[0], times[-1]


def bump(counts: Counter, result: dict) -> None:
    counts["candidate_windows"] += 1
    status = result["status"]
    counts["complete_" + status if status != "incomplete" else "incomplete_three_class"] += 1
    counts["complete_three_class"] += int(result["complete_three_class"])
    state, eligible = result["released_state_seconds"], result["eligible_N_T_P_seconds"]
    for label in ("N", "T", "P"):
        counts[f"overlap_{label}_windows"] += int(state[label] > 0)
    for label in ("T", "P"):
        counts[f"eligible_{label}_overlap_windows"] += int(eligible[label] > 0)
        counts[f"complete_with_{label}_windows"] += int(result["complete_three_class"] and eligible[label] > 0)
    for field, flag in {
        "pending_cut_windows": state["pending_cut"] > 0,
        "explicit_U_windows": state["U"] > 0,
        "unannotated_gap_windows": result["unannotated_seconds"] > 1e-9,
        "ineligible_NTP_windows": result["ineligible_NTP_seconds"] > 1e-9,
        "candidate_or_pending_release_windows": result["candidate_or_pending_release_seconds"] > 0,
        "A_existence_only_windows": result["existence_A_seconds"] > 0 and not result["complete_three_class"],
        "complete_existence_windows": result["complete_existence"],
        "partial_existence_windows": 1e-9 < result["existence_known_fraction"] < 1 - 1e-9,
        "no_existence_supervision_windows": result["existence_known_fraction"] <= 1e-9,
    }.items():
        counts[field] += int(flag)


def released_durations(timeline: Timeline) -> dict:
    totals = Counter()
    for item in timeline.intervals:
        duration = item.end - item.start
        totals[f"released_{item.state}_seconds"] += duration
        if item.three:
            totals[f"eligible_{item.state}_seconds"] += duration
        if item.existence:
            totals["existence_known_seconds"] += duration
        if item.pending_release:
            totals["candidate_or_pending_release_seconds"] += duration
    return {**{f"released_{x}_seconds": totals[f"released_{x}_seconds"] for x in STATES},
            **{f"eligible_{x}_seconds": totals[f"eligible_{x}_seconds"] for x in ("N", "T", "P")},
            "existence_known_seconds": totals["existence_known_seconds"],
            "candidate_or_pending_release_seconds": totals["candidate_or_pending_release_seconds"]}


def clean_counts(counts: Counter) -> dict:
    return {key: counts[key] for key in COUNT_FIELDS}


def merged_supports(supports: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(supports):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def nonoverlap_capacity(samples: list[tuple[float, float, str]]) -> tuple[dict, list[dict]]:
    """Maximum cardinality by earliest finish, separately per video clock."""
    def greedy(items):
        selected, last_end = [], -math.inf
        for start, end, state in sorted(items, key=lambda x: (x[1], x[0], x[2])):
            if start >= last_end:  # Half-open adjacency is valid, no duration fudge.
                selected.append((start, end, state))
                last_end = end
        return selected
    all_selected = greedy(samples)
    selected_counts = Counter(state for _, _, state in all_selected)
    result = {"max_nonoverlap_complete_total": len(all_selected),
              "max_nonoverlap_complete_selected_seconds": math.fsum(b - a for a, b, _ in all_selected),
              "complete_candidate_union_support_seconds": math.fsum(b - a for a, b in merged_supports([(a, b) for a, b, _ in samples]))}
    for state in ("N", "T", "P", "MIX"):
        selected = greedy([item for item in samples if item[2] == state])
        result[f"max_nonoverlap_complete_{state}"] = len(selected)
        result[f"max_nonoverlap_complete_{state}_seconds"] = math.fsum(b - a for a, b, _ in selected)
        result[f"total_optimal_schedule_{state}"] = selected_counts[state]
    return result, [{"support_sec": [a, b], "status": state} for a, b, state in all_selected[:3]]


def union_coverage(supports: list[tuple[float, float]], timeline: Timeline,
                   published: dict) -> dict:
    """Unique grid-supported time, without summing overlapping window spans."""
    merged = merged_supports(supports)
    coverage = Counter()
    coverage["candidate_union_support_seconds"] = math.fsum(b - a for a, b in merged)
    for start, end in merged:
        for item, a, b in timeline.intersect(start, end):
            coverage[f"released_{item.state}_grid_covered_seconds"] += b - a
            if item.three:
                coverage[f"eligible_{item.state}_grid_covered_seconds"] += b - a
    result = {"candidate_union_support_seconds": coverage["candidate_union_support_seconds"]}
    for prefix, states in (("released", STATES), ("eligible", ("N", "T", "P"))):
        for state in states:
            key = f"{prefix}_{state}_grid_covered_seconds"
            result[key] = coverage[key]
            result[f"{prefix}_{state}_grid_uncovered_seconds"] = max(
                0.0, published[f"{prefix}_{state}_seconds"] - coverage[key])
    return result


LENGTH_BINS = (("under_5min", 0, 300), ("5_to_15min", 300, 900),
               ("15_to_30min", 900, 1800), ("30_to_45min", 1800, 2700),
               ("45min_or_more", 2700, math.inf))


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    def quantile(fraction):
        if not ordered:
            return None
        position = (len(ordered) - 1) * fraction
        left, right = math.floor(position), math.ceil(position)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)
    return {"units": len(ordered), "min_seconds": ordered[0] if ordered else None,
            "p25_seconds": quantile(.25), "median_seconds": quantile(.5),
            "p75_seconds": quantile(.75), "max_seconds": ordered[-1] if ordered else None,
            "bins": {name: sum(low <= value < high for value in values) for name, low, high in LENGTH_BINS}}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def audit(release: Path, out: Path, sources_path: Path = DEFAULT_SOURCES,
          bank_path: Path = DEFAULT_BANK, dense_path: Path = DEFAULT_DENSE,
          max_total_windows: int = 160_000) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output is not empty; use a new directory, never overwrite old coverage counts")
    sources, source_identity = read_sources(sources_path)
    endpoints = {uid: coordinate_endpoints(source) for uid, source in sources.items()}
    release_file = release / "released-intervals.json" if release.is_dir() else release
    timelines, release_identity, outside_release = load_release(release_file, set(sources))
    empty = Timeline([])
    durations = {uid: released_durations(timelines.get(uid, empty)) for uid in sources}
    per_unit, per_camera, grid_summaries, examples, identities = [], [], {}, {}, []
    total_raw = 0
    for grid, candidate_path in (("bank", bank_path), ("dense", dense_path)):
        counts = {uid: Counter() for uid in sources}
        supports = defaultdict(list)
        complete_supports = defaultdict(list)
        seen = set()
        example_groups = defaultdict(list)
        digest, raw_count, min_span, max_span = hashlib.sha256(), 0, math.inf, -math.inf
        with candidate_path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                raw_count += 1
                total_raw += 1
                if total_raw > max_total_windows:
                    raise ValueError("Bounded metadata row budget exceeded")
                row = json.loads(raw.decode("utf-8-sig"))
                uid = row["unit_id"]
                if uid not in sources:
                    raise ValueError(f"Candidate outside declared source pool: {uid}")
                try:
                    start, end = window_support(row)
                except ValueError as exc:
                    raise ValueError(f"{grid} line {line_number}: {exc}") from exc
                key = (uid, start, end)
                if key in seen:
                    counts[uid]["duplicate_support_rows"] += 1
                    continue
                seen.add(key)
                supports[uid].append((start, end))
                span = end - start
                min_span, max_span = min(min_span, span), max(max_span, span)
                result = compose(start, end, timelines.get(uid, empty))
                bump(counts[uid], result)
                if result["complete_three_class"]:
                    complete_supports[uid].append((start, end, result["status"]))
                tags = [result["status"]]
                if result["released_state_seconds"]["pending_cut"] > 0:
                    tags.append("pending_cut")
                if result["candidate_or_pending_release_seconds"] > 0:
                    tags.append("candidate_release")
                if result["released_state_seconds"]["A"] > 0:
                    tags.append("A_existence_only")
                for tag in tags:
                    if len(example_groups[tag]) < 3:
                        example_groups[tag].append({"unit_id": uid, "source_camera": sources[uid]["source_camera"],
                            "input_line": line_number, "clip_id": row.get("clip_id"),
                            "frame_indices": row["frame_indices"], "source_pts_seconds": row["source_pts_seconds"],
                            "composition": result})
        identity = {"path": ref(candidate_path), "bytes": candidate_path.stat().st_size,
                    "sha256": digest.hexdigest(), "raw_rows": raw_count,
                    "hash_scope": "JSONL metadata read in the single auditing pass"}
        identities.append(identity)
        capacity_by_unit = {}
        capacity_witnesses = {}
        for uid in sources:
            capacity, witnesses = nonoverlap_capacity(complete_supports[uid])
            capacity_by_unit[uid] = capacity
            capacity_witnesses[uid] = witnesses
            counts[uid].update({key: capacity[key] for key in COUNT_FIELDS if key in capacity})
        total = sum(counts.values(), Counter())
        missing = [uid for uid in sorted(sources) if not counts[uid]["candidate_windows"]]
        gaps = []
        coverage_by_unit = {}
        for uid, source in sorted(sources.items()):
            c, duration = counts[uid], durations[uid]
            coverage = union_coverage(supports[uid], timelines.get(uid, empty), duration)
            coverage_by_unit[uid] = coverage
            published_rows = timelines.get(uid, empty).intervals
            annotation_extent = (published_rows[-1].end - published_rows[0].start) if published_rows else None
            candidate_extent = max(b for _, b in supports[uid]) - min(a for a, _ in supports[uid]) if supports[uid] else None
            span = endpoints[uid]["middleware_PTS_span_seconds"]
            row = {"grid": grid, "unit_id": uid, "source_camera": source["source_camera"],
                   "release_rows_present": uid in timelines, "missing_from_candidate_grid": uid in missing,
                   **clean_counts(c), **duration, **coverage, **capacity_by_unit[uid], **endpoints[uid],
                   "annotation_extent_seconds": annotation_extent, "candidate_extent_seconds": candidate_extent,
                   "candidate_union_fraction_of_middleware_span": coverage["candidate_union_support_seconds"] / span,
                   "complete_union_fraction_of_middleware_span": capacity_by_unit[uid]["complete_candidate_union_support_seconds"] / span,
                   "max_nonoverlap_selected_seconds_fraction_of_middleware_span": capacity_by_unit[uid]["max_nonoverlap_complete_selected_seconds"] / span}
            for label in ("T", "P"):
                row[f"released_{label}_but_zero_overlap"] = duration[f"released_{label}_seconds"] > 0 and not c[f"overlap_{label}_windows"]
                row[f"eligible_{label}_but_zero_complete_support"] = duration[f"eligible_{label}_seconds"] > 0 and not c[f"complete_with_{label}_windows"]
                if row[f"released_{label}_but_zero_overlap"] or row[f"eligible_{label}_but_zero_complete_support"]:
                    gaps.append({"unit_id": uid, "source_camera": source["source_camera"], "state": label,
                                 "released_seconds": duration[f"released_{label}_seconds"],
                                 "eligible_seconds": duration[f"eligible_{label}_seconds"],
                                 "overlap_windows": c[f"overlap_{label}_windows"],
                                 "complete_with_state_windows": c[f"complete_with_{label}_windows"],
                                 "eligible_grid_covered_seconds": coverage[f"eligible_{label}_grid_covered_seconds"],
                                 "eligible_grid_uncovered_seconds": coverage[f"eligible_{label}_grid_uncovered_seconds"]})
            per_unit.append(row)
        for camera in sorted({row["source_camera"] for row in sources.values()}):
            uids = [uid for uid, row in sources.items() if row["source_camera"] == camera]
            camera_counts = sum((counts[uid] for uid in uids), Counter())
            camera_durations = {key: math.fsum(durations[uid][key] for uid in uids) for key in next(iter(durations.values()))}
            camera_coverage = {key: math.fsum(coverage_by_unit[uid][key] for uid in uids)
                               for key in next(iter(coverage_by_unit.values()))}
            camera_capacity_seconds = {key: math.fsum(capacity_by_unit[uid][key] for uid in uids)
                                       for key in next(iter(capacity_by_unit.values())) if key not in COUNT_FIELDS}
            camera_span = math.fsum(endpoints[uid]["middleware_PTS_span_seconds"] for uid in uids)
            per_camera.append({"grid": grid, "source_camera": camera, "pool_units": len(uids),
                               "missing_candidate_units": sum(uid in missing for uid in uids),
                               **clean_counts(camera_counts), **camera_durations, **camera_coverage, **camera_capacity_seconds,
                               "middleware_PTS_span_seconds_sum": camera_span,
                               "max_nonoverlap_selected_seconds_fraction_of_middleware_span": camera_capacity_seconds["max_nonoverlap_complete_selected_seconds"] / camera_span})
        capacity_length_bins = []
        for name, low, high in LENGTH_BINS:
            uids = [uid for uid in sources if low <= endpoints[uid]["middleware_PTS_span_seconds"] < high]
            span_sum = math.fsum(endpoints[uid]["middleware_PTS_span_seconds"] for uid in uids)
            selected_seconds = math.fsum(capacity_by_unit[uid]["max_nonoverlap_complete_selected_seconds"] for uid in uids)
            capacity_length_bins.append({"middleware_span_bin": name, "pool_units": len(uids),
                "middleware_PTS_span_seconds_sum": span_sum,
                "complete_overlapping_candidates": sum(counts[uid]["complete_three_class"] for uid in uids),
                "max_nonoverlap_complete_total": sum(counts[uid]["max_nonoverlap_complete_total"] for uid in uids),
                "max_nonoverlap_complete_selected_seconds": selected_seconds,
                "selected_seconds_fraction_of_middleware_span": selected_seconds / span_sum if span_sum else None})
        grid_summaries[grid] = {**clean_counts(total), "raw_window_rows": raw_count,
            "unique_support_count": len(seen), "actual_span_seconds_min": min_span if seen else None,
            "actual_span_seconds_max": max_span if seen else None,
            "missing_candidate_units": missing, "released_state_candidate_gaps": gaps,
            "unique_time_coverage": {key: math.fsum(row[key] for row in coverage_by_unit.values())
                                     for key in next(iter(coverage_by_unit.values()))},
            "nonoverlap_capacity_seconds": {key: math.fsum(row[key] for row in capacity_by_unit.values())
                                             for key in next(iter(capacity_by_unit.values())) if key not in COUNT_FIELDS},
            "candidate_extent_distribution": distribution([max(b for _, b in supports[uid]) - min(a for a, _ in supports[uid]) for uid in sources if supports[uid]]),
            "capacity_by_middleware_span_bin": capacity_length_bins,
            "scope_caveat": CAVEAT}
        examples[grid] = {tag: example_groups[tag] for tag in ("N", "T", "P", "MIX", "incomplete", "pending_cut", "candidate_release", "A_existence_only")}
        examples[grid]["first_three_of_total_max_nonoverlap_schedule_per_unit"] = capacity_witnesses
    summary = {
        "schema_version": 1, "status": "inherited_grid_semantic_coverage_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_path": ref(release_file.parent), "release_input": release_identity,
        "pool_units": len(sources), "camera_count": len({r['source_camera'] for r in sources.values()}),
        "release_missing_pool_units": sorted(set(sources) - set(timelines)),
        "release_units_outside_pool_ignored": outside_release,
        "middleware_PTS_span_distribution": distribution([row["middleware_PTS_span_seconds"] for row in endpoints.values()]),
        "annotation_extent_distribution": distribution([timeline.intervals[-1].end - timeline.intervals[0].start for timeline in timelines.values() if timeline.intervals]),
        "candidate_inputs": identities, "sources_input": source_identity,
        "tool_input": {"path": ref(Path(__file__)), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "source_fingerprint_policy": {"reused_recorded_fields": ["middleware_sha256", "coord_sha256", "qc_sha256", "media_hash_scope"],
            "bound_by_sources_csv_sha256": source_identity['sha256'],
            "media_or_full_coordinate_tables_rehashed_or_scanned": False,
            "coordinate_endpoint_reads": len(endpoints),
            "coordinate_read_scope": "First and last existing coord CSV records only, with an 8KB tail seek; no full-frame scan or coordinate rehash.",
            "old_forbidden_QC_qualification": "Inherited in the grid; not rerun and not extended to new windows."},
        "support_contract": {"window": "[source_pts_seconds[0], source_pts_seconds[-1])",
            "label": "q=(eligible dN,dT,dP)/full actual span only when fully eligible; r=qT+qP",
            "A": "Only explicitly masked existence information; never supplies a complete three-class target.",
            "unknown": "U, pending cut, implicit gaps, candidate/pending releases and ineligible time are not N and are never removed by renormalization.",
            "pending_cut": "Historical cut is semantic/quality pending, not a newly certified technical fault.",
            "input_training_mask": "eligible_for_input_training is independent of semantic masks; no input/training readiness asserted.",
            "coverage_duration": "Union of inherited window supports, intersected with released intervals; overlapping windows never multiply covered time.",
            "nonoverlap_capacity": "Maximum cardinality interval scheduling by earliest end per unit; half-open endpoint adjacency is allowed; no four-window truncation. Camera/global totals sum independent video clocks.",
            "per_state_capacity": "N/T/P/MIX capacities are optimized separately and are not additive. total_optimal_schedule_* gives composition within the all-state optimum.",
            "length_basis": "Middleware first/last recorded source PTS span, annotation extent and candidate extent are separate quantities; none asserts original-container duration or timestamp origin.",
            "sufficiency": "Physical nonoverlap does not establish statistical independence, learning sufficiency or a new training quota.",
            "partial_existence": "Known positive/negative durations use the whole window denominator; bounds retain the unknown duration."},
        "grids": grid_summaries, "scope_caveat": CAVEAT,
        "training_ready": False, "training_runs": 0, "paid_jobs": 0,
        "video_decodes": 0, "media_hashes_computed": 0,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_csv(out / "per-unit.csv", per_unit)
    write_csv(out / "per-camera.csv", per_camera)
    write_json(out / "fewexamples.json", {"release_input_sha256": release_identity["sha256"],
        "scope_caveat": CAVEAT, "examples": examples})
    return summary


def self_test() -> dict:
    def interval(a, b, label, eligible=True):
        return release_interval({"release_id": f"fixture:{a}:{label}", "start_sec": a, "end_sec": b,
            "released_state": label, "release_status": "bounded_working_label",
            "eligible_for_label_supervision": eligible, "mask_three_class": eligible,
            "mask_existence": eligible, "existence_value": 0 if label == "N" else 1})
    # 1. A genuine crossing receives duration-based soft targets, not end-label
    # or frame-vote labels, and r is exactly T+P.
    result = compose(0, 3, Timeline([interval(0, 1, "N"), interval(1, 2, "T"), interval(2, 3, "P")]))
    assert result["status"] == "MIX" and result["q_N_T_P"] == [1/3, 1/3, 1/3]
    assert math.isclose(result["r_T_plus_P"], 2/3)
    # 2. Unknown time is neither N nor renormalized away.
    result = compose(0, 3, Timeline([interval(0, 2, "P"), interval(2, 3, "U", False)]))
    assert result["q_N_T_P"] is None and result["r_T_plus_P"] is None
    assert result["released_state_seconds"]["N"] == 0
    assert result["existence_positive_fraction_bounds"] == [2/3, 1.0]
    # 3. A nominal P row with masks false supplies no hard supervision.
    result = compose(0, 3, Timeline([interval(0, 3, "P", False)]))
    assert not result["complete_three_class"] and not result["complete_existence"]
    assert result["eligible_N_T_P_seconds"]["P"] == 0 and result["q_N_T_P"] is None
    # 4. Adjacent half-open boundaries contribute once; endpoint-only contact
    # with P contributes zero duration to the left N window.
    timeline = Timeline([interval(0, 1, "N"), interval(1, 2, "P")])
    left, right, both = compose(0, 1, timeline), compose(1, 2, timeline), compose(0, 2, timeline)
    assert left["q_N_T_P"] == [1, 0, 0] and right["q_N_T_P"] == [0, 0, 1]
    assert both["q_N_T_P"] == [.5, 0, .5] and sum(both["released_state_seconds"].values()) == 2
    capacity, _ = nonoverlap_capacity([(0, 5, "P"), (0, 2, "N"), (2, 4, "P"), (4, 6, "T")])
    assert capacity["max_nonoverlap_complete_total"] == 3
    return {"status": "passed", "fixture_cases": 4,
            "cases": ["cross_boundary_soft_targets", "U_never_N_or_renormalized", "masked_P_has_no_hard_target", "half_open_no_duplicate_boundary"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", help="Release directory containing released-intervals.json")
    parser.add_argument("--out", help="New empty output directory; never overwrite previous counts")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES))
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument("--dense", default=str(DEFAULT_DENSE))
    parser.add_argument("--max-total-windows", type=int, default=160_000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if not args.release or not args.out:
        parser.error("--release and --out are required unless --self-test")
    result = audit(resolve(args.release), resolve(args.out), resolve(args.sources),
                   resolve(args.bank), resolve(args.dense), args.max_total_windows)
    print(json.dumps({"status": result["status"], "release_sha256": result["release_input"]["sha256"],
                      "grids": {key: {field: value[field] for field in ("candidate_windows", "complete_N", "complete_T", "complete_P", "complete_MIX", "incomplete_three_class")} for key, value in result["grids"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
