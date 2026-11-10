"""Audit existing real-PTS candidate windows against saved N/P evidence only.

This metadata-only tool intentionally imports the existing pool/annotation
reader. It neither decodes media nor changes labels, grids, splits or training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.video.window_composition import (  # noqa: E402
    CONTRACT_VERSION, Interval, compose_window, dominant_class, normalize_intervals, soft_target,
)
from experiments.tools.prepare_synthetic_video_workflow import (  # noqa: E402
    DEFAULT_ANNOTATIONS, DEFAULT_POOL, fingerprint, path_ref, read_pool, read_saved_annotations, resolve,
)

DEFAULT_WINDOWS = "data/metadata/EXP-017/role-free-plan-v3/windows.jsonl"
COUNTS = ("N", "P", "MIX", "U", "technical_invalid")
EXAMPLES = ("v1_16", "v1_63", "v1_95")
VISUAL_LOG = ".codex/logs/evidence/2026-11-07-v116-annotation-calibration.json"
CURRENT_VISUAL_LOG = ".codex/logs/evidence/2026-11-08-visual-and-storage-evidence.json"


def contract() -> dict:
    return {
        "version": CONTRACT_VERSION, "status": "mathematical_candidate_no_training",
        "task": "Identify an already visible target through whole-window time composition.",
        "base_evidence": "Only explicitly confirmed N and P; legacy persistent weak-state T maps to pending U without rewriting CSV.",
        "new_T": "Whole-window positive-duration composition anchor at r=a; neither physical flow type, intensity, uncertainty, last-frame state nor future risk.",
        "window": "[s,e), L=e-s>0 in a shared real-time coordinate; frame count is not time duration.",
        "ratios": {"r_min": "dP/L", "r_max": "(dP+dU)/L",
                   "dU": "explicit U + legacy T + implicit unannotated time gaps"},
        "target_gate": "Unique mathematical target only for complete explicit N/P coverage and no technical veto. Actual training also requires verified annotation-to-original-pixel time mapping and annotation authorization.",
        "three_anchor_formula": {"a_default": .5, "a_domain": "0<a<1",
                                 "r<=a": ["1-r/a", "r/a", 0],
                                 "r>a": [0, "(1-r)/(1-a)", "(r-a)/(1-a)"],
                                 "order": ["N", "T", "P"], "sum": 1,
                                 "decode": "r_hat=pP+a*pT"},
        "display_rule": "Composition display is argmax g(r_hat) for a model prediction, or argmax g(r) for a mathematical target. Floating-point ties within 1e-12 favor T, then N, then P; pure endpoints remain N and P. At a=.5 this gives composition thresholds .25/.75, inclusive T at both ties.",
        "probability_argmax_rule": "dominant_class(p) is a separate direct probability argmax; it need not equal composition_class(p)=argmax g(pP+a*pT). For p=(.4,.2,.4), r_hat=.5 and composition_class is T, whereas direct probability argmax ties N/P and returns N.",
        "prediction_probability_validation": {"sum_absolute_tolerance": 1e-6, "values": "finite and in [0,1]",
                                              "renormalized": False, "decoded_endpoint_roundoff_clamped": True,
                                              "target_algebra_tolerance_relaxed": False},
        "technical_rule": "Any positive-duration forbidden-time intersection invalidates the complete window; never cut and concatenate. CLI also checks the continuous source-frame support, including the sampled end frame, against frozen forbidden frame ranges, gray_std<20 and PTS backsteps. Selected 16 PTS must be strictly increasing; equal intermediate unsampled source PTS are allowed, as in the existing sampler.",
        "untyped_cut_rule": "Legacy cut is technical_invalid with reason untyped_cut until classified; it is not silently imputed as N or geological U.",
        "invalid_output": "Technical-invalid windows have no ratio ground truth, ratio bounds or target. U windows only have bounds, never a unique target.",
        "examples": {"15_equal_duration_N_1_P": list(soft_target(1 / 16)),
                     "15_equal_duration_N_1_legacy_T": "U: bounds [0,1/16], target null; legacy T supplies neither confirmed P nor half-P.",
                     "nonuniform_frame_sampling": "Fifteen N frames and one P frame alone do not identify the confirmed positive duration."},
        "audit_scope": "Existing cached candidate windows or existing restricted dense-grid windows only; not a new continuous-time grid or all possible windows. Old predictions/FP are never new labels.",
        "sensitivity": "a=0.4,0.5,0.6 only changes anchor encoding; reconstructs the same r. No training or performance claim.",
        "mapping_status": "mapping_unverified", "training_ready": False,
    }


def _read_source_rows(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["unit_id"]: row for row in csv.DictReader(stream)}


def _source_coordinates(row: dict, root: Path, inputs: list) -> dict:
    """Check frozen coordinate/QC evidence, never infer original-pixel mapping."""
    loaded = []
    for field in ("coord", "qc"):
        path = resolve(root, row[field + "_path"])
        identity = fingerprint(path, root)
        if identity["sha256"] != row[field + "_sha256"]:
            raise ValueError("Frozen " + field + " hash mismatch for " + row["unit_id"])
        inputs.append(identity)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            loaded.append(list(csv.DictReader(stream)))
    coord, qc = loaded
    n = int(row["n_frames"])
    if len(coord) != n or len(qc) != n:
        raise ValueError("Coordinate/QC frame count mismatch for " + row["unit_id"])
    times, bad_qc, bad_order = [], [0], [0]
    for index, (frame, quality) in enumerate(zip(coord, qc)):
        if int(frame["frame"]) != index or int(quality["frame"]) != index:
            raise ValueError("Noncontiguous coordinate/QC frame identities")
        timestamp, qt, std = float(frame["pts_time"]), float(quality["pts_time"]), float(quality["gray_std"])
        if not all(math.isfinite(value) for value in (timestamp, qt, std)) or abs(timestamp - qt) > .001001:
            raise ValueError("Invalid coordinate/QC PTS")
        bad_qc.append(bad_qc[-1] + int(std < 20))
        bad_order.append(bad_order[-1] + int(bool(times) and timestamp < times[-1]))
        times.append(timestamp)
    forbidden = json.loads(row["forbidden_ranges_json"])
    for left, right in forbidden:
        if not isinstance(left, int) or not isinstance(right, int) or not 0 <= left < right <= n:
            raise ValueError("Invalid forbidden source-frame range")
    return {"times": times, "forbidden": forbidden, "bad_qc": bad_qc, "bad_order": bad_order}


def _window_support(row: dict, coordinates: dict) -> tuple[float, float, list[str]]:
    indices, times = row.get("frame_indices"), row.get("source_pts_seconds")
    if not isinstance(indices, list) or not isinstance(times, list) or len(indices) != 16 or len(times) != 16:
        raise ValueError("Expected 16 explicit frame_indices and source_pts_seconds; annotation sampling bounds are not window bounds")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("Frame indices must be integers")
    if any(index < 0 or index >= len(coordinates["times"]) for index in indices) or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Invalid or unordered window indices")
    times = [float(value) for value in times]
    if any(not math.isfinite(value) for value in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Window source PTS must be finite and strictly increasing")
    if any(abs(value - coordinates["times"][index]) > 1e-8 for index, value in zip(indices, times)):
        raise ValueError("Candidate source PTS disagrees with frozen source coordinates")
    if not 2.9 - 1e-9 <= times[-1] - times[0] <= 3.1 + 1e-9:
        raise ValueError("Expected existing approximately 3-second model support")
    first, last = indices[0], indices[-1]
    reasons = []
    if any(first < right and last >= left for left, right in coordinates["forbidden"]):
        reasons.append("source_technical_forbidden_frame_range")
    if coordinates["bad_qc"][last + 1] - coordinates["bad_qc"][first]:
        reasons.append("source_gray_std_below_20")
    # At frame i the prefix flags the edge i-1 -> i; exclude the edge before first.
    if coordinates["bad_order"][last + 1] - coordinates["bad_order"][first + 1]:
        reasons.append("source_pts_backstep")
    return times[0], times[-1], reasons


def _summary(rows: list[dict]) -> dict:
    counts = Counter(row["composition"]["status"] for row in rows)
    complete = counts["N"] + counts["P"] + counts["MIX"]
    return {"count": len(rows), "counts": {key: counts[key] for key in COUNTS},
            "confirmed_composition_candidates": complete,
            "genuinely_mixed_count": counts["MIX"],
            "genuinely_mixed_ratio_all_cached_windows": counts["MIX"] / len(rows) if rows else None,
            "genuinely_mixed_ratio_confirmed_composition": counts["MIX"] / complete if complete else None,
            "mapping_unverified_count": len(rows), "training_ready_count": 0,
            "interpretation": "Provisional metadata composition; mapping and supervision not released."}


def _sensitivity(rows: list[dict]) -> dict:
    confirmed = [row["composition"] for row in rows if row["composition"]["target"] is not None]
    result = {"scope": "Same confirmed-composition metadata windows; a does not resolve U or mapping.",
              "candidate_count": len(confirmed), "no_training": True, "anchors": []}
    for anchor in (.4, .5, .6):
        targets = [soft_target(row["r"], anchor) for row in confirmed]
        counts = Counter(dominant_class(target) for target in targets)
        differences = [math.fsum(abs(a - b) for a, b in zip(target, row["target"]))
                       for row, target in zip(confirmed, targets)]
        result["anchors"].append({"a": anchor, "dominant_class_counts": {label: counts[label] for label in ("N", "T", "P")},
                                  "mean_soft_target_N_T_P": [math.fsum(target[index] for target in targets) / len(targets) for index in range(3)] if targets else None,
                                  "mean_L1_change_from_a_0_5": math.fsum(differences) / len(differences) if differences else None,
                                  "changed_target_count_from_a_0_5": sum(value > 1e-12 for value in differences),
                                  "changed_dominant_class_count_from_a_0_5": sum(dominant_class(target) != row["dominant_class"] for row, target in zip(confirmed, targets))})
    return result


def audit(pool_path: Path, annotations_path: Path, windows_path: Path, out: Path,
          *, repo_root: Path = ROOT, command: list[str] | None = None,
          replace_unreviewed: bool = False) -> dict:
    units, excluded = read_pool(pool_path)
    inputs, annotation_audit = read_saved_annotations(annotations_path, units, repo_root)
    inputs[:0] = [fingerprint(pool_path, repo_root), fingerprint(windows_path, repo_root)]
    for filename in ("experiments/common/video/window_composition.py", "experiments/tools/audit_window_composition.py",
                     "experiments/tools/prepare_synthetic_video_workflow.py", "experiments/tests/test_window_composition.py"):
        source_path = repo_root / filename
        if source_path.is_file():
            inputs.append(fingerprint(source_path, repo_root))
    source_rows = _read_source_rows(pool_path)
    normalized = {}
    for uid, unit in units.items():
        normalized[uid] = normalize_intervals(Interval(row["start_sec"], row["end_sec"],
                                                     row["kind"] if row["label"] != "unknown" else "U")
                                               for row in unit["intervals"])
    raw_windows, observed_keys = [], set()
    with windows_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not {"unit_id", "frame_indices", "source_pts_seconds"} <= row.keys():
                    raise ValueError(f"Window actual-PTS schema missing at line {line_number}")
                if row["unit_id"] not in units:
                    raise ValueError("Window outside declared development pool")
                raw_windows.append((line_number, row))
                observed_keys.update(row)
    grouped = {uid: [] for uid in units}
    for line_number, row in raw_windows:
        grouped[row["unit_id"]].append((line_number, row))
    audited, seen, duplicate_count = [], {}, 0
    spans = []
    for uid in sorted(units):
        coordinates = _source_coordinates(source_rows[uid], repo_root, inputs)
        for line_number, row in grouped[uid]:
            start, end, reasons = _window_support(row, coordinates)
            physical_key = (uid, start, end)
            if physical_key in seen:
                duplicate_count += 1
                continue
            seen[physical_key] = True
            composition = compose_window(start, end, normalized[uid], forbidden_ranges=[(start, end)] if reasons else [])
            if reasons:
                composition["reasons"] = sorted(set(composition["reasons"] + reasons))
            audited.append({"unit_id": uid, "source_camera": units[uid]["source_camera"],
                            "input_line": line_number, "clip_id": row.get("clip_id"),
                            "frame_indices": row["frame_indices"], "source_pts_seconds": row["source_pts_seconds"],
                            "composition": composition, "mapping_status": "mapping_unverified",
                            "supervision_target": None, "training_ready": False})
            spans.append(end - start)
    by_unit, by_camera = [], []
    for uid, unit in sorted(units.items()):
        selected = [row for row in audited if row["unit_id"] == uid]
        by_unit.append({"unit_id": uid, "source_camera": unit["source_camera"], **_summary(selected),
                        "legacy_T_to_U_seconds": unit["interval_seconds"]["T"],
                        "legacy_T_status": "pending_no_CSV_rewrite",
                        "saved_N_P_contacts": [left.end for left, right in zip(normalized[uid], normalized[uid][1:])
                                               if left.end == right.start and {left.kind, right.kind} == {"N", "P"}]})
    for camera in sorted({unit["source_camera"] for unit in units.values()}):
        selected_units = [unit for unit in units.values() if unit["source_camera"] == camera]
        by_camera.append({"source_camera": camera, **_summary([row for row in audited if row["source_camera"] == camera]),
                          "unit_count": len(selected_units),
                          "legacy_T_to_U_seconds": round(math.fsum(unit["interval_seconds"]["T"] for unit in selected_units), 9),
                          "legacy_T_status": "pending_no_CSV_rewrite"})
    example_summaries = [row for row in by_unit if row["unit_id"] in EXAMPLES]
    boundary = {"contract_version": CONTRACT_VERSION, "status": "existence_based_interpretation_candidate",
                "CSV_modified": False, "audit_uses_default_saved_v3_legacy_T_as_U": True,
                "v1_16": {"last_clear_negative_observation_sec": 1033.92,
                           "first_visible_target_observation_sec": 1036.04,
                           "onset_bracket_sec": [1033.92, 1036.04], "exact_onset_sec": None,
                           "observation_source": "Current task's visually reviewed sparse-frame observations; existence-based reinterpretation, not old saved annotation truth.",
                           "historical_visual_context": VISUAL_LOG,
                           "current_visual_evidence": CURRENT_VISUAL_LOG,
                           "historical_context_is_new_interpretation": False,
                           "mapping_status": "mapping_unverified",
                           "restriction": "Do not interpolate an exact N/P contact, retain 1038 as a fixed contact, or replace legacy T in this audit."},
                "retained_examples": example_summaries,
                "scope_restriction": "Example counts concern only the specified cached candidate support; zero MIX is not evidence that a full video has no mixed windows."}
    visual_path = repo_root / VISUAL_LOG
    if visual_path.is_file():
        inputs.append(fingerprint(visual_path, repo_root))
    current_visual_path = repo_root / CURRENT_VISUAL_LOG
    boundary["v1_16"]["current_visual_evidence_file_present"] = current_visual_path.is_file()
    if current_visual_path.is_file():
        inputs.append(fingerprint(current_visual_path, repo_root))
    witnesses = [row for row in audited if row["composition"]["status"] not in {"N", "P"} or row["unit_id"] in EXAMPLES]
    input_kind = ("existing_QC_restricted_dense_grid" if "target_start_seconds" in observed_keys and "clip_id" not in observed_keys
                  else "existing_supervision_candidate_bank")
    summary = {"contract_version": CONTRACT_VERSION, "status": "provisional_mapping_unverified",
               "input_kind": input_kind,
               "scope": "Existing approximately-3s cached supports only; not all original-video time, not all possible windows, not epoch exposures.",
               "selection_caveat": "Supervision-bank selection used original N/P cores; existing dense-grid selection used historical cut/QC masks. These inherited restrictions are not removed. Neither denominator establishes full-original-video coverage or absence of mixed windows.",
               "pool_unit_count": len(units), "camera_count": len(by_camera),
               "window_input_schema": {"observed_keys": sorted(observed_keys),
                                       "time_source": "source_pts_seconds[0] and source_pts_seconds[-1]",
                                       "excluded_as_window_time": ["sampling_start_sec", "sampling_end_sec", "target_start_seconds"],
                                       "half_open_support": True, "sample_count": 16,
                                       "dedup_key": ["unit_id", "actual_first_PTS", "actual_last_PTS"],
                                       "raw_window_rows": len(raw_windows), "duplicate_physical_support_rows": duplicate_count,
                                       "actual_span_seconds_min": min(spans) if spans else None,
                                       "actual_span_seconds_max": max(spans) if spans else None},
               "totals": _summary(audited), "by_camera": by_camera, "by_unit": by_unit,
               "legacy_T_to_U": {"seconds": round(math.fsum(unit["interval_seconds"]["T"] for unit in units.values()), 9),
                                  "unit_count": sum(unit["interval_seconds"]["T"] > 0 for unit in units.values()),
                                  "cameras": [row["source_camera"] for row in by_camera if row["legacy_T_to_U_seconds"] > 0],
                                  "status": "pending_no_CSV_rewrite"},
               "retained_examples": example_summaries, "anchor_sensitivity": _sensitivity(audited),
               "annotation_audit": annotation_audit, "excluded_pool_rows": excluded,
               "witnesses": {"path": "window-witnesses.jsonl", "count": len(witnesses),
                             "rule": "All MIX, U, technical_invalid windows plus all cached windows of the three retained examples; pure endpoints of other units remain reproducible from hashed inputs."},
               "inputs": inputs, "command": command,
               "training_ready": False, "training_started": False,
               "audit_notes": [{"status": "corrected_before_acceptance",
                                "issue": "Initial implementation incorrectly vetoed equal intermediate unsampled source PTS as nonincreasing time.",
                                "correction": "Selected 16 PTS remain strictly increasing. Intermediate source timestamps only veto actual backsteps (<), not equality, matching the existing sampler's nondecreasing clean runs.",
                                "discarded_preacceptance_veto_counts": {"supervision_bank": 890, "restricted_dense_grid": 9304},
                                "discarded_counts_are_not_final_results": True,
                                "old_NP_bank_dense_grid_annotations_modified": False}],
               "notes": ["All mathematically complete compositions remain metadata candidates because original-pixel/annotation time closure was not verified here.",
                         "Saved confidence=real is necessary evidence but saved revision training_ready=false remains in force.",
                         "Any old model score, FP or prediction is excluded from annotation evidence."]}
    documents = {"contract.json": contract(), "coverage-summary.json": summary, "boundary-evidence.json": boundary}
    # Do not overwrite an already published historical audit directory.
    if out.exists() and any(out.iterdir()) and not replace_unreviewed:
        raise FileExistsError("Audit output directory must be empty: " + str(out))
    if replace_unreviewed:
        previous = json.loads((out / "coverage-summary.json").read_text(encoding="utf-8"))
        if previous.get("status") != "provisional_mapping_unverified" or previous.get("training_ready") is not False:
            raise ValueError("Only an explicitly authorized unreviewed provisional output may be replaced")
    out.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (out / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "window-witnesses.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in witnesses:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replace-unreviewed", action="store_true",
                        help="Replace this task's explicitly authorized provisional output before acceptance")
    args = parser.parse_args()
    summary = audit(resolve(ROOT, args.pool), resolve(ROOT, args.annotations), resolve(ROOT, args.windows),
                    resolve(ROOT, args.out), command=[sys.executable, "-m", "experiments.tools.audit_window_composition", *sys.argv[1:]],
                    replace_unreviewed=args.replace_unreviewed)
    print(json.dumps({"out": path_ref(resolve(ROOT, args.out), ROOT), "totals": summary["totals"],
                      "legacy_T_to_U": summary["legacy_T_to_U"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
