"""Prepare offline synthetic-video inventories and blocked task candidates.

Reads metadata only. Does not contact a provider, inspect credentials, decode
media, generate samples, modify annotations, or select a training camera split.
An explicit --out directory is required before any JSON is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.data_usage_contract import PROTOCOL_ID, canonical_camera_identity

DEFAULT_POOL = "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
DEFAULT_ANNOTATIONS = "data/metadata/annotation-revisions/npt-20261107-proposals-v3/sources.json"
HISTORICAL_COVERAGE = "data/metadata/scene-coverage/20261107-v1.json"
CURRENT_PROTOCOL = "docs/data/跨场景数据使用规范.md"
CURRENT_LABEL_SPEC = "docs/data/NPT标签定义与量化核查.md"
EVENT_FIELDS = ["video_id", "video_key", "start_sec", "end_sec", "confidence", "kind", "note"]
LABELS = {"neg": "N", "transition": "T", "pos": "P", "cut": "cut"}
CONDITIONS = {
    "no_target": ("无目标背景", "N", "Show an empty or quiet channel without a visible debris-flow surge, flood wave, or landslide."),
    "ordinary_water": ("普通水流", "N", "Show ordinary water flowing through the existing channel. Substantial and large-scale water motion is allowed; do not equate a negative condition with a static or tiny stream. Avoid a debris-laden surge or coarse-debris front."),
    "flood": ("洪水", "N", "Show a watery flood through the existing channel, allowing strong large-scale water motion and changing water extent, without a debris-dominant surge or coarse-debris front. The requested condition still requires content-based label review."),
    "debris_flow_control": ("泥石流正控制", "P", "Show a clearly visible debris-laden surge moving through the existing channel, with a coherent moving sediment-and-debris front."),
}
BLOCKED = "blocked_unfrozen_split"


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def path_ref(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def fingerprint(path: Path, root: Path) -> dict:
    return {"path": path_ref(path, root), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def empty_stats() -> dict:
    return {"interval_seconds": {label: 0.0 for label in ("N", "T", "P", "cut", "unknown")},
            "interval_counts": {label: 0 for label in ("N", "T", "P", "cut", "unknown")}}


def read_pool(path: Path) -> tuple[dict, list[dict]]:
    included, excluded = {}, []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            uid = (row.get("unit_id") or "").strip()
            group = (row.get("group_id") or "").strip()
            parts = group.split("|")
            source = (row.get("source") or "").strip().lower()
            if len(parts) != 3 or parts[0] != "d" or not all(parts[1:]) or source not in {"dongchuan", "video4", "video5"}:
                excluded.append({"unit_id": uid, "group_id": group, "reason": "outside_dongchuan_development_pool"})
                continue
            if not uid or uid in included:
                raise ValueError("Missing or duplicate development unit_id: " + uid)
            included[uid] = {"unit_id": uid, "source": source, "source_camera": parts[1], "group_id": group,
                             "camera_key": canonical_camera_identity({"source_camera": parts[1]}),
                             "source_video_id": row.get("video_id") or uid,
                             "media_candidate_path": row.get("middleware_path") or None,
                             "declared_media_sha256": row.get("middleware_sha256") or None,
                             "training_ready": False, "intervals": [], **empty_stats()}
    return included, excluded


def read_saved_annotations(manifest_path: Path, units: dict, root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    inputs = [fingerprint(manifest_path, root)]
    ignored = {"non_real_rows": 0, "outside_pool_rows": 0}
    unit_sources = {}
    for key, entry in sorted(manifest["sources"].items()):
        csv_value = Path(str(entry["annotations_csv"]).replace("\\", "/"))
        csv_path = (csv_value if csv_value.is_absolute() else
                    resolve(root, csv_value) if csv_value.parts and csv_value.parts[0] == "data" else manifest_path.parent / csv_value)
        source_root = resolve(root, entry["source_id"])
        source = {"source_id": entry["source_id"], "annotations_csv": path_ref(csv_path, root),
                  "annotations_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  "frames_index_candidate_path": path_ref(source_root / "frames_index.csv", root)}
        inputs.append({"path": source["annotations_csv"], "sha256": source["annotations_sha256"]})
        mapping = source_root / "mapping.csv"
        if mapping.is_file():
            inputs.append(fingerprint(mapping, root))
            with mapping.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    uid = (row.get("video_key") or "").strip()
                    if uid in units and (row.get("excluded") or "").strip() != "1":
                        if uid in unit_sources and unit_sources[uid]["source_id"] != source["source_id"]:
                            raise ValueError("Ambiguous source for development unit: " + uid)
                        unit_sources[uid] = source
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            if header[:7] != EVENT_FIELDS:
                raise ValueError("Expected consolidated event fields in the first seven columns: " + str(csv_path))
            for line_number, fields in enumerate(reader, 2):
                if not fields:
                    continue
                row = dict(zip(EVENT_FIELDS, fields[:7]))
                if (row.get("confidence") or "").strip() != "real":
                    ignored["non_real_rows"] += 1
                    continue
                uid = (row.get("video_key") or "").strip()
                if uid not in units:
                    ignored["outside_pool_rows"] += 1
                    continue
                if uid in unit_sources and unit_sources[uid]["source_id"] != source["source_id"]:
                    raise ValueError("Annotations occur in multiple sources for unit: " + uid)
                unit_sources[uid] = source
                try:
                    start, end = float(row["start_sec"]), float(row["end_sec"])
                except (KeyError, ValueError):
                    raise ValueError(f"Invalid real interval: {csv_path}:{line_number}")
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError(f"Invalid real interval: {csv_path}:{line_number}")
                kind = (row.get("kind") or "").strip().lower()
                label = LABELS.get(kind, "unknown")
                unit = units[uid]
                unit["intervals"].append({"start_sec": start, "end_sec": end, "kind": kind,
                                          "label": label, "annotation_line": line_number,
                                          "original_first_seven_fields": row})
                unit["interval_seconds"][label] += end - start
                unit["interval_counts"][label] += 1
    for uid, unit in units.items():
        unit["intervals"].sort(key=lambda interval: interval["start_sec"])
        for first, second in zip(unit["intervals"], unit["intervals"][1:]):
            if first["end_sec"] > second["start_sec"]:
                raise ValueError("Overlapping real intervals for unit: " + uid)
        unit["annotation_source"] = unit_sources.get(uid)
        unit["interval_seconds"] = {label: round(seconds, 9) for label, seconds in unit["interval_seconds"].items()}
        unit["transition_status"] = "legacy_T_semantics_pending_current_label_spec_not_automatically_N_or_P"
    return inputs, {"ignored_rows": ignored, "saved_revision_training_ready": manifest.get("training_ready", False),
                    "units_missing_annotation_source": sorted(set(units) - set(unit_sources))}


def summarize(units: list[dict], **identity) -> dict:
    stats = empty_stats()
    videos_with = {label: 0 for label in stats["interval_seconds"]}
    for unit in units:
        for label in stats["interval_seconds"]:
            stats["interval_seconds"][label] += unit["interval_seconds"][label]
            stats["interval_counts"][label] += unit["interval_counts"][label]
            videos_with[label] += int(unit["interval_counts"][label] > 0)
    stats["interval_seconds"] = {label: round(seconds, 9) for label, seconds in stats["interval_seconds"].items()}
    return {**identity, "video_count": len(units), "source_units": sorted(unit["unit_id"] for unit in units),
            "source_groups": sorted({unit["group_id"] for unit in units}), "videos_with_explicit_intervals": videos_with,
            "training_ready": False, **stats}


def references_for(units: list[dict], preferred_label: str) -> tuple[str, bool, list[dict]]:
    preferred = [unit for unit in units if unit["interval_counts"][preferred_label] > 0]
    needs_edit = not preferred
    references = []
    for unit in preferred or units:
        source = unit["annotation_source"] or {}
        references.append({"source_unit": unit["unit_id"], "source_video_id": unit["source_video_id"],
                           "source_group": unit["group_id"], "source_camera": unit["source_camera"],
                           "media_candidate_path": unit["media_candidate_path"],
                           "frames_index_candidate_path": source.get("frames_index_candidate_path"),
                           "annotations_csv": source.get("annotations_csv"),
                           "candidate_intervals": [{"start_sec": interval["start_sec"], "end_sec": interval["end_sec"],
                                                    "label": interval["label"]} for interval in unit["intervals"]
                                                   if not needs_edit and interval["label"] == preferred_label],
                           "candidate_only": True, "reference_accepted": False,
                           "clean_negative_reference_verified": False,
                           "reference_edit_required": needs_edit,
                           "fallback_scope": "scene_background_only_not_a_clean_N_frame" if needs_edit else None})
    scope = "reference_edit_required" if needs_edit else "explicit_" + preferred_label + "_interval_candidates_first"
    return scope, needs_edit, references


def submission_controls() -> dict:
    return {"matched_N_P_reference_processing_history_required": True,
            "reference_processing_match_status": "unreviewed",
            "shared_reference_processing_history_id": None,
            "raw_P_vs_edited_N_pair_may_submit": False,
            "training_ratio_preregistration_required": True,
            "per_source_quota_preregistration_required": True,
            "training_ratio_preregistered": False,
            "per_source_quota_preregistered": False,
            "holdout_may_select_prompts_models_or_samples": False,
            "legacy_T_can_be_relabelled_as_N_or_P_automatically": False,
            "new_T_requires_current_label_contract": True,
            "pilot_is_class_balanced": False,
            "pilot_samples_are_independent_scenes": False}


def effective_support_requirements() -> dict:
    return {"candidate_total_duration_sec": 5, "minimum_contiguous_effective_support_sec": 3,
            "minimum_distinct_frames": 16, "strictly_increasing_real_pts_required": True,
            "higher_registered_model_requirements_take_precedence": True,
            "first_and_last_effective_frames_must_be_qc_checked": True,
            "fixed_edge_trim_sec": None, "blind_edge_trimming_allowed": False,
            "repeated_frame_padding_allowed": False, "looping_or_splicing_allowed": False,
            "effective_support_qc_status": "not_generated_not_checked",
            "accepted_effective_duration_sec": None}


def build_tasks(cameras: list[dict], units: dict) -> list[dict]:
    tasks = []
    for camera in cameras:
        name = camera["source_camera"]
        camera_units = [units[uid] for uid in camera["source_units"]]
        camera_hash = hashlib.sha256(("d|" + name).encode("utf-8")).hexdigest()[:12]
        for condition, (chinese, desired_label, condition_prompt) in CONDITIONS.items():
            scope, needs_edit, references = references_for(camera_units, desired_label)
            for variant in (1, 2):
                variation = f"camera-{camera_hash}/{condition}/variant-{variant:02d}"
                tasks.append({"job_id": f"syn-{camera_hash}-{condition}-v{variant:02d}",
                              "protocol_id": PROTOCOL_ID, "protocol_path": CURRENT_PROTOCOL,
                              "label_spec_path": CURRENT_LABEL_SPEC,
                              "camera_key": canonical_camera_identity({"source_camera": name}),
                              "source_camera": name, "source_groups": camera["source_groups"], "source_units": camera["source_units"],
                              "condition": condition, "condition_name_zh": chinese, "variant": variant,
                              "duration_sec": 5, "variation_key": variation, "variation_key_is_provider_seed": False,
                              "effective_support_requirements": effective_support_requirements(),
                              "prompt_en": ("Create one continuous 5-second realistic surveillance-camera shot. "
                                            "Keep the reference camera fixed: preserve viewpoint, channel geometry, terrain and background; "
                                            "no pan, tilt, zoom, camera shake, viewpoint change, or scene cuts. " + condition_prompt +
                                            f" Visual variation {variant}: vary only natural motion timing and surface detail, not camera identity."),
                              "reference_candidate_scope": scope, "reference_edit_required": needs_edit,
                              "reference_candidates": references, "selected_reference": None,
                              "desired_training_label": desired_label, "accepted_label": None,
                              "parent_lineage": {"candidate_parent_camera": name, "candidate_parent_event_groups": camera["source_groups"],
                                                 "candidate_parent_units": camera["source_units"], "selected_reference": None,
                                                 "all_descendants_inherit_parent_camera_split": True},
                              "synthetic": True, "independent_scene_increment": 0, "split": "train_candidate_only",
                              "quality_status": "not_generated", "training_ready": False,
                              "submit_ready": False, "status": BLOCKED, "estimated_cost": None,
                              "submission_controls": submission_controls(),
                              "provider_preference": {"service": "Runway", "integration": "ordinary_media_plugin",
                                                      "model": None, "read_model_and_cost_when_connected": True,
                                                      "api_price_assumed": False}})
    return tasks


def workflow_definition() -> dict:
    return {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_path": CURRENT_PROTOCOL,
            "label_spec_path": CURRENT_LABEL_SPEC,
            "status": BLOCKED, "submit_ready": False, "training_ready": False,
            "initial_state": "pending_split", "camera_split_frozen": False,
            "split_policy": {"isolation_unit": "registered_camera_fixed_view", "inner_selection_same_camera_isolation": True,
                             "all_dates_sources_and_descendants_same_side": True, "fixed_fold_count": None,
                             "historical_three_protocols_are_not_current_options": True},
            "effective_support_requirements": effective_support_requirements(),
            "budget": {"authorized": False, "amount": None, "currency": None, "authorization_reference": None},
            "first_batch_scale_only": {"training_camera_count": 2, "tasks_per_camera": 8, "task_count": 16,
                                       "N_intent_count": 12, "P_intent_count": 4,
                                       "class_balanced": False, "independent_new_scene_count": 0,
                                       "selected_cameras": [], "selection_status": "not_selected_pending_frozen_camera_split"},
            "submission_controls": submission_controls(),
            "training_sampling_preregistration": {"status": "not_registered", "required": True,
                                                 "real_to_synthetic_ratio": None, "synthetic_N_to_P_ratio": None,
                                                 "per_source_quota": None},
            "states": {"pending_split": "待划分", "pending_reference": "待参考", "pending_cost_authorization": "待费用授权",
                       "submitted": "已提交", "generated_pending_qc": "生成成功待QC", "rejected": "拒收",
                       "accepted_candidate": "验收候选", "failed_stopped": "失败停止"},
            "transitions": [
                {"from": "pending_split", "to": "pending_reference", "requires": ["frozen_camera_split", "parent_camera_on_training_side"]},
                {"from": "pending_reference", "to": "pending_cost_authorization", "requires": ["reference_review_accepted", "negative_reference_edit_accepted_if_required", "matched_N_P_reference_processing_history_reviewed"]},
                {"from": "pending_cost_authorization", "to": "submitted", "requires": ["explicit_budget_authorization", "connected_plugin_model_and_cost_checked", "training_ratios_and_per_source_quotas_preregistered", "holdout_selection_firewall_checked"]},
                {"from": "submitted", "to": "generated_pending_qc", "requires": ["provider_completed"]},
                {"from": "submitted", "to": "failed_stopped", "requires": ["provider_failed_or_timed_out"]},
                {"from": "generated_pending_qc", "to": "rejected", "requires": ["quality_or_condition_review_failed"]},
                {"from": "generated_pending_qc", "to": "accepted_candidate", "requires": ["quality_review_passed", "actual_contiguous_support_at_least_model_requirement_and_3_seconds", "at_least_16_distinct_increasing_real_pts_frames", "first_and_last_effective_frames_checked_without_blind_trim", "label_reviewed_separately_from_prompt_intent"]},
                {"from": "failed_stopped", "to": "pending_cost_authorization", "requires": ["at_most_one_documented_reasoned_revision", "renewed_budget_authorization"]}],
            "retry_policy": {"infinite_retries": False, "automatic_retry": False, "max_reasoned_revision_retries": 1,
                             "stop_after_second_failure": True, "retry_requires_cost_authorization": True},
            "constraints": {"negative_conditions": ["no_target", "ordinary_water", "flood"],
                            "same_generator_P_control_required": "debris_flow_control",
                            "same_generator_and_model_across_negative_and_P_conditions": True,
                            "matched_reference_processing_history_across_N_and_P_required": True,
                            "raw_P_and_edited_N_are_not_a_submission_ready_control_pair": True,
                            "training_ratios_and_per_source_quotas_must_be_preregistered": True,
                            "held_out_data_not_used_for_prompt_model_or_sample_selection": True,
                            "real_validation_and_test_contain_no_synthetic": True,
                            "all_reference_descendants_remain_in_parent_camera_split": True,
                            "external_test_sources_excluded": True,
                            "synthetic_is_not_a_new_independent_scene": True,
                            "only_frozen_training_cameras_may_proceed": True,
                            "unannotated_gaps_never_imputed_as_N": True,
                            "notes_are_not_label_evidence": True,
                            "T_remains_candidate_not_training_truth": True,
                            "legacy_T_not_automatically_N_or_P": True,
                            "new_T_definition_only_from_current_label_spec": True,
                            "large_water_or_flood_motion_does_not_itself_disqualify_N_intent": True,
                            "five_second_total_length_is_not_effective_support_acceptance": True,
                            "inference_eligibility_uses_technical_QC_not_semantic_exclusion": True,
                            "legacy_cut_reason_unresolved_not_assumed_all_bad_or_all_normal": True,
                            "prompt_intent_is_not_accepted_label": True,
                            "no_reference_upload_or_generation_in_preparation": True}}


def prepare(pool_path: Path, annotations_path: Path, out: Path, repo_root: Path = ROOT) -> dict:
    units, excluded = read_pool(pool_path)
    inputs, annotation_audit = read_saved_annotations(annotations_path, units, repo_root)
    inputs.insert(0, fingerprint(pool_path, repo_root))
    cameras, groups = [], []
    for camera in sorted({unit["source_camera"] for unit in units.values()}):
        selected = [unit for unit in units.values() if unit["source_camera"] == camera]
        cameras.append(summarize(selected, source_camera=camera,
                                 camera_key=canonical_camera_identity({"source_camera": camera})))
    for group in sorted({unit["group_id"] for unit in units.values()}):
        selected = [unit for unit in units.values() if unit["group_id"] == group]
        groups.append(summarize(selected, group_id=group, source_camera=selected[0]["source_camera"]))
    inventory = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_path": CURRENT_PROTOCOL,
                 "label_spec_path": CURRENT_LABEL_SPEC,
                 "scope": "Dongchuan development pool; explicit confidence=real saved-v3 intervals only",
                 "training_ready": False, "inputs": inputs,
                 "historical_coverage_reference_only": HISTORICAL_COVERAGE,
                 "historical_coverage_used_for_current_statistics": False,
                 "totals": summarize(list(units.values())), "cameras": cameras, "event_groups": groups,
                 "units": [units[uid] for uid in sorted(units)], "excluded_pool_rows": excluded,
                 "cameras_missing_explicit_N": [camera["source_camera"] for camera in cameras if not camera["interval_counts"]["N"]],
                 "annotation_audit": annotation_audit,
                 "statistics_caveat": "Explicit interval seconds before reference review or model QC; not training exposure. Gaps are uncounted, cut/unknown are separate, notes never determine labels. T and all task sources remain candidates."}
    tasks = build_tasks(cameras, units)
    task_document = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": BLOCKED, "submit_ready": False, "training_ready": False,
                     "task_count": len(tasks), "tasks_per_camera": 8, "duration_sec": 5, "tasks": tasks}
    workflow = workflow_definition()
    out.mkdir(parents=True, exist_ok=True)
    for filename, document in (("scene-inventory.json", inventory), ("task-candidates.json", task_document), ("workflow.json", workflow)):
        (out / filename).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"scene_inventory": inventory, "task_candidates": task_document, "workflow": workflow}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", required=True, help="Explicit output directory; only three JSON files are written here")
    args = parser.parse_args()
    prepared = prepare(resolve(args.repo_root, args.pool), resolve(args.repo_root, args.annotations),
                       resolve(args.repo_root, args.out), args.repo_root)
    print(json.dumps({"cameras": len(prepared["scene_inventory"]["cameras"]),
                      "development_units": prepared["scene_inventory"]["totals"]["video_count"],
                      "event_groups": len(prepared["scene_inventory"]["event_groups"]),
                      "task_candidates": prepared["task_candidates"]["task_count"], "status": BLOCKED}, ensure_ascii=False))


if __name__ == "__main__":
    main()
