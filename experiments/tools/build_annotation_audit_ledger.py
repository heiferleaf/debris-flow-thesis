"""Join current annotations and bounded visual evidence without editing labels.

Metadata only: no decoding, image hashing, model calls, credentials or network.
Interval identity is unit + half-open time intersection, never row ID alone.
Historical label coverage and actual sampled visual support are kept separate.
Native raw-container probe coordinates are explicitly tentative until aligned
to annotation PTS. No recommendation here creates a final label or release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/metadata/annotation-review/release-audit-20261108-v1"
FIELDS = ["video_id", "video_key", "start_sec", "end_sec", "confidence", "kind", "note"]
LABELS = {"neg": "N", "pos": "P", "transition": "T", "cut": "cut"}
LIMIT = "Sparse interval points, endpoints or short probes only; never whole-time-domain or geological certification."


def path_ref(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def resolve(value, base=ROOT):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else Path(base) / path


def bounds(start, end):
    try:
        start, end = float(start), float(end)
    except (ValueError, TypeError):
        return None
    return [start, end] if math.isfinite(start) and math.isfinite(end) and 0 <= start < end else None


def intersection(first, second):
    if first is None or second is None:
        return None
    return bounds(max(first[0], second[0]), min(first[1], second[1]))


def gap_ranges(rows):
    cursor, gaps = 0.0, []
    for span in sorted(row["range_sec"] for row in rows if row["range_sec"] is not None):
        if span[0] > cursor:
            gaps.append([cursor, span[0]])
        cursor = max(cursor, span[1])
    return gaps  # No invented end-of-video duration or negative labels in gaps.


def history_segments(current, old_rows):
    span = current["range_sec"]
    if span is None:
        return []
    old_rows = [row for row in old_rows if row["unit_id"] == current["unit_id"] and intersection(span, row["range_sec"])]
    points = sorted({*span, *(value for row in old_rows for value in intersection(span, row["range_sec"]))})
    result = []
    for start, end in zip(points, points[1:]):
        linked = [row for row in old_rows if intersection([start, end], row["range_sec"])]
        status = ("no_prior_interval" if not linked else "inherited_same_kind"
                  if all(row["saved_kind"] == current["saved_kind"] for row in linked) else "semantic_kind_changed_since_old_review")
        result.append({"range_sec": [start, end], "status": status,
                       "old_evidence_ids": [row["evidence_id"] for row in linked],
                       "old_kinds": sorted({row["saved_kind"] for row in linked}),
                       "current_kind_preserved": current["saved_kind"],
                       "visual_support_is_only_original_sample_points": True})
    return result


class Inputs:
    def __init__(self):
        self.fingerprints = {}

    def raw(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.fingerprints[path_ref(path)] = {"path": path_ref(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data

    def json(self, path):
        return json.loads(self.raw(path).decode("utf-8-sig"))

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.raw(path).decode("utf-8-sig"))))


def numeric_points(value):
    if isinstance(value, dict):
        return [point for part in value.values() for point in numeric_points(part)]
    if isinstance(value, list):
        return [float(point) for point in value if isinstance(point, (int, float)) and math.isfinite(point)]
    return []


def endpoint_support(record):
    value = record.get("observed_frame_times_sec", record.get("sample_pts_sec", record.get("sample_times_sec", [])))
    if isinstance(value, dict):
        return {side: numeric_points(value.get(side, [])) for side in ("start", "end")}
    points = numeric_points(value)
    middle = len(points) // 2
    return {"start": points[:middle], "end": points[middle:]}


def span_of(points):
    return bounds(min(points), max(points)) if points else None


def artifacts(value):
    """Reference existing paths/hashes only. Never open an image to rehash it."""
    found = {}

    def visit(part):
        if isinstance(part, dict):
            if isinstance(part.get("path"), str) and part["path"].lower().endswith((".jpg", ".jpeg", ".png")):
                found[part["path"]] = dict(part)
            for child in part.values():
                visit(child)
        elif isinstance(part, list):
            for child in part:
                visit(child)
        elif isinstance(part, str) and part.lower().endswith((".jpg", ".jpeg", ".png")):
            found.setdefault(part, {"path": part, "sha256": None, "hash_status": "not_recorded_here_not_rehashed"})

    visit(value)
    return list(found.values())


def candidate_label(value):
    # Only structured reviewer labels: descriptive prose/needs_dynamic is not a label.
    if not isinstance(value, str) or "_or_" in value or value.startswith("QC"):
        return None
    match = re.match(r"^([NPTUA])(?:$|_|;)", value)
    return match.group(1) if match else None


def classify_link(evidence, proposal, current):
    label, actual = proposal.get("candidate_label"), LABELS.get(current["saved_kind"], "unknown")
    if proposal.get("QC"):
        return "specific_QC_review", 15 if actual != "cut" else 35
    if label in {"N", "P", "T"} and actual in {"N", "P", "T"} and label != actual:
        return "specific_countercandidate_pending_adjudication", 10 if evidence["review_type"] == "native_short_probe" else 25
    # Historical "T_supported=false" may rely on the now-withdrawn O requirement.
    # Preserve that inference verbatim; it cannot itself supply T counterevidence.
    if label in {"U", "A"}:
        return "local_type_or_visibility_unresolved", 40 if evidence["review_type"] == "native_short_probe" else 70
    return None, None


def old_reviews(inputs, findings_path):
    findings = inputs.csv(findings_path)
    sample_paths = [ROOT / "data/process/EXP-022/v1-content-review-20261106-v1/sampled-frames.csv",
                    ROOT / "data/process/EXP-022/interval-review-20261105-v1/sampled-frames.csv"]
    samples = [row for path in sample_paths if path.exists() for row in inputs.csv(path)]
    sheets = inputs.csv(ROOT / "data/process/EXP-022/v1-content-review-20261106-v1/sheets.csv")
    old44 = inputs.json(ROOT / "data/process/EXP-022/interval-review-20261105-v1/summary.json")
    evidence = []
    for number, row in enumerate(findings, 2):
        span = bounds(row.get("start"), row.get("end"))
        matches = [sample for sample in samples if sample["unit_id"] == row["unit_id"]
                   and bounds(sample.get("start"), sample.get("end")) == span]
        actualfiles = [{"path": sample.get("path", sample.get("frame_path")),
                        "time_sec": float(sample.get("time", sample.get("frame_time"))),
                        "hash_status": "existing_sample_csv_reference_not_rehashed"} for sample in matches]
        contact = ["data/process/EXP-022/v1-content-review-20261106-v1/" + sheet["sheet"]
                   for sheet in sheets if row["interval_id"] in sheet["interval_ids"].split(";")]
        if row["unit_id"].startswith(("v4_", "v5_")) and matches:
            group = matches[0].get("group", "")
            video = matches[0].get("video_id", "").split("_")[0]
            contact = [path for path in old44.get("contact_sheets", []) if ("v%s-%s" % (group, video)) in path]
        evidence.append({"evidence_id": "old:L%d" % number, "unit_id": row["unit_id"], "review_type": "old_sparse_interval_review",
                         "source_path": path_ref(findings_path), "source_row": number, "range_sec": span,
                         "saved_kind": row["kind"], "actualfiles": actualfiles, "contact_sheets": contact,
                         "sample_times_sec": [item["time_sec"] for item in actualfiles], "proposals": [],
                         "time_basis": "annotation_audit_seconds", "alignment_exact": True,
                         "support_scope": "discrete sampled frames, not the entire historical interval", "raw_record": row})
    return evidence


def json_reviews(inputs, boundary_dir, native_dir):
    evidence, documents = [], {}
    paths = sorted(Path(boundary_dir).glob("review-*.json"))
    native_paths = sorted(set(Path(native_dir).glob("visual-review-*.json")) | {path for path in [Path(native_dir) / "root.json"] if path.exists()})
    for path in paths + native_paths:
        document = inputs.json(path)
        documents[path_ref(path)] = {key: value for key, value in document.items() if key != "records"}
        native = path in native_paths
        for index, record in enumerate(document.get("records", [])):
            native_time = record.get("time", {})
            span = (record.get("nominal_requested_span_sec") or
                    bounds(native_time.get("nominal_requested_start_sec"), native_time.get("nominal_requested_end_exclusive_sec"))) if native else None
            if native and span is None and "nominal_start" in record and "duration" in record:
                span = bounds(record["nominal_start"], record["nominal_start"] + record["duration"])
            support = endpoint_support(record) if not native else {}
            actual = record.get("actualfiles", {})
            time_alignment = record.get("time_alignment", {})
            entry = {"evidence_id": path.stem + ":" + str(index), "unit_id": record["unit_id"],
                     "review_type": "native_short_probe" if native else "boundary_sparse_endpoints",
                     "source_path": path_ref(path), "source_record_index": index, "probe_id": record.get("probe_id"),
                     "range_sec": span, "endpoint_sample_times_sec": support,
                     "actualfiles": artifacts(actual if native else {key: record.get(key) for key in
                         ("reviewed_contact_sheet", "reviewed_frame_paths", "additional_viewed_frame_paths", "enlarged_original_jpegs", "enlarged_frame_paths")}),
                     "contact_sheets": record.get("actually_opened_contact_sheets", actual.get("contact_sheets_actually_opened", []) if isinstance(actual, dict) else []),
                     "time_basis": "raw_nominal_container_seconds" if native else "annotation_audit_sample_PTS",
                     "alignment_exact": bool(time_alignment.get("exact_annotation_alignment_verified", False)) if native else True,
                     "support_scope": "only actually viewed sample frames; nominal probe ranges are not millisecond annotation bindings" if native else "start/end sample points only",
                     "confidence_as_reviewed": record.get("confidence"), "limits": record.get("limits", record.get("limitations", document.get("global_limits"))),
                     "raw_record": record, "proposals": []}
            if native:
                entry["raw_nominal_range_sec"] = span
                entry["annotation_range_sec"] = span if entry["alignment_exact"] else None
                entry["historical_T_mechanism_inference_is_not_current_adjudication"] = True
            if native:
                recommendation = record.get("label_recommendation", {})
                inference = record.get("inference", {})
                text = recommendation.get("label") or (inference.get("local_semantic_candidate") if isinstance(inference, dict) else None)
                entry["proposals"].append({"range_sec": span, "candidate_as_recorded": text, "candidate_label": candidate_label(text),
                                           "T_supported": recommendation.get("T_supported", inference.get("current_T_evidence_sufficient") if isinstance(inference, dict) else None),
                                           "scope": "native nominal probe context; use reviewer scope and usable frames, not all requested seconds"})
                bad = [item for item in actual if isinstance(item, dict) and
                       ("excluded" in item.get("visual_quality_status", "") or item.get("visually_usable") is False)] if isinstance(actual, list) else []
                if bad or (text and "QC" in text):
                    entry["proposals"].append({"range_sec": span, "QC": True, "bad_frame_references": bad,
                                               "scope": "QC on recorded frames/subset only; the full context range is not declared corrupt"})
            else:
                action_labels = {"retain_N_candidate": "N", "N_candidate": "N", "retain_P_candidate": "P", "retain_current_P_candidate": "P"}
                for change in record.get("candidate_changes", []):
                    side = change.get("side")
                    change_span = change.get("sampled_span_sec") or span_of(support.get(side, []))
                    text = change.get("visual_candidate") or action_labels.get(change.get("action"))
                    if text:
                        entry["proposals"].append({"range_sec": change_span, "candidate_as_recorded": text,
                                                   "candidate_label": candidate_label(text), "QC": str(text).startswith("QC"), "side": side})
                for request in record.get("dynamic_probes", []):
                    request_span = request.get("range", request.get("time_range_sec")) or bounds(request.get("start_sec"), request.get("end_sec"))
                    entry["proposals"].append({"range_sec": request_span, "request_only": True,
                                               "question": request.get("question"), "candidate_label": None})
            evidence.append(entry)
    return evidence, [path_ref(path) for path in native_paths], documents


def build(output=DEFAULT_OUT):
    inputs = Inputs()
    pool_path = ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
    pool_rows = inputs.csv(pool_path)
    pool = {row["unit_id"]: row for row in pool_rows if row.get("source") in {"dongchuan", "video4", "video5"}
            and len(row.get("group_id", "").split("|")) == 3 and row["group_id"].startswith("d|")}
    pointer_path = ROOT / "data/metadata/annotation-revisions/current-review.json"
    pointer = inputs.json(pointer_path)
    revision = resolve(pointer["working_annotations_dir"])
    source_manifest = inputs.json(revision / "sources.json")
    current = []
    csv_paths = []
    for source_key, source in source_manifest["sources"].items():
        value = Path(str(source["annotations_csv"]).replace("\\", "/"))
        path = resolve(value) if value.parts and value.parts[0] == "data" else resolve(value, revision)
        csv_paths.append(path)
        stream = csv.reader(io.StringIO(inputs.raw(path).decode("utf-8-sig")))
        if next(stream, [])[:7] != FIELDS:
            raise ValueError("Unexpected event CSV first-seven-field schema")
        for line, fields in enumerate(stream, 2):
            if not fields:
                continue
            raw = dict(zip(FIELDS, fields[:7]))
            current.append({"current_id": source_key + ":L" + str(line), "unit_id": raw.get("video_key", ""),
                            "in_150_pool": raw.get("video_key") in pool, "source_csv": path_ref(path), "source_line": line,
                            "source_sha256": inputs.fingerprints[path_ref(path)]["sha256"],
                            "range_sec": bounds(raw.get("start_sec"), raw.get("end_sec")), "saved_kind": raw.get("kind", ""),
                            "saved_label": LABELS.get(raw.get("kind"), "unknown"), "original_first_seven_fields": raw,
                            "extra_fields": fields[7:], "training_ready": False})
    old_dir = ROOT / "data/metadata/EXP-022/content-review-20261106-v1"
    evidence = old_reviews(inputs, old_dir / "interval-findings.csv")
    old_coverage = inputs.csv(old_dir / "video-coverage.csv")
    fresh, native_paths, review_documents = json_reviews(inputs, ROOT / "data/metadata/annotation-review/boundaries-20261108-v1",
                                                        ROOT / "data/metadata/annotation-review/native-targets-20261108-v2")
    evidence += fresh
    by_unit, evidence_by_unit = defaultdict(list), defaultdict(list)
    for row in current:
        by_unit[row["unit_id"]].append(row)
    for record in evidence:
        record["in_150_pool"] = record["unit_id"] in pool
        evidence_by_unit[record["unit_id"]].append(record)
    queue = []

    def enqueue(row, category, priority, span, evidence_id=None, detail=None, alignment_exact=True):
        item = {"queue_id": "Q%04d" % (len(queue) + 1), "unit_id": row["unit_id"], "in_150_pool": row["in_150_pool"],
                "current_id": row["current_id"], "current_saved_kind": row["saved_kind"], "range_sec": span,
                "priority": priority, "category": category, "evidence_id": evidence_id, "detail": detail,
                "alignment_exact": alignment_exact, "action": "review_then_adjudicate; preserve saved label until a separate decision",
                "final_label": None, "training_ready": False}
        if not alignment_exact:
            item["alignment_requirement"] = "Nominal raw/audit intersection is tentative; establish alignment before changing an annotation boundary"
        queue.append(item)

    for row in current:
        related = evidence_by_unit[row["unit_id"]]
        row["history_segments"] = history_segments(row, [item for item in related if item["review_type"] == "old_sparse_interval_review"])
        row["visual_evidence_links"] = []
        if row["saved_kind"] == "transition":
            enqueue(row, "saved_T_semantics_pending_current_contract", 20, row["range_sec"])
        if row["range_sec"] is None or row["saved_label"] == "unknown":
            enqueue(row, "invalid_range_or_unknown_explicit_kind", 5, row["range_sec"])
        for entry in related:
            if entry["review_type"] == "old_sparse_interval_review":
                overlap = intersection(row["range_sec"], entry["range_sec"])
                if overlap:
                    row["visual_evidence_links"].append({"evidence_id": entry["evidence_id"], "overlap_sec": overlap,
                        "link_type": "historical_label_overlap_only", "actual_sample_times_in_current_row": [point for point in entry["sample_times_sec"] if row["range_sec"][0] <= point < row["range_sec"][1]]})
                continue
            proposals = entry["proposals"]
            if entry["review_type"] == "native_short_probe" and not entry["alignment_exact"]:
                nominal_overlap = intersection(row["range_sec"], entry["raw_nominal_range_sec"])
                if nominal_overlap:
                    row["visual_evidence_links"].append({"evidence_id": entry["evidence_id"], "overlap_sec": None,
                        "raw_nominal_overlap_candidate_sec": nominal_overlap,
                        "link_type": "unit_context_only_nominal_similarity_not_a_time_alignment", "alignment_exact": False})
                continue  # Never use raw nominal overlap as an annotation contradiction.
            spans = ([entry["range_sec"]] if entry["review_type"] == "native_short_probe"
                     else [span_of(points) for points in entry["endpoint_sample_times_sec"].values()])
            for span in spans:
                overlap = intersection(row["range_sec"], span)
                if overlap:
                    row["visual_evidence_links"].append({"evidence_id": entry["evidence_id"], "overlap_sec": overlap,
                        "link_type": "sparse_sample_context_not_continuous_coverage", "alignment_exact": entry["alignment_exact"]})
            for proposal in proposals:
                overlap = intersection(row["range_sec"], proposal.get("range_sec"))
                if not overlap:
                    continue
                category, priority = classify_link(entry, proposal, row)
                if proposal.get("request_only"):
                    category, priority = "requested_probe_only_not_counterevidence", 100
                if category:
                    enqueue(row, category, priority, overlap, entry["evidence_id"], proposal, entry["alignment_exact"])
        important = [item for item in queue if item["current_id"] == row["current_id"] and item["priority"] < 70]
        row["adjudication_status"] = "retain_saved_label_pending_adjudication" if important else "retain_saved_label_bounded_confidence"
        row["confidence_limit"] = LIMIT
        row["whole_interval_visually_certified"] = False
    for entry in fresh:
        if entry["review_type"] != "native_short_probe" or entry["alignment_exact"]:
            continue
        context = {"unit_id": entry["unit_id"], "in_150_pool": entry["in_150_pool"],
                   "current_id": None, "saved_kind": "not_time_aligned"}
        for proposal in entry["proposals"]:
            category = "native_QC_subset_pending_OSD_alignment" if proposal.get("QC") else "native_semantic_proposal_pending_OSD_alignment"
            enqueue(context, category, 15 if proposal.get("QC") else 30, None, entry["evidence_id"],
                    {**proposal, "raw_nominal_range_sec": entry["raw_nominal_range_sec"],
                     "annotation_range_sec": None, "unit_current_ids_context_only": [row["current_id"] for row in by_unit[entry["unit_id"]]]}, False)
    unit_records = []
    all_units = sorted(set(pool) | set(by_unit) | set(evidence_by_unit))
    covered_old = {row["unit_id"] for row in old_coverage}
    for unit in all_units:
        rows, reviews = by_unit[unit], evidence_by_unit[unit]
        gaps = gap_ranges(rows)
        counts = Counter(row["saved_label"] for row in rows)
        unit_records.append({"unit_id": unit, "in_150_pool": unit in pool, "source": pool.get(unit, {}).get("source", "external_to_150"),
            "camera": pool.get(unit, {}).get("group_id", "").split("|")[1] if unit in pool else "",
            "group_id": pool.get(unit, {}).get("group_id", ""), "current_rows": len(rows),
            "current_kind_counts": dict(counts), "old_sparse_review_recorded": unit in covered_old,
            "boundary_review_records": sum(item["review_type"] == "boundary_sparse_endpoints" for item in reviews),
            "native_review_records": sum(item["review_type"] == "native_short_probe" for item in reviews),
            "unlabelled_gaps_between_known_current_rows_sec": gaps, "gaps_are_unknown_not_negative": True,
            "tail_beyond_last_saved_end_not_assessed": True, "queue_items": sum(item["unit_id"] == unit for item in queue),
            "evidence_ids": [item["evidence_id"] for item in reviews], "scope_limit": LIMIT, "training_ready": False})
    pool_current = [row for row in current if row["in_150_pool"]]
    pool_queue = [item for item in queue if item["in_150_pool"]]
    summary = {"schema_version": 1, "current_annotations_dir": path_ref(revision), "pool_units": len(pool),
        "units_with_ledger": sum(row["in_150_pool"] for row in unit_records), "current_rows": len(current),
        "pool_current_rows": len(pool_current), "external_current_rows": len(current) - len(pool_current),
        "external_units_retained_separately": sum(not row["in_150_pool"] for row in unit_records),
        "current_pool_label_counts": dict(Counter(row["saved_label"] for row in pool_current)),
        "pool_units_without_current_rows": sorted(set(pool) - set(by_unit)),
        "pool_units_with_old_sparse_review": len(set(pool) & covered_old),
        "pool_units_with_new_boundary_review": len({item["unit_id"] for item in fresh if item["review_type"] == "boundary_sparse_endpoints" and item["in_150_pool"]}),
        "native_probe_review_records": sum(item["review_type"] == "native_short_probe" for item in fresh),
        "native_review_files": native_paths, "optional_root_review_present": any(Path(path).stem in {"root", "visual-review-root"} for path in native_paths),
        "old_sample_frame_references": sum(len(item["actualfiles"]) for item in evidence if item["review_type"] == "old_sparse_interval_review"),
        "old_review_interval_records": sum(item["review_type"] == "old_sparse_interval_review" for item in evidence),
        "history_segment_status_counts": dict(Counter(atom["status"] for row in pool_current for atom in row["history_segments"])),
        "saved_T_rows": [{"unit_id": row["unit_id"], "current_id": row["current_id"], "range_sec": row["range_sec"]} for row in pool_current if row["saved_kind"] == "transition"],
        "saved_T_units": len({row["unit_id"] for row in pool_current if row["saved_kind"] == "transition"}),
        "new_explicit_T_proposals": [{"unit_id": item["unit_id"], "evidence_id": item["evidence_id"], "proposal": proposal} for item in fresh for proposal in item["proposals"] if proposal.get("candidate_label") == "T"],
        "queue_category_counts_pool": dict(Counter(item["category"] for item in pool_queue)),
        "specific_countercandidate_ranges": [item for item in pool_queue if item["category"] == "specific_countercandidate_pending_adjudication"],
        "native_proposals_without_annotation_time_assignment": [item for item in pool_queue if item["category"].startswith("native_")],
        "pool_gap_units": sum(bool(row["unlabelled_gaps_between_known_current_rows_sec"]) for row in unit_records if row["in_150_pool"]),
        "pool_gap_ranges_unknown": [{"unit_id": row["unit_id"], "ranges_sec": row["unlabelled_gaps_between_known_current_rows_sec"]}
                                    for row in unit_records if row["in_150_pool"] and row["unlabelled_gaps_between_known_current_rows_sec"]],
        "confidence_limit": LIMIT, "native_alignment_limit": "Unaligned native findings are unit context only; nominal raw/audit numerical similarity is not an annotation-time assignment. Use OSD/anchor alignment first.",
        "historical_absence_of_occlusion_is_not_T_counterevidence": True,
        "generic_dynamic_request_is_not_a_U_label": True, "notes_are_not_label_truth": True,
        "label_edits": 0, "decoded_frames": 0, "image_hash_reads": 0, "network_requests": 0, "credentials_read": False, "training_ready": False,
        "inputs": list(inputs.fingerprints.values())}
    # Detect concurrent edits without hashing any media. Fail rather than certify a mixed snapshot.
    for path in csv_paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != inputs.fingerprints[path_ref(path)]["sha256"]:
            raise ValueError("Current labels changed during build; rerun the metadata snapshot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "units.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unit_records[0]))
        writer.writeheader()
        for row in unit_records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    for name, value in [("intervals.json", {"schema_version": 1, "current_intervals": current, "evidence_records": evidence, "review_documents": review_documents, "units": unit_records, "training_ready": False}),
                        ("decision-queue.json", {"schema_version": 1, "items": sorted(queue, key=lambda item: (item["priority"], item["unit_id"], item["range_sec"] or [])), "no_decisions_applied": True}),
                        ("summary.json", summary)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def selftest():
    current = {"unit_id": "u", "range_sec": [2, 6], "saved_kind": "transition"}
    old = [{"unit_id": "u", "range_sec": [0, 4], "saved_kind": "cut", "evidence_id": "renumbered"},
           {"unit_id": "other", "range_sec": [0, 9], "saved_kind": "transition", "evidence_id": "wrong-unit"}]
    atoms = history_segments(current, old)
    assert atoms[0]["range_sec"] == [2.0, 4.0] and atoms[0]["status"] == "semantic_kind_changed_since_old_review"
    assert atoms[1]["status"] == "no_prior_interval"
    assert intersection([0, 2], [2, 3]) is None
    assert gap_ranges([{"range_sec": [0, 2]}, {"range_sec": [3, 5]}]) == [[2, 3]]
    assert candidate_label("N_or_U") is None and candidate_label("QC_U") is None
    assert classify_link({"review_type": "boundary_sparse_endpoints"}, {"request_only": True}, {"saved_kind": "pos"}) == (None, None)
    assert classify_link({"review_type": "native_short_probe"}, {"T_supported": False}, {"saved_kind": "transition"}) == (None, None)
    print("selftest: unit/time join, changed kind, half-open boundary and unknown gap OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    summary = build(args.out)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"inputs", "saved_T_rows", "specific_countercandidate_ranges", "new_explicit_T_proposals", "native_proposals_without_annotation_time_assignment"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
