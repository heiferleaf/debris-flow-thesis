"""Publish a bounded semantic working view; never modify parent events/pointers.

D001/D002 and explicitly supplied root decisions are applied. This is not a training release, geological
certification, technical-QC pass, or a new user signature. Old T remains in
original_kind; unsupported legacy semantics become masked U in the new view,
not a claim that those human observations have been disproved. No media reads.
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
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "data/metadata/annotation-revisions/npt-20261107-proposals-v3"
DEFAULT_OUT = ROOT / "data/metadata/annotation-revisions/npt-20261108-evidence-working-v1"
LEDGER = ROOT / "data/metadata/annotation-review/release-audit-20261108-v1"
POOL = ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
FIELDS = ["video_id", "video_key", "start_sec", "end_sec", "confidence", "kind", "note"]
TECHNICAL_QC = "not_assessed_by_semantic_release"
T_RANGES = {(1036.04, 1038.0), (1038.0, 1040.23)}
FRAME_ROOT = "data/process/video_frames/video1_segment/frames/"
DECISIONS = {
    "D001": {
        "unit_id": "v1_16", "ranges_sec": sorted([list(span) for span in T_RANGES]),
        "released_state": "T", "authority": "root working decision", "new_user_signature": False,
        "independent_geological_certification": False, "prior_user_anchor_acknowledgement": True,
        "evidence_refs": [{"audit_pts_sec": pts, "path": FRAME_ROOT + name} for pts, name in [
            (1033.92, "v1_16_frame_000493_t1034s.jpg"), (1036.04, "v1_16_frame_000494_t1036s.jpg"),
            (1038.12, "v1_16_frame_000495_t1038s.jpg"), (1040.23, "v1_16_frame_000496_t1040s.jpg")]],
        "reason": "Root inspected the existing audit frames: a small front has entered along the upper-right active channel; the current target is visible and identifiable while material/collective-transport details remain partial. Later context only corroborates the identity of that already visible target. No area threshold is used.",
    },
    "D002": {
        "unit_id": "v1_24", "ranges_sec": [[398.0, 402.0]], "released_state": "P", "release_status": "candidate",
        "authority": "root working decision", "new_user_signature": False, "independent_geological_certification": False,
        "evidence_refs": [{"audit_pts_sec": pts, "path": FRAME_ROOT + name} for pts, name in [
            (398.30, "v1_24_frame_000190_t398s.jpg"), (400.40, "v1_24_frame_000191_t400s.jpg"),
            (402.49, "v1_24_frame_000192_t402s.jpg")]],
        "reason": "Root inspected these audit samples and judged the same continuous gray-brown body as neighboring P. Water droplets/light patches remain an independent technical-QC issue. The left edge 398s reuses the existing cut endpoint; it is not a measured physical onset. This is an explicit sampled local extension, not automatic gap filling.",
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path_ref(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def resolve(value, base=ROOT):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else Path(base) / path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_once(path, data):
    """Idempotent identical output only; never overwrite an existing different release."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Existing release file differs; create another version: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def overlap(first, second):
    if first is None or second is None:
        return False
    return max(first[0], second[0]) < min(first[1], second[1])


def load_decisions(path=None):
    """Append explicit source-row or within-row decisions to a new version.

    JSON: {"decisions":{"D003":{"unit_id":..., "ranges_sec":[[s,e]],
    "source_kind":"transition|neg|pos|cut", "released_state":"N|T|P|U", "eligible_for_label_supervision":bool,
    "evidence_level":..., "authority":"root working decision",
    "evidence_refs":[...], "reason":...}}}. source_kind defaults to transition.
    Every range must fit one original row. Partial decisions split that row
    while retaining its original fields and preserving all remaining time.
    D001/D002 cannot be silently replaced, and existing output never changes.
    """
    decisions = dict(DECISIONS)
    if path is None:
        return decisions
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for key, decision in document["decisions"].items():
        if key in DECISIONS:
            if decision != DECISIONS[key]:
                raise ValueError("Built-in D001/D002 are static; do not overwrite them")
            continue
        state, eligible = decision.get("released_state"), decision.get("eligible_for_label_supervision")
        ranges = decision.get("ranges_sec")
        if (not key or not decision.get("unit_id") or state not in {"N", "T", "P", "U"}
                or decision.get("source_kind", "transition") not in {"transition", "neg", "pos", "cut"}
                or type(eligible) is not bool or (state == "U" and eligible)
                or decision.get("authority") != "root working decision" or not decision.get("evidence_level")
                or not decision.get("reason") or not decision.get("evidence_refs")
                or decision.get("new_user_signature", False) is not False
                or decision.get("independent_geological_certification", False) is not False
                or not isinstance(ranges, list) or not ranges):
            raise ValueError("Additional decisions need explicit state/masks, root authority, evidence and ranges")
        for span in ranges:
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in span)
                    or not 0 <= span[0] < span[1]):
                raise ValueError("Invalid explicit decision range")
        decisions[key] = decision
    return decisions


def semantic_state(unit, kind, span):
    if kind in {"neg", "pos"}:
        state = "N" if kind == "neg" else "P"
        return state, "human_reused", True, "Retain the saved explicit %s label; bounded human evidence is reused without a new geological certification or a technical-QC pass." % state
    if kind == "transition" and unit == "v1_16" and tuple(span) in T_RANGES:
        return "T", "root_D001_current_visible_target_partial_detail", True, DECISIONS["D001"]["reason"]
    if kind == "transition":
        return "U", "legacy_T_not_validated_new_semantics", False, "Preserve the original human T value, but its current semantic training validity is not established. U here is isolation pending further evidence, not disproof and not N. Old confidence=real does not override the masks."
    if kind == "cut":
        return "QC_or_semantic_pending", "legacy_cut_reason_not_fully_adjudicated", False, "Preserve the original cut; technical quality and semantic uncertainty have not been separated here. Do not declare every cut corrupt or normal."
    return "U", "unknown_explicit_kind_not_reinterpreted", False, "Unknown explicit kind is retained as masked U; it is never filled as N."


def split_row_spans(original, decisions):
    """Partition only at supported decision edges; never clip or fill a gap."""
    span = [float(original["start_sec"]), float(original["end_sec"])]
    edges = {span[0], span[1]}
    for key, item in decisions.items():
        if (key in DECISIONS or item["unit_id"] != original["video_key"]
                or item.get("source_kind", "transition") != original["kind"]):
            continue
        for change in item["ranges_sec"]:
            if overlap(change, span):
                if change[0] < span[0] or change[1] > span[1]:
                    raise ValueError("A decision range must fit one original row: " + key)
                edges.update(change)
    ordered = sorted(edges)
    return [[left, right] for left, right in zip(ordered, ordered[1:])]


def make_row(original, source, pool, evidence_refs, risk_ids, risk_categories, decisions=None, span_override=None):
    decisions = DECISIONS if decisions is None else decisions
    unit = original["video_key"]
    span = span_override if span_override is not None else [float(original["start_sec"]), float(original["end_sec"])]
    if not all(math.isfinite(value) for value in span) or span[0] < 0 or span[1] <= span[0]:
        raise ValueError("Invalid current interval; a root decision is needed before publication")
    state, level, eligible, reason = semantic_state(unit, original["kind"], span)
    decision = "D001" if state == "T" else None
    extra = [(key, item) for key, item in decisions.items() if key not in DECISIONS
             and item.get("source_kind", "transition") == original["kind"] and item["unit_id"] == unit
             and any(change[0] <= span[0] and span[1] <= change[1] for change in item["ranges_sec"])]
    if len(extra) > 1 or (decision and extra):
        raise ValueError("Conflicting explicit root decisions for the same original interval")
    if extra:
        decision, item = extra[0]
        state, level, eligible, reason = item["released_state"], item["evidence_level"], item["eligible_for_label_supervision"], item["reason"]
    if decision:
        evidence_refs = evidence_refs + [{"decision_id": decision, **item} for item in decisions[decision]["evidence_refs"]]
    camera = pool[unit]["group_id"].split("|")[1] if unit in pool else None
    in_pool = unit in pool
    full_span = [float(original['start_sec']), float(original['end_sec'])]
    release_id = source['current_id'] if span == full_span else '%s:span:%s-%s' % (source['current_id'], span[0], span[1])
    return {
        "release_id": release_id, "unit_id": unit, "in_150_pool": in_pool, "source_camera": camera,
        "group_id": pool[unit]["group_id"] if in_pool else None,
        "source_id": source["source_id"], "source_key": source["source_key"], "current_id": source["current_id"],
        "parent_version": path_ref(PARENT), "parent_sources_sha256": source["parent_sources_sha256"],
        "parent_csv_path": source["parent_csv_path"], "parent_csv_sha256": source["parent_csv_sha256"],
        "legacy_snapshot_csv": source["legacy_snapshot_csv"], "start_sec": span[0], "end_sec": span[1], "interval_convention": "[start,end)",
        "original_kind": original["kind"], "original_first_seven_fields": original, "original_extra_fields": source["extra_fields"],
        "released_state": state if in_pool else None,
        "release_status": ("candidate_supervision_held" if state in {'N','T','P'} and not eligible else "bounded_working_label") if in_pool else "outside_150_release_preserved_only",
        "evidence_level": level if in_pool else "outside_pool_not_semantically_released",
        "state_semantics": reason if in_pool else "Original external metadata is preserved separately and excluded from the 150-unit release.",
        "eligible_for_label_supervision": eligible if in_pool else False,
        "mask_three_class": eligible if in_pool else False, "mask_existence": eligible if in_pool else False,
        "existence_value": (0 if state == "N" else 1 if state in {"T", "P"} else None) if in_pool else None,
        "technical_QC": TECHNICAL_QC, "eligible_for_input_training": False,
        "evidence_refs": evidence_refs, "risk_queue_ids": risk_ids, "risk_categories": risk_categories,
        "decision_id": decision if in_pool else None, "reason": reason if in_pool else "Not a member of the 150 development units",
        "whole_interval_geologically_certified": False, "training_ready": False,
    }


def add_D002(rows):
    unit_rows = [row for row in rows if row["unit_id"] == "v1_24"]
    gap = [398.0, 402.0]
    if (not unit_rows or any(overlap([row["start_sec"], row["end_sec"]], gap) for row in unit_rows)
            or not any(row["end_sec"] == gap[0] for row in unit_rows)
            or not any(row["start_sec"] == gap[1] and row["original_kind"] == "pos" for row in unit_rows)):
        raise ValueError("D002 is not the exact current v1_24 gap; do not apply it to another interval")
    source = unit_rows[0]
    row = {**source, "release_id": "D002:v1_24:398-402", "current_id": "D002:new_gap_interval",
           "start_sec": gap[0], "end_sec": gap[1], "original_kind": None,
           "original_first_seven_fields": None, "original_extra_fields": [], "released_state": "P", "release_status": "candidate",
           "evidence_level": "sampled_local_extension", "state_semantics": DECISIONS["D002"]["reason"],
           "eligible_for_label_supervision": False, "mask_three_class": False, "mask_existence": False,
           "existence_value": 1, "decision_id": "D002", "reason": DECISIONS["D002"]["reason"],
           "supervision_hold_reason": "New local extension candidate held pending independent technical QC; no input pass inferred",
           "evidence_refs": [{"decision_id": "D002", **item} for item in DECISIONS["D002"]["evidence_refs"]],
           "risk_queue_ids": [], "risk_categories": ["new_sampled_extension_pending_independent_technical_QC"],
           "boundary_note": "398s is the old cut endpoint, not a measured true onset"}
    rows.append(row)


def csv_bytes(rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8")


def publish(output=DEFAULT_OUT, decisions_path=None):
    output = Path(output).resolve()
    decisions = load_decisions(decisions_path)
    if output == PARENT.resolve() or PARENT.resolve() in output.parents:
        raise ValueError("A separate new release directory is required")
    source_manifest_bytes = (PARENT / "sources.json").read_bytes()
    source_manifest_sha = sha(source_manifest_bytes)
    sources = json.loads(source_manifest_bytes.decode("utf-8-sig"))["sources"]
    pool_bytes = POOL.read_bytes()
    pool = {row["unit_id"]: row for row in csv.DictReader(io.StringIO(pool_bytes.decode("utf-8-sig")))
            if row.get("source") in {"dongchuan", "video4", "video5"} and row.get("group_id", "").startswith("d|")}
    audit_raw = {name: (LEDGER / name).read_bytes() for name in ("intervals.json", "decision-queue.json", "summary.json")}
    audit = json.loads(audit_raw["intervals.json"].decode("utf-8-sig"))
    queue = json.loads(audit_raw["decision-queue.json"].decode("utf-8-sig"))["items"]
    rows, external, snapshots, source_index = [], [], {}, {}
    original_pool_row_count = 0
    for key, entry in sources.items():
        path_value = Path(str(entry["annotations_csv"]).replace("\\", "/"))
        path = resolve(path_value) if path_value.parts[0] == "data" else resolve(path_value, PARENT)
        raw = path.read_bytes()
        parent_sha = sha(raw)
        snapshot = Path("legacy-input") / (path.parent.name + "/events.csv")
        snapshots[path] = raw
        source_index[key] = {"source_id": entry["source_id"], "parent_csv_path": path_ref(path), "parent_csv_sha256": parent_sha,
                             "legacy_snapshot_csv": snapshot.as_posix(), "legacy_snapshot_sha256": parent_sha}
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
        if next(reader, [])[:7] != FIELDS:
            raise ValueError("Unexpected first-seven-fields event schema")
        for line, fields in enumerate(reader, 2):
            if not fields:
                continue
            original = dict(zip(FIELDS, fields[:7]))
            unit, span = original["video_key"], [float(original["start_sec"]), float(original["end_sec"])]
            matched = [item for item in audit["current_intervals"] if item["unit_id"] == unit and overlap(span, item["range_sec"])]
            risks = [item for item in queue if item["unit_id"] == unit and (item["range_sec"] is None or overlap(span, item["range_sec"]))]
            if (original["kind"] in {"neg", "pos"} and unit in pool and any(item["alignment_exact"]
                    and item["category"] == "specific_countercandidate_pending_adjudication" for item in risks)):
                raise ValueError("An aligned specific N/P countercandidate needs another explicit root decision")
            refs = [{"artifact": "evidence-input/intervals.json", "evidence_id": link["evidence_id"], "link_scope": link["link_type"]}
                    for item in matched for link in item["visual_evidence_links"]]
            refs = list({json.dumps(ref, sort_keys=True): ref for ref in refs}.values())
            source = {**source_index[key], "source_key": key, "current_id": key + ":L" + str(line),
                      "parent_sources_sha256": source_manifest_sha, "extra_fields": fields[7:]}
            if unit in pool:
                original_pool_row_count += 1
            pieces = split_row_spans(original, decisions) if unit in pool else [span]
            for part, piece in enumerate(pieces):
                part_source = dict(source)
                if len(pieces) > 1:
                    part_source["current_id"] = source["current_id"] + ":part" + str(part + 1)
                result = make_row(original, part_source, pool, refs, [risk["queue_id"] for risk in risks],
                                  sorted({risk["category"] for risk in risks}), decisions, piece)
                (rows if unit in pool else external).append(result)
            if pieces[0][0] != span[0] or pieces[-1][1] != span[1] or any(
                    left[1] != right[0] for left, right in zip(pieces, pieces[1:])):
                raise ValueError("Decision partition must preserve all original interval time")
    if len(pool) != 150 or {row["unit_id"] for row in rows} != set(pool):
        raise ValueError("The complete current 150-unit pool must be represented")
    d001 = [row for row in rows if row["decision_id"] == "D001"]
    if {tuple([row["start_sec"], row["end_sec"]]) for row in d001} != T_RANGES or len(d001) != 2:
        raise ValueError("D001 must match exactly the two approved original v1_16 T rows")
    for key, decision in decisions.items():
        if key not in DECISIONS and {tuple([row["start_sec"], row["end_sec"]]) for row in rows if row["decision_id"] == key} != {tuple(span) for span in decision["ranges_sec"]}:
            raise ValueError("Additional decision coverage differs from its explicit ranges: " + key)
    for unit in pool:
        ordered = sorted((row for row in rows if row["unit_id"] == unit), key=lambda row: row["start_sec"])
        if any(left["end_sec"] > right["start_sec"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Overlapping source rows require adjudication before publication")
    add_D002(rows)
    rows.sort(key=lambda row: (row["unit_id"], row["start_sec"], row["end_sec"]))
    counts = Counter(row["released_state"] for row in rows)
    valid_T = [row for row in rows if row["released_state"] == "T" and row["eligible_for_label_supervision"]]
    state_summary = {}
    for state in sorted(counts):
        selected = [row for row in rows if row["released_state"] == state]
        state_summary[state] = {"rows": len(selected), "seconds": round(math.fsum(row["end_sec"] - row["start_sec"] for row in selected), 9),
                                "units": len({row["unit_id"] for row in selected}), "cameras": len({row["source_camera"] for row in selected}),
                                "eligible_label_rows": sum(row["eligible_for_label_supervision"] for row in selected)}
    index = {"schema_version": 1, "release_kind": "bounded_semantic_working_view_not_legacy_events_schema",
             "parent_version": path_ref(PARENT), "parent_sources_sha256": source_manifest_sha,
             "legacy_sources_snapshot": "legacy-input/sources.json", "sources": source_index,
             "released_intervals_csv": "released-intervals.csv", "released_intervals_json": "released-intervals.json",
             "supervision": "supervision.json", "training_ready": False}
    summary = {"schema_version": 1, "parent_version": path_ref(PARENT), "parent_sources_sha256": source_manifest_sha,
               "pool_units": len(pool), "pool_original_rows": original_pool_row_count, "new_D002_rows": 1, "pool_released_rows": len(rows),
               "external_units": len({row["unit_id"] for row in external}), "external_rows": len(external),
               "state_summary": state_summary, "evidence_level_counts": dict(Counter(row["evidence_level"] for row in rows)),
               "reliable_T_cameras": len({row["source_camera"] for row in valid_T}), "reliable_T_units": sorted({row["unit_id"] for row in valid_T}),
               "V0_three_class_release_allowed": False, "V0_hold_reason": "Validated working T exists in only one camera; broader T evidence and the remaining training gates are not complete",
               "technical_QC": TECHNICAL_QC, "training_ready": False, "old_labels_modified": False,
               "ledger_risks_preserved": True, "root_working_decisions": list(decisions), "new_user_signature": False,
               "geological_certification": False, "image_reads": 0, "network_requests": 0, "credentials_read": False,
               "pool_manifest": {"path": path_ref(POOL), "sha256": sha(pool_bytes)},
               "audit_evidence_inputs": [{"path": path_ref(LEDGER / name), "snapshot": "evidence-input/" + name, "sha256": sha(raw)} for name, raw in audit_raw.items()]}
    supervision = {"schema_version": 1, "release_kind": "semantic_label_availability_only", "training_ready": False,
                   "mask_semantics": "These are semantic label masks, never an input technical-QC pass. All actual training also requires its independent technical QC, protocol and authorization gates.",
                   "U_policy": "U and unresolved cut have both masks false regardless of legacy confidence=real; U is never silently N",
                   "decisions": decisions, "rows": rows, "V0_three_class_release_allowed": False}
    if decisions_path is not None:
        summary["additional_decision_input"] = {"path": path_ref(decisions_path), "sha256": sha(Path(decisions_path).read_bytes())}
        if summary["reliable_T_cameras"] != 1:
            summary["V0_hold_reason"] = "Additional root decisions are recorded, but V0 still requires an explicit protocol/evidence/QC release; it is never enabled automatically"
    # Freeze the small reports actually cited by new decisions, not image/video copies.
    cited_reports = {}
    for decision in decisions.values():
        for evidence in decision.get('evidence_refs', []):
            if evidence.get('artifact'):
                artifact = resolve(evidence['artifact'])
                raw = artifact.read_bytes()
                original_path = path_ref(artifact)
                cited_reports[original_path] = (raw, 'evidence-input/decisions/%s-%s' % (sha(raw)[:12], artifact.name))
    summary['decision_evidence_inputs'] = [
        {'path': name, 'sha256': sha(raw), 'snapshot': snapshot}
        for name, (raw, snapshot) in sorted(cited_reports.items())]
    # Verify before writing; metadata source snapshots must form one unchanged read.
    if (PARENT / "sources.json").read_bytes() != source_manifest_bytes or any(path.read_bytes() != raw for path, raw in snapshots.items()):
        raise ValueError("Parent labels changed during publication; rerun against a consistent snapshot")
    files = {"legacy-input/sources.json": source_manifest_bytes, "sources.json": json_bytes(index),
             "released-intervals.csv": csv_bytes(rows), "released-intervals.json": json_bytes({"schema_version": 1, "rows": rows, "training_ready": False}),
             "supervision.json": json_bytes(supervision), "external-intervals.csv": csv_bytes(external),
             "external-intervals.json": json_bytes({"schema_version": 1, "rows": external, "not_in_150_release": True}),
             "summary.json": json_bytes(summary)}
    files.update({source_index[key]["legacy_snapshot_csv"]: snapshots[resolve(value["parent_csv_path"])] for key, value in source_index.items()})
    files.update({"evidence-input/" + name: raw for name, raw in audit_raw.items()})
    files.update({snapshot: raw for raw, snapshot in cited_reports.values()})
    if decisions_path is not None:
        files['additional-decisions.json'] = Path(decisions_path).read_bytes()
    for name, raw in files.items():
        destination = output / name
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError("Existing different release must not be overwritten: " + str(destination))
    for name, raw in files.items():
        write_once(output / name, raw)
    if any(path.read_bytes() != raw for path, raw in snapshots.items()):
        raise ValueError("Parent changed after publication; root must review the version before selecting it")
    return summary


def selftest():
    assert semantic_state("old-T", "transition", [0, 2])[:3] == ("U", "legacy_T_not_validated_new_semantics", False)
    assert semantic_state("gap", None, [2, 3])[0] == "U"
    assert semantic_state("v1_16", "transition", [1036.04, 1038])[0] == "T"
    assert not overlap([0, 2], [2, 3])
    with tempfile.TemporaryDirectory(prefix="annotation-working-release-selftest-") as directory:
        root = Path(directory)
        original = root / "original.csv"
        original.write_bytes(b"video_id,video_key,start_sec,end_sec,confidence,kind,note\r\nx,u,0,2,real,transition,legacy\r\n")
        raw = original.read_bytes()
        write_once(root / "snapshot.csv", raw)
        assert original.read_bytes() == raw == (root / "snapshot.csv").read_bytes()
        source = {"current_id": "test:L2", "source_id": "test", "source_key": "test", "parent_sources_sha256": sha(raw),
                  "parent_csv_path": str(original), "parent_csv_sha256": sha(raw), "legacy_snapshot_csv": "snapshot.csv", "extra_fields": []}
        row = make_row(dict(zip(FIELDS, ["x", "u", "0", "2", "real", "transition", "legacy"])), source, {"u": {"group_id": "d|camera|date"}}, [], [], [])
        assert row["original_kind"] == "transition" and row["released_state"] == "U"
        assert row["original_first_seven_fields"]["confidence"] == "real" and not row["mask_three_class"] and not row["mask_existence"]
        extra_path = root / "decisions.json"
        extra_path.write_bytes(json_bytes({"decisions": {"D003": {"unit_id": "u", "ranges_sec": [[0, 2]],
            "released_state": "N", "eligible_for_label_supervision": True, "evidence_level": "fixture_explicit_review",
            "authority": "root working decision", "evidence_refs": [{"path": "fixture-only-no-image-read"}],
            "reason": "Synthetic test of explicit additional decisions, not a real label decision"}}}))
        updated = make_row(row["original_first_seven_fields"], source, {"u": {"group_id": "d|camera|date"}}, [], [], [], load_decisions(extra_path))
        assert updated["original_kind"] == "transition" and updated["released_state"] == "N" and updated["decision_id"] == "D003"
        # A local correction must retain the unmodified remainder and source row.
        old_negative = dict(zip(FIELDS, ["x", "u", "0", "10", "real", "neg", "human note"]))
        partial = {"D004": {"unit_id": "u", "source_kind": "neg", "ranges_sec": [[4, 6]],
                    "released_state": "U", "eligible_for_label_supervision": False,
                    "evidence_level": "fixture", "reason": "boundary bracket", "evidence_refs": []},
                   "D005": {"unit_id": "u", "source_kind": "neg", "ranges_sec": [[6, 10]],
                    "released_state": "P", "eligible_for_label_supervision": True,
                    "evidence_level": "fixture", "reason": "visible target", "evidence_refs": []}}
        parts = split_row_spans(old_negative, partial)
        assert parts == [[0.0, 4], [4, 6], [6, 10.0]]
        changed = [make_row(old_negative, source, {"u": {"group_id": "d|camera|date"}}, [], [], [], partial, span)
                   for span in parts]
        assert [item["released_state"] for item in changed] == ["N", "U", "P"]
        assert [item["mask_existence"] for item in changed] == [True, False, True]
        assert len({item['release_id'] for item in changed}) == 3
        assert all(item["original_first_seven_fields"] == old_negative for item in changed)
        assert math.fsum(item["end_sec"] - item["start_sec"] for item in changed) == 10
        # Resolving semantic uncertainty in a former cut must not clear QC.
        old_cut = dict(zip(FIELDS, ['x', 'u', '0', '2', 'real', 'cut', 'fog']))
        cut_decision = {'D006': {'unit_id':'u', 'source_kind':'cut', 'ranges_sec':[[0,2]],
                               'released_state':'U', 'eligible_for_label_supervision':False,
                               'evidence_level':'fixture', 'authority':'root working decision',
                               'evidence_refs':[{'path':'fixture'}], 'reason':'Fog hides target region'}}
        extra_path.write_bytes(json_bytes({'decisions':cut_decision}))
        cut_result = make_row(old_cut, source, {'u':{'group_id':'d|camera|date'}}, [], [], [], load_decisions(extra_path))
        assert cut_result['released_state'] == 'U' and not cut_result['mask_existence']
        assert cut_result['technical_QC'] == TECHNICAL_QC and not cut_result['eligible_for_input_training']
        invalid = {"bad": {**partial["D004"], "ranges_sec": [[8, 12]]}}
        try:
            split_row_spans(old_negative, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("A local correction silently extended beyond its source row")
        assert original.read_bytes() == raw
    print("selftest: legacy T/U masks; partial N/U/P correction conserves source time/fields; invalid range rejected; snapshots intact OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--decisions", type=Path, help="Explicit root decisions with source kind/ranges; use a new --out version")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        print(json.dumps(publish(args.out, args.decisions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
