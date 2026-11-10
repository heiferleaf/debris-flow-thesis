"""Create a fresh editable audit copy of an immutable semantic release.

Default is an in-memory dry run; --execute is required to write a new directory.
No images, labels, immutable release files, pointers or services are modified.
CSV confidence=real means a saved review row, not training truth. U remains an
explicit unknown kind. Initial semantic provenance is a sidecar, not a promise
that later editable CSV changes inherit release supervision eligibility.
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
DEFAULT_OUT = ROOT / "data/metadata/annotation-revisions/npt-20261108-review-v4"
FIELDS = ["video_id", "video_key", "start_sec", "end_sec", "confidence", "kind", "note"]
STATE_KIND = {"N": "neg", "P": "pos", "T": "transition", "U": "unknown", "QC_or_semantic_pending": "cut"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def create_copy(release, output=DEFAULT_OUT, *, execute=False):
    release, output = Path(release).resolve(), Path(output).resolve()
    if output == release or release in output.parents:
        raise ValueError("Editable review must be a separate new directory outside the immutable release")
    input_names = ["released-intervals.json", "external-intervals.json", "sources.json", "supervision.json", "legacy-input/sources.json"]
    raw_inputs = {name: (release / name).read_bytes() for name in input_names}
    rows = json.loads(raw_inputs["released-intervals.json"].decode("utf-8-sig"))["rows"]
    external = json.loads(raw_inputs["external-intervals.json"].decode("utf-8-sig"))["rows"]
    release_index = json.loads(raw_inputs["sources.json"].decode("utf-8-sig"))
    legacy_index = json.loads(raw_inputs["legacy-input/sources.json"].decode("utf-8-sig"))
    pool_units = {row["unit_id"] for row in rows}
    if pool_units & {row["unit_id"] for row in external}:
        raise ValueError("External units must not overlap the released pool")
    per_source = defaultdict(list)
    for row in rows:
        if row["released_state"] not in STATE_KIND:
            raise ValueError("Unsupported release state; never default it to P/N/cut")
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("Invalid release interval")
        per_source[row["source_key"]].append(row)
    source_entries = release_index["sources"]
    if set(per_source) - set(source_entries):
        raise ValueError("Release row source is missing from the immutable source index")
    files, sources, provenance = {}, {}, []
    preserved_external = []
    for key, source in source_entries.items():
        source_id = source["source_id"]
        if digest(source_id.encode("utf-8"))[:12] != key:
            raise ValueError("Source key is incompatible with apply_annotations_dir")
        snapshot = (release / source["legacy_snapshot_csv"]).resolve()
        if release not in snapshot.parents:
            raise ValueError("Source snapshot must remain inside the immutable release")
        raw = snapshot.read_bytes()
        raw_inputs[source["legacy_snapshot_csv"]] = raw
        if digest(raw) != source["parent_csv_sha256"]:
            raise ValueError("Immutable parent CSV snapshot SHA does not match")
        table = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
        header, originals = table[0], table[1:]
        if header[:7] != FIELDS:
            raise ValueError("Expected consolidated first-seven-field annotation CSV")
        units = defaultdict(list)
        for original in originals:
            if len(original) >= 7:
                units[original[1]].append(original)
        additions = defaultdict(list)
        for row in per_source[key]:
            unit = row["unit_id"]
            if not units.get(unit):
                raise ValueError("Released unit missing from its original source CSV")
            original = row.get("original_first_seven_fields") or {}
            note = original.get("note", "")
            original_note = note
            kind = STATE_KIND[row["released_state"]]
            if kind == "unknown":
                note += ("\n" if note else "") + "[类型未定 U；仅供复核，不作训练真值]"
            if row.get("decision_id") == "D002":
                note += ("\n" if note else "") + "[D002：新增 P 候选；技术 QC 未评估；训练未就绪]"
            elif row['released_state'] in {'N','T','P'} and not row['eligible_for_label_supervision']:
                note += ("\n" if note else "") + '[工作候选%s；监督隔离，未作训练真值]' % row['released_state']
            extra = row.get("original_extra_fields", [])
            values = [original.get("video_id", units[unit][0][0]), unit, row["start_sec"], row["end_sec"],
                      "real", kind, note] + list(extra)
            values += [""] * max(0, len(header) - len(values))
            additions[unit].append((row, values, original_note))
        for unit, candidates in additions.items():
            candidates.sort(key=lambda item: (float(item[1][2]), float(item[1][3])))
            if any(float(left[1][3]) > float(right[1][2]) for left, right in zip(candidates, candidates[1:])):
                raise ValueError("Released editable intervals would overlap")
        folder = Path(source_id.replace("\\", "/")).name + "--" + key
        relative_csv = folder + "/events.csv"
        output_rows, seen = [], set()
        for original in originals:
            unit = original[1] if len(original) >= 2 else ""
            if unit not in additions:
                output_rows.append(original)  # Complete outside-pool row values, including notes/extras.
                if unit and len(original) >= 7:
                    preserved_external.append((key, unit, original))
                continue
            if unit in seen:
                continue
            seen.add(unit)
            for row, values, original_note in additions[unit]:
                output_rows.append(values)
                provenance.append({"review_csv": relative_csv, "initial_csv_line": len(output_rows) + 1,
                                   "unit_id": unit, "current_id": row["current_id"], "review_kind": values[5],
                                   "original_kind": row.get("original_kind"), "original_note": original_note,
                                   "release_row": row, "scope": "initial export only; later user edits require renewed review"})
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(output_rows)
        # An entirely external source can remain byte-identical, including CSV formatting.
        csv_data = b"\xef\xbb\xbf" + stream.getvalue().encode("utf-8") if additions else raw
        files[relative_csv] = csv_data
        ancestor = legacy_index["sources"][key]
        sources[key] = {"source_id": source_id, "origin_csv": ancestor.get("origin_csv", source["parent_csv_path"]),
                        "origin_sha256": ancestor.get("origin_sha256", source["parent_csv_sha256"]),
                        "annotations_csv": relative_csv,
                        "parent_annotation_csv": reference(snapshot), "parent_annotation_sha256": digest(raw),
                        "initial_annotations_sha256": digest(csv_data),
                        "immutable_release_intervals": reference(release / "released-intervals.json"),
                        "immutable_release_intervals_sha256": digest(raw_inputs["released-intervals.json"]),
                        "supervision_json": reference(release / "supervision.json"), "supervision_sha256": digest(raw_inputs["supervision.json"])}
    external_units = {row["unit_id"] for row in external}
    if {unit for _, unit, _ in preserved_external} != external_units or len(preserved_external) != len(external):
        raise ValueError("Outside-pool source rows do not match the external release inventory")
    manifest = {"schema_version": 1, "sources": sources, "editable_review": True, "training_ready": False,
                "immutable_release": reference(release), "immutable_release_manifest_sha256": digest(raw_inputs["sources.json"]),
                "supervision_json": reference(release / "supervision.json"), "supervision_sha256": digest(raw_inputs["supervision.json"]),
                "csv_confidence_real_means_saved_review_only": True, "unknown_kind_is_not_training_truth": True,
                "editable_csv_is_not_training_manifest": True,
                "provenance_sidecar": "release-provenance.json",
                "supervision_does_not_automatically_follow_later_CSV_edits": True}
    files["sources.json"] = encoded(manifest)
    files["release-provenance.json"] = encoded({"schema_version": 1, "immutable_inputs": [
        {"path": reference(release / name), "sha256": digest(raw)} for name, raw in raw_inputs.items()],
        "initial_released_rows": provenance, "external_rows_preserved": [
            {"source_key": key, "unit_id": unit, "original_row_values": values} for key, unit, values in preserved_external],
        "training_ready": False, "scope": "initial export provenance only, not training eligibility for later mutable edits"})
    report = {"dry_run": not execute, "release": reference(release), "output": reference(output),
              "source_count": len(sources), "pool_units": len(pool_units), "pool_intervals": len(rows),
              "exported_kind_counts_pool": dict(Counter(STATE_KIND[row["released_state"]] for row in rows)),
              "external_units_preserved": len(external_units), "external_rows_preserved": len(preserved_external),
              "training_ready": False, "images_read_or_copied": 0, "current_pointer_changed": False}
    if execute:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Editable target is not empty; refusing to overwrite user review edits")
        for name, raw in raw_inputs.items():
            if (release / name).read_bytes() != raw:
                raise ValueError("Immutable release changed during export")
        output.mkdir(parents=True, exist_ok=True)
        for name, raw in files.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
    return report


def selftest():
    with tempfile.TemporaryDirectory(prefix="annotation-review-copy-") as directory:
        root = Path(directory)
        release, target = root / "immutable", root / "editable"
        source_id = (root / "source").as_posix()
        key = digest(source_id.encode("utf-8"))[:12]
        snapshot = "legacy-input/source/events.csv"
        raw = ("video_id,video_key,start_sec,end_sec,confidence,kind,note\r\n"
               "video,u,0,2,real,transition,原备注\r\nvideo,u,4,6,real,pos,P-note\r\nexternal,e,0,7,real,pos,keep exactly\r\n").encode("utf-8-sig")
        rows = [{"unit_id": "u", "source_key": key, "current_id": "source:L2", "released_state": "U",
                 "start_sec": 0, "end_sec": 2, "original_kind": "transition", "original_first_seven_fields": dict(zip(FIELDS, ["video", "u", "0", "2", "real", "transition", "原备注"]))},
                {"unit_id": "u", "source_key": key, "current_id": "source:L3", "released_state": "P",
                 "start_sec": 4, "end_sec": 6, "original_kind": "pos", "original_first_seven_fields": dict(zip(FIELDS, ["video", "u", "4", "6", "real", "pos", "P-note"]))},
                {"unit_id": "u", "source_key": key, "current_id": "D002:new", "released_state": "P", "decision_id": "D002",
                 "start_sec": 3, "end_sec": 4, "original_kind": None, "original_first_seven_fields": None}]
        fixtures = {snapshot: raw, "released-intervals.json": encoded({"rows": rows}),
                    "external-intervals.json": encoded({"rows": [{"unit_id": "e"}]}), "supervision.json": encoded({"training_ready": False}),
                    "sources.json": encoded({"sources": {key: {"source_id": source_id, "legacy_snapshot_csv": snapshot,
                                              "parent_csv_path": "fixture-parent", "parent_csv_sha256": digest(raw)}}}),
                    "legacy-input/sources.json": encoded({"sources": {key: {"source_id": source_id, "origin_csv": "fixture-origin", "origin_sha256": digest(raw)}}})}
        for name, data in fixtures.items():
            path = release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        assert create_copy(release, target)["dry_run"] and not target.exists()
        create_copy(release, target, execute=True)
        manifest = json.loads((target / "sources.json").read_text(encoding="utf-8"))
        with (target / manifest["sources"][key]["annotations_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.DictReader(stream))
        assert exported[0]["kind"] == "unknown" and exported[0]["note"].startswith("原备注")
        assert not any(float(row["start_sec"]) <= 2.5 < float(row["end_sec"]) for row in exported if row["video_key"] == "u")
        assert "技术 QC 未评估" in next(row for row in exported if row["start_sec"] == "3")["note"]
        assert [row for row in exported if row["video_key"] == "e"][0]["note"] == "keep exactly"
        assert all((release / name).read_bytes() == data for name, data in fixtures.items())
        try:
            create_copy(release, target, execute=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Editable user work must not be overwritten")
    print("selftest: U export, notes, unknown gap, external retention, immutable input and no-overwrite OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        if not args.release:
            parser.error("--release is required")
        print(json.dumps(create_copy(args.release, args.out, execute=args.execute and not args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
