"""EXP-016 metadata freezing and explicit checked H264-to-JPEG preparation.

No training loop. Default is a read-only metadata audit; --extract-from is an
explicit pixel action and --extract-probe uses a separate one-video cache.

Examples:
  python experiments/runspecs/EXP-016/prepare_inputs.py
  python experiments/runspecs/EXP-016/prepare_inputs.py --freeze metadata-v1 \
    --qc-override v1_09=data/process/EXP-016/qc-refresh/v1_09.csv

A metadata freeze alone is NOT a pixel-ready/training-ready declaration.
All outputs are exclusive; an existing output directory is never overwritten.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units_stride import build_forbidden  # noqa: E402
from build_units import QC_STD_LOW  # noqa: E402
from pts_clip_sampling import (complement_ranges, segment_candidates, dense_candidates,
                               fixed_quota_rotation, uniform_indices, tuple_digest,
                               extract_checked_frames, rgb112_from_jpeg)  # noqa: E402

HOLDOUT = "data/metadata/splits/EXP-016-development-holdout.json"
MANIFEST = "data/process/EXP-008-units-v2/unit-manifest.csv"
SRC = "data/process/EXP-008-units-src256"
EVENT_DIRS = {"dongchuan": "video1_segment", "video4": "video4_segment", "video5": "video5_segment"}
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refuse schema-less empty CSV: {path}")
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal_metadata(root, metadata):
    """Exclusive closure over the generated plan, including summary and candidates."""
    metadata = Path(metadata).resolve()
    metadata.relative_to(root / "data/metadata/EXP-016")
    destination = metadata / "frozen-plan-closure.json"
    if destination.exists():
        raise FileExistsError(destination)
    files = sorted(p for p in metadata.iterdir() if p.is_file())
    candidates = sorted((root / "data/process/EXP-016" / metadata.name / "candidates").glob("*.npz"))
    if len(candidates) != 150 or not (metadata / "summary.json").is_file():
        raise ValueError("Cannot seal an incomplete 150-video candidate plan")
    records = [dict(path=p.relative_to(root).as_posix(), size_bytes=p.stat().st_size, sha256=sha256(p))
               for p in files + candidates]
    closure = dict(format_version=1, metadata_path=metadata.relative_to(root).as_posix(),
                   scope="generated_metadata_and_candidate_arrays_no_pixels", files=records,
                   summary_sha256=sha256(metadata / "summary.json"))
    with destination.open("x", encoding="utf-8") as f:
        json.dump(closure, f, ensure_ascii=False, indent=2)
    return destination


def verify_metadata_closure(root, metadata):
    path = metadata / "frozen-plan-closure.json"
    closure = json.loads(path.read_text(encoding="utf-8"))
    if closure["metadata_path"] != metadata.relative_to(root).as_posix():
        raise ValueError("Frozen plan closure directory mismatch")
    for record in closure["files"]:
        item = (root / record["path"]).resolve()
        item.relative_to(root / "data")
        if item.stat().st_size != record["size_bytes"] or sha256(item) != record["sha256"]:
            raise ValueError(f"Frozen generated plan changed: {record['path']}")
    if sha256(metadata / "summary.json") != closure["summary_sha256"]:
        raise ValueError("Frozen summary changed")
    return sha256(path)


def load_tables(root):
    by_key, indices = defaultdict(list), defaultdict(list)
    inputs = {}
    for source, folder in EVENT_DIRS.items():
        events_path = root / f"data/process/video_frames/{folder}/events.csv"
        index_path = events_path.parent / "frames_index.csv"
        inputs[events_path.relative_to(root).as_posix()] = sha256(events_path)
        inputs[index_path.relative_to(root).as_posix()] = sha256(index_path)
        for line, row in enumerate(csv_rows(events_path), 2):
            if row["kind"] not in {"pos", "neg", "cut"}:
                raise ValueError(f"Unreviewed event row: {events_path}:{line}")
            row = dict(row, event_line=line, source=source, events_path=events_path.relative_to(root).as_posix())
            by_key[row["video_key"]].append(row)
        for row in csv_rows(index_path):
            indices[row["video_key"]].append(float(row["t_sec"]))
    return by_key, indices, inputs


def annotation_segments(events, pts, audit_times):
    """Unchanged explicit [s,e), plus independently recorded protocol-9 gaps."""
    explicit, covered = [], []
    for event in sorted(events, key=lambda row: (float(row["start_sec"]), float(row["end_sec"]))):
        a, b = float(event["start_sec"]), float(event["end_sec"])
        if not np.isfinite([a, b]).all() or b <= a:
            raise ValueError("Invalid event interval")
        if covered and a < covered[-1][1] - 1e-9:
            raise ValueError("Overlapping explicit events require review; no first-wins relabel")
        covered.append((a, b))
        if event["kind"] != "cut":
            explicit.append({"segment_id": f"event-line-{event['event_line']}", "kind": event["kind"],
                             "start_sec": a, "end_sec": b, "origin": "explicit_human_event",
                             "event_line": event["event_line"], "audit_frame_count": ""})
    gaps, cursor = [], 0.0
    for a, b in covered:
        if a > cursor:
            gaps.append((cursor, a, "internal_between_annotations"))
        cursor = max(cursor, b)
    if pts[-1] > cursor:
        gaps.append((cursor, float(pts[-1]), "after_final_annotation"))
    gap_records, implicit = [], []
    for i, (a, b, gap_type) in enumerate(gaps):
        # Original annotated endpoints stay claimed. No native-rate upgrade of
        # a gap with <=1 original audit observation.
        n_audit = sum(a < t < b if gap_type == "internal_between_annotations" else a < t <= b
                      for t in audit_times)
        eligible = n_audit >= 2 and b - a >= 2.0
        gap_records.append({"start_sec": a, "end_sec": b, "gap_type": gap_type,
                            "audit_frame_count": n_audit, "native_frame_count": int(((pts > a) & (pts < b)).sum()),
                            "disposition": "protocol9_implicit_negative" if eligible else "insufficient_audit_coverage_not_confirmed_negative"})
        if eligible:
            implicit.append({"segment_id": f"implicit-gap-{i}", "kind": "neg", "start_sec": a,
                             "end_sec": b, "origin": "protocol9_unannotated_not_human_negative",
                             "event_line": "", "audit_frame_count": n_audit})
    return explicit + implicit, gap_records


def load_unit(root, row, overrides):
    uid = row["unit_id"]
    coord_path = root / SRC / "coord" / f"{uid}.csv"
    coord = csv_rows(coord_path)
    pts = np.array([float(r["pts_time"]) for r in coord], dtype=np.float64)
    if [int(r["frame"]) for r in coord] != list(range(len(coord))) or not np.isfinite(pts).all():
        raise ValueError(f"{uid}: invalid coordinate frame sequence/timestamps")
    qp = root / overrides[uid] if uid in overrides else root / SRC / "qc" / f"{uid}.csv"
    if not qp.exists() and uid not in overrides:
        qp = root / "data/process/EXP-008-units-v2/qc" / f"{uid}.csv"
    qc = csv_rows(qp)
    if len(qc) != len(coord) or [int(r["frame"]) for r in qc] != list(range(len(coord))):
        raise ValueError(f"{uid}: QC/coord exact count/order mismatch {len(qc)}/{len(coord)}")
    qt = np.array([float(r["pts_time"]) for r in qc])
    stds = np.array([float(r["gray_std"]) for r in qc])
    if not np.isfinite(qt).all() or not np.isfinite(stds).all() or np.max(abs(qt - pts)) > 0.001001:
        raise ValueError(f"{uid}: QC/coord timestamp or finite-value mismatch")
    source = root / SRC / f"{uid}.mp4"
    if not source.is_file():
        raise ValueError(f"{uid}: missing H264 middleware")
    return pts, stds, source, coord_path, qp


def read_core_review(root):
    rel = "data/metadata/EXP-016/reviews/v1_09-supervision-core-20261030.json"
    path = root / rel
    if not path.exists():
        return None, {}
    review = json.loads(path.read_text(encoding="utf-8"))
    if review["decision"] != "approved_supervision_core" or review["scope"] != "supervision_core_verified":
        raise ValueError("Unrecognized supervision-core review")
    fingerprints = {rel: sha256(path)}
    for evidence in review["evidence"]:
        if sha256(root / evidence["path"]) != evidence["sha256"]:
            raise ValueError("Supervision-core evidence changed")
        fingerprints[evidence["path"]] = evidence["sha256"]
    return review, fingerprints


def run(args):
    root = args.root.resolve()
    split = json.loads((root / HOLDOUT).read_text(encoding="utf-8-sig"))
    if sha256(root / MANIFEST) != split["parent_manifest_sha256"]:
        raise ValueError("Frozen parent manifest fingerprint changed")
    if sha256(root / split["parent_folds"]) != split["parent_folds_sha256"]:
        raise ValueError("Frozen parent folds fingerprint changed")
    rows = csv_rows(root / MANIFEST)
    sides = {uid: "train" for uid in split["train_video_ids"]}
    for uid in split["validation_video_ids"]:
        if uid in sides:
            raise ValueError("Train/validation video overlap")
        sides[uid] = "validation"
    if len(rows) != 150 or set(sides) != {r["unit_id"] for r in rows} or Counter(sides.values()) != {"train": 119, "validation": 31}:
        raise ValueError("Expected exact frozen 150/119/31 source pool")
    train_groups = {r["group_id"] for r in rows if sides[r["unit_id"]] == "train"}
    val_groups = {r["group_id"] for r in rows if sides[r["unit_id"]] == "validation"}
    if train_groups & val_groups:
        raise ValueError("Session-group leakage")
    events, audit_times, fingerprints = load_tables(root)
    core_review, core_fingerprints = read_core_review(root)
    fingerprints.update(core_fingerprints)
    reuse_records = {}
    if args.reuse_success_run:
        reuse_path = (root / args.reuse_success_run).resolve()
        reuse_path.relative_to(root / "data/process/EXP-016")
        prior = json.loads(reuse_path.read_text(encoding="utf-8"))
        reuse_records = {r["unit_id"]: r for r in prior["records"] if r["status"] == "passed"}
        fingerprints[reuse_path.relative_to(root).as_posix()] = sha256(reuse_path)
    fingerprints[HOLDOUT], fingerprints[MANIFEST] = sha256(root / HOLDOUT), sha256(root / MANIFEST)
    for rel in ["experiments/common/video/build_units.py", "experiments/common/video/build_units_v2.py",
                "experiments/common/video/build_units_stride.py", ".agents/skills/data-usage-protocol/SKILL.md"]:
        fingerprints[rel] = sha256(root / rel)
    overrides = dict(item.split("=", 1) for item in args.qc_override)
    if set(overrides) - set(sides):
        raise ValueError("QC override outside source pool")
    if args.freeze:
        if Path(args.freeze).name != args.freeze:
            raise ValueError("Freeze name must be a single directory name")
        metadata = root / "data/metadata/EXP-016" / args.freeze
        process = root / "data/process/EXP-016" / args.freeze
        metadata.mkdir(parents=True, exist_ok=False)
        process.mkdir(parents=True, exist_ok=False)
        (process / "candidates").mkdir()
        windows_file = (metadata / "windows.jsonl").open("x", encoding="utf-8")
        dense_file = (metadata / "dense-eval-windows.jsonl").open("x", encoding="utf-8")
    else:
        metadata = process = windows_file = dense_file = None
    units, segment_rows, gaps_all, assignments, required = [], [], [], [], []
    failures, counts, quota_by_epoch = [], Counter(), [Counter() for _ in range(args.epochs)]
    dense_total = candidates_total = 0
    for row in rows:
        uid, side = row["unit_id"], sides[row["unit_id"]]
        try:
            pts, stds, source, coord_path, qc_path = load_unit(root, row, overrides)
        except (ValueError, FileNotFoundError) as error:
            failures.append({"unit_id": uid, "reason": str(error)})
            continue
        ev = events[uid]
        if any(e["source"] != row["source"] or e["video_id"] != row["video_id"] for e in ev):
            raise ValueError(f"{uid}: events source/video mapping conflict")
        cuts = [(float(e["start_sec"]), float(e["end_sec"])) for e in ev if e["kind"] == "cut"]
        forbidden = build_forbidden(uid, cuts, pts.tolist(), np.flatnonzero(stds < QC_STD_LOW).tolist())
        clean = complement_ranges(forbidden, len(pts))
        segments, gaps = annotation_segments(ev, pts, audit_times[uid])
        gaps_all.extend(dict(unit_id=uid, split=side, **gap) for gap in gaps)
        record = dict(unit_id=uid, split=side, source=row["source"], video_id=row["video_id"],
                      group_id=row["group_id"], video_label=row["label"], n_frames=len(pts),
                      middleware_path=source.relative_to(root).as_posix(), middleware_sha256=sha256(source) if args.freeze else "not_requested",
                      coord_path=coord_path.relative_to(root).as_posix(), coord_sha256=sha256(coord_path),
                      qc_path=qc_path.relative_to(root).as_posix(), qc_sha256=sha256(qc_path),
                      forbidden_ranges_json=json.dumps(forbidden), clean_runs_json=json.dumps(clean))
        record["supervision_timebase_status"] = "supervision_core_verified" if uid == "v1_09" and core_review else "existing_mapping_no_new_known_conflict" if uid != "v1_09" else "review_pending"
        record["reused_cache_evidence_dir"], record["reused_frames_written_sha256"] = "", ""
        record["reused_cache_directory"] = ""
        if uid == "v1_09" and core_review and args.freeze:
            for key in ["middleware_sha256", "coord_sha256", "qc_sha256"]:
                if record[key] != core_review[key]:
                    raise ValueError("Supervision-core reviewed input fingerprint changed")
        units.append(record)
        frame_set, source_candidates, own_label_found = set(), {}, False
        for segment in segments:
            segment["annotation_start_sec"], segment["annotation_end_sec"] = segment["start_sec"], segment["end_sec"]
            sampling_start, sampling_end = segment["start_sec"], segment["end_sec"]
            if uid == "v1_09" and core_review and segment["origin"] == "explicit_human_event":
                if (sampling_start, sampling_end) != (core_review["annotation_start_sec"], core_review["annotation_end_sec"]):
                    raise ValueError("Supervision-core review no longer matches unchanged annotation")
                sampling_start, sampling_end = core_review["sampling_start_sec"], core_review["sampling_end_sec"]
            if not segment["start_sec"] <= sampling_start < sampling_end <= segment["end_sec"]:
                raise ValueError("Sampling bounds would extend an annotation")
            segment["sampling_start_sec"], segment["sampling_end_sec"] = sampling_start, sampling_end
            segment["sampling_scope"] = "supervision_core_verified" if uid == "v1_09" and core_review else "original_annotation"
            candidate = segment_candidates(pts, clean, sampling_start, sampling_end)
            n = len(candidate)
            candidates_total += n
            source_candidates[segment["segment_id"]] = candidate
            k = min(4, n)
            counts[(side, segment["kind"], "segments")] += 1
            counts[(side, segment["kind"], "quota")] += k
            segment_rows.append(dict(unit_id=uid, split=side, group_id=row["group_id"], **segment,
                                     n_candidates=n, quota=k, candidates_sha256=tuple_digest(candidate),
                                     zero_reason="no_legal_3s_tuple_after_quality_and_label_checks" if not n else ""))
            own_label_found |= bool(n and segment["kind"] == row["label"])
            if side == "train":
                schedule, pools = fixed_quota_rotation(n, args.epochs)
                chosen = sorted({i for pool in pools for i in pool})
            else:
                selected = uniform_indices(n, k)
                schedule, chosen = np.array([selected]), selected.tolist()
            ids = {}
            for index in chosen:
                frames = candidate[index]
                clip_id = f"{uid}:{segment['segment_id']}:{tuple_digest(frames)[:20]}"
                ids[index] = clip_id
                frame_set.update(map(int, frames))
                if windows_file:
                    windows_file.write(json.dumps(dict(clip_id=clip_id, unit_id=uid, split=side, group_id=row["group_id"],
                                                       segment_id=segment["segment_id"], kind=segment["kind"], origin=segment["origin"],
                                                       candidate_index=index, frame_indices=frames.tolist(),
                                                       source_pts_seconds=pts[frames].tolist()), ensure_ascii=False) + "\n")
            for epoch, indexes in enumerate(schedule):
                if side == "train":
                    quota_by_epoch[epoch][segment["kind"]] += len(indexes)
                for index in indexes:
                    assignments.append(dict(split=side, epoch=epoch if side == "train" else -1,
                                            clip_id=ids[int(index)], unit_id=uid, segment_id=segment["segment_id"],
                                            kind=segment["kind"], group_id=row["group_id"]))
        if not own_label_found:
            failures.append({"unit_id": uid, "reason": "no_legal_input_matching_own_video_label"})
        if args.freeze:
            np.savez_compressed(process / "candidates" / f"{uid}.npz", **source_candidates)
        prior_frames = {}
        prior = reuse_records.get(uid)
        if prior:
            if any(prior[key] != record[key] for key in ["middleware_sha256", "coord_sha256", "qc_sha256"]):
                raise ValueError(f"{uid}: reuse source/coordinate/QC fingerprint differs")
            prior_csv = root / prior["evidence_dir"] / "frames-written.csv"
            prior_frames = {int(r["frame_index"]): r for r in csv_rows(prior_csv)}
            if set(prior_frames) != frame_set:
                raise ValueError(f"{uid}: new required set differs from successful prior cache")
            record["reused_cache_evidence_dir"] = prior["evidence_dir"]
            record["reused_frames_written_sha256"] = sha256(prior_csv)
            record["reused_cache_directory"] = prior["output_dir"]
        for fi in sorted(frame_set):
            path = (f"{record['reused_cache_directory']}/frame-{fi:08d}.jpg" if prior
                    else f"data/process/EXP-016/{args.freeze or 'UNFROZEN'}/frames/{uid}/frame-{fi:08d}.jpg")
            required.append(dict(unit_id=uid, frame_index=fi, source_pts_seconds=f"{pts[fi]:.9f}",
                                 cache_path=path, split=side,
                                 reused_jpeg_sha256=prior_frames[fi]["jpeg_sha256"] if prior else ""))
        if side == "validation":
            dense, skipped = dense_candidates(pts, clean)
            dense_total += len(dense)
            record["dense_count"] = len(dense)
            record["dense_skipped_runs_json"] = json.dumps(skipped)
            for target, frames in dense:
                if dense_file:
                    dense_file.write(json.dumps(dict(unit_id=uid, target_start_seconds=target,
                                                     frame_indices=frames.tolist(), source_pts_seconds=pts[frames].tolist())) + "\n")
        else:
            record["dense_count"], record["dense_skipped_runs_json"] = "", ""
    if windows_file:
        windows_file.close()
        dense_file.close()
    for epoch in quota_by_epoch:
        if epoch != quota_by_epoch[0]:
            raise ValueError("Epoch P/N quotas changed")
    p, n = quota_by_epoch[0]["pos"], quota_by_epoch[0]["neg"]
    if not p or not n:
        failures.append({"unit_id": "ALL", "reason": "empty_training_class"})
    summary = dict(status="blocked_metadata_incomplete" if failures else "metadata_frozen_pixels_not_ready" if args.freeze else "dry_run_pixels_not_ready",
                   videos_expected=150, videos_metadata_pass=len(units), failures=failures,
                   epochs=args.epochs, candidate_tuples=candidates_total, selected_unique_source_frames=len(required),
                   training_quota_pos=p, training_quota_neg=n, pos_weight=n / p if p else None,
                   fixed_validation_quota=sum(1 for a in assignments if a["split"] == "validation"),
                   dense_validation_metadata_windows=dense_total,
                   segment_counts_and_quotas={"/".join(key): value for key, value in counts.items()},
                   zero_candidate_segments=[f"{r['unit_id']}:{r['segment_id']}" for r in segment_rows if not r["n_candidates"]],
                   gaps_total=len(gaps_all), protocol9_implicit_gaps=[g for g in gaps_all if g["disposition"] == "protocol9_implicit_negative"],
                   preprocessing=dict(cache="256x256 JPEG quality95 BGR source pixels", runtime="JPEG decode BGR to RGB then INTER_LINEAR resize112", mean=MEAN, std=STD, augmentation="none"),
                   independent_test_sources_used=False, timebase_pixel_review="v1_09 supervision core [0,524) verified; full clock remains unresolved" if core_review else "v1_09 audit-to-middleware phase pending",
                   known_timebase_review_pending=[] if core_review else ["v1_09"],
                   full_timebase_mapping_unresolved=["v1_09"], counts_provisional_until_timebase_review=not bool(core_review),
                   supervision_core_review=core_review,
                   pixels_ready=False, training_ready=False,
                   cache_scope="supervised_train_and_checkpoint_selection_only",
                   dense_input_contract="stream validation H264 via recorded frame indices; per decoded BGR frame reproduce cv2 JPEG95 encode/decode in memory, then RGB INTER_LINEAR112 and original mean/std; no dense JPEG cache claimed ready",
                   metadata_fingerprints=fingerprints,
                   command_argv=sys.argv, common_sampling_sha256=sha256(REPO / "experiments/common/video/pts_clip_sampling.py"),
                   entry_sha256=sha256(Path(__file__)))
    if metadata:
        write_csv(metadata / "video-inputs.csv", units)
        write_csv(metadata / "segments.csv", segment_rows)
        write_csv(metadata / "unannotated-gaps.csv", gaps_all)
        write_csv(metadata / "epoch-assignments.csv", assignments)
        write_csv(metadata / "frames-required.csv", required)
        transfers = []
        for unit in units:
            if unit["split"] != "validation":
                continue
            for role, path_key, hash_key in [("dense_h264", "middleware_path", "middleware_sha256"),
                                             ("dense_coordinate", "coord_path", "coord_sha256"),
                                             ("dense_quality", "qc_path", "qc_sha256")]:
                rel = unit[path_key]
                transfers.append(dict(unit_id=unit["unit_id"], role=role, local_path=rel,
                                      remote_project_relative_path=rel, size_bytes=(root / rel).stat().st_size,
                                      sha256=unit[hash_key], transfer_rule="skip_only_if_remote_sha256_equal",
                                      purpose="required_for_selected_checkpoint_dense_evaluation_not_duplicated_locally"))
        write_csv(metadata / "dense-transfer-references.csv", transfers)
        (metadata / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        seal_metadata(root, metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def verify_reused_cache(root, unit, planned_rows, evidence):
    """Read verified JPEGs; no source-video decode, no JPEG copy/re-encoding."""
    evidence.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    prior_csv = root / unit["reused_cache_evidence_dir"] / "frames-written.csv"
    if sha256(prior_csv) != unit["reused_frames_written_sha256"]:
        raise ValueError("Successful cache evidence changed")
    prior = {int(r["frame_index"]): r for r in csv_rows(prior_csv)}
    if set(prior) != {int(r["frame_index"]) for r in planned_rows}:
        raise ValueError("Successful cache/new required frame set mismatch")
    records = []
    for planned in planned_rows:
        frame = int(planned["frame_index"])
        old = prior[frame]
        path = (root / planned["cache_path"]).resolve()
        path.relative_to(root / "data/process/EXP-016")
        data = path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != old["jpeg_sha256"] or actual_sha != planned["reused_jpeg_sha256"]:
            raise ValueError(f"Reused JPEG fingerprint mismatch: {path}")
        if len(data) != int(old["jpeg_size_bytes"]):
            raise ValueError(f"Reused JPEG byte count mismatch: {path}")
        rgb = rgb112_from_jpeg(data)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != old["rgb112_sha256"]:
            raise ValueError(f"Reused JPEG decoded pixels mismatch: {path}")
        records.append(dict(old, action="reused_successful_cache_verified_no_source_decode"))
    write_csv(evidence / "frames-written.csv", records)
    result = dict(status="passed", source_frames=int(unit["n_frames"]), selected_frames=len(records),
                  seconds=time.perf_counter() - start, jpeg_bytes=sum(int(r["jpeg_size_bytes"]) for r in records),
                  source_scan_reused=True, source_scan_evidence_dir=unit["reused_cache_evidence_dir"],
                  reused_frames_written_sha256=unit["reused_frames_written_sha256"],
                  preprocessing="unchanged JPEG bytes->decode->RGB->INTER_LINEAR112")
    (evidence / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def extract_frozen(args):
    root = args.root.resolve()
    metadata = (root / args.extract_from).resolve()
    metadata.relative_to(root / "data/metadata/EXP-016")
    closure_sha256 = verify_metadata_closure(root, metadata)
    summary = json.loads((metadata / "summary.json").read_text(encoding="utf-8"))
    if summary["failures"]:
        raise ValueError("Frozen metadata has unresolved structural failures")
    for rel, fingerprint in summary["metadata_fingerprints"].items():
        if sha256(root / rel) != fingerprint:
            raise ValueError(f"Frozen metadata input changed: {rel}")
    units = csv_rows(metadata / "video-inputs.csv")
    required = defaultdict(list)
    required_rows = defaultdict(list)
    for row in csv_rows(metadata / "frames-required.csv"):
        required[row["unit_id"]].append(int(row["frame_index"]))
        required_rows[row["unit_id"]].append(row)
    all_ids = {row["unit_id"] for row in units}
    excluded = {s for s in args.extract_exclude.split(",") if s}
    only = {args.extract_probe} if args.extract_probe else all_ids - excluded
    if (only | excluded) - all_ids:
        raise ValueError("Unknown extraction unit")
    pending = set(summary.get("known_timebase_review_pending", [])) & only
    if pending:
        raise ValueError(f"Timebase review still pending, extraction refused: {sorted(pending)}")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    evidence_root = root / "data/process/EXP-016" / metadata.name / "pixel-verification" / run_id
    evidence_root.mkdir(parents=True, exist_ok=False)
    records, failed = [], []

    def one(unit):
        uid = unit["unit_id"]
        pts, stds, source, coord, qc = load_unit(root, unit, {uid: unit["qc_path"]})
        for path, key in [(source, "middleware_sha256"), (coord, "coord_sha256"), (qc, "qc_sha256")]:
            if sha256(path) != unit[key]:
                raise ValueError(f"{uid}: frozen input fingerprint changed: {path}")
        output = (root / "data/process/EXP-016/probes" / f"pixel-pipe-{run_id}" / uid / "frames" if args.extract_probe
                  else (root / required_rows[uid][0]["cache_path"]).parent)
        output.resolve().relative_to(root / "data/process/EXP-016")
        evidence = evidence_root / uid
        reuse = bool(unit.get("reused_cache_evidence_dir")) and not args.extract_probe
        result = (verify_reused_cache(root, unit, required_rows[uid], evidence) if reuse
                  else extract_checked_frames(source, pts, stds, required[uid], output, evidence,
                                              args.ffmpeg.resolve(), QC_STD_LOW))
        actual = {int(p.stem.split("-")[-1]) for p in output.glob("frame-*.jpg")}
        if (not reuse and actual != set(required[uid])) or not set(required[uid]).issubset(actual):
            raise ValueError(f"{uid}: cache file set differs from frozen selected frame set")
        if sha256(source) != unit["middleware_sha256"]:
            raise ValueError(f"{uid}: source changed during extraction")
        result.update(unit_id=uid, output_dir=output.relative_to(root).as_posix(),
                      evidence_dir=evidence.relative_to(root).as_posix(),
                      middleware_sha256=unit["middleware_sha256"], coord_sha256=unit["coord_sha256"], qc_sha256=unit["qc_sha256"])
        print(f"PIXELS VERIFIED {uid}: {result['selected_frames']} frames; {result['seconds']:.2f}s", flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=1 if args.extract_probe else args.workers) as executor:
        selected_units = sorted((row for row in units if row["unit_id"] in only),
                                key=lambda row: bool(row.get("reused_cache_evidence_dir")))
        tasks = {executor.submit(one, row): row["unit_id"] for row in selected_units}
        for task in concurrent.futures.as_completed(tasks):
            try:
                records.append(task.result())
            except Exception as error:
                failed.append(dict(unit_id=tasks[task], reason=str(error)))
                print(f"PIXEL FAILURE {tasks[task]}: {error}", flush=True)
    result = dict(status="failed" if failed else "probe_passed" if args.extract_probe else "supervised_pixels_partial" if only != all_ids else "supervised_pixels_all_150_verified",
                  videos_expected_in_experiment=len(all_ids), videos_requested=len(only), videos_verified=len(records),
                  excluded_units=sorted(all_ids - only), failures=failed, records=records,
                  metadata_path=metadata.relative_to(root).as_posix(), metadata_summary_sha256=sha256(metadata / "summary.json"),
                  frozen_plan_closure_sha256=closure_sha256,
                  sampling_code_sha256=sha256(REPO / "experiments/common/video/pts_clip_sampling.py"),
                  entry_code_sha256=sha256(Path(__file__)), command_argv=sys.argv,
                  cache_scope="supervised_only; dense H264 references still required", training_ready=False)
    (evidence_root / "run-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--freeze", default="", help="Exclusive metadata/process subdirectory name; no pixels written")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--qc-override", action="append", default=[], metavar="UNIT=PATH")
    parser.add_argument("--extract-from", default="", help="Frozen EXP-016 metadata directory; explicit extraction mode")
    parser.add_argument("--extract-exclude", default="", help="Explicitly excluded pending units; output remains partial")
    parser.add_argument("--extract-probe", default="", metavar="UNIT", help="One-video probe in a separate probe cache")
    parser.add_argument("--ffmpeg", type=Path, default=Path("D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reuse-success-run", default="", help="Freeze references to genuinely successful per-video caches from a previous run")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        parser.error("epochs must be in 1..20")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be in 1..4")
    if args.extract_from:
        extract_frozen(args)
    else:
        if args.extract_probe or args.extract_exclude:
            parser.error("extraction options require --extract-from")
        run(args)


if __name__ == "__main__":
    main()
