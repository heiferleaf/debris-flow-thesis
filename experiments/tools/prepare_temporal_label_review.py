"""Prepare all-video60-second review blocks and bounded batches of audit panels.

Only metadata and already indexed audit JPEGs are read. No video is opened or
decoded, and preparing a panel never changes the independent human ledger.

Examples (repository root):
  python experiments/tools/prepare_temporal_label_review.py
  python experiments/tools/prepare_temporal_label_review.py --offset 20 --limit 20
  python experiments/tools/prepare_temporal_label_review.py --units v1_16 --limit 20
  python experiments/tools/prepare_temporal_label_review.py --block-ids v1_16-B0018
  python experiments/tools/prepare_temporal_label_review.py --selftest
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the existing pure metadata reader used by the saved-transition audit.
# It preserves actual CSV record numbers through enumerate(...,2), and never
# reads a media file. Frame-field validation below follows load_frames_index's
# basename-only guard but preserves missing records for an honest coverage map.
from experiments.tools.prepare_saved_transition_review import Inputs  # noqa: E402

VERSION = "full-temporal-review-20261108-v1"
STATUS = "prepared_not_reviewed"
DEFAULT_SOURCES = "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
DEFAULT_DOMAIN = "data/metadata/EXP-028/full-coverage-plan-20261108-v1/per-unit.csv"
DEFAULT_RELEASE = "data/metadata/annotation-revisions/npt-20261108-evidence-working-v2/released-intervals.json"
DEFAULT_META = "data/metadata/annotation-review/full-temporal-20261108-v1"
DEFAULT_PANELS = ".codex/docs/.temp/full-temporal-20261108-v1"
LEDGER_FIELDS = ["block_id", "unit_id", "start_seconds", "end_seconds", "required_review", "status",
                 "reviewer", "reviewed_at", "viewed_slots", "decision", "notes"]
CELL_W, CELL_H, HEADER_H, ROW_H = 480, 270, 102, 354
EPS = 1e-9


def reference(path, root=ROOT):
    # Do not resolve a frame junction into an unrecorded external path.
    path = Path(path).absolute()
    try:
        return path.relative_to(Path(root).absolute()).as_posix()
    except ValueError:
        return str(path)


def rooted(path, root=ROOT):
    path = Path(path)
    return path if path.is_absolute() else root / path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def stable_digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def natural_key(value):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def write_json(path, value, *, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def temporal_blocks(duration):
    """Cover the whole known domain [0,maxPTS) without dropping the tail."""
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Known middleware maxPTS must be positive and finite")
    count = math.ceil(duration / 60.0 - 1e-12)
    return [(60.0 * i, min(duration, 60.0 * (i + 1))) for i in range(count)]


def frame_records(rows, folder, *, root=ROOT, allowed_units=None):
    """Parse the existing audit-index schema; stat files without reading JPEGs."""
    grouped = defaultdict(list)
    for record_number, row in enumerate(rows, 2):
        unit = row.get("video_key", "").strip()
        if allowed_units is not None and unit not in allowed_units:
            continue
        name = row.get("frame_file", "").strip().replace("\\", "/")
        if not name or "/" in name or ".." in name:
            raise ValueError("Unsafe audit frame basename in record " + str(record_number))
        timestamp = float(row["t_sec"])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid audit-index time")
        path = folder / "frames" / name
        grouped[unit].append({"unit_id": unit, "actual_pts_seconds": timestamp,
                              "actual_pts_text": row["t_sec"], "frame_file": name,
                              "path": reference(path, root), "file_exists": path.is_file(),
                              "index_csv_path": reference(folder / "frames_index.csv", root),
                              "index_csv_record": record_number,
                              "time_evidence": "existing_audit_index_t_sec; no_new_original_pixel_PTS_verification"})
    for unit, frames in grouped.items():
        frames.sort(key=lambda r: (r["actual_pts_seconds"], r["index_csv_record"]))
        names = [r["frame_file"] for r in frames]
        if len(set(names)) != len(names):
            raise ValueError("Repeated audit frame filename within " + unit)
    return grouped


def select_block_frames(start, end, indexed):
    """Three distinct existing PTS nearest the three targets, inside this block.

    The nearest-unused choice is made for each target, then the selected set is
    displayed in chronological order. This matters when all available points
    are clustered near one end of a block.
    Missing files and shortage of distinct points remain explicit empty slots.
    No borrowing from the next block, copied frames, or fabricated timestamps.
    """
    inside = [r for r in indexed if start <= r["actual_pts_seconds"] < end]
    available = [r for r in inside if r["file_exists"]]
    used_times, slots = set(), []
    for slot, fraction in enumerate((1 / 6, 1 / 2, 5 / 6), 1):
        target = start + (end - start) * fraction
        eligible = [r for r in available if r["actual_pts_seconds"] not in used_times]
        chosen = min(eligible, key=lambda r: (abs(r["actual_pts_seconds"] - target), r["actual_pts_seconds"], r["index_csv_record"])) if eligible else None
        if chosen is None:
            slots.append({"slot": slot, "target_fraction": fraction, "target_seconds": target,
                          "frame": None, "missing_reason": "no_existing_distinct_audit_frame_in_this_block"})
        else:
            used_times.add(chosen["actual_pts_seconds"])
            slots.append({"slot": slot, "target_fraction": fraction, "target_seconds": target,
                          "frame": chosen, "target_error_seconds": chosen["actual_pts_seconds"] - target,
                          "missing_reason": None})
    chronological = sorted((s['frame'] for s in slots if s['frame']),
                           key=lambda r: (r['actual_pts_seconds'], r['index_csv_record']))
    for i, slot in enumerate(slots):
        chosen = chronological[i] if i < len(chronological) else None
        slot['frame'] = chosen
        slot['missing_reason'] = None if chosen else 'no_existing_distinct_audit_frame_in_this_block'
        if chosen:
            slot['target_error_seconds'] = chosen['actual_pts_seconds'] - slot['target_seconds']
        else:
            slot.pop('target_error_seconds', None)
    return {"slots": slots, "indexed_frame_count_in_block": len(inside),
            "existing_frame_count_in_block": len(available),
            "indexed_missing_file_count_in_block": sum(not r["file_exists"] for r in inside),
            "selected_distinct_frame_count": sum(s["frame"] is not None for s in slots),
            "missing_slot_count": sum(s["frame"] is None for s in slots),
            "availability": "three_frames" if all(s["frame"] for s in slots) else "partial_frames" if available else "missing_frames"}


def evidence_reference(old, release_ref):
    """Keep the original evidence object; repository-relative paths take priority."""
    linked = {**old, "raw_reference": dict(old)}
    if "artifact" in old:
        value = str(old["artifact"]).replace("\\", "/")
        path = Path(value)
        linked["artifact"] = value if path.is_absolute() or value.startswith(("data/", ".codex/")) else str(Path(release_ref).parent / path).replace("\\", "/")
        linked["reference_schema"] = "historical_artifact_and_evidence_id"
    elif "path" in old and "audit_pts_sec" in old and "decision_id" in old:
        linked["reference_schema"] = "bounded_visual_decision_point_and_existing_frame"
    else:
        linked["reference_schema"] = "raw_prior_evidence_object_preserved"
    return linked


def label_coverage(start, end, labels, release_ref):
    intersections, gaps, evidence = [], [], []
    cursor = start
    for row in sorted(labels, key=lambda r: float(r["start_sec"])):
        left, right = max(start, float(row["start_sec"])), min(end, float(row["end_sec"]))
        if right <= left:
            continue
        if left < cursor - EPS:
            raise ValueError("Overlapping released intervals")
        if left > cursor:
            gaps.append([cursor, left])
        cursor = right
        intersections.append({"release_id": row["release_id"], "start_seconds": left, "end_seconds": right,
                              "duration_seconds": right - left, "released_state": row["released_state"],
                              "mask_three_class": row["mask_three_class"], "mask_existence": row["mask_existence"],
                              "release_path": release_ref})
        for old in row.get("evidence_refs", []):
            evidence.append({**evidence_reference(old, release_ref), "release_id": row["release_id"], "whole_block_review_satisfied": False})
    if cursor < end:
        gaps.append([cursor, end])
    return {"existing_label_intersections": intersections, "unlabelled_gaps_seconds": gaps,
            "existing_label_coverage_fraction": math.fsum(r["duration_seconds"] for r in intersections) / (end - start),
            "prior_sparse_evidence_refs": evidence,
            "prior_sparse_evidence_completes_this_review": False}


def label_tails(unit, duration, labels, release_ref):
    return [{"unit_id": unit, "release_id": r["release_id"], "known_domain_end_seconds": duration,
             "tail_start_seconds": max(duration, float(r["start_sec"])), "tail_end_seconds": float(r["end_sec"]),
             "released_state": r["released_state"], "status": "tail_extent_unverified", "release_path": release_ref}
            for r in labels if float(r["end_sec"]) > duration + EPS]


def ensure_review_ledger(path, blocks):
    """Create once; never rewrite any human review status during rendering."""
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {r["block_id"] for r in blocks}
        observed = [r["block_id"] for r in rows]
        if len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError("Human review ledger no longer matches this catalog; refused to overwrite")
        return "preserved_existing_review_ledger"
    rows = [{"block_id": b["block_id"], "unit_id": b["unit_id"], "start_seconds": b["start_seconds"],
             "end_seconds": b["end_seconds"], "required_review": True, "status": STATUS,
             "reviewer": "", "reviewed_at": "", "viewed_slots": "", "decision": "", "notes": ""} for b in blocks]
    write_csv(path, rows, LEDGER_FIELDS)
    return "created_all_prepared_not_reviewed"


def filter_blocks(blocks, units=None, block_ids=None, offset=0, limit=20):
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be nonnegative")
    selected = list(blocks)
    if units:
        unknown = set(units) - {b["unit_id"] for b in blocks}
        if unknown:
            raise ValueError("Unknown units: " + ",".join(sorted(unknown)))
        selected = [b for b in selected if b["unit_id"] in set(units)]
    if block_ids:
        unknown = set(block_ids) - {b["block_id"] for b in blocks}
        if unknown:
            raise ValueError("Unknown block ids: " + ",".join(sorted(unknown)))
        selected = [b for b in selected if b["block_id"] in set(block_ids)]
    return selected[offset:offset + limit]


def font(size):
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render_page(blocks, output, *, root=ROOT):
    """At most4 blocks by3 frames; fit each whole image within480x270."""
    if len(blocks) > 4 or not blocks:
        raise ValueError("One page must contain1..4 blocks")
    image = Image.new("RGB", (CELL_W * 3, HEADER_H + ROW_H * len(blocks)), "#edf0f3")
    draw = ImageDraw.Draw(image)
    title, medium, small = font(24), font(20), font(16)
    draw.text((14, 10), "全时间轴60秒分块抽样｜已有审计帧｜已准备、未审查", font=title, fill="#172b3a")
    draw.text((14, 45), "每块近1/6、1/2、5/6处取不同帧；t为现有索引时刻。稀疏抽样不证明块内每一时刻均已核实。", font=small, fill="#624c2e")
    draw.text((14, 72), "本图不显示原标签。可按元数据路径打开原JPEG；未生成新视频帧。", font=small, fill="#4c5b68")
    dimensions = []
    for row_number, block in enumerate(blocks):
        y = HEADER_H + row_number * ROW_H
        draw.rectangle((0, y, CELL_W * 3, y + 37), fill="#dce5eb")
        draw.text((12, y + 5), f"{block['block_id']}   {block['unit_id']}   [{block['start_seconds']:g}, {block['end_seconds']:g})s   {block['source_camera']}", font=medium, fill="#152c3a")
        for column, slot in enumerate(block["slots"]):
            x, frame_y = column * CELL_W, y + 39
            draw.rectangle((x, frame_y, x + CELL_W - 1, frame_y + CELL_H - 1), fill="#17232c")
            frame = slot["frame"]
            if frame is None:
                draw.text((x + 22, frame_y + 103), "缺少现有不同帧：未查看", font=medium, fill="#ffe4bc")
                actual = "实际t=缺失"
                source_size, fitted_size = None, None
            else:
                path = rooted(frame["path"], root)
                if not path.is_file():
                    raise FileNotFoundError("Catalog-selected audit frame disappeared: " + str(path))
                with Image.open(path) as original:
                    source_size = list(original.size)
                    thumb = ImageOps.contain(original.convert("RGB"), (CELL_W, CELL_H), Image.Resampling.LANCZOS)
                fitted_size = list(thumb.size)
                image.paste(thumb, (x + (CELL_W - thumb.width) // 2, frame_y + (CELL_H - thumb.height) // 2))
                actual = f"实际t={frame['actual_pts_text']}s"
            draw.text((x + 8, frame_y + CELL_H + 1), f"{actual}   目标={slot['target_seconds']:.2f}s", font=medium, fill="#172b3a")
            if frame is not None:
                draw.text((x + 8, frame_y + CELL_H + 28), f"索引记录{frame['index_csv_record']}  {frame['frame_file']}", font=small, fill="#4c5b68")
            dimensions.append({"block_id": block["block_id"], "slot": slot["slot"],
                               "source_image_dimensions": source_size, "display_dimensions": fitted_size})
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Refused to overwrite panel: " + str(output))
    image.save(output, "JPEG", quality=92, subsampling=0)
    return {"width": image.width, "height": image.height, "cell_image_bounds": [CELL_W, CELL_H],
            "displayed_frames": dimensions, "sha256": digest_bytes(output.read_bytes()), "size_bytes": output.stat().st_size}


def create_catalog(sources_path, domain_path, release_path, meta, *, root=ROOT):
    inputs = Inputs()
    sources, domains = inputs.csv(sources_path), inputs.csv(domain_path)
    releases = inputs.json(release_path)
    if releases.get("training_ready") is not False:
        raise ValueError("Expected an explicitly nontraining working release")
    by_unit, domain = defaultdict(list), {r["unit_id"]: r for r in domains}
    for row in releases["rows"]:
        by_unit[row["unit_id"]].append(row)
    units = {s["unit_id"] for s in sources}
    if len(units) != len(sources) or units != set(domain) or units != set(by_unit):
        raise ValueError("Sources, known-domain metadata and semantic release units disagree")
    grouped_sources = defaultdict(list)
    for source in sources:
        source_ids = {r["source_id"] for r in by_unit[source["unit_id"]]}
        if len(source_ids) != 1:
            raise ValueError("Ambiguous audit source root for " + source["unit_id"])
        grouped_sources[next(iter(source_ids))].append(source["unit_id"])
        if source["middleware_sha256"] != domain[source["unit_id"]]["physical_video_id"]:
            raise ValueError("Known PTS domain uses different frozen middleware identity")
    frames = {}
    for source_id, members in sorted(grouped_sources.items()):
        folder = rooted(source_id, root)
        frames.update(frame_records(inputs.csv(folder / "frames_index.csv"), folder, root=root, allowed_units=set(members)))
    # Hash only small source/label/index metadata; no image library byte scan.
    tool_data = Path(__file__).read_bytes()
    blocks, tails, unit_reports = [], [], []
    release_ref = reference(release_path, root)
    for source in sorted(sources, key=lambda s: natural_key(s["unit_id"])):
        uid = source["unit_id"]
        duration = float(domain[uid]["max_PTS_seconds"])
        camera = source["group_id"].split("|")[1]
        indexed = frames.get(uid, [])
        times = [r["actual_pts_seconds"] for r in indexed]
        local = []
        for ordinal, (start, end) in enumerate(temporal_blocks(duration), 1):
            lo, hi = bisect_left(times, start), bisect_left(times, end)
            block = {"block_id": f"{uid}-B{ordinal:04d}", "unit_id": uid, "source_camera": camera,
                     "block_ordinal": ordinal, "start_seconds": start, "end_seconds": end,
                     "duration_seconds": end - start, "known_media_domain_end_seconds": duration,
                     "time_domain_scope": "known_frozen_middleware_maxPTS; not_container_last_frame_display_duration",
                     "required_review": True, "status": STATUS, "status_scope": "immutable_preparation_only; authoritative_human_status_in_review-ledger.csv",
                     "training_ready": False, "review_ledger_block_id": f"{uid}-B{ordinal:04d}",
                     **select_block_frames(start, end, indexed[lo:hi]),
                     **label_coverage(start, end, by_unit[uid], release_ref)}
            local.append(block)
        blocks.extend(local)
        unit_tails = label_tails(uid, duration, by_unit[uid], release_ref)
        tails.extend(unit_tails)
        unit_reports.append({"unit_id": uid, "source_camera": camera, "known_media_domain_end_seconds": duration,
                             "domain_scope": "frozen_middleware_maxPTS_not_container_display_duration", "block_count": len(local),
                             "three_frame_blocks": sum(b["selected_distinct_frame_count"] == 3 for b in local),
                             "partially_selectable_blocks": sum(0 < b["selected_distinct_frame_count"] < 3 for b in local),
                             "missing_frame_blocks": sum(b["selected_distinct_frame_count"] == 0 for b in local),
                             "missing_frame_slots": sum(b["missing_slot_count"] for b in local),
                             "indexed_audit_frames": len(indexed), "existing_indexed_audit_frames": sum(r["file_exists"] for r in indexed),
                             "audit_index_points_beyond_known_domain": sum(r["actual_pts_seconds"] >= duration for r in indexed),
                             "tail_extent_unverified_count": len(unit_tails), "required_review": True,
                             "reviewed_blocks_at_preparation": 0})
    meta.mkdir(parents=True, exist_ok=False)
    with (meta / "coverage-blocks.jsonl").open("x", encoding="utf-8") as stream:
        for block in blocks:
            stream.write(json.dumps(block, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    write_csv(meta / "per-unit-coverage.csv", unit_reports, list(unit_reports[0]))
    write_csv(meta / "tail-extent-unverified.csv", tails,
              ["unit_id", "release_id", "known_domain_end_seconds", "tail_start_seconds", "tail_end_seconds", "released_state", "status", "release_path"])
    ledger_action = ensure_review_ledger(meta / "review-ledger.csv", blocks)
    counts = {"video_count": len(sources), "total_blocks": len(blocks),
              "three_frame_blocks": sum(b["selected_distinct_frame_count"] == 3 for b in blocks),
              "blocks_with_any_selected_frame": sum(b["selected_distinct_frame_count"] > 0 for b in blocks),
              "partially_selectable_blocks": sum(0 < b["selected_distinct_frame_count"] < 3 for b in blocks),
              "missing_frame_blocks": sum(b["selected_distinct_frame_count"] == 0 for b in blocks),
              "missing_frame_slots": sum(b["missing_slot_count"] for b in blocks),
              "indexed_missing_frame_files_in_known_domain": sum(b["indexed_missing_file_count_in_block"] for b in blocks),
              "estimated_all_blocks_pages_at_four_blocks_each": math.ceil(len(blocks) / 4),
              "label_tail_extent_unverified_rows": len(tails),
              "known_domain_total_seconds": math.fsum(float(r["max_PTS_seconds"]) for r in domains),
              "reviewed_blocks_at_preparation": 0}
    manifest = {"schema_version": 1, "version": VERSION, "created_utc": now(), "training_ready": False,
                "catalog_status": STATUS, "counts": counts, "catalog": reference(meta / "coverage-blocks.jsonl", root),
                "catalog_sha256": digest_bytes((meta / "coverage-blocks.jsonl").read_bytes()),
                "review_ledger": reference(meta / "review-ledger.csv", root),
                "review_ledger_policy": "create once, do not rewrite on subsequent preparation/render runs; human evidence/status separate from frame paths",
                "ledger_action_at_creation": ledger_action,
                "time_domain": "[0,frozen_middleware_maxPTS), every60 seconds with tail; container last-frame display duration unverified; label tails reported separately",
                "selection": "nearest unused distinct existing indexed t_sec to1/6,1/2,5/6 of each block; earlier time/index record wins ties; never borrow outside block",
                "no_full_temporal_certification": "Every known-time-domain block is registered, but three sparse frames never certify all instants. Old sparse evidence never completes this round's block review.",
                "input_fingerprints": list(inputs.fingerprints.values()),
                "tool": reference(Path(__file__), root), "tool_sha256": digest_bytes(tool_data),
                "constraints": {"source_JPEG_byte_scan": False, "video_decodes": 0, "training_runs": 0,
                                "paid_operations": 0, "label_changes": 0, "source_JPEG_copies": 0,
                                "max_blocks_per_panel": 4, "frames_per_block": 3, "max_frame_cell": [CELL_W, CELL_H]}}
    write_json(meta / "manifest.json", manifest)
    return manifest, blocks


def load_catalog(meta, sources_path, domain_path, release_path, *, root=ROOT):
    manifest = json.loads((meta / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError("Wrong catalog version")
    expected_paths = {reference(p, root) for p in (sources_path, domain_path, release_path)}
    records = {r["path"]: r for r in manifest["input_fingerprints"]}
    if not expected_paths <= set(records):
        raise ValueError("This catalog uses different source, domain or release paths")
    for path, row in records.items():
        if digest_bytes(rooted(path, root).read_bytes()) != row["sha256"]:
            raise ValueError("Frozen catalog input metadata changed; choose a new version: " + path)
    if digest_bytes((meta / "coverage-blocks.jsonl").read_bytes()) != manifest["catalog_sha256"]:
        raise ValueError("Coverage catalog changed")
    blocks = read_jsonl(meta / "coverage-blocks.jsonl")
    ensure_review_ledger(meta / "review-ledger.csv", blocks)
    return manifest, blocks


def generate_batch(manifest, blocks, meta, panels, *, units=None, block_ids=None, offset=0, limit=20, root=ROOT):
    selected = filter_blocks(blocks, units, block_ids, offset, limit)
    panel_index = meta / "panel-index.jsonl"
    existing = read_jsonl(panel_index) if panel_index.exists() else []
    by_id = {r["panel_id"]: r for r in existing}
    records, created, reused = [], 0, 0
    # An evidence-link-only repair may change catalog bytes without changing any
    # block, selected frame, or existing panel. Preserve that original identity.
    selection_catalog_sha = manifest.get("panel_selection_catalog_sha256", manifest["catalog_sha256"])
    for first in range(0, len(selected), 4):
        page_blocks = selected[first:first + 4]
        ids = [b["block_id"] for b in page_blocks]
        panel_id = stable_digest({"layout": VERSION, "catalog_sha256": selection_catalog_sha, "block_ids": ids})[:16]
        output = panels / f"panel-{ids[0]}--{panel_id}.jpg"
        if panel_id in by_id:
            record = by_id[panel_id]
            if not rooted(record["path"], root).is_file():
                raise FileNotFoundError("Previously prepared panel is missing; no silent replacement")
            reused += 1
        else:
            record = {"panel_id": panel_id, "path": reference(output, root), "block_ids": ids,
                      "preparation_status": STATUS, "review_completion_implied": False, "created_utc": now(),
                      **render_page(page_blocks, output, root=root)}
            with panel_index.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            by_id[panel_id] = record
            created += 1
        records.append(record)
    batch_id = stable_digest({"catalog": selection_catalog_sha, "blocks": [b["block_id"] for b in selected]})[:16]
    receipt = {"batch_id": batch_id, "prepared_utc": now(), "training_ready": False, "required_review": True,
               "status": STATUS, "selected_block_ids": [b["block_id"] for b in selected],
               "selected_block_count": len(selected), "panel_count": len(records),
               "newly_created_panels": created, "reused_panels": reused, "panels": records,
               "review_ledger_touched": False, "selectors": {"units": units, "block_ids": block_ids, "offset": offset, "limit": limit}}
    receipts = meta / "batches"
    receipts.mkdir(exist_ok=True)
    receipt_path = receipts / f"batch-{batch_id}.json"
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    all_records = read_jsonl(panel_index) if panel_index.exists() else []
    with (meta / "review-ledger.csv").open(encoding="utf-8-sig", newline="") as stream:
        review_status = Counter(r["status"] for r in csv.DictReader(stream))
    write_json(meta / "summary.json", {"version": VERSION, "updated_utc": now(), "training_ready": False,
                                     **{k: v for k, v in manifest["counts"].items() if k != "reviewed_blocks_at_preparation"},
                                     "historical_reviewed_blocks_at_preparation": manifest["counts"].get("reviewed_blocks_at_preparation", 0),
                                     "authoritative_review_status_counts": dict(review_status),
                                     "current_coarse_reviewed_blocks": review_status["coarse_reviewed"],
                                     "current_prepared_not_reviewed_blocks": review_status[STATUS],
                                     "current_review_counts_source": "review-ledger.csv (read only)",
                                     "prepared_unique_panels": len(all_records),
                                     "prepared_unique_blocks": len({b for r in all_records for b in r["block_ids"]}),
                                     "last_batch": reference(receipt_path, root),
                                     "last_batch_panel_paths": [r["path"] for r in records],
                                     "limits": manifest["no_full_temporal_certification"]}, exclusive=False)
    return receipt


def selftest():
    class Fixtures(unittest.TestCase):
        def test_clustered_available_frames_are_displayed_chronologically(self):
            frames = [{"actual_pts_seconds": t, "file_exists": True, "index_csv_record": i}
                      for i, t in enumerate((540.03, 542.03, 546.08), 2)]
            result = select_block_frames(540, 600, frames)
            self.assertEqual([s['frame']['actual_pts_seconds'] for s in result['slots']],
                             [540.03, 542.03, 546.08])
            self.assertEqual(result['missing_slot_count'], 0)

        def test_zero_anchor_tail_and_no_boundary_reset(self):
            self.assertEqual(temporal_blocks(143.4), [(0, 60), (60, 120), (120, 143.4)])
            self.assertEqual(temporal_blocks(60), [(0, 60)])

        def test_nearest_distinct_actual_times_with_tie(self):
            records = [{"actual_pts_seconds": t, "file_exists": True, "index_csv_record": i} for i, t in enumerate((9, 11, 30, 50), 2)]
            result = select_block_frames(0, 60, records)
            self.assertEqual([s["frame"]["actual_pts_seconds"] for s in result["slots"]], [9, 30, 50])
            self.assertEqual(result["missing_slot_count"], 0)

        def test_missing_files_and_short_tail_keep_empty_slots(self):
            missing = {"actual_pts_seconds": 10, "file_exists": False, "index_csv_record": 2}
            result = select_block_frames(0, 60, [missing])
            self.assertEqual((result["missing_slot_count"], result["indexed_missing_file_count_in_block"]), (3, 1))
            result = select_block_frames(120, 120.2, [{"actual_pts_seconds": 120.1, "file_exists": True, "index_csv_record": 2}])
            self.assertEqual(result["selected_distinct_frame_count"], 1)
            self.assertEqual(result["missing_slot_count"], 2)

        def test_label_gaps_and_unverified_tail_not_hidden(self):
            rows = [{"release_id": "a", "start_sec": 0, "end_sec": 20, "released_state": "N", "mask_three_class": True, "mask_existence": True, "evidence_refs": [{"artifact": "old.json", "evidence_id": "old1"}, {"path": "data/old.jpg", "audit_pts_sec": 10, "decision_id": "D001"}]},
                    {"release_id": "b", "start_sec": 30, "end_sec": 80, "released_state": "P", "mask_three_class": True, "mask_existence": True}]
            result = label_coverage(0, 60, rows, "release/released.json")
            self.assertEqual(result["unlabelled_gaps_seconds"], [[20, 30]])
            self.assertFalse(result["prior_sparse_evidence_refs"][0]["whole_block_review_satisfied"])
            self.assertEqual(label_tails("u", 60, rows, "r")[0]["tail_end_seconds"], 80)

        def test_incremental_selection_and_human_ledger_preserved(self):
            blocks = [{"block_id": "a1", "unit_id": "a", "start_seconds": 0, "end_seconds": 60}, {"block_id": "a2", "unit_id": "a", "start_seconds": 60, "end_seconds": 120}, {"block_id": "b1", "unit_id": "b", "start_seconds": 0, "end_seconds": 60}]
            self.assertEqual([r["block_id"] for r in filter_blocks(blocks, ["a"], offset=1, limit=1)], ["a2"])
            self.assertEqual([r["block_id"] for r in filter_blocks(blocks, block_ids=["b1"])], ["b1"])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "review.csv"
                ensure_review_ledger(path, blocks)
                content = path.read_text(encoding="utf-8-sig").replace(STATUS, "human_reviewed", 1)
                path.write_text(content, encoding="utf-8-sig")
                before = path.read_bytes()
                ensure_review_ledger(path, blocks)
                self.assertEqual(before, path.read_bytes())

        def test_panel_layout_and_incremental_reuse_without_rewrite(self):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                original = root / "source.jpg"
                Image.new("RGB", (1280, 720), "#846e4e").save(original)
                frames = [{"actual_pts_seconds": t, "actual_pts_text": str(t), "path": "source.jpg", "file_exists": True, "index_csv_record": i, "frame_file": "source.jpg"} for i, t in enumerate((10, 30, 50), 2)]
                block = {"block_id": "u-B0001", "unit_id": "u", "source_camera": "camera", "start_seconds": 0, "end_seconds": 60, **select_block_frames(0, 60, frames)}
                meta = root / "meta"
                meta.mkdir()
                ensure_review_ledger(meta / "review-ledger.csv", [block])
                ledger = (meta / "review-ledger.csv").read_bytes()
                manifest = {"catalog_sha256": "a" * 64, "counts": {}, "no_full_temporal_certification": "sparse only"}
                a = generate_batch(manifest, [block], meta, root / "panels", root=root)
                b = generate_batch(manifest, [block], meta, root / "panels", root=root)
                self.assertEqual((a["newly_created_panels"], b["reused_panels"]), (1, 1))
                self.assertEqual(len(read_jsonl(meta / "panel-index.jsonl")), 1)
                self.assertEqual(ledger, (meta / "review-ledger.csv").read_bytes())
                self.assertEqual(a["panels"][0]["width"], 1440)
                self.assertTrue(all(r["display_dimensions"] == [480, 270] for r in a["panels"][0]["displayed_frames"]))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--metadata-dir", default=DEFAULT_META)
    parser.add_argument("--panels-dir", default=DEFAULT_PANELS)
    parser.add_argument("--units", nargs="+")
    parser.add_argument("--block-ids", nargs="+")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20, help="Maximum blocks to render this call;0 means metadata only")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    sources, domain, release = (rooted(value) for value in (args.sources, args.domain, args.release))
    meta, panels = rooted(args.metadata_dir), rooted(args.panels_dir)
    if meta.exists():
        manifest, blocks = load_catalog(meta, sources, domain, release)
    else:
        manifest, blocks = create_catalog(sources, domain, release, meta)
    receipt = generate_batch(manifest, blocks, meta, panels, units=args.units, block_ids=args.block_ids, offset=args.offset, limit=args.limit)
    print(json.dumps({**manifest["counts"], "batch_block_count": receipt["selected_block_count"],
                      "batch_panel_count": receipt["panel_count"], "new_panels": receipt["newly_created_panels"],
                      "reused_panels": receipt["reused_panels"], "panel_paths": [p["path"] for p in receipt["panels"]]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
