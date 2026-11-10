"""Lay out all saved audit JPEGs in current T intervals, without decoding video.

All current transition rows are mapped to their source CSV fingerprints. Adjacent
rows in one video are merged for viewing; each row still gets its own interior
and three existing frames on either side. Images remain in the original cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
POINTER = ROOT / "data/metadata/annotation-revisions/current-review.json"
META = ROOT / "data/metadata/annotation-review/saved-transitions-20261108-v1"
TEMP = ROOT / ".codex/docs/.temp/saved-transitions-20261108-v1"
STATUS = "prepared_not_reviewed"
TITLE = "约2秒稀疏序列 非原帧率"


def ref(path: Path) -> str:
    # Preserve the workspace entry path when a cache directory is a junction.
    return path.absolute().relative_to(ROOT).as_posix()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Inputs:
    def __init__(self) -> None:
        self.fingerprints = {}

    def read(self, path: Path) -> bytes:
        data = path.read_bytes()
        self.fingerprints[ref(path)] = {"path": ref(path), "sha256": sha(data), "bytes": len(data)}
        return data

    def json(self, path: Path) -> dict:
        return json.loads(self.read(path).decode("utf-8-sig"))

    def csv(self, path: Path) -> list[dict]:
        return list(csv.DictReader(io.StringIO(self.read(path).decode("utf-8-sig"))))

    def unchanged(self) -> bool:
        return all(sha((ROOT / path).read_bytes()) == value["sha256"] for path, value in self.fingerprints.items())


def merge_rows(rows: list[dict]) -> list[list[dict]]:
    merged = []
    for row in sorted(rows, key=lambda r: (r["range_sec"][0], r["range_sec"][1])):
        if merged and row["range_sec"][0] <= max(r["range_sec"][1] for r in merged[-1]):
            merged[-1].append(row)
        else:
            merged.append([row])
    return merged


def prepare_selection(rows: list[dict], indexed: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    # Stat only the one video's indexed cache. Never walk/hash the raw-video library.
    available = [f for f in indexed if (ROOT / f["path"]).is_file()]
    roles = defaultdict(list)
    row_summaries = []
    for row in rows:
        a, b = row["range_sec"]
        inside = [f for f in available if a <= f["t_sec"] < b]
        before = [f for f in available if f["t_sec"] < a][-3:]
        after = [f for f in available if f["t_sec"] >= b][:3]
        for role, frames in (("inside_interval", inside), ("context_before", before), ("context_after", after)):
            for frame in frames:
                roles[frame["index_csv_record"]].append({"event_row_id": row["event_row_id"], "role": role})
        row_summaries.append({
            **row,
            "available_inside_count": len(inside),
            "context_before_index_records": [f["index_csv_record"] for f in before],
            "context_after_index_records": [f["index_csv_record"] for f in after],
            "inside_index_records": [f["index_csv_record"] for f in inside],
            "reviewed_frame_count": 0,
            "pending_review_inside_frame_count": len(inside),
            "status": STATUS,
        })
    selected = []
    for frame in available:
        if frame["index_csv_record"] in roles:
            selected.append({**frame, "roles": roles[frame["index_csv_record"]], "review_status": "not_reviewed"})
    missing = [f for f in indexed if any(a <= f["t_sec"] < b for a, b in (r["range_sec"] for r in rows)) and not (ROOT / f["path"]).is_file()]
    for i, frame in enumerate(selected):
        frame["sequence_index"] = i + 1
        frame["gap_from_previous_sec"] = None if i == 0 else round(frame["t_sec"] - selected[i - 1]["t_sec"], 9)
    return selected, row_summaries, missing


def contact_sheets(unit_id: str, frames: list[dict], sheet_dir: Path) -> list[dict]:
    sheet_dir.mkdir(parents=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    tiny = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    outputs = []
    for page, first in enumerate(range(0, len(frames), 16), start=1):
        selected = frames[first:first + 16]
        thumbnails = []
        for frame in selected:
            with Image.open(ROOT / frame["path"]) as image:
                frame["source_image_dimensions"] = list(image.size)
                # Every source image's long side is exactly 480 in the grid.
                factor = 480 / max(image.size)
                size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
                thumb = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            thumbnails.append(thumb)
        image_h = max(image.height for image in thumbnails)
        header_h, label_h, columns = 100, 69, 4
        rows = math.ceil(len(selected) / columns)
        cell_h = image_h + label_h
        sheet = Image.new("RGB", (1920, header_h + rows * cell_h), "#f4f5f7")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f"{unit_id}  |  {TITLE}  |  第{page}页 / 共{math.ceil(len(frames)/16)}页", fill="#172532", font=font)
        draw.text((12, 44), "按索引真实t_sec排序：从左到右、从上到下。实际时间间隔逐格保留；缺帧不补齐，不等于连续原视频。", fill="#775223", font=small)
        draw.text((12, 73), f"本页实际索引时刻 {selected[0]['t_sec_text']}–{selected[-1]['t_sec_text']}秒；图像仅引用已有审计JPEG。", fill="#44515e", font=tiny)
        for i, (frame, image) in enumerate(zip(selected, thumbnails)):
            x, y = (i % columns) * 480, header_h + (i // columns) * cell_h
            sheet.paste(image, (x + (480 - image.width) // 2, y + (image_h - image.height) // 2))
            gap = "首帧" if frame["gap_from_previous_sec"] is None else f"Δ={frame['gap_from_previous_sec']:g}s"
            draw.text((x + 6, y + image_h + 2), f"{unit_id} | t_sec={frame['t_sec_text']} | {gap}", fill="#172532", font=small)
            draw.text((x + 6, y + image_h + 28), frame["frame_file"], fill="#3e4d5c", font=tiny)
            draw.text((x + 6, y + image_h + 49), f"index CSV record {frame['index_csv_record']} | video index {frame['index_in_video']}", fill="#65717d", font=tiny)
            frame["contact_sheet_page"] = page
            frame["contact_sheet_panel"] = i + 1
            frame["contact_image_dimensions"] = list(image.size)
        output = sheet_dir / f"sheet-{page:02d}.jpg"
        sheet.save(output, "JPEG", quality=88, subsampling=0)
        outputs.append({
            "path": ref(output), "sha256": sha(output.read_bytes()), "bytes": output.stat().st_size,
            "page": page, "panel_count": len(selected), "columns": columns, "rows": rows,
            "width": sheet.width, "height": sheet.height,
            "sequence_first": first + 1, "sequence_last": first + len(selected),
            "time_range_sec": [selected[0]["t_sec"], selected[-1]["t_sec"]],
            "frame_index_records": [f["index_csv_record"] for f in selected], "status": STATUS,
        })
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    if META.exists() or TEMP.exists():
        raise SystemExit("Refusing to overwrite existing saved-transition review; choose a new version.")
    inputs = Inputs()
    pointer = inputs.json(POINTER)
    working = ROOT / pointer["working_annotations_dir"]
    sources = inputs.json(working / "sources.json")
    groups, indexes = defaultdict(list), {}
    for source_id, source in sources["sources"].items():
        events_path = working / source["annotations_csv"]
        events = inputs.csv(events_path)
        selected_rows = [(i, row) for i, row in enumerate(events, start=2) if row["kind"] == "transition"]
        if not selected_rows:
            continue
        index_path = ROOT / source["source_id"] / "frames_index.csv"
        by_video = defaultdict(list)
        for record_no, frame in enumerate(inputs.csv(index_path), start=2):
            key = frame["video_key"]
            by_video[key].append({
                "unit_id": key, "t_sec": float(frame["t_sec"]), "t_sec_text": frame["t_sec"],
                "frame_file": frame["frame_file"], "path": ref(index_path.parent / "frames" / frame["frame_file"]),
                "index_csv_path": ref(index_path), "index_csv_record": record_no,
                "index_in_video": len(by_video[key]),
                "frame_hash_status": "existing_cached_JPEG_referenced_not_rehashed",
            })
        for values in by_video.values():
            values.sort(key=lambda f: (f["t_sec"], f["index_csv_record"]))
        indexes.update(by_video)
        for record_no, row in selected_rows:
            unit = row["video_key"]
            groups[unit].append({
                "event_row_id": f"{source_id}:record-{record_no}", "unit_id": unit,
                "range_sec": [float(row["start_sec"]), float(row["end_sec"])],
                "range_convention": "start inclusive, end exclusive",
                "current_kind": row["kind"], "current_confidence": row["confidence"],
                "event_csv_path": ref(events_path), "event_csv_record": record_no,
                "event_csv_sha256": inputs.fingerprints[ref(events_path)]["sha256"],
                "declared_parent_annotation_csv": source.get("parent_annotation_csv"),
                "declared_parent_annotation_sha256": source.get("parent_annotation_sha256"),
                "origin_csv": source.get("origin_csv"), "origin_sha256": source.get("origin_sha256"),
                "raw_event_row": row,
            })
    if len(groups) != 10 or sum(len(rows) for rows in groups.values()) != 13:
        raise SystemExit("Current transition inventory differs from the authorized 10-video / 13-row task.")
    META.mkdir(parents=True)
    TEMP.mkdir(parents=True)
    units = []
    for unit in sorted(groups):
        for number, adjacent in enumerate(merge_rows(groups[unit]), start=1):
            review_id = unit if len(merge_rows(groups[unit])) == 1 else f"{unit}-R{number}"
            frames, mapped_rows, missing = prepare_selection(adjacent, indexes[unit])
            pages = contact_sheets(unit, frames, TEMP / review_id)
            inside_count = sum(any(role["role"] == "inside_interval" for role in frame["roles"]) for frame in frames)
            max_gap = max((frame["gap_from_previous_sec"] for frame in frames[1:]), default=0)
            record = {
                "review_unit_id": review_id, "unit_id": unit, "status": STATUS,
                "current_T_ranges_sec": [row["range_sec"] for row in adjacent],
                "merged_range_sec": [min(row["range_sec"][0] for row in adjacent), max(row["range_sec"][1] for row in adjacent)],
                "event_rows": mapped_rows, "frame_count": len(frames), "inside_unique_frame_count": inside_count,
                "context_only_frame_count": len(frames) - inside_count,
                "reviewed_frame_count": 0, "pending_review_frame_count": len(frames),
                "sheet_count": len(pages), "contact_sheets": pages, "maximum_gap_sec": max_gap,
                "time_range_sec": [frames[0]["t_sec"], frames[-1]["t_sec"]] if frames else None,
                "missing_inside_indexed_frames": missing, "frames": frames,
                "sampling_description": TITLE, "original_framerate_claimed": False,
                "review_note": "New contact sheets were prepared only. Prior sampled/reviewed intervals do not certify these individual frames.",
            }
            units.append(record)
            print(f"{review_id}: rows={len(adjacent)}, inside={inside_count}, context={len(frames)-inside_count}, sheets={len(pages)}, max_gap={max_gap:g}s", flush=True)
    total_bytes = sum(sheet["bytes"] for unit in units for sheet in unit["contact_sheets"])
    if total_bytes >= 80 * 1024 * 1024:
        raise SystemExit("Contact sheets exceeded the authorized 80 MiB temporary footprint.")
    if not inputs.unchanged():
        raise SystemExit("An input metadata file changed during preparation; do not use these sheets as a frozen review.")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": STATUS,
        "purpose": "All existing indexed audit JPEGs inside current T rows, plus three existing context frames per side; no video decoding or label writes.",
        "current_review_pointer": ref(POINTER), "current_working_annotations_dir": ref(working),
        "tool": ref(Path(__file__)), "tool_sha256": sha(Path(__file__).read_bytes()),
        "input_fingerprints": list(inputs.fingerprints.values()),
        "constraints": {"network_calls": 0, "paid_jobs": 0, "training_runs": 0, "video_decodes": 0, "jpeg_source_copies": 0, "label_changes": 0, "max_panels_per_sheet": 16, "contact_image_long_side": 480, "kind_displayed_on_image": False},
        "summary": {
            "video_count": len(groups), "event_row_count": sum(len(rows) for rows in groups.values()), "review_unit_count": len(units),
            "frame_count": sum(unit["frame_count"] for unit in units),
            "inside_unique_frame_count": sum(unit["inside_unique_frame_count"] for unit in units),
            "reviewed_frame_count": 0, "pending_review_frame_count": sum(unit["frame_count"] for unit in units),
            "sheet_count": sum(unit["sheet_count"] for unit in units), "temporary_sheet_bytes": total_bytes,
            "missing_inside_frame_count": sum(len(unit["missing_inside_indexed_frames"]) for unit in units),
        },
        "units": units,
    }
    write_json(META / "manifest.json", manifest)
    summaries = []
    for unit in units:
        summary = {k: unit[k] for k in ("review_unit_id", "unit_id", "status", "current_T_ranges_sec", "frame_count", "inside_unique_frame_count", "reviewed_frame_count", "pending_review_frame_count", "sheet_count", "maximum_gap_sec")}
        summary["contact_sheets"] = [sheet["path"] for sheet in unit["contact_sheets"]]
        summaries.append(summary)
    write_json(META / "summary.json", {"manifest": ref(META / "manifest.json"), "manifest_sha256": sha((META / "manifest.json").read_bytes()), **manifest["summary"], "units": summaries})
    print(json.dumps(manifest["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
