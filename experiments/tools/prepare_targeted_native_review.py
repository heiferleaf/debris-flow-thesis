"""Prepare bounded native-video review images; never change labels or infer classes.

Run from the repository root with its existing Python/Pillow environment. Existing
13 probes are reused by reference. Only the ten explicitly listed new windows
are decoded, with at most two ffmpeg processes and a five-second preroll.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OLD_MANIFEST = ROOT / "data/process/EXP-022/native-semantic-review-20261107-v1/manifest.json"
OUT = ROOT / "data/process/EXP-022/native-semantic-review-20261108-v2"
META = ROOT / "data/metadata/annotation-review/native-targets-20261108-v2"
SOURCES = ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
FFMPEG = ROOT / ".venv/lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe"
# The adjacent 5/6 continuation and disputed upstream interval are prepared first.
TARGETS = [
    ("v1_95", 1396, 1406),
    ("v1_96", 0, 10),
    ("v1_117", 0, 8),
    ("v1_117", 530, 538),
    ("v1_60", 0, 8),
    ("v1_65", 0, 8),
    ("v1_97", 1396, 1406),
    ("v4_10", 0, 8),
    ("v4_13", 0, 8),
    ("v4_16", 0, 8),
]
ALIGNMENT = (
    "Nominal container-relative seek coordinates only; burned OSD/image must be "
    "checked against audit before claiming millisecond annotation alignment."
)


def rel(path: Path) -> str:
    # raw media directories are junctions; preserve their repository entry path.
    return path.absolute().relative_to(ROOT).as_posix()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")]:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def image_records(paths: list[Path], start: float) -> list[dict]:
    result = []
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            width, height = im.size
        result.append({
            "sequence_index": i + 1,
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "width": width,
            "height": height,
            "nominal_container_seconds": start + i / 4,
            "nominal_time_is_not_verified_osd": True,
        })
    return result


def sheets(paths: list[Path], folder: Path, title: str, start: float, old: bool) -> list[dict]:
    """All frames in order; old probes 4x4, new probes in two readable pages."""
    if not paths:
        return []
    per_page = 16 if old or len(paths) <= 32 else 20
    cols, cell_w, label_h, header_h = 4, 960, 36, 90
    with Image.open(paths[0]) as first:
        cell_h = round(first.height * cell_w / first.width)
    result = []
    for page, offset in enumerate(range(0, len(paths), per_page), start=1):
        chunk = paths[offset:offset + per_page]
        rows = (per_page + cols - 1) // cols
        canvas = Image.new("RGB", (cols * cell_w, header_h + rows * (cell_h + label_h)), "#151a21")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), f"{title} | page {page} | continuous 4 fps | frame order left-to-right, top-to-bottom", fill="white", font=font(28))
        draw.text((16, 47), "Nominal container seconds; burned OSD / audit alignment remains UNVERIFIED", fill="#f5cb75", font=font(24))
        for j, path in enumerate(chunk):
            x = (j % cols) * cell_w
            y = header_h + (j // cols) * (cell_h + label_h)
            with Image.open(path) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
            seq = offset + j + 1
            draw.text((x + 10, y + cell_h + 3), f"#{seq:03d}   nominal t={start + (seq - 1) / 4:.2f}s   {path.name}", fill="white", font=font(23))
        target = folder / f"contact-{page:02d}.jpg"
        canvas.save(target, quality=94, subsampling=0)
        result.append({
            "path": rel(target), "sha256": sha(target), "bytes": target.stat().st_size,
            "page": page, "columns": cols, "rows": rows,
            "sequence_first": offset + 1, "sequence_last": offset + len(chunk),
            "width": canvas.width, "height": canvas.height,
            "cell_image_width": cell_w, "cell_image_height": cell_h,
        })
    return result


def date_matches(path: Path, expected: str) -> bool:
    y, m, d = [int(x) for x in expected.split("-")]
    return any(
        f"{y}年{m}月{d}日" in part or expected in part or f"{y}{m:02d}{d:02d}" in part
        for part in path.parts
    )


def annotate_log(record: dict) -> None:
    log_ref = record.get("showinfo_log") or record.get("original_showinfo_log")
    if not log_ref:
        return
    lines = (ROOT / log_ref["path"]).read_text(encoding="utf-8", errors="replace").splitlines()
    points = []
    for line in lines:
        match = re.search(r"\bn:\s*(\d+)\s+pts:\s*[-\d]+\s+pts_time:([\d.eE+-]+)", line)
        if match:
            points.append({"filter_index": int(match[1]), "filter_pts_seconds": float(match[2])})
    delta = record.get("output_seek_delta_seconds", 0)
    selected = [p for p in points if delta <= p["filter_pts_seconds"] < delta + record["requested_duration"]]
    record["showinfo_observations"] = {
        "logged_filter_frame_count_including_preroll": len(points),
        "logged_filter_pts_range_seconds": [points[0]["filter_pts_seconds"], points[-1]["filter_pts_seconds"]] if points else None,
        "output_seek_delta_seconds": delta,
        "selected_filter_count": len(selected),
        "selected_filter_pts_range_seconds": [selected[0]["filter_pts_seconds"], selected[-1]["filter_pts_seconds"]] if selected else None,
        "warning_lines": [line for line in lines if re.search(r"could not find ref|error|invalid|corrupt|concealing|non-existing|reference picture", line, flags=re.IGNORECASE)],
        "note": "showinfo is upstream of output -ss and includes discarded preroll. JPEG001 corresponds to the first selected filter PTS, not necessarily the first showinfo line. Selection uses the command's output seek delta; this does not verify burned OSD alignment or visual integrity.",
    }
    if len(selected) == len(record.get("frames", [])):
        for frame, point in zip(record["frames"], selected):
            frame["selected_filter_pts_seconds"] = point["filter_pts_seconds"]
            frame["selected_filter_index"] = point["filter_index"]


def locate(unit: str, source_rows: dict, mappings: dict, inventory: dict) -> dict:
    source = source_rows.get(unit)
    mapping = mappings.get(unit)
    if not source or not mapping:
        return {"status": "skipped_missing_metadata", "unit_id": unit}
    if source["video_id"] != mapping["video_id"] or source["group_id"].split("|")[-1] != mapping["date"]:
        return {"status": "skipped_metadata_conflict", "unit_id": unit, "source_row": source, "mapping_row": mapping}
    family = "v1" if unit.startswith("v1_") else "v4"
    candidates = [p for p in inventory[family] if p.stem == source["video_id"] and date_matches(p, mapping["date"])]
    evidence = {
        "source_table": rel(SOURCES), "source_table_sha256": sha(SOURCES),
        "source_row": {k: source[k] for k in ("unit_id", "video_id", "group_id")},
        "mapping_table": f"data/process/video_frames/video{'1' if family == 'v1' else '4'}_segment/mapping.csv",
        "mapping_row": mapping, "date_checked": mapping["date"],
        "matching_candidates": [rel(p) for p in candidates],
    }
    if len(candidates) != 1:
        return {"status": "skipped_missing_or_ambiguous_source", "unit_id": unit, "source_resolution": evidence}
    path = candidates[0]
    stat = path.stat()
    return {
        "unit_id": unit, "current_source": rel(path),
        "current_physical_source": str(path.resolve()),
        "source_stat": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "source_resolution": evidence,
        "source_fingerprint_scope": "Current local file path/stat; whole original video was not rehashed.",
        "historical_source_hash_claimed": False,
    }


def reuse_old(old: dict) -> list[dict]:
    result = []
    for probe in old["probes"]:
        unit, start = probe["unit_id"], probe["requested_seek_seconds"]
        old_dir = OLD_MANIFEST.parent / f"{unit}-{start}"
        paths = sorted(p for p in old_dir.glob("*.jpg") if re.fullmatch(r"\d{3}\.jpg", p.name))
        folder = OUT / f"reused-{unit}-{start}"
        folder.mkdir(parents=True, exist_ok=False)
        record = dict(probe)
        record.update({
            "probe_id": f"reused-{unit}-{start}", "kind": "reused_existing_probe",
            "reused_from": {"manifest": rel(OLD_MANIFEST), "manifest_sha256": sha(OLD_MANIFEST), "frame_directory": rel(old_dir)},
            "decode_executed_this_run": False,
            "original_command_preserved": probe["command"],
            "original_exit_code": probe["exit_code"],
            "frames": image_records(paths, start),
            "contact_sheets": sheets(paths, folder, f"REUSED {unit} [{start},{start + probe['requested_duration']})s", start, old=True),
            "status": "sampled_not_yet_visually_reviewed" if len(paths) == 16 else "reuse_incomplete_not_visually_reviewed",
            "alignment": ALIGNMENT,
        })
        log = old_dir / "ffmpeg.log"
        record["original_showinfo_log"] = {"path": rel(log), "sha256": sha(log)} if log.is_file() else None
        annotate_log(record)
        write_json(folder / "probe.json", record)
        result.append(record)
        print(f"REUSED {unit} {start}s: {len(paths)} JPEG, {len(record['contact_sheets'])} contact page", flush=True)
    return result


def decode_target(target: tuple, source: dict) -> dict:
    unit, start, end = target
    probe_id = f"{unit}-{start}-{end}"
    record = dict(source)
    record.update({
        "probe_id": probe_id, "kind": "new_bounded_native_probe",
        "requested_seek_seconds": start, "requested_end_seconds_exclusive": end,
        "requested_duration": end - start, "fps": 4,
        "expected_frame_count": (end - start) * 4,
        "alignment": ALIGNMENT, "visual_review_status": "not_reviewed",
    })
    if record.get("status", "").startswith("skipped_"):
        print(f"SKIPPED {probe_id}: {record['status']}", flush=True)
        return record
    folder = OUT / probe_id
    folder.mkdir(parents=True, exist_ok=False)
    preseek, delta = max(0, start - 5), min(5, start)
    cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "info", "-threads", "2"]
    if preseek:
        cmd += ["-ss", str(preseek)]
    cmd += ["-i", str(ROOT / record["current_source"])]
    if delta:
        cmd += ["-ss", str(delta)]
    cmd += [
        "-t", str(end - start), "-an",
        "-vf", "fps=4,scale=1280:1280:force_original_aspect_ratio=decrease,showinfo",
        "-frames:v", str((end - start) * 4), "-q:v", "2", "-vsync", "0", str(folder / "%03d.jpg"),
    ]
    record.update({
        "command": cmd, "input_preseek_seconds": preseek,
        "output_seek_delta_seconds": delta, "maximum_preroll_seconds": 5,
        "decode_executed_this_run": True,
    })
    log_path = folder / "ffmpeg-showinfo.log"
    began = datetime.now(timezone.utc)
    print(f"START {probe_id}: preseek={preseek}, delta={delta}, duration={end-start}s", flush=True)
    with log_path.open("wb") as log:
        try:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=180)
            record["exit_code"] = run.returncode
        except subprocess.TimeoutExpired:
            record["exit_code"] = None
            record["failure"] = "bounded_probe_timeout_180s"
    record["elapsed_seconds"] = round((datetime.now(timezone.utc) - began).total_seconds(), 3)
    paths = sorted(folder.glob("[0-9][0-9][0-9].jpg"))
    record["frames"] = image_records(paths, start)
    record["showinfo_log"] = {"path": rel(log_path), "sha256": sha(log_path), "bytes": log_path.stat().st_size}
    record["contact_sheets"] = sheets(paths, folder, f"{unit} [{start},{end})s", start, old=False)
    record["status"] = (
        "sampled_not_yet_visually_reviewed"
        if record["exit_code"] == 0 and len(paths) == record["expected_frame_count"]
        else "sample_failed_or_incomplete_not_reviewed"
    )
    annotate_log(record)
    write_json(folder / "probe.json", record)
    print(f"DONE {probe_id}: exit={record['exit_code']}, JPEG={len(paths)}, elapsed={record['elapsed_seconds']}s", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create exactly the fixed bounded review inventory.")
    parser.add_argument("--annotate-existing-logs", action="store_true", help="Read existing logs and record filter PTS selection; no video decoding.")
    args = parser.parse_args()
    if args.annotate_existing_logs:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        for probe in manifest["probes"]:
            annotate_log(probe)
            write_json(OUT / probe["probe_id"] / "probe.json", probe)
        manifest["log_annotation_tool_sha256"] = sha(Path(__file__))
        write_json(OUT / "manifest.json", manifest)
        meta = json.loads((META / "review-inputs.json").read_text(encoding="utf-8"))
        meta["manifest_sha256"] = sha(OUT / "manifest.json")
        write_json(META / "review-inputs.json", meta)
        print("Annotated existing logs only; no ffmpeg or image regeneration.")
        return 0
    if not args.execute:
        parser.print_help()
        return 0
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing review directory: {OUT}")
    if not FFMPEG.is_file():
        raise SystemExit(f"Existing local ffmpeg is missing: {FFMPEG}")
    old = json.loads(OLD_MANIFEST.read_text(encoding="utf-8-sig"))
    with SOURCES.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = {r["unit_id"]: r for r in csv.DictReader(handle)}
    mappings = {}
    for family in (1, 4):
        with (ROOT / f"data/process/video_frames/video{family}_segment/mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
            mappings.update({r["video_key"]: r for r in csv.DictReader(handle)})
    inventory = {
        "v1": [p for n in (1, 2) for p in (ROOT / f"data/raw/video_dataset_{n}").rglob("*.mp4")],
        "v4": list((ROOT / "data/raw/video_dataset_3").rglob("*.mp4")),
    }
    locations = {unit: locate(unit, source_rows, mappings, inventory) for unit, _, _ in TARGETS}
    OUT.mkdir(parents=True)
    META.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Reuse 13 existing 16-frame probes without decoding; decode only ten fixed short native windows; no training, scores, label writes or visual type conclusions.",
        "generated_by": rel(Path(__file__)), "generated_by_sha256": sha(Path(__file__)),
        "maximum_concurrent_ffmpeg": 2, "alignment": ALIGNMENT,
        "source_resolution": locations, "probes": [],
    }
    manifest["probes"] = reuse_old(old)
    write_json(OUT / "manifest.json", manifest)
    new_records = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(decode_target, target, locations[target[0]]): target for target in TARGETS}
        for future in as_completed(jobs):
            target = jobs[future]
            new_records[target] = future.result()
            manifest["probes"] = manifest["probes"][:len(old["probes"]) ] + [new_records[t] for t in TARGETS if t in new_records]
            write_json(OUT / "manifest.json", manifest)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["counts"] = {
        "reused_probes": len(old["probes"]), "new_probes": len(new_records),
        "jpeg_frames_reused": sum(len(r["frames"]) for r in manifest["probes"] if r["kind"] == "reused_existing_probe"),
        "jpeg_frames_new": sum(len(r.get("frames", [])) for r in new_records.values()),
        "contact_sheets": sum(len(r.get("contact_sheets", [])) for r in manifest["probes"]),
        "reviewed_probes": 0,
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(META / "review-inputs.json", {
        "manifest": rel(OUT / "manifest.json"), "manifest_sha256": sha(OUT / "manifest.json"),
        "counts": manifest["counts"], "alignment": ALIGNMENT,
        "status": "prepared_only_not_visually_reviewed", "training_runs": 0, "paid_jobs": 0,
        "label_changes": 0, "model_scores_generated": False,
    })
    with (META / "probe-index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probe_id", "kind", "unit_id", "nominal_start", "duration", "status", "frame_count", "contact_sheets_json"])
        for probe in manifest["probes"]:
            writer.writerow([probe["probe_id"], probe["kind"], probe["unit_id"], probe["requested_seek_seconds"], probe["requested_duration"], probe["status"], len(probe.get("frames", [])), json.dumps([s["path"] for s in probe.get("contact_sheets", [])])])
    print(json.dumps(manifest["counts"]), flush=True)
    return 0 if all(r["status"] == "sampled_not_yet_visually_reviewed" for r in new_records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
