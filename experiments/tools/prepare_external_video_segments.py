"""Reproduce the manually reviewed 2026-10-30 external-video derivatives.

Raw inputs are read only. Frame bounds are zero-based, half-open, and refer to
decoded source presentation order. Crops remove the publisher's composite
canvas without resizing the image content. These are candidates, not labels or
a frozen test split. A temporal edit is never joined into one continuous clip.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from fractions import Fraction
import hashlib
import json
import os
import re
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import stat
import subprocess
import time
from zipfile import ZipFile
import zlib


SOURCES = {
    "usgs-hpcc-20220803": "data/raw/web-research-20261029/positive/USGS_HPCC_20220803_FinalCut.mp4",
    "usgs-chalkcliffs-20160818": "data/raw/web-research-20261030/positive/USGS_ChalkCliffs_2016-08-18.mp4",
    "usgs-chalkcliffs-20170719": "data/raw/web-research-20261030/positive/USGS_ChalkCliffs_2017-07-19.mp4",
}

DOWNLOAD_IDS = {
    "usgs-hpcc-20220803": "USGS-HPCC-20220803",
    "usgs-chalkcliffs-20160818": "USGS-CC-2016-0818",
    "usgs-chalkcliffs-20170719": "USGS-CC-2017-0719",
}


def source_path(root: Path, sid: str) -> Path:
    """Prefer the uncut download archive after the authorized raw promotion."""
    archive = root / "data/raw/source-downloads-archive" / DOWNLOAD_IDS[sid] / Path(SOURCES[sid]).name
    return archive if archive.exists() else root / SOURCES[sid]


def active_relpath(sid: str, view: str, part: str) -> Path:
    if sid == "usgs-hpcc-20220803":
        return Path(SOURCES[sid])
    base = Path("data/raw/web-research-20261030/positive")
    if sid == "usgs-chalkcliffs-20160818":
        return base / f"USGS_ChalkCliffs_2016-08-18_{view.split('-')[0]}.mp4"
    if view == "cam01-zoom-left":
        return base / "references" / f"USGS_ChalkCliffs_2017-07-19_cam01_zoom_{part}.mp4"
    return base / f"USGS_ChalkCliffs_2017-07-19_cam01_{part}.mp4"


def extract_usu_archive(root: Path, ffprobe: Path, resume_verified: bool = False) -> None:
    """Safely extract the reviewed USU archive; never move or remove sources."""
    archive = root / "data/raw/source-downloads-archive/USU-21047541-ARCHIVE/Example_videos_USU_fixed_river_cameras.zip"
    if not archive.exists():
        archive = root / "data/raw/web-research-20261030/negative/Example_videos_USU_fixed_river_cameras.zip"
    expected_hash = "bc7d920f23619e54be060c254febbf425f2a1266afe7296f0940771fb3fc2181"
    if sha256(archive) != expected_hash:
        raise ValueError("USU ZIP fingerprint differs from reviewed download")
    destination = (root / "data/raw/web-research-20261030/negative/USU_fixed_river_cameras").resolve()
    data_root = (root / "data").resolve()
    destination.relative_to(data_root)
    inventory = root / "data/metadata/external-videos/usu-members-20261030.csv"
    if inventory.exists():
        raise FileExistsError(inventory)
    rows = []
    with ZipFile(archive) as z:
        planned = []
        targets = set()
        for member in z.infolist():
            name = member.filename
            posix = PurePosixPath(name.replace("\\", "/"))
            windows = PureWindowsPath(name)
            if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts or ":" in name:
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            if not posix.parts or posix.parts[0] != "Example_videos":
                raise ValueError(f"Unexpected ZIP root: {name!r}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"ZIP symlink refused: {name!r}")
            if member.is_dir():
                continue
            relative = Path(*posix.parts[1:])
            target = (destination / relative).resolve()
            target.relative_to(destination)
            target.relative_to(data_root)
            key = str(target).casefold()
            if key in targets or (target.exists() and not resume_verified):
                raise FileExistsError(f"Existing or duplicate ZIP target refused: {target}")
            targets.add(key)
            if relative.suffix.lower() == ".mp4":
                if len(relative.parts) != 2 or relative.parts[0] not in {"BSF", "First dam", "UWRL"}:
                    raise ValueError(f"Unexpected MP4 site path: {name}")
            elif relative.as_posix() != "Readme.txt":
                raise ValueError(f"Unexpected non-video ZIP member: {name}")
            planned.append((member, relative, target))
        if sum(p.suffix.lower() == ".mp4" for _, p, _ in planned) != 13:
            raise ValueError("Expected exactly thirteen MP4 ZIP members")
        for member, relative, target in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = 0
            fingerprint = hashlib.sha256()
            length = 0
            if target.exists():
                # Explicit recovery only: verify both complete byte streams and
                # never open an existing target for writing.
                with z.open(member) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        length += len(chunk)
                        checksum = zlib.crc32(chunk, checksum)
                        fingerprint.update(chunk)
            else:
                with z.open(member) as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        length += len(chunk)
                        checksum = zlib.crc32(chunk, checksum)
                        fingerprint.update(chunk)
            if length != member.file_size or checksum != member.CRC:
                raise ValueError(f"ZIP CRC/size mismatch: {member.filename}")
            if sha256(target) != fingerprint.hexdigest():
                raise ValueError(f"Extracted file SHA256 mismatch: {target}")
            # ZIP dates are metadata, often export dates rather than filming dates.
            timestamp = time.mktime(member.date_time + (0, 0, -1))
            os.utime(target, (timestamp, timestamp))
            row = {"download_source_id": "USU-21047541-ARCHIVE", "zip_relpath": archive.relative_to(root).as_posix(),
                   "zip_sha256": expected_hash, "zip_member": member.filename, "local_path": target.relative_to(root).as_posix(),
                   "size_bytes": length, "zip_crc32": f"{checksum:08x}", "sha256": fingerprint.hexdigest(),
                   "zip_member_datetime": "-".join(map(str, member.date_time[:3])) + " " + ":".join(map(str, member.date_time[3:])),
                   "crc_and_sha_verification": "passed", "media_type": "video" if target.suffix.lower() == ".mp4" else "readme"}
            print(f"EXTRACTED CRC/SHA VERIFIED {relative.as_posix()}", flush=True)
            rows.append(row)
    def verify_media(row: dict) -> dict:
        if row["media_type"] != "video":
            return row
        target = root / row["local_path"]
        info = probe(ffprobe, target, frames=True, allow_recorded_sei_warning=True)
        stream = info["streams"][0]
        frames = info["frames"]
        if not frames or len(frames) != int(stream["nb_frames"]):
            raise ValueError(f"Decoded frame count mismatch: {target}")
        timestamps = [int(frame["best_effort_timestamp"]) for frame in frames]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Nonmonotonic video PTS: {target}")
        qc = root / "data/process/external-videos/usu-fixed-river-cameras/qc"
        qc.mkdir(parents=True, exist_ok=True)
        (qc / f"{target.parent.name}--{target.stem}.probe.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        warnings = info.get("_recorded_decoder_warnings", [])
        row.update({"site_id": target.parent.name, "width": stream["width"], "height": stream["height"],
                    "fps": stream["avg_frame_rate"], "container_frames": stream["nb_frames"],
                    "duration_seconds": stream["duration"], "decode_verification": "full_decode_count_and_monotonic_pts_passed" + ("_with_recorded_SEI_warnings" if warnings else ""),
                    "decoder_warning_count": len(warnings), "decoder_warnings_json": json.dumps(warnings)})
        print(f"MEDIA VERIFIED {target.parent.name}/{target.name}; frames={len(frames)}; SEI warnings={len(warnings)}", flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(verify_media, rows))
    if sha256(archive) != expected_hash:
        raise ValueError("USU ZIP changed during extraction")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with inventory.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"INVENTORY {inventory}", flush=True)

# sid, view, temporal block, start, end, crop(x,y,w,h), quality, overlay, duplicate
SEGMENTS = [
    ("usgs-hpcc-20220803", "cam01", "part01", 149, 4582, (0, 0, 1920, 1080),
     "rain_droplets_blur_difficult_candidate", "top-right location/date caption retained", ""),
    ("usgs-chalkcliffs-20160818", "cam01-left", "part01", 150, 3904, (32, 34, 828, 466),
     "visible_flow_sensor_obstruction_left", "publisher canvas and logo excluded by crop", ""),
    ("usgs-chalkcliffs-20160818", "cam02-right", "part01", 150, 3904, (886, 34, 366, 652),
     "visible_flow_distant_narrow_view", "publisher canvas and logo excluded by crop", ""),
    ("usgs-chalkcliffs-20170719", "cam01-original-right", "part01", 150, 1853, (886, 28, 368, 664),
     "narrow_view_guide_overlay_candidate", "white zoom-region guide retained; animates during intro", ""),
    ("usgs-chalkcliffs-20170719", "cam01-zoom-left", "part01", 211, 1853, (38, 30, 816, 458),
     "digitally_enlarged_same_camera_reference_only", "publisher canvas, logo and outer white border excluded by crop", "cam01-original-right/part01"),
    ("usgs-chalkcliffs-20170719", "cam01-original-right", "part02", 1913, 3162, (886, 28, 368, 664),
     "narrow_view_guide_overlay_candidate", "white zoom-region guide retained", ""),
    ("usgs-chalkcliffs-20170719", "cam01-zoom-left", "part02", 1913, 3162, (38, 30, 816, 458),
     "digitally_enlarged_same_camera_reference_only", "publisher canvas, logo and outer white border excluded by crop", "cam01-original-right/part02"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(ffprobe: Path, path: Path, frames: bool = False, allow_recorded_sei_warning: bool = False) -> dict:
    fields = "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames:format=duration,size"
    if frames:
        fields += ":frame=best_effort_timestamp,duration"
    command = [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries", fields, "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    warnings = [line for line in result.stderr.splitlines() if line.strip()]
    if warnings:
        if not allow_recorded_sei_warning or any(not re.fullmatch(r"\[h264 @ [^\]]+\] SEI type 5 size \d+ truncated at \d+", line) for line in warnings):
            raise RuntimeError(f"Decoder reported errors for {path}: {result.stderr}")
        info["_recorded_decoder_warnings"] = warnings
    return info


def frame_times(info: dict) -> list[Fraction]:
    stream = info["streams"][0]
    tb = Fraction(stream["time_base"])
    times = [int(f["best_effort_timestamp"]) * tb for f in info["frames"]]
    step = 1 / Fraction(stream["avg_frame_rate"])
    assert len(times) == int(stream["nb_frames"]), "Frame count mismatch"
    assert times and times[0] == 0, "Unexpected source start"
    assert all(b - a == step for a, b in zip(times, times[1:])), "Nonuniform source timestamps need separate review"
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="Create derivatives; default prints the reviewed plan")
    parser.add_argument("--extract-usu-zip", action="store_true", help="Extract and fully verify thirteen reviewed USU videos")
    parser.add_argument("--resume-usu-verified", action="store_true", help="Resume interrupted extraction by verifying existing full files byte-for-byte; never overwrite")
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/external-videos/segment-preparation-20261030.csv"))
    args = parser.parse_args()
    root = args.root.resolve()
    ffmpeg = args.ffmpeg_dir.resolve() / "ffmpeg.exe"
    ffprobe = args.ffmpeg_dir.resolve() / "ffprobe.exe"
    if args.extract_usu_zip:
        extract_usu_archive(root, ffprobe, args.resume_usu_verified)
        return
    if not args.execute:
        print(json.dumps(SEGMENTS, ensure_ascii=False, indent=2))
        return
    manifest = root / args.manifest
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing provenance: {manifest}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for sid, rel in SOURCES.items():
        path = source_path(root, sid)
        info = probe(ffprobe, path, frames=True)
        times = frame_times(info)
        qc = root / "data/process/external-videos" / sid / "qc"
        qc.mkdir(parents=True, exist_ok=True)
        (qc / "source-frames-probe.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        inputs[sid] = (path, info, times, sha256(path))

    def convert(spec: tuple) -> dict:
        sid, view, part, start, end, crop, quality, overlay, duplicate = spec
        source, source_info, times, source_hash = inputs[sid]
        stream = source_info["streams"][0]
        step = 1 / Fraction(stream["avg_frame_rate"])
        assert 0 <= start < end <= len(times)
        x, y, w, h = crop
        assert all(v % 2 == 0 for v in crop), "yuv420 crop must be even"
        assert x + w <= stream["width"] and y + h <= stream["height"]
        name = f"{sid}_{view}_{part}_f{start:06d}-{end:06d}"
        output = root / active_relpath(sid, view, part)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite derivative: {output}")
        filters = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y}"
        command = [str(ffmpeg), "-hide_banner", "-nostdin", "-n", "-i", str(source),
                   "-map", "0:v:0", "-vf", filters, "-an", "-fps_mode", "passthrough",
                   "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-threads", "4",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        qc = root / "data/process/external-videos" / sid / "qc"
        (qc / f"{name}.command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True)
        (qc / f"{name}.encode.log").write_text(result.stderr, encoding="utf-8")
        result.check_returncode()
        derived_info = probe(ffprobe, output, frames=True)
        derived_times = frame_times(derived_info)
        assert len(derived_times) == end - start, "Dropped or duplicated frames"
        assert derived_times == [t - times[start] for t in times[start:end]], "Presentation timestamp changed"
        derived_stream = derived_info["streams"][0]
        assert (derived_stream["width"], derived_stream["height"]) == (w, h)
        (qc / f"{name}.probe.json").write_text(json.dumps(derived_info, indent=2), encoding="utf-8")
        endpoint = times[end] if end < len(times) else times[-1] + step
        print(f"VERIFIED {name}: {end-start} frames, {float(endpoint-times[start]):.6f}s", flush=True)
        event = sid + (":event-date-unverified" if sid.endswith("20170719") else ":event01")
        return {
            "source_id": sid, "original_event_id": event, "camera_id": "cam01" if sid.endswith("20170719") else view.split("-")[0],
            "view_id": view, "temporal_block_id": part,
            "source_relpath": source.relative_to(root).as_posix(), "derived_relpath": output.relative_to(root).as_posix(),
            "start_frame": start, "end_frame_exclusive": end, "source_pts_start_seconds": f"{float(times[start]):.9f}",
            "source_pts_end_exclusive_seconds": f"{float(endpoint):.9f}",
            "source_time_base": stream["time_base"], "source_pts_start_ticks": int(times[start] / Fraction(stream["time_base"])),
            "source_pts_end_exclusive_ticks": int(endpoint / Fraction(stream["time_base"])),
            "fps": stream["avg_frame_rate"], "source_width": stream["width"], "source_height": stream["height"],
            "crop_x": x, "crop_y": y, "crop_width": w, "crop_height": h,
            "derived_width": w, "derived_height": h, "derived_frame_count": len(derived_times),
            "derived_duration_seconds": f"{float(len(derived_times)*step):.9f}",
            "status": "candidate", "label_status": "candidate_positive_source_framewise_labels_unreviewed",
            "split_status": "unassigned_not_frozen_test", "quality": quality,
            "overlay_notes": overlay, "same_camera_duplicate_of": duplicate,
            "count_as_independent_event": "no_derivative_of_original_event",
            "continuity_notes": "do_not_join_across_part01_part02_editorial_black_fade" if sid.endswith("20170719") else "single_continuous_view_within_segment",
            "source_date_notes": "in_frame_July14_2017_filename_July19_2017_unresolved" if sid.endswith("20170719") else "",
            "source_sha256": source_hash, "derived_sha256": sha256(output),
            "command_json": json.dumps(command, ensure_ascii=False), "verification": "full_decode_frame_count_and_all_pts_and_dimensions_passed",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(convert, SEGMENTS))
    for sid, (source, _, _, original_hash) in inputs.items():
        assert sha256(source) == original_hash, f"Raw source changed: {sid}"
    with manifest.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"MANIFEST {manifest}", flush=True)


if __name__ == "__main__":
    main()
