"""Encode one repeated-frame 10s preview per generated dry scene, entirely locally.

No model/provider requests are made. Generated reference images and their sidecars
are read only. Re-running skips videos matching the image SHA256 and parameters.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "data/metadata/synthetic-video/dry-scenes-20261108-v1/source-inventory.json"
DEFAULT_OUTPUT = ROOT / "data/process/synthetic-video/dry-scenes-20261108-v1"
PARAMETERS = {"duration_seconds": 10, "fps": 25, "width": 1280, "height": 720,
              "codec": "libx264", "profile": "high", "pixel_format": "yuv420p", "crf": 18,
              "geometry": "uniform contain with black padding; no stretching",
              "audio": False, "physical_motion": False, "unique_source_images": 1}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def find_ffmpeg(explicit=None):
    if explicit:
        return str(Path(explicit).resolve())
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    bundled = sorted((ROOT / ".venv/Lib/site-packages/imageio_ffmpeg/binaries").glob("ffmpeg*.exe"))
    if bundled:
        return str(bundled[0])
    raise FileNotFoundError("未找到 ffmpeg；请用 --ffmpeg 指定现有本地程序。")


def inspect_video(path, ffmpeg):
    """Decode without image caches; verify frame count, timing and unique frames."""
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
                             "-an", "-f", "framemd5", "-"], capture_output=True, text=True, check=True)
    timebase, dimensions, frames = None, None, []
    for line in result.stdout.splitlines():
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
        elif line.startswith("#dimensions 0:"):
            dimensions = [int(value) for value in line.split(":", 1)[1].strip().split("x")]
        elif line and not line.startswith("#"):
            fields = [value.strip() for value in line.split(",")]
            frames.append((int(fields[2]), int(fields[3]), fields[-1]))
    if not frames or timebase is None or dimensions is None:
        raise ValueError("ffmpeg 未返回完整的解码时序。")
    duration = float((frames[-1][0] + frames[-1][1] - frames[0][0]) * timebase)
    report = {"decoded_frames": len(frames), "decoded_unique_frames": len({frame[2] for frame in frames}),
              "duration_seconds": duration, "width": dimensions[0], "height": dimensions[1],
              "time_base": str(timebase), "size_bytes": Path(path).stat().st_size,
              "verification": "full ffmpeg framemd5 decode; no frame-image cache", "checked_at": now()}
    if (len(frames) != 250 or abs(duration - 10) > 0.001 or dimensions != [1280, 720]):
        raise ValueError("静态视频解码检查未通过：" + json.dumps(report))
    return report


def encode_preview(image, destination, ffmpeg):
    temporary = destination.with_name("static-10s.encoding.mp4")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "25",
               "-i", str(image), "-t", "10", "-vf",
               "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
               "-r", "25", "-an", "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
               "-crf", str(PARAMETERS["crf"]), "-profile:v", PARAMETERS["profile"],
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        technical = inspect_video(temporary, ffmpeg)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return technical, command


def build(inventory_path=DEFAULT_INVENTORY, output_directory=DEFAULT_OUTPUT, ffmpeg=None, *, root=ROOT):
    root, output_directory = Path(root), Path(output_directory)
    inventory = read_json(inventory_path)
    items = inventory.get("items", [])
    if not items:
        raise ValueError("场景清单没有 items，未进行编码。")
    ffmpeg = find_ffmpeg(ffmpeg)
    rows = []
    for item in items:
        image = root / item["output_path"]
        destination = image.parent / "static-10s.mp4"
        record_path = image.parent / "static-10s.json"
        row = {"index": item["index"], "scene_id": item["scene_id"], "camera": item["camera"],
               "image_path": item["output_path"], "video_path": destination.relative_to(root).as_posix(),
               "image_available": image.is_file(), "video_status": "waiting_for_image", "training_ready": False}
        if image.is_file():
            image_hash = sha256(image)
            previous = read_json(record_path)
            if (destination.is_file() and previous.get("source_sha256") == image_hash
                    and previous.get("parameters") == PARAMETERS
                    and previous.get("video_sha256") == sha256(destination)):
                row.update(video_status="complete", operation="skipped_matching_output", record=previous)
            else:
                try:
                    technical, command = encode_preview(image, destination, ffmpeg)
                    record = {"schema_version": 1, "created_at": now(), "scene_id": item["scene_id"],
                              "source_path": item["output_path"], "source_sha256": image_hash,
                              "parameters": PARAMETERS, "technical": technical, "command": command,
                              "video_sha256": sha256(destination), "video_generation_fee_cny": 0,
                              "meaning": "10 seconds of playback from one repeated source image; no observed physical time or motion",
                              "compression_note": "No source motion; lossy H264 compression may introduce small decoded-frame differences.",
                              "training_ready": False}
                    write_json(record_path, record)
                    row.update(video_status="complete", operation="encoded_locally", record=record)
                except (subprocess.CalledProcessError, OSError, ValueError) as exc:
                    row.update(video_status="failed", error=str(exc))
        rows.append(row)
        print(f'{item["index"]:02d} {item["camera"]}: {row["video_status"]}', flush=True)
    catalog = {"schema_version": 1, "updated_at": now(), "inventory_path": str(Path(inventory_path).resolve()),
               "parameters": PARAMETERS, "video_generation_fee_cny": 0, "training_ready": False,
               "summary": {"expected": len(rows), "image_available": sum(row["image_available"] for row in rows),
                           "video_complete": sum(row["video_status"] == "complete" for row in rows),
                           "video_failed": sum(row["video_status"] == "failed" for row in rows)}, "items": rows}
    write_json(output_directory / "catalog.json", catalog)
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--watch", action="store_true", help="Incrementally encode newly completed references")
    parser.add_argument("--interval", type=float, default=12)
    args = parser.parse_args(argv)
    while True:
        catalog = build(args.inventory, args.output_directory, args.ffmpeg)
        if not args.watch:
            return int(bool(catalog["summary"]["video_failed"]))
        time.sleep(max(2, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
