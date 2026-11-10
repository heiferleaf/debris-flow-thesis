"""Compare two FFmpeg decoders on the same original video without saving frames.

This diagnostic never changes source videos, historical intermediates or training
inputs. BGR frames are held in memory; commands, logs and summary JSON are saved
under .codex/logs. A clean exit does not establish that source content is intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/tools"))
from build_units_src256 import unit_video_rows  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(exe: Path, src: Path, seconds: float, destination: Path) -> dict:
    version = subprocess.run(
        [str(exe), "-version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True, timeout=15,
    ).stdout.splitlines()[0]
    # No input seek, corrupt-frame discarding, interpolation or output encoder.
    # FFmpeg 9 removed -vsync. Use the equivalent output passthrough setting
    # supported by each version, and record it in the exact command.
    version_match = re.search(r"ffmpeg version n?(\d+)", version)
    if not version_match:
        raise RuntimeError("Cannot determine decoder version for passthrough option")
    sync_option = (["-fps_mode", "passthrough"]
                   if int(version_match.group(1)) >= 5 else ["-vsync", "0"])
    command = [
        str(exe), "-nostdin", "-hide_banner", "-v", "info",
        "-threads", "1", "-i", str(src), "-t", str(seconds),
        "-map", "0:v:0", "-an", "-vf", "scale=256:256,showinfo",
        *sync_option, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    try:
        proc = subprocess.run(command, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write((exc.stderr or b"").decode("utf-8", errors="replace"))
        raise RuntimeError("Decoder timed out; partial stderr was preserved") from exc
    stderr = proc.stderr.decode("utf-8", errors="replace")
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(stderr)
    frame_bytes = 256 * 256 * 3
    if len(proc.stdout) % frame_bytes:
        raise RuntimeError("Decoder returned a partial BGR frame")
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 256, 256, 3)
    stds = np.array([
        float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std())
        for frame in frames
    ])
    # showinfo can log an additional frame before the output -t filter stops.
    # Keep only the PTS entries corresponding to actually delivered raw frames.
    parsed = re.findall(
        r"\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*(\S+)", stderr,
    )
    if [int(index) for index, _ in parsed] != list(range(len(parsed))):
        raise RuntimeError("showinfo frame numbers are not contiguous from zero")
    pts_all = [float(value) for _, value in parsed]
    if not all(np.isfinite(value) for value in pts_all):
        raise RuntimeError("Non-finite showinfo PTS")
    if len(pts_all) < len(frames):
        raise RuntimeError("Missing showinfo PTS for delivered frames")
    pts = pts_all[:len(frames)]
    warnings = [line for line in stderr.splitlines() if re.search(
        r"Could not|Error|corrupt|missing reference|Invalid|failed|failure",
        line, re.I,
    )]
    valid = proc.returncode == 0 and bool(len(frames))
    return {
        "version": version, "exe_sha256": sha256(exe), "command": command,
        "status": "decoded" if valid else "failed",
        "error": None if valid else "Nonzero decoder exit or no delivered frames; inspect stderr",
        "returncode": proc.returncode, "frames_delivered": len(frames),
        "raw_bgr_sha256": hashlib.sha256(proc.stdout).hexdigest(),
        "showinfo_frames": len(pts_all), "showinfo_tail_pts": pts_all[len(frames):],
        "gray_std_below_20": int((stds < 20).sum()) if valid else None,
        "gray_std_min": float(stds.min()) if len(stds) else None,
        "gray_std_median": float(np.median(stds)) if len(stds) else None,
        "pts_delivered": pts, "gray_std_per_frame": stds.tolist(),
        "matched_warning_count": len(warnings), "matched_warnings": warnings,
        "warning_count_scope": "Non-exhaustive text matches; not comparable as severity across versions",
        "stderr_path": str(destination),
        "note": "Grayscale threshold is a screening rule, not proof of damage.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", default="v1_110")
    parser.add_argument("--old", type=Path, default=Path(imageio_ffmpeg.get_ffmpeg_exe()))
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 30:
        parser.error("In-memory diagnostic duration must be in (0, 30] seconds")
    for exe in (args.old, args.new):
        if not exe.is_file():
            parser.error(f"Missing decoder executable: {exe}")
    source_map = dict(unit_video_rows())
    requested = [value.strip() for value in args.units.split(",") if value.strip()]
    if not requested or any(uid not in source_map for uid in requested):
        parser.error("Unknown or empty video IDs")
    if len(set(requested)) != len(requested):
        parser.error("Repeated video IDs are not allowed")
    args.out = args.out.resolve()
    try:
        args.out.relative_to((REPO / ".codex/logs").resolve())
    except ValueError:
        parser.error("Diagnostic output must stay inside .codex/logs")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f"{uid}{suffix}").exists() for uid in requested
           for suffix in (".json", "-old.log", "-new.log")):
        parser.error("Output exists; choose a new diagnostic directory")
    for uid in requested:
        src = source_map[uid]
        result = {
            "unit_id": uid, "source": str(src.resolve()),
            "source_size": src.stat().st_size, "source_sha256": sha256(src),
            "seconds": args.seconds, "saved_frames": False,
            "script_sha256": sha256(Path(__file__)),
            "comparison_scope": "Delivered frames before encoding, first source interval only",
        }
        for label, exe in (("old", args.old), ("new", args.new)):
            log = args.out / f"{uid}-{label}.log"
            try:
                result[label] = decode(exe, src, args.seconds, log)
            except Exception as exc:
                result[label] = {
                    "returncode": None, "frames_delivered": 0,
                    "exe": str(exe.resolve()), "error_type": type(exc).__name__,
                    "error": str(exc), "stderr_path": str(log) if log.exists() else None,
                }
        with (args.out / f"{uid}.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        summary = {key: {field: result[key].get(field) for field in (
            "version", "returncode", "frames_delivered", "gray_std_below_20", "matched_warning_count", "error",
        )} for key in ("old", "new")}
        print(json.dumps({"unit_id": uid, **summary}, ensure_ascii=False), flush=True)
        if any(result[key]["returncode"] != 0 or not result[key]["frames_delivered"]
               for key in ("old", "new")):
            raise RuntimeError("Decoder comparison failed; inspect saved logs")


if __name__ == "__main__":
    main()
