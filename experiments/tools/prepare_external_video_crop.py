"""Stage one explicitly bounded external-video crop; dry run unless --execute.

Source frames are zero-based, half-open presentation-order coordinates. A prior
fingerprinted full-source audit supplies integer PTS; only the new derivative is
fully decoded for verification. This tool never promotes media or assigns truth.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.tools.prepare_external_video_segments import frame_times, probe, sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ffmpeg-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-frames-csv", type=Path, required=True)
    parser.add_argument("--source-frames-sha256", required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--source-summary-sha256", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True, help="Exclusive")
    parser.add_argument("--crop", type=int, nargs=4, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--thumbnail-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(2, 4), default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(path: Path) -> Path:
        result = (root / path).resolve()
        result.relative_to(root)
        return result

    source = resolve(args.source)
    frames_csv = resolve(args.source_frames_csv)
    source_summary = resolve(args.source_summary)
    target = resolve(args.output)
    metadata = resolve(args.metadata_dir)
    thumbs = resolve(args.thumbnail_dir)
    require(target != source, "Source and output must differ")
    require(not target.exists() and not metadata.exists(), "Existing output/provenance refused")
    start, end = args.start_frame, args.end_frame
    x, y, width, height = args.crop
    require(0 <= start < end and x >= 0 and y >= 0 and width > 0 and height > 0,
            "Invalid frame interval or crop")
    require(all(value % 2 == 0 for value in args.crop), "YUV420 crop coordinates must be even")
    ffmpeg = args.ffmpeg_dir.resolve() / "ffmpeg.exe"
    ffprobe = args.ffmpeg_dir.resolve() / "ffprobe.exe"
    filters = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,crop={width}:{height}:{x}:{y}"
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-n", "-threads", str(args.threads),
               "-i", str(source), "-map", "0:v:0", "-vf", filters, "-an", "-fps_mode", "passthrough",
               "-c:v", "libx264", "-crf", "10", "-preset", "veryfast", "-threads", str(args.threads),
               "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", "-movflags", "+faststart", str(target)]
    plan = dict(source_id=args.source_id, source_path=source.relative_to(root).as_posix(),
                output_path=target.relative_to(root).as_posix(), source_frame_start=start,
                source_frame_end_exclusive=end, frame_count=end-start, crop_xywh=args.crop,
                metadata_dir=metadata.relative_to(root).as_posix(), command=command,
                framewise_label="unassigned", test_split="unassigned", independent_event_increment=0,
                raw_move_performed=False, catalog_changed=False, training_or_scoring_used=False)
    if not args.execute:
        print(json.dumps(dict(execute=False, **plan), indent=2, ensure_ascii=False))
        return 0

    metadata.mkdir(parents=True, exist_ok=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(status="failed_or_partial", **plan)
    helper = ROOT / "experiments/tools/prepare_external_video_segments.py"
    try:
        require(sha256(source) == args.source_sha256, "Source SHA256 mismatch")
        require(sha256(frames_csv) == args.source_frames_sha256, "Prior frame evidence SHA256 mismatch")
        require(sha256(source_summary) == args.source_summary_sha256, "Prior summary SHA256 mismatch")
        old = json.loads(source_summary.read_text(encoding="utf-8"))
        require(old["source_id"] == args.source_id and
                old["original_sha256"] == old["source_sha256_after"] == args.source_sha256 and
                old["size_bytes"] == source.stat().st_size and old["pts_strictly_increasing"] and
                old["decoder_errors"] == [], "Prior source integrity audit is inconsistent")
        with frames_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == old["full_decoded_frames"] and
                [int(row["frame"]) for row in rows] == list(range(len(rows))), "Source frame coordinate gap")
        # Header-only source probe, with no -show_frames: do not repeat the source audit.
        source_info = probe(ffprobe, source)
        write_json(metadata / "source-container-probe.json", source_info)
        stream = source_info["streams"][0]
        require(all(stream[key] == old["container"]["streams"][0][key] for key in
                    ("width", "height", "avg_frame_rate", "time_base", "nb_frames")), "Source container changed")
        source_tb = Fraction(stream["time_base"])
        times = frame_times(dict(streams=[stream], frames=[dict(best_effort_timestamp=row["pts_ticks"]) for row in rows]))
        require(end <= len(times) and x + width <= stream["width"] and y + height <= stream["height"],
                "Frame bounds or crop exceed source")
        shutil.copyfile(Path(__file__), metadata / "generator-source.py")
        shutil.copyfile(helper, metadata / "reused-helper-source.py")
        fingerprints = dict(source_path=source.relative_to(root).as_posix(), source_sha256=args.source_sha256,
                            source_bytes=source.stat().st_size, source_frames_csv=frames_csv.relative_to(root).as_posix(),
                            source_frames_sha256=args.source_frames_sha256,
                            source_summary=source_summary.relative_to(root).as_posix(), source_summary_sha256=args.source_summary_sha256,
                            generator_sha256=sha256(Path(__file__)), helper_sha256=sha256(helper),
                            reused_functions=["sha256", "probe", "frame_times"], full_source_decode_reused=True)
        write_json(metadata / "source-fingerprints.json", fingerprints)
        write_json(metadata / "contract.json", dict(**plan, source_time_base=str(source_tb),
                   source_frame_count=len(times), source_dimensions=[stream["width"], stream["height"]],
                   recipe="frame trim; PTS-STARTPTS; crop without resize; libx264 CRF10 veryfast; no audio/forced-r/interpolation",
                   lossy_reencoding=True, pixel_bit_identity_claim=False, argv=sys.argv,
                   label_note="Candidate positive source context is not frozen framewise truth or a frozen test assignment"))
        write_json(metadata / "environment.json", dict(python=sys.version,
                   ffmpeg_version=subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()[0],
                   ffmpeg_sha256=sha256(ffmpeg), ffprobe_sha256=sha256(ffprobe)))
        write_json(metadata / "encode-command.json", command)
        with (metadata / "encode.log").open("xb") as log:
            encoded = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log, timeout=300)
        require(encoded.returncode == 0, "Encoding failed; see encode.log")
        # One complete decode of the derivative verifies every delivered frame and all streams.
        verify_command = [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-show_frames", "-of", "json", str(target)]
        write_json(metadata / "output-probe-command.json", verify_command)
        verified = subprocess.run(verify_command, capture_output=True, text=True, timeout=120)
        with (metadata / "output-probe-stderr.txt").open("x", encoding="utf-8") as handle:
            handle.write(verified.stderr)
        require(verified.returncode == 0 and not verified.stderr.strip(), "Derivative decode reported errors")
        info = json.loads(verified.stdout)
        write_json(metadata / "output-probe.json", info)
        require(len(info["streams"]) == 1 and info["streams"][0]["codec_type"] == "video", "Unexpected audio/non-video stream")
        require(all(frame["media_type"] == "video" and
                    (frame["width"], frame["height"], frame["pix_fmt"]) == (width, height, "yuv420p")
                    for frame in info["frames"]), "Per-frame dimensions/pixel format mismatch")
        actual = frame_times(info)
        expected = [value - times[start] for value in times[start:end]]
        require(len(actual) == end-start and actual == expected, "Frame loss/duplication or changed PTS")
        out_stream = info["streams"][0]
        require(out_stream["codec_name"] == "h264" and out_stream["time_base"] == "1/30000", "Unexpected output codec/time base")
        out_tb = Fraction(out_stream["time_base"])
        with (metadata / "frame-map.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_frame_index", "source_pts_ticks", "source_time_base", "source_pts_seconds",
                             "new_frame_index", "new_pts_ticks", "new_time_base", "new_pts_seconds", "relative_pts_equal_exact"])
            for index, timestamp in enumerate(actual):
                writer.writerow([start+index, rows[start+index]["pts_ticks"], str(source_tb), format(float(times[start+index]), ".12f"),
                                 index, info["frames"][index]["best_effort_timestamp"], str(out_tb), format(float(timestamp), ".12f"), True])
        thumbnail_records = []
        for index in (0, len(actual)//2, len(actual)-1):
            image = thumbs / f"output-f{index:04d}-source-f{start+index:04d}.jpg"
            require(not image.exists(), "Existing thumbnail refused")
            # Seek immediately before the exact requested frame; this decodes only a local GOP.
            seek = max(Fraction(0), actual[index] - Fraction(1, 1000))
            thumbnail_command = [str(ffmpeg), "-hide_banner", "-nostdin", "-n", "-v", "error", "-threads", str(args.threads),
                                 "-ss", format(float(seek), ".12f"), "-i", str(target), "-frames:v", "1", "-vf", "scale=820:390", str(image)]
            thumbnail = subprocess.run(thumbnail_command, capture_output=True, text=True, timeout=30)
            require(thumbnail.returncode == 0 and not thumbnail.stderr.strip() and image.is_file(), "Thumbnail extraction failed")
            thumbnail_records.append(dict(output_frame_index=index, source_frame_index=start+index,
                                          path=image.relative_to(root).as_posix(), command=thumbnail_command, sha256=sha256(image)))
        write_json(metadata / "thumbnail-extraction.json", thumbnail_records)
        require(sha256(source) == args.source_sha256, "Source changed during processing")
        last_duration = int(info["frames"][-1]["duration"]) * out_tb
        result.update(status="derivative_verified_pending_root_promotion", source_sha256_after=args.source_sha256,
                      output_sha256=sha256(target), output_size_bytes=target.stat().st_size,
                      output_frame_count=len(actual), dimensions_every_frame=[width, height], output_time_base=str(out_tb),
                      exact_relative_pts_match=True, source_pts_start_ticks=int(rows[start]["pts_ticks"]),
                      source_pts_end_exclusive_ticks=int(rows[end]["pts_ticks"]) if end < len(rows) else None,
                      source_time_base=str(source_tb), output_fps=out_stream["avg_frame_rate"],
                      endpoint_span_seconds=float(actual[-1]), last_frame_display_duration_seconds=float(last_duration),
                      frame_coverage_seconds=float(actual[-1] + last_duration),
                      container_video_duration_seconds=out_stream["duration"], container_format_duration_seconds=info["format"]["duration"],
                      audio_stream_count=0, lossy_reencoding=True, pixel_bit_identity_claim=False,
                      full_output_decode_count=1, full_source_decode_reused=True,
                      thumbnail_paths=[record["path"] for record in thumbnail_records])
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        result.update(elapsed_seconds=time.monotonic()-started)
        write_json(metadata / "summary.json", result)
        write_json(metadata / "output-manifest.json", [dict(path=file.relative_to(root).as_posix(),
                   size_bytes=file.stat().st_size, sha256=sha256(file)) for file in sorted(metadata.iterdir()) if file.is_file()])
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "derivative_verified_pending_root_promotion" else 1


if __name__ == "__main__":
    sys.exit(main())
