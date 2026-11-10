"""One-source native Las Lomas technical audit; no labels or dataset promotion."""
from pathlib import Path
import csv
import json
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tools import audit_external_video_inputs as common

SOURCE = ROOT / "data/raw/web-research-20261102/followup/USGS_LasLomas_20220901.mp4"
EXPECTED_SHA256 = "086ce5e2a358e8b69b71c96f8a37c0bdf2bd5b581d16f1a86536f92f32d708a1"
EXPECTED_BYTES = 92535889
OUTPUT = ROOT / "data/process/external-videos/las-lomas-native-audit-v1"
HISTORY = ROOT / "data/metadata/external-videos/followup-20261102-v1"


def main():
    import cv2
    import numpy as np

    began = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    before = common.sha(SOURCE)
    if before != EXPECTED_SHA256 or SOURCE.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Source fingerprint differs from the authorized original")
    read_only_inputs = [HISTORY / name for name in
                        ["acquisition.json", "probe.json", "registration.json"]]
    read_only_inputs += [common.SELECTION,
                        ROOT / "data/metadata/external-videos/catalog.csv",
                        ROOT / "data/metadata/external-videos/source-downloads.csv"]
    input_hashes = {common.rel(p): common.sha(p) for p in read_only_inputs}
    code_paths = [Path(__file__).resolve(), Path(common.__file__).resolve(),
                  ROOT / "experiments/common/video/pts_clip_sampling.py"]
    common.write(OUTPUT / "contract.json", dict(
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        invocation=sys.argv, source_id="USGS-LAS-LOMAS-VIDEO-20220901",
        source_path=common.rel(SOURCE), source_sha256=before,
        source_size_bytes=SOURCE.stat().st_size,
        inherited_source_page="https://www.usgs.gov/media/videos/las-lomas-debris-flow",
        inherited_license="Public Domain; inherited from acquisition.json, no new web check",
        inherited_webpage_duration_seconds=68,
        duration_discrepancy_status="Retain webpage 68 s vs original video-track 59.192467 s; unresolved",
        original_records_read_only_sha256=input_hashes,
        code_sha256={common.rel(p): common.sha(p) for p in code_paths},
        decode_budget=dict(full_sequential_passes=1, timeout_seconds=240,
                           decoder_threads=2, filter_threads=1, benchmark_passes=0),
        quality_rules=dict(native_luma_statistics="FFmpeg showinfo native Y-plane mean/std; rounded log values",
                           low_texture="std < 20", nearly_uniform="std < 2",
                           dark="std < 2 and mean < 20",
                           bright="std < 2 and mean > 235",
                           gray="std < 2 and 20 <= mean <= 235",
                           scene_change="FFmpeg scene_score > 0.1",
                           pts_gap="delta > max(0.5 seconds, 5 * positive-delta median)",
                           inherited_from=common.rel(Path(common.__file__).resolve()),
                           interpretation="Screening candidates only; no damaged-frame ground truth or content exclusion"),
        scope="Technical native decode/PTS/numeric quality only; no semantic judgement, labels, training, scoring, split/test freeze, selection update, or catalog edit"))
    version_command = [str(common.FFMPEG), "-version"]
    version = subprocess.run(version_command, capture_output=True, text=True, check=True, timeout=15)
    common.write(OUTPUT / "environment.json", dict(
        python_executable=sys.executable, python_version=sys.version,
        platform=platform.platform(), numpy_version=np.__version__, opencv_version=cv2.__version__,
        ffmpeg_version_command=version_command, ffmpeg_version_stdout=version.stdout,
        ffmpeg_version_stderr=version.stderr,
        ffmpeg_path=str(common.FFMPEG), ffmpeg_sha256=common.sha(common.FFMPEG),
        ffprobe_path=str(common.FFPROBE), ffprobe_sha256=common.sha(common.FFPROBE)))
    command = [str(common.FFPROBE), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(SOURCE)]
    common.write(OUTPUT / "probe-command.json", command)
    probe = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    (OUTPUT / "probe-stderr.log").write_text(probe.stderr, encoding="utf-8")
    info = json.loads(probe.stdout)
    common.write(OUTPUT / "container-probe.json", info)
    if probe.stderr.strip():
        raise ValueError("Probe reported diagnostic errors; raw stderr retained")
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    expected = int(stream["nb_frames"])
    if expected != 1774 or stream["duration"] != "59.192467":
        raise ValueError("Probe differs from inherited source properties; preserve before investigating")
    log, elapsed = common.decode(SOURCE, OUTPUT, timeout=240)
    rows, errors, warnings = common.parse_log(log, stream)
    diagnostic = common.describe(rows)
    # Only retain native technical statistics; no prospective model-input grids.
    diagnostic.pop("container_clock_sampling")
    gap_limit = diagnostic["gap_candidate_threshold_seconds"]
    for i, row in enumerate(rows):
        delta = float(row["pts_seconds"]) - float(rows[i-1]["pts_seconds"]) if i else None
        mean, std = row["native_luma_mean"], row["native_luma_std"]
        row.update(delta_pts_seconds="" if delta is None else format(delta, ".12f"),
                   pts_duplicate_candidate=int(delta == 0) if delta is not None else 0,
                   pts_backward_candidate=int(delta < 0) if delta is not None else 0,
                   pts_gap_candidate=int(delta > gap_limit) if delta is not None else 0,
                   low_texture_candidate_std_below20=int(std < 20),
                   nearly_uniform_candidate_std_below2=int(std < 2),
                   dark_low_texture_candidate=int(std < 2 and mean < 20),
                   bright_low_texture_candidate=int(std < 2 and mean > 235),
                   gray_low_texture_candidate=int(std < 2 and 20 <= mean <= 235),
                   scene_change_candidate=int(row["scene_score"] > .1))
    with (OUTPUT / "frame-pts-quality.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    after = common.sha(SOURCE)
    inputs_after = {common.rel(p): common.sha(p) for p in read_only_inputs}
    matched = len(rows) == expected
    passed = matched and not errors and before == after
    summary = dict(
        status="native_technical_audit_completed" if passed else "native_technical_audit_failed",
        source_id="USGS-LAS-LOMAS-VIDEO-20220901", source_path=common.rel(SOURCE),
        source_sha256_before=before, source_sha256_after=after, source_unchanged=before == after,
        source_size_bytes=SOURCE.stat().st_size,
        native_dimensions=[stream["width"], stream["height"]],
        codec=stream["codec_name"], pix_fmt=stream["pix_fmt"],
        frame_rate=stream["avg_frame_rate"], time_base=stream["time_base"],
        container_reported_video_frames=expected, decoded_count_matches_container=matched,
        video_track_duration_seconds=float(stream["duration"]),
        webpage_duration_seconds_inherited=68,
        duration_discrepancy_unresolved=True,
        explicit_frame_or_time_limit_applied=False,
        full_native_eof_evidence=dict(completed_without_decode_limit=True,
            decoder_exit_code=0, decoded_all_container_reported_frames=matched,
            decoder_errors_found=len(errors),
            conclusion="Unbounded input pass terminated normally at native end; no temporal truncation or frame-count limit was used"),
        decode_seconds=elapsed, decoder_errors=errors, decoder_warnings=warnings,
        original_records_sha256_before=input_hashes,
        original_records_sha256_after=inputs_after,
        original_records_unchanged=input_hashes == inputs_after,
        quality_candidates_are_not_damage_truth=True,
        full_semantic_review_performed=False, fixed_camera_full_film_claimed=False,
        labels_assigned=False, final_usable_judgement_made=False,
        selected_18_modified=False, formal_test_frozen=False, model_scores_read=False,
        source_media_rewritten=False, decoded_images_saved=0,
        wall_seconds=time.monotonic()-began,
        **diagnostic)
    common.write(OUTPUT / "machine-summary.json", summary)
    common.write(OUTPUT / "manifest-sha256.json", {
        p.name: common.sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()})
    print(json.dumps({k: summary[k] for k in
                     ["status", "decoded_frames", "native_dimensions", "pts_strictly_increasing",
                      "source_unchanged", "original_records_unchanged", "decode_seconds"]}, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
