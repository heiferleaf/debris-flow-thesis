"""Read-only, full sequential external-media diagnostics; no label/QC masks."""
from pathlib import Path
from fractions import Fraction
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.pts_clip_sampling import dense_candidates

META = ROOT / "data/metadata/external-videos/external-input-audit-20261102-v1"
PROCESS = ROOT / "data/process/external-videos/external-input-audit-20261102-v1"
SELECTION = ROOT / "data/metadata/external-videos/label-review-20261101-v2/selection.json"
CONFIRMATION = SELECTION.with_name("silverado-human-confirmation-20261102.json")
FFMPEG = Path("D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe")
FFPROBE = FFMPEG.with_name("ffprobe.exe")
FRAME = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+\d.eE]+).*?s:(\d+)x(\d+).*?mean:\[([^\]]+)\]\s+stdev:\[([^\]]+)\]")
MOVDIR = "data/raw/web-research-20261102/review/ChalkCliffs2019_Firehose_Flood/"
MOVS = [
    ("USGS-CC-2019-0508-REVIEW-PART01", "050819_firehose_flood.MOV", "daa7148a6c0832ba49ec17c387121bb8a25a35408e4d73f171727e79981aec1f", 28575),
    ("USGS-CC-2019-0508-REVIEW-PART02", "050819_firehose_flood_pt2.MOV", "7c92cbdf27503ad24f6214d1ad6af3c3351475ab891ef1539a065bf5cc8ad11e", 25005),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def ranges(mask):
    indices = np.flatnonzero(mask)
    return [[int(x[0]), int(x[-1]) + 1] for x in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(x)]


def describe(rows):
    pts = np.array([float(r["pts_seconds"]) for r in rows])
    means = np.array([float(r["native_luma_mean"]) for r in rows])
    stds = np.array([float(r["native_luma_std"]) for r in rows])
    scenes = np.array([float(r["scene_score"]) for r in rows])
    if not len(rows) or not all(np.isfinite(v).all() for v in [pts, means, stds, scenes]):
        raise ValueError("Missing/nonfinite full-frame diagnostics")
    delta = np.diff(pts)
    positive = delta[delta > 0]
    median = float(np.median(positive)) if len(positive) else None
    gap_limit = max(.5, 5 * median) if median is not None else .5
    # Numerical clock interruptions only. Quality/cut candidates are NOT masks.
    breaks = np.flatnonzero((delta <= 0) | (delta > gap_limit)) + 1
    bounds = [0] + breaks.tolist() + [len(rows)]
    runs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    windows, skipped = dense_candidates(pts, runs)
    return dict(decoded_frames=len(rows), first_pts_seconds=float(pts[0]), last_pts_seconds=float(pts[-1]),
                container_pts_span_seconds=float(pts[-1] - pts[0]), pts_strictly_increasing=bool(np.all(delta > 0)),
                duplicate_pts_after_frames=np.flatnonzero(delta == 0).tolist(),
                backward_pts_after_frames=np.flatnonzero(delta < 0).tolist(),
                positive_delta_median_seconds=median, gap_candidate_threshold_seconds=gap_limit,
                gap_candidates=[dict(after_frame=int(i), delta_seconds=float(delta[i])) for i in np.flatnonzero(delta > gap_limit)],
                native_luma_mean_range=[float(means.min()), float(means.max())],
                native_luma_std_range=[float(stds.min()), float(stds.max())],
                low_texture_std_below20_ranges=ranges(stds < 20),
                nearly_uniform_std_below2_ranges=ranges(stds < 2),
                dark_low_texture_candidates=ranges((stds < 2) & (means < 20)),
                bright_low_texture_candidates=ranges((stds < 2) & (means > 235)),
                gray_low_texture_candidates=ranges((stds < 2) & (means >= 20) & (means <= 235)),
                scene_score_max=float(scenes.max()),
                cut_candidates=[dict(frame=int(i), pts_seconds=float(pts[i]), scene_score=float(scenes[i])) for i in np.flatnonzero(scenes > .1)],
                container_clock_sampling=dict(target_span_seconds=3, distinct_real_frames=16, stride_seconds=1.5,
                    pts_numeric_runs=[list(x) for x in runs], available_window_count=len(windows),
                    unavailable_targets=skipped, no_cross_file_windows=True, quality_or_content_masks_applied=False,
                    example_indices=[list(map(int, ix)) for _, ix in (windows[:1] + windows[-1:] if len(windows) > 1 else windows)]))


def parse_log(path, stream):
    rows, scenes = [], []
    errors, warnings = [], []
    timebase = Fraction(stream["time_base"])
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        raw = f.read()
    normalized, _ = normalize_progress(raw)
    with io.StringIO(normalized) as f:
        for line in f:
            m = FRAME.search(line)
            if m:
                if int(m[1]) != len(rows) or (int(m[4]), int(m[5])) != (stream["width"], stream["height"]):
                    raise ValueError("Nonsequential frame identity or native dimensions changed")
                rows.append(dict(frame=int(m[1]), pts_ticks=int(m[2]), pts_seconds=format(int(m[2]) * float(timebase), ".12f"),
                                 showinfo_pts_seconds=m[3], native_luma_mean=float(m[6].split()[0]), native_luma_std=float(m[7].split()[0])))
            if "config in time_base:" in line:
                tb = line.split("config in time_base:")[1].split(",")[0].strip()
                if Fraction(tb) != timebase:
                    raise ValueError("Filter timebase differs from frozen container stream")
            if "lavfi.scene_score=" in line:
                scenes.append(float(line.split("lavfi.scene_score=")[-1].strip()))
            if re.search(r"Error while decoding|Invalid NAL|corrupt decoded|error submitting|Invalid data found", line, re.I):
                errors.append(line.strip())
            elif re.search(r"warning|truncated|invalid|error", line, re.I) and "configuration:" not in line:
                warnings.append(line.strip())
    if len(rows) != len(scenes):
        raise ValueError("Full-frame native QC/scene accounting mismatch")
    for row, score in zip(rows, scenes):
        row["scene_score"] = score
    return rows, errors, list(dict.fromkeys(warnings))


def normalize_progress(raw):
    """Remove only FFmpeg CR progress messages, including in-line collisions.

    Retain all showinfo/metadata fields; remove an attached repeated logger
    prefix when a progress write interrupts a mean/std/checksum field.
    Original bytes/logs are immutable and bound in reconciliation provenance.
    """
    def remove_progress(match):
        progress = match.group().split("\r", 1)[0]
        allowed = {"frame", "fps", "q", "size", "Lsize", "time", "bitrate", "speed", "elapsed", "N", "A", "kB", "KiB", "kbits", "s", "x"}
        if set(re.findall(r"[A-Za-z_]+", progress)) - allowed:
            raise ValueError("Non-progress diagnostic inside CR progress; preserve original log")
        fields = re.findall(r"([A-Za-z]+)=", progress)
        if fields not in (["frame", "fps", "q", "size", "time", "bitrate", "speed", "elapsed"],
                          ["frame", "fps", "q", "Lsize", "time", "bitrate", "speed", "elapsed"]):
            raise ValueError("Unrecognized CR progress fields; preserve original log")
        return ""
    return re.subn(r"frame=\s*\d+\s+fps=[^\r\n]*\belapsed=[^\r\n]*\r(?!\n)"
                   r"(?:\[Parsed_(?:showinfo|metadata)_\d+ @ [0-9a-fA-F]+\] )?", remove_progress, raw)


def decode(source, folder, *, limit=None, timeout=1200):
    name = "benchmark" if limit else "full-sequential-decode"
    command = [str(FFMPEG), "-hide_banner", "-nostats", "-nostdin", "-noautorotate", "-copyts", "-threads", "2", "-i", str(source),
               "-map", "0:v:0", "-filter_threads", "1", "-vf", "showinfo,select='gte(scene,0)',metadata=mode=print:key=lavfi.scene_score", "-an"]
    if limit:
        command += ["-frames:v", str(limit)]
    command += ["-fps_mode", "passthrough", "-f", "null", "-"]
    write(folder / (name + "-command.json"), command)
    began = time.monotonic()
    with (folder / (name + ".log")).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log)
        write(folder / (name + "-process.json"), dict(pid=process.pid, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), command=command))
        print(json.dumps(dict(stage="decoding", source=source.name, pid=process.pid, limit=limit)), flush=True)
        try:
            result = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
            raise RuntimeError("Bounded local decode timeout; partial log preserved")
    seconds = time.monotonic() - began
    write(folder / (name + "-exit.json"), dict(exit_code=result, elapsed_seconds=seconds))
    if result:
        raise RuntimeError("FFmpeg nonzero exit " + str(result))
    return folder / (name + ".log"), seconds


def review(row, benchmark=False):
    started = time.monotonic()
    uid = row["source_id"]
    folder = PROCESS / uid
    folder.mkdir()
    result = dict(row, status="failed", technical_status="failed", formal_test_frozen=False, model_scores_read=False)
    try:
        source = ROOT / row["media_path"]
        before = sha(source)
        if before != row["media_sha256_inherited"]:
            raise ValueError("Actual media SHA differs from selected/source evidence")
        result.update(source_sha256_before=before, size_bytes=source.stat().st_size)
        command = [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(source)]
        write(folder / "probe-command.json", command)
        probe = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        if probe.stderr.strip():
            raise ValueError("Container probe reported errors: " + probe.stderr[:500])
        info = json.loads(probe.stdout)
        write(folder / "container-probe.json", info)
        stream = info["streams"][0]
        expected = int(stream["nb_frames"])
        if expected != int(row["proposed_end_frame_exclusive"]):
            raise ValueError("Container count differs from unchanged selected media range")
        if uid == "EIDC-SHEAF-202408-27K30W":
            prior = ROOT / "data/process/external-videos/review-20261101-v1" / uid
            proof = json.loads((prior / "machine-summary.json").read_text(encoding="utf-8"))
            if proof["original_sha256"] != before or proof["source_sha256_after"] != before or proof["full_decoded_frames"] != expected or proof["decoder_errors"]:
                raise ValueError("Previous same-SHA complete decode proof mismatch")
            with (prior / "frame-pts-quality.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if [int(r["frame"]) for r in rows] != list(range(expected)):
                raise ValueError("Previous complete QC frame identity mismatch")
            result["reused_evidence"] = {rel(p): sha(p) for p in [prior / "machine-summary.json", prior / "frame-pts-quality.csv", prior / "decode-command.json", prior / "full-sequential-decode.log"]}
            errors, warnings, seconds = [], [], 0.0
            technical = "reused"
        else:
            if benchmark:
                log, seconds = decode(source, folder, limit=90, timeout=60)
                frames, _, _ = parse_log(log, stream)
                estimate = seconds * expected / max(1, len(frames))
                write(folder / "benchmark-summary.json", dict(observed_frames=len(frames), seconds=seconds, projected_full_seconds=estimate, provisional_only=True))
                print(json.dumps(dict(stage="benchmark_complete", source=source.name, frames=len(frames), seconds=seconds, projected_full_seconds=estimate)), flush=True)
            log, seconds = decode(source, folder)
            rows, errors, warnings = parse_log(log, stream)
            technical = "completed"
        if len(rows) != expected:
            raise ValueError("Full decode frame count mismatch: actual=%d expected=%d" % (len(rows), expected))
        with (folder / "frame-pts-quality.csv").open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result.update(describe(rows), decode_seconds=seconds, decoder_errors=errors, decoder_warnings=warnings,
                      native_dimensions=[stream["width"], stream["height"]], pix_fmt=stream.get("pix_fmt"),
                      orientation_metadata=dict(tags=stream.get("tags", {}), side_data_list=stream.get("side_data_list", [])),
                      frame_csv=rel(folder / "frame-pts-quality.csv"), source_sha256_after=sha(source))
        if errors or result["source_sha256_after"] != before:
            raise ValueError("Decoder errors or source SHA changed; no acceptance")
        result.update(technical_status=technical, status="pending_content_review" if row["proposed_video_label"] is None else technical)
    except Exception as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc))
    result["wall_seconds"] = time.monotonic() - started
    result["output_sha256"] = {rel(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    write(META / (uid + ".json"), result)
    print(json.dumps(dict(stage="video_finished", source_id=uid, status=result["status"], technical_status=result["technical_status"], decoded_frames=result.get("decoded_frames"), wall_seconds=result["wall_seconds"], error=result.get("error"))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--resume-empty-bootstrap", action="store_true", help="Only resume empty directories left before any contract/media output")
    parser.add_argument("--reconcile-existing", action="store_true", help="No decoding; append corrected diagnostics from existing exit-zero logs")
    args = parser.parse_args()
    if args.reconcile_existing:
        return reconcile_existing()
    if args.resume_empty_bootstrap:
        if not all(p.is_dir() and not list(p.iterdir()) for p in [META, PROCESS]):
            raise ValueError("Bootstrap resume requires both output directories completely empty")
        write(META / "bootstrap-failure.json", dict(error="Initial CLI failed before writing contract or starting decoder: relative __file__ path in provenance", media_processed=0, original_command=[str(x) for x in sys.argv if x != "--resume-empty-bootstrap"]))
    else:
        META.mkdir(parents=True, exist_ok=False)
        PROCESS.mkdir(parents=True, exist_ok=False)
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    selected = selection["selected_media"]
    if len(selected) != 18 or sum(r["proposed_video_label"] == "pos" for r in selected) != 7 or sum(r["proposed_video_label"] == "neg" for r in selected) != 11:
        raise ValueError("Unchanged 18=7P11N selection required")
    extra = [dict(source_id=uid, media_path=MOVDIR + filename, media_sha256_inherited=digest,
                  proposed_video_label=None, proposed_start_frame="0", proposed_end_frame_exclusive=str(n),
                  capture_clock_status="unknown", original_event_id="2019-source-association-unverified") for uid, filename, digest, n in MOVS]
    rows = extra + selected
    version = subprocess.run([str(FFMPEG), "-version"], capture_output=True, text=True, check=True).stdout
    write(META / "contract.json", dict(command=sys.argv, code_path=rel(Path(__file__)), code_sha256=sha(__file__),
        ffmpeg_sha256=sha(FFMPEG), ffprobe_sha256=sha(FFPROBE), ffmpeg_version=version,
        inputs={rel(p): sha(p) for p in [SELECTION, CONFIRMATION, Path(__file__), ROOT / "experiments/common/video/pts_clip_sampling.py"]},
        media=rows, selected_18_labels_unchanged=True, unassigned_additional_media=2,
        quality="native Y luma mean/std rounded by FFmpeg showinfo; std<20 low texture, std<2 near-uniform; candidates only, not bad-frame labels or model RGB112 QC",
        cuts="existing FFmpeg scene-score >0.1 candidate rule; no automatic exclusion or physical-cut assertion",
        subtitles="retain existing selection class-bearing subtitle exclusion; native full-frame numeric QC cannot detect/clear all subtitles; no OCR/new content judgement",
        clock="real distinct decoded source frames under container PTS; physical capture clock unknown; no rotation, enhancement, frame insertion or media rewrite",
        scope="technical input audit only; no final test freeze, label change, independent-event increment, model inference or semantic full-film review"))
    began = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(review, row, i == 0) for i, row in enumerate(rows)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: [r["source_id"] for r in rows].index(x["source_id"]))
    fields = ["source_id", "media_path", "proposed_video_label", "status", "technical_status", "decoded_frames", "native_dimensions", "pts_strictly_increasing", "wall_seconds", "error"]
    with (META / "video-status.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    failed = [r["source_id"] for r in results if r["technical_status"] == "failed"]
    write(META / "summary.json", dict(status="technical_audit_failed" if failed else "technical_audit_complete_content_and_input_freeze_pending", media_count=len(results),
        selected_media=18, selected_pos=7, selected_neg=11, unassigned_media=2, elapsed_seconds=time.monotonic()-began,
        technical_completed=sum(r["technical_status"] == "completed" for r in results), technical_reused=sum(r["technical_status"] == "reused" for r in results), failed=failed,
        decoded_frames=sum(r.get("decoded_frames", 0) for r in results), formal_test_frozen=False,
        input_source_closure=all(r.get("source_sha256_before") == r.get("source_sha256_after") == r["media_sha256_inherited"] for r in results),
        output_sha256={rel(p): sha(p) for p in sorted(META.glob("*.json"))}, model_scores_read=False))
    return 1 if failed else 0


def reconcile_existing():
    began = time.monotonic()
    contract = json.loads((META / "contract.json").read_text(encoding="utf-8"))
    initial_source = META / "decoder-source-first-attempt.py"
    if sha(initial_source) != contract["code_sha256"]:
        raise ValueError("Initial source snapshot identity mismatch")
    if (META / "reconciliation.json").exists():
        raise FileExistsError("Refusing to overwrite reconciliation")
    results, changes = [], []
    for original in contract["media"]:
        uid = original["source_id"]
        prior_path = META / (uid + ".json")
        result = json.loads(prior_path.read_text(encoding="utf-8"))
        folder = PROCESS / uid
        source = ROOT / original["media_path"]
        actual = sha(source)
        if actual != original["media_sha256_inherited"] or result.get("source_sha256_before") != actual:
            raise ValueError("Actual unchanged source SHA failed during reanalysis: " + uid)
        if result["technical_status"] == "failed":
            if result.get("error") != "ValueError: Nonsequential frame identity or native dimensions changed":
                raise ValueError("Not the predeclared CR-progress parser failure: " + uid)
            receipt = json.loads((folder / "full-sequential-decode-exit.json").read_text(encoding="utf-8"))
            if receipt["exit_code"] != 0:
                raise ValueError("Nonzero decoder exit cannot be reconciled")
            stream = json.loads((folder / "container-probe.json").read_text(encoding="utf-8"))["streams"][0]
            log = folder / "full-sequential-decode.log"
            raw = log.read_bytes().decode("utf-8", errors="replace")
            normalized, count = normalize_progress(raw)
            rows, errors, warnings = parse_log(log, stream)
            if not count or errors or len(rows) != int(stream["nb_frames"]) or len(rows) != int(original["proposed_end_frame_exclusive"]):
                raise ValueError("Strict full-frame/scene/error closure failed after CR progress removal")
            csv_path = folder / "frame-pts-quality.reconciled.csv"
            with csv_path.open("x", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result.update(describe(rows), decode_seconds=receipt["elapsed_seconds"], decoder_errors=errors, decoder_warnings=warnings,
                native_dimensions=[stream["width"], stream["height"]], pix_fmt=stream.get("pix_fmt"),
                orientation_metadata=dict(tags=stream.get("tags", {}), side_data_list=stream.get("side_data_list", [])),
                frame_csv=rel(csv_path), source_sha256_after=actual, technical_status="completed",
                status="pending_content_review" if original["proposed_video_label"] is None else "completed",
                error=None, initial_parse_failure_preserved=rel(prior_path))
            provenance = dict(source_id=uid, original_log_sha256=sha(log), original_record_sha256=sha(prior_path),
                removed_CR_progress_messages=count, normalized_log_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                original_log_unchanged=True, decoder_rerun=False, recovered_frames=len(rows),
                method="Remove only known FFmpeg frame/fps/elapsed CR messages plus attached repeated logger prefix, preserve every showinfo/scene field, then enforce exact contiguous counters/count/dimensions/finite diagnostics")
            result["log_reconciliation"] = provenance
            result["output_sha256"] = {rel(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
            write(META / (uid + ".reconciled.json"), result)
            changes.append(provenance)
        elif result.get("source_sha256_after") != actual:
            raise ValueError("Existing complete diagnostic source closure failed")
        results.append(result)
    fields = ["source_id", "media_path", "proposed_video_label", "status", "technical_status", "decoded_frames", "native_dimensions", "pts_strictly_increasing", "wall_seconds", "error"]
    with (META / "video-status.reconciled.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    proof = dict(status="all_20_technical_records_closed_content_and_input_freeze_pending", media_count=len(results),
        selected_media=18, selected_pos=7, selected_neg=11, unassigned_media=2,
        technical_completed=sum(r["technical_status"] == "completed" for r in results),
        technical_reused=sum(r["technical_status"] == "reused" for r in results),
        decoded_frames=sum(r["decoded_frames"] for r in results), input_source_closure=True,
        initial_failed_records_preserved=True, decoder_reruns=0, reconciled_records=changes,
        code_sha256=sha(__file__), code_path=rel(__file__), command=sys.argv,
        contract_sha256=sha(META / "contract.json"), initial_summary_sha256=sha(META / "summary.json"),
        diagnostic_reanalysis_seconds=time.monotonic()-began, formal_test_frozen=False, model_scores_read=False,
        output_sha256={rel(p): sha(p) for p in sorted(META.iterdir()) if p.is_file()})
    write(META / "reconciliation.json", proof)
    print(json.dumps({k: proof[k] for k in ["status", "media_count", "decoded_frames", "technical_completed", "technical_reused", "diagnostic_reanalysis_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
