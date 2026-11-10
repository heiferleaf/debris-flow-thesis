"""One sequential, no-seek decode of the current v5_08 fixed candidate.

Writes only technical evidence. Does not edit annotations or eligibility.
"""
import argparse
import csv
import hashlib
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    before = source.stat()
    command = [str(Path(args.ffmpeg).resolve()), "-hide_banner", "-loglevel", "info",
               "-threads", "2", "-i", str(source), "-an", "-sn", "-vf",
               "scale=256:144,format=rgb24,showinfo", "-vsync", "0", "-pix_fmt",
               "rgb24", "-f", "rawvideo", "pipe:1"]
    environment = subprocess.run([command[0], "-version"], capture_output=True, text=True).stdout
    (out / "ffmpeg-version.txt").write_text(environment, encoding="utf-8")
    (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    pattern = re.compile(r" n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(\S+)")
    pending = queue.Queue()
    records, samples, errors = [], [], []
    started = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 20)

    def drain():
        with (out / "stderr.txt").open("w", encoding="utf-8") as logfile:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                logfile.write(line)
                match = pattern.search(line)
                if match:
                    pending.put((int(match[1]), int(match[2]), float(match[3])))
                elif re.search(r"(?i)(error|invalid|missing|corrupt|non.monoton|could not|NAL unit)", line):
                    errors.append(line.strip())
        pending.put(None)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    targets = [("support3_" + str(i).zfill(2), i * .19) for i in range(16)]
    targets += [("support7p5_" + str(i).zfill(2), i * .49) for i in range(16)]
    targets += [("navigation_" + str(i).zfill(2), float(i * 120)) for i in range(18)]
    targets += [("quality_" + str(i).zfill(2), 1760.0 + i * .5) for i in range(31)]
    targets.sort(key=lambda x: x[1])
    cursor, n, previous_hash, previous_low = 0, 0, None, False
    previous_frame = None
    candidate_runs = []
    active_run = None
    frame_bytes = 256 * 144 * 3

    def save(name, frame, record):
        path = out / "frames" / (name + ".png")
        ok = cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Failed to write " + str(path))
        samples.append(dict(name=name, frame_file=str(path.relative_to(out)), **record))

    with (out / "frame-statistics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["n", "pts", "seconds", "gray_std", "neutral_fraction", "rgb_sha256", "same_as_previous"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while True:
            data = proc.stdout.read(frame_bytes)
            if not data:
                break
            while len(data) < frame_bytes:
                extra = proc.stdout.read(frame_bytes - len(data))
                if not extra:
                    raise RuntimeError("Partial raw frame at EOF")
                data += extra
            meta = pending.get(timeout=120)
            if meta is None or meta[0] != n:
                raise RuntimeError("showinfo/raw frame mismatch at " + str(n))
            _, pts, seconds = meta
            frame = np.frombuffer(data, dtype=np.uint8).reshape((144, 256, 3))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            sha = hashlib.sha256(data).hexdigest()
            row = dict(n=n, pts=pts, seconds=seconds, gray_std=float(gray.std()),
                       neutral_fraction=float((np.ptp(frame, axis=2) <= 2).mean()),
                       rgb_sha256=sha, same_as_previous=(sha == previous_hash))
            writer.writerow(row)
            records.append(row)
            while cursor < len(targets) and seconds >= targets[cursor][1]:
                save(targets[cursor][0], frame, row)
                cursor += 1
            low = row["gray_std"] < 20
            if low and not previous_low:
                active_run = dict(start_n=n, start_sec=seconds, min_gray_std=row["gray_std"])
                if previous_frame is not None:
                    save("candidate_" + str(len(candidate_runs)).zfill(2) + "_before", previous_frame, records[-2])
                save("candidate_" + str(len(candidate_runs)).zfill(2) + "_first", frame, row)
            elif low:
                active_run["min_gray_std"] = min(active_run["min_gray_std"], row["gray_std"])
            elif previous_low:
                active_run.update(end_n=n - 1, end_sec=records[-2]["seconds"], next_sec=seconds)
                candidate_runs.append(active_run)
                save("candidate_" + str(len(candidate_runs) - 1).zfill(2) + "_last", previous_frame, records[-2])
                save("candidate_" + str(len(candidate_runs) - 1).zfill(2) + "_after", frame, row)
                active_run = None
            previous_hash, previous_low, previous_frame = sha, low, frame.copy()
            n += 1
            if n % 2000 == 0:
                print(json.dumps(dict(decoded_frames=n, seconds=seconds, elapsed=round(time.monotonic()-started, 2))), flush=True)
    if active_run:
        active_run.update(end_n=n - 1, end_sec=records[-1]["seconds"], next_sec=None)
        candidate_runs.append(active_run)
    if previous_frame is not None:
        save("final", previous_frame, records[-1])
    rc = proc.wait()
    thread.join()
    after = source.stat()
    unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    diffs = np.diff([r["seconds"] for r in records])
    jumps = [dict(previous_n=i, before=records[i]["seconds"], after=records[i+1]["seconds"], delta=float(d))
             for i, d in enumerate(diffs) if d <= 0 or d > .25]
    support = {}
    for prefix, duration in [("support3_", 3.0), ("support7p5_", 7.5)]:
        group = [s for s in samples if s["name"].startswith(prefix)]
        support[prefix] = dict(requested_window_sec=duration, selected_frames=len(group),
                               distinct_decode_indices=len({s["n"] for s in group}),
                               distinct_pixel_sha256=len({s["rgb_sha256"] for s in group}),
                               first_last_sec=[group[0]["seconds"],group[-1]["seconds"]] if group else None,
                               min_gray_std=min([s["gray_std"] for s in group], default=None))
    summary = dict(source=str(source), source_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
                   source_sha256=digest(source), source_stat_unchanged=unchanged, command=command,
                   returncode=rc, sequential_from_start_without_seek=True, eof_reached=True,
                   decoded_real_frames=n, first_last_sec=[records[0]["seconds"], records[-1]["seconds"]],
                   elapsed_sec=round(time.monotonic()-started,3),
                   positive_pts_delta_median=float(np.median(diffs[diffs>0])),
                   max_pts_delta=float(max(diffs)), nonpositive_pts_delta_count=int((diffs<=0).sum()),
                   jumps_over_025_or_nonpositive=jumps, low_std_candidate_runs=candidate_runs,
                   error_warning_line_count=len(errors), error_warning_unique=sorted(set(errors)),
                   exact_adjacent_rgb_repeat_count=sum(r["same_as_previous"] for r in records),
                   support_examples=support,
                   limits=["Low standard deviation and timestamp diagnostics are technical candidates, not semantic labels.",
                           "No fps interpolation or vsync duplication was used; all images correspond to actual decoder output.",
                           "Timestamps describe the current fixed stream; repair/pre-fix mapping remains unknown.",
                           "No annotations, mapping flags, source bytes, or training data were changed."])
    (out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
