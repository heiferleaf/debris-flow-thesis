# -*- coding: utf-8 -*-
"""Small sequential native-video probe. No random seek or training changes."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--times", required=True, help="Comma-separated native reported seconds")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = sorted(set(float(t) for t in args.times.split(",")))
    before = source.stat()
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Native video failed to open")
    declared = dict(fps=cap.get(cv2.CAP_PROP_FPS), frames=cap.get(cv2.CAP_PROP_FRAME_COUNT),
                    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samples, timeline, cursor, index = [], [], 0, 0
    started = time.monotonic()
    try:
        while cursor < len(targets) and cap.grab():
            seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timeline.append(dict(decoded_index=index, reported_seconds=seconds))
            if seconds >= targets[cursor]:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError("Native retrieve failed at index {}".format(index))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                neutral = (np.max(frame, axis=2) - np.min(frame, axis=2) <= 2)
                name = "native-{:06d}-t{:.3f}.jpg".format(index, seconds)
                preview = cv2.resize(frame, (960, int(round(960 * frame.shape[0] / frame.shape[1]))))
                if not cv2.imwrite(str(output / name), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError("Preview write failed")
                samples.append(dict(requested_seconds=targets[cursor], decoded_index=index,
                                    reported_seconds=seconds, gray_std=float(gray.std()),
                                    nearly_neutral_pixel_fraction=float(neutral.mean()), preview=name))
                while cursor < len(targets) and seconds >= targets[cursor]:
                    cursor += 1
            index += 1
            if index > 20000:
                raise RuntimeError("Probe guard reached; timeline may have stalled")
    finally:
        cap.release()
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    with (output / "reported-timeline.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["decoded_index", "reported_seconds"])
        writer.writeheader()
        writer.writerows(timeline)
    report = dict(source=str(source), source_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns,
                  decoder="OpenCV CAP_FFMPEG sequential grab/retrieve; no seek", opencv_version=cv2.__version__,
                  declared=declared, sampled=samples, decoded_frames=index,
                  elapsed_seconds=round(time.monotonic() - started, 3),
                  all_requested_samples_reached=(cursor == len(targets)),
                  limits=["Native reported time is not silently substituted for audit or middleware seconds.",
                          "Statistics locate candidates; actual previews need visual inspection.",
                          "Only requested early interval probed, not whole-video correctness certification."])
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
