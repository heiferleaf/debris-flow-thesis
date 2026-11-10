"""Describe fixed-threshold windows; do not tune thresholds or rescore models."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("bce", "cross", "same")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(run, out):
    scores_path = run / "three-dense/window-scores.csv"
    videos_path = run / "analysis-v2/video-comparison.csv"
    dense_path = run / "three-dense/result.json"
    with videos_path.open(encoding="utf-8-sig", newline="") as f:
        video_rows = list(csv.DictReader(f))
    videos = {r["unit_id"]: r for r in video_rows}
    if len(videos) != len(video_rows) or len(videos) != 31:
        raise ValueError("Expected 31 unique development videos")
    expected = json.loads(dense_path.read_text(encoding="utf-8"))["videos"]
    values = defaultdict(lambda: defaultdict(list))
    last_start = {}
    coincident_starts = 0
    with scores_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["unit_id"]
            if uid not in videos:
                raise ValueError("Unrecognized video identity")
            start, end = float(row["start"]), float(row["end"])
            if (not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start or start < last_start.get(uid, -1)):
                raise ValueError("Nonfinite or disordered window range")
            # Distinct frozen grids can map to the same actual first PTS.
            # Preserve all original rows, including that overlap; do not dedupe.
            coincident_starts += int(start == last_start.get(uid, -1))
            last_start[uid] = start
            for method in METHODS:
                score = float(row[method])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Nonfinite or out-of-range score")
                values[uid][method].append(score)
    if set(values) != set(videos) or set(expected) != set(videos):
        raise ValueError("Video coverage differs from completed dense evaluation")
    records = []
    for uid in sorted(videos):
        v = videos[uid]
        label = int(v["label"])
        if label not in (0, 1):
            raise ValueError("Invalid video label")
        for method in METHODS:
            scores = values[uid][method]
            if len(scores) != expected[uid]["windows"]:
                raise ValueError("Missing or duplicate dense windows")
            if max(scores) != float(v[method]):
                raise ValueError("Dense maximum differs from audited video score")
            records.append(dict(unit_id=uid, group_id=v["group_id"], video_label=label,
                method=method, windows=len(scores), above_threshold=sum(s >= .5 for s in scores),
                minimum=min(scores), maximum=max(scores), video_pred=int(max(scores) >= .5)))
    totals = {}
    for method in METHODS:
        negative = [r for r in records if r["method"] == method and r["video_label"] == 0]
        totals[method] = dict(negative_videos=len(negative),
            negative_windows=sum(r["windows"] for r in negative),
            above_threshold_negative_windows=sum(r["above_threshold"] for r in negative),
            false_positive_videos=sum(r["video_pred"] for r in negative))
    if sum(r["windows"] for r in records if r["method"] == "bce") != 32567:
        raise ValueError("Expected 32567 completed dense windows")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "per-video-window-counts.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    result = dict(scope="Descriptive analysis of already-scored development windows only",
        threshold=.5, new_inference=False, threshold_search=False,
        coincident_actual_start_pairs=coincident_starts,
        limitations=["Overlapping windows are correlated, not independent errors or alarm episodes",
                     "Positive video labels do not supply positive labels for every window",
                     "Counts describe only seven negative development videos; no independent test",
                     "This does not replace the registered EXP-018 primary endpoint"],
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (scores_path, videos_path, dense_path)},
        totals=totals, generator_sha256=sha(Path(__file__)))
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "generator-source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(totals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "data/results/exp018/paired-s7-20261101-v1")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    describe(args.run.resolve(), (args.out or args.run / "false-alarm-description-v1").resolve())
