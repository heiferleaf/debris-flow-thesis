"""Describe saved development errors at the original threshold; never fit or score a model."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "data/results/exp022/linear-cv-s7-20261104-v1"
REPORT = ROOT / "data/results/exp022/report-linear-cv-s7-20261104-v1-v1"
OUT = ROOT / "data/results/exp022/error-diagnosis-20261105-v2"
SEGMENTS = ROOT / "data/metadata/EXP-016/metadata-v4/segments.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_window(start: float, end: float, segments: list[dict]) -> str:
    kinds = {s["kind"] for s in segments
             if float(s["annotation_start_sec"]) <= start
             and end < float(s["annotation_end_sec"])}
    if "pos" in kinds and "neg" in kinds:
        return "conflict"
    if "pos" in kinds:
        return "pos"
    if "neg" in kinds:
        return "neg"
    return "boundary_or_unannotated"


def longest_positive_run(windows: list[dict]) -> dict:
    best = {"count": 0, "start": None, "end": None}
    current = []
    for row in windows:
        if row["score"] < 0.5:
            current = []
            continue
        if current and abs(row["target_start_seconds"] - current[-1]["target_start_seconds"] - 1.5) > 1e-5:
            current = []
        current.append(row)
        if len(current) > best["count"]:
            best = {"count": len(current), "start": current[0]["start"], "end": current[-1]["end"]}
    return best


def build() -> dict:
    if OUT.exists():
        raise FileExistsError("Use a new analysis version; do not overwrite")
    pairs = read_csv(REPORT / "video-pairs.csv")
    segments = defaultdict(list)
    for row in read_csv(SEGMENTS):
        segments[row["unit_id"]].append(row)
    all_windows = defaultdict(list)
    for fold in range(1, 6):
        for row in read_csv(RUN / f"f{fold:02d}-dense/window-scores.csv"):
            for key in ["score", "start", "end", "target_start_seconds"]:
                row[key] = float(row[key])
            row["annotation_relation"] = label_window(row["start"], row["end"], segments[row["unit_id"]])
            all_windows[row["unit_id"]].append(row)
    statistics = []
    representatives = []
    conflicts = 0
    for pair in pairs:
        unit = pair["unit_id"]
        windows = sorted(all_windows[unit], key=lambda r: r["target_start_seconds"])
        values = np.asarray([r["score"] for r in windows], dtype=np.float64)
        assert len(windows) == int(pair["windows"])
        assert abs(float(values.max()) - float(pair["linear_score"])) < 1e-7
        label, prediction = int(pair["label"]), int(pair["linear_prediction"])
        outcome = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[label, prediction]
        full_pos = [r for r in windows if r["annotation_relation"] == "pos"]
        full_neg = [r for r in windows if r["annotation_relation"] == "neg"]
        conflicts += sum(r["annotation_relation"] == "conflict" for r in windows)
        run = longest_positive_run(windows)
        pmax = max((r["score"] for r in full_pos), default=None)
        item = {**pair, "outcome": outcome,
                "unique_frame_tuples": len({r["frame_tuple_sha256"] for r in windows}),
                "score_min": float(values.min()), "score_q25": float(np.quantile(values, .25)),
                "score_median": float(np.median(values)), "score_q75": float(np.quantile(values, .75)),
                "score_q90": float(np.quantile(values, .9)), "score_max": float(values.max()),
                "positive_windows": int((values >= .5).sum()),
                "positive_fraction": float((values >= .5).mean()),
                "longest_positive_grid_run": run["count"],
                "longest_run_start": run["start"], "longest_run_end": run["end"],
                "fully_pos_windows": len(full_pos), "fully_neg_windows": len(full_neg),
                "boundary_or_unannotated_windows": sum(r["annotation_relation"] == "boundary_or_unannotated" for r in windows),
                "fully_pos_max": pmax,
                "fully_pos_positive_windows": sum(r["score"] >= .5 for r in full_pos),
                "fully_neg_positive_windows": sum(r["score"] >= .5 for r in full_neg),
                "video_TP_without_positive_interval_detection": outcome == "TP" and (pmax is None or pmax < .5)}
        statistics.append(item)
        if outcome in {"FP", "FN"}:
            peak = max(windows, key=lambda r: r["score"])
            median = min(windows, key=lambda r: abs(r["score"] - float(np.median(values))))
            for role, window in [("peak", peak), ("median", median)]:
                representatives.append({"unit_id": unit, "fold": int(pair["fold"]), "outcome": outcome,
                                        "group_id": pair["group_id"], "selection": role, **window})
    assert conflicts == 0
    errors = [r for r in statistics if r["outcome"] in {"FP", "FN"}]
    assert Counter(r["outcome"] for r in errors) == {"FP": 21, "FN": 4}
    assert sum(len(v) for v in all_windows.values()) == 143470
    # Already trained/evaluated controls; their model scores are not new evaluations.
    for unit in ["v1_105", "v1_106", "v1_35", "v1_37", "v1_69", "v1_70", "v1_72", "v1_95", "v1_98", "v1_99", "v1_100"]:
        pair = next(r for r in statistics if r["unit_id"] == unit)
        candidates = [w for w in all_windows[unit] if w["annotation_relation"] == "pos"]
        if candidates:
            window = max(candidates, key=lambda r: r["score"])
            representatives.append({"unit_id": unit, "fold": int(pair["fold"]), "outcome": pair["outcome"],
                                    "group_id": pair["group_id"], "selection": "existing_positive_control", **window})
    group_stats = []
    for group in sorted({r["group_id"] for r in statistics}):
        members = [r for r in statistics if r["group_id"] == group]
        counts = Counter(r["outcome"] for r in members)
        group_stats.append({"group_id": group, "fold": int(members[0]["fold"]),
                            **{k: counts[k] for k in ["TP", "FP", "FN", "TN"]}})
    summary = {"scope": "Saved-score descriptive diagnosis at frozen0.5; no model forward, training, threshold fitting or label changes.",
               "videos": len(statistics), "physical_windows": sum(len(v) for v in all_windows.values()),
               "error_counts": dict(Counter(r["outcome"] for r in errors)),
               "FP_groups": dict(Counter(r["group_id"] for r in errors if r["outcome"] == "FP")),
               "FN_groups": dict(Counter(r["group_id"] for r in errors if r["outcome"] == "FN")),
               "FP_grid_runs_at_most_one": sum(r["outcome"] == "FP" and r["longest_positive_grid_run"] <= 1 for r in errors),
               "FP_grid_runs_at_least_ten": sum(r["outcome"] == "FP" and r["longest_positive_grid_run"] >= 10 for r in errors),
               "TP_without_detection_in_fully_pos_interval": [r["unit_id"] for r in statistics if r["video_TP_without_positive_interval_detection"]],
               "FN_positive_interval_stats": [{k:r[k] for k in ["unit_id", "fully_pos_windows", "fully_pos_max", "score_median", "score_max"]} for r in errors if r["outcome"] == "FN"],
               "annotation_boundary_policy": "Frozen half-open intervals [a,b); first sampled PTS>=a, last sampled PTS<b.",
               "supersedes": "error-diagnosis-20261105-v1 inclusive-end classification; primary run metrics unchanged.",
               "limits": ["Adjacent windows overlap; counts and grid support spans are not independent events or physical event durations.",
                          "Annotation relations use frozen declared source-time intervals; visual truth is not inferred from model score.",
                          "Representatives are selected by saved score for diagnosis, not to estimate full-dataset visual-label accuracy."],
               "input_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [REPORT / "video-pairs.csv", SEGMENTS, Path(__file__).resolve()]}}
    OUT.mkdir(parents=True)
    write_csv(OUT / "all-video-statistics.csv", statistics)
    write_csv(OUT / "error-videos.csv", errors)
    write_csv(OUT / "group-statistics.csv", group_stats)
    write_csv(OUT / "representative-windows.csv", representatives)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
