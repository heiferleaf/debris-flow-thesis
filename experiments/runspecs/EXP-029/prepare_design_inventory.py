# -*- coding: utf-8 -*-
"""Read-only label/grid inventory for an N/P/T experiment design; never decodes media."""
import argparse
import bisect
import collections
import csv
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint(path, root):
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out")
    destination.add_argument("--candidate-roles-out")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.candidate_roles_out:
        design_path = root / "experiments/runspecs/EXP-029/design.json"
        scope_path = root / "data/metadata/annotation-review/recheck-20261108-v2/all-dongchuan-scope.csv"
        candidate = read_json(design_path)["candidate_partition"]
        scope = [r for r in read_csv(scope_path)
                 if r["in_current150_development_pool"].lower() == "true"]
        blocks = {key: [r["unit_id"] for r in scope if r["camera"] in views]
                  for key, views in candidate["blocks"].items()}
        flattened = [key for members in blocks.values() for key in members]
        assert len(flattened) == len(set(flattened)) == len(scope) == 150
        folds = []
        outer_exposure = collections.Counter()
        for rule in candidate["folds"]:
            roles = {role: sorted(key for b in rule[role] for key in blocks[b])
                     for role in ("inner_train", "inner_val", "outer_eval")}
            all_members = [key for members in roles.values() for key in members]
            assert len(all_members) == len(set(all_members)) == 150
            roles["outer_train"] = sorted(roles["inner_train"] + roles["inner_val"])
            folds.append(dict(fold=rule["fold"], roles=roles,
                              counts={r: len(keys) for r, keys in roles.items()}))
            outer_exposure.update(roles["outer_eval"])
        assert set(outer_exposure) == set(flattened) and set(outer_exposure.values()) == {1}
        result = dict(status=candidate["status"], frozen=False,
                      physical_scene_independence_verified=False,
                      training_authorized=False, blocks=blocks, folds=folds,
                      checks={"scope_video_count": 150, "sides_disjoint": True,
                              "same_block_not_split": True, "outer_each_video_once": True},
                      evidence=[fingerprint(p, root) for p in (design_path, scope_path)])
        target = (root / args.candidate_roles_out).resolve()
        target.relative_to((root / "data/metadata").resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"block_counts": {k: len(v) for k, v in blocks.items()},
                          "fold_counts": [f["counts"] for f in folds], "checks": result["checks"]},
                         ensure_ascii=False, indent=2))
        return
    output = (root / args.out).resolve()
    output.relative_to((root / "data/metadata").resolve())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refuse to overwrite an existing non-empty inventory")
    output.mkdir(parents=True, exist_ok=True)
    current = root / "data/metadata/annotation-revisions/current-review.json"
    revision = root / read_json(current)["working_annotations_dir"]
    scope_path = root / "data/metadata/annotation-review/recheck-20261108-v2/all-dongchuan-scope.csv"
    scope = {r["unit_id"]: r for r in read_csv(scope_path)
             if r["in_current150_development_pool"].lower() == "true"}
    assert len(scope) == 150
    input_files = [current, revision / "sources.json", scope_path]
    labels = collections.defaultdict(list)
    kind_map = {"neg": "N", "transition": "T", "pos": "P", "cut": "cut"}
    for source in read_json(revision / "sources.json")["sources"].values():
        if source.get("audit_only"):
            continue
        path = revision / source["annotations_csv"]
        input_files.append(path)
        for row in read_csv(path):
            key = row["video_key"]
            if key in scope:
                assert row["kind"] in kind_map, row
                labels[key].append({"start": float(row["start_sec"]),
                                    "end": float(row["end_sec"]),
                                    "kind": kind_map[row["kind"]]})
    units = {}
    interval_rows = []
    for key in scope:
        intervals = sorted(labels[key], key=lambda r: (r["start"], r["end"]))
        assert intervals, key
        for i, row in enumerate(intervals):
            assert row["end"] > row["start"], (key, row)
            if i:
                assert row["start"] >= intervals[i - 1]["end"] - 1e-6, (key, intervals)
            interval_rows.append(dict(video_key=key, camera=scope[key]["camera"], **row))
        kinds = sorted({r["kind"] for r in intervals if r["kind"] != "cut"})
        duration = {c: sum(r["end"] - r["start"] for r in intervals if r["kind"] == c)
                    for c in ("N", "T", "P", "cut")}
        units[key] = dict(video_key=key, camera=scope[key]["camera"],
                          label_group="all_" + kinds[0] if len(kinds) == 1 else "mixed",
                          kinds="|".join(kinds), **{c + "_seconds": duration[c] for c in duration},
                          intervals=intervals, starts=[r["start"] for r in intervals],
                          grid_counts=collections.Counter(), legacy_pass_counts=collections.Counter())
    grid_path = root / "data/metadata/EXP-028/full-coverage-plan-20261108-v1/windows.jsonl"
    input_files.append(grid_path)
    grid_count = 0
    with grid_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            unit = units[row["unit_id"]]
            center = (row["target_start_seconds"] + row["target_end_seconds"]) / 2
            index = bisect.bisect_right(unit["starts"], center) - 1
            state = "unlabelled"
            if index >= 0 and center < unit["intervals"][index]["end"]:
                state = unit["intervals"][index]["kind"]
            unit["grid_counts"][state] += 1
            if row.get("legacy_guard_pass") and row.get("gray_QC_pass") and row.get("PTS_pass"):
                unit["legacy_pass_counts"][state] += 1
            grid_count += 1
    unit_rows = []
    cameras = {}
    for unit in units.values():
        row = {k: unit[k] for k in ("video_key", "camera", "label_group", "kinds",
                                   "N_seconds", "T_seconds", "P_seconds", "cut_seconds")}
        for c in ("N", "T", "P", "cut", "unlabelled"):
            row["nominal_grid_" + c] = unit["grid_counts"][c]
            row["legacy_guard_grid_" + c] = unit["legacy_pass_counts"][c]
        unit_rows.append(row)
        camera = cameras.setdefault(unit["camera"], dict(camera=unit["camera"], videos=0,
                                                         N_videos=0, T_videos=0, P_videos=0,
                                                         N_seconds=0., T_seconds=0., P_seconds=0.))
        camera["videos"] += 1
        for c in ("N", "T", "P"):
            camera[c + "_videos"] += int(unit[c + "_seconds"] > 0)
            camera[c + "_seconds"] += unit[c + "_seconds"]
    summary = dict(schema_version=1, status="design_inventory_only_not_training_release",
                   annotations_dir=revision.relative_to(root).as_posix(),
                   videos=len(units), cameras=len(cameras),
                   label_groups=dict(collections.Counter(u["label_group"] for u in units.values())),
                   label_seconds={c: round(sum(u[c + "_seconds"] for u in units.values()), 6)
                                  for c in ("N", "T", "P", "cut")},
                   T_video_keys=[u["video_key"] for u in units.values() if u["T_seconds"] > 0],
                   T_camera_count=sum(c["T_videos"] > 0 for c in cameras.values()),
                   shortest_state_interval_seconds=min(r["end"] - r["start"] for r in interval_rows
                                                       if r["kind"] != "cut"),
                   shortest_T_interval_seconds=min(r["end"] - r["start"] for r in interval_rows
                                                   if r["kind"] == "T"),
                   full_grid_count=grid_count,
                   nominal_anchor_counts={c: sum(u["grid_counts"][c] for u in units.values())
                                          for c in ("N", "T", "P", "cut", "unlabelled")},
                   legacy_guard_anchor_counts={c: sum(u["legacy_pass_counts"][c] for u in units.values())
                                               for c in ("N", "T", "P", "cut", "unlabelled")},
                   boundaries="Intervals are half-open; anchor is nominal 3s-window midpoint.",
                   limitations=["Numeric annotation-time overlay only; no new source-pixel alignment or decode.",
                                "Nominal grid counts are not prepared/usable training samples.",
                                "Legacy guards may include semantic exclusions; their intersection is diagnostic only.",
                                "Camera identities are not certified independent physical scenes.",
                                "N/P/T video support counts overlap; interval counts are not independent events."],
                   training_started=False, source_files=[fingerprint(p, root) for p in input_files])
    write_csv(output / "per-video.csv", unit_rows)
    write_csv(output / "per-camera.csv", list(cameras.values()))
    write_csv(output / "intervals.csv", interval_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "source_files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
