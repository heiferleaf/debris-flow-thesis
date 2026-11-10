"""Reserve final-evaluation media without copying videos or changing human labels."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path("data/metadata/external-videos/human-review-20261104-v1")
HISTORY = Path("data/results/temporal/EXP-006-20260924-addrmap-session-pred/segment-manifest-20260924-addrmap.csv")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Frozen version exists; choose a new version, never overwrite")
    session = json.loads((ROOT / REVIEW / "session-manifest.json").read_text(encoding="utf-8"))
    annotation_bytes = (ROOT / REVIEW / "annotations.json").read_bytes()
    annotations = json.loads(annotation_bytes)
    videos = session["manifest"]["videos"]
    if len(videos) != 27 or set(annotations["annotations"]) != {x["id"] for x in videos}:
        raise ValueError("Review membership changed; inspect before making a new version")
    with (ROOT / "data/metadata/external-videos/catalog.csv").open(encoding="utf-8-sig", newline="") as f:
        catalog = {x["source_id"]: x for x in csv.DictReader(f)}
    with (ROOT / HISTORY).open(encoding="utf-8-sig", newline="") as f:
        old = [x for x in csv.DictReader(f) if x["source"] == "deflow"]
    if {x["video_id"] for x in old} != {"cam1", "cam2"} or any(x["split"] != "train" for x in old):
        raise ValueError("Expected historical DeFlow training evidence changed")
    with (ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv").open(encoding="utf-8-sig", newline="") as f:
        current = list(csv.DictReader(f))
    if len(current) != 150 or any("v3_" in json.dumps(x) for x in current):
        raise ValueError("Current development-pool membership changed")

    # Explicit scene identities: same view across dates remains one scene.
    scenes = {
        "v001": "USGS:HPCC:cam01", "v002": "USGS:ChalkCliffs:left-view",
        "v003": "USGS:ChalkCliffs:right-view", "v004": "USGS:ChalkCliffs:right-view",
        "v005": "USGS:ChalkCliffs:right-view", "v006": "EIDC:Sheaf:fixed-view",
        "v007": "USGS:Elk:cam01", "v021": "USGS:Silverado:cam01",
        "v022": "USGS:Silverado:cam01", "v023": "USGS:LasLomas:source-view",
        "v024": "USGS:Calwood:station-view01",
        "v025": "USGS:ChalkCliffs:Firehose", "v026": "USGS:ChalkCliffs:Firehose",
        "v027": "USGS:ChalkCliffs:Firehose",
    }
    rows = []
    for video in videos:
        vid = video["id"]
        label = annotations["annotations"][vid]
        source = ROOT / video["source_path"]
        if source.stat().st_size != video["size_bytes"]:
            raise ValueError("Current media size changed: " + vid)
        cat = catalog.get(video["source_id"], {})
        scene = scenes.get(vid) or cat["split_group"]
        role = {"positive": "primary_external_reserved", "negative": "primary_external_reserved",
                "exclude": "excluded_by_human", "uncertain": "excluded_unresolved_truth"}.get(label["label"])
        if role is None:
            raise ValueError("Unreviewed video cannot silently enter final evaluation: " + vid)
        firehose = vid in {"v025", "v026", "v027"}
        event = ("USGS:Firehose:flood-050819:source-recording" if vid in {"v025", "v026"}
                 else "USGS:Firehose:debrisflow-080319:source-recording" if vid == "v027"
                 else cat["original_event_id"])
        scene_evidence = ("human annotations: v004/v005" if vid in {"v003", "v004", "v005"}
                          else "human annotations: v019/v020" if vid in {"v014", "v015", "v016", "v017", "v018", "v019", "v020"}
                          else "human annotations: v025" if firehose else "existing catalog camera/source grouping")
        rows.append({
            "video_id": vid, "legacy_video_id": "", "source_id": video["source_id"],
            "source_path": video["source_path"], "size_bytes": video["size_bytes"],
            "sha256": video["sha256"], "hash_policy": "inherited_existing_verified_metadata_size_checked",
            "video_label": label["label"], "evaluation_role": role, "scene_group": scene,
            "scene_group_evidence": scene_evidence, "event_group": event,
            "event_group_status": "source_recording_identity_not_a_verified_independent_event_count",
            "parent_media_id": cat.get("download_source_id") or ("USGS:Firehose:flood-050819" if vid in {"v025", "v026"} else video["source_id"]),
            "project_exposure": "no_prior_project_use_identified_in_existing_source_records",
            "label_evidence": f"human-review revision {annotations['revision']}",
            "positive_intervals_json": json.dumps(label.get("positive_intervals", []), ensure_ascii=False),
            "rotation_clockwise_degrees": label.get("rotation", 0),
            "notes": label.get("notes", ""),
        })
    for number, cam in enumerate(["cam1", "cam2"], 28):
        path = Path("data/raw/deflow/Data") / (cam + ".mp4")
        rows.append({
            "video_id": f"v{number:03d}", "legacy_video_id": "v3_0" + cam[-1],
            "source_id": "ETH-DeFlow-" + cam, "source_path": path.as_posix(),
            "size_bytes": (ROOT / path).stat().st_size, "sha256": digest(ROOT / path),
            "hash_policy": "new_sha256_for_this_reservation",
            "video_label": "positive", "evaluation_role": "historically_exposed_supplementary_retest",
            "scene_group": "DeFlow:" + cam,
            "scene_group_evidence": "existing user-confirmed EXP006 scene protocol; v3 mapping",
            "event_group": "DeFlow:unresolved-event-link",
            "event_group_status": "event_relationship_unknown_do_not_count_as_two_independent_events",
            "parent_media_id": "ETH-DeFlow-" + cam,
            "project_exposure": "EXP006_training_and_validation_model_selection",
            "label_evidence": "user confirmed both retained DeFlow videos positive; existing pos segments",
            "positive_intervals_json": "[]", "rotation_clockwise_degrees": 0,
            "notes": "Whole-video positive; historical interval is not a newly audited onset or full-duration claim. Not in current 150-video development pool.",
        })
    primary = [x for x in rows if x["evaluation_role"] == "primary_external_reserved"]
    negatives = [x for x in primary if x["video_label"] == "negative"]
    if len(rows) != 29 or any(x["legacy_video_id"] for x in primary):
        raise AssertionError("Historical material leaked into primary external evaluation")
    pins = {str(p).replace("\\", "/"): digest(ROOT / p) for p in [
        REVIEW / "session-manifest.json", Path("data/metadata/external-videos/catalog.csv"),
        HISTORY, Path("data/metadata/splits/EXP-006-session-folds.json"),
        Path("data/metadata/EXP-017/role-free-plan-v3/sources.csv"),
        Path(__file__).resolve().relative_to(ROOT)]}
    pins[(REVIEW / "annotations.json").as_posix()] = hashlib.sha256(annotation_bytes).hexdigest()
    summary = {"registered_media": len(rows), "primary_external_video_labels": dict(Counter(x["video_label"] for x in primary)),
               "primary_negative_videos_by_scene": dict(sorted(Counter(x["scene_group"] for x in negatives).items())),
               "primary_scene_count": len({x["scene_group"] for x in primary}),
               "supplementary_historically_exposed_positive_videos": 2,
               "excluded_by_human": sum(x["evaluation_role"] == "excluded_by_human" for x in rows),
               "excluded_unresolved_truth": sum(x["evaluation_role"] == "excluded_unresolved_truth" for x in rows)}
    protocol = {
        "schema": "reserved-final-video-evaluation/v1", "date": "2026-11-05",
        "annotation_revision_snapshot": annotations["revision"], "membership_and_video_labels_frozen": True,
        "input_manifest_frozen": False, "model_scoring_performed": False, "scoring_allowed_by_this_manifest": False,
        "purpose": "Reserve media and truth for final whole-video evaluation after model/input/decision rules are fixed",
        "truth_unit": "whole video; a positive interval is evidence, never an input-selection mask",
        "unmarked_time_is_negative": False,
        "primary_and_historical_results": "Report separately; never combine DeFlow with independent external primary metrics",
        "scene_metrics": {"micro_FPR": "sum(FP_g)/sum(N_g)", "macro_scene_FPR": "mean(FP_g/N_g) over scenes with negative videos",
                          "additional": "Report per-scene counts, file TP/FP/FN/TN and Recall; class-conditional scene Recall where meaningful. No negative members means no FPR for that scene."},
        "correlation": "Retain same-scene files; do not treat windows/files/multiple views as independent events or compute file-IID confidence intervals",
        "input_policy": "Use whole valid footage and label-independent PTS/QC sampling, preserve human rotation; do not trim to positive intervals or turn unmarked time into negatives",
        "selection_policy": "No use for training, choosing epochs/thresholds/architecture or rescuing a failed method; freeze model and inference rules before first scoring",
        "version_policy": "Later human changes require an explicit new version, never overwrite this label snapshot",
        "scope_limit": "Small selected-source evaluation; site/class background association remains and does not support arbitrary-site generalization",
        "pins": pins, "summary": summary,
    }
    out.mkdir(parents=True)
    (out / "human-annotations.snapshot.json").write_bytes(annotation_bytes)
    with (out / "media.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out / "protocol.json", protocol)
    write_json(out / "checks.json", {"status": "passed", "unique_video_ids": len({x['video_id'] for x in rows}),
        "active_development_videos": len(current), "deflow_in_active_development": False,
        "historical_deflow_training_rows": len(old), "source_labels_modified": False,
        "video_bytes_copied": 0, "summary": summary})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/metadata/external-videos/final-evaluation-20261105-v1")
    print(json.dumps(build(parser.parse_args().out), ensure_ascii=False, indent=2))
