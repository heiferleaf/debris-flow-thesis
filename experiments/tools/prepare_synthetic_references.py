"""Prepare negative-only v3 metadata from v2 without opening or changing images.

The historical v2 image-producing function is retained below for traceability,
but is never called by the CLI. The default reuses all v2 reference records.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.synthetic_runway import RunwayWorkflow, atomic_json

DEFAULT_SOURCE = ROOT / "data/metadata/synthetic-video/workflow-preparation-20261108-v2"
DEFAULT_OUTPUT = ROOT / "data/metadata/synthetic-video/workflow-preparation-20261108-v3"
SINGLE_PREVIEW = "data/metadata/synthetic-video/preview-gen4turbo-20261108-v1/manifest.json"
CONDITIONS = {"no_target", "ordinary_water", "flood"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _legacy_v2_reference_producer():
    from PIL import Image, ImageOps, ImageDraw, ImageFont

    source = ROOT / "data/metadata/synthetic-video/workflow-preparation-20261108-v1"
    original = json.loads((source / "task-candidates.json").read_text(encoding="utf-8-sig"))
    inventory = json.loads((source / "scene-inventory.json").read_text(encoding="utf-8-sig"))
    units = {u["unit_id"]:u for u in inventory["units"]}
    output = ROOT / "data/metadata/synthetic-video/workflow-preparation-20261108-v2"
    images = ROOT / "data/process/synthetic-video/references-20261108-v2"
    output.mkdir(parents=True, exist_ok=True); images.mkdir(parents=True, exist_ok=True)
    groups = {}
    for task in original["tasks"]:
        groups.setdefault(task["camera_key"], task)
    references = {}
    for camera, task in groups.items():
        candidates = sorted(task["reference_candidates"], key=lambda r: (not r["source_unit"].startswith("v5_"), r["source_unit"]))
        choices = []
        for candidate in candidates:
            index_path = ROOT / candidate["frames_index_candidate_path"]
            with index_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [r for r in csv.DictReader(stream) if r.get("video_key") == candidate["source_unit"]]
            intervals = candidate["candidate_intervals"]
            if not intervals and task["reference_edit_required"]:
                intervals = [i for i in units[candidate["source_unit"]]["intervals"] if i["label"] == "P"]
            for interval in intervals:
                start, end = interval["start_sec"], interval["end_sec"]
                target = 300.0 if camera == "s|鸡冠山下游" and start <= 300 < end else (start+end)/2
                for row in sorted(rows, key=lambda r: abs(float(r["t_sec"])-target)):
                    if start <= float(row["t_sec"]) < end:
                        path = index_path.parent / "frames" / row["frame_file"]
                        if path.is_file():
                            choices.append((candidate, interval, row, path));break
            if choices:
                break
        if not choices:
            references[camera] = {"status": "no_cached_candidate", "accepted": False};continue
        candidate, interval, row, path = choices[0]
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            original_size = list(im.size)
            # Preserve the whole fixed view. No center crop or aspect distortion.
            normalized = ImageOps.pad(im.convert("RGB"), (1280,720), method=Image.Resampling.LANCZOS, color=(0,0,0))
        name = digest(camera.encode("utf-8"))[:12] + ".jpg"
        target_path = images / name
        normalized.save(target_path, "JPEG", quality=90)
        references[camera] = {"status": "local_reference_candidate_needs_content_review", "accepted": False,
            "source_camera":task["source_camera"],"source_unit":candidate["source_unit"],"source_group":candidate["source_group"],
            "source_frame":path.relative_to(ROOT).as_posix(),"source_frame_sha256":digest(raw),
            "source_time_sec":float(row["t_sec"]),"source_annotation":interval,"source_dimensions":original_size,
            "path":target_path.relative_to(ROOT).as_posix(),"sha256":digest(target_path.read_bytes()),
            "dimensions":[1280,720],"geometry_operation":"uniform contain and letterbox; no target removal",
            "source_aspect_ratio_preserved":True,"reference_edit_required":bool(candidate.get("reference_edit_required")),
            "fixed_view_content_verified":False,"flow_direction_verified":False,
            "shared_processing_history":"reference-candidate-v2/"+name,"synthetic":False}
    jobs = []
    for old in original["tasks"]:
        task = dict(old); ref = references[old["camera_key"]]
        task["duration_sec"] = 10
        task["prompt_en"] = task["prompt_en"].replace("5-second", "10-second")
        if task["condition"] == "no_target":
            task["prompt_en"] += " Preserve the channel bed without a debris-flow body. Gentle ordinary water is allowed; the camera remains locked."
        task["provider_request"]={"model":"gen4_turbo","ratio":"1280:720","duration":10,
            "seed":int(digest(task["job_id"].encode("utf-8"))[:8],16),"promptText":task["prompt_en"]}
        task["reference_candidate"] = ref
        task["selected_reference"] = None
        task["provider_preference"]={"service":"Runway Dev","integration":"official_rest_api","model":"gen4_turbo",
            "api_version":"2024-11-06","price_snapshot_usd_per_output_second":.05,
            "price_source":"https://docs.dev.runwayml.com/guides/pricing/"}
        task["estimated_video_generation_usd"] = .5
        task["estimated_cost_caveat"]="Base10sec generation only; excludes reference editing, tax, retries; not authorization"
        task["split"]="unassigned";task["submit_ready"]=False;task["training_ready"]=False
        task["status"]="pending_label_review_split_reference_and_cost_authorization"
        task["effective_support_requirements"]={**task["effective_support_requirements"],
            "candidate_total_duration_sec":10,"minimum_contiguous_effective_support_sec":7.5,
            "reason":"Supports both3sec and7.5sec candidate inputs without assuming5sec is sufficient",
            "accepted_effective_duration_sec":None}
        jobs.append(task)
    document={"schema_version":2,"provider":"runway_dev","tasks":jobs,"task_count":len(jobs),"duration_sec":10,
        "parent_manifest":(source/"task-candidates.json").relative_to(ROOT).as_posix(),
        "parent_sha256":digest((source/"task-candidates.json").read_bytes()),
        "camera_split_frozen":False,"train_camera_keys":[],"submit_ready":False,"training_ready":False,
        "public_base_video_estimate_usd":len(jobs)*.5,"initial_pilot_base_estimate_usd":16*.5,
        "initial_pilot_camera_candidates":["s|鸡冠山下游","s|门前沟上游"],
        "pilot_rationale":"One accepted ordinary-flow reference and one missing-N view; pending frozen train roles",
        "network_requests_performed":0,"generated_count":0}
    (output/"task-candidates.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"reference-candidates.json").write_text(json.dumps({"references":references,"count":len(references),
        "no_reference_was_accepted_or_uploaded":True},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    thumbs=[];font_path=Path("C:/Windows/Fonts/msyh.ttc")
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    for ref in references.values():
        if "path" not in ref:continue
        with Image.open(ROOT/ref["path"]) as im: thumb=im.resize((480,270))
        cell=Image.new("RGB",(480,310),"white");cell.paste(thumb,(0,40))
        ImageDraw.Draw(cell).text((8,7),ref["source_camera"]+" / "+ref["source_unit"],fill="black",font=font)
        thumbs.append(cell)
    sheet=Image.new("RGB",(480*3,310*((len(thumbs)+2)//3)),"#ddd")
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*310))
    temp=ROOT/".codex/docs/.temp/synthetic-reference-preparation-20261108";temp.mkdir(parents=True,exist_ok=True)
    sheet.save(temp/"reference-candidates.jpg",quality=85)
    print(json.dumps({"cameras":len(references),"local_reference_candidates":len(thumbs),"jobs":len(jobs),
        "generation_duration_sec":10,"network_requests":0,"training_ready":False},ensure_ascii=False))


def path_ref(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


class OfflineTransport:
    def request(self, *args, **kwargs):
        raise RuntimeError("Offline preparation cannot make API requests")

    def download(self, *args, **kwargs):
        raise RuntimeError("Offline preparation cannot download media")


def prepare(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    historical = {source, DEFAULT_SOURCE, ROOT / "data/metadata/synthetic-video/workflow-preparation-20261108-v1"}
    if any(output == path.resolve() or path.resolve() in output.parents for path in historical):
        raise ValueError("Output must be a new metadata version outside v1/v2/source directories")
    parent_manifest, parent_references = source / "task-candidates.json", source / "reference-candidates.json"
    manifest_bytes, reference_bytes = parent_manifest.read_bytes(), parent_references.read_bytes()
    original = json.loads(manifest_bytes.decode("utf-8-sig"))
    reference_document = json.loads(reference_bytes.decode("utf-8-sig"))
    references = reference_document["references"]
    parent_jobs = {task["job_id"]: task for task in original["tasks"]}
    jobs = []
    for old in original["tasks"]:
        if old["condition"] not in CONDITIONS:
            continue
        task = deepcopy(old)
        if (task["desired_training_label"] != "N" or task["duration_sec"] != 10
                or task["provider_request"]["model"] != "gen4_turbo" or task["provider_request"]["duration"] != 10):
            raise ValueError("Expected v2 ten-second Gen4Turbo negative candidates")
        if task["reference_candidate"] != references[task["camera_key"]]:
            raise ValueError("Task reference differs from its v2 shared reference record")
        task.update(augmentation_scope="negative_only", split="unassigned", submit_ready=False,
                    training_ready=False, selected_reference=None, accepted_label=None,
                    status="pending_label_review_split_reference_and_cost_authorization")
        controls = task["submission_controls"]
        for key in ("matched_N_P_reference_processing_history_required", "reference_processing_match_status",
                    "shared_reference_processing_history_id", "raw_P_vs_edited_N_pair_may_submit"):
            controls.pop(key, None)
        controls.update(positive_pair_required=False, processing_history_review_required=True,
                        same_source_shared_processing_history_required=True,
                        training_ratio_preregistered=False, per_source_quota_preregistered=False)
        task["reference_review_requirements"] = {
            "processing_history_reviewed": False,
            "processing_history_id_candidate": task["reference_candidate"].get("shared_processing_history"),
            "selected_reference_must_be_content_accepted": True,
            "same_source_shared_history_required": True,
            "missing_N_reference_requires_edit_and_acceptance": task["reference_edit_required"],
        }
        task["provider_preference"].update(selection_role="low_cost_baseline_expansion_candidate",
                                           quality_priority_claim=False)
        task["estimated_video_generation_usd"] = 0.5
        task["estimated_cost_caveat"] = "10s Gen4Turbo base estimate only; excludes reference editing, tax and retries; not authorization"
        jobs.append(task)
    cameras = {task["camera_key"] for task in jobs}
    counts = Counter((task["camera_key"], task["condition"]) for task in jobs)
    if (cameras != set(references) or len(jobs) != len(cameras) * 6
            or any(counts[(camera, condition)] != 2 for camera in cameras for condition in CONDITIONS)
            or len({task["job_id"] for task in jobs}) != len(jobs)):
        raise ValueError("Expected two variants of each negative condition per reference camera")
    for task in jobs:
        parent = parent_jobs[task["job_id"]]
        if (task["provider_request"] != parent["provider_request"]
                or task["parent_lineage"] != parent["parent_lineage"]
                or task["reference_candidates"] != parent["reference_candidates"]):
            raise ValueError("Request/seed or inherited parent lineage changed")
    current_paths = {
        "current_candidate_manifest": path_ref(output / "task-candidates.json"),
        "current_reference_manifest": path_ref(output / "reference-candidates.json"),
        "current_offline_plan": path_ref(output / "offline-execution-plan.json"),
        "single_preview_manifest": SINGLE_PREVIEW,
        "historical_parent_manifest": path_ref(parent_manifest),
        "historical_parent_references": path_ref(parent_references),
        "reference_images_reused_from": "data/process/synthetic-video/references-20261108-v2",
        "reference_candidate_path_base": "repository_root (preserved v2 records)",
        "execution_selected_reference_path_base": "manifest_directory_or_absolute (adapt when accepting a candidate)",
        "generated_reference_images": 0,
    }
    document = {
        **{key: value for key, value in original.items() if key != "tasks"},
        "schema_version": 3, "augmentation_scope": "negative_only", "tasks": jobs,
        "task_count": len(jobs), "tasks_per_camera": 6, "camera_count": len(cameras),
        "condition_counts": dict(sorted(Counter(task["condition"] for task in jobs).items())),
        "parent_manifest": path_ref(parent_manifest), "parent_sha256": digest(manifest_bytes),
        "parent_reference_manifest": path_ref(parent_references), "parent_reference_sha256": digest(reference_bytes),
        "all_parent_job_ids_seeds_and_reference_records_preserved": True,
        "camera_split_frozen": False, "train_camera_keys": [], "submit_ready": False, "training_ready": False,
        "public_base_video_estimate_usd": len(jobs) * 0.5, "budget_authorized": False,
        "initial_pilot_task_count": 12, "initial_pilot_base_estimate_usd": 6.0,
        "pilot_rationale": "Expansion-scale candidate only: 2 cameras x 3 negative conditions x 2 variants; no train roles frozen or batch authorized",
        "initial_expansion_candidate": {"camera_count": 2, "conditions_per_camera": 3, "variants_per_condition": 2,
                                        "task_count": 12, "base_estimate_usd": 6.0,
                                        "camera_candidates": original.get("initial_pilot_camera_candidates", []),
                                        "camera_selection_frozen": False, "authorized": False},
        "positive_synthetic_diagnostic": {"required": False, "new_tasks": 0,
                                          "historical_v2_tasks": sum(task["condition"] == "debris_flow_control" for task in original["tasks"]),
                                          "use": "Optional later style diagnostic only if generation-artifact concerns arise"},
        "current_single_preview": {"manifest": SINGLE_PREVIEW, "task_count": 1,
                                   "status": "awaiting_explicit_cost_authorization", "automatic_submission": False},
        "provider_selection_role": "Gen4Turbo low-cost candidate; not a quality-priority claim",
        "current_paths": current_paths, "network_requests_performed": 0, "credentials_read": False, "generated_count": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "task-candidates.json", document)
    atomic_json(output / "reference-candidates.json", {
        **reference_document, "schema_version": 3, "count": len(references),
        "parent_manifest": path_ref(parent_references), "parent_sha256": digest(reference_bytes),
        "reference_records_reused_without_change": True, "image_files_copied_or_modified": 0,
    })
    atomic_json(output / "current-paths.json", current_paths)
    workflow = RunwayWorkflow(output / "task-candidates.json", output / "task-state.json", OfflineTransport())
    plan = workflow.plan()
    if plan["network_requests"] != 0 or plan["credentials_read"] or plan["estimated_base_cost_usd"] != "%.2f" % (len(jobs) * 0.5):
        raise ValueError("Offline plan did not preserve the expected base-only estimate")
    atomic_json(output / "offline-execution-plan.json", plan)
    return {"output": path_ref(output), "cameras": len(cameras), "reference_candidates": len(references),
            "tasks": len(jobs), "conditions": document["condition_counts"],
            "duration_sec": 10, "base_estimate_usd": document["public_base_video_estimate_usd"],
            "expansion_candidate_tasks": 12, "expansion_base_estimate_usd": 6.0,
            "network_requests": 0, "credentials_read": False, "image_files_modified": 0,
            "all_tasks_unassigned_and_not_ready": True,
            "offline_plan_blocker_counts": dict(Counter(reason for task in plan["jobs"] for reason in task["blockers"]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
