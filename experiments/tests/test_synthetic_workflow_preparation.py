"""One offline fixture covering source eligibility, labels, identity, and gates."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/prepare_synthetic_video_workflow.py"
SPEC = importlib.util.spec_from_file_location("synthetic_workflow_preparation", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class OfflinePreparationTest(unittest.TestCase):
    def test_explicit_saved_labels_references_stable_jobs_and_blocked_workflow(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-workflow-preparation-") as directory:
            root = Path(directory)
            source_root = root / "data/process/video_frames/fixture"
            source_root.mkdir(parents=True)
            version = root / "saved-v3"
            (version / "fixture").mkdir(parents=True)
            pool = root / "pool.csv"
            with pool.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["unit_id", "source", "video_id", "group_id", "video_label", "middleware_path"])
                writer.writerows([
                    ["a", "dongchuan", "source-a", "d|camera A|2026-01-01", "pos", "media/a.mp4"],
                    ["b", "dongchuan", "source-b", "d|camera A|2026-01-02", "neg", "media/b.mp4"],
                    ["c", "dongchuan", "source-c", "d|camera B|2026-01-03", "pos", "media/c.mp4"],
                    ["d", "video4", "source-d", "d|camera A|2026-01-01", "pos", "media/d.mp4"],
                    ["e", "video5", "source-e", "d|camera C|2026-01-05", "neg", "media/e.mp4"],
                    ["external", "external_test", "test-video", "e|external camera|2026-01-04", "neg", "media/test.mp4"],
                ])
            with (source_root / "mapping.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video_key", "video_id", "excluded"])
                writer.writerows([[uid, "source-" + uid, 0] for uid in ("a", "b", "c", "external")])
            events = version / "fixture/events.csv"
            with events.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(tool.EVENT_FIELDS + ["unrelated_extra_column"])
                writer.writerows([
                    ["source-a", "a", 0, 2, "real", "neg", "explicit N", "preserve source file"],
                    ["source-a", "a", 2, 4, "real", "cut", "not a negative", "extra"],
                    ["source-a", "a", 4, 5, "real", "mystery", "not positive", "extra"],
                    ["source-a", "a", 5, 500, "auto", "neg", "must be ignored", "extra"],
                    ["source-b", "b", 10, 12, "real", "pos", "note says N but is not truth", "extra"],
                    ["source-c", "c", 0, 2, "real", "transition", "candidate only", "extra"],
                    ["source-c", "c", 3, 4, "real", "pos", "gap 2-3 remains unknown", "extra"],
                    ["test-video", "external", 0, 999, "real", "neg", "external test excluded", "extra"],
                ])
            manifest = version / "sources.json"
            source_entries = {"fixture": {"source_id": "data/process/video_frames/fixture", "annotations_csv": "fixture/events.csv"}}
            additional_inputs = []
            for name, uid, kind, end in (("video4", "d", "pos", 3), ("video5", "e", "neg", 4)):
                extra_source = root / "data/process/video_frames" / name
                extra_source.mkdir()
                extra_mapping = extra_source / "mapping.csv"
                extra_mapping.write_text(f"video_key,video_id,excluded\n{uid},source-{uid},0\n", encoding="utf-8")
                extra_events = version / name / "events.csv"
                extra_events.parent.mkdir()
                with extra_events.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(tool.EVENT_FIELDS)
                    writer.writerow(["source-" + uid, uid, 0, end, "real", kind, "cross-source fixture"])
                source_entries[name] = {"source_id": "data/process/video_frames/" + name, "annotations_csv": name + "/events.csv"}
                additional_inputs.extend([extra_mapping, extra_events])
            manifest.write_text(json.dumps({"training_ready": False, "sources": source_entries}), encoding="utf-8")
            input_paths = [pool, events, manifest, source_root / "mapping.csv"] + additional_inputs
            input_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
            out = root / "out"
            result = tool.prepare(pool, manifest, out, root)
            self.assertEqual({path.name for path in out.iterdir()},
                             {"scene-inventory.json", "task-candidates.json", "workflow.json"})
            inventory = json.loads((out / "scene-inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["totals"]["video_count"], 5)
            self.assertEqual(inventory["totals"]["interval_seconds"], {"N": 6, "T": 2, "P": 6, "cut": 2, "unknown": 1})
            self.assertEqual(len(inventory["event_groups"]), 4)
            self.assertEqual(inventory["cameras_missing_explicit_N"], ["camera B"])
            self.assertEqual(inventory["annotation_audit"]["ignored_rows"], {"non_real_rows": 1, "outside_pool_rows": 1})
            self.assertEqual(inventory["excluded_pool_rows"][0]["unit_id"], "external")
            cameras = {camera["source_camera"]: camera for camera in inventory["cameras"]}
            self.assertEqual(cameras["camera B"]["interval_seconds"]["N"], 0)  # cut, gaps, and unknown never supply N.
            self.assertEqual(cameras["camera A"]["videos_with_explicit_intervals"]["N"], 1)
            self.assertEqual(cameras["camera A"]["interval_seconds"]["P"], 5)  # Saved labels merge across sources; old video_label/note cannot override P.
            self.assertEqual(cameras["camera A"]["video_count"], 3)
            merged_group = next(group for group in inventory["event_groups"] if group["group_id"] == "d|camera A|2026-01-01")
            self.assertEqual(merged_group["source_units"], ["a", "d"])
            self.assertEqual(merged_group["interval_seconds"]["P"], 3)
            self.assertEqual(cameras["camera C"]["source_units"], ["e"])
            self.assertEqual({unit["unit_id"]: unit["source"] for unit in inventory["units"]},
                             {"a": "dongchuan", "b": "dongchuan", "c": "dongchuan", "d": "video4", "e": "video5"})
            self.assertTrue(all(not unit["training_ready"] for unit in inventory["units"]))
            tasks = result["task_candidates"]["tasks"]
            self.assertEqual(len(tasks), 24)
            self.assertEqual(len({task["job_id"] for task in tasks}), 24)
            for name in cameras:
                camera_tasks = [task for task in tasks if task["source_camera"] == name]
                self.assertEqual(len(camera_tasks), 8)
                self.assertEqual({condition: sum(task["condition"] == condition for task in camera_tasks)
                                  for condition in tool.CONDITIONS}, {condition: 2 for condition in tool.CONDITIONS})
            for task in tasks:
                self.assertEqual(task["status"], "blocked_unfrozen_split")
                self.assertFalse(task["submit_ready"])
                self.assertFalse(task["training_ready"])
                self.assertIsNone(task["accepted_label"])
                self.assertIsNone(task["estimated_cost"])
                self.assertEqual(task["duration_sec"], 5)
                self.assertEqual(task["split"], "train_candidate_only")
                self.assertEqual(task["independent_scene_increment"], 0)
                self.assertNotIn("external", task["source_units"])
                self.assertFalse(task["variation_key_is_provider_seed"])
                self.assertEqual(task["protocol_id"], "cross-camera-fixed-view-v1")
                self.assertEqual(task["camera_key"], "s|" + task["source_camera"])
                self.assertEqual(task["label_spec_path"], "docs/data/NPT标签定义与量化核查.md")
                support = task["effective_support_requirements"]
                self.assertEqual(support["minimum_contiguous_effective_support_sec"], 3)
                self.assertEqual(support["minimum_distinct_frames"], 16)
                self.assertTrue(support["strictly_increasing_real_pts_required"])
                self.assertTrue(support["first_and_last_effective_frames_must_be_qc_checked"])
                self.assertIsNone(support["fixed_edge_trim_sec"])
                self.assertFalse(support["repeated_frame_padding_allowed"])
                self.assertFalse(support["blind_edge_trimming_allowed"])
                self.assertIsNone(support["accepted_effective_duration_sec"])
                if task["condition"] in {"ordinary_water", "flood"}:
                    self.assertIn("large-scale", task["prompt_en"])
                    self.assertNotIn("small ordinary water stream", task["prompt_en"])
                controls = task["submission_controls"]
                self.assertTrue(controls["matched_N_P_reference_processing_history_required"])
                self.assertFalse(controls["raw_P_vs_edited_N_pair_may_submit"])
                self.assertTrue(controls["training_ratio_preregistration_required"])
                self.assertTrue(controls["per_source_quota_preregistration_required"])
                self.assertFalse(controls["holdout_may_select_prompts_models_or_samples"])
                if task["desired_training_label"] == "N" and task["source_camera"] == "camera A":
                    self.assertEqual(task["reference_candidate_scope"], "explicit_N_interval_candidates_first")
                    self.assertEqual([ref["source_unit"] for ref in task["reference_candidates"]], ["a"])
                    self.assertEqual(task["reference_candidates"][0]["candidate_intervals"], [{"start_sec": 0, "end_sec": 2, "label": "N"}])
                if task["desired_training_label"] == "N" and task["source_camera"] == "camera B":
                    self.assertTrue(task["reference_edit_required"])
                    self.assertEqual(task["reference_candidate_scope"], "reference_edit_required")
                    self.assertTrue(all(ref["reference_edit_required"] and not ref["clean_negative_reference_verified"]
                                        and not ref["candidate_intervals"] for ref in task["reference_candidates"]))
            units = {unit["unit_id"]: unit for unit in inventory["units"]}
            reordered = tool.build_tasks(list(reversed(inventory["cameras"])), dict(reversed(list(units.items()))))
            self.assertEqual({task["job_id"]: task["variation_key"] for task in tasks},
                             {task["job_id"]: task["variation_key"] for task in reordered})
            workflow = result["workflow"]
            self.assertFalse(workflow["budget"]["authorized"])
            self.assertEqual(workflow["first_batch_scale_only"]["selected_cameras"], [])
            self.assertEqual(workflow["first_batch_scale_only"]["task_count"], 16)
            self.assertEqual(workflow["first_batch_scale_only"]["N_intent_count"], 12)
            self.assertEqual(workflow["first_batch_scale_only"]["P_intent_count"], 4)
            self.assertFalse(workflow["first_batch_scale_only"]["class_balanced"])
            self.assertEqual(workflow["first_batch_scale_only"]["independent_new_scene_count"], 0)
            self.assertEqual(workflow["submission_controls"], tool.submission_controls())
            self.assertEqual(workflow["training_sampling_preregistration"]["status"], "not_registered")
            self.assertTrue(workflow["constraints"]["matched_reference_processing_history_across_N_and_P_required"])
            self.assertTrue(workflow["constraints"]["held_out_data_not_used_for_prompt_model_or_sample_selection"])
            self.assertFalse(workflow["retry_policy"]["infinite_retries"])
            self.assertEqual(workflow["retry_policy"]["max_reasoned_revision_retries"], 1)
            self.assertTrue(workflow["constraints"]["real_validation_and_test_contain_no_synthetic"])
            self.assertTrue(workflow["constraints"]["all_reference_descendants_remain_in_parent_camera_split"])
            self.assertTrue(workflow["split_policy"]["all_dates_sources_and_descendants_same_side"])
            self.assertTrue(workflow["split_policy"]["inner_selection_same_camera_isolation"])
            self.assertIsNone(workflow["split_policy"]["fixed_fold_count"])
            self.assertTrue(workflow["constraints"]["legacy_T_not_automatically_N_or_P"])
            self.assertTrue(workflow["constraints"]["legacy_cut_reason_unresolved_not_assumed_all_bad_or_all_normal"])
            self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}, input_hashes)


if __name__ == "__main__":
    unittest.main()
