"""Execution-boundary fixtures; all HTTP and media receipt operations are mocked."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.common.video.synthetic_runway import (
    API_BASE, API_VERSION, MAX_DATA_URI_BYTES, RunwayWorkflow, digest,
)
from experiments.tools.run_synthetic_video_jobs import main


class MockTransport:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.download_calls = []

    def request(self, method, url, headers, body=None):
        self.calls.append((method, url, dict(headers), body))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        status, response = value
        return status, json.dumps(response).encode("utf-8")

    def download(self, url, stream):
        self.download_calls.append(url)
        stream.write(b"mock-video-bytes-not-a-QC-accepted-video")


class SyntheticRunwayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path, self.state_path = self.root / "manifest.json", self.root / "state.json"
        self.authorization_path = self.root / "authorization.json"
        self.reference = self.root / "accepted.jpg"
        # Header-only transport fixture; acceptance is an input, not image QC by this client.
        self.reference.write_bytes(b"\xff\xd8\xff\xe0accepted-reference-fixture\xff\xd9")
        self.job = {
            "job_id": "syn-camera-no_target-v01", "source_camera": "registered-camera-A",
            "synthetic": True, "split": "train", "submit_ready": True, "status": "reviewed_for_submission",
            "desired_training_label": "N", "accepted_label": None, "training_ready": False,
            "provider_request": {"model": "gen4_turbo", "ratio": "1280:720", "duration": 10,
                                 "promptText": "A continuous 10-second shot from the same fixed camera, with ordinary flowing water.", "seed": 123},
            "selected_reference": {
                "path": "accepted.jpg", "sha256": digest(self.reference.read_bytes()), "accepted": True,
                "source_camera": "registered-camera-A", "split": "train", "lineage_complete": True,
                "processing_history_accepted": True, "processing_history_id": "matched-edit-v1",
                "parent_lineage": [{"source_camera": "registered-camera-A", "split": "train",
                                    "source_unit": "original-1", "source_group": "camera-A|date-1", "source_media_sha256": "ab" * 32}],
            },
            "submission_controls": {"reference_processing_match_status": "accepted",
                                    "shared_reference_processing_history_id": "matched-edit-v1",
                                    "training_ratio_preregistered": True, "per_source_quota_preregistered": True,
                                    "holdout_may_select_prompts_models_or_samples": False},
        }
        self.manifest = {"schema_version": 1, "camera_partition": {"status": "frozen", "split_id": "fixture-split",
                         "roles": {"registered-camera-A": "train", "registered-camera-B": "test"}}, "jobs": [self.job]}
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def authorization(self, cap="0.50"):
        self.authorization_path.write_text(json.dumps({
            "approved": True, "provider": "runway_dev", "authorization_id": "fixture-only-not-real-approval",
            "approved_by": "fixture", "approved_at": "2026-11-08T00:00:00Z",
            "manifest_sha256": digest(self.manifest_path.read_bytes()), "state_path": str(self.state_path),
            "allowed_job_ids": [job["job_id"] for job in self.manifest["jobs"]], "max_cost_usd": cap,
        }), encoding="utf-8")
        return self.authorization_path

    def execute(self, workflow, ids=None, cap="0.50"):
        return workflow.submit(ids or [self.job["job_id"]], execute=True,
                               authorization_path=self.authorization_path, max_cost_usd=cap)

    def preview_fixture(self):
        self.job.update(kind="negative_candidate", preview_only=True, training_ready=False)
        self.job.pop("split")
        self.job.pop("submission_controls")
        self.job["selected_reference"].pop("split")
        self.job["selected_reference"].pop("processing_history_id")
        self.job["selected_reference"].pop("processing_history_accepted")
        self.job["selected_reference"]["parent_lineage"][0]["split"] = "development_source_not_external_test"
        self.manifest = {"schema_version": 1, "purpose": "media_preview", "preview_only": True, "training_ready": False,
                         "reference_use": {"external_test": False, "frozen_test_cameras": [],
                                           "checked_current_frozen_test_list": True}, "jobs": [self.job]}
        self.write_manifest()

    def test_default_plan_and_dry_run_read_no_key_make_no_request_or_state(self):
        transport = MockTransport()
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("no credential read")):
            plan = workflow.plan()
            self.assertEqual(plan["estimated_base_cost_usd"], "0.50")
            self.assertEqual(plan["jobs"][0]["blockers"], [])
            self.assertNotIn("data:image/jpeg;base64,", json.dumps(plan))
            workflow.submit([self.job["job_id"]])
            workflow.poll([self.job["job_id"]])
            workflow.download([self.job["job_id"]], self.root / "out")
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["submit", "--manifest", str(self.manifest_path), "--execute", "--dry-run"]), 0)
                self.assertEqual(json.loads(output.getvalue())["network_requests"], 0)
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.state_path.exists())
        self.assertFalse((self.root / "out").exists())
        # The original preparation schema remains blocked; seed/10s aren't silently filled in.
        self.manifest = {"status": "blocked_unfrozen_split", "tasks": [{"job_id": "original-88-candidate", "submit_ready": False,
                         "split": "train_candidate_only", "duration_sec": 5, "prompt_en": "5-second shot"}]}
        self.write_manifest()
        blocked = RunwayWorkflow(self.manifest_path, self.state_path, transport).plan()
        self.assertTrue(blocked["jobs"][0]["blockers"])
        self.assertFalse(blocked["jobs"][0]["submit_ready_in_source"])
        self.assertIsNone(blocked["estimated_base_cost_usd"])
        self.assertEqual(blocked["unpriced_job_count"], 1)

    def test_authorization_train_lineage_and_reference_gates_precede_network(self):
        transport = MockTransport()
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("must gate first")):
            with self.assertRaises(ValueError):
                workflow.submit([self.job["job_id"]], execute=True)
            self.authorization()
            auth = json.loads(self.authorization_path.read_text())
            auth["approved"] = False
            self.authorization_path.write_text(json.dumps(auth))
            with self.assertRaises(ValueError):
                self.execute(workflow)
            original = deepcopy(self.manifest)
            for change in ("test_camera", "test_ancestor", "missing_history", "unaccepted", "blocked", "hash_changed"):
                with self.subTest(change=change):
                    self.manifest = deepcopy(original)
                    job = self.manifest["jobs"][0]
                    if change == "test_camera":
                        job["source_camera"] = "registered-camera-B"
                    elif change == "test_ancestor":
                        job["selected_reference"]["parent_lineage"][0]["source_camera"] = "registered-camera-B"
                    elif change == "missing_history":
                        job["selected_reference"]["processing_history_accepted"] = False
                    elif change == "unaccepted":
                        job["selected_reference"]["accepted"] = False
                    elif change == "blocked":
                        job["status"] = "blocked_unfrozen_split"
                    else:
                        job["selected_reference"]["sha256"] = "00" * 32
                    self.write_manifest()
                    self.authorization()
                    with self.assertRaises(ValueError):
                        self.execute(RunwayWorkflow(self.manifest_path, self.state_path, transport))
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.state_path.exists())

    def test_submit_poll_download_and_resume_never_resubmit_or_leak_key(self):
        self.authorization()
        task_id, secret = "provider-task-1", "fixture-api-secret-do-not-persist"
        url = "https://cdn.example.test/generated.mp4?signature=fixture"
        transport = MockTransport([(200, {"id": task_id}), (200, {"id": task_id, "status": "SUCCEEDED", "output": [url]})])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value=secret):
            result = self.execute(workflow)
            self.assertEqual(result["jobs"][self.job["job_id"]]["status"], "submitted")
            request_body = json.loads(transport.calls[0][3])
            self.assertEqual(set(request_body), {"model", "ratio", "duration", "promptText", "seed", "promptImage"})
            self.assertEqual(request_body["duration"], 10)
            self.assertTrue(request_body["promptImage"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(transport.calls[0][1], API_BASE + "/v1/image_to_video")
            self.assertEqual(transport.calls[0][2]["X-Runway-Version"], API_VERSION)
            result = workflow.poll([self.job["job_id"]], execute=True)
            self.assertEqual(result["jobs"][self.job["job_id"]]["status"], "succeeded")
        output_dir = self.root / "received"
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("download needs no key")):
            result = workflow.download([self.job["job_id"]], output_dir, execute=True)
            RunwayWorkflow(self.manifest_path, self.state_path, transport).download([self.job["job_id"]], output_dir, execute=True)
            self.execute(RunwayWorkflow(self.manifest_path, self.state_path, transport))
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(transport.download_calls, [url])
        receipt = result["jobs"][self.job["job_id"]]["downloads"]["0"]
        self.assertEqual(digest(Path(receipt["path"]).read_bytes()), receipt["sha256"])
        self.assertFalse(result["training_ready"])
        self.assertIsNone(result["jobs"][self.job["job_id"]]["accepted_label"])
        self.assertNotIn(secret, self.state_path.read_text())

    def test_unknown_post_receipt_reserves_budget_and_does_not_repeat(self):
        self.authorization()
        transport = MockTransport([TimeoutError("server might have accepted")])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value="fixture-secret"):
            result = self.execute(workflow)
        self.assertEqual(result["jobs"][self.job["job_id"]]["status"], "submission_unknown")
        self.assertEqual(result["reserved_cost_cents"], 50)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("no retry credential read")):
            result = self.execute(RunwayWorkflow(self.manifest_path, self.state_path, transport))
            with self.assertRaises(ValueError):
                workflow.poll([self.job["job_id"]], execute=True)
        self.assertEqual(len(transport.calls), 1)
        self.assertNotIn("server might have accepted", self.state_path.read_text())

    def test_budget_is_cumulative_and_never_exceeds_either_cap(self):
        second = deepcopy(self.job)
        second["job_id"] = "syn-camera-no_target-v02"
        self.manifest["jobs"].append(second)
        self.write_manifest()
        self.authorization("1.00")
        transport = MockTransport([(200, {"id": "task-1"})])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        both = [self.job["job_id"], second["job_id"]]
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value="fixture-secret"):
            with self.assertRaises(ValueError):
                self.execute(workflow, both, cap="0.99")
            self.assertEqual(transport.calls, [])
            self.execute(workflow, cap="0.50")
            with self.assertRaises(ValueError):
                self.execute(workflow, [second["job_id"]], cap="0.50")
            self.authorization("0.50")
            with self.assertRaises(ValueError):
                self.execute(workflow, [second["job_id"]], cap="10.00")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(json.loads(self.state_path.read_text())["jobs"][self.job["job_id"]]["reserved_cost_cents"], 50)

    def test_encoded_limit_and_failed_or_non_https_outputs_not_downloaded(self):
        with self.reference.open("wb") as stream:
            stream.truncate(MAX_DATA_URI_BYTES)
        self.job["selected_reference"]["sha256"] = digest(self.reference.read_bytes())
        self.write_manifest()
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, MockTransport())
        self.assertIn("encoded data URI", workflow.plan()["jobs"][0]["blockers"][0])
        # Start another fixture submission using the original tiny accepted reference.
        self.reference.write_bytes(b"\xff\xd8\xfffixture\xff\xd9")
        self.job["selected_reference"]["sha256"] = digest(self.reference.read_bytes())
        self.write_manifest()
        self.authorization()
        transport = MockTransport([(200, {"id": "task-1"}),
                                   (200, {"id": "task-1", "status": "SUCCEEDED", "output": ["http://unsafe.test/video"]}),
                                   (200, {"id": "task-1", "status": "FAILED", "failure": "fixture-secret rejected", "failureCode": "MODERATION"})])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value="fixture-secret"):
            self.execute(workflow)
            result = workflow.poll([self.job["job_id"]], execute=True)
            self.assertEqual(result["jobs"][self.job["job_id"]]["status"], "submitted")
            with self.assertRaises(ValueError):
                workflow.download([self.job["job_id"]], self.root / "out", execute=True)
            result = workflow.poll([self.job["job_id"]], execute=True)
            self.assertEqual(result["jobs"][self.job["job_id"]]["status"], "failed")
            self.assertEqual(result["jobs"][self.job["job_id"]]["failure_code"], "MODERATION")
            self.assertNotIn("fixture-secret", self.state_path.read_text())
        self.assertEqual(transport.download_calls, [])

    def test_each_model_has_explicit_schema_and_correct_offline_cost(self):
        cases = [("gen4_turbo", "1280:720", True, "0.50"),
                 ("gen4.5", "1280:720", True, "1.20"),
                 ("seedance2_5", "1280:720", True, "3.00"),
                 ("kling3.0_pro", "1920:1080", False, "1.20")]
        jobs = []
        for index, (model, ratio, supports_seed, _) in enumerate(cases):
            job = deepcopy(self.job)
            job["job_id"] = "model-fixture-%d" % index
            job["provider_request"].update(model=model, ratio=ratio)
            if not supports_seed:
                job["provider_request"].pop("seed")
            jobs.append(job)
        self.manifest["jobs"] = jobs
        self.write_manifest()
        transport = MockTransport()
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("offline model check")):
            plan = workflow.plan()
            self.assertEqual(plan["estimated_base_cost_usd"], "5.90")
            self.assertEqual(plan["unpriced_job_count"], 0)
            for job, row, (model, ratio, supports_seed, cost) in zip(jobs, plan["jobs"], cases):
                with self.subTest(model=model):
                    self.assertEqual(row["base_generation_cost_usd"], cost)
                    self.assertEqual(row["blockers"], [])
                    body, _ = workflow.validate_job(job)
                    self.assertEqual(body["ratio"], ratio)
                    self.assertEqual(body["duration"], 10)
                    self.assertEqual("seed" in body, supports_seed)
                    self.assertNotIn("referenceVideos", body)
                    self.assertNotIn("referenceAudio", body)
                    if model != "gen4_turbo":
                        self.assertEqual(len(body["promptImage"]), 1)
                        self.assertEqual(body["promptImage"][0]["position"], "first")
                    if model in {"seedance2_5", "kling3.0_pro"}:
                        self.assertIs(body["audio"], False)
                    if model == "seedance2_5":
                        self.assertIs(body["draft"], False)
                        self.assertEqual(body["outputFormat"], "mp4")
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(["--list-models"]), 0)
                catalog = json.loads(output.getvalue())
                self.assertEqual(catalog["seedance2_5"]["base_cost_usd"], "3.00")
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.state_path.exists())

    def test_model_parameter_mismatches_reject_before_key_or_network(self):
        cases = [("seedance2_5", {"ratio": "adaptive"}),
                 ("seedance2_5", {"ratio": "1920:1080"}),
                 ("seedance2_5", {"audio": True}),
                 ("seedance2_5", {"draft": True}),
                 ("seedance2_5", {"referenceVideos": ["https://example.test/reference.mp4"]}),
                 ("seedance2_5", {"referenceAudio": []}),
                 ("kling3.0_pro", {"ratio": "1280:720"}),
                 ("kling3.0_pro", {"audio": True}),
                 ("kling3.0_pro", {"seed": 123}),
                 ("gen4.5", {"outputFormat": "prores"}),
                 ("unverified-model", {})]
        transport = MockTransport()
        original_spec = deepcopy(self.job["provider_request"])
        for model, invalid in cases:
            with self.subTest(model=model, invalid=invalid):
                spec = deepcopy(original_spec)
                spec.update(model=model, ratio="1920:1080" if model == "kling3.0_pro" else "1280:720")
                if model == "kling3.0_pro":
                    spec.pop("seed")
                spec.update(invalid)
                self.manifest["jobs"][0]["provider_request"] = spec
                self.write_manifest()
                self.authorization("10.00")
                workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
                self.assertIsNone(workflow.plan()["estimated_base_cost_usd"])
                with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("bad model gate first")):
                    with self.assertRaises(ValueError):
                        self.execute(workflow, cap="10.00")
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.state_path.exists())

    def test_seedance_reserves_three_dollars_and_cannot_use_turbo_budget(self):
        self.job["provider_request"]["model"] = "seedance2_5"
        self.write_manifest()
        self.authorization("3.00")
        transport = MockTransport([(200, {"id": "seedance-task-1"})])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("budget gate first")):
            with self.assertRaises(ValueError):
                self.execute(workflow, cap="0.50")
            self.authorization("2.99")
            with self.assertRaises(ValueError):
                self.execute(workflow, cap="3.00")
        self.assertEqual(transport.calls, [])
        self.authorization("3.00")
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value="fixture-secret"):
            result = self.execute(workflow, cap="3.00")
        self.assertEqual(result["reserved_cost_cents"], 300)
        payload = json.loads(transport.calls[0][3])
        self.assertEqual(payload["promptImage"][0]["position"], "first")
        self.assertIs(payload["audio"], False)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("do not resubmit")):
            self.execute(RunwayWorkflow(self.manifest_path, self.state_path, transport), cap="3.00")
        self.assertEqual(len(transport.calls), 1)
        # A persisted Seedance job cannot masquerade as a 50-cent reservation.
        state = json.loads(self.state_path.read_text())
        state["jobs"][self.job["job_id"]]["reserved_cost_cents"] = 50
        self.state_path.write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            self.execute(workflow, cap="3.00")

    def test_single_preview_needs_no_formal_partition_or_P_pair_but_still_needs_cost_authorization(self):
        self.preview_fixture()
        transport = MockTransport([(200, {"id": "preview-task-1"})])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("no authorized execution")):
            plan = workflow.plan()
            self.assertEqual(plan["purpose"], "media_preview")
            self.assertEqual(plan["jobs"][0]["blockers"], [])
            self.assertEqual(plan["estimated_base_cost_usd"], "0.50")
            self.assertNotIn("freeze camera_partition", json.dumps(plan["adaptation_required"]))
            self.assertNotIn("N/P", json.dumps(plan["adaptation_required"]))
            self.assertFalse(self.state_path.exists())
            with self.assertRaises(ValueError):
                workflow.submit([self.job["job_id"]], execute=True)
        self.assertEqual(transport.calls, [])
        # This approval exists only in the synthetic fixture; no real authorization is created.
        self.authorization()
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value="fixture-secret"):
            result = self.execute(workflow)
        receipt = result["jobs"][self.job["job_id"]]
        self.assertTrue(receipt["preview_only"])
        self.assertFalse(receipt["training_ready"])
        self.assertIsNone(receipt["accepted_label"])
        use = json.loads(self.state_path.read_text())["camera_uses"][0]
        self.assertEqual(use["activity"], "generation_preview")
        self.assertEqual(use["source_camera"], self.job["source_camera"])
        self.assertFalse(use["independent_holdout_for_choices_using_this_preview"])

    def test_preview_cannot_bypass_frozen_or_external_test_reference(self):
        self.preview_fixture()
        original = deepcopy(self.manifest)
        transport = MockTransport()
        for case in ("frozen_list", "partition_test", "external_scope", "reference_test", "parent_test"):
            with self.subTest(case=case):
                self.manifest = deepcopy(original)
                job = self.manifest["jobs"][0]
                if case == "frozen_list":
                    self.manifest["reference_use"]["frozen_test_cameras"] = [job["source_camera"]]
                elif case == "partition_test":
                    self.manifest["camera_partition"] = {"status": "frozen", "roles": {job["source_camera"]: "test"}}
                elif case == "external_scope":
                    self.manifest["reference_use"]["external_test"] = True
                elif case == "reference_test":
                    job["selected_reference"]["split"] = "external_test"
                else:
                    job["selected_reference"]["parent_lineage"][0]["split"] = "test"
                self.write_manifest()
                self.authorization()
                workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
                self.assertTrue(workflow.plan()["jobs"][0]["blockers"])
                with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("test source gate first")):
                    with self.assertRaises(ValueError):
                        self.execute(workflow)
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.state_path.exists())

    def test_preview_rejects_multiple_jobs_and_pretended_training_readiness(self):
        self.preview_fixture()
        original = deepcopy(self.manifest)
        transport = MockTransport()
        for case in ("multiple", "training_ready", "manifest_training_ready", "wrong_kind", "not_preview"):
            with self.subTest(case=case):
                self.manifest = deepcopy(original)
                job = self.manifest["jobs"][0]
                if case == "multiple":
                    second = deepcopy(job)
                    second["job_id"] = "second-preview-job"
                    self.manifest["jobs"].append(second)
                elif case == "training_ready":
                    job["training_ready"] = True
                elif case == "manifest_training_ready":
                    self.manifest["training_ready"] = True
                elif case == "wrong_kind":
                    job["kind"] = "training_positive"
                else:
                    job["preview_only"] = False
                self.write_manifest()
                self.authorization()
                workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
                with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("preview scope gate first")):
                    with self.assertRaises(ValueError):
                        self.execute(workflow)
        self.assertEqual(transport.calls, [])
        self.assertFalse(self.state_path.exists())

    def test_negative_only_formal_batch_needs_reviewed_source_history_not_P_pair(self):
        jobs = []
        for condition in ("no_target", "ordinary_water", "flood"):
            job = deepcopy(self.job)
            job.update(job_id="negative-only-" + condition, condition=condition)
            job["submission_controls"].pop("reference_processing_match_status")
            job["submission_controls"].pop("shared_reference_processing_history_id")
            reference = job["selected_reference"]
            reference.pop("processing_history_accepted")
            reference.update(processing_history_reviewed=True, processing_history_id="negative-source-processing-v1")
            jobs.append(job)
        self.manifest.update(augmentation_scope="negative_only", jobs=jobs)
        self.write_manifest()
        ready = deepcopy(self.manifest)
        transport = MockTransport([(200, {"id": "negative-task-%d" % index}) for index in range(3)])
        workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
        with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("offline N-only plan")):
            plan = workflow.plan()
            self.assertTrue(all(not row["blockers"] for row in plan["jobs"]))
            self.assertEqual(plan["augmentation_scope"], "negative_only")
            self.assertEqual(plan["estimated_base_cost_usd"], "1.50")
        self.authorization("1.50")
        with patch("experiments.common.video.synthetic_runway.load_api_secret", return_value="fixture-secret"):
            result = self.execute(workflow, [job["job_id"] for job in jobs], cap="1.50")
        self.assertEqual(len(transport.calls), 3)
        self.assertTrue(all(row["status"] == "submitted" and row["accepted_label"] is None
                            and row["augmentation_scope"] == "negative_only" for row in result["jobs"].values()))
        # The old schema still requires its original matched N/P controls.
        self.manifest.pop("augmentation_scope")
        self.write_manifest()
        legacy = RunwayWorkflow(self.manifest_path, self.state_path, transport).plan()
        self.assertTrue(all("Matched N/P" in row["blockers"][0] for row in legacy["jobs"]))
        for case in ("P_condition", "P_label", "unreviewed_history", "source_history_conflict", "missing_quota"):
            with self.subTest(case=case):
                self.manifest = deepcopy(ready)
                job = self.manifest["jobs"][0]
                job["job_id"] = "new-attempt-" + case
                if case == "P_condition":
                    job["condition"] = "debris_flow_control"
                elif case == "P_label":
                    job["desired_training_label"] = "P"
                elif case == "unreviewed_history":
                    job["selected_reference"]["processing_history_reviewed"] = False
                elif case == "source_history_conflict":
                    # Another accepted reference from the same parent cannot use another history.
                    self.manifest["jobs"][1]["selected_reference"]["processing_history_id"] = "different-processing"
                else:
                    job["submission_controls"]["per_source_quota_preregistered"] = False
                self.write_manifest()
                self.authorization("10.00")
                workflow = RunwayWorkflow(self.manifest_path, self.state_path, transport)
                with patch("experiments.common.video.synthetic_runway.load_api_secret", side_effect=AssertionError("N-only contract gate first")):
                    with self.assertRaises(ValueError):
                        self.execute(workflow, [job["job_id"]], cap="10.00")
        self.assertEqual(len(transport.calls), 3)


if __name__ == "__main__":
    unittest.main()
