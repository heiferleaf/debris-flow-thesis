"""Wan execution tests: dummy credentials, mocked provider and temporary media only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.common.video.synthetic_wan import (
    API_BASE, PROFILE, WanWorkflow, WorkflowError, digest, technical_qc,
)


class MockTransport:
    def __init__(self, responses=(), download_error=None):
        self.responses, self.calls, self.downloads = list(responses), [], []
        self.download_error = download_error

    def request(self, method, url, headers, body=None):
        self.calls.append((method, url, headers, body))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        status, value = result
        return status, json.dumps(value).encode("utf-8")

    def download(self, url, stream):
        self.downloads.append(url)
        if self.download_error:
            raise self.download_error
        stream.write(b"mock-mp4-not-an-accepted-video")


class WanFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "data/accepted.jpg"
        self.reference.parent.mkdir()
        self.reference.write_bytes(b"\xff\xd8\xffdummy-reference\xff\xd9")
        self.manifest_path = self.root / "manifest.json"
        self.state_path = self.root / "state.json"
        self.auth_path = self.root / "authorization.json"
        self.job_id = "preview-fixture-01"
        self.manifest = {"schema_version": 1, "purpose": "media_preview", "training_ready": False,
            "reference_use": {"external_test": False, "checked_current_frozen_test_list": True, "frozen_test_cameras": []},
            "scenarios": [{"scenario_id": "camera-a", "camera_name": "camera A", "unit_id": "unit-1", "ref_time_sec": 300,
                "reference_path": "data/accepted.jpg", "reference_sha256": digest(self.reference.read_bytes()),
                "label_state": "user_accepted_N_anchor_preview_only", "reference_status": "accepted_for_single_media_preview",
                "needs_target_removal": False}],
            "jobs": [{"id": self.job_id, "scenario_id": "camera-a", "title": "Fixture preview", "kind": "negative_candidate",
                "source_camera": "camera A", "preview_only": True, "training_ready": False, "accepted_label": None,
                "prompt_zh": "固定镜头，普通水流，地貌稳定。", "prompt_en": "A locked-off camera records ordinary water in one 10-second shot.",
                "settings": deepcopy(PROFILE), "selected_reference": {"path": "data/accepted.jpg",
                    "sha256": digest(self.reference.read_bytes()), "accepted": True, "source_camera": "camera A",
                    "lineage_complete": True, "parent_lineage": [{"source_camera": "camera A", "source_unit": "unit-1",
                        "source_group": "cameraA-date1", "source_media_sha256": "ab" * 32, "split": "development"}]}}]}
        self.write_manifest()
        self.secret_reads = []

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def secret(self):
        self.secret_reads.append(True)
        return "dummy-not-a-real-key"

    def workflow(self, responses=(), execute=True, transport=None):
        return WanWorkflow(self.manifest_path, self.state_path, root=self.root,
            execute=execute, authorization_path=self.auth_path, transport=transport or MockTransport(responses),
            secret_loader=self.secret)

    def authorize(self, workflow, **changes):
        auth = {"approved": True, "provider": "dashscope", "model": PROFILE["model"],
            "authorization_id": "fixture-only-not-real-approval", "approved_by": "test-fixture",
            "approved_at": "2026-11-08T00:00:00Z", "manifest_sha256": workflow.manifest_sha256,
            "state_path": str(self.state_path), "allowed_job_ids": [self.job_id], "max_jobs": 1,
            "request_sha256_by_job": {self.job_id: workflow.request_identity(self.job_id)[1]},
            "max_cost_cny": "1.50", "automatic_recharge": False, "automatic_retry": False}
        auth.update(changes)
        self.auth_path.write_text(json.dumps(auth), encoding="utf-8")

    @staticmethod
    def receipt(status="PENDING", **fields):
        return 200, {"output": {"task_id": "fixture-task-123", "task_status": status, **fields}, "request_id": "fixture-request-1"}


class SyntheticWanTest(WanFixture):
    def test_uploaded_reference_is_the_exact_authorized_byte_snapshot(self):
        import base64
        authorized_bytes = self.reference.read_bytes()
        workflow = self.workflow(responses=[self.receipt()])
        self.authorize(workflow)
        def key_loader():
            self.reference.write_bytes(b"changed-after-reference-capture")
            return self.secret()
        workflow.secret_loader = key_loader
        workflow.submit(self.job_id)
        uploaded = json.loads(workflow.transport.calls[0][3].decode("utf-8"))["input"]["img_url"]
        self.assertEqual(base64.b64decode(uploaded.split(",", 1)[1]), authorized_bytes)

    def test_default_plan_is_offline_and_creates_no_state(self):
        workflow = self.workflow(execute=False)
        with patch("experiments.common.video.synthetic_wan.load_wan_secret", side_effect=AssertionError("must not read real key")):
            plan = workflow.plan(self.job_id)
            self.assertEqual(plan["network_requests"], 0)
            self.assertFalse(plan["credentials_read"])
            self.assertFalse(plan["submission_allowed"])
            with self.assertRaises(WorkflowError) as raised:
                workflow.submit(self.job_id)
            self.assertEqual(raised.exception.code, "execution_not_authorized")
        self.assertFalse(self.state_path.exists())
        self.assertEqual(workflow.transport.calls, [])
        self.assertEqual(self.secret_reads, [])

    def test_gates_reject_unsigned_wrong_budget_and_mutated_request_before_key(self):
        workflow = self.workflow()
        for changes in ({"approved": False}, {"max_cost_cny": "3.00"}, {"automatic_retry": True},
                        {"state_path": str(self.root / "different.json")}, {"max_jobs": True}):
            self.authorize(workflow, **changes)
            with self.assertRaises(WorkflowError):
                workflow.submit(self.job_id)
        self.authorize(workflow)
        workflow.edit(self.job_id, {"prompt_en": "The revised ordinary stream prompt."})
        with self.assertRaises(WorkflowError):
            workflow.submit(self.job_id)
        self.assertEqual(self.secret_reads, [])
        self.assertEqual(workflow.transport.calls, [])

    def test_unknown_post_is_reserved_and_never_posted_again(self):
        workflow = self.workflow([TimeoutError()])
        self.authorize(workflow)
        first = workflow.submit(self.job_id)
        second = workflow.submit(self.job_id)
        self.assertEqual(first["status"], "submission_unknown")
        self.assertEqual(second["reserved_cost_cents"], 150)
        self.assertEqual(len(workflow.transport.calls), 1)
        self.assertEqual(len(self.secret_reads), 1)
        self.assertEqual(first["error"]["code"], "submission_receipt_unknown")
        with self.assertRaises(WorkflowError):
            workflow.poll(self.job_id)
        with self.assertRaises(WorkflowError):
            workflow.edit(self.job_id, {"prompt_en": "Cannot rewrite a charged task."})

    def test_mock_submit_query_download_qc_chain_and_exact_wan_payload(self):
        workflow = self.workflow([self.receipt(), self.receipt("RUNNING"), self.receipt("SUCCEEDED", video_url="https://example.invalid/result.mp4")])
        self.authorize(workflow)
        row = workflow.submit(self.job_id)
        self.assertEqual(row["remote_task_id"], "fixture-task-123")
        method, url, headers, raw = workflow.transport.calls[0]
        payload = json.loads(raw)
        self.assertEqual((method, url), ("POST", API_BASE + "/api/v1/services/aigc/video-generation/video-synthesis"))
        self.assertEqual(headers["X-DashScope-Async"], "enable")
        self.assertEqual(payload["model"], "wan2.6-i2v-flash")
        self.assertEqual(payload["parameters"], {k: v for k, v in PROFILE.items() if k != "model"})
        self.assertTrue(payload["input"]["img_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("ratio", payload)
        self.assertNotIn("promptText", payload)
        self.assertEqual(workflow.poll(self.job_id)["status"], "running")
        # Calls within 15s return current state without another remote query.
        self.assertEqual(workflow.poll(self.job_id)["status"], "running")
        self.assertEqual(len(workflow.transport.calls), 2)
        with workflow.locked_state() as state:
            state["jobs"][self.job_id]["last_poll_attempt_at"] = "2020-01-01T00:00:00+00:00"
            workflow.persist(state)
        self.assertEqual(workflow.poll(self.job_id)["status"], "succeeded")
        received = workflow.download(self.job_id)
        self.assertTrue(workflow.result_path(self.job_id).is_file())
        workflow.download(self.job_id)
        self.assertEqual(len(workflow.transport.downloads), 1)
        self.assertIsNone(received["accepted_label"])
        with patch("experiments.common.video.synthetic_wan.shutil.which", return_value=None):
            report = workflow.qc(self.job_id)
        self.assertEqual(report["technical_status"], "pending")
        self.assertFalse(report["training_ready"])
        persisted = self.state_path.read_text(encoding="utf-8")
        self.assertNotIn("dummy-not-a-real-key", persisted)
        self.assertNotIn("data:image/jpeg;base64,", persisted)

    def test_failure_cancel_unknown_and_poll_error_retain_task_and_budget(self):
        for remote in ("FAILED", "CANCELED", "UNKNOWN"):
            with self.subTest(remote=remote):
                if self.state_path.exists():
                    self.state_path.unlink()
                workflow = self.workflow([self.receipt(), self.receipt(remote, code="FixtureFailure", message="dummy-not-a-real-key")])
                self.authorize(workflow)
                workflow.submit(self.job_id)
                row = workflow.poll(self.job_id)
                self.assertEqual(row["provider_status"], remote)
                self.assertEqual(row["reserved_cost_cents"], 150)
                self.assertNotIn("dummy-not-a-real-key", json.dumps(row))
                with self.assertRaises(WorkflowError):
                    workflow.download(self.job_id)
        self.state_path.unlink()
        workflow = self.workflow([self.receipt(), TimeoutError()])
        self.authorize(workflow)
        workflow.submit(self.job_id)
        row = workflow.poll(self.job_id)
        self.assertEqual(row["error"]["code"], "poll_failed")
        self.assertEqual(row["remote_task_id"], "fixture-task-123")
        self.assertEqual(len(workflow.transport.calls), 2)

    def test_failed_download_keeps_known_task_and_cleans_partial(self):
        transport = MockTransport([self.receipt(), self.receipt("SUCCEEDED", video_url="https://example.invalid/result.mp4")], OSError("fixture"))
        workflow = self.workflow(transport=transport)
        self.authorize(workflow)
        workflow.submit(self.job_id)
        workflow.poll(self.job_id)
        row = workflow.download(self.job_id)
        self.assertEqual(row["error"]["code"], "download_failed")
        self.assertIsNone(workflow.result_path(self.job_id))
        self.assertFalse(list((workflow.output_dir / self.job_id).glob("*.part")))

    def test_terminal_create_response_still_has_a_downloadable_receipt(self):
        workflow = self.workflow([self.receipt("SUCCEEDED", video_url="https://example.invalid/fast-result.mp4")])
        self.authorize(workflow)
        row = workflow.submit(self.job_id)
        self.assertEqual(row["status"], "succeeded")
        self.assertEqual(row["output_video_url"], "https://example.invalid/fast-result.mp4")
        workflow.download(self.job_id)
        self.assertEqual(len(workflow.transport.calls), 1)
        self.assertEqual(len(workflow.transport.downloads), 1)

    def test_reference_fingerprint_external_scope_and_non_draft_edit_are_blocked(self):
        workflow = self.workflow()
        self.authorize(workflow)
        self.reference.write_bytes(b"changed")
        with self.assertRaises(WorkflowError):
            workflow.submit(self.job_id)
        self.assertFalse(self.secret_reads)
        self.assertTrue(workflow.plan(self.job_id)["blockers"])

    def test_technical_qc_success_still_leaves_physics_and_label_unaccepted(self):
        probe = type("Result", (), {"stdout": json.dumps({"streams": [{"codec_type": "video", "width": 1280,
            "height": 720, "duration": "10.0"}]}).encode("utf-8")})()
        decode = type("Result", (), {"returncode": 0, "stderr": b""})()
        with patch("experiments.common.video.synthetic_wan.subprocess.run", side_effect=[probe, decode]):
            report = technical_qc(self.root / "dummy.mp4", ffprobe="fixture-probe", ffmpeg="fixture-decode")
        self.assertEqual(report["technical_status"], "passed")
        self.assertIsNone(report["accepted_label"])
        self.assertEqual(report["physical_quality_status"], "pending_human_or_assistant_review")


if __name__ == "__main__":
    unittest.main()
