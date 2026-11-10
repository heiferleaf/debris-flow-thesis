"""Local HTTP behavior only; no external provider, actual key or production state."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
from urllib import error, request

from experiments.tests.test_synthetic_wan import WanFixture
from experiments.tools.synthetic_video_workbench_server import Workbench, handler_for


class SyntheticWorkbenchTest(WanFixture):
    def setUp(self):
        super().setUp()
        for module in ("experiments.tools.synthetic_video_workbench_server", "experiments.common.video.synthetic_wan"):
            mocked = patch(module + ".credential_configured", return_value=True)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.workflow_instance = self.workflow(execute=False)
        assets = self.root / "assets"
        assets.mkdir()
        (assets / "index.html").write_text("<html>fixture</html>", encoding="utf-8")
        self.workbench = Workbench(self.workflow_instance, assets=assets)
        # A temporary loopback-only test socket, never the user-facing 8772 server.
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.workbench))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.base = "http://127.0.0.1:%d" % self.server.server_port

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def call(self, path, method="GET", value=None, headers=None):
        headers = {"Content-Type": "application/json", **(headers or {})}
        data = None if value is None else json.dumps(value).encode("utf-8")
        try:
            response = request.urlopen(request.Request(self.base + path, data=data, method=method, headers=headers))
        except error.HTTPError as response:
            return response.code, json.loads(response.read())
        with response:
            body = response.read()
            return response.status, json.loads(body) if response.headers.get_content_type() == "application/json" else body

    def test_snapshot_default_stage_truth_and_no_hidden_remote_progress(self):
        status, value = self.call("/api/workbench")
        self.assertEqual(status, 200)
        self.assertEqual(value["provider"]["model"], "wan2.6-i2v-flash")
        job = value["jobs"][0]
        self.assertIsNone(job["progress"])
        self.assertIsNone(job["remote_task_id"])
        self.assertIsNone(job["local_video_url"])
        self.assertFalse(job["submission_allowed"])
        self.assertEqual([s["id"] for s in job["stages"]], ["reference", "prompt", "authorization", "submit", "generation", "qc", "result"])
        self.assertEqual(self.workflow_instance.transport.calls, [])
        self.assertEqual(self.secret_reads, [])
        self.assertFalse(self.state_path.exists())

    def test_draft_save_and_offline_plan_then_submit_blocks_without_reading_key(self):
        path = "/api/jobs/" + self.job_id
        status, snapshot = self.call(path, "PATCH", {"prompt_en": "Ordinary water, stable channel, no debris flow."})
        self.assertEqual(status, 200)
        self.assertIn("Ordinary water", snapshot["jobs"][0]["prompt_en"])
        status, result = self.call(path + "/plan", "POST", {})
        self.assertEqual(status, 200)
        self.assertFalse(result["plan"]["credentials_read"])
        self.assertEqual(result["plan"]["network_requests"], 0)
        status, result = self.call(path + "/submit", "POST", {})
        self.assertEqual(status, 403)
        self.assertEqual(result["error"]["code"], "execution_not_authorized")
        self.assertEqual(self.secret_reads, [])
        self.assertEqual(self.workflow_instance.transport.calls, [])

    def test_legacy_submit_cannot_self_approve_or_expose_arbitrary_files(self):
        path = "/api/jobs/" + self.job_id
        status, _ = self.call(path + "/submit", "POST", {"approved": True, "execute": True})
        self.assertEqual(status, 400)
        status, _ = self.call(path, "PATCH", {"settings": {"duration": 15}})
        self.assertEqual(status, 400)
        for path in ("/../../authorization.json", "/.codex/secrets/dashscope.key.dpapi", "/media/reference/../../state.json", "/media/result/unregistered"):
            self.assertEqual(self.call(path)[0], 404)
        self.assertEqual(self.call("/api/workbench", headers={"Host": "malicious.example"})[0], 403)
        self.assertEqual(self.call("/api/jobs/" + self.job_id + "/plan", "POST", {}, {"Origin": "https://malicious.example"})[0], 403)

    def prepare(self):
        status, response = self.call("/api/jobs/" + self.job_id + "/prepare-confirmation", "POST", {},
                                     {"Origin": self.base})
        self.assertEqual(status, 200, response)
        return response["confirmation"]

    @staticmethod
    def consent(confirmation):
        return {"confirmation_token": confirmation["token"],
                "request_sha256": confirmation["request_sha256"],
                "accepted_upload": True, "max_cost_cny": 1.5}

    def test_opening_and_cancelling_confirmation_is_offline(self):
        c = self.prepare()
        self.assertEqual(c["estimated_cny"], 1.5)
        self.assertEqual(c["reference_url"], "/media/reference/camera-a")
        self.assertFalse(self.state_path.exists())
        self.assertFalse((self.state_path.parent / "workbench-consent").exists())
        self.assertEqual(self.secret_reads, [])
        self.assertEqual(self.workflow_instance.transport.calls, [])
        # No follow-up request is the browser's cancel action.
        self.assertTrue(self.call("/api/workbench")[1]["jobs"][0]["user_confirmation_available"])
        path = "/api/jobs/" + self.job_id + "/confirm-submit"
        invalid = self.consent(c)
        invalid["accepted_upload"] = False
        self.assertEqual(self.call(path, "POST", invalid, {"Origin": self.base})[0], 403)
        invalid = self.consent(c)
        invalid["max_cost_cny"] = 9
        self.assertEqual(self.call(path, "POST", invalid, {"Origin": self.base})[0], 403)
        self.assertEqual(self.secret_reads, [])

    def test_only_final_bound_confirmation_submits_once_and_survives_restart(self):
        c = self.prepare()
        self.workflow_instance.transport.responses.append(self.receipt())
        path = "/api/jobs/" + self.job_id + "/confirm-submit"
        self.assertEqual(self.call(path, "POST", self.consent(c), {"Origin": self.base})[0], 200)
        self.assertEqual(len(self.workflow_instance.transport.calls), 1)
        self.assertEqual(len(self.secret_reads), 1)
        status, response = self.call(path, "POST", self.consent(c), {"Origin": self.base})
        self.assertEqual(status, 409)
        self.assertEqual(len(self.workflow_instance.transport.calls), 1)
        restored = Workbench(self.workflow_instance, assets=self.workbench.assets)
        snapshot = restored.snapshot()
        self.assertEqual(snapshot["jobs"][0]["remote_task_id"], "fixture-task-123")
        self.assertEqual(snapshot["jobs"][0]["cost_reserved_cny"], 1.5)
        self.assertFalse(snapshot["jobs"][0]["user_confirmation_available"])
        self.assertTrue(restored.workflow_for(self.job_id).execute)

    def test_stale_prompt_expired_token_and_foreign_origin_never_read_key(self):
        prefix = "/api/jobs/" + self.job_id
        self.assertEqual(self.call(prefix + "/prepare-confirmation", "POST", {})[0], 403)
        self.assertEqual(self.call(prefix + "/prepare-confirmation", "POST", {},
                                   {"Origin": "https://other.example"})[0], 403)
        c = self.prepare()
        self.workbench.workflow.edit(self.job_id, {"prompt_en": "Changed prompt that must be reviewed again."})
        status, response = self.call(prefix + "/confirm-submit", "POST", self.consent(c), {"Origin": self.base})
        self.assertEqual(status, 409)
        self.assertEqual(response["error"]["code"], "request_changed")
        c = self.prepare()
        self.workbench.confirmations[self.job_id]["deadline"] = 0
        self.assertEqual(self.call(prefix + "/confirm-submit", "POST", self.consent(c), {"Origin": self.base})[0], 409)
        self.assertEqual(self.secret_reads, [])
        self.assertEqual(self.workflow_instance.transport.calls, [])

    def test_edit_back_to_original_prompt_does_not_resurrect_confirmation(self):
        c = self.prepare()
        prefix = "/api/jobs/" + self.job_id
        original_prompt = self.workflow_instance.draft(self.job_id)["prompt_en"]
        self.assertEqual(self.call(prefix, "PATCH", {"prompt_en": "A different draft."})[0], 200)
        self.assertEqual(self.call(prefix, "PATCH", {"prompt_en": original_prompt})[0], 200)
        self.assertEqual(self.call(prefix + "/confirm-submit", "POST", self.consent(c),
                                   {"Origin": self.base})[0], 409)
        self.assertEqual(self.secret_reads, [])
        self.assertEqual(self.workflow_instance.transport.calls, [])

    def test_separate_state_does_not_collide_with_old_user_receipt(self):
        c = self.prepare()
        first = self.workbench.consent_path(self.job_id, c["request_sha256"])
        other = self.workflow(execute=False)
        other.state_path = self.root / "separate-state.json"
        second = Workbench(other, assets=self.workbench.assets).consent_path(self.job_id, c["request_sha256"])
        self.assertNotEqual(first, second)

    def test_only_registered_reference_and_static_file_are_served(self):
        self.assertEqual(self.call("/")[0], 200)
        status, data = self.call("/media/reference/camera-a")
        self.assertEqual(status, 200)
        self.assertEqual(data, self.reference.read_bytes())
        self.reference.write_bytes(b"tampered")
        self.assertEqual(self.call("/media/reference/camera-a")[0], 404)

    def test_user_rejection_blocks_both_confirmation_steps_and_direct_submit(self):
        confirmation = self.prepare()
        acceptance = self.workflow_instance.manifest_path.parent / "reference-acceptance.json"
        acceptance.parent.mkdir(parents=True, exist_ok=True)
        reference = self.workflow_instance.job(self.job_id)["selected_reference"]["path"]
        acceptance.write_text(json.dumps({"items": [{"path": reference, "status": "rejected_by_user",
            "reason": "background geometry/texture redrawn unnaturally", "training_ready": False}]}), encoding="utf-8")
        snapshot = self.call("/api/workbench")[1]
        self.assertFalse(snapshot["jobs"][0]["user_confirmation_available"])
        self.assertFalse(snapshot["jobs"][0]["submission_allowed"])
        prefix = "/api/jobs/" + self.job_id
        for suffix, payload in (("prepare-confirmation", {}), ("confirm-submit", self.consent(confirmation)), ("submit", {})):
            status, value = self.call(prefix + "/" + suffix, "POST", payload, {"Origin": self.base})
            self.assertEqual(status, 409)
            self.assertEqual(value["error"]["code"], "reference_rejected_by_user")
        self.assertEqual(self.secret_reads, [])
        self.assertEqual(self.workflow_instance.transport.calls, [])
        self.assertFalse(self.state_path.exists())


if __name__ == "__main__":
    unittest.main()
