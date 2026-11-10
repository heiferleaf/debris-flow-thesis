"""Offline checks for paid-submit boundaries and honest media acceptance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.common.video.minimax_preview_workflow import MinimaxPreview, atomic_json
from experiments.common.video import minimax_official as official


class PreviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        Image.new("RGB", (320, 180), "grey").save(self.root / "data/dry.png")
        self.workflow = MinimaxPreview(self.root, secret_loader=lambda: self.fail("Unexpected credential read"))
        atomic_json(self.workflow.inventory, {"items": [{"scene_id": "camera-test", "camera": "test", "index": 1,
                    "source_path": "data/dry.png", "output_path": "data/dry.png"}]})
        self.draft = {"scene_id": "camera-test", "polygon": [[.1,.2],[.8,.2],[.8,.7],[.1,.7]],
                      "flow": [[.7,.3],[.2,.6]], "duration": 4, "workspace": "test-space",
                      "prompt": "test ordinary water", "reference_ok": True}

    def test_read_save_prepare_do_not_read_key_or_call_network(self):
        with patch.object(self.workflow.http, "get", side_effect=AssertionError("network")), patch.object(self.workflow.http, "post", side_effect=AssertionError("network")):
            self.workflow.snapshot()
            self.workflow.save_draft(self.draft)
            prepared = self.workflow.prepare(self.draft)
        self.assertEqual(prepared["intent"]["estimated_cny"], .64)
        self.assertEqual(prepared["intent"]["resolution"], "480P")
        self.assertEqual(prepared["intent"]["input_mode"], "image_url+image_url")
        self.assertFalse(self.workflow.history())

    def test_explicit_consent_and_single_use_token(self):
        prepared = self.workflow.prepare(self.draft)
        with self.assertRaises(ValueError):
            self.workflow.submit({"token": prepared["token"], "approved": False})
        prepared = self.workflow.prepare(self.draft)
        with patch("experiments.common.video.minimax_preview_workflow.threading.Thread") as thread:
            created = self.workflow.submit({"token": prepared["token"], "approved": True})
            self.assertEqual(thread.call_count, 1)
            with self.assertRaises(ValueError):
                self.workflow.submit({"token": prepared["token"], "approved": True})
        self.assertTrue(self.workflow.run_path(created["id"]).is_file())

    def test_changed_reference_aborts_before_paid_request(self):
        prepared = self.workflow.prepare(self.draft)
        Image.new("RGB", (320,180), "blue").save(self.root / "data/dry.png")
        with self.assertRaisesRegex(ValueError, "场景图已经变化"):
            self.workflow.submit({"token": prepared["token"], "approved": True})
        self.assertFalse(self.workflow.history())

    def test_missing_geometry_and_unknown_scene_block_prepare(self):
        for change in ({"flow": []}, {"polygon": []}, {"scene_id": "not-registered"}, {"reference_ok": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.workflow.prepare({**self.draft, **change})

    def test_provider_success_does_not_imply_quality_success(self):
        identity = "minimax-water-v2-test"
        atomic_json(self.workflow.jobs / identity / "state.json", {"status": "succeeded", "scene_id": "camera-test"})
        (self.workflow.jobs / identity / "candidate.mp4").write_bytes(b"fixture")
        row = self.workflow.history()[0]
        self.assertEqual(row["review"], {})
        with self.assertRaises(ValueError):
            self.workflow.review({"job_id": identity, "outcome": "accepted", "checks": {"motion": False}})
        self.workflow.review({"job_id": identity, "outcome": "rejected", "note": "static video"})
        self.assertEqual(self.workflow.history()[0]["review"]["outcome"], "rejected")
        self.assertEqual(json.loads(self.workflow.run_path(identity).read_text())["status"], "succeeded")
        self.assertFalse(self.workflow.snapshot()["training_ready"])

    def test_active_remote_job_blocks_new_paid_attempt(self):
        atomic_json(self.workflow.jobs / "minimax-water-v2-test" / "state.json", {"status": "submission_unknown"})
        with self.assertRaisesRegex(ValueError, "已有未结束"):
            self.workflow.prepare(self.draft)

    def test_paid_payload_uses_two_references_and_selected_duration(self):
        prepared = self.workflow.prepare(self.draft)
        with patch("experiments.common.video.minimax_preview_workflow.threading.Thread"):
            identity = self.workflow.submit({"token": prepared["token"], "approved": True})["id"]
        self.workflow.secret_loader = lambda: "offline-test-secret"
        with patch.object(self.workflow, "poll"), patch.object(self.workflow.http, "post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {"task_id": "offline-task"}
            self.workflow.perform(identity)
        body = post.call_args.kwargs["json"]
        self.assertEqual(post.call_args.args[0], "https://api.minimax.cn/v2/video_generation")
        self.assertEqual(body["model"], "MiniMax-H3-Lite")
        self.assertEqual(body["resolution"], "480P")
        images=body["content"][1:]
        self.assertEqual(len(images),2)
        self.assertTrue(all(x["type"] == "image_url" and x["role"] == "reference_image" for x in images))
        self.assertTrue(all(x["image_url"]["url"].startswith("data:image/png;base64,") for x in images))
        self.assertEqual(body["duration"], 4)
        self.assertEqual(body["ratio"], "16:9")
        self.assertEqual(post.call_count, 1)

    def test_expensive_profile_is_blocked_before_network(self):
        value=self.workflow.prepare(self.draft)["intent"]
        for change in ({"duration":10}, {"resolution":"768P"}, {"model":"MiniMax-H3"}, {"estimated_cny":5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                official.request_body({**value,**change},self.root/'data/dry.png',self.root/'data/dry.png')

    def test_official_key_never_falls_back_to_bailian_key(self):
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'nonsecret-bailian-fixture'},clear=True):
            self.assertFalse(official.configured(self.root))
            with self.assertRaisesRegex(ValueError,'官方'):
                official.load_secret(self.root)

    def test_static_video_raises_low_motion_warning(self):
        import cv2
        import numpy as np
        path=self.root/'static.mp4'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(320,180))
        try:
            rng=np.random.RandomState(1)
            frame=rng.randint(0,256,(180,320,3),dtype=np.uint8)
            for _ in range(32):writer.write(frame)
        finally:
            writer.release()
        report=official.motion_check(path,self.draft['polygon'])
        self.assertEqual(report['flag'],'low_motion')
        self.assertEqual(report['frames_sampled'],9)

    def test_three_variant_batch_is_bounded_and_cannot_be_created_twice(self):
        from experiments.tools.compare_minimax_water_prompts import create_batch
        atomic_json(self.workflow.jobs/'minimax-official-f6349ffaed0d0fd9/state.json',
                    {'status':'succeeded','prompt':'baseline ordinary water'})
        # This test uses a fixture scene under the real preview's scene ID.
        inventory=json.loads(self.workflow.inventory.read_text())
        inventory['items'][0]['scene_id']='camera-c9c07aee7f28'
        atomic_json(self.workflow.inventory,inventory)
        self.workflow.save_draft({**self.draft,'scene_id':'camera-c9c07aee7f28'})
        with patch.object(self.workflow.http,'post',side_effect=AssertionError('network')):
            batch=create_batch(self.workflow,execute=True)
        self.assertEqual(len(batch['variants']),3)
        self.assertEqual(batch['total_estimated_cny'],1.92)
        self.assertEqual(len({r['intent']['reference_sha256'] for r in batch['variants']}),1)
        self.assertEqual(len({r['intent']['locator_sha256'] for r in batch['variants']}),1)
        with self.assertRaises(ValueError):create_batch(self.workflow,execute=True)

    def test_wet_first_frame_is_one_strict_frame_not_two_soft_references(self):
        wet=self.root/'data/process/synthetic-video/wet-first-frame-20261109-v1/camera-test/wet-first-frame.png'
        wet.parent.mkdir(parents=True)
        Image.new('RGB',(320,180),'blue').save(wet)
        prepared=self.workflow.prepare({**self.draft,'input_mode':'first_frame'})
        self.assertEqual(prepared['intent']['reference_kind'],'wet')
        body=official.request_body(prepared['intent'],wet,self.root/'must-not-be-read.png')
        self.assertEqual(len(body['content']),2)
        self.assertEqual(body['content'][1]['role'],'first_frame')
        self.assertNotIn('ratio',body)

    def test_standard_quote_binds_new_user_image_and_two_yuan_profile(self):
        wet=self.root/'data/process/synthetic-video/user-water-region-20261109-v2/camera-test/wet-first-frame.png'
        wet.parent.mkdir(parents=True)
        Image.new('RGB',(320,180),'blue').save(wet)
        prepared=self.workflow.prepare({**self.draft,'input_mode':'first_frame','profile':'standard-water-v2'})
        intent=prepared['intent']
        self.assertEqual(intent['reference_kind'],'wet_user')
        self.assertEqual(intent['estimated_cny'],2.0)
        body=official.request_body(intent,wet)
        self.assertEqual((body['model'],body['resolution'],body['duration']),('MiniMax-H3','768P',4))
        self.assertEqual(len(body['content']),2)
        self.assertEqual(body['content'][1]['role'],'first_frame')
        with self.assertRaises(ValueError):official.request_body({**intent,'estimated_cny':.64},wet)
        atomic_json(self.workflow.jobs/'minimax-official-standardfixture/state.json',
                    {'status':'failed','batch_id':'user-water-region-20261109-v2'})
        with self.assertRaisesRegex(ValueError,'重复付费'):
            self.workflow.submit({'token':prepared['token'],'approved':True})

    def test_wet_preview_phase_stops_after_two_created_jobs(self):
        wet=self.root/'data/process/synthetic-video/wet-first-frame-20261109-v1/camera-test/wet-first-frame.png'
        wet.parent.mkdir(parents=True)
        Image.new('RGB',(320,180),'blue').save(wet)
        for i in range(2):
            atomic_json(self.workflow.jobs/('minimax-official-fixture'+str(i))/'state.json',
                        {'status':'failed','batch_id':'wet-first-frame-20261109-v1'})
        prepared=self.workflow.prepare({**self.draft,'input_mode':'first_frame'})
        with self.assertRaisesRegex(ValueError,'两条'):
            self.workflow.submit({'token':prepared['token'],'approved':True})


if __name__ == "__main__":
    unittest.main()
