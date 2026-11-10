"""Local-only dry scene encoding, refresh and paid-route lock checks."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib import error, request

from PIL import Image
from experiments.tools.build_dry_scene_previews import build, find_ffmpeg, sha256, write_json
from experiments.tools.synthetic_video_workbench_server import DrySceneGallery, handler_for


class DryScenePreviewTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "scenes/camera-test/dry-reference.png"
        self.image.parent.mkdir(parents=True)
        self.source = self.root / "source.png"
        Image.new("RGB", (64, 36), (120, 140, 110)).save(self.source)
        self.inventory = self.root / "metadata/source-inventory.json"
        write_json(self.inventory, {"items": [{"index": 1, "scene_id": "camera-test", "camera": "测试机位",
            "source_path": "source.png", "output_path": "scenes/camera-test/dry-reference.png",
            "metadata_path": "metadata/camera-test.json"}]})

    def test_real_10s_encode_single_source_and_matching_rerun_skips(self):
        self.image.write_bytes(self.source.read_bytes())
        catalog = build(self.inventory, self.root / "scenes", find_ffmpeg(), root=self.root)
        row = catalog["items"][0]
        self.assertEqual(row["video_status"], "complete")
        self.assertEqual(row["record"]["technical"]["duration_seconds"], 10)
        self.assertEqual(row["record"]["technical"]["decoded_frames"], 250)
        self.assertEqual(row["record"]["parameters"]["unique_source_images"], 1)
        self.assertEqual(row["record"]["parameters"]["profile"], "high")
        self.assertLess(row["record"]["technical"]["size_bytes"], 3 * 1024 * 1024)
        self.assertFalse(row["record"]["parameters"]["physical_motion"])
        with patch("experiments.tools.build_dry_scene_previews.encode_preview", side_effect=AssertionError("must skip")):
            repeated = build(self.inventory, self.root / "scenes", find_ffmpeg(), root=self.root)
        self.assertEqual(repeated["items"][0]["operation"], "skipped_matching_output")

    def test_snapshot_refreshes_real_image_revision_and_stale_video_states(self):
        gallery = DrySceneGallery(self.inventory, root=self.root)
        self.assertEqual(gallery.snapshot()["summary"]["pending"], 1)
        self.image.write_bytes(self.source.read_bytes())
        write_json(self.root / "metadata/camera-test.json", {"fluid_suspected_in_channel": True})
        (self.image.parent / "static-10s.mp4").write_bytes(b"fixture-video")
        write_json(self.image.parent / "static-10s.json", {"source_sha256": "stale-image"})
        scene = gallery.snapshot()["scenes"][0]
        self.assertEqual(scene["image_status"], "needs_revision")
        self.assertIsNone(scene["video_url"])
        self.assertEqual(gallery.media("dry", "camera-test"), self.image)
        self.assertIsNone(gallery.media("dry", "not-registered"))

    def test_old_paid_submission_and_edit_routes_are_locked_without_provider(self):
        gallery = DrySceneGallery(self.inventory, root=self.root)
        with patch("experiments.tools.synthetic_video_workbench_server.WanWorkflow", side_effect=AssertionError("provider forbidden")), \
             patch("experiments.tools.synthetic_video_workbench_server.credential_configured", side_effect=AssertionError("credentials forbidden")):
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(gallery))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = "http://127.0.0.1:%d" % server.server_port
                for method, suffix in (("POST", "/confirm-submit"), ("POST", "/submit"), ("PATCH", "")):
                    with self.assertRaises(error.HTTPError) as raised:
                        request.urlopen(request.Request(base + "/api/jobs/old-paid-draft" + suffix,
                            data=b"{}", method=method, headers={"Content-Type": "application/json"}))
                    self.assertEqual(raised.exception.code, 403)
                    self.assertEqual(json.loads(raised.exception.read())["error"]["code"], "paid_generation_locked")
                with request.urlopen(base + "/api/workbench") as response:
                    self.assertFalse(json.load(response)["paid_submission_enabled"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
