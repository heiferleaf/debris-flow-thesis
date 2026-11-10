# -*- coding: utf-8 -*-
"""新视频推理的关键门禁。运行：.venv/Scripts/python.exe experiments/tests/test_infer_video_protocol.py"""
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import infer_video as iv


def make_run(root, label="clip-labels-v3-128.npz",
             digest="fa2c53c1a0f93d62869b1aeebbc80f4b"):
    root.mkdir(parents=True, exist_ok=True)
    run = root / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({
        "script": "train_ft.py", "clip_labels": "/old/path/" + label,
        "md5": {"clip_labels": digest},
    }), encoding="utf-8")
    return run


class TestInferProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sampling_requires_matching_training_evidence(self):
        run = make_run(self.root)
        self.assertEqual(iv.sampling_from_run(run, "ft"), "v3")
        with self.assertRaisesRegex(SystemExit, "拒绝出分"):
            iv.sampling_from_run(run, "mil")

    def test_frozen_model_loaders_require_five_unless_fold_requested(self):
        class FakeHead:
            def load_state_dict(self, _state):
                pass

            def eval(self):
                pass

        for kind in ("mil", "strong"):
            run = self.root / kind
            run.mkdir()
            pattern = iv.HEADS[kind][1]
            for fold in range(1, 5):
                (run / pattern.replace("*", str(fold))).touch()
            with self.assertRaisesRegex(SystemExit, "缺第 \\[5\\] 折.*拒绝出分"):
                iv.load_fold_heads(run, kind)
            with patch.dict(iv.HEADS, {kind: (FakeHead, pattern, kind == "strong")}), \
                 patch.object(iv.torch, "load", return_value={}):
                heads = iv.load_fold_heads(run, kind, folds=[2])
            self.assertEqual(list(heads), [2])
            with self.assertRaisesRegex(SystemExit, "缺第 \\[5\\] 折.*拒绝出分"):
                iv.load_fold_heads(run, kind, folds=[5])
            (run / pattern.replace("*", "5")).touch()
            with patch.dict(iv.HEADS, {kind: (FakeHead, pattern, kind == "strong")}), \
                 patch.object(iv.torch, "load", return_value={}):
                complete = iv.load_fold_heads(run, kind)
            self.assertEqual(list(complete), [1, 2, 3, 4, 5])
            (run / pattern.replace("*", "01")).touch()
            with patch.dict(iv.HEADS, {kind: (FakeHead, pattern, kind == "strong")}), \
                 patch.object(iv.torch, "load", return_value={}):
                self.assertEqual(list(iv.load_fold_heads(run, kind, folds=[2])), [2])
            with self.assertRaisesRegex(SystemExit, "重复.*拒绝出分"):
                iv.load_fold_heads(run, kind)

    def test_ft_loader_requires_five_unless_fold_requested(self):
        run = self.root / "ft"
        run.mkdir()
        for fold in range(1, 5):
            (run / f"ft_strong_f{fold}.pt").touch()
        with self.assertRaisesRegex(SystemExit, "缺第 \\[5\\] 折.*拒绝出分"):
            iv.load_ft_nets(run)
        fake_module = types.ModuleType("clip_scores_ft")
        called = []

        def fake_build(_run, fold, _pattern):
            called.append(fold)
            return object()

        fake_module.build = fake_build
        with patch.dict(sys.modules, {"clip_scores_ft": fake_module}):
            nets = iv.load_ft_nets(run, folds=[2])
        self.assertEqual(list(nets), [2])
        self.assertEqual(called, [2])
        with self.assertRaisesRegex(SystemExit, "缺第 \\[5\\] 折.*拒绝出分"):
            iv.load_ft_nets(run, folds=[5])
        (run / "ft_strong_f5.pt").touch()
        called.clear()
        with patch.dict(sys.modules, {"clip_scores_ft": fake_module}):
            complete = iv.load_ft_nets(run)
        self.assertEqual(list(complete), [1, 2, 3, 4, 5])
        self.assertEqual(called, [1, 2, 3, 4, 5])

    def test_conflicting_sampling_refuses(self):
        run = make_run(self.root, label="clip-labels-v2-128.npz")
        with self.assertRaisesRegex(SystemExit, "拒绝出分"):
            iv.sampling_from_run(run, "ft")

    def test_full_frame_gray_scan_and_eof_gate(self):
        video = self.root / "sample.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (256, 256))
        self.assertTrue(writer.isOpened())
        gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        normal = np.repeat(gradient[:, :, None], 3, axis=2)
        gray = np.full((256, 256, 3), 128, dtype=np.uint8)
        for i in range(32):
            writer.write(gray if i == 5 else normal)
        writer.release()
        self.assertEqual(iv.scan_gray_video(video, 32), [5])
        for n in (31, 33):
            with self.assertRaisesRegex(SystemExit, "拒绝出分"):
                iv.scan_gray_video(video, n)

    def test_even_256_input_transcodes_by_default_and_prepared_skips(self):
        video = self.root / "already-256.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (256, 256))
        self.assertTrue(writer.isOpened())
        frame = np.repeat(np.tile(np.arange(256, dtype=np.uint8), (256, 1))[:, :, None],
                          3, axis=2)
        for _ in range(32):
            writer.write(frame)
        writer.release()
        with patch.object(iv, "REPO", self.root):
            middleware, record_path = iv.prepare_input_video(video)
            self.assertNotEqual(middleware, video)
            self.assertTrue(record_path.is_file())
            self.assertEqual(iv.first_frame_size(middleware), (256, 256))
            same, no_record = iv.prepare_input_video(video, prepared=True)
            self.assertEqual(same, video)
            self.assertIsNone(no_record)

    def test_both_sampling_modes_convert_raw_video_and_keep_provenance(self):
        video = self.root / "raw.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        self.assertTrue(writer.isOpened())
        gradient = np.tile(np.arange(320, dtype=np.uint8), (240, 1))
        frame = np.repeat(gradient[:, :, None], 3, axis=2)
        for _ in range(160):
            writer.write(frame)
        writer.release()
        with self.assertRaisesRegex(SystemExit, "256×256.*拒绝出分"):
            iv.prepare_input_video(video, prepared=True)
        modes = (("v2", "clip-labels-v2-128.npz", "07ad71057234c990fb9498a5c5e59a0c"),
                 ("v3", "clip-labels-v3-128.npz", "fa2c53c1a0f93d62869b1aeebbc80f4b"))
        for name, label, digest in modes:
            run = make_run(self.root / name, label, digest)
            with ExitStack() as stack:
                stack.enter_context(patch.object(iv, "REPO", self.root))
                stack.enter_context(patch.object(iv, "load_ft_nets", return_value={1: object()}))
                stack.enter_context(patch.object(
                    iv, "score_clips_ft", side_effect=lambda _d, _n, n, progress=None:
                    {1: np.ones(n, dtype=np.float32)}))
                out = iv.infer(video, run, self.root / (name + "-out"), folds=[1], kind="ft")
            self.assertEqual(out["sampling"], name)
            self.assertEqual(out["video"], str(video))
            self.assertEqual(iv.first_frame_size(Path(out["middleware"])), (256, 256))
            self.assertEqual(out["n_frames"], 160)
            self.assertEqual(out["n_clips"], 10 if name == "v2" else 2)
            if name == "v3":
                self.assertEqual(out["frame_stride"], 5)
                self.assertEqual(out["slots"], [0, 76])
            record = json.loads(Path(out["transcode_record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "validated")
            self.assertEqual(record["source"], str(video.resolve()))
            self.assertTrue(Path(record["ffmpeg_log"]).is_file())
            self.assertTrue(Path(record["middleware"]).is_file())

    def test_failed_conversion_keeps_log_and_refuses(self):
        bad = self.root / "not-a-video.mp4"
        bad.write_text("invalid video", encoding="utf-8")
        with patch.object(iv, "REPO", self.root):
            with self.assertRaisesRegex(SystemExit, "转码失败.*拒绝出分"):
                iv.prepare_input_video(bad)
        records = list((self.root / ".codex/docs/.temp/infer-video/transcodes").glob("*/source.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertTrue(Path(record["ffmpeg_log"]).is_file())

    def test_converted_video_pts_mismatch_refuses_with_record(self):
        raw = self.root / "raw-mismatch.avi"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        self.assertTrue(writer.isOpened())
        frame = np.repeat(np.tile(np.arange(320, dtype=np.uint8), (240, 1))[:, :, None],
                          3, axis=2)
        for _ in range(160):
            writer.write(frame)
        writer.release()
        run = make_run(self.root / "mismatch")
        with patch.object(iv, "REPO", self.root), \
             patch.object(iv, "probe_pts", return_value=[i / 10 for i in range(150)]):
            with self.assertRaisesRegex(SystemExit, "时间戳.*不符.*拒绝出分"):
                iv.infer(raw, run, kind="ft")
        records = list((self.root / ".codex/docs/.temp/infer-video/transcodes").glob("*/source.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "rejected")
        self.assertTrue(Path(record["ffmpeg_log"]).is_file())

    def test_v3_inference_uses_stride_and_blocks(self):
        run = make_run(self.root)
        video = self.root / "dummy.mp4"
        video.touch()
        seen = {}

        def fake_extract(_video, frames, out_dir, **_kw):
            seen["frames"] = frames
            out_dir.mkdir(parents=True)
            return len(frames), 160, None, []

        with ExitStack() as stack:
            stack.enter_context(patch.object(iv, "prepare_input_video", return_value=(video, None)))
            stack.enter_context(patch.object(iv, "probe_pts", return_value=[i / 10 for i in range(160)]))
            stack.enter_context(patch.object(iv, "probe_video", return_value=(160, 10.0)))
            stack.enter_context(patch.object(iv, "scan_gray_video", return_value=[]))
            stack.enter_context(patch.object(iv, "training_zones", return_value=([], 0, 0)))
            stack.enter_context(patch.object(iv, "extract_stride", side_effect=fake_extract))
            stack.enter_context(patch.object(iv, "load_ft_nets", return_value={1: object()}))
            stack.enter_context(patch.object(
                iv, "score_clips_ft", side_effect=lambda _d, _n, n, progress=None:
                {1: np.ones(n, dtype=np.float32)}))
            out = iv.infer(video, run, self.root / "out", folds=[1], kind="ft")
        self.assertEqual(out["sampling"], "v3")
        self.assertEqual(out["frame_stride"], 5)
        self.assertEqual(out["slots"], [0, 76])
        self.assertEqual(seen["frames"][0], (0, 0, 0))
        self.assertEqual(seen["frames"][15], (0, 75, 15))
        self.assertEqual(seen["frames"][16], (1, 76, 0))
        self.assertEqual((self.root / "out/frames/slots.txt").read_text(), "0;76")

    def test_missing_pts_refuses_before_scoring(self):
        run = make_run(self.root)
        video = self.root / "dummy.mp4"
        video.touch()
        with patch.object(iv, "prepare_input_video", return_value=(video, None)), \
             patch.object(iv, "probe_pts", return_value=None):
            with self.assertRaisesRegex(SystemExit, "拒绝出分"):
                iv.infer(video, run, kind="ft")


if __name__ == "__main__":
    unittest.main()
