"""Dense metrics and real sequential/filter decoder agreement."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.dense_evaluation import (checked_frames, dense_batches, video_metrics,
    compatible_pixel_reference, load_selected_checkpoint, observe_decoder_line)
from experiments.common.video.supervised_plan import digest


class DenseTests(unittest.TestCase):
    def test_interleaved_raw_muxer_warning_does_not_hide_real_decoder_errors(self):
        state = dict(decoded=0, max_pts_error=0., raw_muxer_dts_warnings=0)
        errors, diagnostics = [], []
        pts = np.array([0., 1.])
        observe_decoder_line('[Parsed_showinfo_0] n: 0 pts: 0 pts_time: 0 s:256x256', pts, state, errors, diagnostics)
        warning = 'Application provided invalid, non monotonically increasing dts to muxer in stream 0: 4394 >= 4394'
        observe_decoder_line('[Parsed_showinfo_0] color_range:pc'+warning, pts, state, errors, diagnostics)
        self.assertEqual((state['decoded'], state['raw_muxer_dts_warnings']), (1, 1))
        self.assertEqual((errors, diagnostics), ([], []))
        observe_decoder_line('[Parsed_showinfo_0] n: 1 pts: 1 pts_time: 1 s:256x256 '+warning+' invalid NAL unit',
                             pts, state, errors, diagnostics)
        self.assertEqual(state['decoded'], 2)
        self.assertEqual(state['raw_muxer_dts_warnings'], 2)
        self.assertIn('invalid NAL unit', diagnostics[0])
        observe_decoder_line('[Parsed_showinfo_0] n: 3 pts: 3 pts_time: 3 s:256x256', pts, state, errors, diagnostics)
        self.assertTrue(errors)

    def test_no_legal_window_returns_unknown_without_decoding(self):
        info = {}
        with patch('experiments.common.video.dense_evaluation.checked_frames', side_effect=AssertionError('must not decode')):
            self.assertEqual(list(dense_batches('unused', [], [], 'unused', info, filter_path='unused')), [])
        self.assertEqual(info['windows'], 0)
        self.assertNotIn('decoded', info)
        self.assertEqual(video_metrics([dict(label=1, score=None)])['unknown'], 1)

    def test_only_completed_selected_checkpoint_accepted(self):
        parent = ROOT / '.codex/docs/.temp/dense-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            folder = Path(folder)
            path = folder / 'best.pt'
            checkpoint = dict(profile=dict(mode='rgb_zero', plan_sha256='plan', smoke=False,
                                          requested_epochs=20, requested_optimizer_updates=420), completed_epochs=3)
            torch.save(checkpoint, path)
            result = dict(status='supervised_complete_dense_evaluation_pending', completed_epochs=20,
                          optimizer_steps=420, best_epoch=3,
                          selected_checkpoint=dict(filename='best.pt', sha256=digest(path), completed_epochs=3))
            result_file = folder / 'training-result.json'
            result_file.write_text(json.dumps(result), encoding='utf-8')
            actual, sha = load_selected_checkpoint(path, 'plan', 'rgb_zero')
            self.assertEqual(actual['completed_epochs'], 3)
            self.assertEqual(sha, digest(path))
            last = folder / 'last.pt'
            torch.save(checkpoint, last)
            with self.assertRaises(ValueError):
                load_selected_checkpoint(last, 'plan', 'rgb_zero')
            with self.assertRaises(ValueError):
                load_selected_checkpoint(path, 'other-plan', 'rgb_zero')
            result['status'] = 'smoke_passed'
            result_file.write_text(json.dumps(result), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_selected_checkpoint(path, 'plan', 'rgb_zero')
            result['status'] = 'supervised_complete_dense_evaluation_pending'
            result_file.write_text(json.dumps(result), encoding='utf-8')
            def concurrent_update(*args, **kwargs):
                with path.open('ab') as f:
                    f.write(b'changed')
                return checkpoint
            with patch('experiments.common.video.dense_evaluation.torch.load', side_effect=concurrent_update):
                with self.assertRaisesRegex(ValueError, 'changed while loading'):
                    load_selected_checkpoint(path, 'plan', 'rgb_zero')

    def test_pixel_reference_inherits_only_identical_input_resources(self):
        parent = ROOT / '.codex/docs/.temp/dense-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            original = dict(resource='resource-v4', sources=[dict(path='pixels', sha256='abc')])
            source = Path(folder) / 'source-plan.json'
            source.write_text(json.dumps(original), encoding='utf-8')
            reference = dict(status='dense_pixels_verified', plan_sha256=digest(source))
            compatible_pixel_reference(original, 'new-assignment-plan', reference, source)
            changed = dict(resource='resource-v4', sources=[dict(path='pixels', sha256='different')])
            with self.assertRaises(ValueError):
                compatible_pixel_reference(changed, 'new-assignment-plan', reference, source)

    def test_video_metrics_threshold_ties_and_unknown(self):
        rows = [dict(label=1, score=.5), dict(label=1, score=.8),
                dict(label=0, score=.5), dict(label=0, score=.2)]
        r = video_metrics(rows)
        self.assertEqual((r['TP'], r['FP'], r['FN'], r['TN']), (2, 1, 0, 1))
        self.assertEqual(r['AUC'], .875)
        rows[0]['score'] = None
        self.assertEqual(video_metrics(rows)['status'], 'incomplete_coverage_metrics_withheld')

    def test_stream_windows_reuse_source_frames_without_padding(self):
        rng = np.random.default_rng(5)
        frames = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(24)]
        windows = [dict(frame_indices=list(range(16))), dict(frame_indices=list(range(8, 24)))]
        with patch('experiments.common.video.dense_evaluation.checked_frames', return_value=iter(enumerate(frames))):
            info = {}
            batches = list(dense_batches('unused', np.arange(24), windows, 'unused', info, batch_size=1))
        self.assertEqual(info['windows'], 2)
        self.assertEqual(batches[0][0].shape, (1, 16, 112, 112, 3))
        np.testing.assert_array_equal(batches[0][0][0, 8:], batches[1][0][0, :8])

    def test_real_decoder_selection_matches_full_physical_order(self):
        ffmpeg = Path('D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe')
        if not ffmpeg.is_file():
            self.skipTest('Local portable FFmpeg fixture unavailable')
        parent = ROOT / '.codex/docs/.temp/dense-tests'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            video = Path(folder) / 'fixture.mp4'
            rng = np.random.default_rng(7)
            frames = rng.integers(0, 256, (64, 256, 256, 3), dtype=np.uint8)
            subprocess.run([str(ffmpeg), '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', '256x256', '-r', '16', '-i', 'pipe:0', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', str(video)], input=frames.tobytes(), check=True)
            pts = np.arange(64) / 16
            all_frames = dict(checked_frames(video, pts, ffmpeg, {}))
            selected = [0, 3, 7, 32, 63]
            actual = dict(checked_frames(video, pts, ffmpeg, {}, selected, Path(folder) / 'select.txt'))
            self.assertEqual(list(actual), selected)
            for i in selected:
                np.testing.assert_array_equal(actual[i], all_frames[i])
            with self.assertRaises(RuntimeError):
                list(checked_frames(video, pts + .1, ffmpeg, {}))


if __name__ == '__main__':
    unittest.main()
