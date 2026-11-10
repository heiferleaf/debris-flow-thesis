"""Meaningful boundary/quota fixtures for metadata preparation, without media."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common/video"))
from pts_clip_sampling import (segment_candidates, nearest_window, fixed_quota_rotation,
                               dense_candidates, complement_ranges, selection_runs,
                               jpeg95_bytes, rgb112_from_jpeg, validate_pixel_std)


class InputContractTests(unittest.TestCase):
    def test_forbidden_middle_never_bridged(self):
        pts = np.arange(0, 10.05, .05)
        clean = complement_ranges([(70, 100)], len(pts))
        windows = segment_candidates(pts, clean, 0, 10)
        self.assertTrue(len(windows))
        for window in windows:
            self.assertTrue(window[-1] < 70 or window[0] >= 100)
            self.assertTrue(np.all(np.diff(window) > 0))
            self.assertTrue(np.all(np.diff(pts[window]) > 0))
        self.assertEqual(len(segment_candidates(pts, [], 0, 10)), 0)

    def test_no_borrowing_nominal_span_from_tolerance(self):
        pts = np.arange(0, 2.96, .05)
        self.assertEqual(len(segment_candidates(pts, [(0, len(pts))], 0, 4)), 0)

    def test_label_boundary_and_dedup(self):
        pts = np.arange(0, 8.05, .05)
        windows = segment_candidates(pts, [(0, len(pts))], 2, 6)
        self.assertTrue(len(windows))
        self.assertTrue(np.all(pts[windows[:, 0]] >= 2))
        self.assertTrue(np.all(pts[windows[:, -1]] < 6))
        self.assertEqual(len(np.unique(windows, axis=0)), len(windows))

    def test_duplicate_pts_and_backstep(self):
        pts = np.sort(np.concatenate((np.arange(0, 5, .05), [.2, .4, .6])))
        windows = segment_candidates(pts, [(0, len(pts))], 0, 5)
        self.assertTrue(np.all(np.diff(pts[windows], axis=1) > 0))
        with self.assertRaises(ValueError):
            nearest_window([0, .2, .1], 0, 3, 0)
        with self.assertRaises(ValueError):
            nearest_window([0, float("nan"), 3], 0, 3, 0)

    def test_every_epoch_has_fixed_quota_and_all_strata(self):
        for n in [0, 1, 2, 3, 4, 5, 17, 81, 130]:
            schedule, pools = fixed_quota_rotation(n, 20)
            k = min(4, n)
            self.assertEqual(schedule.shape, (20, k))
            for epoch in schedule:
                self.assertEqual(len(set(epoch)), k)
            for slot, pool in enumerate(pools):
                self.assertTrue(1 <= len(pool) <= 20)
                self.assertTrue(set(schedule[:, slot]).issubset(pool))

    def test_dense_grid_uses_only_quality_runs_and_end_alignment(self):
        pts = np.arange(0, 10.05, .05)
        windows, skipped = dense_candidates(pts, [(0, 41), (50, len(pts))])
        self.assertTrue(skipped)
        self.assertTrue(windows)
        self.assertAlmostEqual(windows[-1][0], pts[-1] - 3)
        self.assertTrue(all(ix[0] >= 50 for _, ix in windows))

    def test_ffmpeg_selection_runs_are_exact_for_large_irregular_sets(self):
        rng = np.random.default_rng(7)
        for selected in [list(range(4000)), list(range(0, 8000, 4)),
                         sorted(rng.choice(20000, 3500, replace=False).tolist()), [2], [2, 7]]:
            expanded = [i for a, b, step in selection_runs(selected) for i in range(a, b + 1, step)]
            self.assertEqual(expanded, selected)

    def test_jpeg_cache_and_dense_memory_share_color_and_shape_contract(self):
        bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        bgr[:, :, 2] = 240
        rgb = rgb112_from_jpeg(jpeg95_bytes(bgr))
        self.assertEqual(rgb.shape, (112, 112, 3))
        self.assertGreater(int(rgb[:, :, 0].min()), 230)
        self.assertLess(int(rgb[:, :, 2].max()), 10)

    def test_actual_pixel_threshold_is_not_relaxed_by_qc_rounding(self):
        validate_pixel_std(20.01, 20.0, 20.0)
        for actual in [19.96, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                validate_pixel_std(actual, 20.0, 20.0)


if __name__ == "__main__":
    unittest.main()
