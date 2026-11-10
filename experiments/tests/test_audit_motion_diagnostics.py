"""Numerical checks for the three failure modes that would alter this audit."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "motion", Path(__file__).resolve().parents[1] / "tools/audit_motion_diagnostics.py")
motion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(motion)


class MotionDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.a = cv2.GaussianBlur(rng.integers(20, 235, (180, 280), dtype=np.uint8), (3, 3), .6)
        self.channel = np.zeros_like(self.a, bool)
        self.channel[60:150, 140:250] = True
        self.background = np.zeros_like(self.a, bool)
        self.background[20:140, 20:100] = True

    def test_identical_frames_have_no_measurable_change(self):
        result = motion.measure_pair(self.a, self.a.copy(), self.channel, self.background, 2., .5)
        self.assertTrue(result["diagnostic_evaluable"])
        self.assertLess(result["channel_diff_mean_gray"], 1e-6)
        self.assertEqual(result["channel_joint_exceed_background_fraction"], 0.)

    def test_global_translation_removed_and_actual_dt_scale_preserved(self):
        b = cv2.warpAffine(self.a, np.float32([[1, 0, 2], [0, 1, -1]]),
                           (self.a.shape[1], self.a.shape[0]), borderMode=cv2.BORDER_REFLECT101)
        result = motion.measure_pair(self.a, b, self.channel, self.background, 2., .5)
        self.assertTrue(result["diagnostic_evaluable"])
        self.assertAlmostEqual(result["global_dx_original_px"], 4., delta=.5)
        self.assertLess(result["channel_diff_mean_gray"], 6.)
        # Move only a textured channel by 2 processed px. Background stays fixed.
        b = self.a.copy()
        b[50:160, 125:265] = np.roll(self.a[50:160, 125:265], 2, axis=1)
        two = motion.measure_pair(self.a, b, self.channel, self.background, 2., .5)
        four = motion.measure_pair(self.a, b, self.channel, self.background, 4., .5)
        self.assertGreater(two["channel_flow_q50_px_s"], 1.)
        self.assertAlmostEqual(two["channel_flow_q50_px_s"] / four["channel_flow_q50_px_s"], 2., places=5)
        self.assertGreater(two["channel_joint_exceed_background_fraction"], .5)

    def test_flat_bad_or_sparse_pair_is_diagnostic_only_not_semantic_label(self):
        a = np.full_like(self.a, 120)
        result = motion.measure_pair(a, a.copy(), self.channel, self.background, 6., .5)
        self.assertFalse(result["diagnostic_evaluable"])
        self.assertIn("interval_too_large", result["diagnostic_flags"])
        self.assertIn("channel_low_texture", result["diagnostic_flags"])
        self.assertNotIn("class", result)
        self.assertNotIn("state", result)


if __name__ == "__main__":
    unittest.main()
