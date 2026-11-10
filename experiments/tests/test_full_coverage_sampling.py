"""Metadata-only decision fixtures for the full-coverage diagnostic contract."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.video.full_coverage_sampling import (  # noqa: E402
    deduplicate_physical, diagnostic_loss_coefficients, epoch_order, full_grid,
    integrate_release, physical_window_key, update_scenarios, validate_release,
)


def interval(start, end, state, mask=True):
    return {"start_sec": start, "end_sec": end, "released_state": state,
            "mask_three_class": mask, "mask_existence": mask,
            "existence_value": int(state != "N") if state in {"N", "T", "P"} else None}


class FullCoverageSamplingTest(unittest.TestCase):
    def test_fixed_anchor_has_no_label_or_QC_phase_reset(self):
        pts = np.arange(201, dtype=float) * .05 + .017
        windows = list(full_grid(pts))
        self.assertEqual([round(r["target_start_seconds"], 6) for r in windows], [.017, 3.017, 6.017])
        # A semantic boundary/QC cut at1.2s never becomes a new start.
        release = validate_release([interval(.017, 1.2, "N"), interval(1.2, 10.017, "P")])
        self.assertTrue(integrate_release(windows[0]["actual_start_seconds"], windows[0]["actual_end_seconds"], release)["semantic_three_class_complete"])
        self.assertEqual(windows[1]["frame_indices"][0], 60)

    def test_actual_PTS_cross_boundary_q_and_existence_r(self):
        result = integrate_release(.02, 3.02, validate_release([
            interval(0, 1.02, "N"), interval(1.02, 2.02, "T"), interval(2.02, 4, "P")]))
        for value in result["q_N_T_P"]:
            self.assertAlmostEqual(value, 1 / 3)
        self.assertAlmostEqual(result["r_existence"], 2 / 3)

    def test_U_cut_gap_and_maskfalse_P_never_complete_or_renormalize(self):
        for middle in (interval(1, 2, "U", False), interval(1, 2, "QC_or_semantic_pending", False), interval(1, 2, "P", False)):
            result = integrate_release(0, 3, validate_release([interval(0, 1, "N"), middle, interval(2, 3, "P")]))
            self.assertIsNone(result["q_N_T_P"])
            self.assertIsNone(result["r_existence"])
            self.assertAlmostEqual(sum(result["observed_N_T_P_fraction"]), 2 / 3)
            self.assertEqual(result["r_existence_bounds"], [1 / 3, 2 / 3])
        self.assertFalse(integrate_release(0, 3, validate_release([interval(0, 1, "N")]))["semantic_three_class_complete"])

    def test_physical_alias_dedup_and_truth_conflict(self):
        key = physical_window_key("a" * 64, list(range(16)))
        a = {"physical_window_key": key, "unit_id": "a", "source_camera": "c",
             "composition": {"q_N_T_P": [1, 0, 0]}, "legacy_guard_pass": True, "gray_QC_pass": True, "PTS_pass": True}
        b = {**a, "unit_id": "alias"}
        unique, duplicates = deduplicate_physical([a, b])
        self.assertEqual((len(unique), len(duplicates)), (1, 1))
        with self.assertRaises(ValueError):
            deduplicate_physical([a, {**b, "composition": {"q_N_T_P": [0, 0, 1]}}])

    def test_video_camera_equal_coefficients_and_static_exception(self):
        rows = []
        for camera, video, n in (("c1", "long", 8), ("c1", "short", 2), ("c2", "other", 3)):
            for i in range(n):
                rows.append({"physical_window_key": camera + video + str(i), "unit_id": video,
                             "physical_video_id": video, "source_camera": camera,
                             "composition": {"q_N_T_P": [1.0, 0.0, 0.0]}, "motion_state": "unknown"})
        coefficients, cameras = diagnostic_loss_coefficients(rows)
        self.assertAlmostEqual(sum(r["loss_coefficient_w"] for r in coefficients), len(rows))
        for camera in ("c1", "c2"):
            self.assertAlmostEqual(sum(r["loss_coefficient_w"] for r in coefficients if r["source_camera"] == camera), len(rows) / 2)
        long = sum(r["loss_coefficient_w"] for r in coefficients if r["physical_video_id"] == "long")
        short = sum(r["loss_coefficient_w"] for r in coefficients if r["physical_video_id"] == "short")
        self.assertAlmostEqual(long, short)
        static_rows = [{**r, "motion_state": "confirmed_static_N" if r["physical_video_id"] in {"long", "short"} else "confirmed_nonstatic", "source_camera": "c"} for r in rows]
        _, report = diagnostic_loss_coefficients(static_rows)
        self.assertAlmostEqual(report[0]["static_N_factor"], .5)
        # Unknown is part of the undiscounted complement D, not a motion label.
        _, report = diagnostic_loss_coefficients([{**r, "motion_state": "unknown" if r["motion_state"] == "confirmed_nonstatic" else r["motion_state"]} for r in static_rows])
        self.assertAlmostEqual(report[0]["static_N_factor"], .5)
        _, report = diagnostic_loss_coefficients([{**r, "motion_state": "confirmed_static_N"} for r in static_rows])
        self.assertEqual(report[0]["static_N_factor"], 1)

    def test_tail_update_count_and_epoch_uniqueness(self):
        for k in (0, 1, 24, 25, 49):
            report = update_scenarios(k)
            self.assertEqual(report["one_traversal_updates"], math.ceil(k / 24))
            self.assertEqual(report["twenty_traversal_scenario_updates"], 20 * math.ceil(k / 24))
            self.assertEqual(report["loss_denominator"], 24)
        rows = [{"physical_window_key": str(i)} for i in range(25)]
        self.assertEqual(sorted(epoch_order(rows)), sorted(str(i) for i in range(25)))
        self.assertEqual(epoch_order(rows, 7), epoch_order(rows, 7))


if __name__ == "__main__":
    unittest.main()
