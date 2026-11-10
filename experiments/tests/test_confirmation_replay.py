"""One small EXP-024 boundary fixture; no saved real scores/media/models."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('confirmation_replay', ROOT / 'experiments/runspecs/EXP-024/replay.py')
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def windows(times, scores, run_ids=None):
    run_ids = [0] * len(times) if run_ids is None else run_ids
    return [dict(target_start_seconds=t, start=t + .01, end=t + 3.01, score=s,
        frame_tuple_sha256=tool.tuple_digest([100 * i + n for n in range(16)]), clean_run_index=run)
        for i, (t, s, run) in enumerate(zip(times, scores, run_ids))]


class ConfirmationFixture(unittest.TestCase):
    def test_fixed_confirmation_boundaries(self):
        normal = tool.describe(list(reversed(windows([0., 1.5, 3.], [.8, .6, .4]))))
        self.assertEqual((normal['A_score'], normal['B_score'], normal['legal_pairs']), (.8, .6, 2))
        self.assertEqual((normal['A_prediction'], normal['B_prediction']), (1, 1))
        self.assertAlmostEqual(normal['A_first_trigger_end_seconds'], 3.01)
        self.assertAlmostEqual(normal['B_first_trigger_end_seconds'], 4.51)
        self.assertAlmostEqual(normal['relative_output_wait_seconds'], 1.5)
        short = tool.describe(windows([0., .2], [.8, .8]))
        self.assertEqual(short['excluded_pairs_short_tail_interval'], 1)
        self.assertIsNone(short['B_score']); self.assertIsNone(short['B_prediction'])
        self.assertIsNone(short['B_first_trigger_end_seconds']); self.assertFalse(short['covered'])
        gap = tool.describe(windows([0., 3.], [.8, .8]))
        self.assertEqual(gap['excluded_pairs_missing_window_gap'], 1)
        cross = tool.describe(windows([0., 1.5], [.8, .8], [0, 1]))
        self.assertEqual(cross['excluded_pairs_quality_or_cut'], 1)
        for case in (gap, cross):
            self.assertEqual(case['legal_pairs'], 0); self.assertIsNone(case['B_prediction'])
        negative = tool.describe(windows([0., 1.5], [.4999999701976776, .5]))
        self.assertEqual(negative['B_prediction'], 0)
        self.assertIsNone(negative['B_first_trigger_end_seconds']); self.assertIsNone(negative['relative_output_wait_seconds'])
        never = tool.describe(windows([0., 1.5], [.1, .2]))
        self.assertIsNone(never['A_first_trigger_end_seconds']); self.assertIsNone(never['B_first_trigger_end_seconds'])
        same = windows([0., 1.5], [.5, .5]); same[1]['frame_tuple_sha256'] = same[0]['frame_tuple_sha256']
        self.assertEqual(tool.describe(same)['excluded_pairs_repeated_physical_tuple'], 1)
        late = tool.describe(windows([0., 1.5, 3., 4.5, 6.], [.8, .1, .1, .7, .7]))
        self.assertAlmostEqual(late['relative_output_wait_seconds'], 6.)
        self.assertEqual(tool.clean_run_index([0, 1, 2, 3], [[0, 4], [5, 8]]), 0)
        self.assertIsNone(tool.clean_run_index([3, 4, 5], [[0, 4], [5, 8]]))
        self.assertEqual(tool.describe(windows([0., 1.5000005], [.5, .5]))['B_prediction'], 1)
        self.assertIsNone(tool.describe(windows([0., 1.500002], [.5, .5]))['B_prediction'])


if __name__ == '__main__':
    unittest.main()
