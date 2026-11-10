"""Only persistence-boundary checks; no model, video or result artifacts."""
import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('exp021_error_fixture', ROOT / 'experiments/runspecs/EXP-021/analyze_errors.py')
tool = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(tool)


def rows(times, scores):
    return [dict(unit_id='fixture', target_start_seconds=t, start=t, end=t + 3, score=s) for t, s in zip(times, scores)]


class ErrorTests(unittest.TestCase):
    def test_sparse_spike_persistent_and_video_only_FN(self):
        times = [1.5 * i for i in range(10)]
        sparse = tool.describe(rows(times, [.1] * 5 + [.5] + [.1] * 4), 0)
        persistent = tool.describe(rows(times, [.8] * 10), 0)
        self.assertEqual((sparse['positive_fraction'], sparse['longest_positive_run_windows']), (.1, 1))
        self.assertEqual((persistent['positive_fraction'], persistent['longest_positive_run_windows']), (1., 10))
        self.assertEqual(sparse['video_case'], 'FP'); self.assertEqual(persistent['false_positive_windows'], 10)
        positive = tool.describe(rows(times, [.1] * 10), 1)
        self.assertEqual(positive['video_case'], 'FN'); self.assertIsNone(positive['false_positive_windows'])
        self.assertNotIn('FN_windows', positive)

    def test_gap_tail_alignment_and_saved_FP32_threshold(self):
        case = tool.describe(list(reversed(rows([0., 1.5, 3., 7.5, 9., 9.2], [.5] * 6))), 0)
        self.assertEqual(case['longest_positive_run_windows'], 3)
        boundary = tool.describe(rows([0., 1.5, 3.], [.4999999701976776, .5, .5000000596046448]), 0)
        self.assertEqual((boundary['positive_windows'], boundary['longest_positive_run_windows']), (2, 2))
        self.assertEqual(tool.describe(rows([0., 1.500002], [.5, .5]), 0)['longest_positive_run_windows'], 1)
        mixed = rows([0., 1.5], [.8, .8]); mixed[1]['unit_id'] = 'another'
        with self.assertRaises(ValueError): tool.describe(mixed, 0)


if __name__ == '__main__': unittest.main()
