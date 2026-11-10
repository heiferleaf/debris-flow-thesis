"""Shared quota and coverage invariants, no video/pixel work."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.video.supervised_plan import bucket_schedule


class PlanTests(unittest.TestCase):
    def test_one_video_not_multiplied_by_cut_rows(self):
        banks = [[f'{s}-{i}' for i in range(80)] for s in range(6)]
        rows = bucket_schedule(banks)
        self.assertTrue(all(len(set(row)) == 4 for row in rows))
        self.assertEqual({cid.split('-')[0] for row in rows[:2] for cid in row}, set(map(str, range(6))))

    def test_small_segments_not_discarded_and_same_epoch_never_repeated(self):
        for sizes in ((1,), (4,), (5,), (6,), (1, 80), (2, 80),
                      (1, 1), (1, 1, 5), (1, 1, 1, 1, 1, 1), (80,)):
            banks = [[f'{s}-{i}' for i in range(size)] for s, size in enumerate(sizes)]
            rows = bucket_schedule(banks)
            k = min(4, sum(sizes))
            self.assertTrue(all(len(row) == len(set(row)) == k for row in rows))
            seen = {cid.split('-')[0] for row in rows[:(len(sizes) + k - 1) // k] for cid in row}
            self.assertEqual(seen, set(map(str, range(len(sizes)))))

    def test_whole_span_strata_and_determinism(self):
        rows = bucket_schedule([list(range(80))])
        self.assertEqual(rows[0], [0, 20, 40, 60])
        self.assertEqual(rows[-1], [19, 39, 59, 79])
        self.assertEqual(rows, bucket_schedule([list(range(80))]))

    def test_multiple_segment_banks_reach_their_ends(self):
        for n_segments in (2, 3, 6):
            banks = [[s * 1000 + i for i in range(80)] for s in range(n_segments)]
            rows = bucket_schedule(banks)
            used = {x for row in rows for x in row}
            for bank in banks:
                self.assertIn(bank[0], used)
                self.assertIn(bank[-1], used)
        self.assertEqual(bucket_schedule([list(range(80))], epochs=1)[0], [0, 26, 53, 79])
        self.assertEqual(bucket_schedule([list(range(80)), [1000]], epochs=1)[0][-1], 1000)


if __name__ == '__main__':
    unittest.main()
