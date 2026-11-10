"""EXP-017 metadata-only split/quota/prefix/class-weight fixtures."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.video.cv_supervised_plan import (
    representative_indices, video_schedules, validate_partitions, training_weight,
    prefix_epochs, validate_epoch_quota,
)


class CVPlanTests(unittest.TestCase):
    def test_original_twenty_epoch_representatives_then_shared_quota(self):
        selected, rotation = representative_indices(800)
        self.assertEqual(len(selected), 80)
        self.assertEqual((selected[0], selected[-1]), (0, 799))
        self.assertEqual(len(rotation), 20)
        banks = dict(pos=[['p%d' % i for i in selected], ['short']], neg=[['n%d' % i for i in selected]])
        epochs, fixed = video_schedules(banks)
        self.assertTrue(all(len(r) == 8 and len(set(r)) == 8 for r in epochs))
        self.assertEqual(len(fixed), 8)
        self.assertIn('short', set(epochs[0] + epochs[1]))
        for e in (1, 2, 7, 20):
            self.assertEqual(prefix_epochs(epochs, e), epochs[:e])
        with self.assertRaises(ValueError): prefix_epochs(epochs[:3], 2)
        with self.assertRaises(ValueError): prefix_epochs(epochs, 0)

    def test_group_isolation_and_union_are_required(self):
        sources = {u: {'group_id': g} for u, g in [('a', 'g1'), ('b', 'g1'), ('c', 'g2'), ('d', 'g3')]}
        good = dict(inner_train=['a', 'b'], inner_val=['c'], outer_train=['a', 'b', 'c'], outer_eval=['d'])
        validate_partitions(good, sources)
        for bad in (
            dict(inner_train=['a'], inner_val=['b', 'c'], outer_train=['a', 'b', 'c'], outer_eval=['d']),
            dict(inner_train=['a', 'b'], inner_val=['c'], outer_train=['a', 'b'], outer_eval=['d']),
            dict(inner_train=['a', 'b'], inner_val=['c'], outer_train=['a', 'b', 'c'], outer_eval=['c', 'd'])):
            with self.assertRaises(ValueError): validate_partitions(bad, sources)

    def test_class_weight_uses_only_supplied_training_ids(self):
        class ReadGuard(dict):
            def __getitem__(self, key):
                if key.startswith('heldout'): raise AssertionError('Heldout label was read for training N/P')
                return super().__getitem__(key)
        bank = ReadGuard(p={'label': 1}, n1={'label': 0}, n2={'label': 0}, heldout_p={'label': 1})
        self.assertEqual(training_weight(['p', 'n1', 'n2'], bank), 2.)
        with self.assertRaises(ValueError): training_weight(['p'], bank)

    def test_video_class_and_original_segment_quota_not_multiplied(self):
        bank = {str(i): dict(unit_id='u', kind='pos', segment_id='s%d' % (i // 4), label=1)
                for i in range(8)}
        validate_epoch_quota(list(bank)[:4], bank)
        with self.assertRaises(ValueError): validate_epoch_quota(list(bank), bank)
        for i in bank: bank[i]['unit_id'] = 'u' + i
        # Even distinct videos cannot share one actual original-segment key;
        # original segment keys include unit_id, and remain <=4 windows each.
        validate_epoch_quota(list(bank), bank)
        with self.assertRaises(ValueError): validate_epoch_quota(['0', '0'], bank)


if __name__ == '__main__':
    unittest.main()
