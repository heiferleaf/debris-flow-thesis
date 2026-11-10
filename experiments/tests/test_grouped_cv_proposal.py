"""Group isolation and inner/outer membership contracts for proposed CV."""
import unittest
from experiments.splits.code.plan_grouped_cv import propose


class GroupedCVProposalTests(unittest.TestCase):
    def test_outer_partition_and_inner_isolation(self):
        rows = [dict(unit_id=f'g{g:02d}-{label}-{i}', session_group=f'group-{g:02d}', label=label,
                     pos_quota=4 if label == 'pos' else 0, neg_quota=4 if label == 'neg' else 0)
                for g in range(15) for label in ('pos', 'neg') for i in range(2)]
        result = propose(rows, restarts=8)
        seen = []
        for f in result['folds']:
            train = set(f['inner_train']['video_ids'])
            val = set(f['inner_validation']['video_ids'])
            outer = set(f['outer_evaluation']['video_ids'])
            self.assertFalse(train & val or train & outer or val & outer)
            self.assertEqual(train | val, set(f['outer_refit_train']['video_ids']))
            self.assertEqual(len(train | val | outer), len(rows))
            self.assertFalse(set(f['inner_train']['group_ids']) & set(f['inner_validation']['group_ids']))
            seen.extend(outer)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {r['unit_id'] for r in rows})

    def test_duplicate_video_rejected(self):
        row = dict(unit_id='same', session_group='g', label='pos', pos_quota=4, neg_quota=0)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            propose([row, row])


if __name__ == '__main__':
    unittest.main()
