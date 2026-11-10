"""New EXP-025 risk: all five outer folds must partition videos and event groups."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('confirmation_cv', ROOT / 'experiments/runspecs/EXP-025/compare.py')
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class OuterMappingFixture(unittest.TestCase):
    def test_each_video_and_event_group_belongs_to_one_of_all_five_folds(self):
        sources = {'v%d' % fold: dict(group_id='event%d' % fold) for fold in range(1, 6)}
        folds = [dict(fold=fold, outer_eval=dict(video_ids=['v%d' % fold])) for fold in range(1, 6)]
        self.assertEqual(tool.outer_membership(folds, sources), {'v%d' % fold: fold for fold in range(1, 6)})
        duplicate = deepcopy(folds)
        duplicate[1]['outer_eval']['video_ids'].append('v1')
        missing = deepcopy(folds)
        missing[4]['outer_eval']['video_ids'] = []
        split_group = deepcopy(sources)
        split_group['v2']['group_id'] = split_group['v1']['group_id']
        for bad_folds, bad_sources in ((duplicate, sources), (missing, sources), (folds[:4], sources), (folds, split_group)):
            with self.assertRaises(ValueError):
                tool.outer_membership(bad_folds, bad_sources)


if __name__ == '__main__':
    unittest.main()
