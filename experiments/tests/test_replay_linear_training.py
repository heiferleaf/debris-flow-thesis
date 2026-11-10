"""One pure fixture for selected-E identity and role-local negative bags; no torch/model/files."""
import copy
import csv
import hashlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import importlib
_M = importlib.import_module('experiments.runspecs.EXP-017.replay_training')
index_lines, select_epoch, summarize = _M.index_lines, _M.select_epoch, _M.summarize


class LookupOnly(dict):
    def __iter__(self):
        raise AssertionError('Must not traverse the clip bank')

    def items(self):
        raise AssertionError('Must not traverse the clip bank')

    def values(self):
        raise AssertionError('Must not traverse the clip bank')


class SelectedExposureFixture(unittest.TestCase):
    def test_identity_exclusions_and_negative_bag_scope(self):
        raw_index = b'\xef\xbb\xbfcache_path,unit_id\r\ndata/frame.jpg,video\r\n'
        digest = hashlib.sha256()
        records = list(csv.DictReader(index_lines(io.BytesIO(raw_index), digest)))
        self.assertEqual(records[0]['cache_path'], 'data/frame.jpg')
        self.assertEqual(digest.hexdigest(), hashlib.sha256(raw_index).hexdigest())
        def clip(uid, label, offset):
            return dict(unit_id=uid, label=label, group_id='group',
                frame_indices=list(range(offset, offset + 16)),
                source_pts_seconds=[offset + i / 5 for i in range(16)], frames=[{} for _ in range(16)])
        # Same positive video contributes both a positive and a negative supervised clip.
        bank = LookupOnly(p=clip('positive_video', 1, 0), n=clip('positive_video', 0, 100),
                          val=clip('val_video', 0, 0))
        inputs = dict(clip_bank=bank, epoch_clip_ids=[['n', 'p'] for _ in range(20)], validation_clip_ids=['val'])
        history = dict(completed_epochs=10, train_clip_ids=['n', 'p'], samples=2, train_P=1, train_N=1)
        selected = select_epoch(inputs, history, 10, 1, 1)
        self.assertEqual([r['clip_id'] for r in selected], ['n', 'p'])
        for mutation in ['order', 'label', 'physical_duplicate', 'validation']:
            data = copy.deepcopy(dict(inputs, clip_bank={cid: bank[cid] for cid in ['p', 'n', 'val']}))
            log = copy.deepcopy(history)
            if mutation == 'order':
                log['train_clip_ids'] = ['p', 'n']
            elif mutation == 'label':
                data['clip_bank']['n']['label'] = 1
            elif mutation == 'physical_duplicate':
                data['clip_bank']['n'] = data['clip_bank']['p']
            else:
                data['validation_clip_ids'] = ['n']
            with self.assertRaises(ValueError):
                select_epoch(data, log, 10, 1, 1)
        for row, score in zip(selected, [0.5, 0.9]):
            row.update(score=score, logit=0.0)
        result = summarize(selected)
        self.assertTrue(result['any_negative_bag_false_positive'])
        self.assertEqual(result['negative_bag_count'], 1)
        self.assertEqual(result['negative_bags'][0]['unit_id'], 'positive_video')
        self.assertEqual(result['negative_bags'][0]['clip_ids'], ['n'])
        # A second role is summarized independently, not merged into the first role's bag.
        selected[0]['score'] = 0.4999999701976776  # adjacent FP32 value below 0.5
        self.assertFalse(summarize(selected)['any_negative_bag_false_positive'])
        selected[0]['score'] = float('nan')
        with self.assertRaises(ValueError):
            summarize(selected)


if __name__ == '__main__':
    unittest.main()
