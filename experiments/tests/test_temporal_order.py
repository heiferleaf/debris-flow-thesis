"""Physical-key/RNG/pixel-shape fixtures shared by supervised and dense callers."""
from pathlib import Path
import json
import random
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.video import temporal_order as order

MEDIA = '0123456789abcdef' * 4
FRAMES = list(range(0, 32, 2))


class TemporalOrderTests(unittest.TestCase):
    def test_canonical_json_and_fixture_are_frozen(self):
        expected = '{"frame_indices":[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30],"media_sha256":"' + MEDIA + '","permutation_seed":1901,"version":"exp019-temporal-order-v1"}'
        self.assertEqual(order.physical_window_json(MEDIA, FRAMES), expected)
        record = order.window_order_record(MEDIA, FRAMES)
        self.assertEqual(record['physical_window_sha256'], '5c8bccbf2d0650c9cbf4311323c7db68fa8bd3d5eb5bbd3427d319751252de80')
        self.assertEqual(record['permutation'], [5, 12, 8, 2, 9, 11, 13, 0, 1, 3, 10, 15, 14, 7, 4, 6])
        self.assertEqual(record['chronological_sequence_sha256'], 'e384aabddccc28c71ca8512c2a6c48f9df28d2089318483c6df299623f7262af')
        self.assertEqual(record['fixed_permutation_sequence_sha256'], 'df0d274a3335d3e1a1aa07692626ac594273f93f7d12d48d6699dd54047621ea')
        self.assertEqual(sorted(record['permutation']), list(range(16)))
        self.assertEqual(record, order.window_order_record(MEDIA.upper(), np.array(FRAMES, dtype=np.int64)))
        self.assertEqual(json.loads(record['canonical_json'])['frame_indices'], FRAMES)

    def test_alias_roles_and_entrypoints_do_not_change_physical_key_or_rng(self):
        python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        outputs = []
        for role in ('supervised_train', 'fixed_validation', 'dense_inference'):
            row = {'media_sha256': MEDIA, 'frame_indices': FRAMES, 'label': role == 'supervised_train', 'clip_id': role}
            outputs.append(order.temporal_indices(row['media_sha256'], row['frame_indices'], 'fixed_permutation'))
        self.assertEqual(outputs[0], outputs[1]); self.assertEqual(outputs[0], outputs[2])
        self.assertEqual(random.getstate(), python)
        now = np.random.get_state(); self.assertEqual(now[0], numpy[0]); np.testing.assert_array_equal(now[1], numpy[1]); self.assertEqual(now[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertNotEqual(order.window_order_record('f' * 64, FRAMES)['physical_window_sha256'], order.window_order_record(MEDIA, FRAMES)['physical_window_sha256'])
        self.assertNotEqual(order.window_order_record(MEDIA, [i+1 for i in FRAMES])['physical_window_sha256'], order.window_order_record(MEDIA, FRAMES)['physical_window_sha256'])

    def test_sort_collision_breaks_by_original_position(self):
        with patch.object(order, '_position_key', return_value=b'collision'):
            self.assertEqual(order.temporal_indices(MEDIA, FRAMES, 'fixed_permutation'), tuple(range(16)))

    def test_model_T_axis_only_keeps_same_pixels_shape_and_zero_D(self):
        rgb = np.arange(16 * 2 * 3 * 3, dtype=np.uint8).reshape(16, 2, 3, 3)
        original = rgb.copy(); canonical = FRAMES[:]
        chronological = order.apply_window_order(rgb, MEDIA, FRAMES, 'chronological')
        permuted = order.apply_window_order(rgb, MEDIA, FRAMES, 'fixed_permutation')
        self.assertEqual(permuted.shape, rgb.shape)
        np.testing.assert_array_equal(chronological, rgb)
        np.testing.assert_array_equal(np.sort(permuted.reshape(16, -1), axis=0), np.sort(rgb.reshape(16, -1), axis=0))
        np.testing.assert_array_equal(rgb, original); self.assertEqual(FRAMES, canonical)
        normalized = torch.cat([torch.from_numpy(rgb).permute(3, 0, 1, 2).float(), torch.zeros(3, 16, 2, 3)], 0)
        changed = order.apply_window_order(normalized, MEDIA, FRAMES, 'fixed_permutation', time_axis=1)
        self.assertEqual(changed.shape, (6, 16, 2, 3)); self.assertEqual(changed[3:].count_nonzero(), 0)
        torch.testing.assert_close(changed[:3], normalized[:3, order.temporal_indices(MEDIA, FRAMES, 'fixed_permutation')])

    def test_invalid_identity_or_noncanonical_frames_or_shape_refused(self):
        for frames in (FRAMES[:-1], list(reversed(FRAMES)), [0]*16, [float(i) for i in FRAMES], [True]+FRAMES[1:], [-1]+FRAMES[1:]):
            with self.assertRaises(ValueError): order.temporal_indices(MEDIA, frames, 'fixed_permutation')
        with self.assertRaises(ValueError): order.temporal_indices('bad', FRAMES, 'fixed_permutation')
        with self.assertRaises(ValueError): order.temporal_indices(MEDIA, FRAMES, 'random_each_epoch')
        with self.assertRaises(ValueError): order.apply_window_order(np.zeros((15, 2)), MEDIA, FRAMES, 'chronological')


if __name__ == '__main__': unittest.main()
