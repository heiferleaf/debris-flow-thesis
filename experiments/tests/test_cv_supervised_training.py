"""Finite CPU-only fits with an injected tiny model; never downloads K400."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video import cv_supervised_training as core
from experiments.common.video.pts_clip_sampling import jpeg95_bytes, rgb112_from_jpeg
from experiments.common.video.supervised_runtime import ClipInput, FrameRef, train_effective_batch


class Mean(nn.Module):
    def forward(self, x):
        return x.mean((2, 3, 4))


class Tiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained is not True:
            raise AssertionError('Core changed public initialization intent')
        self.backbone = nn.Sequential(Mean(), nn.Linear(6, 3), nn.BatchNorm1d(3))
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        return self.head(self.backbone(x))

    def train(self, mode=True):
        super().train(mode)
        self.backbone[2].eval()
        return self


class NoOuterRead(dict):
    def __getitem__(self, key):
        if key == 'outer_secret':
            raise AssertionError('Outer-held-out metadata was read')
        return super().__getitem__(key)


class CVFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        parent = (ROOT / '.codex/docs/.temp').resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='cv-fit-fixture-', dir=str(parent))
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, parent)
        self.addCleanup(self.temp.cleanup)
        self.bank, self.refs = NoOuterRead(), {}
        self.pixels = []
        for i in range(16):
            encoded = jpeg95_bytes(np.full((256, 256, 3), 25 + 10 * i, dtype=np.uint8))
            p = self.root / ('frame-%02d.jpg' % i)
            p.write_bytes(encoded)
            self.pixels.append(FrameRef(str(p), hashlib.sha256(encoded).hexdigest(),
                                       hashlib.sha256(rgb112_from_jpeg(encoded).tobytes()).hexdigest()))
        proof = self.root / 'accepted-cache-and-plan-fixture.json'
        proof.write_text('{"cpu_fixture":true}', encoding='utf-8')
        self.fingerprints = {proof.name: hashlib.sha256(proof.read_bytes()).hexdigest()}

    def add_clips(self, prefix, n, positives):
        ids = []
        for j in range(n):
            cid = '%s-%02d' % (prefix, j)
            label = int(j < positives)
            class_position = j if label else j - positives
            uid = '%s-video-%d-%02d' % (prefix, label, class_position // 4)
            indices = list(range(j * 20, j * 20 + 16))
            pts = [j * 4. + i * .2 for i in range(16)]
            self.bank[cid] = dict(clip_id=cid, unit_id=uid, group_id=uid + '-group',
                segment_id='seg', kind='pos' if label else 'neg', label=label,
                frame_indices=indices, source_pts_seconds=pts)
            for i, ref in zip(indices, self.pixels):
                self.refs[(uid, i)] = ref
            ids.append(cid)
        return ids

    def epochs(self, ids):
        return [ids[i % len(ids):] + ids[:i % len(ids)] for i in range(20)]

    def fit(self, name, stage, schedule, **kwargs):
        return core.fit(self.root, self.root / name, stage=stage, fold=1, bank=self.bank,
            epoch_clip_ids=schedule, verified_frame_index=self.refs,
            input_fingerprints=self.fingerprints, device='cpu', workers=0,
            cache_mib=2, model_factory=Tiny, **kwargs)

    def test_inner_earliest_tie_outer_fresh_and_only_selected_prefix(self):
        train = self.add_clips('inner', 25, 17)
        val = self.add_clips('validation', 2, 1)
        self.bank['outer_secret'] = object()
        values = iter((1., 0., 0.))
        def scores(_net, u8, _transform, microbatch):
            return torch.full((len(u8),), next(values))
        real_adam, optimizers = torch.optim.Adam, []
        def adam(*args, **kwargs):
            result = real_adam(*args, **kwargs)
            self.assertFalse(result.state)
            optimizers.append(result)
            return result
        with patch.object(core, 'score_clips', side_effect=scores), patch.object(core.torch.optim, 'Adam', side_effect=adam):
            inner = self.fit('inner-fit', 'inner_select', self.epochs(train),
                             validation_clip_ids=val, max_epochs=3)
        self.assertEqual(inner['selected_epochs'], 2)  # Epochs2 and3 have EXACT equal BCE.
        self.assertEqual(inner['optimizer_steps'], 6)
        self.assertEqual(inner['pos_weight'], 8 / 17)
        history = [json.loads(x) for x in (self.root / 'inner-fit/epochs.jsonl').read_text().splitlines()]
        self.assertTrue(all(h['actual_effective_batch_sizes'] == [13, 12] for h in history))
        self.assertEqual(history[0]['train_clip_ids'], train)
        self.assertEqual(history[1]['train_clip_ids'], self.epochs(train)[1])
        outer_ids = train + val
        with patch.object(core, 'score_clips', side_effect=AssertionError('Outer validation called')), patch.object(core.torch.optim, 'Adam', side_effect=adam):
            outer = self.fit('outer-fit', 'fixed_outer', self.epochs(outer_ids),
                selected_epochs=inner['selected_epochs'], expected_initial_state_sha256=inner['initial_state_sha256'])
        self.assertEqual(outer['initial_state_sha256'], inner['initial_state_sha256'])
        self.assertIsNot(optimizers[0], optimizers[1])
        self.assertEqual(outer['completed_epochs'], 2)
        self.assertEqual(outer['optimizer_steps'], 4)
        self.assertEqual(outer['pos_weight'], 9 / 18)
        self.assertFalse(outer['fixed_outer_validation_evaluated'])
        self.assertNotIn('best_validation_unweighted_bce', outer)
        self.assertFalse((self.root / 'outer-fit/best.pt').exists())
        self.assertFalse(list((self.root / 'outer-fit').glob('validation-epoch-*.json')))
        records = [json.loads(x) for x in (self.root / 'outer-fit/epochs.jsonl').read_text().splitlines()]
        self.assertEqual([r['train_clip_ids'] for r in records], self.epochs(outer_ids)[:2])
        self.assertTrue(all(r['actual_effective_batch_sizes'] == [14, 13] for r in records))
        for name in ('selected.pt', 'last.pt'):
            checkpoint = torch.load(self.root / 'outer-fit' / name, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['completed_epochs'], 2)
            self.assertEqual({int(v['step']) for v in checkpoint['optimizer']['state'].values()}, {4})
        for p in self.root.glob('*-fit/profile.json'):
            profile = json.loads(p.read_text())
            self.assertEqual(profile['optimizer']['initial_state_entries'], 0)
            self.assertEqual(profile['execution_kind'], 'cpu_fixture')
            self.assertEqual(profile['mode'], 'rgb_zero')

    def test_small_quota_fit_keeps_finite_bad_predictions_and_dynamic_weight(self):
        train = self.add_clips('small', 7, 2)
        val = self.add_clips('holdout', 3, 2)
        def terrible(_net, u8, _transform, microbatch):
            return torch.full((len(u8),), -80.)
        with patch.object(core, 'score_clips', side_effect=terrible):
            result = self.fit('bad-but-finite', 'inner_select', self.epochs(train),
                              validation_clip_ids=val, max_epochs=2)
        self.assertEqual(result['status'], 'fit_complete')
        self.assertEqual(result['pos_weight'], 2.5)
        self.assertEqual(result['selected_epochs'], 1)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertGreater(result['best_validation_unweighted_bce'], 50)
        self.assertTrue((self.root / 'bad-but-finite/best.pt').exists())
        self.assertTrue((self.root / 'bad-but-finite/last.pt').exists())

    def test_outer_validation_and_changed_initialization_rejected(self):
        train = self.add_clips('train', 5, 3)
        val = self.add_clips('val', 2, 1)
        with self.assertRaisesRegex(ValueError, 'must not receive validation'):
            self.fit('leak', 'fixed_outer', self.epochs(train), validation_clip_ids=val,
                     selected_epochs=2, expected_initial_state_sha256='0' * 64)
        self.assertFalse((self.root / 'leak').exists())
        with self.assertRaisesRegex(ValueError, 'initialization differs'):
            self.fit('bad-init', 'fixed_outer', self.epochs(train), selected_epochs=2,
                     expected_initial_state_sha256='0' * 64)
        self.assertTrue((self.root / 'bad-init/failure.json').exists())

    def test_production_inner_requires_frozen_sha_and_mismatch_precedes_optimizer(self):
        train=self.add_clips('guard-train',5,3);val=self.add_clips('guard-val',2,1)
        with patch.object(core,'VideoClassifier') as factory:
            with self.assertRaisesRegex(ValueError,'frozen initialization SHA'):
                core.fit(self.root,self.root/'missing-init',stage='inner_select',fold=1,bank=self.bank,
                    epoch_clip_ids=self.epochs(train),verified_frame_index=self.refs,
                    input_fingerprints=self.fingerprints,validation_clip_ids=val,
                    max_epochs=1,device='cpu',workers=0)
            factory.assert_not_called()
        with patch.object(core.torch.optim,'Adam') as optimizer:
            with self.assertRaisesRegex(ValueError,'initialization differs'):
                self.fit('wrong-inner-init','inner_select',self.epochs(train),validation_clip_ids=val,
                    max_epochs=1,expected_initial_state_sha256='0'*64)
            optimizer.assert_not_called()
        failure=json.loads((self.root/'wrong-inner-init/failure.json').read_text())
        self.assertEqual(failure['optimizer_steps'],0)

    def test_nonfinite_validation_fails_without_completed_result(self):
        train = self.add_clips('train', 5, 3)
        val = self.add_clips('val', 2, 1)
        with patch.object(core, 'score_clips', return_value=torch.tensor([float('nan'), 0.])):
            with self.assertRaisesRegex(FloatingPointError, 'Nonfinite inner validation'):
                self.fit('nonfinite', 'inner_select', self.epochs(train), validation_clip_ids=val, max_epochs=1)
        self.assertTrue((self.root / 'nonfinite/failure.json').exists())
        self.assertFalse((self.root / 'nonfinite/training-result.json').exists())

    def test_full_bank_class_quota_and_source_hash_gates(self):
        train = self.add_clips('train', 5, 3)
        val = self.add_clips('val', 2, 1)
        with self.assertRaisesRegex(ValueError, 'twenty-epoch'):
            self.fit('short', 'inner_select', self.epochs(train)[:2], validation_clip_ids=val, max_epochs=1)
        bad = self.epochs(train); bad[19] = train[:-1]
        with self.assertRaisesRegex(ValueError, 'class quota'):
            self.fit('changed-quota', 'inner_select', bad, validation_clip_ids=val, max_epochs=1)
        (self.root / next(iter(self.fingerprints))).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'input SHA mismatch'):
            self.fit('changed-proof', 'inner_select', self.epochs(train), validation_clip_ids=val, max_epochs=1)

    def test_train_and_fixed_validation_reject_fifth_video_class_window(self):
        train = self.add_clips('quota', 6, 5)
        val = self.add_clips('validation', 2, 1)
        self.bank[train[4]]['unit_id'] = self.bank[train[0]]['unit_id']
        with self.assertRaisesRegex(ValueError, 'quota|segment64'):
            self.fit('five-train', 'inner_select', self.epochs(train), validation_clip_ids=val, max_epochs=1)
        self.assertFalse((self.root / 'five-train').exists())
        train_ok = self.add_clips('other', 5, 3)
        val_bad = self.add_clips('five-validation', 6, 5)
        self.bank[val_bad[4]]['unit_id'] = self.bank[val_bad[0]]['unit_id']
        with self.assertRaisesRegex(ValueError, 'quota|segment64'):
            self.fit('five-val', 'inner_select', self.epochs(train_ok), validation_clip_ids=val_bad, max_epochs=1)
        self.assertFalse((self.root / 'five-val').exists())

    def test_tail_microbatch_scaling_matches_full_batch_update(self):
        torch.manual_seed(11)
        for n in (13, 12, 7):
            ref = Tiny().train()
            actual = deepcopy(ref).train()
            u8 = torch.randint(0, 256, (n, 16, 2, 2, 3), dtype=torch.uint8)
            labels = (torch.arange(n) % 2).float()
            transform = ClipInput('rgb_zero')
            opt1, opt2 = torch.optim.Adam(ref.parameters(), lr=1e-3), torch.optim.Adam(actual.parameters(), lr=1e-3)
            opt1.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(ref(transform(u8)).reshape(-1), labels,
                    pos_weight=torch.tensor(2.5), reduction='mean')
            loss.backward(); opt1.step()
            result = train_effective_batch(actual, opt2, u8, labels, transform, 2.5, microbatch=8)
            self.assertAlmostEqual(result['loss'], loss.item(), places=6)
            self.assertEqual(result['n'], n)
            self.assertEqual(result['optimizer_steps'], 1)
            for a, b in zip(actual.parameters(), ref.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
