"""EXP-018 schedule and FP32 objective contracts; CPU fixtures, no rental."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.paired_plan import build_epoch, validate_epoch, validate_window
from experiments.common.video.paired_runtime import train_paired_batch, validate_pairs
from experiments.common.video.supervised_runtime import ClipInput, train_effective_batch
from experiments.common.video.supervised_plan import digest


def fixture():
    bank = {}
    def add(group, kind, count):
        for i in range(count):
            cid = '%s:%s:%03d' % (group, kind, i)
            bank[cid] = dict(clip_id=cid, group_id=group, split='train', kind=kind,
                             unit_id='%s:%s:%02d' % (group, kind, i // 4), segment_id='s')
    for i, quota in enumerate((4, 4, 4, 4, 12, 16, 16, 20)):
        add('g%d' % i, 'pos', quota)
        add('g%d' % i, 'neg', quota)
    add('pos_only', 'pos', 276)
    add('neg_only', 'neg', 68)
    return bank


class TinyDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv3d(6, 4, 1), nn.BatchNorm3d(4),
                                      nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Dropout(.5), nn.Linear(6, 1))
    def forward(self, x):
        return self.head(self.backbone(x)).reshape(-1)
    def train(self, mode=True):
        super().train(mode)
        self.backbone[1].eval()
        return self


class PlanTests(unittest.TestCase):
    def test_schedule_closure_quotas_and_independent_prng(self):
        bank = fixture()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        rng = np.random.default_rng(7)
        epochs = [build_epoch(list(bank), bank, rng, epoch=i) for i in range(20)]
        for epoch in epochs:
            validate_epoch(epoch, list(bank), bank)
            self.assertEqual(len(epoch['batches']), 21)
            self.assertEqual(Counter(len(b['same_pairs']) for b in epoch['batches']), {4: 20, 0: 1})
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertEqual(epochs[0], build_epoch(list(bank), bank, np.random.default_rng(7), epoch=0))
        self.assertNotEqual(epochs[0]['batches'], epochs[1]['batches'])

    def test_invalid_pairs_coverage_and_source_quota_refused(self):
        bank = fixture()
        good = build_epoch(list(bank), bank, np.random.default_rng(7))
        for mutate in ('duplicate', 'cross_group', 'cross_block', 'label', 'quota'):
            row, changed = deepcopy(good), deepcopy(bank)
            b = next(b for b in row['batches'] if b['same_pairs'])
            if mutate == 'duplicate':
                b['clip_ids'][0] = b['clip_ids'][1]
            elif mutate == 'cross_group':
                b['cross_pairs'] = deepcopy(b['same_pairs'])
            elif mutate == 'cross_block':
                p, n = b['same_pairs'][0]
                b['same_pairs'][0] = [p, (n + 8) % 24]
            elif mutate == 'label':
                changed[b['clip_ids'][b['same_pairs'][0][0]]]['kind'] = 'neg'
            else:
                for r in changed.values():
                    if r['group_id'] == 'pos_only':
                        r['unit_id'] = 'one_video'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_epoch(row, list(bank), changed)

    def test_window_closed_class_interval_and_quality_run(self):
        row = dict(clip_id='c', unit_id='u', group_id='g', segment_id='s', split='train',
                   kind='pos', frame_indices=list(range(16)), source_pts_seconds=np.linspace(0, 3, 16).tolist())
        seg = dict(unit_id='u', group_id='g', segment_id='s', split='train', kind='pos',
                   sampling_start_sec='0', sampling_end_sec='3.01')
        video = dict(unit_id='u', group_id='g', split='train', clean_runs_json='[[0, 16]]')
        validate_window(row, seg, video)
        for mutation in ('end', 'start', 'cut', 'label', 'nan', 'duplicate_frame'):
            r, s, v = deepcopy(row), deepcopy(seg), deepcopy(video)
            if mutation == 'end': s['sampling_end_sec'] = '3'
            elif mutation == 'start': s['sampling_start_sec'] = '0.01'
            elif mutation == 'cut': v['clean_runs_json'] = '[[0, 8], [8, 16]]'
            elif mutation == 'label': s['kind'] = 'neg'
            elif mutation == 'nan': r['source_pts_seconds'][4] = float('nan')
            else: r['frame_indices'][4] = r['frame_indices'][3]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_window(r, s, v)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.net = TinyDropoutNet().train()
        self.u8 = torch.randint(0, 256, (24, 16, 3, 4, 3), dtype=torch.uint8)
        self.labels = torch.tensor(([1.] * 4 + [0.] * 4) * 3)
        self.pairs = [[i, i + 4] for i in range(4)]
        self.transform = ClipInput('rgb_zero')

    def test_dropout_chunk_reference_gradient_and_bn_fixed(self):
        ref, actual = deepcopy(self.net), deepcopy(self.net)
        opt = torch.optim.Adam(actual.parameters(), lr=1e-3)
        opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-3)
        before = {k: v.clone() for k, v in actual.named_buffers()}
        rng = torch.get_rng_state().clone()
        # Independent reference retains three same-sized graphs, then one backward.
        logits = [ref(self.transform(x)) for x in self.u8.split(8)]
        bce = sum(nn.functional.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor(.4),
                  reduction='sum') for z, y in zip(logits, self.labels.split(8)))
        pair_sum = nn.functional.softplus(logits[0][4:] - logits[0][:4]).sum()
        loss = bce / 24 + .1 * (21 / 80) * pair_sum
        loss.backward()
        expected_rng = torch.get_rng_state().clone()
        opt_ref.step()
        torch.set_rng_state(rng)
        result = train_paired_batch(actual, opt, self.u8, self.labels, self.transform, .4,
                                    self.pairs, rank_lambda=.1)
        self.assertEqual(result['pair_count'], 4)
        self.assertAlmostEqual(result['pair_sum'], pair_sum.item(), places=6)
        self.assertAlmostEqual(result['loss'], loss.item(), places=6)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for p, q in zip(actual.parameters(), ref.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
        for k, v in actual.named_buffers():
            self.assertTrue(torch.equal(v, before[k]))
        self.assertGreater(actual.backbone[1].weight.grad.norm(), 0)

    def test_lambda_zero_matches_existing_bce_and_dropout_rng(self):
        a, b = deepcopy(self.net), deepcopy(self.net)
        oa, ob = torch.optim.Adam(a.parameters()), torch.optim.Adam(b.parameters())
        rng = torch.get_rng_state().clone()
        old = train_effective_batch(a, oa, self.u8, self.labels, self.transform, .4, 8)
        old_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        new = train_paired_batch(b, ob, self.u8, self.labels, self.transform, .4,
                                 self.pairs, rank_lambda=0.)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(old['loss'], new['bce'])
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_micro_closure_labels_and_no_reused_endpoints(self):
        for pairs in ([[0, 8]], [[0, 1]], [[4, 0]], [[0, 4], [0, 5]], [[.5, 4]], [[True, 4]], [[0, 24]]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                validate_pairs(pairs, self.labels, 8)

    def test_stable_extremes_and_nonfinite_refuses_update(self):
        class Extreme(TinyDropoutNet):
            def forward(self, x):
                return super().forward(x) * 0 + self.head[-1].bias[0] * 1e4
        finite = Extreme().train()
        with torch.no_grad(): finite.head[-1].bias.fill_(1.)
        r = train_paired_batch(finite, torch.optim.Adam(finite.parameters()), self.u8,
                               self.labels, self.transform, .4, self.pairs)
        self.assertTrue(np.isfinite(r['loss']))
        for gradient_failure in (False, True):
            net = deepcopy(self.net)
            opt = torch.optim.Adam(net.parameters())
            before = deepcopy(net.state_dict())
            handle = None
            if gradient_failure:
                handle = next(net.parameters()).register_hook(lambda g: g * float('nan'))
            else:
                with torch.no_grad(): net.head[-1].bias.fill_(float('inf'))
                before = deepcopy(net.state_dict())
            with self.assertRaises(FloatingPointError):
                train_paired_batch(net, opt, self.u8, self.labels, self.transform, .4, self.pairs)
            self.assertFalse(opt.state)
            for k, v in net.state_dict().items():
                torch.testing.assert_close(v, before[k], equal_nan=True)
            if handle: handle.remove()

    def test_training_entry_preserves_checkpoint_and_validation_contract(self):
        from experiments.common.video import paired_training
        from experiments.common.video.dense_evaluation import load_selected_checkpoint
        from torch.utils.data import Dataset
        bank = fixture()
        train_ids = list(bank)
        val_ids = ['validation-%03d' % i for i in range(136)]
        rows = [(c, (), int(bank[c]['kind'] == 'pos')) for c in train_ids]
        rows += [(c, (), int(i < 96)) for i, c in enumerate(val_ids)]
        by_id = {r[0]: i for i, r in enumerate(rows)}
        base_dir = ROOT / 'data/metadata/EXP-016/training-plan-v2'
        plan_dir = ROOT / 'data/metadata/EXP-018/paired-plan-v1'
        plan = json.loads((plan_dir / 'plan.json').read_text(encoding='utf-8'))
        plan['validation_clip_ids'] = val_ids
        rng = np.random.default_rng(7)
        epochs = [build_epoch(train_ids, bank, rng, i) for i in range(20)]
        validation = [by_id[c] for c in val_ids]
        class FakePixels(Dataset):
            def __init__(self, input_rows, **_): self.rows = input_rows
            def __len__(self): return len(self.rows)
            def __getitem__(self, i):
                return np.full((16, 3, 4, 3), i % 251, dtype=np.uint8), self.rows[i][2], i
        results, states = [], []
        temp_root = ROOT / '.codex/docs/.temp/paired-tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            with patch.object(paired_training, '_execution_sources', return_value={
                    Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(Path(__file__).resolve())}), \
                 patch.object(paired_training, 'load_pair_plan', return_value=(plan, epochs)), \
                 patch.object(paired_training, 'load_inputs', return_value=(
                    {'pos_weight': 148 / 356}, rows, {i: list(range(504)) for i in range(20)}, validation)), \
                 patch.object(paired_training, 'ClipDataset', FakePixels), \
                 patch.object(paired_training, 'VideoClassifier', lambda pretrained: TinyDropoutNet()):
                for mode in ('bce', 'cross', 'same'):
                    out = Path(folder) / mode
                    result = paired_training.train(ROOT, plan_dir, out, mode, epochs=1,
                                                    device='cpu', workers=0, cache_mib=0)
                    results.append(result)
                    states.append(torch.get_rng_state().clone())
                    self.assertEqual(result['optimizer_steps'], 21)
                    self.assertEqual(result['completed_epochs'], 1)
                    ckpt, _ = load_selected_checkpoint(out / 'best.pt', digest(base_dir / 'plan.json'), 'rgb_zero')
                    self.assertFalse(ckpt['profile']['checkpoint_backbone'])
                    self.assertEqual(ckpt['profile']['objective_mode'], mode)
                    self.assertTrue((out / 'last.pt').is_file())
                    self.assertEqual(len(json.loads((out / 'validation-epoch-01.json').read_text())), 136)
                    record = json.loads((out / 'epochs.jsonl').read_text())
                    self.assertEqual(record['samples'], 504)
                    self.assertEqual(record['train_pair_count'], 80)
                    self.assertEqual(record['train_clip_ids'], [c for b in epochs[0]['batches'] for c in b['clip_ids']])
        self.assertEqual(len({r['initial_state_sha256'] for r in results}), 1)
        self.assertTrue(all(torch.equal(states[0], s) for s in states))


if __name__ == '__main__':
    unittest.main()
