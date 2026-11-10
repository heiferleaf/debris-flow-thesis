"""Small synthetic contracts; no production weights, video, JPEGs or training."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video import training_replay_diagnostic as d
from experiments.common.video.supervised_runtime import ClipInput


def window(uid, start, label=0, group='group'):
    return dict(unit_id=uid, group_id=group, label=label, frame_indices=list(range(start, start+16)),
                source_pts_seconds=list(np.linspace(0, 3, 16)),
                frames=[dict(path=f'/f/{uid}/{i}', jpeg_sha256='a'*64, rgb112_sha256='b'*64)
                        for i in range(start, start+16)])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(6, 512))
        self.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.5), nn.Linear(32, 1))
        self.calls = 0

    def forward(self, x):
        if self.training or torch.is_grad_enabled() or x.device.type != 'cpu' or x.dtype != torch.float32:
            raise AssertionError('Replay execution contract broken')
        self.calls += 1
        return self.head(self.backbone(x)).reshape(-1)


class ExposureAndSelection(unittest.TestCase):
    def test_actual_epochs_and_alias_dedup(self):
        bank = dict(n=window('neg', 0), p=window('pos', 0, 1),
                    alias=window('neg', 0), unexecuted=window('unseen', 100))
        schedule = [['n', 'p'], ['alias', 'p']] + [['unexecuted', 'p']]*18
        history = [dict(completed_epochs=i+1, train_clip_ids=ids, samples=2,
                        train_P=1, train_N=1, finite=True) for i, ids in enumerate(schedule[:2])]
        rows, counts = d.exposure_candidates(dict(clip_bank=bank, epoch_clip_ids=schedule,
                                                 validation_clip_ids=None), history, 2)
        self.assertEqual(counts, dict(P=1, N=1))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['unit_id'] for r in rows}, {'pos', 'neg'})
        self.assertEqual(next(r for r in rows if not r['label'])['exposure_epochs'], [1, 2])
        self.assertEqual(next(r for r in rows if not r['label'])['exposure_clip_ids'], ['n', 'alias'])
        with self.assertRaises(ValueError):
            d.exposure_candidates(dict(clip_bank=bank, epoch_clip_ids=schedule,
                                       validation_clip_ids=None), history, 20)

    def test_group_first_video_cap_shortage_and_score_independence(self):
        rows = []
        for group in ('a', 'b', 'c'):
            for uid in range(2):
                for start in range(4):
                    row = window(f'{group}{uid}', start*20, group=group)
                    row.update(clip_id=f'{group}{uid}:{start}', score=object())
                    rows.append(row)
        chosen, coverage = d.choose_panel(rows, 3, target=3)
        self.assertEqual({r['group_id'] for r in chosen}, {'a', 'b', 'c'})
        first = d.choose_panel(rows, 3)[0]
        second = d.choose_panel(list(reversed(rows)), 3)[0]
        self.assertEqual([r['selection_sha256'] for r in first], [r['selection_sha256'] for r in second])
        # Six distinct videos get one window each before second windows.
        self.assertEqual(len({r['unit_id'] for r in first[:6]}), 6)
        _, coverage = d.choose_panel(rows, 3)
        self.assertEqual(len(first), 12)
        self.assertEqual(coverage['0']['shortage'], 20)
        self.assertEqual(coverage['1']['shortage'], 32)
        self.assertTrue(all(n == 2 for n in coverage['0']['per_video_counts'].values()))
        self.assertEqual(len({(r['unit_id'], tuple(r['frame_indices'])) for r in first}), len(first))

    def test_heterogeneous_groups_exhaust_global_unused_videos_first(self):
        rows = [window('small-only', start*20, group='small') for start in range(2)]
        rows.extend(window(f'large-{i}', 0, group='large') for i in range(3))
        for row in rows:
            row['clip_id'] = row['unit_id']+str(row['frame_indices'][0])
        selected, _ = d.choose_panel(rows, 1, target=4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({r['unit_id'] for r in selected}), 4)
        selected, _ = d.choose_panel(rows, 1, target=5)
        self.assertEqual(len({r['unit_id'] for r in selected[:4]}), 4)
        self.assertEqual(len(selected), 5)


class MetricsAndRoute(unittest.TestCase):
    def test_general_weight_panel_constant(self):
        reference = d.constant_reference(32, 32, .4)
        self.assertAlmostEqual(reference['q_star'], 2/7)
        self.assertNotEqual(reference['q_star'], .5)
        optimum = -(12.8*math.log(2/7)+32*math.log(5/7))/64
        self.assertAlmostEqual(reference['optimum_weighted_bce'], optimum)
        self.assertAlmostEqual(reference['q_half_weighted_bce'], .7*math.log(2))
        self.assertAlmostEqual(d.constant_reference(10, 4, .4)['q_star'], .5)
        rows = [dict(label=0, logit=0., score=.5), dict(label=1, logit=2., score=d.probability(2.))]
        metrics = d.panel_metrics(rows, .4)
        self.assertEqual((metrics['TN'], metrics['FP']), (0, 1))
        self.assertEqual(metrics['AUC'], 1.)
        self.assertEqual(metrics['AUC_score_kind'], 'FP32_logit')
        self.assertAlmostEqual(metrics['original_weight_panel_bce'], (math.log(2)+.4*d.bce(2., 1))/2)

    def test_fp64_variation_and_zero_layer(self):
        stats = d.LayerStatistics()
        samples = np.array([[1e8+1, 0], [1e8-1, 0]], dtype=np.float64)
        for row in samples:
            stats.add(row)
        summary = stats.summary()
        self.assertEqual(summary['centered_sum_squares'], 2.)
        self.assertAlmostEqual(summary['relative_sample_variation'],
                               np.sum((samples-samples.mean(axis=0))**2)/(np.sum(samples**2)+1e-12))
        self.assertEqual(summary['always_zero_unit_fraction'], .5)
        zero = d.LayerStatistics()
        zero.add([0, 0]); zero.add([0, 0])
        self.assertEqual(zero.summary()['relative_sample_variation'], 0.)
        self.assertTrue(zero.summary()['all_zero'])
        self.assertEqual(zero.summary()['always_zero_unit_fraction'], 1.)
        with self.assertRaises(FloatingPointError):
            zero.add([math.nan, 0])

    def test_partial_never_triggers_a_training_branch(self):
        folds = [dict(fold=i, status='complete', panel_quota_complete=True, selected_windows=64,
                      scored_windows=64, metrics=dict(P=32, N=32, AUC=.9, negative_Recall=0.))
                 for i in range(1, 6)]
        self.assertEqual(d.frozen_route(folds), 'calibration_candidate')
        low = deepcopy(folds); low[2]['metrics']['AUC'] = .79
        self.assertEqual(d.frozen_route(low), 'linear_head_candidate')
        for field, value in (('status', 'partial'), ('panel_quota_complete', False), ('scored_windows', 63)):
            partial = deepcopy(low); partial[0][field] = value
            self.assertEqual(d.frozen_route(partial), 'undetermined')
        for field, value in (('AUC', math.nan), ('N', 0)):
            partial = deepcopy(low); partial[0]['metrics'][field] = value
            self.assertEqual(d.frozen_route(partial), 'undetermined')
        self.assertEqual(d.frozen_route(low[:4]), 'undetermined')


class ExecutionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_once_cpu_eval_no_grad_and_hooks(self):
        torch.manual_seed(7)
        net = Tiny().eval()
        pixels = torch.zeros((1, 16, 2, 2, 3), dtype=torch.uint8)
        value, layers, fingerprint = d.score_one(net, ClipInput('rgb_zero').eval(), pixels)
        self.assertEqual(net.calls, 1)
        self.assertTrue(math.isfinite(value))
        self.assertEqual({k: len(v) for k, v in layers.items()}, d.LAYER_WIDTHS)
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertTrue(all(len(m._forward_hooks) == 0 for m in net.modules()))
        net.train()
        with self.assertRaises(ValueError):
            d.score_one(net, ClipInput('rgb_zero').eval(), pixels)

    def test_supervisor_hard_timeout_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared = root/'prepared'; prepared.mkdir()
            (prepared/'manifest.json').write_text('{}', encoding='utf-8')
            entry = root/'fake_worker.py'
            entry.write_text('import time\ntime.sleep(5)\n', encoding='utf-8')
            result = d.replay(root, prepared, entry, seconds=1)
            self.assertTrue(result['timed_out'])
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['routing'], 'undetermined')
            self.assertLess(result['seconds'], 4)
            with self.assertRaises(FileExistsError):
                d.replay(root, prepared, entry, seconds=1)

    def test_durable_score_survives_stale_progress_without_complete_route(self):
        row = dict(physical_sha256='id', unit_id='u', group_id='g', label=0)
        fold = dict(fold=1, windows=[row], original_pos_weight=.4,
                    panel_quota_complete=False, coverage={})
        score = dict(row, fold=1, logit=0., score=.5, bce=math.log(2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'window-scores.jsonl').write_text(json.dumps(score)+'\n', encoding='utf-8')
            folds, error = d.accepted_prefix(path, dict(folds=[fold]), dict(folds=[]))
            self.assertIsNone(error)
            self.assertEqual(folds[0]['scored_windows'], 1)
            self.assertEqual(folds[0]['layer_summary_windows'], 0)
            self.assertEqual(folds[0]['status'], 'partial')
            self.assertEqual(d.frozen_route(folds), 'undetermined')

    def test_truncated_ascii_or_utf8_tail_preserves_durable_prefix(self):
        row = dict(physical_sha256='id', unit_id='u', group_id='中文组', label=0)
        fold = dict(fold=1, windows=[row], original_pos_weight=.4,
                    panel_quota_complete=False, coverage={})
        score = dict(row, fold=1, logit=0., score=.5, bce=math.log(2))
        prefix = json.dumps(score, ensure_ascii=False).encode('utf-8')+b'\n'
        for tail in (b'{"group_id":', b'{"group_id":"\xe6'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                (path/'window-scores.jsonl').write_bytes(prefix+tail)
                folds, issue = d.accepted_prefix(path, dict(folds=[fold]), dict(folds=[]))
                self.assertEqual(issue['line'], 2)
                self.assertEqual(folds[0]['scored_windows'], 1)
                self.assertEqual(folds[0]['status'], 'partial')
                self.assertEqual(d.frozen_route(folds), 'undetermined')


if __name__ == '__main__':
    unittest.main()
