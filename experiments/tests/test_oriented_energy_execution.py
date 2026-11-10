"""Small new-risk fixtures: fusion fit contract, shared dense EOF, unique gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from experiments.tests import test_cv_supervised_training as fixtures
from experiments.common.video import oriented_energy_training as core
from experiments.common.video.oriented_energy_model import EnergyHead
from experiments.common.video.oriented_energy_plan import epoch_block
from experiments.common.video.oriented_energy_evaluation import score_stream
from experiments.common.video.oriented_energy_metrics import gate
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.supervised_runtime import ClipInput


class TinyFusion(fixtures.Tiny):
    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        self.head = EnergyHead(self.head)

    def forward(self, x):
        d = torch.arange(1, 113, dtype=x.dtype, device=x.device).div(112).expand(len(x), 112)
        return self.head(self.backbone(x), d)


class FakeEnergy(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0

    def forward(self, x, modes):
        self.calls += 1
        return {m: torch.ones(len(x), 112) for m in modes}


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(.2)); self.energy = FakeEnergy()

    def forward(self, x, descriptions=None):
        return x.mean((1, 2, 3, 4)) + self.weight


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_new_fit_ordinary_epoch_loop_metadata_and_energy_group(self):
        f = fixtures.CVFitTests('runTest'); f.setUp()
        self.addCleanup(f.doCleanups)
        train, val = f.add_clips('train', 25, 17), f.add_clips('val', 2, 1)
        result = core.fit(f.root, f.root / 'fusion', mode='B', kernel_path='unused_cpu_fixture', kernel_sha256='f' * 64,
            fold=1, bank=f.bank, epoch_clip_ids=f.epochs(train), validation_clip_ids=val,
            verified_frame_index=f.refs, input_fingerprints=f.fingerprints, max_epochs=2,
            model_factory=TinyFusion, device='cpu', workers=0, cache_mib=2)
        self.assertEqual(result['completed_epochs'], 2); self.assertEqual(result['optimizer_steps'], 4)
        profile = json.loads((f.root / 'fusion/profile.json').read_text())
        self.assertEqual(profile['energy_mode'], 'B'); self.assertEqual(profile['dataloader_generator_seed'], 100007)
        self.assertEqual(profile['optimizer']['initial_state_entries'], 0)
        checkpoint = torch.load(f.root / 'fusion/last.pt', map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['state_dict']['head.energy_weight'].shape, (112,))
        self.assertEqual(checkpoint['optimizer']['param_groups'][1]['lr'], .001)
        self.assertEqual(checkpoint['state_dict']['backbone.2.num_batches_tracked'].item(), 0)
        history = [json.loads(s) for s in (f.root / 'fusion/epochs.jsonl').read_text().splitlines()]
        self.assertEqual([r['train_clip_ids'] for r in history], f.epochs(train)[:2])
        self.assertTrue(all(r['actual_effective_batch_sizes'] == [13, 12] for r in history))
        self.assertEqual(epoch_block(ROOT / 'experiments/common/video/oriented_energy_training.py'),
                         epoch_block(ROOT / 'experiments/common/video/cv_supervised_training.py'))

    def test_shared_batch_descriptors_and_no_score_acceptance_before_EOF(self):
        models = {m: FakeModel().eval() for m in ('A', 'B', 'C')}
        def stream(info):
            meta = [dict(unit_id='fixture', frame_indices=list(range(i * 16, (i + 1) * 16)),
                source_pts_seconds=[i * 3 + j / 5 for j in range(16)], target_start_seconds=i * 3) for i in range(2)]
            yield np.zeros((2, 16, 2, 2, 3), dtype=np.uint8), meta
            info.update(windows=2, rgb112_digest='canonical-fixture')
        rows = {a: [] for a in models}; info = {}
        result = score_stream(stream(info), info, 'canonical-fixture', models, ClipInput('rgb_zero'),
            {a: r.append for a, r in rows.items()})
        self.assertEqual(result['windows'], 2)
        self.assertEqual(models['B'].energy.calls, 1); self.assertEqual(models['C'].energy.calls, 0)
        self.assertTrue(all(len(r) == 2 for r in rows.values()))
        info = {}
        with self.assertRaisesRegex(ValueError, 'EOF'):
            score_stream(stream(info), info, 'wrong-digest', models, ClipInput('rgb_zero'), {a: lambda r: None for a in models})

    def test_shared_video_not_double_counted_and_mixed_role_repair_not_promoted(self):
        sources = {'p': dict(video_label='pos', group_id='g1'), 'shared': dict(video_label='neg', group_id='g1'),
                   'n1': dict(video_label='neg', group_id='g1'), 'n2': dict(video_label='neg', group_id='g2')}
        roles = {'1': dict(video_ids=['p', 'shared', 'n1']), '4': dict(video_ids=['p', 'shared', 'n2'])}
        dense = {}
        for fold in ('1', '4'):
            dense[fold] = {}
            for arm in ('A', 'B', 'C'):
                rows = [dict(unit_id=u, label=int(sources[u]['video_label'] == 'pos'), group_id=sources[u]['group_id'],
                    score=.8 if u == 'p' or arm != 'C' else .2) for u in roles[fold]['video_ids']]
                dense[fold][arm] = dict(status='inner_role_dense_complete', scores=rows, metrics=video_metrics(rows))
        fragment = {f: {a: dict(passed=True) for a in ('A', 'B', 'C')} for f in ('1', '4')}
        result = gate(dense, fragment, sources, roles)
        self.assertEqual(len(result['references']['A']['R']), 3)  # Four repaired roles, only three unique videos.
        self.assertTrue(result['promote_to_matched_formal_design_only'])
        changed = deepcopy(dense)
        for arm in ('A', 'B'):
            next(r for r in changed['4'][arm]['scores'] if r['unit_id'] == 'shared')['score'] = .2
        next(r for r in changed['4']['C']['scores'] if r['unit_id'] == 'shared')['score'] = .8
        for arm in ('A', 'B', 'C'):
            changed['4'][arm]['metrics'] = video_metrics(changed['4'][arm]['scores'])
        result = gate(changed, fragment, sources, roles)
        self.assertIn('shared', result['references']['A']['W'])
        self.assertNotIn('shared', result['references']['A']['R'])
        self.assertIn('shared', result['references']['A']['role_inconsistent'])
        self.assertFalse(result['promote_to_matched_formal_design_only'])
        fragment['1']['B']['passed'] = False
        self.assertFalse(gate(dense, fragment, sources, roles)['promote_to_matched_formal_design_only'])


if __name__ == '__main__':
    unittest.main()
