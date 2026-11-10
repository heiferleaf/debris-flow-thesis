"""Only EXP020's new risks; tiny CPU pixels/models, no real-data fit/download."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import tempfile
import time
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/runspecs/EXP-020'))
from experiments.common.video import cv_supervised_training as old
from experiments.common.video import head_comparison_training as new
from experiments.tests import test_cv_supervised_training as cpu_fixtures
from plan import epoch_block, verify_baseline, read, check_initialization
SPEC = importlib.util.spec_from_file_location('exp020_pipeline_fixture', ROOT / 'experiments/runspecs/EXP-020/pipeline.py')
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
import torch
from torch import nn


class TinyDropout(cpu_fixtures.Tiny):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)
        self.head = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Dropout(.5), nn.Linear(5, 1))


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.fixture = cpu_fixtures.CVFitTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_deep7_old_new_exact_updates_exposure_selection_and_empty_adam(self):
        f = self.fixture
        train = f.add_clips('train', 25, 17)
        val = f.add_clips('validation', 2, 1)
        kwargs = dict(fold=1, bank=f.bank, epoch_clip_ids=f.epochs(train),
            validation_clip_ids=val, verified_frame_index=f.refs, input_fingerprints=f.fingerprints,
            device='cpu', workers=0, cache_mib=2, model_factory=TinyDropout, max_epochs=3)
        a = old.fit(f.root, f.root / 'old', stage='inner_select', **kwargs)
        b = new.fit(f.root, f.root / 'new', head='deep', seed=7, **kwargs)
        for k in ('selected_epochs', 'initial_state_sha256', 'final_state_sha256', 'optimizer_steps', 'best_validation_unweighted_bce'):
            self.assertEqual(a[k], b[k])
        def history(name):
            return [json.loads(s) for s in (f.root / name / 'epochs.jsonl').read_text().splitlines()]
        for x, y in zip(history('old'), history('new')):
            for k in ('train_weighted_bce', 'validation_unweighted_bce', 'train_clip_ids', 'validation_clip_ids',
                      'model_state_sha256', 'actual_effective_batch_sizes', 'grad_norm_mean', 'grad_norm_max'):
                self.assertEqual(x[k], y[k])
            self.assertEqual(y['actual_effective_batch_sizes'], [13, 12])
        for name in ('old', 'new'):
            profile = read(f.root / name / 'profile.json')
            self.assertEqual(profile['optimizer']['initial_state_entries'], 0)
            checkpoint = torch.load(f.root / name / 'last.pt', map_location='cpu', weights_only=False)
            # Tiny BN remains fixed despite Dropout-enabled train updates.
            self.assertTrue(torch.equal(checkpoint['state_dict']['backbone.2.running_mean'], torch.zeros(3)))
            self.assertTrue(torch.equal(checkpoint['state_dict']['backbone.2.running_var'], torch.ones(3)))
            self.assertEqual(checkpoint['state_dict']['backbone.2.num_batches_tracked'].item(), 0)

    def test_exact_epoch_block_and_all20_original_fold_exposures(self):
        self.assertEqual(epoch_block(ROOT / 'experiments/common/video/cv_supervised_training.py'),
                         epoch_block(ROOT / 'experiments/common/video/head_comparison_training.py'))
        folds = read(ROOT / 'data/metadata/EXP-017/role-free-plan-v3/folds.json')
        for fold, n, p, sizes in ((1, 408, 288, [24] * 17), (4, 412, 292, [23] * 16 + [22] * 2)):
            order = folds[fold - 1]['inner_train']['epoch_clip_ids']
            self.assertEqual(len(order), 20)
            for row in order:
                self.assertEqual(len(row), n)
                self.assertEqual(len(set(row)), n)
                sampler = new.EpochBatches(24)
                sampler.set_order(list(range(n)))
                self.assertEqual([len(x) for x in sampler], sizes)
            self.assertEqual(folds[fold - 1]['inner_train']['counts']['pos_windows'], p)

    def test_reuse_rejects_outer_role_changed_label_and_schedule(self):
        f = self.fixture
        train = f.add_clips('train', 4, 2)
        val = f.add_clips('val', 2, 1)
        schedule = f.epochs(train)
        folder = f.root / 'baseline'; folder.mkdir()
        inputs = dict(epoch_clip_ids=schedule, validation_clip_ids=val, clip_bank={c: f.bank[c] for c in train + val})
        (folder / 'inputs.json').write_text(json.dumps(inputs))
        (folder / 'profile.json').write_text(json.dumps(dict(stage='fixed_outer', seed=7)))
        result = dict(status='fit_complete', stage='fixed_outer', fold=1, completed_epochs=20)
        (folder / 'training-result.json').write_text(json.dumps(result))
        (folder / 'epochs.jsonl').write_text('')
        fold = dict(fold=1, inner_train=dict(epoch_clip_ids=schedule), inner_val=dict(clip_ids=val))
        with self.assertRaisesRegex(ValueError, 'role'):
            verify_baseline(f.root, folder, fold, f.bank)
        result['stage'] = 'inner_select'
        (folder / 'training-result.json').write_text(json.dumps(result))
        (folder / 'profile.json').write_text(json.dumps(dict(stage='inner_select', seed=7)))
        records = []
        predictions = [dict(clip_id=c, label=f.bank[c]['label'], logit=0.) for c in val]
        for i in range(20):
            records.append(dict(train_clip_ids=schedule[i], optimizer_steps=1, validation_unweighted_bce=.6931471805599453))
            (folder / ('validation-epoch-%02d.json' % (i + 1))).write_text(json.dumps(predictions))
        result['selected_epochs'] = 1
        (folder / 'training-result.json').write_text(json.dumps(result))
        (folder / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
        verify_baseline(f.root, folder, fold, f.bank)
        changed = deepcopy(f.bank); changed[train[0]]['label'] = 0
        with self.assertRaisesRegex(ValueError, 'identity/label'):
            verify_baseline(f.root, folder, fold, changed)
        changed = deepcopy(fold); changed['inner_train']['epoch_clip_ids'][19].reverse()
        with self.assertRaisesRegex(ValueError, 'schedule'):
            verify_baseline(f.root, folder, changed, f.bank)

    def test_forecast_never_cancels_next_fit_and_technical_failure_closes(self):
        f = self.fixture
        timer = f.root / 'timer.json'; timer.write_text('{"operator_verified":true}')
        directory = f.root / 'plan'; directory.mkdir(); (directory / 'plan.json').write_text('{}')
        items = ['f01-linear-s7', 'f01-deep-s8']
        plan = dict(fit_order=items, items=[], source_fingerprints={}, code_sha256={})
        def invoke(name, failed=False):
            calls = []
            def run(command, **kwargs):
                item = command[command.index('--item') + 1]; calls.append(item)
                target = Path(command[command.index('--out') + 1]); target.mkdir()
                (target / ('failure.json' if failed else 'training-result.json')).write_text('{}')
                return SimpleNamespace(returncode=1 if failed else 0)
            output = f.root / name
            with patch.object(pipeline, 'load', return_value=(plan, None, None, None, {})), \
                 patch.object(pipeline, 'NEW_CODE', []), \
                 patch.object(pipeline, 'os', SimpleNamespace(name='posix')), \
                 patch.object(pipeline.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 * 1024 ** 3)), \
                 patch.object(pipeline.subprocess, 'run', side_effect=run), \
                 patch.object(pipeline, 'comparison', return_value=dict(status='fixture_complete')):
                argv = ['--plan', str(directory), '--out', str(output), '--money-deadline-unix', str(time.time() + 30),
                        '--platform-timer-receipt', str(timer)]
                if failed:
                    with self.assertRaisesRegex(RuntimeError, 'Technical fit failure'):
                        pipeline.main(argv)
                else:
                    pipeline.main(argv)
            return output, calls
        output, calls = invoke('successful-matrix')
        self.assertEqual(calls, items)
        forecast = read(output / ('forecast-after-' + items[0] + '.json'))
        self.assertGreater(forecast['expected_remaining_seconds'], forecast['actual_remaining_fee_seconds'])
        self.assertEqual(forecast['policy'], 'advisory_only')
        self.assertEqual(read(output / 'closed-run.json')['status'], 'run_closed')
        output, calls = invoke('failed-matrix', failed=True)
        self.assertEqual(calls, items[:1])
        self.assertEqual(read(output / 'closed-run.json')['status'], 'run_failed_closed')

    def test_local_environment_observation_is_not_accepted_as_production_proof(self):
        from experiments.common.video.head_comparison_training import WEIGHTS_SHA
        proof = dict(status='local_environment_observation_not_production_proof',
            public_weights_sha256=WEIGHTS_SHA, initializations={})
        path = self.fixture.root / 'observation.json'; path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError, 'local observations are insufficient'):
            check_initialization(self.fixture.root, path)

    def test_shell_bounded_shutdown_preserves_failure_and_ignores_receipt_write_failure(self):
        bash = Path('C:/Program Files/Git/bin/bash.exe')
        if not bash.is_file():
            self.skipTest('Isolated Git Bash fixture unavailable')
        root = self.fixture.root
        source = (ROOT / 'experiments/runspecs/EXP-020/run.sh').read_text(encoding='utf-8')
        entry = '/root/miniconda3/bin/python -B experiments/runspecs/EXP-020/pipeline.py'
        self.assertEqual(source.splitlines()[-1], entry + ' "$@"')
        # Only the temporary copy substitutes executable paths; production has
        # no command-injection/test hook and no real shutdown is called here.
        script = source.replace('cd "$(dirname "$0")/../../.."', 'cd "$test_workdir"')
        script = script.replace('/usr/bin/timeout', '"$test_timeout"').replace('/usr/bin/shutdown', '"$test_shutdown"')
        script = script.replace(entry, '/bin/bash "$test_pipeline"')
        (root / 'run-fixture.sh').write_text(script, encoding='utf-8')
        (root / 'timeout.sh').write_text('''#!/bin/bash
[[ "$1" == '--kill-after=2s' && "$2" == '15s' ]] || exit 99
shift 2
exec "$@"
''', encoding='utf-8')
        (root / 'shutdown.sh').write_text('printf requested >> "$test_calls"\nexit "$test_shutdown_code"\n', encoding='utf-8')
        (root / 'pipeline.sh').write_text('exit "$test_pipeline_code"\n', encoding='utf-8')
        for i, (fit_code, shutdown_code, evidence_blocked, expected) in enumerate(((0, 0, False, 0), (0, 9, False, 125), (7, 9, False, 7), (0, 0, True, 0))):
            output = root / ('out-%d' % i); output.mkdir()
            if evidence_blocked:
                (output / 'shutdown-receipt.json').mkdir()
            calls = root / ('calls-%d.txt' % i)
            # Empty PATH reproduces the real launch environment: the explicit
            # interpreter/shutdown executables must not rely on shell lookup.
            env = dict(os.environ, PATH='', test_workdir=root.as_posix(), test_timeout=(root / 'timeout.sh').as_posix(),
                test_shutdown=(root / 'shutdown.sh').as_posix(), test_pipeline=(root / 'pipeline.sh').as_posix(),
                test_calls=calls.as_posix(), test_shutdown_code=str(shutdown_code), test_pipeline_code=str(fit_code))
            result = subprocess.run([str(bash), str(root / 'run-fixture.sh'), '--out', output.as_posix()], env=env,
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(calls.read_text(), 'requested')
            if not evidence_blocked:
                self.assertEqual(read(output / 'shutdown-receipt.json')['shutdown_command_returncode'], shutdown_code)


if __name__ == '__main__':
    unittest.main()
