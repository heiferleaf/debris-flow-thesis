"""Small parser/unique-role boundary fixtures, no models or real-result output."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('report_exp021_fixture', ROOT / 'experiments/runspecs/EXP-021/report.py')
report = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_shared_roles_repair_dedup_and_reverse_role_failure(self):
        videos = []
        pools = {1: ['p', 'shared', 'n1'], 4: ['p', 'shared', 'n2']}
        for f, pool in pools.items():
            for arm in ('A', 'B', 'C'):
                for uid in pool:
                    y = int(uid == 'p'); score = .8 if y or arm != 'C' else .2
                    videos.append(dict(fold=f, arm=arm, unit_id=uid, group_id='g2' if uid == 'n2' else 'g1', label=y,
                        score=score, prediction=int(score >= .5)))
        fragment = [dict(fold=f, arm=a, fragment_floor_pass=True) for f in (1, 4) for a in ('B', 'C')]
        verdict, _, _ = report.unique_gates(videos, fragment)
        self.assertEqual(len(verdict['references']['A']['R']), 3)
        self.assertEqual(verdict['unique_group_macro_FPR'], dict(A=1., B=1., C=0.))
        self.assertTrue(verdict['promote_to_matched_formal_design_only'])
        changed = deepcopy(videos)
        for row in changed:
            if row['fold'] == 4 and row['unit_id'] == 'shared':
                row['score'] = .8 if row['arm'] == 'C' else .2; row['prediction'] = int(row['score'] >= .5)
        verdict, _, _ = report.unique_gates(changed, fragment)
        self.assertIn('shared', verdict['references']['A']['W'])
        self.assertNotIn('shared', verdict['references']['A']['R'])
        self.assertIn('shared', verdict['references']['A']['role_inconsistent'])
        self.assertEqual(verdict['unique_group_FPR']['A']['g1'], .75)
        self.assertEqual(verdict['unique_group_FPR']['C']['g1'], .25)
        self.assertFalse(verdict['promote_to_matched_formal_design_only'])
        fragment[0]['fragment_floor_pass'] = False
        self.assertFalse(report.unique_gates(videos, fragment)[0]['promote_to_matched_formal_design_only'])

    def test_twenty_epoch_parser_rejects_nonfinite_and_uncollected(self):
        parent = ROOT / '.codex/docs/.temp'; parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='report-021-fixture-', dir=str(parent)) as temp:
            directory = Path(temp) / 'f01-B-s7'; directory.mkdir()
            def write(name, value):
                (directory / name).write_text(json.dumps(value), encoding='utf-8')
            train = ['t%d' % i for i in range(408)]; val = ['v%d' % i for i in range(100)]
            labels = {c: int(i < 288) for i, c in enumerate(train)}
            labels.update({c: int(i < 72) for i, c in enumerate(val)})
            schedule = dict(epoch_clip_ids=[train for _ in range(20)], validation_clip_ids=val)
            profile = dict(fold=1, seed=7, execution_kind='registered_fit', energy_mode='B', mode='rgb_zero', requested_epochs=20)
            result = dict(status='fit_complete', stage='inner_select', fold=1, completed_epochs=20, selected_epochs=1,
                optimizer_steps=340, seconds=20., initial_state_sha256='a' * 64,
                selected_checkpoint=dict(filename='best.pt', completed_epochs=1, sha256='b' * 64))
            write('profile.json', profile); write('training-result.json', result)
            history = []
            loss = math.log1p(math.exp(-1.))
            for e in range(1, 21):
                history.append(dict(completed_epochs=e, train_clip_ids=train, samples=408, optimizer_steps=17, total_updates=e * 17,
                    actual_effective_batch_sizes=[24] * 17, train_P=288, train_N=120, seconds=1., train_weighted_bce=.4,
                    validation_unweighted_bce=loss))
                write('validation-epoch-%02d.json' % e, [dict(clip_id=c, label=labels[c], logit=1. if labels[c] else -1.) for c in val])
            def history_file():
                (directory / 'epochs.jsonl').write_text('\n'.join(json.dumps(r) for r in history), encoding='utf-8')
            history_file()
            row, epochs = report.fit_review(report.Inputs(), directory, 1, 'B', schedule, labels)
            self.assertEqual(row['selected_E'], 1); self.assertEqual(row['FP64_earliest_min_E'], 1)
            self.assertEqual(row['optimizer_updates'], 340); self.assertEqual(len(epochs), 20)
            history[10]['validation_unweighted_bce'] = float('nan'); history_file()
            with self.assertRaisesRegex(ValueError, 'BCE'):
                report.fit_review(report.Inputs(), directory, 1, 'B', schedule, labels)
            (directory / 'training-result.json').unlink()
            with self.assertRaises(report.Incomplete):
                report.fit_review(report.Inputs(), directory, 1, 'B', schedule, labels)


if __name__ == '__main__':
    unittest.main()
