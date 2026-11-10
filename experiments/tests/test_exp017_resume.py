"""Finite CPU evidence/control fixtures; no model, media, SSH or rental action."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

from experiments.common.video import cv_resume_execution as resume
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.pts_clip_sampling import tuple_digest
from experiments.common.video.supervised_plan import digest, write_json
from experiments.common.video.supervised_runtime import FrameRef


def first_fold_fixture(root):
    parent = root / 'parent'; parent.mkdir()
    bank = {}; refs = {}; metadata = {}
    for cid, label in [('p', 1), ('n', 0)]:
        bank[cid] = dict(label=label, unit_id=cid, group_id=cid, frame_indices=list(range(16)),
                         source_pts_seconds=[j / 5 for j in range(16)])
        frames = []
        for i in range(16):
            relative = 'cache/%s/%d.jpg' % (cid, i)
            refs[(cid, i)] = FrameRef(str(root / relative), '1' * 64, '2' * 64)
            frames.append(dict(path='/root/autodl-tmp/debris-flow-thesis-exp016/' + relative,
                               jpeg_sha256='1' * 64, rgb112_sha256='2' * 64))
        metadata[cid] = dict(bank[cid], frames=frames)
    fold = dict(fold=1, inner_train={'epoch_clip_ids': [['p', 'n']] * 20},
                inner_val={'clip_ids': ['p', 'n']}, outer_train={'epoch_clip_ids': [['p', 'n']] * 20},
                outer_eval={'video_ids': ['u%02d' % i for i in range(33)]})
    frozen = dict(code_sha256={'code.py': 'a' * 64}, input_fingerprints={'metadata.json': 'b' * 64},
                  dense_reference_sha256='c' * 64)
    sources = {}; windows = {}; reference = {}; scores = []; evidence = {}; csv_rows = []
    for i, u in enumerate(fold['outer_eval']['video_ids']):
        label = int(i % 2 == 0); value = .7 if label else .2
        sources[u] = dict(video_label='pos' if label else 'neg', group_id='g' + u)
        w = dict(frame_indices=list(range(16)), target_start_seconds=0., source_pts_seconds=[j / 5 for j in range(16)])
        windows[u] = [w]; reference[u] = dict(decoded=16, rgb112_digest='d' * 64)
        evidence[u] = dict(decoded=16, windows=1, score=value, canonical_reference_matched=True, rgb112_digest='d' * 64)
        scores.append(dict(unit_id=u, label=label, group_id='g' + u, fold=1, score=value, windows=1))
        csv_rows.append(dict(unit_id=u, target_start_seconds=0., start=0., end=3., frame_tuple_sha256=tuple_digest(w['frame_indices']), score=value))
    context = dict(root=root, folds=[fold], bank=bank, refs=refs, sources=sources,
                   dense={'windows': windows, 'reference': {'videos': reference}})
    for stage, role, epochs in [('inner', 'inner_select', 20), ('outer', 'fixed_outer', 3)]:
        d = parent / 'fold-01' / stage; d.mkdir(parents=True)
        validation = ['p', 'n'] if stage == 'inner' else None
        write_json(d / 'inputs.json', dict(clip_bank=metadata, epoch_clip_ids=[['p', 'n']] * 20, validation_clip_ids=validation))
        parsed_sha = resume._canonical(dict(metadata=metadata, epoch_clip_ids=[['p', 'n']] * 20,
            validation_clip_ids=validation, files=frozen['input_fingerprints']))
        profile = dict(execution_kind='registered_fit', stage=role, fold=1, seed=7, mode='rgb_zero',
            initial_state_sha256=resume.INITIAL_SHA, expected_initial_state_sha256=resume.INITIAL_SHA,
            effective_batch=24, microbatch=8, checkpoint_backbone=False, requested_epochs=epochs,
            input_fingerprints=frozen['input_fingerprints'], supervised_pos_weight=1., parsed_inputs_sha256=parsed_sha)
        write_json(d / 'profile.json', profile)
        selected = 'best.pt' if stage == 'inner' else 'selected.pt'
        (d / selected).write_bytes(b'selected'); (d / 'last.pt').write_bytes(b'last')
        result = dict(status='fit_complete', stage=role, fold=1, completed_epochs=epochs, selected_epochs=3,
            initial_state_sha256=resume.INITIAL_SHA, fixed_outer_validation_evaluated=False,
            input_fingerprints=frozen['input_fingerprints'], optimizer_steps=epochs,
            profile_sha256=digest(d / 'profile.json'), parsed_inputs_sha256=parsed_sha,
            selected_checkpoint=dict(filename=selected, completed_epochs=3, sha256=digest(d / selected)),
            last_checkpoint=dict(filename='last.pt', completed_epochs=epochs, sha256=digest(d / 'last.pt')))
        write_json(d / 'training-result.json', result)
        with (d / 'epochs.jsonl').open('x', encoding='utf-8') as stream:
            for epoch in range(1, epochs + 1):
                row = dict(completed_epochs=epoch, train_clip_ids=['p', 'n'], samples=2, train_P=1, train_N=1,
                    optimizer_steps=1, total_updates=epoch, finite=True, train_weighted_bce=.3,
                    initial_state_sha256=resume.INITIAL_SHA, parsed_inputs_sha256=parsed_sha,
                    profile_sha256=digest(d / 'profile.json'))
                if stage == 'inner':
                    z = 2. if epoch == 3 else 1.
                    pred = [dict(clip_id='p', label=1, logit=z), dict(clip_id='n', label=0, logit=-z)]
                    write_json(d / ('validation-epoch-%02d.json' % epoch), pred)
                    row['validation_unweighted_bce'] = math.log1p(math.exp(-z))
                stream.write(json.dumps(row) + '\n')
    d = parent / 'fold-01/dense'; d.mkdir()
    with (d / 'window-scores.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    outer = json.loads((parent / 'fold-01/outer/training-result.json').read_text())
    dense = dict(status='outer_fold_complete', fold=1, selected_epochs=3, plan_sha256=resume.PLAN_SHA,
        checkpoint_sha256=outer['selected_checkpoint']['sha256'], input_fingerprints=frozen['input_fingerprints'],
        threshold=.5, aggregation='max', outer_selection_use=False, independent_test_used=False,
        pixel_reference_sha256=frozen['dense_reference_sha256'], window_scores_sha256=digest(d / 'window-scores.csv'),
        scores=scores, videos=evidence, metrics=video_metrics(scores))
    write_json(d / 'result.json', dense)
    write_json(parent / 'pipeline-failure.json', dict(status='incomplete_five_fold_metrics_withheld', error_type='RuntimeError', error=resume.REFUSAL))
    write_json(parent / 'fold-01/remaining-budget.json', dict(status='refused', seconds=100., remaining_seconds=90.))
    hashes = {r['path']: r['sha256'] for r in resume.artifact_inventory(parent)}
    hashes['source-snapshot/code.py'] = 'a' * 64
    return context, parent, frozen, hashes


class ResumeFixtures(unittest.TestCase):
    def test_complete_first_fold_reuse_and_missing_33_video_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, parent, frozen, hashes = first_fold_fixture(Path(tmp))
            before = resume.artifact_inventory(parent)
            result = resume.validate_first_fold(c, parent, frozen, hashes)
            self.assertEqual(result['inner']['selected_epochs'], 3)
            self.assertNotEqual(digest(parent / 'fold-01/inner/inputs.json'), result['inner']['parsed_inputs_sha256'])
            self.assertEqual(before, resume.artifact_inventory(parent))
            p = parent / 'fold-01/dense/result.json'; d = json.loads(p.read_text()); d['scores'].pop()
            p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                resume.validate_first_fold(c, parent, frozen, hashes)

    def test_changed_sha_missing_artifact_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p / 'a').write_bytes(b'original'); rows = resume.artifact_inventory(p)
            resume.verify_inventory(p, rows)
            (p / 'a').write_bytes(b'altered!')
            with self.assertRaises(ValueError): resume.verify_inventory(p, rows)
            (p / 'a').unlink()
            with self.assertRaises(ValueError): resume.verify_inventory(p, rows)
            for name in ['../x', '/x', 'a\\b', 'a/./b', 'C:/x']:
                with self.assertRaises(ValueError): resume.plain(p, name)

    def test_budget_uses_actual_fees_not_45h_estimate(self):
        start = 10000.; money = 7 - 1.05 - (7200 - 6325) * .1 / 3600 - .1
        end = start + money / 1.48 * 3600; timer = math.floor(end / 60) * 60
        b = resume.continuation_budget(start + 2, start, timer, 1.48, 20000, 6325, 1.05, 3.92)
        self.assertEqual(b['gpu_target_policy'], 'advisory_only_not_a_stop_gate')
        self.assertLessEqual(end - timer, 60)
        for kwargs in [(timer - 900, 1.05, 3.92), (timer, 7., 9.87), (timer, 1.05, 20.)]:
            with self.assertRaises(ValueError):
                resume.continuation_budget(start + 2, start, kwargs[0], 1.48, 2154, 6325, kwargs[1], kwargs[2])

    def test_forecast_overshoot_executes_fresh_2_through_5_and_no_parent_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'child'; out.mkdir(); sources = {}; dense = {}; index = 0
            for fold, positives in enumerate([23, 23, 23, 22, 22], 1):
                scores = []
                for j in range(30):
                    uid = 'u%d' % index; label = int(j < positives); count = 1026 if index == 0 else 956
                    sources[uid] = dict(video_label='pos' if label else 'neg', group_id='g%d' % fold)
                    scores.append(dict(unit_id=uid, label=label, group_id='g%d' % fold, score=.7 if label else .2, windows=count))
                    index += 1
                dense[fold] = dict(fold=fold, scores=scores, metrics=video_metrics(scores))
            c = dict(root=root, sources=sources, folds=[dict(fold=i) for i in range(1, 6)], dense={'windows': {}})
            first = dict(inner={'selected_epochs': 3}, outer={}, dense=dense[1]); calls = []
            def train(ctx, path, fold, stage, inner=None):
                calls.append((fold, stage))
                if stage == 'inner_select': self.assertIsNone(inner)
                else: self.assertEqual(inner, {'selected_epochs': 4, 'from_fold': fold})
                return dict(selected_epochs=4, from_fold=fold)
            def evaluate(ctx, path, fold, outer, E, ffmpeg):
                self.assertEqual(E, 4); return dense[fold]
            summary, complete, origins = resume.run_remaining(c, out, first, None,
                {'deadline_unix': time.time() + 100}, lambda *a: {'seconds': 10**9}, train=train, evaluate=evaluate)
            self.assertEqual(calls, [(i, stage) for i in range(2, 6) for stage in ['inner_select', 'fixed_outer']])
            self.assertFalse((out / 'fold-01').exists()); self.assertEqual(summary['P'], 113); self.assertIsNone(summary['pooled_auc'])
            self.assertEqual([r['reused'] for r in origins], [True, False, False, False, False])
            self.assertTrue(all(json.loads((out / ('fold-%02d' % i) / 'timing-advisory.json').read_text())['forecast_exceeds_remaining'] for i in range(2, 6)))
            with self.assertRaisesRegex(ValueError, 'Actual authorized'):
                resume.run_remaining(c, out, first, None, {'deadline_unix': time.time() - 1}, lambda *a: {}, train=train)


if __name__ == '__main__':
    unittest.main()
