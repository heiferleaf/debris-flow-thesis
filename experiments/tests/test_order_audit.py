"""Finite offline EXP-019 audit fixtures; no model, media or real run reads."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('exp019_audit', ROOT/'experiments/runspecs/EXP-019/audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_stable_probabilities_and_bce(self):
        zs = [-100., -43., -1., 0., 1., 43., 100.]
        ps = [audit.sigmoid(z) for z in zs]
        self.assertTrue(0 < ps[0] < ps[1] < ps[2] < ps[3] < ps[4] <= ps[5] <= ps[6])
        for z, p in zip(zs, ps):
            q = math.exp(-abs(z))
            self.assertEqual(p, 1/(1+q) if z >= 0 else q/(1+q))
            self.assertTrue(math.isfinite(audit.bce(z, 0)))
            self.assertTrue(math.isfinite(audit.bce(z, 1)))
        self.assertAlmostEqual(audit.bce(-100., 1), 100.)
        with self.assertRaises(ValueError):
            audit.bce(float('nan'), 1)

    def test_metrics_ties_single_class_and_gate(self):
        rows = [dict(unit_id=str(i), label=y, score=s) for i, (y, s) in enumerate(
            [(1, .5), (1, .9), (0, .5), (0, .1)])]
        m = audit.metrics(rows)
        self.assertEqual((m['TP'], m['FP'], m['FN'], m['TN']), (2, 1, 0, 1))
        self.assertEqual(m['AUC'], .875)
        self.assertIsNone(audit.metrics(rows[:2])['AUC'])
        def summary(fp, fn, auc=.7, span=.2):
            return dict(clip_metrics=dict(AUC=auc), probability_range=span,
                        video_metrics=dict(FP=fp, FN=fn))
        def gate(a, b): return audit.interpret(dict(zip(audit.MODES, (a, b))))
        self.assertEqual(gate(summary(1, 2), summary(2, 2))['classification'], 'positive_development_signal')
        self.assertEqual(gate(summary(1, 2), summary(1, 2))['classification'], 'same_error_counts_scoring_comparison_only')
        self.assertEqual(gate(summary(1, 2, .5), summary(2, 2))['classification'], 'optimization_stability_only')
        self.assertEqual(gate(summary(2, 2), summary(1, 2))['classification'], 'permuted_has_fewer_errors_no_automatic_promotion')
        self.assertEqual(gate(summary(0, 3), summary(1, 2))['classification'], 'mixed_error_tradeoff')
        self.assertFalse(gate(summary(1, 2), summary(2, 2))['another_run_automatically_authorized'])

    def test_validation_identity_labels_and_earliest_recorded_selection(self):
        ids = ['p', 'n']
        truth = {'p': 1, 'n': 0}
        preds = [dict(clip_id='p', label=1, logit=1.), dict(clip_id='n', label=0, logit=-1.)]
        loss, _, _ = audit.validation_summary(preds, ids, truth)
        self.assertAlmostEqual(loss, math.log1p(math.exp(-1)))
        self.assertEqual(audit.selected_epoch([dict(validation_unweighted_bce=x) for x in (2., 1., 1.)]), 2)
        for changed in (preds[::-1], preds+[preds[0]], [dict(preds[0], label=0), preds[1]],
                        [dict(preds[0], logit=float('inf')), preds[1]]):
            with self.assertRaises(ValueError):
                audit.validation_summary(changed, ids, truth)
        audit.near(loss+1e-7, loss, 'FP32 reduction tolerance', audit.BCE_ATOL)
        with self.assertRaises(ValueError):
            audit.near(loss+.001, loss, 'changed loss', audit.BCE_ATOL)

    def test_order_mapping_and_dense_exact_identity(self):
        def window(uid, offset):
            frames = list(range(offset, offset+16))
            record = audit.window_order_record('a'*64, frames)
            return dict(unit_id=uid, media_sha256='a'*64, frame_indices=frames,
                        source_pts_seconds=[i*.2 for i in frames], order_record=record)
        ws = [window('p', 0), window('p', 16), window('n', 0)]
        orders = [dict(unit_id=w['unit_id'], frame_indices=w['frame_indices'],
                       source_pts_seconds=w['source_pts_seconds'], **w['order_record']) for w in ws]
        rows = [dict(unit_id=w['unit_id'], start=str(w['source_pts_seconds'][0]),
                     end=str(w['source_pts_seconds'][-1]), physical_window_sha256=w['order_record']['physical_window_sha256'],
                     chronological=str(a), fixed_permutation=str(b)) for w, a, b in zip(ws, (.2, .8, .3), (.4, .7, .6))]
        # A physical key may appear in two files, but each (unit, key) occurs once.
        maxima, counts = audit.dense_windows(rows, orders, ws)
        self.assertEqual(counts, {'p': 2, 'n': 1})
        self.assertEqual(maxima['chronological'], {'p': .8, 'n': .3})
        for badrows, badorders in ((rows[:-1], orders), (rows+[rows[0]], orders+[orders[0]]),
                                   (rows, orders[::-1])):
            with self.assertRaises(ValueError): audit.dense_windows(badrows, badorders, ws)
        broken = copy.deepcopy(orders)
        broken[0]['permutation'] = list(range(16))
        with self.assertRaises(ValueError): audit.dense_windows(rows, broken, ws)
        altered = copy.deepcopy(rows)
        altered[0]['end'] = '99'
        with self.assertRaises(ValueError): audit.dense_windows(altered, orders, ws)
        mappings = [dict(clip_id='clip', order_mode='fixed_permutation', **ws[0]['order_record'])]
        audit.supervised_mapping(mappings, {'clip': ws[0]}, 'fixed_permutation')
        with self.assertRaises(ValueError): audit.supervised_mapping(mappings, {'clip': ws[0]}, 'chronological')

    def test_rng_closed_and_corruption_rejected(self):
        def state(c): return dict(cpu=c*64, cuda=[c*64])
        history = [dict(rng_before=state('a'), rng_after=state('b')),
                   dict(rng_before=state('b'), rng_after=state('c'))]
        audit.rng_contract(history, copy.deepcopy(history))
        bad = copy.deepcopy(history); bad[1]['rng_before'] = state('d')
        with self.assertRaises(ValueError): audit.rng_contract(history, bad)
        bad = copy.deepcopy(history); bad[0]['rng_after'] = dict(cpu='bad', cuda=[])
        with self.assertRaises(ValueError): audit.rng_contract(bad, bad)

    def test_digest_closure_and_frozen_truth(self):
        sources = {'p': dict(video_label='pos', group_id='g1'), 'n': dict(video_label='neg', group_id='g2')}
        maxima = {m: {'p': .8, 'n': .2} for m in audit.MODES}
        counts = {'p': 1, 'n': 1}
        dense = dict(videos={u: dict(windows=1, accepted_windows=1, input_order_windows=1,
            canonical_reference_matched=True, rgb112_digest=u*64, maxima={m: maxima[m][u] for m in audit.MODES}) for u in sources},
            scores={m: [dict(unit_id=u, label=int(sources[u]['video_label']=='pos'), group_id=sources[u]['group_id'],
                            score=maxima[m][u], windows=1) for u in sources] for m in audit.MODES})
        ref = dict(videos={u: dict(windows=1, rgb112_digest=u*64) for u in sources})
        audit.video_closure(dense, ref, sources, maxima, counts)
        for change in ('truth', 'digest', 'max'):
            broken = copy.deepcopy(dense)
            if change == 'truth': broken['scores']['chronological'][1]['label'] = 1
            if change == 'digest': broken['videos']['p']['rgb112_digest'] = 'changed'
            if change == 'max': broken['videos']['p']['maxima']['chronological'] = .9
            with self.assertRaises(ValueError): audit.video_closure(broken, ref, sources, maxima, counts)

    def test_label_free_reference_compatible_historical_plan(self):
        # The pixel-only reference predates training-plan-v2; compare identical
        # data sources, never require equal governance/assignment-plan SHA.
        original = dict(resource='canonical', sources=[dict(path='frames', sha256='abc')])
        base = dict(original, version=2)
        reference = dict(status='dense_pixels_verified', plan_sha256='old')
        audit.pixel_reference_contract(reference, base, 'new', original, 'old')
        for bad in (dict(original, resource='different'), dict(original, sources=[])):
            with self.assertRaises(ValueError):
                audit.pixel_reference_contract(reference, base, 'new', bad, 'old')
        with self.assertRaises(ValueError):
            audit.pixel_reference_contract(dict(reference, status='dense_evaluation_complete'), base, 'new', original, 'old')

    def test_missing_run_is_incomplete_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root/'data/results/exp019/run'; run.mkdir(parents=True)
            out = run/'audit-v1'
            result = audit.audit(run, out, root=root, argv=['fixture'])
            self.assertEqual(result['status'], 'incomplete')
            self.assertFalse(result['metrics_accepted'])
            self.assertIn('pipeline-result.json', ' '.join(result['missing_inputs']))
            self.assertTrue((out/'generator-source.py').is_file())
            with self.assertRaises(FileExistsError): audit.audit(run, out, root=root)
            malformed = run/'bad.json'; malformed.write_text('{"x": 1, "x": 2}', encoding='utf-8')
            with self.assertRaises(ValueError): audit.Evidence(root).read(malformed)
            malformed.write_text('{"x": NaN}', encoding='utf-8')
            with self.assertRaises(ValueError): audit.Evidence(root).read(malformed)
            (run/'pipeline-failure.json').write_text('{}', encoding='utf-8')
            failed = audit.audit(run, run/'audit-failed', root=root)
            self.assertEqual(failed['status'], 'failed')
            self.assertIsNone(failed['registered_gate'])

    def test_cli_incomplete_nonzero_and_sha_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'small.txt'; path.write_text('before', encoding='utf-8')
            evidence = audit.Evidence(root); evidence.bind(path)
            path.write_text('after', encoding='utf-8')
            with self.assertRaises(ValueError): evidence.close()
            # Exercise main's actual status->exit mapping with only temporary files.
            original_root = audit.ROOT
            try:
                audit.ROOT = root
                run = root/'data/results/exp019/run'; run.mkdir(parents=True)
                # audit's default root is bound at definition; inject only that
                # path through a wrapper, retaining its real output logic.
                original_audit = audit.audit
                def fixture_run(*args, **kwargs):
                    kwargs['root'] = root
                    return original_audit(*args, **kwargs)
                audit.audit = fixture_run
                result = audit.main(['--run', str(run), '--out', str(run/'audit-v1')])
                self.assertEqual(result, 2)
                for i, name in enumerate(('hard-time-limit.json', 'hard-stop-shutdown.json', 'hard-stop-shutdown.log')):
                    receipt = run/name; receipt.write_text('{}', encoding='utf-8')
                    output = run/('audit-hard-stop-%d'%i)
                    self.assertEqual(audit.main(['--run', str(run), '--out', str(output)]), 1)
                    report = json.loads((output/'audit.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['status'], 'failed')
                    self.assertFalse(report['metrics_accepted'])
                    receipt.unlink()
            finally:
                audit.ROOT = original_root
                audit.audit = original_audit


if __name__ == '__main__': unittest.main()
