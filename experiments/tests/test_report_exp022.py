"""Small EXP022 reporting fixtures: identities, ties, group gates and closure."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('report_exp022_fixture', ROOT / 'experiments/runspecs/EXP-022/report.py')
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def video(uid, group, label, score, fold=1):
    return dict(unit_id=uid, group_id=group, label=label, score=f32(score), fold=fold, windows=1)


class ReportTests(unittest.TestCase):
    def test_macro_is_not_micro_and_requires_two_improved_groups(self):
        # One small group and one nine-video group carry equal macro weight.
        deep = [video('p', 'g0', 1, .8), video('small', 'g1', 0, .8)]
        deep += [video('n%d' % i, 'g2', 0, .8) for i in range(9)]
        linear = deepcopy(deep); linear[1]['score'] = f32(.2)
        result, _ = report.paired_comparison(deep, linear)
        self.assertEqual(result['linear']['FPR'], .9)
        self.assertEqual(result['linear_macro_negative_group_FPR'], .5)
        self.assertFalse(result['supports_overall_head_simplification_video_improvement'])
        self.assertFalse(result['gate_components']['at_least_two_negative_groups_improved'])
        linear[2]['score'] = f32(.2)
        result, _ = report.paired_comparison(deep, linear)
        self.assertTrue(result['supports_overall_head_simplification_video_improvement'])
        linear[0]['score'] = f32(.2)
        result, _ = report.paired_comparison(deep, linear)
        self.assertFalse(result['supports_overall_head_simplification_video_improvement'])
        self.assertEqual(result['regression_cases'][0]['change'], 'new_false_negative')
        self.assertEqual(result['regression_cases'][0]['unit_id'], 'p')

    def test_pairing_rejects_duplicate_missing_and_changed_identity(self):
        deep = [video('p', 'g0', 1, .8), video('n', 'g1', 0, .8)]
        for field, value in [('label', 1), ('group_id', 'wrong'), ('fold', 2), ('windows', 2)]:
            with self.subTest(field=field):
                linear = deepcopy(deep); linear[1][field] = value
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    report.paired_comparison(deep, linear)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.paired_comparison(deep, deep + [deep[0]])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            report.paired_comparison(deep, deep[:1])

    def test_saved_probability_ties_and_threshold_are_distinct(self):
        rows = [video('p', 'g0', 1, .5), video('n', 'g1', 0, .5)]
        scores = report.metrics(rows)
        self.assertEqual(scores['AUC'], .5)
        self.assertEqual((scores['TP'], scores['FP']), (1, 1))
        # A genuine FP32 neighbour stays distinct; there is no epsilon tie rule.
        rows[0]['score'] = f32(.5 + 2 ** -24)
        self.assertEqual(report.metrics(rows)['AUC'], 1.)
        self.assertNotIn('AUC', report.metrics(rows, auc=False))

    def test_window_max_and_paired_physical_identity(self):
        with tempfile.TemporaryDirectory(prefix='report-exp022-') as temp:
            path = Path(temp) / 'window-scores.csv'
            row = video('p', 'g', 1, .8)
            info = dict(score=row['score'], windows=1, canonical_reference_matched=True, rgb112_digest='x')
            result = dict(scores=[row], videos={'p': info}, metrics=report.metrics([row]))
            reference = dict(videos={'p': dict(windows=1, rgb112_digest='x')})
            def save(value):
                with path.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['unit_id', 'frame_tuple_sha256', 'score', 'target_start_seconds', 'start', 'end'])
                    writer.writeheader(); writer.writerow(dict(unit_id='p', frame_tuple_sha256='a' * 64,
                        score=value, target_start_seconds=0, start=0, end=3))
                result['window_scores_sha256'] = report.sha(path)
            save(row['score'])
            keys, stats = report.window_review(report.Inputs(), path, result, reference)
            self.assertEqual(stats['n'], 1)
            with self.assertRaisesRegex(ValueError, 'physical window identity'):
                report.window_review(report.Inputs(), path, result, reference, {('p', 'b' * 64): (0, 0, 3)})
            save(f32(.7))
            with self.assertRaisesRegex(ValueError, 'closure differs'):
                report.window_review(report.Inputs(), path, result, reference)

    def test_incomplete_run_withholds_full_report_and_creates_no_output(self):
        with tempfile.TemporaryDirectory(prefix='report-exp022-') as temp:
            root = Path(temp); run = root / 'run'; run.mkdir(); out = root / 'derived'
            with self.assertRaises(report.Incomplete):
                report.generate(root, root / 'plan', run, out)
            self.assertFalse(out.exists())
            (run / 'closed-run.json').write_text(json.dumps(dict(status='run_failed_closed', files=[])), encoding='utf-8')
            with self.assertRaisesRegex(report.Incomplete, 'not fully closed'):
                report.generate(root, root / 'plan', run, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
