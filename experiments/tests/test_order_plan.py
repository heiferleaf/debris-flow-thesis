"""Small metadata-only checks for the frozen A template and provenance separation."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.order_plan import (
    import_epochs, validate_history_order, verify_snapshot_file, _check_window,
)


def fixture():
    bank = {}
    for i in range(504):
        kind = 'pos' if i < 356 else 'neg'
        local = i if kind == 'pos' else i - 356
        bank[str(i)] = dict(kind=kind, split='train', unit_id=kind + str(local // 4), segment_id='s')
    order = list(reversed(list(bank)))
    rows = [dict(epoch=e, batches=[dict(clip_ids=order[i:i + 24]) for i in range(0, 504, 24)]) for e in range(20)]
    original = {e: list(bank) for e in range(20)}
    return bank, rows, original


class OrderPlanTests(unittest.TestCase):
    def test_A_batch_order_is_imported_without_resampling_or_reshuffling(self):
        bank, rows, original = fixture()
        result = import_epochs(rows, original, bank)
        self.assertEqual(result[0]['clip_ids'], list(reversed(list(bank))))
        self.assertEqual(result[19]['batches'], rows[19]['batches'])
        self.assertTrue(all(len(r['clip_ids']) == 504 and len(r['batches']) == 21 for r in result))

    def test_batch_labels_membership_quota_and_full_epoch_coverage_refused(self):
        bank, rows, original = fixture()
        for mutation in ('duplicates', 'missing_batch', 'role', 'class', 'quota', 'epoch'):
            b, r = deepcopy(bank), deepcopy(rows)
            if mutation == 'duplicates': r[0]['batches'][0]['clip_ids'][0] = r[0]['batches'][0]['clip_ids'][1]
            elif mutation == 'missing_batch': r[0]['batches'].pop()
            elif mutation == 'role': b['0']['split'] = 'validation'
            elif mutation == 'class': b['0']['kind'] = 'neg'
            elif mutation == 'quota': b['4']['unit_id'] = b['0']['unit_id']
            else: r[-1]['epoch'] = 18
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): import_epochs(r, original, b)

    def test_historical_actual_train_order_and_update_contract(self):
        bank, rows, original = fixture(); imported = import_epochs(rows, original, bank)
        history = [dict(completed_epochs=i+1, samples=504, optimizer_steps=21, total_updates=21*(i+1),
            train_clip_ids=r['clip_ids'][:], objective_mode='bce', rank_lambda=0.) for i,r in enumerate(imported)]
        validate_history_order(history, imported)
        bad = deepcopy(history); bad[1]['train_clip_ids'][0:2] = reversed(bad[1]['train_clip_ids'][0:2])
        with self.assertRaises(ValueError): validate_history_order(bad, imported)
        bad = deepcopy(history); bad[-1]['total_updates'] = 419
        with self.assertRaises(ValueError): validate_history_order(bad, imported)

    def test_old_snapshot_and_current_governance_are_distinct_guards(self):
        folder = ROOT / '.codex/docs/.temp/order-plan-fixtures'; folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            snapshot = Path(name) / 'old-card.md'; live = Path(name) / 'current-card.md'
            snapshot.write_bytes(b'old authorized scope'); live.write_bytes(b'current authorized scope')
            old_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            verify_snapshot_file(snapshot, old_sha)
            with self.assertRaises(ValueError): verify_snapshot_file(live, old_sha)
            snapshot.write_bytes(b'tampered archive')
            with self.assertRaises(ValueError): verify_snapshot_file(snapshot, old_sha)

    def test_dense_QC_is_label_independent_canonical_PTS_remains_unchanged(self):
        pts = np.linspace(0, 3, 16); std = np.full(16, 30.)
        window = dict(unit_id='u', frame_indices=list(range(16)), source_pts_seconds=pts.tolist())
        video = dict(split='validation', group_id='g', clean_runs_json='[[0,16]]')
        original = deepcopy(window); _check_window(window, video, pts, std, supervised=False)
        self.assertEqual(window, original)
        bad = std.copy(); bad[8] = 19.9
        with self.assertRaises(ValueError): _check_window(window, video, pts, bad, supervised=False)
        with self.assertRaises(ValueError): _check_window(window, video, pts+.01, std, supervised=False)
        with self.assertRaises(ValueError): _check_window(window, {**video,'clean_runs_json':'[[0,8],[8,16]]'}, pts, std, supervised=False)


if __name__ == '__main__': unittest.main()
