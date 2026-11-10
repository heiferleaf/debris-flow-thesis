"""Finite existing-JPEG fixtures; no source-video decoder or trained model."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video import order_cache_acceptance as acceptance
from experiments.common.video.pts_clip_sampling import jpeg95_bytes, rgb112_from_jpeg
from experiments.common.video.supervised_plan import digest


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fixture(folder):
    root = Path(folder); plan_dir = root / 'data/metadata/EXP-019/order-plan-fixture'
    plan_dir.mkdir(parents=True)
    rng = np.random.default_rng(19)
    refs = []
    for i in range(16):
        bgr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        encoded = jpeg95_bytes(bgr); relative = 'cache/frame-%02d.jpg' % i
        (root / relative).parent.mkdir(exist_ok=True); (root / relative).write_bytes(encoded)
        refs.append(dict(unit_id='u', frame_index=i, source_pts_seconds=i / 5,
            cache_path=relative, size_bytes=len(encoded), jpeg_sha256=hashlib.sha256(encoded).hexdigest(),
            rgb112_sha256=hashlib.sha256(rgb112_from_jpeg(encoded).tobytes()).hexdigest(),
            source_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
    write_rows(root / acceptance.INDEX, refs)
    write_rows(plan_dir / 'supervised-cache-requirements.csv', refs)
    bank = {'c': dict(unit_id='u', frame_indices=list(range(16)), source_pts_seconds=[i / 5 for i in range(16)])}
    copied = ['experiments/common/video/order_cache_acceptance.py', 'experiments/runspecs/EXP-019/verify_cache.py',
              'experiments/common/video/pts_clip_sampling.py', 'experiments/common/video/order_plan.py',
              'experiments/common/video/temporal_order.py', 'experiments/common/video/supervised_plan.py']
    for name in copied:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT / name).read_bytes())
    helper = 'experiments/common/video/pts_clip_sampling.py'
    plan = dict(data_fingerprints=[dict(path=acceptance.INDEX, sha256=digest(root / acceptance.INDEX)),
                                  dict(path=helper, sha256=digest(root / helper))],
        current_source_fingerprints=[], files={'supervised-cache-requirements.csv': digest(plan_dir / 'supervised-cache-requirements.csv')},
        pixels_ready=False, training_ready=False, cache_acceptance=None)
    (plan_dir / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    return root, plan_dir, plan, bank, refs


class CacheAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = ROOT / '.codex/docs/.temp/order-cache-fixtures'; cls.temp.mkdir(parents=True, exist_ok=True)

    def test_actual_byte_RGB112_and_dimension_checks(self):
        with tempfile.TemporaryDirectory(dir=self.temp) as folder:
            root, _, _, _, refs = fixture(folder)
            ref = {k:v for k,v in refs[0].items() if k != 'source_bgr_sha256'}
            row = acceptance.verify_jpeg(root, ref)
            self.assertEqual(row['actual_jpeg_sha256'], ref['jpeg_sha256'])
            self.assertEqual(row['actual_rgb112_sha256'], ref['rgb112_sha256'])
            wrong = dict(ref, rgb112_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'RGB112 SHA'): acceptance.verify_jpeg(root, wrong)
            with self.assertRaises(ValueError): acceptance.verify_jpeg(root, dict(ref, cache_path='../escape.jpg'))
            path = root / ref['cache_path']; data = bytearray(path.read_bytes()); data[-10] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'JPEG SHA'): acceptance.verify_jpeg(root, ref)
            small = jpeg95_bytes(np.zeros((128, 128, 3), dtype=np.uint8)); path.write_bytes(small)
            wrong = dict(ref, size_bytes=len(small), jpeg_sha256=hashlib.sha256(small).hexdigest())
            with self.assertRaises(ValueError): acceptance.verify_jpeg(root, wrong)

    def test_original_index_and_canonical_requirement_coverage_are_strict(self):
        with tempfile.TemporaryDirectory(dir=self.temp) as folder:
            root, plan_dir, plan, bank, refs = fixture(folder)
            result, expected = acceptance.checked_requirements(root, plan_dir, plan, bank)
            self.assertEqual(len(result), 16); self.assertEqual(expected, digest(root / acceptance.INDEX))
            duplicated = deepcopy(bank); duplicated['alias'] = deepcopy(bank['c'])
            self.assertEqual(len(acceptance.required_physical_frames(duplicated)), 16)
            duplicated['alias']['source_pts_seconds'][0] = -.01
            with self.assertRaises(ValueError): acceptance.required_physical_frames(duplicated)
            write_rows(root / acceptance.INDEX, refs[:-1])
            broken = deepcopy(plan); broken['data_fingerprints'][0]['sha256'] = digest(root / acceptance.INDEX)
            with self.assertRaises(ValueError): acceptance.checked_requirements(root, plan_dir, broken, bank)

    def test_threaded_index_order_and_failure_partial_evidence(self):
        with tempfile.TemporaryDirectory(dir=self.temp) as folder:
            root, plan_dir, plan, bank, _ = fixture(folder)
            refs, _ = acceptance.checked_requirements(root, plan_dir, plan, bank)
            a = acceptance.verify_required_rows(root, refs, root/'one.csv', root/'fail-one.jsonl', workers=1)
            b = acceptance.verify_required_rows(root, refs, root/'four.csv', root/'fail-four.jsonl', workers=4)
            self.assertEqual(a, b); self.assertEqual(a['verified'], 16)
            self.assertEqual((root/'one.csv').read_bytes(), (root/'four.csv').read_bytes())
            (root / refs[0]['cache_path']).unlink()
            with self.assertRaises(FileNotFoundError):
                acceptance.verify_required_rows(root, refs, root/'partial.csv', root/'failure.jsonl', workers=2)
            self.assertTrue((root/'failure.jsonl').is_file())
            self.assertIn('verified_before_failure', (root/'failure.jsonl').read_text(encoding='utf-8'))

    def test_complete_proof_matches_existing_training_consumer_and_never_promotes_plan(self):
        from experiments.common.video.order_training import verify_cache_acceptance
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=self.temp) as folder:
                root, plan_dir, plan, bank, refs = fixture(folder)
                out = root/'data/metadata/EXP-019/acceptance-fixture'
                original_plan, original_index = (plan_dir/'plan.json').read_bytes(), (root/acceptance.INDEX).read_bytes()
                if fail: (root / refs[0]['cache_path']).unlink()
                parsed = (plan, bank, [], [], [], {})
                with patch.object(acceptance, 'load_order_plan', return_value=parsed), \
                     patch.object(acceptance, '__file__', str(root/'experiments/common/video/order_cache_acceptance.py')):
                    if fail:
                        with self.assertRaises(RuntimeError): acceptance.accept_order_cache(root, plan_dir, out, workers=2)
                    else:
                        result = acceptance.accept_order_cache(root, plan_dir, out, workers=2)
                proof = json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
                self.assertEqual((plan_dir/'plan.json').read_bytes(), original_plan)
                self.assertEqual((root/acceptance.INDEX).read_bytes(), original_index)
                if fail:
                    self.assertEqual(proof['status'], 'order_supervised_cache_failed')
                    self.assertEqual(proof['failed_required_frames'], 1)
                    with self.assertRaises(ValueError): verify_cache_acceptance(root, plan_dir, plan, bank, out/'acceptance.json')
                else:
                    self.assertEqual(result, proof); self.assertEqual(proof['status'], 'order_supervised_cache_ready')
                    self.assertEqual(proof['verified_required_frame_count'], 16)
                    self.assertEqual(proof['failed_required_frames'], 0)
                    verify_cache_acceptance(root, plan_dir, plan, bank, out/'acceptance.json')
                    self.assertEqual(proof['verified_required_frames_index_sha256'], digest(out/'verified-required-frames.csv'))
                with self.assertRaises(FileExistsError): acceptance.accept_order_cache(root, plan_dir, out)


if __name__ == '__main__': unittest.main()
