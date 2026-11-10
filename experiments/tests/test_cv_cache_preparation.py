"""Small synthetic JPEG/array/receipt fixtures; never decode real media."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from experiments.common.video import cv_cache_preparation as prep
from experiments.common.video.pts_clip_sampling import jpeg95_bytes, rgb112_from_jpeg
from experiments.common.video.order_cache_acceptance import verify_required_rows


class CVCacheTests(unittest.TestCase):
    def test_requirement_difference_and_immutable_old_index(self):
        required = [dict(unit_id='u', frame_index=i, source_pts_seconds=i*.1, frozen_qc_std=30.) for i in (1,2)]
        old = [dict(required[0], gray_std=30.01, jpeg_sha256='a'*64, rgb112_sha256='b'*64, source_bgr_sha256='c'*64)]
        before = copy.deepcopy(old)
        needed, reused, missing, original = prep.split_requirements(required, old)
        self.assertEqual(set(needed), {('u',1),('u',2)})
        self.assertEqual(set(reused), {('u',1)})
        self.assertEqual(set(missing), {('u',2)})
        self.assertEqual(old, before)
        for bad in (old+old, [dict(old[0], source_pts_seconds=99.)], [dict(old[0], gray_std=19.99)],
                    [dict(old[0], frozen_qc_std=30.1)]):
            with self.assertRaises(ValueError): prep.split_requirements(required, bad)

    def test_complete_receipt_resume_and_partial_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); artifact=root/'pixels.jpg'; artifact.write_bytes(b'immutable')
            receipt=root/'complete.json'; contract='a'*64
            self.assertIsNone(prep.read_seal(root, receipt, contract, 'u'))
            # Presence of partial files does not create a completion receipt.
            prep.seal(root, receipt, contract, 'u', dict(frames=1), [artifact])
            self.assertEqual(prep.read_seal(root, receipt, contract, 'u')['payload'], dict(frames=1))
            with self.assertRaises(ValueError): prep.read_seal(root, receipt, 'b'*64, 'u')
            with self.assertRaises(ValueError): prep.read_seal(root, receipt, contract, 'different')
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError): prep.read_seal(root, receipt, contract, 'u')

    def test_output_contract_exclusive_and_resume_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); c=dict(root=root); out=root/'data/process/EXP-017/fixture'
            frozen=dict(source_sha256={}, plan_sha256='a'*64)
            prep.open_output(c, out, frozen, False)
            with self.assertRaises(ValueError): prep.open_output(c, out, frozen, False)
            prep.open_output(c, out, frozen, True)
            with self.assertRaises(ValueError): prep.open_output(c, out, dict(frozen, plan_sha256='b'*64), True)
            with self.assertRaises(ValueError): prep.open_output(c, root/'data/raw/forbidden', frozen, False)

    def test_archived_source_and_initial_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); c=dict(root=root); code=root/'helper.py'; code.write_bytes(b'frozen')
            out=root/'data/process/EXP-017/fixture'
            frozen=dict(source_sha256={'helper.py':hashlib.sha256(code.read_bytes()).hexdigest()})
            prep.open_output(c,out,frozen,False)
            (out/'source-snapshot/helper.py').write_bytes(b'changed')
            with self.assertRaises(ValueError): prep.open_output(c,out,frozen,True)
            failed=root/'data/metadata/EXP-017/failed-fixture'
            prep.record_cli_failure(root,failed,'fixture',ValueError('bad input'))
            failures=list(failed.rglob('failure.json')); self.assertEqual(len(failures),1)
            self.assertEqual(json.loads(failures[0].read_text('utf-8'))['status'],'failed_no_new_readiness')
            self.assertFalse((failed/'acceptance.json').exists())

    def test_actual_jpeg_and_rgb112_acceptance_not_index_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); rng=np.random.default_rng(3); requirements=[]
            for i in range(2):
                bgr=rng.integers(0,256,(256,256,3),dtype=np.uint8); encoded=jpeg95_bytes(bgr)
                path=root/('frame%d.jpg'%i); path.write_bytes(encoded)
                requirements.append(dict(unit_id='u', frame_index=i, source_pts_seconds=i*.1, cache_path=path.name,
                    size_bytes=len(encoded), jpeg_sha256=hashlib.sha256(encoded).hexdigest(),
                    rgb112_sha256=hashlib.sha256(rgb112_from_jpeg(encoded).tobytes()).hexdigest()))
            result=verify_required_rows(root, requirements, root/'verified.csv', root/'failed.jsonl', workers=1)
            self.assertEqual(result['verified'],2); self.assertEqual(result['failed'],0)
            before=(root/'frame0.jpg').read_bytes()
            broken=copy.deepcopy(requirements); broken[0]['rgb112_sha256']='0'*64
            with self.assertRaises(ValueError):
                verify_required_rows(root, broken, root/'wrong-rgb.csv', root/'wrong-rgb.jsonl', workers=1)
            self.assertTrue((root/'wrong-rgb.jsonl').is_file())
            self.assertEqual((root/'frame0.jpg').read_bytes(),before)
            (root/'frame1.jpg').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                verify_required_rows(root, requirements, root/'wrong-jpeg.csv', root/'wrong-jpeg.jsonl', workers=1)

    def test_dense_pixels_identity_overlap_and_final_closure(self):
        windows=[dict(unit_id='u',target_start_seconds=float(start)/10,
                      frame_indices=list(range(start,start+16)), source_pts_seconds=[i/10 for i in range(start,start+16)])
                 for start in (0,8)]
        pixels={i:np.full((112,112,3),i,np.uint8) for i in range(24)}
        batch=np.stack([np.stack([pixels[i] for i in w['frame_indices']]) for w in windows])
        info=dict(windows=2,rgb112_digest='d'*64,minimum_selected_gray_std=25.)
        result=prep.consume_dense(iter([(batch,windows)]),info,windows)
        self.assertEqual(len(result),24)
        for array,meta,details in ((batch[:1],windows[:1],info), (batch,windows,dict(info,windows=1)),
                                   (batch.astype(np.float32),windows,info), (batch,windows,dict(info,minimum_selected_gray_std=19.9))):
            with self.assertRaises(ValueError): prep.consume_dense(iter([(array,meta)]),details,windows)
        broken=batch.copy(); broken[1,0,0,0,0] ^= 1
        with self.assertRaises(ValueError): prep.consume_dense(iter([(broken,windows)]),info,windows)
        changed=copy.deepcopy(windows); changed[0]['source_pts_seconds'][0]=999.
        with self.assertRaises(ValueError): prep.consume_dense(iter([(batch,changed)]),info,windows)

    def test_numeric_comparison_preserves_old_version(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new=Path(temp)/'v2',Path(temp)/'v3';old.mkdir();new.mkdir()
            for name in prep.NUMERIC_FILES:
                (old/name).write_bytes(b'identical');(new/name).write_bytes(b'identical')
            prep.compare_numeric_plans(old,new)
            (new/'folds.json').write_bytes(b'changed')
            with self.assertRaises(ValueError): prep.compare_numeric_plans(old,new)
            self.assertEqual((old/'folds.json').read_bytes(),b'identical')


if __name__=='__main__': unittest.main()
