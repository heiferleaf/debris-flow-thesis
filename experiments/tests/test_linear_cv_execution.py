"""Three targeted CPU contracts; no real model, pixels, decoding or fitting."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video import linear_cv_execution as execution


def completed(stage, fold, E=6):
    return dict(status='fit_complete', stage=stage, fold=fold, completed_epochs=20 if stage == 'inner_select' else E,
        initial_state_sha256=execution.INITIAL_SHA, fixed_outer_validation_evaluated=False, selected_epochs=E,
        selected_checkpoint=dict(filename='best.pt' if stage == 'inner_select' else 'selected.pt'))


class LinearCVContracts(unittest.TestCase):
    def test_frozen_epoch_and_exact_stage_order_shell(self):
        self.assertEqual(execution.epoch_block(ROOT / 'experiments/common/video/linear_cv_training.py'),
                         execution.epoch_block(ROOT / 'experiments/common/video/cv_supervised_training.py'))
        spec = importlib.util.spec_from_file_location('linear_pipeline', ROOT / 'experiments/runspecs/EXP-022/pipeline.py')
        pipeline = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)
        self.assertEqual(pipeline.stage_order(), ['f01-outer','f01-dense','f02-inner','f02-outer','f02-dense',
            'f03-inner','f03-outer','f03-dense','f04-outer','f04-dense','f05-inner','f05-outer','f05-dense'])
        shell = (ROOT / 'experiments/runspecs/EXP-022/run.sh').read_bytes()
        self.assertNotIn(b'\r', shell)
        self.assertFalse(shell.startswith(b'\xef\xbb\xbf'))
        self.assertIn(b'/root/miniconda3/bin/python -B experiments/runspecs/EXP-022/pipeline.py', shell)
        self.assertIn(b'/usr/bin/timeout --kill-after=2s 15s', shell)

    def test_outer_only_own_E_full_schedule_prefix_and_no_validation(self):
        outer = [['outer-E%d' % e] for e in range(20)]
        inner = [['inner-E%d' % e] for e in range(20)]
        fold = dict(inner_train=dict(epoch_clip_ids=inner), inner_val=dict(clip_ids=['fixed-val']),
                    outer_train=dict(epoch_clip_ids=outer))
        context = dict(root=ROOT, folds=[fold]*5, bank={}, fingerprints={'frozen':'sha'}, plan=dict(new_inner_folds=[2,3,5]))
        calls = []
        def fake_fit(root, out, **kw):
            calls.append(kw)
            return completed(kw['stage'], kw['fold'], kw.get('selected_epochs', 6))
        with patch.object(execution, 'frame_refs', return_value={}), patch.object(execution, 'fit', fake_fit), \
             patch.object(execution, 'read', return_value=dict(head='linear', execution_kind='registered_fit')):
            execution.train_stage(context, Path('unused'), 2, 'fixed_outer', completed('inner_select',2,6))
            self.assertEqual(calls[0]['epoch_clip_ids'], outer)  # Full20 passed; core takes prefix[:E].
            self.assertEqual(calls[0]['selected_epochs'], 6)
            self.assertNotIn('validation_clip_ids', calls[0])
            self.assertNotIn('model_factory', calls[0])
            self.assertNotIn('state_dict', calls[0])
            self.assertEqual(calls[0]['expected_initial_state_sha256'], execution.INITIAL_SHA)
            execution.train_stage(context, Path('unused'), 2, 'inner_select')
            self.assertEqual(calls[1]['validation_clip_ids'], ['fixed-val'])
            with self.assertRaises(ValueError):
                execution.train_stage(context, Path('unused'), 1, 'inner_select')
            with self.assertRaises(ValueError):
                execution.train_stage(context, Path('unused'), 2, 'fixed_outer', completed('inner_select',3,6))

    def test_reuse_wrong_role_label_schedule_or_initialization_rejected(self):
        train, val = ['p','n'], ['v']
        schedule = [list(train) for _ in range(20)]
        def row(uid,label):
            return dict(unit_id=uid, group_id=uid, label=label, frame_indices=list(range(16)), source_pts_seconds=list(range(16)))
        bank = dict(p=row('trainP',1), n=row('trainN',0), v=row('val',1))
        inputs = dict(epoch_clip_ids=schedule, validation_clip_ids=val, clip_bank=copy.deepcopy(bank))
        fold = dict(fold=1,inner_train=dict(epoch_clip_ids=schedule),inner_val=dict(clip_ids=val))
        profile = dict(head='linear',head_parameters=513,seed=7,stage='inner_select',fold=1,
            initial_state_sha256=execution.INITIAL_SHA,backbone_state_sha256=execution.BACKBONE_SHA,
            model_class='experiments.common.video.head_comparison_training.LinearVideoClassifier',mode='rgb_zero',
            dataloader_generator_seed=100007, effective_batch=24,microbatch=8,
            environment=dict(precision='FP32',amp=False),optimizer=dict(name='Adam',lr_backbone=1e-4,lr_head=1e-3,
                betas=[.9,.999],eps=1e-8,weight_decay=0,initial_state_entries=0))
        history = [dict(completed_epochs=e,train_clip_ids=train,validation_clip_ids=val,optimizer_steps=1,
                        validation_unweighted_bce=0 if e==6 else 1) for e in range(1,21)]
        result = completed('inner_select',1,6)
        execution.validate_reuse(profile,result,inputs,history,fold,bank,6)
        for damage in ('role','label','schedule','initial','nan'):
            p,r,x,h = map(copy.deepcopy,(profile,result,inputs,history))
            if damage=='role':p['stage']='fixed_outer'
            elif damage=='label':x['clip_bank']['n']['label']=1
            elif damage=='schedule':x['epoch_clip_ids'][0]=['n','p']
            elif damage=='initial':p['initial_state_sha256']='c184'+'0'*60
            else:h[0]['validation_unweighted_bce']=math.nan
            with self.assertRaises(ValueError):execution.validate_reuse(p,r,x,h,fold,bank,6)


if __name__ == '__main__':
    unittest.main()
