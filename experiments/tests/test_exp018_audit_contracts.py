"""Reject mislabeled objectives and predictions linked to different weights."""
import copy
from pathlib import Path
import runpy
import unittest
API=runpy.run_path(str(Path(__file__).resolve().parents[1]/'runspecs/EXP-018/audit.py'))


class Lineage(unittest.TestCase):
    def test_objective_and_plan_are_bound(self):
        profile=dict(objective_mode='same',rank_lambda=.1,rank_scale=21/80,
                     pair_plan_sha256='p',pair_schedule_sha256='s')
        API['objective_contract'](profile,'same','p','s')
        for key,value in [('objective_mode','bce'),('rank_lambda',0.),('pair_plan_sha256','old'),('pair_schedule_sha256','old')]:
            changed=dict(profile);changed[key]=value
            with self.assertRaises(ValueError):API['objective_contract'](changed,'same','p','s')

    def test_predictions_bound_to_selected_weights_inputs_and_pipeline(self):
        selected={m:m+'.sha' for m in ('bce','cross','same')}
        dense=dict(checkpoint_sha256=selected,checkpoint_input_modes={m:'rgb_zero' for m in selected},metrics={'bce':{'FP':2}})
        pipeline=dict(metrics=dense['metrics'])
        API['dense_model_contract'](dense,pipeline,selected)
        for key,value in [('checkpoint_sha256',{**selected,'same':'other.sha'}),
                          ('checkpoint_input_modes',{m:'rgb_diff' for m in selected}),('metrics',{'bce':{'FP':0}})]:
            changed=copy.deepcopy(dense);changed[key]=value
            with self.assertRaises(ValueError):API['dense_model_contract'](changed,pipeline,selected)


if __name__=='__main__':unittest.main()
