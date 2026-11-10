"""EXP-019 finite CPU fixtures: no K400 download, video decode or real fit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.common.video.order_training import (
    MODES,_fit,ordered_batch,order_records,score_order_batch,train_order_batch,verify_cache_acceptance,check_initial_state,
)
from experiments.common.video.order_dense_evaluation import _score_stream
from experiments.common.video.paired_runtime import train_paired_batch
from experiments.common.video.r2plus1d_classifier import configure_fp32
from experiments.common.video.supervised_runtime import ClipInput


class Mean(nn.Module):
    def forward(self,x):return x.mean((2,3,4))


class Tiny(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        if not pretrained:raise AssertionError('Fixture must receive the explicit public-initialization intent')
        self.backbone=nn.Sequential(Mean(),nn.Linear(6,3),nn.BatchNorm1d(3))
        self.head=nn.Sequential(nn.Linear(3,3),nn.ReLU(),nn.Dropout(.5),nn.Linear(3,1))
    def forward(self,x):return self.head(self.backbone(x)).reshape(-1)
    def train(self,mode=True):
        super().train(mode);self.backbone[2].eval();return self


class FirstFrame(Tiny):
    def forward(self,x):
        # Model parameters participate with coefficient0 only for score-shape API.
        return x[:,0,0].mean((1,2))+0*super().forward(x)


def window(j):
    return dict(clip_id='clip-%04d'%j,unit_id='u%d'%j,group_id='g%d'%j,
                media_sha256='a'*64,frame_indices=list(range(j*20,j*20+16)),
                source_pts_seconds=[j*4+i*.2 for i in range(16)])


class TinyDataset(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        x=torch.tensor([(i*7+j*11)%256 for j in range(16)],dtype=torch.uint8)
        return x[:,None,None,None].expand(16,2,2,3).clone(),self.rows[i][2],i


class OrderExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        parent=(ROOT/'.codex/docs/.temp').resolve();parent.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(prefix='order-cpu-fixture-',dir=str(parent));self.addCleanup(temp.cleanup)
        root=Path(temp.name).resolve();self.assertEqual(root.parent,parent)
        bank={};rows=[]
        for i in range(640):
            w=window(i);w['kind']='pos' if (i<356 or 504<=i<600) else 'neg'
            bank[w['clip_id']]=w;rows.append((w['clip_id'],(),int(w['kind']=='pos')))
        ids=[r[0] for r in rows[:504]]
        epochs=[dict(epoch=e,batches=[dict(clip_ids=ids[j:j+24]) for j in range(0,504,24)]) for e in range(20)]
        plan=dict(plan_sha256='0'*64,base_plan_sha256='1'*64,paired_plan_sha256='2'*64,paired_epochs_sha256='3'*64)
        proof=root/'proof.json';proof.write_text('{"CPU_fixture":true}')
        fingerprints={'proof.json':hashlib.sha256(proof.read_bytes()).hexdigest()}
        return root,plan,dict(pos_weight=148/356),rows,epochs,list(range(504,640)),bank,fingerprints,TinyDataset(rows)

    def test_A_exact_old_lambda0_gradient_update_dropout_RNG(self):
        configure_fp32(7);old=Tiny().train();new=deepcopy(old).train()
        optimizer_old=torch.optim.Adam(old.parameters(),lr=1e-3)
        optimizer_new=torch.optim.Adam(new.parameters(),lr=1e-3)
        u8=torch.arange(24*16*2*2*3,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(24,16,2,2,3)
        labels=(torch.arange(24)%2==0).float();windows=[window(i) for i in range(24)]
        transform=ClipInput('rgb_zero');rng=torch.get_rng_state().clone()
        a=train_paired_batch(old,optimizer_old,u8,labels,transform,148/356,[(0,1),(8,9),(16,17)],rank_lambda=0.)
        after=torch.get_rng_state().clone();torch.set_rng_state(rng)
        b,_=train_order_batch(new,optimizer_new,u8,labels,transform,148/356,windows,'chronological')
        self.assertEqual(a['bce_sum'],b['bce_sum']);self.assertEqual(a['loss'],b['loss'])
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        for x,y in zip(old.parameters(),new.parameters()):
            self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(x.grad,y.grad))

    def test_B_same_frame_multiset_zero_D_stable_physical_mapping_no_RNG(self):
        windows=[window(1),window(2)]
        u8=torch.arange(2*16*2*2*3).remainder(256).to(torch.uint8).reshape(2,16,2,2,3)
        before=deepcopy(windows);py=random.getstate();npstate=np.random.get_state();tr=torch.get_rng_state().clone()
        ordered,records=ordered_batch(u8,windows,'fixed_permutation')
        repeat,again=ordered_batch(u8,windows,'fixed_permutation')
        self.assertEqual(windows,before);self.assertTrue(torch.equal(ordered,repeat));self.assertEqual(records,again)
        for i,record in enumerate(records):
            self.assertEqual(sorted(record['permutation']),list(range(16)))
            self.assertTrue(torch.equal(ordered[i],u8[i,record['permutation']]))
            self.assertEqual(sorted(x.numpy().tobytes() for x in ordered[i]),sorted(x.numpy().tobytes() for x in u8[i]))
        self.assertEqual(ClipInput('rgb_zero')(ordered)[:,3:].count_nonzero(),0)
        self.assertEqual(py,random.getstate());self.assertTrue(torch.equal(tr,torch.get_rng_state()))
        after=np.random.get_state();self.assertEqual(npstate[0],after[0]);np.testing.assert_array_equal(npstate[1],after[1]);self.assertEqual(npstate[2:],after[2:])
        alias=dict(windows[0],clip_id='unrelated-name',label=0,split='other',epoch=17)
        self.assertEqual(order_records([alias]),order_records([windows[0]]))

    def test_validation_and_dense_share_branch_order_digest_failure_withholds(self):
        windows=[window(3),window(4)]
        u8=torch.arange(16,dtype=torch.uint8).view(1,16,1,1,1).expand(2,16,2,2,3).clone()*10
        configure_fp32(7);net=FirstFrame().eval();transform=ClipInput('rgb_zero')
        models={mode:(net,transform) for mode in MODES};expected={}
        for mode in MODES:expected[mode]=torch.sigmoid(score_order_batch(net,u8,transform,windows,mode)[0]).numpy()
        info={}
        def batches():
            yield u8,windows
            info.update(rgb112_digest='canonical-before-permutation',windows=2)
        captured=[]
        result=_score_stream(batches(),info,'canonical-before-permutation',models,lambda score,order:captured.append((score,order)))
        self.assertEqual(result['accepted_windows'],2)
        for i,(score,order) in enumerate(captured):
            self.assertEqual(order['permutation'],order_records([windows[i]])[0]['permutation'])
            for mode in MODES:self.assertEqual(score[mode],float(expected[mode][i]))
        info.clear()
        with self.assertRaisesRegex(ValueError,'provisional scores withheld'):_score_stream(batches(),info,'wrong-summary',models)
        poisoned=dict(windows[0],frame_indices=[0]*16)
        with self.assertRaises(ValueError):ordered_batch(u8[:1],[poisoned],'fixed_permutation')

    def test_full20_420_same_initialization_earliest_BCE_with_finite_bad_scores(self):
        root,plan,base,rows,epochs,validation,bank,fp,ds=self.fixture()
        import experiments.common.video.order_training as module
        real_score=module.score_order_batch
        modes_seen=[]
        def finite_bad(net,u8,transform,windows,mode):
            modes_seen.append(mode)
            # Constant finite poor prediction is retained; exact BCE ties keep1.
            return torch.full((len(u8),),-20.),order_records(windows)
        with patch.object(module,'score_order_batch',side_effect=finite_bad):
            warm=_fit(root,root/'runtime-probe','chronological',plan,base,rows,epochs,validation,bank,fp,
                      'cpu',0,0,_test_only=dict(dataset=ds,model_factory=Tiny),_probe=True)
            a=_fit(root,root/'A','chronological',plan,base,rows,epochs,validation,bank,fp,
                   'cpu',0,0,_test_only=dict(dataset=ds,model_factory=Tiny))
            b=_fit(root,root/'B','fixed_permutation',plan,base,rows,epochs,validation,bank,fp,
                   'cpu',0,0,expected_initial_state_sha256=a['initial_state_sha256'],
                   _test_only=dict(dataset=ds,model_factory=Tiny))
        self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256'])
        self.assertEqual(warm['status'],'runtime_probe_complete')
        self.assertEqual(warm['completed_epochs'],1);self.assertEqual(warm['optimizer_steps'],21)
        self.assertEqual(a['initial_state_sha256'],warm['initial_state_sha256'])
        self.assertNotEqual(a['initial_state_sha256'],warm['final_state_sha256'])
        self.assertNotIn('selected_checkpoint',warm)
        self.assertFalse(list((root/'runtime-probe').glob('*.pt')))
        self.assertEqual(set(modes_seen),set(MODES))
        for name,result in [('A',a),('B',b)]:
            self.assertEqual(result['completed_epochs'],20);self.assertEqual(result['optimizer_steps'],420)
            self.assertEqual(result['best_epoch'],1)
            self.assertGreater(result['best_validation_bce'],10)
            history=[json.loads(x) for x in (root/name/'epochs.jsonl').read_text().splitlines()]
            self.assertEqual(len(history),20)
            self.assertTrue(all(h['samples']==504 and h['optimizer_steps']==21 for h in history))
            self.assertEqual(history[-1]['total_updates'],420)
            self.assertEqual(history[0]['train_clip_ids'],[r[0] for r in rows[:504]])
            self.assertTrue((root/name/'best.pt').exists());self.assertTrue((root/name/'last.pt').exists())
        ha=[json.loads(x) for x in (root/'A/epochs.jsonl').read_text().splitlines()]
        hb=[json.loads(x) for x in (root/'B/epochs.jsonl').read_text().splitlines()]
        self.assertEqual([h['rng_before'] for h in ha],[h['rng_before'] for h in hb])
        self.assertEqual([h['rng_after'] for h in ha],[h['rng_after'] for h in hb])
        self.assertEqual(a['profile']['loader_generator_seed'],100007)

    def test_nonfinite_update_fails_and_keeps_failure_record(self):
        root,plan,base,rows,epochs,validation,bank,fp,ds=self.fixture()
        class Bad(Tiny):
            def forward(self,x):return super().forward(x)*float('nan')
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite'):
            _fit(root,root/'failed','chronological',plan,base,rows,epochs,validation,bank,fp,
                 'cpu',0,0,_test_only=dict(dataset=ds,model_factory=Bad))
        failure=json.loads((root/'failed/failure.json').read_text())
        self.assertEqual(failure['optimizer_steps'],0)
        self.assertFalse((root/'failed/training-result.json').exists())

    def test_independent_acceptance_requires_plan_index_and_full_coverage(self):
        root,_,_,_,_,_,_,_,_=self.fixture()
        directory=root/'plan';directory.mkdir();(directory/'plan.json').write_text('{"fixture":true}')
        plan_sha=hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()
        index=root/'verified.csv';index.write_text('cpu-fixture-index')
        index_sha=hashlib.sha256(index.read_bytes()).hexdigest()
        plan=dict(data_fingerprints=[dict(path='verified.csv',sha256=index_sha)],pixels_ready=False,training_ready=False)
        bank={str(i):window(i) for i in range(2)}
        with self.assertRaisesRegex(ValueError,'Separate actual'):
            verify_cache_acceptance(root,directory,plan,bank,None)
        path=root/'acceptance.json'
        proof=dict(status='order_supervised_cache_ready',order_plan_sha256=plan_sha,
            verified_cache_index='verified.csv',verified_cache_index_sha256=index_sha,
            required_frame_count=32,verified_required_frame_count=32,failed_required_frames=0)
        path.write_text(json.dumps(proof));verify_cache_acceptance(root,directory,plan,bank,path)
        self.assertFalse(plan['pixels_ready']);self.assertFalse(plan['training_ready'])
        for changes in [dict(order_plan_sha256='0'*64),dict(verified_required_frame_count=31),
                        dict(failed_required_frames=1),dict(verified_cache_index_sha256='0'*64)]:
            path.write_text(json.dumps(dict(proof,**changes)))
            with self.assertRaises(ValueError):verify_cache_acceptance(root,directory,plan,bank,path)

    def test_production_A_requires_plan_initialization_without_optional_override(self):
        actual='a'*64;plan=dict(expected_initial_state_sha256=actual)
        check_initial_state(actual,plan,'chronological')
        for wrong in ({},dict(expected_initial_state_sha256='b'*64),dict(expected_initial_state_sha256='not-a-hash')):
            with self.assertRaisesRegex(ValueError,'frozen plan'):
                check_initial_state(actual,wrong,'chronological')
        with self.assertRaisesRegex(ValueError,'frozen plan'):
            check_initial_state('b'*64,plan,'chronological','b'*64)
        with self.assertRaisesRegex(ValueError,'measured A'):
            check_initial_state(actual,plan,'fixed_permutation')
        with self.assertRaisesRegex(ValueError,'same measured'):
            check_initial_state(actual,plan,'fixed_permutation','b'*64)
        check_initial_state(actual,plan,'fixed_permutation',actual)


if __name__=='__main__':unittest.main()
