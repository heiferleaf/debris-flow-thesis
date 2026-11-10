"""EXP-017 orchestration regression tests: mocks/small files, no model or video.

Only .codex/docs/.temp fixtures are written. Production orchestration, budget,
deployment validation, metrics and offline BCE guards execute without CUDA.
"""
from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.common.video import cv_execution as execution
from experiments.common.video import cv_dense_evaluation as dense
from experiments.common.video.supervised_plan import digest,write_json
from experiments.common.video.supervised_runtime import FrameRef


def module(name):
    spec=importlib.util.spec_from_file_location('exp017_fixture_'+name,ROOT/('experiments/runspecs/EXP-017/'+name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


pipeline,audit,train=module('pipeline'),module('audit'),module('train')


class DormantWatchdog:
    """The real watchdog is tested separately with every external action mocked."""
    def __init__(self,**kwargs):self.kwargs=kwargs
    def start(self):pass
    def join(self,timeout):pass


class NoOuterValidation(dict):
    def __getitem__(self,key):
        if key=='outer_eval':raise AssertionError('A training stage read outer-held-out data')
        return super().__getitem__(key)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'.codex/docs/.temp';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='cv-execution-fixture-',dir=str(parent))
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.plan=self.root/'data/metadata/EXP-017/plan';self.plan.mkdir(parents=True)
        self.planfile=self.file('data/metadata/EXP-017/plan/plan.json',b'{"fixture":true}')
        self.acceptance=self.file('data/metadata/EXP-017/accepted.json',b'{"fixture":true}')
        self.reference=self.file('data/metadata/EXP-017/dense-reference.json',b'{"fixture":true}')
        self.ffmpeg=self.file('fake-ffmpeg.bin',b'CPU fixture, never executed')
        self.code=self.file('fake-execution.py',b'# CPU fixture, never executed\n')
        self.metadata=self.file('data/metadata/EXP-017/input.json',b'{"fixture":true}')
        sources,media,windows={},[],{}
        self.scores=[];self.folds=[]
        for fold in range(1,6):
            uids=[];scores=[];negatives=(7,7,7,8,8)[fold-1]
            for j in range(30):
                uid='fixture-f%d-u%02d'%(fold,j);uids.append(uid)
                path=self.file('media/'+uid+'.bin',b'not a video: '+uid.encode('ascii'))
                source=dict(unit_id=uid,video_label='neg' if j<negatives else 'pos',group_id='fixture-group-'+uid,
                    middleware_path=path.relative_to(self.root).as_posix(),middleware_sha256=digest(path))
                sources[uid]=source
                media.append(dict(unit_id=uid,path=source['middleware_path'],sha256=source['middleware_sha256'],size_bytes=path.stat().st_size))
                windows[uid]=[dict(unit_id=uid,frame_indices=list(range(16)),source_pts_seconds=[i*.2 for i in range(16)],target_start_seconds=0.)]
                scores.append(dict(unit_id=uid,label=int(source['video_label']=='pos'),group_id=source['group_id'],score=.2 if j<negatives else .8,windows=1))
            self.scores.append(scores)
            self.folds.append(dict(fold=fold,inner_train=dict(epoch_clip_ids=[['inner-%d-%d'%(fold,e)] for e in range(20)]),
                inner_val=dict(clip_ids=['val-%d'%fold]),outer_train=dict(epoch_clip_ids=[['outer-%d-%d'%(fold,e)] for e in range(20)]),
                outer_eval=dict(video_ids=uids)))
        refs,index={},{}
        for i in range(2):
            uid=list(sources)[i];path=self.file('cache/frame%d.bin'%i,b'not a JPEG; never decoded')
            k=(uid,0);refs[k]=FrameRef(str(path),digest(path),'a'*64)
            index[k]=dict(cache_path=path.relative_to(self.root).as_posix(),jpeg_sha256=digest(path),rgb112_sha256='a'*64,size_bytes=str(path.stat().st_size))
        self.context=dict(root=self.root,plan_dir=self.plan,plan=dict(source_fingerprints=[dict(path=self.metadata.relative_to(self.root).as_posix(),sha256=digest(self.metadata),size_bytes=self.metadata.stat().st_size)]),
            bank={},refs=refs,cache_index=index,sources=sources,folds=self.folds,
            fingerprints={self.acceptance.relative_to(self.root).as_posix():digest(self.acceptance)},
            dense=dict(path=self.reference,reference=dict(media_refs=media),windows=windows))
        self.manifest=self.root/'deployment.csv';self.write_manifest()
        self.preflight=self.root/'data/process/EXP-017/check/preflight.json'

    def file(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p

    def write_manifest(self,rows=None):
        rows=rows if rows is not None else list(execution.deployment_records(self.context,[self.code]).values())
        with self.manifest.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','kind','sha256','size_bytes']);w.writeheader();w.writerows(rows)

    def args(self,stage,out):
        result=['--stage',stage,'--plan',str(self.plan),'--acceptance',str(self.acceptance),'--dense-reference',str(self.reference),
            '--out',str(out),'--ffmpeg',str(self.ffmpeg),'--transfer-manifest',str(self.manifest),'--prior-cpu-seconds','0']
        if stage=='full':
            receipt=self.file('timer-fixture.json',b'{"fixture_not_a_real_timer":true}')
            result+=['--preflight',str(self.preflight),'--gpu-start-unix','100','--platform-deadline-unix','14000',
                '--platform-shutdown-receipt',str(receipt),'--prior-stage-spend-yuan','0','--prior-total-spend-yuan','2.87']
        return result

    def pipeline_patches(self,stack):
        stack.enter_context(patch.object(pipeline,'ROOT',self.root))
        stack.enter_context(patch.object(pipeline,'load_execution',side_effect=lambda *a:deepcopy(self.context)))
        stack.enter_context(patch.object(pipeline,'code_paths',return_value=[self.code]))
        stack.enter_context(patch.object(pipeline.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)))
        stack.enter_context(patch.object(pipeline.threading,'Thread',DormantWatchdog))

    def check(self):
        with ExitStack() as stack:
            self.pipeline_patches(stack)
            pipeline.main(self.args('check',self.preflight.parent))

    def full(self,name,*,slow=False,fail_fold=None,expired=False):
        out=self.root/'data/results/exp017'/name;calls=[]
        def fit(c,path,fold,stage,inner_result=None):
            calls.append((fold,stage))
            if fold==fail_fold:raise RuntimeError('synthetic fit failure')
            if stage=='fixed_outer':self.assertEqual(inner_result['selected_epochs'],3)
            return dict(stage=stage,selected_epochs=3,seconds=1000. if slow else 1.,optimizer_steps=1)
        def evaluate(c,path,fold,outer,E,ffmpeg):
            calls.append((fold,'dense'));self.assertEqual(E,3)
            rows=deepcopy(self.scores[fold-1])
            return dict(fold=fold,scores=rows,metrics=execution.video_metrics(rows),seconds=1.)
        with ExitStack() as stack:
            self.pipeline_patches(stack)
            stack.enter_context(patch.object(pipeline,'train_stage',side_effect=fit))
            stack.enter_context(patch.object(pipeline,'evaluate_fold',side_effect=evaluate))
            stack.enter_context(patch.object(pipeline.time,'time',side_effect=[100.]*5+[20000.] if expired else None,return_value=100.))
            try:pipeline.main(self.args('full',out))
            except Exception as error:return out,calls,error
        return out,calls,None

    def test_preflight_cannot_omit_media_or_required_jpegs(self):
        required=execution.deployment_records(self.context,[self.code]);rows=list(required.values())
        self.assertEqual(sum(r['kind']=='source_media' for r in rows),150)
        self.assertEqual(sum(r['kind']=='supervised_JPEG' for r in rows),2)
        for n,kinds in enumerate(({'source_media'},{'supervised_JPEG'},{'source_media','supervised_JPEG'})):
            self.write_manifest([r for r in rows if r['kind'] not in kinds])
            out=self.root/('data/process/EXP-017/truncated-%d'%n)
            with ExitStack() as stack:
                self.pipeline_patches(stack)
                with self.assertRaises(ValueError):pipeline.main(self.args('check',out))
            self.assertTrue((out/'pipeline-failure.json').exists());self.assertFalse((out/'preflight.json').exists())
        for change in ('size','sha','duplicate','extra'):
            changed=deepcopy(rows)
            if change=='size':changed[0]['size_bytes']+=1
            elif change=='sha':changed[0]['sha256']='0'*64
            elif change=='duplicate':changed.append(changed[0])
            else:changed.append(dict(changed[0],path='not-required'))
            with self.assertRaises(ValueError):execution.validate_deployment_manifest(changed,required)
        self.write_manifest();self.check()
        self.assertEqual(json.loads(self.preflight.read_text('utf-8'))['status'],'cv_static_preflight_passed')

    def test_actual_pipeline_fold_order_and_complete_summary(self):
        self.check();out,calls,error=self.full('complete')
        self.assertIsNone(error)
        self.assertEqual(calls,[(f,s) for f in range(1,6) for s in ('inner_select','fixed_outer','dense')])
        result=json.loads((out/'pipeline-result.json').read_text('utf-8'))
        self.assertEqual((result['summary']['P'],result['summary']['N']),(113,37))
        self.assertIsNone(result['summary']['pooled_auc'])

    def test_refused_forecast_fit_failure_and_final_deadline_withhold_summary(self):
        self.check()
        for name,kwargs in (('forecast',dict(slow=True)),('fit-failure',dict(fail_fold=3)),('expired',dict(expired=True))):
            out,calls,error=self.full(name,**kwargs)
            self.assertIsNotNone(error);self.assertTrue((out/'pipeline-failure.json').exists())
            self.assertFalse((out/'pipeline-result.json').exists())
            if name=='forecast':
                self.assertEqual(calls,[(1,'inner_select'),(1,'fixed_outer'),(1,'dense')])
                self.assertEqual(json.loads((out/'fold-01/remaining-budget.json').read_text('utf-8'))['status'],'refused')
            if name=='expired':self.assertIsInstance(error,TimeoutError)

    def test_training_route_inner_only_and_fresh_outer_prefix(self):
        f=NoOuterValidation(deepcopy(self.folds[0]));c=dict(self.context,folds=[f]+self.folds[1:])
        seen=[]
        def fake_fit(root,out,**kwargs):
            seen.append(kwargs);out=Path(out);out.mkdir()
            write_json(out/'profile.json',dict(execution_kind='registered_fit',initial_state_sha256=execution.INITIAL_SHA))
            outer=kwargs['stage']=='fixed_outer';E=kwargs['selected_epochs'] if outer else 20
            return dict(status='fit_complete',stage=kwargs['stage'],fold=1,completed_epochs=E,selected_epochs=E if outer else 3,
                initial_state_sha256=execution.INITIAL_SHA,fixed_outer_validation_evaluated=False,
                selected_checkpoint=dict(filename='selected.pt' if outer else 'best.pt'))
        with patch.object(execution,'fit',side_effect=fake_fit):
            inner=execution.train_stage(c,self.root/'inner',1,'inner_select')
            outer=execution.train_stage(c,self.root/'outer',1,'fixed_outer',inner)
            with self.assertRaises(ValueError):execution.train_stage(c,self.root/'bad',1,'fixed_outer',dict(inner,selected_epochs=0))
        self.assertEqual(seen[0]['validation_clip_ids'],f['inner_val']['clip_ids'])
        self.assertNotIn('validation_clip_ids',seen[1]);self.assertEqual(seen[1]['epoch_clip_ids'],f['outer_train']['epoch_clip_ids'])
        self.assertEqual(seen[1]['epoch_clip_ids'][:seen[1]['selected_epochs']],f['outer_train']['epoch_clip_ids'][:3])
        self.assertEqual(outer['completed_epochs'],3)
        self.assertTrue(all(r['expected_initial_state_sha256']==execution.INITIAL_SHA for r in seen))
        self.assertTrue(all(not set(r)&{'checkpoint','optimizer','state_dict','model_factory'} for r in seen))

    def test_budget_limits_and_forecast_ignore_score_quality(self):
        valid=dict(now=100.,start=100.,platform_end=14000.,price=1.48,prior_gpu=0.,prior_cpu=0.,prior_stage=0.,prior_total=2.87)
        budget=pipeline.budget_deadline(**valid);self.assertLess(budget['deadline_unix'],14000.)
        for key,value in (('price',1.49),('prior_gpu',14400.),('prior_cpu',7201.),('platform_end',15000.),('prior_total',float('nan')),('prior_stage',7.)):
            with self.assertRaises(ValueError):pipeline.budget_deadline(**dict(valid,**{key:value}))
        evidence=[dict(inner=dict(seconds=10.,optimizer_steps=2),outer=dict(seconds=20.,optimizer_steps=4),
            dense=dict(seconds=30.,scores=[dict(windows=30,score=.99)],metrics=dict(AUC=1.)))]
        expected=pipeline.remaining_forecast(evidence,self.folds[1:],self.context['dense']['windows'])
        evidence[0]['dense']['scores'][0]['score']=.01;evidence[0]['dense']['metrics']['AUC']=0.
        self.assertEqual(expected,pipeline.remaining_forecast(evidence,self.folds[1:],self.context['dense']['windows']))
        self.assertEqual(expected['worst_case_remaining_updates'],160)

    def test_missing_fold_or_duplicate_outer_cannot_summarize(self):
        results=[dict(fold=i+1,scores=rows,metrics=execution.video_metrics(rows)) for i,rows in enumerate(self.scores)]
        execution.summarize_folds(results,self.context['sources'])
        with self.assertRaises(ValueError):execution.summarize_folds(results[:4],self.context['sources'])
        broken=deepcopy(results);broken[-1]['scores'][0]=deepcopy(broken[0]['scores'][0])
        with self.assertRaises(ValueError):execution.summarize_folds(broken,self.context['sources'])

    def test_shutdown_attempt_even_when_evidence_writes_fail(self):
        stop=Mock();stop.wait.return_value=False
        for shutdown_failure in (False,True):
            with patch.object(pipeline,'write_json',side_effect=OSError('synthetic disk failure')), \
                 patch.object(pipeline,'_shutdown',side_effect=RuntimeError('synthetic shutdown failure') if shutdown_failure else None,return_value=dict(returncode=0)) as shutdown, \
                 patch.object(pipeline.os,'_exit',side_effect=SystemExit) as exit_process:
                with self.assertRaises(SystemExit):pipeline._hard_stop(0.,self.root,stop)
                shutdown.assert_called_once_with(self.root);exit_process.assert_called_once_with(124)

    def test_nonfinite_bce_and_standalone_training_refused(self):
        predictions=[dict(logit=0.,label=1),dict(logit=0.,label=0)]
        value=audit.bce(predictions);self.assertEqual(audit.checked_validation_bce(predictions,value),value)
        for recorded in (float('nan'),float('inf'),-float('inf'),-1.,value+.001):
            with self.assertRaises(ValueError):audit.checked_validation_bce(predictions,recorded)
        with patch.object(audit,'bce',return_value=float('nan')):
            with self.assertRaises(ValueError):audit.checked_validation_bce(predictions,value)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
            train.main(['--stage','inner_select','--fold','1','--device','cuda'])
        self.assertEqual(caught.exception.code,2)

    def test_dense_scores_provisional_until_canonical_eof(self):
        row=dict(unit_id='u',target_start_seconds=0.,frame_indices=list(range(16)),source_pts_seconds=[i*.2 for i in range(16)])
        pixels=np.zeros((1,16,1,1,3),np.uint8);info=dict(windows=1,rgb112_digest='expected')
        with patch.object(dense,'score_clips',return_value=torch.tensor([0.])):
            self.assertEqual(dense.score_stream([(pixels,[row])],info,'expected',None,None)['score'],.5)
            with self.assertRaises(ValueError):dense.score_stream([(pixels,[row])],info,'wrong',None,None)
            with self.assertRaises(ValueError):dense.score_stream([(pixels,[row]),(pixels,[row])],info,'expected',None,None)
        with patch.object(dense,'score_clips',return_value=torch.tensor([float('nan')])):
            with self.assertRaises(FloatingPointError):dense.score_stream([(pixels,[row])],info,'expected',None,None)


if __name__=='__main__':unittest.main()
