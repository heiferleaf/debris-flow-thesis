"""Small offline provenance/fit/dense audit fixtures; no model or media calls."""
import csv
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.common.video.supervised_plan import digest
from experiments.common.video.supervised_runtime import EpochBatches, FrameRef
from experiments.common.video.cv_supervised_training import _canonical
from experiments.tests.test_exp017_resume import first_fold_fixture

SOURCE = Path(__file__).resolve().parents[1] / 'runspecs/EXP-017/audit_resume.py'
SPEC = importlib.util.spec_from_file_location('resume_audit', SOURCE)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def dense_fixture(root):
    directory=root/M.CHILD/'fold-02';d=directory/'dense';d.mkdir(parents=True)
    windows={};sources={};scores=[];videos={};rows=[];refs={}
    for uid,label,values in [('p',1,[.6,.8]),('n',0,[.2,.4])]:
        sources[uid]=dict(video_label='pos' if label else 'neg',group_id='g'+uid)
        windows[uid]=[]
        for i,value in enumerate(values):
            w=dict(frame_indices=list(range(i*100,i*100+16)),source_pts_seconds=[i+j/5 for j in range(16)],target_start_seconds=float(i))
            windows[uid].append(w)
            rows.append(dict(unit_id=uid,target_start_seconds=i,start=i,end=i+3,frame_tuple_sha256=M.tuple_digest(w['frame_indices']),score=value))
        scores.append(dict(unit_id=uid,label=label,group_id='g'+uid,fold=2,score=max(values),windows=2))
        refs[uid]=dict(decoded=200,rgb112_digest='e'*64)
        videos[uid]=dict(refs[uid],canonical_reference_matched=True,score=max(values),windows=2)
        (d/(uid+'-filter.txt')).write_text('select=fixture',encoding='utf-8')
    frozen=dict(input_fingerprints={'metadata.json':'a'*64},dense_reference_sha256='b'*64)
    outer=dict(selected_epochs=3,selected_checkpoint={'sha256':'c'*64})
    result=dict(status='outer_fold_complete',fold=2,selected_epochs=3,plan_sha256=M.PLAN_SHA,
        checkpoint_sha256='c'*64,pixel_reference_sha256='b'*64,input_fingerprints=frozen['input_fingerprints'],
        threshold=.5,aggregation='max',outer_selection_use=False,independent_test_used=False,
        scores=scores,videos=videos,metrics=M.video_metrics(scores),
        metrics_by_group={g:M.video_metrics([r for r in scores if r['group_id']==g]) for g in ['gn','gp']})
    context=dict(sources=sources,dense={'windows':windows,'reference':{'videos':refs}})
    fold=dict(fold=2,outer_eval={'video_ids':['p','n']})
    def publish(result_rows=rows, payload=result):
        with (d/'window-scores.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(result_rows)
        payload['window_scores_sha256']=digest(d/'window-scores.csv')
        (d/'result.json').write_text(json.dumps(payload),encoding='utf-8')
    publish()
    return directory,fold,outer,frozen,context,rows,result,publish


def rich_fit_fixture(root):
    context,parent,frozen,_=first_fold_fixture(root)
    code={'/root/autodl-tmp/debris-flow-thesis-exp016/experiments/common/video/cv_supervised_training.py':'9'*64}
    frozen['code_sha256']={'experiments/common/video/cv_supervised_training.py':'9'*64}
    for stage,epochs in [('inner',20),('outer',3)]:
        d=parent/'fold-01'/stage
        p=json.loads((d/'profile.json').read_text())
        rule='minimum_fixed_inner_unweighted_BCE_earliest_exact_tie' if stage=='inner' else 'fixed_E_from_inner_no_validation_or_best_selection'
        p.update(objective='ordinary_weighted_bce',pretrained='KINETICS400_V1',
            model_class='experiments.common.video.r2plus1d_classifier.VideoClassifier',
            expected_inner_initial_state_sha256=None if stage=='inner' else M.INITIAL_SHA,
            workers=2,cache_mib=128,frozen_schedule_epochs=20,selection_rule=rule,planned_optimizer_steps=epochs,
            train_P=1,train_N=1,code_sha256=code,
            optimizer=dict(name='Adam',lr_backbone=1e-4,lr_head=1e-3,betas=[.9,.999],eps=1e-8,weight_decay=0,initial_state_entries=0),
            environment=dict(device='cuda',precision='FP32',amp=False,tf32_matmul=False,tf32_cudnn=False,cudnn_benchmark=False,cudnn_deterministic=True))
        (d/'profile.json').write_text(json.dumps(p),encoding='utf-8');profile_sha=digest(d/'profile.json')
        history=[json.loads(x) for x in (d/'epochs.jsonl').read_text().splitlines()]
        for i,row in enumerate(history):
            row.update(actual_effective_batch_sizes=[2],learning_rates=[1e-4,1e-3],grad_norm_mean=0.,grad_norm_max=0.,seconds=.1,
                       code_sha256=code,profile_sha256=profile_sha,model_state_sha256='f'*64)
            if stage=='inner':row.update(validation_samples=2,validation_clip_ids=['p','n'],best_epoch=1 if i<2 else 3)
        (d/'epochs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history),encoding='utf-8')
        result=json.loads((d/'training-result.json').read_text())
        result.update(selection_rule=rule,code_sha256=code,profile_sha256=profile_sha,train_P=1,train_N=1,pos_weight=1.,final_state_sha256='f'*64)
        if stage=='inner':result['best_validation_unweighted_bce']=history[2]['validation_unweighted_bce']
        (d/'training-result.json').write_text(json.dumps(result),encoding='utf-8')
    return context,parent,frozen


class AuditFixtures(unittest.TestCase):
    def test_fixed_parent_child_origins_and_false_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/M.PARENT;child=root/M.CHILD;child.mkdir(parents=True)
            b=dict(parent_execution_inputs_sha256='a'*64,files_sha256='b'*64)
            lineage=dict(parent_run_id=parent.name,parent_path=M.PARENT,parent_binding_path=M.BINDING,parent_binding_sha256=M.BINDING_SHA,
                parent_execution_inputs_sha256='a'*64,parent_artifacts_csv_sha256='b'*64,reused_folds=[1],newly_executed_folds=[2,3,4,5],
                inherited_parent_weights=False,inherited_parent_optimizer=False)
            p=dict(resume_lineage=lineage,fold_artifact_origins=[dict(fold=i,reused=i==1,run_id=parent.name if i==1 else child.name,
                path=(M.PARENT if i==1 else M.CHILD)+'/fold-%02d'%i) for i in range(1,6)])
            f={'resume_lineage':copy.deepcopy(lineage)}
            paths=M.origins_and_lineage(root,parent,child,p,f,b)
            self.assertEqual(paths[0],parent/'fold-01');self.assertEqual(paths[-1],child/'fold-05')
            bad=copy.deepcopy(p);bad['fold_artifact_origins'][0]['path']=M.CHILD+'/fold-01'
            with self.assertRaises(ValueError):M.origins_and_lineage(root,parent,child,bad,f,b)
            bad=copy.deepcopy(p);bad['resume_lineage']['inherited_parent_weights']=True
            with self.assertRaises(ValueError):M.origins_and_lineage(root,parent,child,bad,f,b)
            (child/'fold-01').mkdir()
            with self.assertRaises(ValueError):M.origins_and_lineage(root,parent,child,p,f,b)

    def test_missing_folds_are_incomplete_but_child_failure_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/M.PARENT;child=root/M.CHILD;parent.mkdir(parents=True);child.mkdir(parents=True)
            args=(parent,child,root/'plan')
            result=M.audit(*args,root/'audit-incomplete',root/'proof',root/'manifest',root=root)
            self.assertEqual(result['status'],'incomplete');self.assertNotIn('summary',result)
            for i,name in enumerate(M.FAILURES):
                failure=child/name;failure.write_text('{}')
                result=M.audit(*args,root/('audit-failed-%d'%i),root/'proof',root/'manifest',root=root)
                self.assertEqual(result['status'],'audit_failed');self.assertTrue(result['metrics_withheld'])
                failure.unlink()  # Isolated synthetic fixture only.

    def test_real_schema_fit_bce_and_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,parent,frozen=rich_fit_fixture(root);e=M.Evidence(root)
            inner,outer=M.audit_fit(c,parent/'fold-01',c['folds'][0],frozen,e)
            self.assertEqual(inner['selected_epochs'],3)
            self.assertNotEqual(digest(parent/'fold-01/inner/inputs.json'),inner['parsed_inputs_sha256'])
            self.assertIn('parent/fold-01/inner/best.pt',e.hashes)
            path=parent/'fold-01/inner/epochs.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            rows[1]['validation_unweighted_bce']=float('nan');path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError):M.audit_fit(c,parent/'fold-01',c['folds'][0],frozen,M.Evidence(root))

    def test_frozen_balanced_batches_508_412_512_reject_legacy_tail_and_reorder(self):
        expected={508:[24]*2+[23]*20,412:[23]*16+[22]*2,512:[24]*6+[23]*16}
        for n in expected:
            with self.subTest(samples=n),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);c,parent,frozen=rich_fit_fixture(root);fold=c['folds'][0]
                sampler=EpochBatches(24);sampler.set_order(range(n));batches=list(sampler)
                actual=[len(b) for b in batches];steps=len(batches)
                self.assertEqual(actual,expected[n]);self.assertEqual([i for b in batches for i in b],list(range(n)))
                ids=['x%04d'%i for i in reversed(range(n))];metadata={}
                for i,cid in enumerate(ids):
                    row=dict(unit_id=cid,group_id=cid,label=i%2,frame_indices=list(range(16)),source_pts_seconds=[j/5 for j in range(16)])
                    c['bank'][cid]=row;frames=[]
                    for j in range(16):
                        relative='cache/%s/%d.jpg'%(cid,j);ref=FrameRef(str(root/relative),'1'*64,'2'*64)
                        c['refs'][(cid,j)]=ref
                        frames.append(dict(path='/root/autodl-tmp/debris-flow-thesis-exp016/'+relative,jpeg_sha256=ref.jpeg_sha256,rgb112_sha256=ref.rgb112_sha256))
                    metadata[cid]=dict(row,frames=frames)
                for stage,role,epochs in [('inner','inner_train',20),('outer','outer_train',3)]:
                    fold[role]['epoch_clip_ids']=[list(ids) for _ in range(20)];d=parent/'fold-01'/stage
                    old=json.loads((d/'inputs.json').read_text());val=old['validation_clip_ids']
                    bank=dict(metadata)
                    if val:bank.update({cid:old['clip_bank'][cid] for cid in val})
                    parsed=_canonical(dict(metadata=bank,epoch_clip_ids=fold[role]['epoch_clip_ids'],validation_clip_ids=val,files=frozen['input_fingerprints']))
                    (d/'inputs.json').write_text(json.dumps(dict(clip_bank=bank,epoch_clip_ids=fold[role]['epoch_clip_ids'],validation_clip_ids=val)),encoding='utf-8')
                    profile=json.loads((d/'profile.json').read_text());profile.update(parsed_inputs_sha256=parsed,planned_optimizer_steps=epochs*steps,train_P=n//2,train_N=n//2)
                    (d/'profile.json').write_text(json.dumps(profile),encoding='utf-8');profile_sha=digest(d/'profile.json')
                    history=[json.loads(x) for x in (d/'epochs.jsonl').read_text().splitlines()]
                    for e,row in enumerate(history):
                        row.update(train_clip_ids=list(ids),samples=n,train_P=n//2,train_N=n//2,optimizer_steps=steps,total_updates=(e+1)*steps,
                                   actual_effective_batch_sizes=list(actual),profile_sha256=profile_sha,parsed_inputs_sha256=parsed)
                    (d/'epochs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history),encoding='utf-8')
                    result=json.loads((d/'training-result.json').read_text());result.update(profile_sha256=profile_sha,parsed_inputs_sha256=parsed,optimizer_steps=epochs*steps,train_P=n//2,train_N=n//2)
                    (d/'training-result.json').write_text(json.dumps(result),encoding='utf-8')
                inner,outer=M.audit_fit(c,parent/'fold-01',fold,frozen,M.Evidence(root))
                self.assertEqual(inner['optimizer_steps'],20*steps);self.assertEqual(outer['optimizer_steps'],3*steps)
                path=parent/'fold-01/inner/epochs.jsonl';history=[json.loads(x) for x in path.read_text().splitlines()]
                legacy=[24]*(n//24)+([n%24] if n%24 else []);self.assertNotEqual(legacy,actual)
                history[0]['actual_effective_batch_sizes']=legacy
                path.write_text(''.join(json.dumps(x)+'\n' for x in history),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Epoch schedule'):
                    M.audit_fit(c,parent/'fold-01',fold,frozen,M.Evidence(root))
                history[0]['actual_effective_batch_sizes']=actual;history[0]['train_clip_ids'][0:2]=reversed(history[0]['train_clip_ids'][0:2])
                path.write_text(''.join(json.dumps(x)+'\n' for x in history),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Epoch schedule'):
                    M.audit_fit(c,parent/'fold-01',fold,frozen,M.Evidence(root))

    def test_dense_recomputes_metrics_and_rejects_duplicate_missing_or_drift(self):
        mutations=['duplicate_video','missing_window','metric_drift','group_metric_drift','pixel_digest','label','nan_score']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory,fold,outer,frozen,c,rows,p,publish=dense_fixture(root);e=M.Evidence(root)
            d,seen=M.audit_dense(directory,fold,outer,frozen,c,e)
            self.assertEqual(len(seen),4);self.assertEqual(d['metrics']['TP'],1)
            self.assertTrue(any(n.startswith(M.CHILD+'/fold-02/') for n in e.hashes))
        for mutation in mutations:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);directory,fold,outer,frozen,c,rows,p,publish=dense_fixture(root)
                if mutation=='duplicate_video':p['scores'].append(copy.deepcopy(p['scores'][0]))
                elif mutation=='missing_window':rows.pop()
                elif mutation=='metric_drift':p['metrics']['FP']+=1
                elif mutation=='group_metric_drift':p['metrics_by_group']['gn']['FP']+=1
                elif mutation=='pixel_digest':p['videos']['p']['rgb112_digest']='0'*64
                elif mutation=='label':p['scores'][0]['label']=0
                elif mutation=='nan_score':rows[0]['score']='nan'
                publish(rows,p)
                with self.assertRaises(ValueError):M.audit_dense(directory,fold,outer,frozen,c,M.Evidence(root))


if __name__=='__main__':unittest.main()
