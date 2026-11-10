"""Recompute all pilot decisions from saved predictions, never retrain or retune."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.tools.audit_supervised_pair import metrics, near, require, sha
from experiments.common.video.paired_plan import load_pair_plan

MODES=('bce','cross','same')


def objective_contract(profile,mode,pair_sha,schedule_sha):
    require(profile.get('objective_mode')==mode,'Objective folder/profile mismatch')
    require(profile.get('rank_lambda')==(0. if mode=='bce' else .1)
            and profile.get('rank_scale')==21/80,'Ranking objective differs from card')
    require(profile.get('pair_plan_sha256')==pair_sha and profile.get('pair_schedule_sha256')==schedule_sha,
            'Profile refers to another pair plan')


def dense_model_contract(dense,pipeline,selected):
    require(dense.get('checkpoint_sha256')==selected,'Dense scores came from different checkpoints')
    require(dense.get('checkpoint_input_modes')=={m:'rgb_zero' for m in MODES},'Dense input mode changed')
    require(pipeline.get('metrics')==dense.get('metrics'),'Pipeline and dense metrics disagree')


def bce(z,y):return max(z,0)-z*y+math.log1p(math.exp(-abs(z)))


def sigmoid(z):
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))


def audit(run,out):
    run,out=Path(run).resolve(),Path(out).resolve()
    run.relative_to(ROOT/'data/results/exp018')
    out.relative_to(ROOT/'data/results/exp018')
    sources={}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    pipeline=read(run/'pipeline-result.json')
    require(pipeline['status']=='completed','Incomplete pipeline')
    paired,epochs=load_pair_plan(ROOT,ROOT/'data/metadata/EXP-018/paired-plan-v1')
    require(pipeline['pair_plan_sha256']==sha(ROOT/'data/metadata/EXP-018/paired-plan-v1/plan.json'),'Pair plan mismatch')
    with (ROOT/paired['base_plan_path']/'assignments.csv').open(encoding='utf-8-sig',newline='') as f:
        val=[r for r in csv.DictReader(f) if r['split']=='validation']
    expected={r['clip_id']:int(r['kind']=='pos') for r in val}
    require(len(expected)==136 and sum(expected.values())==96,'Fixed validation identity changed')
    summaries,profiles,selected_shas={},[],{}
    for mode in MODES:
        result=read(run/mode/'training-result.json'); profile=read(run/mode/'profile.json')
        require(result['status']=='supervised_complete_dense_evaluation_pending'
                and result['completed_epochs']==20 and result['optimizer_steps']==420,'Training incomplete')
        require(profile==result['profile'] and profile['mode']=='rgb_zero' and profile['seed']==7
                and profile['requested_epochs']==20 and profile['requested_optimizer_updates']==420
                and profile['plan_sha256']==paired['base_plan_sha256'] and not profile['smoke'],'Profile mismatch')
        profiles.append(profile)
        objective_contract(profile,mode,sha(ROOT/'data/metadata/EXP-018/paired-plan-v1/plan.json'),paired['files']['epochs.jsonl'])
        hp=run/mode/'epochs.jsonl'; sources[hp.relative_to(ROOT).as_posix()]=sha(hp)
        history=[json.loads(s) for s in hp.read_text(encoding='utf-8').splitlines()]
        require(len(history)==20,'Missing epochs')
        for ep,record in enumerate(history):
            order=[cid for batch in epochs[ep]['batches'] for cid in batch['clip_ids']]
            require(record['completed_epochs']==ep+1 and record['samples']==504
                    and record['optimizer_steps']==21 and record['total_updates']==21*(ep+1)
                    and record['train_clip_ids']==order,'Changed endpoint order or update budget')
            require(record.get('train_pair_count')==80 and record.get('objective_mode')==mode
                    and record.get('rank_lambda')==profile['rank_lambda'] and record.get('rank_scale')==21/80,
                    'Changed epoch pair objective')
            predictions=read(run/mode/f'validation-epoch-{ep+1:02d}.json')
            require(len(predictions)==136 and {p['clip_id'] for p in predictions}==set(expected),'Missing/duplicated validation')
            require(all(p['label']==expected[p['clip_id']] and math.isfinite(p['logit']) for p in predictions),'Invalid labels/logits')
            recomputed=sum(bce(p['logit'],p['label']) for p in predictions)/136
            near(recomputed,record['validation_unweighted_bce'],'Saved logits do not reproduce logged BCE',2e-6)
            require(math.isfinite(record['train_loss']) and 0<record['grad_norm_mean']<float('inf'),'Invalid training trace')
        selected=min(range(20),key=lambda e:history[e]['validation_unweighted_bce'])+1
        require(result['best_epoch']==selected,'Changed checkpoint selection')
        selected_file=result['selected_checkpoint']
        require(selected_file['filename']=='best.pt' and selected_file['completed_epochs']==selected
                and sha(run/mode/'best.pt')==selected_file['sha256'],'Selected weight mismatch')
        selected_shas[mode]=selected_file['sha256']
        predictions=read(run/mode/f'validation-epoch-{selected:02d}.json')
        clip_rows=[dict(unit_id=p['clip_id'],label=p['label'],score=sigmoid(p['logit'])) for p in predictions]
        by_video={}
        for uid in ('v1_122','v1_63','v1_82'):
            ps=[p['logit'] for p in predictions if p['label'] and p['clip_id'].startswith(uid+':')]
            ns=[p['logit'] for p in predictions if not p['label'] and p['clip_id'].startswith(uid+':')]
            require(len(ps)==len(ns)==4,'Mixed-video window identity changed')
            wrong=sum((a<b)+.5*(a==b) for a in ps for b in ns)
            by_video[uid]=dict(P=4,N=4,inversion_pairs=wrong,total_pairs=16,rate=wrong/16)
        values=[r['score'] for r in clip_rows]
        summaries[mode]=dict(best_epoch=selected,clip_bce_fp64=sum(bce(p['logit'],p['label']) for p in predictions)/136,
            clip_metrics=metrics(clip_rows),probability_range=max(values)-min(values),
            mixed_video_inversion_macro=sum(x['rate'] for x in by_video.values())/3,mixed_videos=by_video)
    for field in ('initial_state_sha256','seed','effective_batch','microbatch','precision','checkpoint_backbone',
                  'requested_epochs','requested_optimizer_updates'):
        require(all(p[field]==profiles[0][field] for p in profiles),'Unmatched three-condition profile '+field)
    dense=read(run/'three-dense/result.json')
    require(dense['status']=='dense_evaluation_complete' and set(dense['metrics'])==set(MODES),'Incomplete dense result')
    dense_model_contract(dense,pipeline,selected_shas)
    source=ROOT/'data/metadata/EXP-016/metadata-v4'
    with (source/'video-inputs.csv').open(encoding='utf-8-sig',newline='') as f:
        videos=[r for r in csv.DictReader(f) if r['split']=='validation']
    expected_windows=defaultdict(list)
    with (source/'dense-eval-windows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);expected_windows[r['unit_id']].append((r['source_pts_seconds'][0],r['source_pts_seconds'][-1]))
    maximum={m:defaultdict(lambda:None) for m in MODES}; counts=defaultdict(int)
    wp=run/'three-dense/window-scores.csv';sources[wp.relative_to(ROOT).as_posix()]=sha(wp)
    with wp.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(reader.fieldnames==['unit_id','start','end',*MODES],'Missing, duplicate or extra score columns')
        for r in reader:
            require(set(r)==set(reader.fieldnames) and all(value is not None for value in r.values()),'Malformed score row')
            uid=r['unit_id']; i=counts[uid]
            require(uid in expected_windows and i<len(expected_windows[uid]),'Extra dense window')
            start,end=expected_windows[uid][i]
            near(float(r['start']),start,'Changed dense start',1e-8);near(float(r['end']),end,'Changed dense end',1e-8)
            counts[uid]+=1
            for m in MODES:
                score=float(r[m]);require(math.isfinite(score) and 0<=score<=1,'Nonfinite/out-of-range dense score')
                old=maximum[m][uid];maximum[m][uid]=score if old is None else max(old,score)
    require(len(counts)==31 and sum(counts.values())==32567,'Incomplete full-video coverage')
    for uid,windows in expected_windows.items():require(counts[uid]==len(windows),'Missing dense window')
    reference=read(ROOT/'data/process/EXP-016/dense-pixel-verification-v3/result.json')
    for uid in counts:
        require(dense['videos'][uid]['rgb112_digest']==reference['videos'][uid]['rgb112_digest'],'Changed dense pixels')
    for m in MODES:
        video_rows=[dict(unit_id=v['unit_id'],label=int(v['video_label']=='pos'),score=maximum[m][v['unit_id']]) for v in videos]
        stored=dense['scores'][m]
        require(len(stored)==31 and len({r['unit_id'] for r in stored})==31,'Duplicate/missing video identity')
        stored={r['unit_id']:r for r in stored}
        require(set(stored)=={v['unit_id'] for v in videos},'Stored video identities changed')
        for v in videos:
            r=stored[v['unit_id']]
            require(r['label']==int(v['video_label']=='pos') and r['group_id']==v['group_id']
                    and r['windows']==counts[v['unit_id']],'Stored video truth/group/coverage changed')
            near(r['score'],maximum[m][v['unit_id']],'Stored max differs from full score CSV',1e-12)
        computed=metrics(video_rows)
        for k,value in computed.items():near(value,dense['metrics'][m][k],'Dense metrics disagree '+m+'/'+k,1e-10)
        summaries[m]['video_metrics']=computed
    a,b,c=[summaries[m] for m in MODES]
    checks=dict(baseline_not_degenerate=a['clip_metrics']['AUC']>.5 and a['probability_range']>=.05,
        mixed_inversion_improves_both=all(c['mixed_video_inversion_macro']<=x['mixed_video_inversion_macro']-.05+1e-12 for x in (a,b)),
        bce_no_worse_than_both=all(c['clip_bce_fp64']<=x['clip_bce_fp64']+1e-8 for x in (a,b)),
        video_fp_fn_no_worse_than_both=all(c['video_metrics'][k]<=x['video_metrics'][k] for k in ('FP','FN') for x in (a,b)))
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='audit_passed',scope='three-condition developmental pilot; no independent test or full CV',
        summaries=summaries,registered_gate=checks,advance_condition_met=all(checks.values()),
        another_run_automatically_authorized=False,source_sha256=sources,
        protocol_status='section6 EXP-018 shared quota exception; section10 full dual CV incomplete')
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'generator-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=report['status'],registered_gate=checks,advance=report['advance_condition_met'])))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();audit(a.run,a.out)
