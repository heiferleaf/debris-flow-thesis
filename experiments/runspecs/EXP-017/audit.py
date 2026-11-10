"""Offline EXP-017 evidence/metric audit. No model execution or media decoding."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.common.video.cv_execution import INITIAL_SHA,PLAN_SHA,summarize_folds,check_fit_result
from experiments.common.video.cv_supervised_plan import load_cv_plan
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.pts_clip_sampling import tuple_digest
from experiments.common.video.supervised_plan import csv_rows,digest,jsonl,write_json


def bce(logits):
    return math.fsum(max(r['logit'],0)-r['label']*r['logit']+math.log1p(math.exp(-abs(r['logit']))) for r in logits)/len(logits)


def checked_validation_bce(logits,recorded):
    recomputed=bce(logits)
    if (not math.isfinite(recorded) or not math.isfinite(recomputed) or recorded<0 or recomputed<0
            or abs(recomputed-recorded)>1e-5):
        raise ValueError('Inner saved-logit BCE differs or is nonfinite')
    return recomputed


def audit(run,plan_dir,out):
    run,plan_dir,out=(Path(p).resolve() for p in (run,plan_dir,out))
    if out.exists():raise FileExistsError(out)
    if digest(plan_dir/'plan.json')!=PLAN_SHA:raise ValueError('Unregistered plan')
    plan,bank,_,_,folds=load_cv_plan(ROOT,plan_dir)
    sources={r['unit_id']:r for r in csv_rows(plan_dir/'sources.csv')}
    pipeline=json.loads((run/'pipeline-result.json').read_text(encoding='utf-8'))
    if pipeline['status']!='complete_pending_independent_audit':raise ValueError('Incomplete pipeline')
    for name in ('pipeline-failure.json','hard-time-limit.json','hard-stop-shutdown.json','hard-stop-shutdown.log'):
        if (run/name).exists():raise ValueError('Failure/hard-stop receipt prevents complete acceptance')
    frozen=json.loads((run/'execution-inputs.json').read_text(encoding='utf-8'))
    for relative,expected in frozen['code_sha256'].items():
        if digest(run/'source-snapshot'/relative)!=expected:raise ValueError('Execution source snapshot differs')
    for relative,expected in frozen['input_fingerprints'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Frozen execution input changed')
    reference=json.loads((ROOT/frozen['dense_reference_path']).read_text(encoding='utf-8'))
    if digest(ROOT/frozen['dense_reference_path'])!=frozen['dense_reference_sha256'] or digest(ROOT/reference['grid_path'])!=reference['grid_sha256']:
        raise ValueError('Dense reference/grid bytes differ')
    expected_grid={}
    for row in jsonl(ROOT/reference['grid_path']):
        key=(row['unit_id'],tuple_digest(row['frame_indices']))
        if key in expected_grid:raise ValueError('Repeated frozen physical grid window')
        expected_grid[key]=row
    results=[];fit_summaries=[];hashes={};out.mkdir(parents=True,exist_ok=False)
    def consume(path):
        hashes[path.relative_to(ROOT).as_posix()]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    try:
        for f in folds:
            number=f['fold'];directory=run/('fold-%02d'%number)
            inner=consume(directory/'inner/training-result.json');check_fit_result(inner,'inner_select',number,20)
            E=inner['selected_epochs'];outer=consume(directory/'outer/training-result.json');check_fit_result(outer,'fixed_outer',number,E)
            losses=[]
            for stage,role,n_epochs,result in (('inner','inner_train',20,inner),('outer','outer_train',E,outer)):
                profile=consume(directory/stage/'profile.json')
                if (profile['execution_kind']!='registered_fit' or profile['initial_state_sha256']!=INITIAL_SHA
                        or profile['expected_initial_state_sha256']!=INITIAL_SHA or profile['stage']!=('inner_select' if stage=='inner' else 'fixed_outer')
                        or profile['seed']!=7 or profile['fold']!=number):raise ValueError('Unregistered fit profile')
                if profile['input_fingerprints']!=frozen['input_fingerprints']:raise ValueError('Fit uses another execution/input closure')
                history=jsonl(directory/stage/'epochs.jsonl');hashes[(directory/stage/'epochs.jsonl').relative_to(ROOT).as_posix()]=digest(directory/stage/'epochs.jsonl')
                if len(history)!=n_epochs:raise ValueError('Wrong fit length')
                updates=0
                for e,row in enumerate(history):
                    ids=f[role]['epoch_clip_ids'][e];P=sum(bank[c]['label'] for c in ids);N=len(ids)-P
                    updates+=math.ceil(len(ids)/24)
                    if (row['train_clip_ids']!=ids or row['samples']!=len(ids) or row['train_P']!=P or row['train_N']!=N
                            or row['optimizer_steps']!=math.ceil(len(ids)/24) or row['total_updates']!=updates or row['finite'] is not True
                            or not math.isfinite(row['train_weighted_bce']) or row['initial_state_sha256']!=INITIAL_SHA):raise ValueError('Fit schedule/weight/finite record differs')
                    if stage=='inner':
                        pred=consume(directory/stage/('validation-epoch-%02d.json'%(e+1)))
                        if [r['clip_id'] for r in pred]!=f['inner_val']['clip_ids'] or any(r['label']!=bank[r['clip_id']]['label'] or not math.isfinite(r['logit']) for r in pred):raise ValueError('Inner validation identities/labels differ')
                        checked_validation_bce(pred,row['validation_unweighted_bce'])
                        losses.append(row['validation_unweighted_bce'])
                    elif any(k.startswith('validation') for k in row):raise ValueError('Outer fit contains validation selection data')
                if updates!=result['optimizer_steps'] or not math.isclose(profile['supervised_pos_weight'],N/P,abs_tol=1e-12):raise ValueError('Dynamic fit weight/updates differ')
                for key in ('selected_checkpoint','last_checkpoint'):
                    ref=result[key];path=directory/stage/ref['filename']
                    if ref['completed_epochs']!=(E if key=='selected_checkpoint' else n_epochs):raise ValueError('Checkpoint recorded epoch differs')
                    if digest(path)!=ref['sha256']:raise ValueError('Checkpoint bytes changed')
                    hashes[path.relative_to(ROOT).as_posix()]=ref['sha256']
            if min(range(20),key=lambda i:losses[i])+1!=E:raise ValueError('Inner E is not earliest recorded minimum')
            dense=consume(directory/'dense/result.json')
            if (dense['status']!='outer_fold_complete' or dense['fold']!=number or dense['selected_epochs']!=E
                    or dense['checkpoint_sha256']!=outer['selected_checkpoint']['sha256']
                    or {r['unit_id'] for r in dense['scores']}!=set(f['outer_eval']['video_ids'])):raise ValueError('Wrong outer coverage/selected model')
            path=directory/'dense/window-scores.csv'
            if digest(path)!=dense['window_scores_sha256']:raise ValueError('Window score CSV changed')
            maxima={};counts={};seen=set()
            for row in csv_rows(path):
                uid=row['unit_id'];value=float(row['score']);key=(uid,row['frame_tuple_sha256'])
                if key in seen or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Duplicate/nonfinite dense prediction')
                grid=expected_grid[key]
                if (abs(float(row['target_start_seconds'])-grid['target_start_seconds'])>1e-8
                        or abs(float(row['start'])-grid['source_pts_seconds'][0])>1e-8
                        or abs(float(row['end'])-grid['source_pts_seconds'][-1])>1e-8):raise ValueError('Dense prediction not on canonical frozen grid')
                seen.add(key);maxima[uid]=max(maxima.get(uid,0.),value);counts[uid]=counts.get(uid,0)+1
            for row in dense['scores']:
                uid=row['unit_id']
                if (row['score']!=maxima[uid] or row['windows']!=counts[uid] or dense['videos'][uid]['canonical_reference_matched'] is not True
                        or dense['videos'][uid]['rgb112_digest']!=reference['videos'][uid]['rgb112_digest']
                        or row['group_id']!=sources[uid]['group_id'] or row['label']!=int(sources[uid]['video_label']=='pos')):raise ValueError('Dense max/identity/group closure differs')
            if seen!={k for k in expected_grid if k[0] in set(f['outer_eval']['video_ids'])}:raise ValueError('Incomplete canonical outer grid')
            if video_metrics(dense['scores'])!=dense['metrics']:raise ValueError('Fold video metrics differ')
            results.append(dense);fit_summaries.append(dict(fold=number,selected_epochs=E,inner_updates=inner['optimizer_steps'],outer_updates=outer['optimizer_steps']))
        summary=summarize_folds(results,sources)
        if summary!=pipeline['summary']:raise ValueError('Pooled counts/fold-only AUC summary differs')
        answer=dict(status='audit_passed',summary=summary,fits=fit_summaries,source_sha256=hashes,
            audit_source_sha256=digest(Path(__file__)),plan_sha256=PLAN_SHA,no_model_execution=True,no_media_decode=True,
            limitations='Checkpoint bytes hashed, not independently replayed; historically reused development pool; invoices/platform-off state separately verified by operator.')
        write_json(out/'audit.json',answer);return answer
    except Exception as e:
        write_json(out/'failure.json',dict(status='audit_failed',error_type=type(e).__name__,error=str(e)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run','plan','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.run,a.plan,a.out),ensure_ascii=True))


if __name__=='__main__':main()
