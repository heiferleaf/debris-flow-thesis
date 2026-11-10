"""EXP-017 execution contracts; no partition search, decoding or rental APIs."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from experiments.common.video.cv_supervised_plan import load_cv_plan, validate_partitions
from experiments.common.video.cv_supervised_training import fit
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, rooted
from experiments.common.video.supervised_runtime import FrameRef, PIXEL_VERSION

PLAN_SHA='70394126cd15d015c4250bbea8f1c9a0c928141d92f00a03d66279a576c2c9b8'
INITIAL_SHA='c1847e1c0987ff003bff99805ab795af8eb59b07db210c7999e377b337ee9b81'


def _key(row):return row['unit_id'],int(row['frame_index'])


def deployment_records(context,code_paths):
    """Rebuild mandatory deployment identities from accepted inputs, not a CSV.

    Media/JPEG SHA and sizes come from actual producer evidence. Only small
    metadata/code are read here; a paid run must not rescan every media byte.
    """
    root=context['root'];records={}
    def add(path,kind,expected=None,size=None):
        path=Path(path).resolve();name=path.relative_to(root).as_posix()
        row=dict(path=name,kind=kind,sha256=digest(path) if expected is None else expected,
                 size_bytes=path.stat().st_size if size is None else size)
        if type(row['size_bytes']) is not int or row['size_bytes']<0:raise ValueError('Invalid required deployment size')
        previous=records.get(name)
        if previous and any(previous[k]!=row[k] for k in ('sha256','size_bytes')):
            raise ValueError('Conflicting required deployment identity: '+name)
        if previous and previous['kind']=='execution_source':return
        records[name]=row
    for p in code_paths:add(p,'execution_source')
    for p in context['plan_dir'].rglob('*'):
        if p.is_file():add(p,'plan_metadata')
    for name,expected in context['fingerprints'].items():add(root/name,'accepted_input_metadata',expected)
    for row in context['plan']['source_fingerprints']:
        add(root/row['path'],'frozen_source_metadata',row['sha256'],row['size_bytes'])
    if context['dense'] is None:raise ValueError('Full150-video dense reference required for deployment')
    media=context['dense']['reference']['media_refs']
    if len(media)!=150 or {r['unit_id'] for r in media}!=set(context['sources']):
        raise ValueError('Missing/duplicate150 hash-closed media references')
    for row in media:
        source=context['sources'][row['unit_id']]
        if row['path']!=source['middleware_path'] or row['sha256']!=source['middleware_sha256']:
            raise ValueError('Required media differs from frozen source')
        add(root/row['path'],'source_media',row['sha256'],row['size_bytes'])
    for key,ref in context['refs'].items():
        row=context['cache_index'][key]
        if (rooted(root,row['cache_path'])!=Path(ref.path).resolve()
                or row['jpeg_sha256']!=ref.jpeg_sha256 or row['rgb112_sha256']!=ref.rgb112_sha256):
            raise ValueError('Required JPEG differs from accepted FrameRef')
        add(root/row['cache_path'],'supervised_JPEG',row['jpeg_sha256'],int(row['size_bytes']))
    return records


def validate_deployment_manifest(manifest,required):
    """Refuse truncated, extra, duplicate or altered transfer declarations."""
    indexed={r['path']:r for r in manifest}
    if not manifest or len(indexed)!=len(manifest) or set(indexed)!=set(required):
        raise ValueError('Transfer manifest does not cover the exact required deployment')
    for name,expected in required.items():
        actual=indexed[name]
        if (actual['sha256']!=expected['sha256'] or actual['kind']!=expected['kind']
                or int(actual['size_bytes'])!=expected['size_bytes']):
            raise ValueError('Required transfer identity differs: '+name)


def verified_frames(root,plan_dir,acceptance):
    """Consume independently checked bytes/pixels; never infer readiness from paths."""
    acceptance=Path(acceptance).resolve();acceptance.relative_to(root)
    proof=json.loads(acceptance.read_text(encoding='utf-8'))
    expected={_key(r):r for r in csv_rows(plan_dir/'frames-required.csv')}
    if (proof.get('status')!='cv_supervised_cache_ready' or proof.get('spec_id')!='EXP-017'
            or proof.get('plan_sha256')!=digest(plan_dir/'plan.json') or proof.get('pixel_version')!=PIXEL_VERSION
            or proof.get('actual_JPEG_byte_SHA_checked') is not True or proof.get('actual_RGB112_pixel_SHA_checked') is not True
            or any(type(proof.get(k)) is not int or proof[k]!=len(expected) for k in
                   ('required_frame_count','verified_required_frame_count','actual_verified_JPEG_reads'))
            or proof.get('failed_required_frames')!=0 or proof.get('unverified_required_frames')!=0):
        raise ValueError('A complete independently accepted CV pixel proof is required')
    fingerprints={acceptance.relative_to(root).as_posix():digest(acceptance)}
    def bound(path_key,hash_key):
        p=rooted(root,proof[path_key]);fingerprints[proof[path_key]]=proof[hash_key]
        if digest(p)!=proof[hash_key]:raise ValueError('Acceptance child SHA changed: '+path_key)
        return p
    merged=csv_rows(bound('verified_cache_index','verified_cache_index_sha256'))
    actual=csv_rows(bound('verified_required_frames_index','verified_required_frames_index_sha256'))
    bound('producer_result','producer_result_sha256')
    index={_key(r):r for r in merged}
    if len(index)!=len(merged) or len(actual)!=len(expected) or {_key(r) for r in actual}!=set(expected):
        raise ValueError('Duplicate/missing/extra accepted physical frame identities')
    refs={}
    for r in actual:
        k=_key(r);m=index[k];e=expected[k]
        if (r['status']!='verified' or r['rgb112_shape']!='112x112x3' or r['rgb112_dtype']!='uint8'
                or r['actual_jpeg_sha256']!=r['jpeg_sha256'] or r['actual_rgb112_sha256']!=r['rgb112_sha256']
                or any(r[x]!=m[x] for x in ('cache_path','jpeg_sha256','rgb112_sha256'))
                or int(r['size_bytes'])!=int(m['size_bytes'])
                or abs(float(r['source_pts_seconds'])-float(e['source_pts_seconds']))>1e-8):
            raise ValueError('Accepted physical JPEG/RGB record differs from frozen requirement')
        refs[k]=FrameRef(str(rooted(root,r['cache_path'])),r['jpeg_sha256'],r['rgb112_sha256'])
    return proof,refs,index,fingerprints


def load_execution(root,plan_dir,acceptance,dense_reference=None):
    root,plan_dir=Path(root).resolve(),Path(plan_dir).resolve()
    plan_dir.relative_to(root)
    if digest(plan_dir/'plan.json')!=PLAN_SHA:raise ValueError('Unregistered EXP-017 v3 plan')
    plan,bank,epochs,fixed,folds=load_cv_plan(root,plan_dir)
    sources={r['unit_id']:r for r in csv_rows(plan_dir/'sources.csv')}
    if len(sources)!=150 or Counter(r['video_label'] for r in sources.values())!=Counter(pos=113,neg=37) or len(folds)!=5:
        raise ValueError('Incomplete registered development pool/folds')
    held=[]
    for i,f in enumerate(folds,1):
        if f['fold']!=i:raise ValueError('Fold order changed')
        validate_partitions({k:f[k]['video_ids'] for k in ('inner_train','inner_val','outer_train','outer_eval')},sources)
        held.extend(f['outer_eval']['video_ids'])
    if len(held)!=150 or set(held)!=set(sources):raise ValueError('Outer coverage must be150 unique videos')
    proof,refs,index,fingerprints=verified_frames(root,plan_dir,acceptance)
    fingerprints[(plan_dir/'plan.json').relative_to(root).as_posix()]=PLAN_SHA
    for name in ('folds.json','windows.jsonl','epochs.jsonl','fixed-validation.json','frames-required.csv','sources.csv'):
        fingerprints[(plan_dir/name).relative_to(root).as_posix()]=plan['files'][name]
    context=dict(root=root,plan_dir=plan_dir,plan=plan,bank=bank,folds=folds,sources=sources,
        proof=proof,refs=refs,cache_index=index,fingerprints=fingerprints,dense=None)
    if dense_reference is not None:
        path=Path(dense_reference).resolve();path.relative_to(root)
        d=json.loads(path.read_text(encoding='utf-8'))
        if (d.get('status')!='cv_dense_pixels_verified' or d.get('plan_sha256')!=PLAN_SHA
                or d.get('video_count')!=150 or d.get('windows')!=143470 or set(d['videos'])!=set(sources)
                or d['preprocessing']['pixel_version']!=PIXEL_VERSION):raise ValueError('Incomplete150-video label-free dense reference')
        grid=rooted(root,d['grid_path'])
        if digest(grid)!=d['grid_sha256'] or digest(rooted(root,d['preprocessing']['helper_path']))!=d['preprocessing']['helper_sha256']:
            raise ValueError('Dense grid/pixel helper differs')
        windows=jsonl(grid);grouped={uid:[] for uid in sources}
        for row in windows:grouped[row['unit_id']].append(row)
        if len(windows)!=143470 or any(len(grouped[u])!=d['videos'][u]['windows'] for u in sources):raise ValueError('Dense reference/grid count mismatch')
        fingerprints[path.relative_to(root).as_posix()]=digest(path);fingerprints[d['grid_path']]=d['grid_sha256']
        context['dense']=dict(path=path,reference=d,windows=grouped)
    return context


def check_fit_result(result,stage,fold,expected_epochs):
    if (result['status']!='fit_complete' or result['stage']!=stage or result['fold']!=fold
            or result['completed_epochs']!=expected_epochs or result['initial_state_sha256']!=INITIAL_SHA
            or result['fixed_outer_validation_evaluated'] is not False):raise ValueError('Incomplete/unregistered CV fit')
    if stage=='inner_select' and not 1<=result['selected_epochs']<=20:raise ValueError('Invalid inner selected E')
    if stage=='fixed_outer' and (result['selected_epochs']!=expected_epochs or result['selected_checkpoint']['filename']!='selected.pt'):
        raise ValueError('Outer must use fixed E selected checkpoint, never outer best')


def train_stage(context,out,fold,stage,inner_result=None,*,max_epochs=20):
    """Production route only. No model factory, held-out tensors, or stage reshuffle."""
    if type(fold) is not int or not 1<=fold<=5 or stage not in ('inner_select','fixed_outer'):raise ValueError('Invalid CV stage/fold')
    f=context['folds'][fold-1]
    import os
    import torch
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    kwargs=dict(stage=stage,fold=fold,bank=context['bank'],verified_frame_index=context['refs'],
        input_fingerprints=context['fingerprints'],expected_initial_state_sha256=INITIAL_SHA,
        device='cuda',workers=2,cache_mib=128)
    if stage=='inner_select':
        if max_epochs!=20 or inner_result is not None:raise ValueError('Formal inner stage is full20, no continuation')
        kwargs.update(epoch_clip_ids=f['inner_train']['epoch_clip_ids'],validation_clip_ids=f['inner_val']['clip_ids'],max_epochs=20)
        epochs=20
    else:
        if inner_result is None:raise ValueError('Outer requires the actual completed inner result')
        check_fit_result(inner_result,'inner_select',fold,20)
        kwargs.update(epoch_clip_ids=f['outer_train']['epoch_clip_ids'],selected_epochs=inner_result['selected_epochs'],
            expected_initial_state_sha256=inner_result['initial_state_sha256'])
        epochs=inner_result['selected_epochs']
    result=fit(context['root'],out,**kwargs);check_fit_result(result,stage,fold,epochs)
    profile=json.loads((Path(out)/'profile.json').read_text(encoding='utf-8'))
    if profile['execution_kind']!='registered_fit' or profile['initial_state_sha256']!=INITIAL_SHA:raise ValueError('Fixture/unregistered checkpoint')
    return result


def summarize_folds(results,sources):
    """Pool confusion counts only. Never compute an AUC across different models."""
    if len(results)!=5 or [r['fold'] for r in results]!=list(range(1,6)):raise ValueError('Five registered folds required')
    rows=[x for r in results for x in r['scores']]
    if (len(rows)!=150 or len({r['unit_id'] for r in rows})!=150 or {r['unit_id'] for r in rows}!=set(sources)
            or any(r['label']!=int(sources[r['unit_id']]['video_label']=='pos')
                   or r['group_id']!=sources[r['unit_id']]['group_id']
                   or not math.isfinite(r['score']) or not 0<=r['score']<=1 for r in rows)):
        raise ValueError('Incomplete/duplicate outer-held-out coverage')
    p=sum(r['label'] for r in rows);n=len(rows)-p
    tp=sum(r['label']==1 and r['score']>=.5 for r in rows);fp=sum(r['label']==0 and r['score']>=.5 for r in rows)
    if (p,n)!=(113,37):raise ValueError('Pooled label counts changed')
    auc=[r['metrics']['AUC'] for r in results]
    if not all(v is not None and math.isfinite(v) for v in auc):raise ValueError('Invalid per-fold AUC')
    return dict(status='five_fold_development_complete',P=p,N=n,TP=tp,FP=fp,FN=p-tp,TN=n-fp,
        Recall=tp/p,FPR=fp/n,Precision=tp/(tp+fp) if tp+fp else 0.,F1=2*tp/(2*tp+fp+p-tp) if 2*tp+fp+p-tp else 0.,
        per_fold_auc=auc,mean_fold_auc=statistics.mean(auc),sample_std_fold_auc=statistics.stdev(auc),
        pooled_auc=None,pooled_auc_reason='Different outer models; pooled scores are not a single-model AUC.',
        negative_group_metrics={g:video_metrics([r for r in rows if r['group_id']==g]) for g in sorted({r['group_id'] for r in rows if r['label']==0})},
        scope='Session-group development CV; historically reused pool, not unseen-camera or independent external testing')
