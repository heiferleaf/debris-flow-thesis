"""One-time outer-fold RGB evaluation; single streaming decode per held-out video."""
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.common.video.cv_execution import INITIAL_SHA,PLAN_SHA,check_fit_result
from experiments.common.video.dense_evaluation import dense_batches,video_metrics
from experiments.common.video.pts_clip_sampling import tuple_digest
from experiments.common.video.r2plus1d_classifier import VideoClassifier,configure_fp32,state_digest
from experiments.common.video.supervised_plan import csv_rows,digest,write_json
from experiments.common.video.supervised_runtime import ClipInput,score_clips


def score_stream(batches,info,expected_digest,net,transform,on_window=None):
    """No accepted score until the canonical RGB112 stream closes at video EOF."""
    maximum=None;seen=set();count=0
    for pixels,metadata in batches:
        u8=torch.from_numpy(pixels) if isinstance(pixels,np.ndarray) else pixels
        values=torch.sigmoid(score_clips(net,u8,transform,microbatch=8)).cpu().numpy()
        if len(values)!=len(metadata) or not np.isfinite(values).all():raise FloatingPointError('Invalid outer dense score')
        for row,value in zip(metadata,values):
            key=tuple_digest(row['frame_indices'])
            if key in seen:raise ValueError('Repeated physical dense window')
            seen.add(key);value=float(value);maximum=value if maximum is None else max(maximum,value)
            if on_window:on_window(dict(unit_id=row['unit_id'],target_start_seconds=row['target_start_seconds'],
                start=row['source_pts_seconds'][0],end=row['source_pts_seconds'][-1],frame_tuple_sha256=key,score=value))
            count+=1
    if not count or info.get('windows')!=count or info.get('rgb112_digest')!=expected_digest:
        raise ValueError('Canonical pixel/grid closure failed; provisional scores withheld')
    return dict(score=maximum,windows=count,canonical_reference_matched=True)


def evaluate_fold(context,out,fold,outer_directory,selected_epochs,ffmpeg,device='cuda'):
    out,outer_directory=Path(out).resolve(),Path(outer_directory).resolve()
    if out.exists():raise FileExistsError(out)
    if context['dense'] is None:raise ValueError('A full local150-video pixel reference is required')
    f=context['folds'][fold-1];uids=f['outer_eval']['video_ids']
    result=json.loads((outer_directory/'training-result.json').read_text(encoding='utf-8'))
    check_fit_result(result,'fixed_outer',fold,selected_epochs)
    path=outer_directory/'selected.pt'
    checkpoint_sha=digest(path)
    if checkpoint_sha!=result['selected_checkpoint']['sha256']:raise ValueError('Selected fixed-E weight changed')
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    profile=checkpoint['profile']
    if (checkpoint['completed_epochs']!=selected_epochs or profile['fold']!=fold or profile['stage']!='fixed_outer'
            or profile['execution_kind']!='registered_fit' or profile['mode']!='rgb_zero' or profile['seed']!=7
            or profile['requested_epochs']!=selected_epochs or profile['initial_state_sha256']!=INITIAL_SHA
            or profile['input_fingerprints']!=context['fingerprints']):raise ValueError('Unregistered fixed-E outer checkpoint')
    configure_fp32(7);cv2.setNumThreads(1);torch.set_num_threads(4)
    net=VideoClassifier(pretrained=False).to(device).eval();net.load_state_dict(checkpoint['state_dict'])
    if state_digest(net)!=checkpoint['state_sha256'] or state_digest(net)!=result['final_state_sha256']:raise ValueError('Outer model tensor digest differs')
    transform=ClipInput('rgb_zero').to(device);out.mkdir(parents=True,exist_ok=False)
    scores=[];evidence={};started=time.perf_counter()
    try:
        with (out/'window-scores.csv').open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['unit_id','target_start_seconds','start','end','frame_tuple_sha256','score']);writer.writeheader()
            for uid in uids:
                source=context['sources'][uid];info={}
                pts=np.array([float(r['pts_time']) for r in csv_rows(context['root']/source['coord_path'])])
                windows=context['dense']['windows'][uid]
                known={i:r for (u,i),r in context['cache_index'].items() if u==uid}
                batches=dense_batches(context['root']/source['middleware_path'],pts,windows,ffmpeg,info,known,
                    batch_size=8,filter_path=out/(uid+'-filter.txt'))
                closed=score_stream(batches,info,context['dense']['reference']['videos'][uid]['rgb112_digest'],net,transform,writer.writerow)
                evidence[uid]=dict(info,**closed)
                # The label does not select/filter the grid or participate in E selection.
                scores.append(dict(unit_id=uid,group_id=source['group_id'],label=int(source['video_label']=='pos'),
                    score=closed['score'],windows=closed['windows'],fold=fold))
        if set(evidence)!=set(uids) or sum(r['windows'] for r in scores)!=sum(len(context['dense']['windows'][u]) for u in uids):
            raise ValueError('Incomplete outer-fold dense coverage')
        if digest(path)!=checkpoint_sha:raise ValueError('Weight changed during evaluation')
        for relative,expected in context['fingerprints'].items():
            if digest(context['root']/relative)!=expected:raise ValueError('Input changed during evaluation: '+relative)
        result=dict(status='outer_fold_complete',fold=fold,selected_epochs=selected_epochs,plan_sha256=PLAN_SHA,
            scores=scores,metrics=video_metrics(scores),metrics_by_group={g:video_metrics([r for r in scores if r['group_id']==g]) for g in sorted({r['group_id'] for r in scores})},
            videos=evidence,window_scores_sha256=digest(out/'window-scores.csv'),checkpoint_sha256=checkpoint_sha,
            pixel_reference_sha256=digest(context['dense']['path']),input_fingerprints=context['fingerprints'],
            seconds=time.perf_counter()-started,threshold=.5,aggregation='max',outer_selection_use=False,
            canonical_digest_timing='single pass; scores provisional until per-video EOF closure',independent_test_used=False)
        write_json(out/'result.json',result);return result
    except Exception as error:
        write_json(out/'failure.json',dict(status='failed_outer_metrics_withheld',fold=fold,error_type=type(error).__name__,
            error=str(error),provisional_window_scores_not_accepted=True,accepted_videos=list(evidence)))
        raise
