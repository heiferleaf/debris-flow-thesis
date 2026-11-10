"""EXP-019 shared canonical decoding, branch-specific T order, gated acceptance.

Canonical RGB112 digests are accumulated before permutation. Scores are provisional
until each video's final digest closes against the existing label-free reference.
No second decoder pass or extra up-front full-video pixel scan is introduced.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import torch

from experiments.common.video.dense_evaluation import (
    compatible_pixel_reference, dense_batches, load_selected_checkpoint, video_metrics,
)
from experiments.common.video.order_training import MODES, order_records, score_order_batch, verify_cache_acceptance
from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32
from experiments.common.video.supervised_plan import csv_rows, digest, rooted, verify_plan, write_json
from experiments.common.video.supervised_runtime import ClipInput


def _score_stream(batches, info, expected_digest, models, on_window=None):
    """Score canonical batches; return no accepted maxima if digest closure fails.

    Finite CPU fixtures inject this generator, never a video decoder or checkpoint.
    Metadata contains only window provenance, never labels for input exclusion.
    """
    if set(models) != set(MODES):
        raise ValueError('Require exactly the two registered order branches')
    maxima = {mode: None for mode in MODES}
    seen = set()
    count = 0
    for u8, metadata in batches:
        canonical = torch.from_numpy(u8) if isinstance(u8, np.ndarray) else u8
        records = order_records(metadata)
        results = {}
        for mode, (net, transform) in models.items():
            logits, current = score_order_batch(net, canonical, transform, metadata, mode)
            if current != records:
                raise ValueError('Supervised/dense physical order record mismatch')
            values = torch.sigmoid(logits).numpy()
            if len(values) != len(metadata) or not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite/invalid dense score')
            results[mode] = values
        for i, (row, record) in enumerate(zip(metadata, records)):
            key = record['physical_window_sha256']
            if key in seen:
                raise ValueError('Duplicate physical dense window')
            seen.add(key)
            output = dict(unit_id=row['unit_id'], start=row['source_pts_seconds'][0],
                          end=row['source_pts_seconds'][-1], physical_window_sha256=key)
            for mode in MODES:
                value = float(results[mode][i])
                output[mode] = value
                maxima[mode] = value if maxima[mode] is None else max(maxima[mode], value)
            if on_window:
                on_window(output, dict(unit_id=row['unit_id'], frame_indices=row['frame_indices'],
                                       source_pts_seconds=row['source_pts_seconds'], **record))
            count += 1
    # Aggregate equality is knowable only after the shared canonical stream ends.
    if info.get('rgb112_digest') != expected_digest or info.get('windows') != count or not count:
        raise ValueError('Canonical dense pixel/reference closure failed; provisional scores withheld')
    return dict(maxima=maxima, accepted_windows=count, input_order_windows=len(seen))


def evaluate(root, plan_dir, out, ffmpeg, checkpoints, *, pixel_reference, cache_acceptance, device='cuda'):
    from experiments.common.video.order_plan import load_order_plan
    root, plan_dir, out = (Path(p).resolve() for p in (root, plan_dir, out))
    if out.exists(): raise FileExistsError(out)
    if set(checkpoints) != set(MODES): raise ValueError('Need both completed order branches')
    plan, bank, epochs, validation, dense_rows, sources = load_order_plan(root, plan_dir)
    acceptance_path,acceptance=verify_cache_acceptance(root,plan_dir,plan,bank,cache_acceptance)
    plan_sha = digest(plan_dir/'plan.json')
    base_dir = rooted(root, plan['base_plan_path'])
    base = verify_plan(root, base_dir)
    resource = root/base['resource']
    video_ids = {bank[c]['unit_id'] for c in validation}
    videos = [sources[uid] for uid in sorted(video_ids)]
    grouped = defaultdict(list)
    for row in dense_rows: grouped[row['unit_id']].append(row)
    if len(videos)!=31 or set(grouped)!=video_ids or sum(map(len,grouped.values()))!=32567:
        raise ValueError('The original31-video/32567-window dense grid changed')
    for ref in csv_rows(resource/'dense-transfer-references.csv'):
        path=root/ref['local_path']
        if path.stat().st_size!=int(ref['size_bytes']) or digest(path)!=ref['sha256']:
            raise ValueError('Frozen dense source/coordinate/QC fingerprint changed')
    reference_path=Path(pixel_reference).resolve()
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    compatible_pixel_reference(base,digest(base_dir/'plan.json'),reference,reference_path.parent/'source-plan.json')
    if set(reference['videos'])!=video_ids:raise ValueError('Pixel reference coverage differs')
    known=defaultdict(dict)
    index_path=next(root/r['path'] for r in base['sources'] if r['path'].endswith('verified-cache-files.csv'))
    for row in csv_rows(index_path):
        if row['unit_id'] in video_ids:known[row['unit_id']][int(row['frame_index'])]=row
    names=['order_dense_evaluation.py','order_training.py','temporal_order.py','order_plan.py',
           'dense_evaluation.py','supervised_runtime.py','pts_clip_sampling.py','r2plus1d_classifier.py']
    fingerprints={('experiments/common/video/'+name):digest(root/'experiments/common/video'/name) for name in names}
    for path in plan_dir.iterdir():
        if path.is_file():fingerprints[path.relative_to(root).as_posix()]=digest(path)
    fingerprints[reference_path.relative_to(root).as_posix()]=digest(reference_path)
    fingerprints[acceptance_path.relative_to(root).as_posix()]=digest(acceptance_path)
    configure_fp32(7);cv2.setNumThreads(1);torch.set_num_threads(4)
    device=torch.device(device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested evaluation device unavailable')
    models,loaded_hashes,profiles={},{},[]
    for mode in MODES:
        checkpoint,loaded_hashes[mode]=load_selected_checkpoint(checkpoints[mode],plan_sha,'rgb_zero')
        profile=checkpoint['profile']
        if (profile.get('spec_id')!='EXP-019' or profile.get('order_mode')!=mode or profile.get('fixture')
                or profile.get('permutation_seed')!=1901 or profile.get('seed')!=7
                or profile.get('requested_epochs')!=20 or profile.get('requested_optimizer_updates')!=420):
            raise ValueError('Checkpoint is not a completed registered order condition')
        for key in ('base_plan_sha256','paired_plan_sha256'):
            if profile[key]!=plan[key]:raise ValueError('Checkpoint frozen data source differs')
        net=VideoClassifier(pretrained=False).to(device).eval();net.load_state_dict(checkpoint['state_dict'])
        models[mode]=(net,ClipInput('rgb_zero').to(device));profiles.append(profile)
    for key in ('initial_state_sha256','seed','effective_batch','microbatch','workers','loader_generator_seed',
                'cpu_threads','precision','base_plan_sha256','paired_plan_sha256','paired_schedule_sha256','cache_acceptance_sha256'):
        if profiles[0][key]!=profiles[1][key]:raise ValueError('Unmatched order branches: '+key)
    if profiles[0]['cache_acceptance_sha256']!=digest(acceptance_path):
        raise ValueError('Training and dense acceptance proofs differ')
    out.mkdir(parents=True,exist_ok=False)
    evidence,scores={}, {mode:[] for mode in MODES}
    began=time.perf_counter()
    try:
        with (out/'window-scores.csv').open('x',encoding='utf-8-sig',newline='') as f, (out/'window-orders.jsonl').open('x',encoding='utf-8') as mapping:
            writer=csv.DictWriter(f,fieldnames=['unit_id','start','end','physical_window_sha256',*MODES]);writer.writeheader()
            def record(score,order):
                writer.writerow(score);mapping.write(json.dumps(order,ensure_ascii=True)+'\n')
            for video in videos:
                uid=video['unit_id'];info={}
                pts=np.array([float(r['pts_time']) for r in csv_rows(root/video['coord_path'])])
                stream=dense_batches(root/video['middleware_path'],pts,grouped[uid],ffmpeg,info,known[uid],filter_path=out/(uid+'-filter.txt'))
                closed=_score_stream(stream,info,reference['videos'][uid]['rgb112_digest'],models,record)
                evidence[uid]=dict(info,canonical_reference_matched=True,**closed)
                # Only after canonical closure is the true video label consulted.
                for mode in MODES:
                    scores[mode].append(dict(unit_id=uid,label=int(video['video_label']=='pos'),
                        score=closed['maxima'][mode],group_id=video['group_id'],windows=closed['accepted_windows']))
        if len(evidence)!=31 or sum(v['accepted_windows'] for v in evidence.values())!=32567:
            raise ValueError('Incomplete dense coverage; metrics withheld')
        for name,fingerprint in fingerprints.items():
            if digest(root/name)!=fingerprint:raise RuntimeError('Evaluation source/reference changed')
        for mode,path in checkpoints.items():
            if digest(path)!=loaded_hashes[mode]:raise ValueError('Checkpoint changed during evaluation')
        result=dict(status='dense_evaluation_complete',spec_id='EXP-019',plan_sha256=plan_sha,
            videos=evidence,scores=scores,metrics={mode:video_metrics(r) for mode,r in scores.items()},
            metrics_by_group={mode:{g:video_metrics([r for r in rows if r['group_id']==g]) for g in sorted({r['group_id'] for r in rows})} for mode,rows in scores.items()},
            checkpoint_sha256=loaded_hashes,pixel_reference_sha256=digest(reference_path),code_and_input_sha256=fingerprints,
            input_order_mapping_sha256=digest(out/'window-orders.jsonl'),window_scores_sha256=digest(out/'window-scores.csv'),
            canonical_digest_timing='single-pass canonical stream before gather; provisional inference; final per-video closure before acceptance',
            independent_test_used=False,scope='reused development validation, not independent generalization',
            seconds=time.perf_counter()-began,environment=dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,platform=platform.platform(),device=str(device),precision='FP32'))
        write_json(out/'result.json',result)
        return result
    except Exception as error:
        write_json(out/'failure.json',dict(status='failed_metrics_withheld',accepted_videos=list(evidence),
            provisional_window_scores_not_a_completed_result=True,error_type=type(error).__name__,error=str(error)))
        raise
