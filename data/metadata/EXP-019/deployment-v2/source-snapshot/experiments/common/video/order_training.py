"""EXP-019: EXP-018 ordinary-BCE cadence with an input T-axis gather only.

Canonical physical windows/PTS/QC stay untouched. No old implementation is
patched. This module does not authorize paid execution or rent cloud hardware.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.common.video.paired_runtime import train_paired_batch
from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
from experiments.common.video.supervised_plan import digest, jsonl, rooted, write_json
from experiments.common.video.supervised_runtime import ClipDataset, ClipInput, EpochBatches, init_worker, score_clips
from experiments.common.video.supervised_training import _environment, _save_checkpoint, load_inputs

MODES = ('chronological', 'fixed_permutation')


def order_records(windows):
    from experiments.common.video.temporal_order import window_order_record
    records = []
    for w in windows:
        record = window_order_record(w['media_sha256'], w['frame_indices'])
        if w.get('order_record') is not None and w['order_record'] != record:
            raise ValueError('Frozen order record differs from canonical physical window')
        records.append(record)
    return records


def ordered_batch(u8, windows, mode):
    """Validate physical keys, then gather T only; never change metadata or RNG."""
    from experiments.common.video.temporal_order import temporal_indices
    if mode not in MODES or u8.dtype != torch.uint8 or u8.ndim != 5 or u8.shape[1] != 16 or u8.shape[-1] != 3 or len(u8) != len(windows):
        raise ValueError('Expected canonical uint8 [N,16,H,W,3] and matching windows')
    records = order_records(windows)
    permutations = [temporal_indices(w['media_sha256'], w['frame_indices'], mode) for w in windows]
    if mode == 'chronological':
        if any(tuple(p) != tuple(range(16)) for p in permutations):
            raise ValueError('Chronological identity permutation changed')
        return u8, records
    index = torch.tensor(permutations, dtype=torch.long, device=u8.device)
    index = index[:, :, None, None, None].expand_as(u8)
    return torch.gather(u8, 1, index), records


def train_order_batch(net, optimizer, u8, labels, transform, pos_weight, windows, mode):
    ordered, records = ordered_batch(u8, windows, mode)
    # Empty ranking endpoints change no forward/backward/RNG relative to lambda0;
    # EXP-018's auxiliary pair reductions consumed no RNG and had zero coefficient.
    result = train_paired_batch(net, optimizer, ordered, labels, transform, pos_weight,
                                (), microbatch=8, rank_lambda=0., rank_scale=21 / 80)
    return result, records


def score_order_batch(net, u8, transform, windows, mode):
    ordered, records = ordered_batch(u8, windows, mode)
    return score_clips(net, ordered, transform, microbatch=8), records


def _rng_digest():
    result = dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
    if torch.cuda.is_available():
        result['cuda'] = [hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()]
    return result


def _finite_state(net):
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in net.state_dict().values()):
        raise FloatingPointError('Nonfinite model parameter/buffer')


def check_initial_state(initial_sha, plan, mode, paired_initial_sha=None, *, test_only=False):
    """Production A/B always bind the frozen K400 state; B also binds measured A."""
    if mode not in MODES:raise ValueError('Unknown input order')
    if not test_only:
        frozen=plan.get('expected_initial_state_sha256')
        if (not isinstance(frozen,str) or len(frozen)!=64
                or any(c not in '0123456789abcdef' for c in frozen)
                or initial_sha!=frozen):
            raise ValueError('Initialization differs from the frozen plan state SHA')
        if mode=='fixed_permutation' and paired_initial_sha is None:
            raise ValueError('B also requires the measured A initialization SHA')
    if paired_initial_sha is not None and initial_sha!=paired_initial_sha:
        raise ValueError('Order branches did not restart from the same measured initial state')


def _execution_sources(root):
    names = ['order_training.py', 'temporal_order.py', 'order_plan.py', 'paired_plan.py', 'paired_runtime.py',
             'supervised_runtime.py', 'supervised_training.py', 'supervised_plan.py',
             'r2plus1d_classifier.py', 'pts_clip_sampling.py']
    paths = [root / 'experiments/common/video' / name for name in names]
    paths += [root / 'docs/process/experiments/EXP-019.md',
              root / 'docs/process/experiments/EXP-006-数据使用方案.md',
              root / '.agents/skills/data-usage-protocol/SKILL.md']
    for name in ('train.py','pipeline.py','run.sh','prepare.py','verify_cache.py','prepare_transfer.py'):
        entry=root/'experiments/runspecs/EXP-019'/name
        if entry.exists():paths.append(entry)
    return {p.relative_to(root).as_posix(): digest(p) for p in paths}


def verify_cache_acceptance(root, plan_dir, plan, bank, acceptance_path):
    """Read a separate actual-pixel acceptance proof; never promote metadata flags.

    Required producer contract: status=order_supervised_cache_ready, bound
    order_plan_sha256, verified_cache_index + verified_cache_index_sha256,
    required_frame_count = verified_required_frame_count, failed_required_frames=0.
    The independent producer must actually verify JPEG and RGB112 fingerprints;
    this consumer never creates or infers that proof from an existing index.
    """
    if acceptance_path is None:
        raise ValueError('Separate actual JPEG/RGB112 acceptance is required; metadata flags are not proof')
    path=Path(acceptance_path).resolve();path.relative_to(root)
    record=json.loads(path.read_text(encoding='utf-8'))
    required={(w['unit_id'],i) for w in bank.values() for i in w['frame_indices']}
    if (record.get('status')!='order_supervised_cache_ready'
            or record.get('order_plan_sha256')!=digest(plan_dir/'plan.json')
            or type(record.get('required_frame_count')) is not int
            or type(record.get('verified_required_frame_count')) is not int
            or type(record.get('failed_required_frames')) is not int
            or record['required_frame_count']!=len(required)
            or record['verified_required_frame_count']!=len(required)
            or record.get('failed_required_frames')!=0):
        raise ValueError('Cache acceptance is incomplete or bound to another order plan')
    refs={r['path']:r['sha256'] for r in plan['data_fingerprints']}
    index=record['verified_cache_index'];expected=record['verified_cache_index_sha256']
    if refs.get(index)!=expected or digest(rooted(root,index))!=expected:
        raise ValueError('Accepted cache index differs from the frozen original data')
    return path,record


def _load_contract(root, plan_dir, cache_acceptance):
    from experiments.common.video.order_plan import load_order_plan
    plan, bank, epochs, validation, dense_rows, sources = load_order_plan(root, plan_dir)
    acceptance_path,acceptance=verify_cache_acceptance(root,plan_dir,plan,bank,cache_acceptance)
    plan=dict(plan,cache_acceptance_path=acceptance_path.relative_to(root).as_posix(),
              cache_acceptance_sha256=digest(acceptance_path))
    base_dir = rooted(root, plan['base_plan_path'])
    base, rows, original_schedule, old_validation = load_inputs(root, base_dir, check_files=True)
    by_id = {r[0]: i for i, r in enumerate(rows)}
    canonical = {r['clip_id']: r for r in jsonl(root / base['resource'] / 'windows.jsonl')}
    if len(epochs) != 20 or len(validation) != 136 or validation != [rows[i][0] for i in old_validation]:
        raise ValueError('Frozen validation identity/order or twenty-epoch schedule changed')
    for ep, row in enumerate(epochs):
        ids = [cid for b in row['batches'] for cid in b['clip_ids']]
        if (row['epoch'] != ep or len(row['batches']) != 21 or any(len(b['clip_ids']) != 24 for b in row['batches'])
                or len(ids) != 504 or len(set(ids)) != 504
                or set(ids) != {rows[i][0] for i in original_schedule[ep]}):
            raise ValueError('Original EXP-018 batch exposure/order contract differs')
    for cid in by_id:
        w, old = bank[cid], canonical[cid]
        if (rows[by_id[cid]][2] != int(w['kind'] == 'pos')
                or any(w[k] != old[k] for k in ('unit_id','group_id','kind','frame_indices','source_pts_seconds'))):
            raise ValueError('Order plan altered canonical bank labels/frames/PTS')
    order_records([bank[cid] for cid in by_id])
    return plan, base, rows, epochs, old_validation, bank


def _fit(root, out, mode, plan, base, rows, epochs, validation, bank, fingerprints,
         device, workers, cache_mib, *, expected_initial_state_sha256=None, _test_only=None, _probe=False):
    """Internal loop. _test_only is CPU fixture injection, never a CLI/config key."""
    if mode not in MODES or out.exists():
        raise ValueError('Unknown input order or existing output directory')
    if type(workers) is not int or workers < 0 or type(cache_mib) is not int or cache_mib < 0:
        raise ValueError('Invalid loader/cache configuration')
    fixture = _test_only is not None
    if type(_probe) is not bool:raise ValueError('Probe selector must be explicit bool')
    actual_epochs=1 if _probe else 20
    device = torch.device(device)
    if fixture and (device.type != 'cpu' or workers != 0):
        raise ValueError('Test-only injection is CPU/no-worker only')
    if not fixture and (workers != 2 or len(epochs) != 20):
        raise ValueError('EXP-019 freezes workers2 and full twenty epochs')
    if len(epochs) != 20 or len(validation) != 136:
        raise ValueError('Require twenty full epochs and136 fixed validation clips')
    by_id = {r[0]: i for i, r in enumerate(rows)}
    if len(by_id) != len(rows): raise ValueError('Duplicate dataset clip')
    def check():
        for path, fingerprint in fingerprints.items():
            if digest(root / path) != fingerprint:
                raise RuntimeError('Frozen execution input/source changed: ' + path)
    check()
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device.type not in ('cpu','cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested device unavailable; no fallback')
    configure_fp32(7)
    net = (_test_only['model_factory'](pretrained=True) if fixture else VideoClassifier(pretrained=True)).to(device)
    transform = ClipInput('rgb_zero').to(device)
    initial_sha = state_digest(net)
    check_initial_state(initial_sha,plan,mode,expected_initial_state_sha256,test_only=fixture)
    optimizer = torch.optim.Adam([dict(params=net.backbone.parameters(), lr=1e-4),
                                  dict(params=net.head.parameters(), lr=1e-3)],
                                 betas=(.9,.999), eps=1e-8, weight_decay=0, amsgrad=False)
    if optimizer.state: raise ValueError('Adam must start empty')
    _finite_state(net)
    profile = dict(format_version=1, spec_id='EXP-019', order_mode=mode, mode='rgb_zero',
        objective_mode='bce', seed=7, permutation_seed=1901, initial_state_sha256=initial_sha,
        expected_paired_initial_state_sha256=expected_initial_state_sha256,
        frozen_plan_initial_state_sha256=plan.get('expected_initial_state_sha256'),
        old_initialization_reference='c1847e1c0987ff003bff99805ab795af8eb59b07db210c7999e377b337ee9b81',
        plan_sha256=plan['plan_sha256'], base_plan_sha256=plan['base_plan_sha256'],
        paired_plan_sha256=plan['paired_plan_sha256'], paired_schedule_sha256=plan['paired_epochs_sha256'],
        cache_acceptance_sha256=plan.get('cache_acceptance_sha256'),
        pretrained='injected_tiny_cpu_fixture' if fixture else 'KINETICS400_V1',
        fixture=fixture, smoke=_probe, architecture='r2plus1d_18_six_channel', effective_batch=24,
        microbatch=8, workers=workers, persistent_workers=workers>0, prefetch_factor=2 if workers else None,
        loader_generator_seed=100007, cpu_threads=torch.get_num_threads(), cache_mib_per_worker=cache_mib,
        checkpoint_backbone=False, supervised_pos_weight=base['pos_weight'], rank_lambda=0., rank_scale=21/80,
        requested_epochs=actual_epochs, requested_optimizer_updates=21*actual_epochs, precision='FP32',
        aggregation='max',score_threshold=.5,clip_len=16,span_seconds=3.,inference_stride_seconds=1.5,
        input_pipeline='canonical JPEG95/RGB112 -> T-axis order -> MEAN_STD/RGB+zeroD',
        execution_fingerprints=fingerprints,
        environment=(_environment(device) if not fixture else
                     dict(torch=torch.__version__,opencv=cv2.__version__,device=str(device),
                          amp=False,tf32_matmul=False,tf32_cudnn=False,cudnn_deterministic=True)))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'profile.json',profile)
    with (out/'input-order-mapping.jsonl').open('x',encoding='utf-8') as f:
        for cid in sorted(by_id):
            f.write(json.dumps(dict(clip_id=cid,order_mode=mode,**order_records([bank[cid]])[0]),ensure_ascii=True)+'\n')
    ds = _test_only['dataset'] if fixture else ClipDataset(rows,cache_bytes=cache_mib*1024**2)
    sampler=EpochBatches(24)
    loader=DataLoader(ds,batch_sampler=sampler,num_workers=workers,persistent_workers=workers>0,
        pin_memory=device.type=='cuda',worker_init_fn=init_worker,generator=torch.Generator().manual_seed(100007),
        **({'prefetch_factor':2} if workers else {}))
    initial_bn={n:v.detach().cpu().clone() for n,v in net.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    best_loss,best_epoch,updates=float('inf'),None,0
    began=time.perf_counter()
    try:
        with (out/'epochs.jsonl').open('x',encoding='utf-8') as history:
            for epoch in range(actual_epochs):
                started=time.perf_counter(); rng_before=_rng_digest()
                specs=epochs[epoch]['batches'];order=[by_id[c] for b in specs for c in b['clip_ids']]
                if len(specs)!=21 or any(len(b['clip_ids'])!=24 for b in specs) or len(order)!=504 or len(set(order))!=504:
                    raise ValueError('EXP-018 physical batch schedule changed')
                sampler.set_order(order);seen=[];count=steps=0;bce_sum=objective_sum=0.;grads=[]
                for batch_index,(u8,labels,selected) in enumerate(loader):
                    ids=[rows[int(i)][0] for i in selected]
                    if ids!=specs[batch_index]['clip_ids']:raise ValueError('Frozen batch order changed')
                    net.train()
                    measured,_=train_order_batch(net,optimizer,u8,labels,transform,base['pos_weight'],[bank[c] for c in ids],mode)
                    _finite_state(net)
                    count+=measured['n'];steps+=1;updates+=1;bce_sum+=measured['bce_sum'];objective_sum+=measured['loss']
                    seen.extend(map(int,selected));grads.append(measured['grad_norm'])
                if seen!=order or count!=504 or steps!=21:raise ValueError('Dropped/reordered samples or updates')
                sampler.set_order(validation);val_sum=0.;val_count=0;predictions=[];val_seen=[]
                class_sums,class_counts={0:0.,1:0.},{0:0,1:0}
                for u8,labels,selected in loader:
                    ids=[rows[int(i)][0] for i in selected]
                    logits,_=score_order_batch(net,u8,transform,[bank[c] for c in ids],mode)
                    losses=nn.functional.binary_cross_entropy_with_logits(logits,labels.float(),reduction='none')
                    val_sum+=float(losses.sum());val_count+=len(labels);val_seen.extend(map(int,selected))
                    for label in (0,1):
                        class_sums[label]+=float(losses[labels==label].sum());class_counts[label]+=int((labels==label).sum())
                    predictions.extend(dict(clip_id=rows[int(i)][0],label=int(y),logit=float(z)) for i,y,z in zip(selected,labels,logits))
                if val_seen!=validation or val_count!=136 or class_counts!={0:40,1:96}:raise ValueError('Fixed validation order/count changed')
                val_loss=val_sum/val_count
                if not np.isfinite(val_loss) or not np.isfinite(bce_sum) or not np.isfinite(grads).all():raise FloatingPointError('Nonfinite epoch')
                for name,value in net.named_buffers():
                    if name in initial_bn and not torch.equal(value.cpu(),initial_bn[name]):raise RuntimeError('BN statistics changed')
                check()
                if val_loss<best_loss:
                    best_loss,best_epoch=val_loss,epoch+1
                    if not _probe:_save_checkpoint(out/'best.pt',net,optimizer,profile,epoch+1)
                write_json(out/('validation-epoch-%02d.json'%(epoch+1)),predictions)
                record=dict(completed_epochs=epoch+1,train_loss=objective_sum/steps,train_weighted_bce=bce_sum/count,
                    train_bce_sum=bce_sum,validation_unweighted_bce=val_loss,best_epoch=best_epoch,samples=count,
                    optimizer_steps=steps,total_updates=updates,train_clip_ids=[rows[i][0] for i in seen],
                    validation_clip_ids=[rows[i][0] for i in val_seen],validation_P=96,validation_N=40,
                    validation_pos_bce=class_sums[1]/96,validation_neg_bce=class_sums[0]/40,
                    order_mode=mode,rank_lambda=0.,grad_norm_mean=float(np.mean(grads)),grad_norm_max=max(grads),
                    initial_state_sha256=initial_sha,model_state_sha256=state_digest(net),
                    rng_before=rng_before,rng_after=_rng_digest(),seconds=time.perf_counter()-started)
                history.write(json.dumps(record,allow_nan=False)+'\n');history.flush()
        if not _probe:_save_checkpoint(out/'last.pt',net,optimizer,profile,20)
        if updates!=21*actual_epochs:raise ValueError('Update budget changed')
        check()
        result=dict(status='runtime_probe_complete' if _probe else 'supervised_complete_dense_evaluation_pending',completed_epochs=actual_epochs,best_epoch=best_epoch,
            best_validation_bce=best_loss,initial_state_sha256=initial_sha,final_state_sha256=state_digest(net),
            optimizer_steps=updates,profile=profile,independent_test_used=False,
            input_order_mapping_sha256=digest(out/'input-order-mapping.jsonl'),seconds=time.perf_counter()-began)
        if not _probe:
            result['selected_checkpoint']=dict(filename='best.pt',sha256=digest(out/'best.pt'),completed_epochs=best_epoch)
        write_json(out/'training-result.json',result)
        return result
    except Exception as error:
        write_json(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),optimizer_steps=updates))
        raise


def train(root,plan_dir,out,mode,device='cuda',workers=2,cache_mib=128,expected_initial_state_sha256=None,cache_acceptance=None):
    return _production_fit(root,plan_dir,out,mode,device,workers,cache_mib,expected_initial_state_sha256,cache_acceptance,False)


def probe(root,plan_dir,out,mode,device='cuda',workers=2,cache_mib=128,expected_initial_state_sha256=None,cache_acceptance=None):
    """One separate full epoch for timing/correctness; never writes usable weights."""
    return _production_fit(root,plan_dir,out,mode,device,workers,cache_mib,expected_initial_state_sha256,cache_acceptance,True)


def _production_fit(root,plan_dir,out,mode,device,workers,cache_mib,expected_initial_state_sha256,cache_acceptance,is_probe):
    root,plan_dir,out=(Path(p).resolve() for p in (root,plan_dir,out))
    if out.exists():raise FileExistsError(out)
    plan,base,rows,epochs,validation,bank=_load_contract(root,plan_dir,cache_acceptance)
    # Local plan hash is an execution attribute, not a change to the frozen plan.
    plan=dict(plan,plan_sha256=digest(plan_dir/'plan.json'))
    fingerprints=_execution_sources(root)
    fingerprints[plan['cache_acceptance_path']]=plan['cache_acceptance_sha256']
    for path in plan_dir.iterdir():
        if path.is_file():fingerprints[path.relative_to(root).as_posix()]=digest(path)
    return _fit(root,out,mode,plan,base,rows,epochs,validation,bank,fingerprints,
                device,workers,cache_mib,expected_initial_state_sha256=expected_initial_state_sha256,_probe=is_probe)
