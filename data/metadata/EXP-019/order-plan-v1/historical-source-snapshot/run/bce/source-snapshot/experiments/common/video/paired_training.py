"""EXP-018 common A/B/C execution; only the frozen pair relationship changes."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.common.video.paired_plan import load_pair_plan
from experiments.common.video.paired_runtime import train_paired_batch
from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
from experiments.common.video.supervised_plan import digest, rooted, write_json
from experiments.common.video.supervised_runtime import ClipDataset, ClipInput, EpochBatches, init_worker, score_clips
from experiments.common.video.supervised_training import _environment, _save_checkpoint, load_inputs


def _execution_sources(root):
    names = [
        'experiments/common/video/paired_plan.py', 'experiments/common/video/paired_runtime.py',
        'experiments/common/video/paired_training.py', 'experiments/common/video/supervised_plan.py',
        'experiments/common/video/supervised_runtime.py', 'experiments/common/video/supervised_training.py',
        'experiments/common/video/r2plus1d_classifier.py', 'experiments/common/video/pts_clip_sampling.py',
        'experiments/runspecs/EXP-018/prepare.py', 'experiments/runspecs/EXP-018/train.py',
        'docs/process/experiments/EXP-018.md', 'docs/process/experiments/EXP-006-数据使用方案.md',
        '.agents/skills/data-usage-protocol/SKILL.md',
    ]
    controller = root / 'experiments/runspecs/EXP-018/run.py'
    if controller.is_file():
        names.append(controller.relative_to(root).as_posix())
    sources = {name: digest(root / name) for name in names}
    # Imported code must be the actual selected root checkout, not an earlier copy.
    for obj in (VideoClassifier, ClipInput, train_paired_batch, load_pair_plan, load_inputs):
        actual = Path(sys.modules[obj.__module__].__file__).resolve()
        if actual != rooted(root, actual.relative_to(root).as_posix()):
            raise ValueError('Imported training implementation is outside the selected root')
    return sources


def _check_fingerprints(root, sources, plan_dir, paired_sha, base_dir, base_sha, epochs_sha):
    if any(digest(root / name) != sha for name, sha in sources.items()):
        raise RuntimeError('Training source/card/protocol changed during execution')
    if (digest(plan_dir / 'plan.json') != paired_sha or digest(plan_dir / 'epochs.jsonl') != epochs_sha
            or digest(base_dir / 'plan.json') != base_sha):
        raise RuntimeError('Training/base/pair schedule changed during execution')


def train(root, plan_dir, out, mode, seed=7, epochs=20, device='cuda', workers=2,
          cache_mib=128, microbatch=8, smoke=False):
    root, plan_dir, out = (Path(p).resolve() for p in (root, plan_dir, out))
    if mode not in ('bce', 'cross', 'same') or seed != 7:
        raise ValueError('EXP-018 registers bce/cross/same and seed7 only')
    if (isinstance(epochs, bool) or epochs not in (1, 20) or microbatch != 8
            or not isinstance(workers, int) or isinstance(workers, bool) or workers < 0
            or not isinstance(cache_mib, int) or isinstance(cache_mib, bool) or cache_mib < 0):
        raise ValueError('Registered execution is twenty epochs (one for probe), microbatch8')
    if out.exists():
        raise FileExistsError(out)
    sources = _execution_sources(root)
    pair_sha = digest(plan_dir / 'plan.json')
    pair_plan, epoch_rows = load_pair_plan(root, plan_dir)
    base_dir = rooted(root, pair_plan['base_plan_path'])
    base_sha = pair_plan['base_plan_sha256']
    epochs_sha = pair_plan['files']['epochs.jsonl']
    base, rows, original_schedule, validation = load_inputs(root, base_dir)
    by_id = {r[0]: i for i, r in enumerate(rows)}
    for epoch, spec in enumerate(epoch_rows):
        ids = [c for b in spec['batches'] for c in b['clip_ids']]
        if set(ids) != {rows[i][0] for i in original_schedule[epoch]}:
            raise ValueError('Real FrameRef loader and paired schedule disagree')
    if [rows[i][0] for i in validation] != pair_plan['validation_clip_ids']:
        raise ValueError('Real FrameRef loader and fixed validation disagree')
    _check_fingerprints(root, sources, plan_dir, pair_sha, base_dir, base_sha, epochs_sha)
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA are registered')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    configure_fp32(seed)
    net = VideoClassifier(pretrained=True).to(device)
    transform = ClipInput('rgb_zero').to(device)
    initial_sha = state_digest(net)
    optimizer = torch.optim.Adam([dict(params=net.backbone.parameters(), lr=1e-4),
                                  dict(params=net.head.parameters(), lr=1e-3)],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    actual_epochs = 1 if smoke else epochs
    profile = dict(format_version=1, architecture='r2plus1d_18_six_channel', mode='rgb_zero',
                   objective_mode=mode, pretrained='KINETICS400_V1', seed=seed,
                   initial_state_sha256=initial_sha, plan_sha256=base_sha,
                   base_plan_path=pair_plan['base_plan_path'], base_plan_sha256=base_sha,
                   pair_plan_sha256=pair_sha, pair_schedule_sha256=epochs_sha,
                   aggregation='max', score_threshold=.5, span_seconds=3., clip_len=16,
                   inference_stride_seconds=1.5, supervised_pos_weight=base['pos_weight'],
                   effective_batch=24, microbatch=8, checkpoint_backbone=False,
                   workers=workers, cache_mib_per_worker=cache_mib, model_count=1,
                   precision='FP32', input_pipeline='JPEG95/RGB112/MEAN_STD/signed_difference_v1',
                   rank_lambda=0. if mode == 'bce' else pair_plan['rank_lambda'],
                   rank_scale=pair_plan['rank_scale'], pairs_per_full_epoch=80,
                   code_sha256=sources, environment=_environment(device), smoke=smoke,
                   requested_epochs=actual_epochs,
                   requested_optimizer_updates=1 if smoke else 21 * epochs,
                   independent_test_used=False)
    out.mkdir(parents=True, exist_ok=False)
    for name, sha in sources.items():
        target = out / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        if digest(target) != sha:
            raise RuntimeError('Source changed while taking execution snapshot')
    shutil.copyfile(plan_dir / 'plan.json', out / 'paired-plan.json')
    shutil.copyfile(plan_dir / 'epochs.jsonl', out / 'paired-epochs.jsonl')
    write_json(out / 'profile.json', profile)
    write_json(out / 'provenance.json', dict(argv=sys.argv, code=sources,
               base_plan_sha256=base_sha, pair_plan_sha256=pair_sha, pair_schedule_sha256=epochs_sha))
    ds = ClipDataset(rows, cache_bytes=cache_mib * 1024**2)
    sampler = EpochBatches(24)
    generator = torch.Generator().manual_seed(seed + 100000)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device.type == 'cuda',
                        worker_init_fn=init_worker, generator=generator,
                        **({'prefetch_factor': 2} if workers else {}))
    initial_bn = {name: value.detach().cpu().clone() for name, value in net.named_buffers()
                  if 'running_' in name or 'num_batches_tracked' in name}
    began = time.perf_counter()
    best_loss, best_epoch, updates = float('inf'), None, 0
    with (out / 'epochs.jsonl').open('x', encoding='utf-8') as history:
        for epoch in range(actual_epochs):
            started = time.perf_counter()
            specs = epoch_rows[epoch]['batches']
            if smoke:
                specs = [next(b for b in specs if b['same_pairs'])]
            order = [by_id[c] for b in specs for c in b['clip_ids']]
            sampler.set_order(order)
            count, pair_count, steps = 0, 0, 0
            bce_sum, pair_sum, objective_sum = 0., 0., 0.
            seen, grad_norms = [], []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for batch_index, (u8, labels, selected) in enumerate(loader):
                spec = specs[batch_index]
                if [rows[int(i)][0] for i in selected] != spec['clip_ids']:
                    raise ValueError('Frozen batch order changed before optimizer update')
                net.train()  # score_clips sets eval; every update restores Dropout training.
                pairs = spec['cross_pairs'] if mode == 'cross' else spec['same_pairs']
                measured = train_paired_batch(net, optimizer, u8, labels, transform,
                            base['pos_weight'], pairs, microbatch=8,
                            rank_lambda=profile['rank_lambda'], rank_scale=profile['rank_scale'])
                count += measured['n']; pair_count += measured['pair_count']; steps += 1; updates += 1
                bce_sum += measured['bce_sum']; pair_sum += measured['pair_sum']
                objective_sum += measured['loss']
                seen.extend(map(int, selected)); grad_norms.append(measured['grad_norm'])
            if (seen != order or count != len(order) or steps != len(specs)
                    or (not smoke and (count != 504 or pair_count != 80 or steps != 21))):
                raise ValueError('Dropped/duplicated samples, pairs or optimizer updates')
            val_indices = ([next(i for i in validation if rows[i][2] == y) for y in (0, 1)]
                           if smoke else validation)
            sampler.set_order(val_indices)
            val_sum, val_count, predictions = 0., 0, []
            class_sums, class_counts = {0: 0., 1: 0.}, {0: 0, 1: 0}
            val_seen = []
            for u8, labels, selected in loader:
                logits = score_clips(net, u8, transform, microbatch=8)
                losses = nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                val_sum += float(losses.sum()); val_count += len(labels)
                val_seen.extend(map(int, selected))
                for label in (0, 1):
                    class_sums[label] += float(losses[labels == label].sum())
                    class_counts[label] += int((labels == label).sum())
                predictions.extend(dict(clip_id=rows[int(i)][0], label=int(y), logit=float(z))
                                   for i, y, z in zip(selected, labels, logits))
            if val_seen != val_indices or val_count != len(val_indices):
                raise ValueError('Fixed validation coverage/order changed')
            val_loss = val_sum / val_count
            if not np.isfinite(val_loss):
                raise FloatingPointError('Nonfinite unweighted validation BCE')
            for name, value in net.named_buffers():
                if name in initial_bn and not torch.equal(value.cpu(), initial_bn[name]):
                    raise RuntimeError('BN statistics changed: %s' % name)
            _check_fingerprints(root, sources, plan_dir, pair_sha, base_dir, base_sha, epochs_sha)
            if val_loss < best_loss:  # Strictly smaller; exact ties retain the earlier epoch.
                best_loss, best_epoch = val_loss, epoch + 1
                if not smoke:
                    _save_checkpoint(out / 'best.pt', net, optimizer, profile, epoch + 1)
            write_json(out / ('validation-epoch-%02d.json' % (epoch + 1)), predictions)
            record = dict(completed_epochs=epoch + 1, train_loss=objective_sum / steps,
                          train_weighted_bce=bce_sum / count, train_bce_sum=bce_sum,
                          train_pair_sum=pair_sum, train_pair_count=pair_count,
                          train_pair_mean=pair_sum / pair_count if pair_count else None,
                          objective_mode=mode, rank_lambda=profile['rank_lambda'], rank_scale=profile['rank_scale'],
                          validation_unweighted_bce=val_loss, best_epoch=best_epoch,
                          samples=count, optimizer_steps=steps, total_updates=updates,
                          train_clip_ids=[rows[i][0] for i in seen],
                          learning_rates=[g['lr'] for g in optimizer.param_groups],
                          grad_norm_mean=float(np.mean(grad_norms)), grad_norm_max=max(grad_norms),
                          validation_P=class_counts[1], validation_N=class_counts[0],
                          validation_pos_bce=class_sums[1] / class_counts[1],
                          validation_neg_bce=class_sums[0] / class_counts[0],
                          seconds=time.perf_counter() - started,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
            history.write(json.dumps(record, allow_nan=False) + '\n'); history.flush()
            print(json.dumps({k: v for k, v in record.items() if k != 'train_clip_ids'}), flush=True)
    if not smoke:
        _save_checkpoint(out / 'last.pt', net, optimizer, profile, actual_epochs)
    final_sha = state_digest(net)
    if initial_sha == final_sha:
        raise RuntimeError('Parameters did not change')
    _check_fingerprints(root, sources, plan_dir, pair_sha, base_dir, base_sha, epochs_sha)
    result = dict(status='smoke_passed' if smoke else 'supervised_complete_dense_evaluation_pending',
                  completed_epochs=actual_epochs, best_epoch=best_epoch, best_validation_bce=best_loss,
                  initial_state_sha256=initial_sha, final_state_sha256=final_sha,
                  optimizer_steps=updates, seconds=time.perf_counter() - began, profile=profile,
                  independent_test_used=False,
                  protocol_status='EXP-018 authorized section6 exception; section10 full dual-protocol CV not completed')
    if not smoke:
        result['selected_checkpoint'] = dict(filename='best.pt', sha256=digest(out / 'best.pt'),
                                               completed_epochs=best_epoch)
    write_json(out / 'training-result.json', result)
    return result
