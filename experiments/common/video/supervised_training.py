"""EXP-016 supervised execution; the input plan is immutable and external."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, rooted, verify_plan, write_json
from experiments.common.video.supervised_runtime import (ClipDataset, ClipInput, EpochBatches,
                                                        FrameRef, init_worker, score_clips, train_effective_batch)


def load_inputs(root, plan_dir, check_files=False):
    root, plan_dir = Path(root).resolve(), Path(plan_dir).resolve()
    plan = verify_plan(root, plan_dir)
    assignments = csv_rows(plan_dir / 'assignments.csv')
    selected = {r['clip_id'] for r in assignments}
    resource = root / plan['resource']
    bank = {r['clip_id']: r for r in jsonl(resource / 'windows.jsonl') if r['clip_id'] in selected}
    if len(bank) != len(selected):
        raise ValueError('Assignment references absent clip')
    completion = json.loads((root / 'data/process/EXP-016/metadata-v4/cache-completion.json').read_text(encoding='utf-8'))
    wanted = {(r['unit_id'], i) for r in bank.values() for i in r['frame_indices']}
    frames, parents = {}, {}
    with (root / completion['verified_cache_index']).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['unit_id'], int(row['frame_index']))
            if key in wanted:
                relative = PurePosixPath(row['cache_path'])
                if relative.is_absolute() or '..' in relative.parts or ':' in row['cache_path']:
                    raise ValueError('Cache path must be project-relative')
                parent = str(relative.parent)
                if parent not in parents:
                    parents[parent] = rooted(root, parent)
                p = parents[parent] / relative.name
                if check_files and (not p.is_file() or p.stat().st_size != int(row['size_bytes'])):
                    raise ValueError(f'Missing/changed cached image: {p}')
                frames[key] = FrameRef(str(p), row['jpeg_sha256'], row['rgb112_sha256'])
    if set(frames) != wanted:
        raise ValueError('Verified cache index does not cover assigned frames')
    ids = sorted(bank)
    indices = {cid: i for i, cid in enumerate(ids)}
    rows = [(cid, tuple(frames[(bank[cid]['unit_id'], i)] for i in bank[cid]['frame_indices']),
             int(bank[cid]['kind'] == 'pos')) for cid in ids]
    schedule = {ep: [] for ep in range(plan['epochs'])}
    validation = []
    for row in assignments:
        target = validation if row['split'] == 'validation' else schedule[int(row['epoch'])]
        target.append(indices[row['clip_id']])
        clip = bank[row['clip_id']]
        if any(row[k] != clip[k] for k in ('unit_id', 'kind', 'split')):
            raise ValueError('Assignment label/source mismatch')
    for values in schedule.values():
        positives = sum(rows[i][2] for i in values)
        if (len(values), positives, len(set(values))) != (504, 356, 504):
            raise ValueError('Training quota changed')
    if len(validation) != 136 or sum(rows[i][2] for i in validation) != 96 or len(set(validation)) != 136:
        raise ValueError('Fixed validation quota changed')
    return plan, rows, schedule, validation


def _environment(device):
    import cv2
    import torchvision
    return dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                opencv=cv2.__version__, platform=platform.platform(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                precision='FP32', amp=False, tf32_matmul=False, tf32_cudnn=False,
                cudnn_benchmark=False, cudnn_deterministic=True)


def _save_checkpoint(path, net, optimizer, profile, epoch):
    tmp = path.with_suffix('.tmp')
    torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    optimizer=optimizer.state_dict(), profile=profile, completed_epochs=epoch), tmp)
    os.replace(tmp, path)


def initial_diagnostics(net, u8, labels, device):
    """No optimizer step or stochastic head; establish the real-network gates."""
    net.eval()
    rgb = ClipInput('rgb_zero').to(device)(u8[:1].to(device))
    diff = ClipInput('rgb_diff').to(device)(u8[:1].to(device))
    with torch.no_grad():
        a, b = net(rgb), net(diff)
    torch.testing.assert_close(a, b, rtol=0., atol=0.)
    if diff[:, 3:].count_nonzero() == 0:
        raise ValueError('Diagnostic clip has no temporal difference')
    net.zero_grad(set_to_none=True)
    loss = nn.functional.binary_cross_entropy_with_logits(net(diff), labels[:1].to(device).float())
    loss.backward()
    d_grad = net.backbone.stem[0].weight.grad[:, 3:].norm()
    bn = next(m for m in net.backbone.modules() if isinstance(m, nn.BatchNorm3d))
    gamma_grad = bn.weight.grad.norm()
    if not (torch.isfinite(d_grad) and d_grad > 0 and torch.isfinite(gamma_grad) and gamma_grad > 0):
        raise RuntimeError('Difference stem/BN affine diagnostic gradient failed')
    result = dict(initial_logits_equal=True, difference_stem_grad=float(d_grad.cpu()),
                  bn_gamma_grad=float(gamma_grad.cpu()), optimizer_steps=0)
    net.zero_grad(set_to_none=True)
    net.train()
    return result


def train(root, plan_dir, out, mode, seed=7, epochs=20, device='cuda', workers=2,
          cache_mib=128, microbatch=8, checkpoint=False, smoke=False):
    root, plan_dir, out = map(Path, (root, plan_dir, out))
    code_paths = [Path(__file__), Path(sys.modules[VideoClassifier.__module__].__file__),
                  Path(sys.modules[ClipInput.__module__].__file__),
                  root / 'experiments/common/video/supervised_plan.py',
                  root / 'experiments/common/video/pts_clip_sampling.py',
                  root / 'experiments/runspecs/EXP-016/train.py',
                  root / 'experiments/runspecs/EXP-016/run.py',
                  root / 'docs/process/experiments/EXP-016.md',
                  root / 'docs/process/experiments/EXP-006-数据使用方案.md']
    execution_sha = {p.relative_to(root).as_posix(): digest(p) for p in code_paths}
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if mode not in ('rgb_zero', 'rgb_diff') or seed not in (7, 8):
        raise ValueError('Unregistered input mode or seed')
    if not 1 <= epochs <= 20 or not 1 <= microbatch <= 8 or workers < 0 or cache_mib < 0:
        raise ValueError('Unregistered execution budget')
    if out.exists():
        raise FileExistsError(out)
    plan, rows, schedule, validation = load_inputs(root, plan_dir)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    configure_fp32(seed)
    net = VideoClassifier(pretrained=True).to(device)
    transform = ClipInput(mode).to(device)
    initial_sha = state_digest(net)
    optimizer = torch.optim.Adam([dict(params=net.backbone.parameters(), lr=1e-4),
                                  dict(params=net.head.parameters(), lr=1e-3)],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
    profile = dict(format_version=1, architecture='r2plus1d_18_six_channel', mode=mode,
                   pretrained='KINETICS400_V1', seed=seed, initial_state_sha256=initial_sha,
                   plan_sha256=digest(plan_dir / 'plan.json'), aggregation='max', score_threshold=.5,
                   span_seconds=3., clip_len=16, inference_stride_seconds=1.5,
                   supervised_pos_weight=plan['pos_weight'], effective_batch=24,
                   microbatch=microbatch, checkpoint_backbone=checkpoint, workers=workers,
                   cache_mib_per_worker=cache_mib, model_count=1, precision='FP32',
                   input_pipeline='JPEG95/RGB112/MEAN_STD/signed_difference_v1',
                   environment=_environment(device), smoke=smoke)
    profile.update(requested_epochs=1 if smoke else epochs,
                   requested_optimizer_updates=1 if smoke else 21 * epochs)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'profile.json', profile)
    write_json(out / 'provenance.json', dict(argv=sys.argv, code=execution_sha))
    ds = ClipDataset(rows, cache_bytes=cache_mib * 1024**2)
    batch_sampler = EpochBatches(24)
    batch_sampler.set_order(schedule[0])
    generator = torch.Generator().manual_seed(seed + 100000)
    loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device.type == 'cuda',
                        worker_init_fn=init_worker, generator=generator,
                        **({'prefetch_factor': 2} if workers else {}))
    if smoke:
        epochs = 1
        train_indices = [next(i for i in schedule[0] if rows[i][2] == y) for y in (0, 1)]
        validation = [next(i for i in validation if rows[i][2] == y) for y in (0, 1)]
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, update_count = float('inf'), None, 0
    initial_bn = {name: value.detach().cpu().clone() for name, value in net.named_buffers()
                  if 'running_' in name or 'num_batches_tracked' in name}
    began = time.perf_counter()
    diagnostics = None
    if smoke or epochs == 1:
        first = schedule[0][0]
        u8, y, _ = ds[first]
        diagnostics = initial_diagnostics(net, torch.from_numpy(u8[None]), torch.tensor([y]), device)
        if state_digest(net) != initial_sha:
            raise RuntimeError('Diagnostic changed initialization')
        write_json(out / 'initial-diagnostics.json', diagnostics)
    with (out / 'epochs.jsonl').open('x', encoding='utf-8') as history:
        for epoch in range(epochs):
            started = time.perf_counter()
            net.train()
            current = train_indices if smoke else schedule[epoch]
            order = [current[i] for i in rng.permutation(len(current))]
            batch_sampler.set_order(order)
            sums, count, steps = 0., 0, 0
            seen, grad_norms = [], []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for u8, labels, selected in loader:
                result = train_effective_batch(net, optimizer, u8, labels, transform,
                                               plan['pos_weight'], microbatch, checkpoint)
                sums += result['loss'] * result['n']
                count += result['n']
                steps += 1
                update_count += 1
                seen.extend(map(int, selected))
                grad_norms.append(result['grad_norm'])
            if seen != order or count != len(order) or steps != (len(order) + 23) // 24:
                raise ValueError('Dropped/duplicated samples or wrong update count')
            batch_sampler.set_order(validation)
            val_sum, val_count, predictions = 0., 0, []
            val_class_sum, val_class_count = {0: 0., 1: 0.}, {0: 0, 1: 0}
            for u8, labels, selected in loader:
                logits = score_clips(net, u8, transform, microbatch)
                losses = nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                val_sum += float(losses.sum())
                for label in (0, 1):
                    val_class_sum[label] += float(losses[labels == label].sum())
                    val_class_count[label] += int((labels == label).sum())
                val_count += len(labels)
                predictions.extend(dict(clip_id=rows[int(i)][0], label=int(y), logit=float(z))
                                   for i, y, z in zip(selected, labels, logits))
            val_loss = val_sum / val_count
            if not np.isfinite(val_loss):
                raise FloatingPointError('Nonfinite validation loss')
            for name, value in net.named_buffers():
                if name in initial_bn and not torch.equal(value.cpu(), initial_bn[name]):
                    raise RuntimeError(f'BN statistics changed: {name}')
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch = val_loss, epoch + 1
                if not smoke:
                    _save_checkpoint(out / 'best.pt', net, optimizer, profile, epoch + 1)
            write_json(out / f'validation-epoch-{epoch+1:02d}.json', predictions)
            record = dict(completed_epochs=epoch + 1, train_loss=sums / count,
                          validation_unweighted_bce=val_loss, best_epoch=best_epoch,
                          samples=count, optimizer_steps=steps, total_updates=update_count,
                          learning_rates=[g['lr'] for g in optimizer.param_groups],
                          train_clip_ids=[rows[i][0] for i in seen],
                          grad_norm_mean=float(np.mean(grad_norms)), grad_norm_max=max(grad_norms),
                          validation_P=val_class_count[1], validation_N=val_class_count[0],
                          validation_pos_bce=val_class_sum[1] / val_class_count[1],
                          validation_neg_bce=val_class_sum[0] / val_class_count[0],
                          seconds=time.perf_counter() - started,
                          peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
            history.write(json.dumps(record, allow_nan=False) + '\n')
            history.flush()
            print(json.dumps({k: v for k, v in record.items() if k != 'train_clip_ids'}), flush=True)
    if not smoke:
        _save_checkpoint(out / 'last.pt', net, optimizer, profile, epochs)
    final_sha = state_digest(net)
    if initial_sha == final_sha:
        raise RuntimeError('Parameters did not change')
    result = dict(status='smoke_passed' if smoke else 'supervised_complete_dense_evaluation_pending',
                  completed_epochs=epochs, best_epoch=best_epoch, best_validation_bce=best_loss,
                  initial_state_sha256=initial_sha, final_state_sha256=final_sha,
                  optimizer_steps=update_count, seconds=time.perf_counter() - began,
                  profile=profile, independent_test_used=False,
                  protocol_status='EXP-016 exception to section6; section10 full CV not completed')
    result['initial_diagnostics'] = diagnostics
    if not smoke:
        result['selected_checkpoint'] = dict(filename='best.pt', sha256=digest(out / 'best.pt'),
                                              completed_epochs=best_epoch)
    if any(digest(root / path) != sha for path, sha in execution_sha.items()):
        raise RuntimeError('Training source/card/protocol changed during execution')
    write_json(out / 'training-result.json', result)
    return result
