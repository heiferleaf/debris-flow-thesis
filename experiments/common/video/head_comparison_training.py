"""EXP020 paired inner fits. The EXP017 epoch execution is retained verbatim."""
from __future__ import annotations
import math
import os
from pathlib import Path
import platform
import sys
import time
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from experiments.common.video.cv_supervised_training import (_sha, _canonical, _json, _integer,
    _finite_state, _checkpoint, _prepare)
from experiments.common.video.cv_supervised_plan import validate_epoch_quota
from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
from experiments.common.video.supervised_runtime import (
    ClipDataset, ClipInput, EpochBatches, init_worker, score_clips, train_effective_batch)

WEIGHTS_SHA = '91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8'


class LinearVideoClassifier(VideoClassifier):
    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)
        self.head = nn.Linear(512, 1)


def require_local_weights():
    path = Path(torch.hub.get_dir()) / 'checkpoints/r2plus1d_18-91a641e6.pth'
    if not path.is_file() or _sha(path) != WEIGHTS_SHA:
        raise ValueError('Require locally present exact public K400 weights; no download allowed')
    return path


def initialization_records(enforce_original=True):
    import gc
    require_local_weights()
    records = {}
    for head in ('deep', 'linear'):
        for seed in (7, 8):
            configure_fp32(seed)
            net = (VideoClassifier if head == 'deep' else LinearVideoClassifier)(pretrained=True)
            _finite_state(net)
            records[head + '-s' + str(seed)] = dict(initial_state_sha256=state_digest(net),
                backbone_state_sha256=state_digest(net.backbone), stem_state_sha256=state_digest(net.backbone.stem),
                head_parameters=sum(p.numel() for p in net.head.parameters()))
            del net
            gc.collect()
    if enforce_original and records['deep-s7']['initial_state_sha256'] != 'c1847e1c0987ff003bff99805ab795af8eb59b07db210c7999e377b337ee9b81':
        raise ValueError('Original deep seed7 initialization mismatch')
    if len({r['backbone_state_sha256'] for r in records.values()}) != 1 or len({r['stem_state_sha256'] for r in records.values()}) != 1:
        raise ValueError('Common K400 backbone/stem changed across heads/seeds')
    if any(r['head_parameters'] != (513 if k.startswith('linear') else 76033) for k, r in records.items()):
        raise ValueError('Registered head capacity changed')
    return records


def fit(root, out, *, head, seed, fold, bank, epoch_clip_ids, verified_frame_index,
        input_fingerprints, validation_clip_ids=None, selected_epochs=None,
        expected_initial_state_sha256=None, max_epochs=20, device='cuda', workers=2,
        cache_mib=128, model_factory=None):
    """EXP020 inner-only paired head fit; factory injection is CPU-fixture-only."""
    stage = 'inner_select'
    if head not in ('deep', 'linear') or type(seed) is not int or seed not in (7, 8):
        raise ValueError('Only registered head/seed combinations are supported')
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    _integer(fold, 1)
    _integer(workers, 0)
    _integer(cache_mib, 0)
    _integer(max_epochs, 1, 20)
    if not validation_clip_ids or selected_epochs is not None:
        raise ValueError('Inner-only comparison requires fixed validation and no outer E')
    epochs = max_epochs
    if model_factory is None or expected_initial_state_sha256 is not None:
        if (not isinstance(expected_initial_state_sha256, str) or len(expected_initial_state_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_initial_state_sha256)):
            raise ValueError('Production fits require a frozen initialization SHA')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Unsupported/unavailable device; no silent fallback')
    if model_factory is not None and (device.type != 'cpu' or workers != 0):
        raise ValueError('Injected model factories are only for isolated CPU/no-worker fixtures')
    if not input_fingerprints:
        raise ValueError('Require frozen plan and accepted-cache input fingerprints')
    files = {}
    for relative, fingerprint in input_fingerprints.items():
        path = (root / relative).resolve()
        path.relative_to(root)
        if len(fingerprint) != 64 or _sha(path) != fingerprint:
            raise ValueError('Frozen fit input SHA mismatch: ' + relative)
        files[str(path)] = fingerprint
    code = {}
    for module in (sys.modules[__name__], sys.modules[ClipDataset.__module__],
                   sys.modules[VideoClassifier.__module__], sys.modules[init_worker.__module__],
                   sys.modules[validate_epoch_quota.__module__], sys.modules[_prepare.__module__]):
        path = Path(module.__file__).resolve()
        code[str(path)] = _sha(path)
    pixel_path = Path(sys.modules['experiments.common.video.pts_clip_sampling'].__file__).resolve()
    code[str(pixel_path)] = _sha(pixel_path)
    rows, schedule, validation, metadata, counts = _prepare(bank, epoch_clip_ids, validation_clip_ids, verified_frame_index)
    planned_steps = sum(math.ceil(len(order) / 24) for order in schedule[:epochs])
    input_digest = _canonical(dict(metadata=metadata, epoch_clip_ids=epoch_clip_ids,
                                   validation_clip_ids=validation_clip_ids, files=input_fingerprints))
    def unchanged():
        for path, fingerprint in {**files, **code}.items():
            if _sha(path) != fingerprint:
                raise RuntimeError('Fit input/source changed during execution: ' + path)
    unchanged()
    out.mkdir(parents=True, exist_ok=False)
    total_updates, completed = 0, 0
    began = time.perf_counter()
    try:
        import cv2
        cv2.setNumThreads(1)
        configure_fp32(seed)
        if model_factory is None:
            require_local_weights()
        factory = (VideoClassifier if head == 'deep' else LinearVideoClassifier) if model_factory is None else model_factory
        net = factory(pretrained=True).to(device)
        transform = ClipInput('rgb_zero').to(device)
        _finite_state(net)
        initial_sha = state_digest(net)
        if expected_initial_state_sha256 is not None and initial_sha != expected_initial_state_sha256:
            raise ValueError('Fresh initialization differs from frozen/inner public-K400 initialization')
        optimizer = torch.optim.Adam([dict(params=net.backbone.parameters(), lr=1e-4),
                                      dict(params=net.head.parameters(), lr=1e-3)],
                                     betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
        if optimizer.state:
            raise ValueError('Fresh Adam must have empty state')
        profile = dict(stage=stage, fold=fold, seed=seed, head=head, mode='rgb_zero', objective='ordinary_weighted_bce',
            execution_kind='cpu_fixture' if model_factory is not None else 'registered_fit',
            model_class=type(net).__module__ + '.' + type(net).__qualname__,
            pretrained='KINETICS400_V1' if model_factory is None else 'injected_cpu_fixture_no_download',
            initial_state_sha256=initial_sha, backbone_state_sha256=state_digest(net.backbone),
            stem_state_sha256=state_digest(net.backbone.stem) if hasattr(net.backbone, 'stem') else None,
            head_parameters=sum(p.numel() for p in net.head.parameters()),
            initialization_order='full VideoClassifier then replace Linear(512,1); no reseed' if head == 'linear' else 'unchanged VideoClassifier',
            dataloader_generator_seed=100007, expected_initial_state_sha256=expected_initial_state_sha256,
            expected_inner_initial_state_sha256=expected_initial_state_sha256 if stage == 'fixed_outer' else None,
            effective_batch=24, microbatch=8, checkpoint_backbone=False, workers=workers, cache_mib=cache_mib,
            train_P=counts[1], train_N=counts[0], supervised_pos_weight=counts[0] / counts[1],
            requested_epochs=epochs, planned_optimizer_steps=planned_steps, frozen_schedule_epochs=20,
            selection_rule='minimum_fixed_inner_unweighted_BCE_earliest_exact_tie' if stage == 'inner_select' else 'fixed_E_from_inner_no_validation_or_best_selection',
            input_fingerprints=dict(input_fingerprints), parsed_inputs_sha256=input_digest, code_sha256=code,
            optimizer=dict(name='Adam', lr_backbone=1e-4, lr_head=1e-3, betas=[.9,.999], eps=1e-8, weight_decay=0, initial_state_entries=0),
            environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                platform=platform.platform(), device=str(device), precision='FP32', amp=False,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic))
        _json(out / 'profile.json', profile)
        profile_sha = _sha(out / 'profile.json')
        _json(out / 'inputs.json', dict(clip_bank=metadata, epoch_clip_ids=epoch_clip_ids,
                                      validation_clip_ids=validation_clip_ids))
        ds = ClipDataset(rows, cache_bytes=cache_mib * 1024 ** 2)
        sampler = EpochBatches(24)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=workers,
            persistent_workers=workers > 0, pin_memory=device.type == 'cuda',
            worker_init_fn=init_worker, generator=torch.Generator().manual_seed(100007),
            **({'prefetch_factor': 2} if workers else {}))
        bn_initial = {k: t.detach().cpu().clone() for k, t in net.named_buffers()
                      if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        best_loss, best_epoch = math.inf, None
        with (out / 'epochs.jsonl').open('x', encoding='utf-8') as history:
            for epoch, order in enumerate(schedule[:epochs], 1):
                started = time.perf_counter()
                net.train()
                sampler.set_order(order)
                seen, sums, count, steps, grads, batch_sizes = [], 0., 0, 0, [], []
                for u8, labels, selected in loader:
                    result = train_effective_batch(net, optimizer, u8, labels, transform,
                        profile['supervised_pos_weight'], microbatch=8, checkpoint_backbone=False)
                    if (result['optimizer_steps'] != 1 or result['n'] != len(labels)
                            or not math.isfinite(result['loss']) or not math.isfinite(result['grad_norm'])):
                        raise FloatingPointError('Invalid/nonfinite effective-batch result')
                    _finite_state(net)
                    seen.extend(map(int, selected))
                    sums += result['loss'] * result['n']
                    count += result['n']
                    grads.append(result['grad_norm'])
                    batch_sizes.append(result['n'])
                    steps += 1
                    total_updates += 1
                if seen != order or count != len(order) or steps != math.ceil(len(order) / 24):
                    raise RuntimeError('Dropped/reordered samples or optimizer-step mismatch')
                record = dict(completed_epochs=epoch, train_weighted_bce=sums / count,
                    samples=count, train_P=counts[1], train_N=counts[0], optimizer_steps=steps,
                    total_updates=total_updates, actual_effective_batch_sizes=batch_sizes,
                    train_clip_ids=[rows[i][0] for i in seen], learning_rates=[g['lr'] for g in optimizer.param_groups],
                    grad_norm_mean=float(np.mean(grads)), grad_norm_max=max(grads), finite=True,
                    initial_state_sha256=initial_sha, model_state_sha256=state_digest(net),
                    profile_sha256=profile_sha, parsed_inputs_sha256=input_digest, code_sha256=code)
                if stage == 'inner_select':
                    sampler.set_order(validation)
                    predictions, val_losses, val_ids = [], [], []
                    for u8, labels, selected in loader:
                        logits = score_clips(net, u8, transform, microbatch=8)
                        losses = nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                        if not torch.isfinite(logits).all() or not torch.isfinite(losses).all():
                            raise FloatingPointError('Nonfinite inner validation')
                        val_losses.extend(map(float, losses))
                        val_ids.extend(map(int, selected))
                        predictions.extend(dict(clip_id=rows[int(i)][0], label=int(y), logit=float(z))
                                           for i, y, z in zip(selected, labels, logits))
                    if val_ids != validation or len(val_losses) != len(validation):
                        raise RuntimeError('Fixed inner validation coverage/order mismatch')
                    val_loss = math.fsum(val_losses) / len(val_losses)
                    if not math.isfinite(val_loss):
                        raise FloatingPointError('Nonfinite inner validation BCE')
                    if val_loss < best_loss:  # Strict comparison keeps the earliest EXACT tie.
                        best_loss, best_epoch = val_loss, epoch
                        _checkpoint(out / 'best.pt', net, optimizer, profile, epoch)
                    _json(out / ('validation-epoch-%02d.json' % epoch), predictions)
                    record.update(validation_unweighted_bce=val_loss, validation_samples=len(validation),
                                  best_epoch=best_epoch, validation_clip_ids=[rows[i][0] for i in val_ids])
                for name, tensor in net.named_buffers():
                    if name in bn_initial and not torch.equal(tensor.detach().cpu(), bn_initial[name]):
                        raise RuntimeError('BN statistics changed: ' + name)
                if not math.isfinite(record['train_weighted_bce']):
                    raise FloatingPointError('Nonfinite epoch loss')
                unchanged()
                record['seconds'] = time.perf_counter() - started
                history.write(json.dumps(record, allow_nan=False) + '\n')
                history.flush()
                completed = epoch
        if completed != epochs or total_updates != planned_steps:
            raise RuntimeError('Fit incomplete or update count differs from frozen prefix')
        _checkpoint(out / 'last.pt', net, optimizer, profile, epochs)
        selected_name = 'best.pt'
        result = dict(status='fit_complete', stage=stage, fold=fold, completed_epochs=completed,
            selected_epochs=best_epoch if stage == 'inner_select' else epochs,
            selection_rule=profile['selection_rule'], initial_state_sha256=initial_sha,
            final_state_sha256=state_digest(net), optimizer_steps=total_updates,
            train_P=counts[1], train_N=counts[0], pos_weight=profile['supervised_pos_weight'],
            fixed_outer_validation_evaluated=False, input_fingerprints=dict(input_fingerprints),
            parsed_inputs_sha256=input_digest, code_sha256=code, profile_sha256=profile_sha,
            selected_checkpoint=dict(filename=selected_name, sha256=_sha(out / selected_name),
                                     completed_epochs=best_epoch if stage == 'inner_select' else epochs),
            last_checkpoint=dict(filename='last.pt', sha256=_sha(out / 'last.pt'), completed_epochs=epochs),
            seconds=time.perf_counter() - began)
        if stage == 'inner_select':
            result['best_validation_unweighted_bce'] = best_loss
        unchanged()
        _json(out / 'training-result.json', result)
        return result
    except Exception as error:
        _json(out / 'failure.json', dict(status='failed', stage=stage, fold=fold,
              completed_epochs=completed, optimizer_steps=total_updates,
              error_type=type(error).__name__, error=str(error), seconds=time.perf_counter() - began))
        raise
