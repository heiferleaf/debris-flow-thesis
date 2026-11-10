"""Reusable ordinary-RGB inner selection and fixed-E fresh refit core.

Consumes a parsed, immutable plan and an independently verified FrameRef index.
Does not partition videos, generate windows, decode video, evaluate outer data,
rent hardware or decide whether a paid/real run is authorized.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.common.video.cv_supervised_plan import validate_epoch_quota
from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
from experiments.common.video.supervised_runtime import (
    ClipDataset, ClipInput, EpochBatches, FrameRef, init_worker, score_clips, train_effective_batch,
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def _integer(value, low, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError('Invalid integer execution argument')
    return value


def _finite_state(net):
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in net.state_dict().values()):
        raise FloatingPointError('Nonfinite model parameter/buffer')


def _checkpoint(path, net, optimizer, profile, epoch):
    _finite_state(net)
    temp = path.with_suffix('.tmp')
    if temp.exists():
        raise FileExistsError(temp)
    with temp.open('xb') as f:
        torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                        optimizer=optimizer.state_dict(), profile=profile, completed_epochs=epoch,
                        state_sha256=state_digest(net)), f)
    os.replace(temp, path)


def _prepare(bank, epoch_clip_ids, validation_clip_ids, verified_frame_index):
    if len(epoch_clip_ids) != 20:
        raise ValueError('Require the already frozen full twenty-epoch bank/schedule')
    schedule = [list(order) for order in epoch_clip_ids]
    for order in schedule:
        if not order or len(order) != len(set(order)) or any(not isinstance(cid, str) for cid in order):
            raise ValueError('Empty/duplicate/non-string training identity')
        validate_epoch_quota(order, bank)
    val = [] if validation_clip_ids is None else list(validation_clip_ids)
    if len(val) != len(set(val)):
        raise ValueError('Duplicate fixed validation clip')
    if val:
        validate_epoch_quota(val, bank)
    train_ids = set().union(*map(set, schedule))
    if train_ids & set(val):
        raise ValueError('Train/validation clip overlap')
    ids = sorted(train_ids | set(val))
    rows, metadata = [], {}
    for cid in ids:
        w = bank[cid]  # Deliberately never inspect unrelated/outer-held-out bank rows.
        label = w['label']
        indices, pts = w['frame_indices'], w['source_pts_seconds']
        if (w['clip_id'] != cid or type(label) is not int or label not in (0, 1)
                or w['kind'] != ('pos' if label else 'neg') or len(indices) != 16 or len(pts) != 16
                or any(type(i) is not int or i < 0 for i in indices)
                or not all(math.isfinite(p) for p in pts)
                or any(a >= b for a, b in zip(indices, indices[1:]))
                or any(a >= b for a, b in zip(pts, pts[1:]))
                or not 2.9 - 1e-9 <= pts[-1] - pts[0] <= 3.1 + 1e-9):
            raise ValueError('Invalid bank label/frame/PTS identity')
        refs = tuple(verified_frame_index[(w['unit_id'], i)] for i in indices)
        if any(not isinstance(ref, FrameRef) for ref in refs):
            raise ValueError('Require independently verified FrameRef values; no path guessing')
        rows.append((cid, refs, label))
        metadata[cid] = dict(unit_id=w['unit_id'], group_id=w['group_id'], label=label,
                             frame_indices=list(indices), source_pts_seconds=list(pts),
                             frames=[dict(path=f.path, jpeg_sha256=f.jpeg_sha256,
                                          rgb112_sha256=f.rgb112_sha256) for f in refs])
    for key in ('unit_id', 'group_id'):
        if {metadata[c][key] for c in train_ids} & {metadata[c][key] for c in val}:
            raise ValueError('Train/validation video or group overlap')
    counts = [Counter(metadata[c]['label'] for c in order) for order in schedule]
    if not counts[0][0] or not counts[0][1] or any(c != counts[0] for c in counts):
        raise ValueError('Both fit-training classes and fixed per-epoch class quota required')
    train_units = [{metadata[c]['unit_id'] for c in order} for order in schedule]
    if any(s != train_units[0] for s in train_units):
        raise ValueError('Fit-training video pool changed across frozen epochs')
    by_id = {cid: i for i, cid in enumerate(ids)}
    return rows, [[by_id[c] for c in order] for order in schedule], [by_id[c] for c in val], metadata, counts[0]


def fit(root, out, *, stage, fold, bank, epoch_clip_ids, verified_frame_index,
        input_fingerprints, validation_clip_ids=None, selected_epochs=None,
        expected_initial_state_sha256=None, max_epochs=20, device='cuda', workers=2,
        cache_mib=128, model_factory=None):
    """Fit one stage only; caller owns plan loading and fold orchestration.

    stage='inner_select': fixed validation IDs required, run at most20 epochs;
      choose minimum unweighted BCE, strict improvement preserves earliest ties.
      Production requires the frozen expected initialization SHA before updates.
    stage='fixed_outer': validation must be None, selected_epochs=E and the inner
      initial-state SHA are required. Fresh seed7/K400 model + empty Adam; execute
      exactly the firstE rows of the original twenty-row outer-training schedule.

    input_fingerprints maps repository-relative files to SHA256. The caller must
    include the frozen plan and independently accepted cache-index/manifest files;
    each selected JPEG/RGB112 is additionally checked by the existing PixelCache.
    model_factory is a Python-only CPU-fixture injection, never a run-config key.
    Its default calls VideoClassifier(pretrained=True); no optimizer/state injection.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    _integer(fold, 1)
    _integer(workers, 0)
    _integer(cache_mib, 0)
    _integer(max_epochs, 1, 20)
    if stage not in ('inner_select', 'fixed_outer'):
        raise ValueError('Unknown fit stage')
    if stage == 'inner_select':
        if not validation_clip_ids or selected_epochs is not None:
            raise ValueError('Inner selection requires fixed validation and no outer selection arguments')
        epochs = max_epochs
    else:
        if validation_clip_ids is not None:
            raise ValueError('Outer fixed-E fit must not receive validation identities/tensors')
        epochs = _integer(selected_epochs, 1, 20)
        if (not isinstance(expected_initial_state_sha256, str)
                or len(expected_initial_state_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_initial_state_sha256)):
            raise ValueError('Outer refit requires inner initialization SHA')
    if (stage == 'fixed_outer' or model_factory is None or expected_initial_state_sha256 is not None):
        if (not isinstance(expected_initial_state_sha256, str)
                or len(expected_initial_state_sha256) != 64
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
                   sys.modules[validate_epoch_quota.__module__]):
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
        configure_fp32(7)
        factory = VideoClassifier if model_factory is None else model_factory
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
        profile = dict(stage=stage, fold=fold, seed=7, mode='rgb_zero', objective='ordinary_weighted_bce',
            execution_kind='cpu_fixture' if model_factory is not None else 'registered_fit',
            model_class=type(net).__module__ + '.' + type(net).__qualname__,
            pretrained='KINETICS400_V1' if model_factory is None else 'injected_cpu_fixture_no_download',
            initial_state_sha256=initial_sha, expected_initial_state_sha256=expected_initial_state_sha256,
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
        if stage == 'fixed_outer':
            _checkpoint(out / 'selected.pt', net, optimizer, profile, epochs)
        selected_name = 'best.pt' if stage == 'inner_select' else 'selected.pt'
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
