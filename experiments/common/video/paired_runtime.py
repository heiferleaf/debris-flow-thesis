"""Same-forward EXP-018 ranking supervision, with immediate microblock backward."""
from __future__ import annotations
from numbers import Integral

import numpy as np
import torch
from torch import nn

from experiments.common.video.supervised_runtime import (
    _check_model_state, _check_optimizer, _fp32_device, _integer,
)


def validate_pairs(pairs, labels, microbatch=8):
    microbatch = _integer(microbatch, 1, 'Microbatch')
    result, seen = [], set()
    for pair in pairs:
        if (len(pair) != 2 or any(not isinstance(i, Integral) or isinstance(i, (bool, np.bool_))
                                  or not 0 <= i < len(labels) for i in pair)):
            raise ValueError('Pairs require two valid integer local indices')
        p, n = map(int, pair)
        if p // microbatch != n // microbatch:
            raise ValueError('Pair must be closed inside one microblock')
        if p in seen or n in seen or p == n:
            raise ValueError('Each endpoint may occur in at most one pair')
        if labels[p] != 1 or labels[n] != 0:
            raise ValueError('Pairs must be ordered positive, negative')
        seen.update((p, n))
        result.append((p, n))
    return tuple(result)


def train_paired_batch(net, optimizer, u8, labels, transform, pos_weight, pairs,
                       microbatch=8, rank_lambda=.1, rank_scale=21 / 80):
    """One update; all ranking endpoints live in the current eight-clip graph.

    sum(weighted BCE)/actual effective N + lambda*(21/80)*sum(softplus(zN-zP)).
    Lambda zero uses the identical backward expression as the old BCE runtime.
    Pair reductions use no RNG and never add a model forward or detach logits.
    """
    _check_model_state(net)
    microbatch = _integer(microbatch, 1, 'Microbatch')
    if microbatch != 8:
        raise ValueError('EXP-018 freezes microbatch=8')
    if len(u8) == 0 or len(labels) != len(u8) or labels.ndim != 1:
        raise ValueError('Expected a nonempty effective batch and binary label vector')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Labels must be binary')
    if not np.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError('A fixed positive N/P class weight is required')
    if (not np.isfinite(rank_lambda) or rank_lambda < 0
            or not np.isfinite(rank_scale) or rank_scale <= 0):
        raise ValueError('Ranking weights must be finite and nonnegative')
    pairs = validate_pairs(pairs, labels, microbatch)
    device = _fp32_device(net, transform)
    _check_optimizer(net, optimizer)
    target = labels.to(device, dtype=torch.float32, non_blocking=True)
    weight = torch.tensor(pos_weight, device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    bce_sum, pair_sum = torch.zeros((), device=device), torch.zeros((), device=device)
    try:
        for start in range(0, len(u8), microbatch):
            stop = min(start + microbatch, len(u8))
            pixels = u8[start:stop].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=False):
                logits = net(transform(pixels)).reshape(-1)
                if len(logits) != stop - start or not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite or invalid training logits; update refused')
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, target[start:stop], pos_weight=weight, reduction='sum')
                local = [(p - start, n - start) for p, n in pairs if start <= p < stop]
                ranking = (nn.functional.softplus(logits[[n for _, n in local]]
                                                  - logits[[p for p, _ in local]]).sum()
                           if local else torch.zeros((), device=device))
                loss = bce / len(u8)
                if rank_lambda:
                    loss = loss + rank_lambda * rank_scale * ranking
                if not torch.isfinite(bce) or not torch.isfinite(ranking) or not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; optimizer update refused')
                bce_sum += bce.detach()
                pair_sum += ranking.detach()
                loss.backward()
            del loss, ranking, bce, logits, pixels
        missing = [name for name, p in net.named_parameters() if p.grad is None]
        if missing:
            raise RuntimeError('Trainable parameters received no gradient: %s' % missing)
        grads = [p.grad.detach().norm() for p in net.parameters()]
        grad_norm = torch.stack(grads).norm()
        if not torch.isfinite(bce_sum) or not torch.isfinite(pair_sum) or not torch.isfinite(grad_norm):
            raise FloatingPointError('Nonfinite loss/gradient; optimizer update refused')
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    bce_value, pair_value = float(bce_sum.cpu()) / len(u8), float(pair_sum.cpu())
    return dict(loss=bce_value + rank_lambda * rank_scale * pair_value, bce=bce_value,
                bce_sum=float(bce_sum.cpu()), pair_sum=pair_value, pair_count=len(pairs),
                rank_lambda=rank_lambda, rank_scale=rank_scale, n=len(u8),
                grad_norm=float(grad_norm.cpu()), optimizer_steps=1)
