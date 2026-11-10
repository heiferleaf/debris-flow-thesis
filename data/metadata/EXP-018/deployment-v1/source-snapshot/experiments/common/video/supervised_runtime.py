"""Reusable FP32 execution for supervised video clips; no dataset decisions.

Cache only immutable JPEG -> RGB112 pixels. Never cache a changing backbone's
features. Batch boundaries belong to the caller's frozen experiment plan.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from experiments.common.video.pts_clip_sampling import rgb112_from_jpeg

PIXEL_VERSION = "jpeg95-bgr-to-rgb-inter-linear-112-v1"
MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def _integer(value, minimum, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class FrameRef:
    path: str
    jpeg_sha256: str
    rgb112_sha256: str


class PixelCache:
    """Process-local LRU with a strict pixel-byte bound (metadata is additional).

    A miss checks the frozen JPEG and processed-pixel hashes. Hits assume the
    experiment's immutable input snapshot; changing source files mid-run is not
    supported. The key also contains the preprocessing version.
    """

    def __init__(self, max_bytes=128 * 1024**2):
        self.max_bytes = _integer(max_bytes, 0, "Cache budget")
        self.used_bytes = self.hits = self.misses = 0
        self.entries = OrderedDict()

    def get(self, ref: FrameRef):
        key = (ref.path, ref.jpeg_sha256, ref.rgb112_sha256, PIXEL_VERSION)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        encoded = Path(ref.path).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != ref.jpeg_sha256:
            raise ValueError(f"JPEG fingerprint mismatch: {ref.path}")
        pixels = rgb112_from_jpeg(encoded)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != ref.rgb112_sha256:
            raise ValueError(f"RGB112 fingerprint mismatch: {ref.path}")
        pixels.setflags(write=False)
        if pixels.nbytes <= self.max_bytes:
            while self.entries and self.used_bytes + pixels.nbytes > self.max_bytes:
                _, old = self.entries.popitem(last=False)
                self.used_bytes -= old.nbytes
            self.entries[key] = pixels
            self.used_bytes += pixels.nbytes
        return pixels


class ClipDataset(Dataset):
    """Immutable clip bank: each row is (clip_id, tuple[16 FrameRef], label).

    The epoch sampler sends global row indices to persistent workers. There is
    no mutable per-epoch dataset state to leave stale in a worker process.
    """

    def __init__(self, rows, cache_bytes=128 * 1024**2):
        checked = []
        for cid, frames, label in rows:
            frames = tuple(frames)
            if (len(frames) != 16 or not isinstance(label, Real)
                    or label not in (0, 1)):
                raise ValueError("Each clip needs 16 frozen frames and a binary label")
            checked.append((str(cid), frames, int(label)))
        self.rows = tuple(checked)
        if len({row[0] for row in self.rows}) != len(self.rows):
            raise ValueError("Clip IDs must be unique")
        self.cache_bytes = _integer(cache_bytes, 0, "Cache budget")
        self.cache = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.cache is None:
            self.cache = PixelCache(self.cache_bytes)
        _, frames, label = self.rows[index]
        # stack makes an owned, writable array, never exposing cached pixels.
        return np.stack([self.cache.get(ref) for ref in frames]), label, index


class EpochBatches(Sampler):
    """Parent-process batch sampler; rebalance a tail without dropping samples."""

    def __init__(self, target_batch=24):
        self.target_batch = _integer(target_batch, 1, "Batch target")
        self.batches = ()

    def set_order(self, indices):
        order = tuple(_integer(i, 0, "Clip index") for i in indices)
        if not order or len(set(order)) != len(order) or min(order) < 0:
            raise ValueError("An epoch needs distinct nonnegative clip indices")
        n_batches = (len(order) + self.target_batch - 1) // self.target_batch
        self.batches = tuple(tuple(map(int, x)) for x in np.array_split(order, n_batches))

    def __iter__(self):
        # DataLoader consumes this in its parent; no Dataset mutation is needed.
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def init_worker(_worker_id):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class ClipInput(nn.Module):
    """Shared [N,16,H,W,3] uint8 -> [N,6,16,H,W] FP32, on chosen device."""

    def __init__(self, mode):
        super().__init__()
        if mode not in ("rgb_zero", "rgb_diff"):
            raise ValueError("Unknown input mode")
        self.mode = mode
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1, 1))

    def forward(self, u8):
        if u8.dtype != torch.uint8 or u8.ndim != 5 or u8.shape[1] != 16 or u8.shape[-1] != 3:
            raise ValueError("Expected uint8 [N,16,H,W,3]")
        rgb = u8.permute(0, 4, 1, 2, 3).float().div_(255.0)
        rgb = (rgb - self.mean) / self.std
        diff = torch.zeros_like(rgb)
        if self.mode == "rgb_diff":
            diff[:, :, 1:] = rgb[:, :, 1:] - rgb[:, :, :-1]
        return torch.cat((rgb, diff), dim=1).contiguous()


def _check_model_state(net):
    if not net.training:
        raise ValueError("Training model must be in train mode")
    if any(not p.requires_grad for p in net.parameters()):
        raise ValueError("This finetuning path requires all model parameters trainable")
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError("Microbatch equivalence requires frozen BN statistics")


def _fp32_device(net, transform):
    parameters = tuple(net.parameters())
    if not parameters:
        raise ValueError("Model has no parameters")
    device = parameters[0].device
    for tensor in (*parameters, *net.buffers(), *transform.buffers()):
        if tensor.device != device or (tensor.is_floating_point() and tensor.dtype != torch.float32):
            raise ValueError("Model and preprocessing must share one device and FP32 dtype")
    if device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError("FP32 execution requires both TF32 switches disabled and recorded by the caller")
    return device


def _check_optimizer(net, optimizer):
    expected = {id(p) for p in net.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer must cover every trainable model parameter exactly once")


def _forward(net, x, checkpoint_backbone):
    if not checkpoint_backbone:
        return net(x).reshape(-1)
    from torch.utils.checkpoint import checkpoint
    x.requires_grad_(True)
    h = checkpoint(net.backbone, x, use_reentrant=False)
    return net.head(h).reshape(-1)


def train_effective_batch(net, optimizer, u8, labels, transform, pos_weight,
                          microbatch=8, checkpoint_backbone=False):
    """Immediate backward per micro; exactly ONE optimizer update per batch.

    Loss is sum(weighted BCE) / actual effective batch length, including unequal
    final micros. No augmentation, AMP, clipping, scheduler or hidden resampling.
    Dropout is stochastic: splitting changes RNG consumption versus a monolithic
    head call. Paired methods must use the SAME split; only dropout-disabled or
    explicitly fixed-mask fixtures claim gradient equivalence to one full call.
    """
    _check_model_state(net)
    microbatch = _integer(microbatch, 1, "Microbatch")
    if len(u8) == 0 or len(labels) != len(u8):
        raise ValueError("Invalid effective batch")
    if labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be a binary vector")
    if not np.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("Both training classes and a fixed positive N/P are required")
    device = _fp32_device(net, transform)
    _check_optimizer(net, optimizer)
    target = labels.to(device, dtype=torch.float32, non_blocking=True)
    weight = torch.tensor(pos_weight, device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for start in range(0, len(u8), microbatch):
        stop = min(start + microbatch, len(u8))
        pixels = u8[start:stop].to(device, non_blocking=True)
        # An outer caller's autocast context must not silently alter this run.
        with torch.autocast(device_type=device.type, enabled=False):
            logits = _forward(net, transform(pixels), checkpoint_backbone)
            chunk_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target[start:stop], pos_weight=weight, reduction="sum")
            loss_sum += chunk_loss.detach()
            (chunk_loss / len(u8)).backward()
        # Only this micro's graph exists; never accumulate graphs in a list.
        del chunk_loss, logits, pixels
    missing = [name for name, p in net.named_parameters() if p.grad is None]
    if missing:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"Trainable parameters received no gradient: {missing}")
    grads = [p.grad.detach().norm() for p in net.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("No parameter received a gradient")
    grad_norm = torch.stack(grads).norm()
    if not torch.isfinite(loss_sum) or not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite loss/gradient; optimizer update refused")
    optimizer.step()
    return {"loss": float(loss_sum.cpu()) / len(u8),
            "n": len(u8), "grad_norm": float(grad_norm.cpu()),
            "optimizer_steps": 1}


@torch.inference_mode()
def score_clips(net, u8, transform, microbatch=8):
    """Return raw logits on CPU; model/input work runs on the model's device."""
    microbatch = _integer(microbatch, 1, "Microbatch")
    if len(u8) == 0:
        raise ValueError("Expected a nonempty scoring batch")
    device = _fp32_device(net, transform)
    net.eval()
    output = []
    for start in range(0, len(u8), microbatch):
        with torch.autocast(device_type=device.type, enabled=False):
            x = transform(u8[start:start + microbatch].to(device, non_blocking=True))
            output.append(net(x).reshape(-1))
    result = torch.cat(output)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite prediction")
    return result.cpu()
