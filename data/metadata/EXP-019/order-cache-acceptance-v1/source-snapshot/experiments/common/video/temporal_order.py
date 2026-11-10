"""EXP-019 physical-window order: stdlib only, no mutable state or RNG calls."""
from __future__ import annotations
import hashlib
import json
from numbers import Integral
import re

VERSION = 'exp019-temporal-order-v1'
PERMUTATION_SEED = 1901
MODES = ('chronological', 'fixed_permutation')


def _canonical(media_sha256, frame_indices):
    if not isinstance(media_sha256, str) or re.fullmatch(r'[0-9a-fA-F]{64}', media_sha256) is None:
        raise ValueError('Expected the actual middleware SHA256 as64 hexadecimal characters')
    indices = tuple(frame_indices)
    if (len(indices) != 16 or any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:]))):
        raise ValueError('Expected16 strictly increasing nonnegative canonical physical frame indices')
    return media_sha256.lower(), tuple(map(int, indices))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def physical_window_json(media_sha256, frame_indices):
    media, indices = _canonical(media_sha256, frame_indices)
    return _json(dict(version=VERSION, permutation_seed=PERMUTATION_SEED,
                      media_sha256=media, frame_indices=list(indices)))


def _position_key(canonical_bytes, position):
    return hashlib.sha256(canonical_bytes + b'|position|' + str(position).encode('ascii')).digest()


def temporal_indices(media_sha256, frame_indices, mode):
    if mode not in MODES: raise ValueError('Unknown EXP-019 temporal input mode')
    canonical = physical_window_json(media_sha256, frame_indices).encode('utf-8')
    if mode == 'chronological': return tuple(range(16))
    return tuple(sorted(range(16), key=lambda i: (_position_key(canonical, i), i)))


def _sequence_sha(media, canonical_indices, positions, mode):
    value = dict(version=VERSION, permutation_seed=PERMUTATION_SEED, mode=mode,
                 media_sha256=media, ordered_frame_indices=[canonical_indices[i] for i in positions])
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def window_order_record(media_sha256, frame_indices):
    media, indices = _canonical(media_sha256, frame_indices)
    canonical = physical_window_json(media, indices)
    permutation = temporal_indices(media, indices, 'fixed_permutation')
    return dict(version=VERSION, permutation_seed=PERMUTATION_SEED,
                canonical_json=canonical, physical_window_sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
                permutation=list(permutation),
                chronological_sequence_sha256=_sequence_sha(media, indices, range(16), 'chronological'),
                fixed_permutation_sequence_sha256=_sequence_sha(media, indices, permutation, 'fixed_permutation'))


def apply_window_order(value, media_sha256, frame_indices, mode, time_axis=0):
    """Return one NumPy/Torch temporal view/copy; canonical metadata is untouched.

    Unbatched uint8[T,H,W,RGB] uses axis0; normalized[channels,T,H,W] uses
    axis1. A batched caller can obtain per-window positions via temporal_indices
    and perform its native gather. No tensor library is imported by this module.
    """
    if not hasattr(value, 'ndim') or not isinstance(time_axis, Integral) or isinstance(time_axis, bool):
        raise ValueError('Expected an array/tensor and integer temporal axis')
    if not -value.ndim <= time_axis < value.ndim: raise ValueError('Temporal axis outside input shape')
    axis = int(time_axis) % value.ndim
    if value.shape[axis] != 16: raise ValueError('Temporal input dimension must equal16')
    positions = temporal_indices(media_sha256, frame_indices, mode)
    slices = [slice(None)] * value.ndim; slices[axis] = list(positions)
    return value[tuple(slices)]
