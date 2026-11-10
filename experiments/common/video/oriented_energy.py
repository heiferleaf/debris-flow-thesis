"""Fixed EXP021 centered directional Gaussian energy; no labels or learned state.

The direct FP64 reference uses the frozen full 3-D kernels. The separable
implementation expands each directional derivative into the ten degree-three
Cartesian Gaussian-derivative products. B pools squared slice contributions;
C pools their squared sum. No descriptor cache or cross-pixel-source reuse.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'centered-g3-27x4-valid7-regions2-eps-quantization-v1'
CARD_SHA = 'da613e7e44fbf46e1029690bcc649715cfc1513d6665d0026228eccf30254394'
BASIS = tuple((x, y, 3 - x - y) for x in range(4) for y in range(4 - x))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def build_kernels():
    """Exact diagonal trigonometric ties preserve the specified x,y,t tie order."""
    h = math.sqrt(.5)
    circle = [(1., 0.), (h, h), (0., 1.), (-h, h), (-1., 0.), (-h, -h), (0., -1.), (h, -h)]
    normals = [np.array([0., 0., 1.])]
    normals += [np.array([s * x, s * y, 1.]) / math.sqrt(s * s + 1.) for s in (.5, 1., 2.) for x, y in circle]
    normals += [np.array([1., 0., 0.]), np.array([0., 1., 0.])]
    directions = []
    for n in normals:
        axis = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = axis - np.dot(axis, n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        directions.extend([u, h * (u + v), v, h * (-u + v)])
    normals, directions = np.array(normals), np.array(directions)
    r = np.arange(-3, 4, dtype=np.float64)
    tt, yy, xx = np.meshgrid(r, r, r, indexing='ij')
    gaussian = (2 * math.pi) ** -1.5 * np.exp(-(xx ** 2 + yy ** 2 + tt ** 2) / 2)
    dot = directions[:, 0, None, None, None] * xx + directions[:, 1, None, None, None] * yy + directions[:, 2, None, None, None] * tt
    raw = (3 * dot - dot ** 3) * gaussian
    scale = float(1 / np.sqrt(np.mean(np.sum(raw ** 2, axis=(1, 2, 3)))))
    kernels = raw * scale
    g = np.exp(-r ** 2 / 2) / math.sqrt(2 * math.pi)
    derivatives = np.array([g, -r * g, (r ** 2 - 1) * g, (3 * r - r ** 3) * g])
    spatial = np.array([derivatives[y, :, None] * derivatives[x, None, :] for x, y, t in BASIS])
    temporal = np.array([derivatives[t] for x, y, t in BASIS])
    coefficients = np.array([[math.factorial(3) / (math.factorial(x) * math.factorial(y) * math.factorial(t))
        * m[0] ** x * m[1] ** y * m[2] ** t for x, y, t in BASIS] for m in directions]) * scale
    reconstructed = np.einsum('ok,kt,kyx->otyx', coefficients, temporal, spatial)
    if not np.allclose(reconstructed, kernels, atol=1e-14, rtol=1e-12):
        raise ValueError('Separable Gaussian derivative expansion differs from direct formula')
    q2 = (.299 ** 2 + .587 ** 2 + .114 ** 2) / (12 * 255 ** 2)
    epsilon = q2 * float(np.sum(kernels ** 2))
    return dict(kernels=kernels, normals=normals, directions=directions, spatial=spatial, temporal=temporal,
                coefficients=coefficients, global_scale=np.array(scale), q2=np.array(q2), epsilon=np.array(epsilon))


def write_artifact(directory):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    data = build_kernels()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'kernels.npz').open('xb') as f:
        np.savez(f, **data)
    manifest = dict(version=VERSION, card_sha256=CARD_SHA, artifact_sha256=sha(directory / 'kernels.npz'),
        source_sha256=sha(Path(__file__).resolve()), sigma=1, support=[-3, 3], planes=27, directions_per_plane=4,
        common_global_scale=float(data['global_scale']), mean_kernel_squared_norm=float(np.mean(np.sum(data['kernels'] ** 2, axis=(1, 2, 3)))),
        maximum_absolute_kernel_DC=float(np.max(np.abs(data['kernels'].sum(axis=(1, 2, 3))))),
        q2=float(data['q2']), epsilon=float(data['epsilon']), epsilon_fitted=False,
        plane_order='static; speeds0.5,1,2 each theta0..7*pi/4; infinite-horizontal; infinite-vertical',
        region_order=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        channel_order='each region:27 normalized plane energies then epsilon fraction',
        input='inverse fixed ClipInput RGB normalization and clamp[0,1]; gray0.299R+0.587G+0.114B; subtract ONE clip-global mean; no clamp after centering',
        output_valid_shape=[10, 106, 106], descriptor_dimensions=112,
        constant_energy_zero_but_stable_fraction_one=True,
        array_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in data.items()})
    with (directory / 'manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return manifest


def read_artifact(path, expected_sha=None):
    path = Path(path)
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('Fixed kernel artifact SHA differs')
    with np.load(str(path), allow_pickle=False) as f:
        data = {name: f[name].copy() for name in f.files}
    shapes = dict(kernels=(108, 7, 7, 7), normals=(27, 3), directions=(108, 3), spatial=(10, 7, 7),
                  temporal=(10, 7), coefficients=(108, 10), global_scale=(), q2=(), epsilon=())
    if set(data) != set(shapes) or any(data[k].shape != shape or data[k].dtype != np.float64
            or not np.isfinite(data[k]).all() for k, shape in shapes.items()) or data['epsilon'] <= 0:
        raise ValueError('Malformed fixed kernel artifact')
    return data


def pool_planes(response):
    """[N,108,T,H,W] -> [N,4,27], mean over valid T and each exact region."""
    n, channels, t, h, w = response.shape
    if channels != 108 or h % 2 or w % 2:
        raise ValueError('Require108 directions and exactly even2x2 spatial partition')
    return response.reshape(n, 27, 4, t, 2, h // 2, 2, w // 2).square().mean((3, 5, 7)).sum(2).permute(0, 2, 3, 1).reshape(n, 4, 27)


def normalize_energy(energy, epsilon):
    if not torch.isfinite(energy).all() or (energy < 0).any():
        raise FloatingPointError('Nonfinite/negative squared energy')
    denominator = energy.sum(-1, keepdim=True) + epsilon
    return torch.cat((energy / denominator, epsilon.expand_as(denominator) / denominator), dim=-1).reshape(len(energy), 112)


def reference_energies(gray, data, center=True):
    """Independent direct3D C and direct spatial-slice B; intended for tiny FP64 fixtures."""
    gray = gray.to(dtype=torch.float64)
    if center:
        gray = gray - gray.mean((2, 3, 4), keepdim=True)
    kernels = torch.as_tensor(data['kernels'], dtype=gray.dtype, device=gray.device)
    full = F.conv3d(gray, kernels[:, None])
    b = torch.zeros((len(gray), 4, 27), dtype=gray.dtype, device=gray.device)
    for tau in range(7):
        contribution = F.conv3d(gray[:, :, tau:tau + full.shape[2]], kernels[:, None, tau:tau + 1])
        b += pool_planes(contribution)
    return {'B': b, 'C': pool_planes(full)}


class OrientedEnergy(nn.Module):
    def __init__(self, data, dtype=torch.float32):
        super().__init__()
        self.register_buffer('spatial', torch.as_tensor(data['spatial'], dtype=dtype)[:, None])
        self.register_buffer('temporal', torch.as_tensor(data['temporal'], dtype=dtype)[:, None, :, None, None])
        self.register_buffer('coefficients', torch.as_tensor(data['coefficients'], dtype=dtype))
        self.register_buffer('epsilon', torch.as_tensor(data['epsilon'], dtype=dtype))

    def energies(self, gray, modes=('B', 'C'), center=True):
        if (gray.ndim != 5 or gray.shape[1] != 1 or min(gray.shape[2:]) < 7 or not torch.isfinite(gray).all()
                or not modes or len(set(modes)) != len(modes) or set(modes) - {'B', 'C'}):
            raise ValueError('Invalid finite grayscale shape or energy modes')
        if center:
            wide = gray.to(dtype=torch.float64)
            gray = (wide - wide.mean((2, 3, 4), keepdim=True)).to(dtype=gray.dtype)
        n, _, t, h, w = gray.shape
        spatial = F.conv2d(gray[:, 0].reshape(n * t, 1, h, w), self.spatial)
        spatial = spatial.reshape(n, t, 10, h - 6, w - 6).permute(0, 2, 1, 3, 4).contiguous()
        output = {}
        if modes == ('C',) or set(modes) == {'C'}:
            bases = F.conv3d(spatial, self.temporal, groups=10)
            response = F.conv3d(bases, self.coefficients[:, :, None, None, None])
            output['C'] = pool_planes(response)
        else:
            b, full = None, None
            for tau in range(7):
                weights = self.coefficients * self.temporal[:, 0, tau, 0, 0][None]
                contribution = F.conv3d(spatial[:, :, tau:tau + t - 6], weights[:, :, None, None, None])
                if 'B' in modes:
                    current = pool_planes(contribution)
                    b = current if b is None else b + current
                if 'C' in modes:
                    full = contribution if full is None else full + contribution
            if 'B' in modes:
                output['B'] = b
            if 'C' in modes:
                output['C'] = pool_planes(full)
        return output

    def from_gray(self, gray, modes=('B', 'C'), center=True):
        return {mode: normalize_energy(e, self.epsilon) for mode, e in self.energies(gray, modes, center).items()}

    @torch.no_grad()
    def forward(self, normalized, modes=('B', 'C')):
        from experiments.common.video.supervised_runtime import MEAN, STD
        if normalized.ndim != 5 or tuple(normalized.shape[1:]) != (6, 16, 112, 112):
            raise ValueError('Production descriptor requires actual6x16x112x112 ClipInput pixels')
        mean = normalized.new_tensor(MEAN).reshape(1, 3, 1, 1, 1)
        std = normalized.new_tensor(STD).reshape(1, 3, 1, 1, 1)
        rgb = (normalized[:, :3] * std + mean).clamp(0, 1)
        gray = (rgb * normalized.new_tensor([.299, .587, .114]).reshape(1, 3, 1, 1, 1)).sum(1, keepdim=True)
        return self.from_gray(gray, modes=modes, center=True)
