"""One bounded set of mathematical/new-fusion checks; no real clips or training."""
import copy
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.common.video.oriented_energy import (OrientedEnergy, build_kernels, normalize_energy,
    pool_planes, reference_energies, sha, write_artifact)
from experiments.common.video.oriented_energy_model import EnergyClassifier, EnergyHead
from experiments.common.video.head_comparison_training import LinearVideoClassifier, require_local_weights
from experiments.common.video.r2plus1d_classifier import configure_fp32, state_digest
from experiments.common.video.supervised_runtime import ClipInput


class EnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.data = build_kernels()
        cls.op = OrientedEnergy(cls.data, dtype=torch.float64)
        parent = ROOT / '.codex/docs/.temp'; parent.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='energy-fixture-', dir=str(parent))
        cls.path = Path(cls.temp.name) / 'kernels'
        write_artifact(cls.path)
        cls.kernel = cls.path / 'kernels.npz'; cls.kernel_sha = sha(cls.kernel)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_direct_FP64_and_separable_B_C_with_fixed_normalization(self):
        gray = torch.randn((2, 1, 9, 10, 10), generator=torch.Generator().manual_seed(2101), dtype=torch.float64)
        reference = reference_energies(gray, self.data)
        optimized = self.op.energies(gray)
        single_c = self.op.energies(gray, ('C',))
        for mode in ('B', 'C'):
            torch.testing.assert_close(optimized[mode], reference[mode], atol=1e-7, rtol=1e-5)
            a = normalize_energy(optimized[mode], self.op.epsilon)
            b = normalize_energy(reference[mode], self.op.epsilon)
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
            self.assertEqual(a.shape, (2, 112))
            torch.testing.assert_close(a.reshape(2, 4, 28).sum(-1), torch.ones(2, 4, dtype=torch.float64), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(single_c['C'], optimized['C'], atol=1e-7, rtol=1e-5)
        self.assertAlmostEqual(np.mean(np.sum(self.data['kernels'] ** 2, axis=(1, 2, 3))), 1., places=13)
        self.assertLess(np.max(np.abs(self.data['kernels'].sum((1, 2, 3)))), 1e-13)

    def test_centered_cross_term_identity(self):
        g = torch.randn((1, 1, 9, 10, 10), generator=torch.Generator().manual_seed(2102), dtype=torch.float64)
        g = g - g.mean((2, 3, 4), keepdim=True)
        k = torch.from_numpy(self.data['kernels'])
        contributions = [F.conv3d(g[:, :, tau:tau + 3], k[:, None, tau:tau + 1]) for tau in range(7)]
        sum_a = sum(contributions)
        cross = torch.zeros_like(sum_a)
        for i in range(7):
            for j in range(i + 1, 7):
                cross += 2 * contributions[i] * contributions[j]
        def average_signed(x):
            return x.reshape(1, 27, 4, 3, 2, 2, 2, 2).mean((3, 5, 7)).sum(2).permute(0, 2, 3, 1).reshape(1, 4, 27)
        c, b = pool_planes(sum_a), sum(pool_planes(x) for x in contributions)
        torch.testing.assert_close(c - b, average_signed(cross), atol=1e-7, rtol=1e-5)

    def test_low_contrast_FP32_centering_meets_same_FP64_tolerance(self):
        g = (torch.randn((2, 1, 9, 10, 10), generator=torch.Generator().manual_seed(17121), dtype=torch.float64) * 1e-4 + .4).float()
        reference = reference_energies(g.double(), self.data)
        operator = OrientedEnergy(self.data)
        actual = operator.from_gray(g)
        for mode in ('B', 'C'):
            expected = normalize_energy(reference[mode], self.op.epsilon)
            torch.testing.assert_close(actual[mode].double(), expected, atol=1e-7, rtol=1e-5)

    def test_constant_brightness_shift_and_C_DC_cancellation(self):
        constant = torch.full((1, 1, 9, 10, 10), 12.25, dtype=torch.float64)
        for mode, e in self.op.energies(constant).items():
            self.assertEqual(torch.count_nonzero(e).item(), 0)
            d = normalize_energy(e, self.op.epsilon).reshape(1, 4, 28)
            self.assertEqual(torch.count_nonzero(d[:, :, :27]).item(), 0)
            self.assertTrue(torch.equal(d[:, :, 27], torch.ones(1, 4, dtype=torch.float64)))
        g = torch.randn((1, 1, 9, 10, 10), generator=torch.Generator().manual_seed(2103), dtype=torch.float64)
        a, shifted = self.op.from_gray(g), self.op.from_gray(g + 7.)
        for mode in ('B', 'C'):
            torch.testing.assert_close(a[mode], shifted[mode], atol=1e-7, rtol=1e-5)
        before = self.op.energies(g + 7., ('C',), center=False)['C']
        after = self.op.energies(g + 7., ('C',), center=True)['C']
        torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-5)

    def test_actual_shape_FP32_and_synthetic_positive_x_motion(self):
        t = torch.arange(16).reshape(16, 1, 1)
        x = torch.arange(112).reshape(1, 1, 112)
        pixels = ((.5 + .3 * torch.cos((x - t) * (2 * np.pi / 12))).expand(16, 112, 112) * 255).round().to(torch.uint8)
        u8 = pixels[None, :, :, :, None].expand(1, 16, 112, 112, 3).contiguous()
        operator = OrientedEnergy(self.data)
        started = time.perf_counter()
        both = operator(ClipInput('rgb_zero')(u8))
        self.assertEqual(set(both), {'B', 'C'})
        self.assertTrue(all(d.shape == (1, 112) and torch.isfinite(d).all() for d in both.values()))
        # Normal (1,0,1)/sqrt2 matches +x sampling motion; opposite direction is
        # weaker. This is not a unique physical-speed estimator/aperture proof.
        planes = both['C'].reshape(1, 4, 28)[0, :, :27].mean(0)
        self.assertGreater(planes[9].item(), planes[13].item())
        print('synthetic16x112 joint B/C descriptor seconds=%.4f' % (time.perf_counter() - started))

    def test_nonzero_description_zero_weight_preserves_RGB_logits_and_gradients(self):
        torch.manual_seed(2104)
        a = nn.Linear(512, 1)
        b = EnergyHead(copy.deepcopy(a))
        features_a = torch.randn(4, 512, requires_grad=True)
        features_b = features_a.detach().clone().requires_grad_()
        description = torch.rand(4, 112) + .1
        labels = torch.tensor([0., 1., 1., 0.])
        logit_a, logit_b = a(features_a).reshape(-1), b(features_b, description)
        self.assertTrue(torch.equal(logit_a, logit_b))
        F.binary_cross_entropy_with_logits(logit_a, labels).backward()
        F.binary_cross_entropy_with_logits(logit_b, labels).backward()
        for x, y in ((a.weight.grad, b.rgb.weight.grad), (a.bias.grad, b.rgb.bias.grad), (features_a.grad, features_b.grad)):
            self.assertTrue(torch.equal(x, y))
        self.assertGreater(b.energy_weight.grad.abs().sum().item(), 0)

    def test_energy_head_refuses_broadcast_and_nonfinite_description(self):
        head = EnergyHead(nn.Linear(512, 1))
        features = torch.ones(3, 512)
        for description in (torch.ones(112), torch.ones(1, 112), torch.full((3, 112), float('nan')),
                            torch.full((3, 112), float('inf'))):
            with self.assertRaisesRegex(ValueError, 'finite'):
                head(features, description)

    def test_model_initial_RGB_RNG_modes_and_optimizer_group(self):
        try:
            require_local_weights()
        except ValueError:
            self.skipTest('Existing exact K400 unavailable; never download')
        configure_fp32(7)
        a = LinearVideoClassifier(pretrained=True)
        rgb_sha, rng = state_digest(a), torch.get_rng_state().clone()
        del a
        configure_fp32(7)
        b = EnergyClassifier(self.kernel, self.kernel_sha, 'B')
        self.assertEqual(b.rgb_component_sha256(), rgb_sha)  # Local env observation, not cloud f3 proof.
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        initial = state_digest(b)
        self.assertEqual(sum(p.numel() for p in b.head.parameters()), 625)
        b.train()
        self.assertTrue(all(not m.training for m in b.backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)))
        optimizer = torch.optim.Adam([dict(params=b.backbone.parameters(), lr=1e-4), dict(params=b.head.parameters(), lr=1e-3)])
        self.assertFalse(optimizer.state)
        self.assertTrue(any(p is b.head.energy_weight for p in optimizer.param_groups[1]['params']))
        self.assertTrue(all(p.requires_grad for p in b.parameters()))
        del optimizer, b
        configure_fp32(7)
        c = EnergyClassifier(self.kernel, self.kernel_sha, 'C')
        self.assertEqual(state_digest(c), initial)
        self.assertEqual(c.rgb_component_sha256(), rgb_sha)


if __name__ == '__main__':
    unittest.main()
