"""Numerical and input invariants for faster video execution, CPU/no rental."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.pts_clip_sampling import jpeg95_bytes, rgb112_from_jpeg
from experiments.common.video.supervised_runtime import (
    ClipDataset, ClipInput, EpochBatches, FrameRef, PixelCache,
    init_worker, score_clips, train_effective_batch,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv3d(6, 3, 1), nn.BatchNorm3d(3),
                                      nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        return self.head(self.backbone(x))

    def train(self, mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        return self


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_unequal_micro_gradient_and_adam_update_match_full_batch(self):
        torch.manual_seed(71)
        for count in (1, 7, 10, 24):
            reference = TinyNet().train()
            actual = deepcopy(reference).train()
            u8 = torch.randint(0, 256, (count, 16, 4, 5, 3), dtype=torch.uint8)
            y = (torch.arange(count) % 2).float()
            transform = ClipInput('rgb_diff')
            opt_ref = torch.optim.Adam(reference.parameters(), lr=1e-3)
            opt_actual = torch.optim.Adam(actual.parameters(), lr=1e-3)
            bn_before = actual.backbone[1].running_mean.clone()
            for _ in range(2):
                opt_ref.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    reference(transform(u8)).flatten(), y,
                    pos_weight=torch.tensor(0.37), reduction='mean')
                loss.backward()
                opt_ref.step()
                result = train_effective_batch(actual, opt_actual, u8, y, transform,
                                               0.37, microbatch=3)
                self.assertAlmostEqual(result['loss'], loss.item(), places=6)
                for p_ref, p_actual in zip(reference.parameters(), actual.parameters()):
                    # FP32 Conv3d/reduction ordering changes with batch shape;
                    # compare numeric agreement, not bitwise identity.
                    torch.testing.assert_close(p_actual.grad, p_ref.grad, rtol=1e-4, atol=2e-6)
                    torch.testing.assert_close(p_actual, p_ref, rtol=1e-4, atol=2e-6)
                for state in opt_actual.state.values():
                    self.assertEqual(int(state['step']), _ + 1)
            self.assertTrue(torch.equal(bn_before, actual.backbone[1].running_mean))

    def test_live_bn_refused_and_nonfinite_does_not_update(self):
        net = TinyNet().train()
        opt = torch.optim.Adam(net.parameters())
        u8 = torch.zeros((2, 16, 3, 4, 3), dtype=torch.uint8)
        y = torch.tensor([0., 1.])
        net.backbone[1].train()
        with self.assertRaisesRegex(ValueError, 'frozen BN'):
            train_effective_batch(net, opt, u8, y, ClipInput('rgb_zero'), 1)
        net.train()
        with torch.no_grad():
            net.head.weight.fill_(float('nan'))
        before = deepcopy(opt.state_dict())
        with self.assertRaises(FloatingPointError):
            train_effective_batch(net, opt, u8, y, ClipInput('rgb_zero'), 1)
        self.assertEqual(before, opt.state_dict())

    def test_signed_temporal_difference_zero_first_and_reverse_sign(self):
        u8 = torch.arange(16, dtype=torch.uint8).view(1, 16, 1, 1, 1).expand(1, 16, 2, 3, 3)
        inputs = ClipInput('rgb_diff')(u8)
        self.assertTrue(torch.equal(inputs[:, 3:, 0], torch.zeros_like(inputs[:, 3:, 0])))
        torch.testing.assert_close(inputs[:, 3:, 1:], inputs[:, :3, 1:] - inputs[:, :3, :-1])
        reverse = ClipInput('rgb_diff')(u8.flip(1))
        torch.testing.assert_close(reverse[:, 3:, 1:], -inputs[:, 3:, 1:].flip(2))
        self.assertEqual(ClipInput('rgb_zero')(u8)[:, 3:].count_nonzero(), 0)

    def test_tail_balancing_preserves_order_and_updates(self):
        for n in (1, 23, 24, 25, 49, 504):
            sampler = EpochBatches(24)
            order = list(reversed(range(n)))
            sampler.set_order(order)
            batches = list(sampler)
            self.assertEqual([i for b in batches for i in b], order)
            sizes = [len(b) for b in batches]
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            self.assertLessEqual(max(sizes), 24)
        self.assertEqual(len(batches), 21)

    def test_invalid_label_and_fractional_indices_never_coerced(self):
        refs = (FrameRef('unused', '0' * 64, '0' * 64),) * 16
        for bad in (0.5, 1.9, -0.1, '1', float('nan')):
            with self.assertRaises(ValueError):
                ClipDataset([('clip', refs, bad)])
        with self.assertRaises(ValueError):
            EpochBatches().set_order([0, 1.9])
        for bad in (1.5, True):
            with self.assertRaises(ValueError):
                EpochBatches(bad)
        sampler = EpochBatches()
        sampler.set_order([np.int64(1), np.int64(2)])
        self.assertEqual(list(sampler), [(1, 2)])

    def test_fp32_enforced_inside_autocast_and_optimizer_coverage(self):
        net = TinyNet().train()
        transform = ClipInput('rgb_diff')
        u8 = torch.randint(0, 256, (3, 16, 3, 4, 3), dtype=torch.uint8)
        y = torch.tensor([0., 1., 1.])
        dtypes = []
        handle = net.backbone[0].register_forward_hook(lambda _m, _i, o: dtypes.append(o.dtype))
        opt = torch.optim.Adam(net.parameters())
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            train_effective_batch(net, opt, u8, y, transform, 0.5, 2)
            score_clips(net, u8, transform, 2)
        handle.remove()
        self.assertTrue(dtypes)
        self.assertEqual(set(dtypes), {torch.float32})
        net.train()
        incomplete = torch.optim.Adam(net.head.parameters())
        with self.assertRaisesRegex(ValueError, 'every trainable'):
            train_effective_batch(net, incomplete, u8, y, transform, 0.5)
        net.backbone[1].weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'all model parameters trainable'):
            train_effective_batch(net, opt, u8, y, transform, 0.5)
        net.backbone[1].weight.requires_grad_(True)
        net.unused = nn.Parameter(torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, 'received no gradient'):
            train_effective_batch(net, torch.optim.Adam(net.parameters()), u8, y, transform, 0.5)
        del net.unused
        net.double()
        with self.assertRaisesRegex(ValueError, 'FP32'):
            score_clips(net, u8, transform)

    def test_optional_checkpoint_matches_regular_micro_gradients(self):
        torch.manual_seed(92)
        regular = TinyNet().train()
        checkpointed = deepcopy(regular).train()
        u8 = torch.randint(0, 256, (5, 16, 3, 4, 3), dtype=torch.uint8)
        y = torch.tensor([0., 1., 1., 0., 1.])
        transform = ClipInput('rgb_diff')
        a = torch.optim.Adam(regular.parameters(), lr=1e-3)
        b = torch.optim.Adam(checkpointed.parameters(), lr=1e-3)
        train_effective_batch(regular, a, u8, y, transform, 0.5, 2)
        train_effective_batch(checkpointed, b, u8, y, transform, 0.5, 2, True)
        for pa, pb in zip(regular.parameters(), checkpointed.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad)
            torch.testing.assert_close(pa, pb)

    def test_inference_matches_full_eval_and_frozen_bn(self):
        torch.manual_seed(17)
        net = TinyNet().eval()
        transform = ClipInput('rgb_diff')
        u8 = torch.randint(0, 256, (7, 16, 4, 5, 3), dtype=torch.uint8)
        with torch.no_grad():
            expected = net(transform(u8)).flatten()
        torch.testing.assert_close(score_clips(net, u8, transform, 3), expected)

    def test_cache_integrity_budget_and_persistent_worker_indices(self):
        temp_root = ROOT / '.codex/docs/.temp/runtime-tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            refs = []
            for i in range(4):
                bgr = np.full((256, 256, 3), 35 + i * 20, dtype=np.uint8)
                data = jpeg95_bytes(bgr)
                path = Path(folder) / f'{i}.jpg'
                path.write_bytes(data)
                refs.append(FrameRef(str(path), hashlib.sha256(data).hexdigest(),
                                     hashlib.sha256(rgb112_from_jpeg(data).tobytes()).hexdigest()))
            cache = PixelCache(112 * 112 * 3)
            a = cache.get(refs[0])
            self.assertIs(cache.get(refs[0]), a)
            cache.get(refs[1])
            self.assertEqual(cache.used_bytes, a.nbytes)
            self.assertEqual(len(cache.entries), 1)
            bad = FrameRef(refs[0].path, '0' * 64, refs[0].rgb112_sha256)
            with self.assertRaisesRegex(ValueError, 'JPEG fingerprint'):
                cache.get(bad)
            rows = [(f'clip-{i}', (ref,) * 16, i % 2) for i, ref in enumerate(refs)]
            ds = ClipDataset(rows, cache_bytes=2 * a.nbytes)
            batches = EpochBatches(2)
            batches.set_order([0, 2])
            ld = DataLoader(ds, batch_sampler=batches, num_workers=2,
                            persistent_workers=True, worker_init_fn=init_worker)
            self.assertEqual(torch.cat([item[2] for item in ld]).tolist(), [0, 2])
            batches.set_order([3, 1])
            self.assertEqual(torch.cat([item[2] for item in ld]).tolist(), [3, 1])
            del ld


if __name__ == '__main__':
    unittest.main()
