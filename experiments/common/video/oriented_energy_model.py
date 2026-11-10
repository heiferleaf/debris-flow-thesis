"""EXP021 linear RGB component + exactly112 zero-initialized energy weights."""
from pathlib import Path
import torch
from torch import nn

from experiments.common.video.head_comparison_training import LinearVideoClassifier, require_local_weights
from experiments.common.video.r2plus1d_classifier import state_digest
from experiments.common.video.oriented_energy import OrientedEnergy, read_artifact

RGB_INITIAL_SHA = 'f3b66df8d405b5c1bbd00bb77f6f04e6168a5b858918ae720c2acc706775d288'
BACKBONE_INITIAL_SHA = '0093965b8b24276e58332f96fa152da970c04ac130f08cc3ad9b90b183ba0143'


class EnergyHead(nn.Module):
    def __init__(self, rgb):
        super().__init__()
        self.rgb = rgb
        self.energy_weight = nn.Parameter(torch.zeros(112))

    def forward(self, features, description):
        if (features.ndim != 2 or features.shape[1] != self.rgb.in_features
                or description.ndim != 2 or tuple(description.shape) != (len(features), 112)
                or not torch.isfinite(description).all()):
            raise ValueError('Energy description must be finite[N,112], with no batch broadcasting')
        return self.rgb(features).reshape(-1) + description.matmul(self.energy_weight)


class EnergyClassifier(LinearVideoClassifier):
    def __init__(self, kernel_path, kernel_sha256, mode, pretrained=True):
        if mode not in ('B', 'C'):
            raise ValueError('Only the two preregistered energy modes')
        if pretrained:
            require_local_weights()
        super().__init__(pretrained=pretrained)
        # Zeros/buffer copies consume no random numbers after original A init.
        self.head = EnergyHead(self.head)
        self.energy = OrientedEnergy(read_artifact(kernel_path, kernel_sha256))
        self.energy_mode = mode
        self.kernel_sha256 = kernel_sha256

    def rgb_component_sha256(self):
        view = nn.Module()
        view.backbone = self.backbone
        view.head = self.head.rgb
        return state_digest(view)

    def check_fresh(self):
        if (self.rgb_component_sha256() != RGB_INITIAL_SHA or state_digest(self.backbone) != BACKBONE_INITIAL_SHA
                or torch.count_nonzero(self.head.energy_weight).item() != 0
                or sum(p.numel() for p in self.head.parameters()) != 625):
            raise ValueError('Production RGB component/zero energy initialization differs from registered A')

    def forward(self, x, descriptions=None):
        description = self.energy(x, (self.energy_mode,))[self.energy_mode] if descriptions is None else descriptions[self.energy_mode]
        return self.head(self.backbone(x), description).reshape(-1)
