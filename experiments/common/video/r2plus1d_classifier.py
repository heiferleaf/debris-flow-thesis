"""One shared six-channel R(2+1)D classifier for training and prediction."""
from __future__ import annotations
import hashlib
import torch
from torch import nn


class VideoClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.backbone.fc = nn.Identity()
        old = self.backbone.stem[0]
        new = nn.Conv3d(6, old.out_channels, old.kernel_size, old.stride,
                        old.padding, dilation=old.dilation, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
        self.backbone.stem[0] = new
        self.head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(32, 1))

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, x):
        return self.head(self.backbone(x)).reshape(-1)


def state_digest(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode('utf-8'))
        h.update(str(value.dtype).encode('ascii'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def configure_fp32(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
