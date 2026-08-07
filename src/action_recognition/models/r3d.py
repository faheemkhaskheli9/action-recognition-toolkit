"""3D CNN baseline: ResNet-style spatiotemporal convolutions (Kinetics-pretrained).

Sees short-range motion directly through 3D kernels, unlike cnn_lstm which
only aggregates per-frame features after the fact.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .registry import register_model


@register_model("r3d18")
class R3D18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = torchvision.models.video.R3D_18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.video.r3d_18(weights=weights)
        if freeze_backbone:
            for param in net.parameters():
                param.requires_grad = False
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dataset yields (B, T, C, H, W); this architecture wants (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)
