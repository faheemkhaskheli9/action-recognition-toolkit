"""Baseline: a 2D CNN backbone (per-frame) feeding an LSTM over time.

Cheap to train from scratch on small custom datasets since only the LSTM +
classifier head need to learn from little data when the backbone is pretrained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .registry import register_model


@register_model("cnn_lstm")
class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = torchvision.models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(feature_dim, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        features = self.backbone(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        output, _ = self.lstm(features)
        last_step = output[:, -1, :]
        return self.classifier(last_step)
