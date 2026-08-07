"""Frame-level transforms applied identically across a clip's sampled frames."""
from __future__ import annotations

import torch
from torchvision.transforms import v2 as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 112, train: bool = False) -> T.Compose:
    ops = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        ops += [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops += [
            T.Resize(int(image_size * 1.15), antialias=True),
            T.CenterCrop(image_size),
        ]
    ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)
