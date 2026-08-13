"""Shared torch device resolution: cuda > mps > cpu unless a specific
device is requested (CLI --device flag / config `device` field). Used by
training, inference, and the detect/track pipeline, so "a GPU is available
but nothing asked for it" doesn't silently mean everything runs on CPU.
"""
from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
