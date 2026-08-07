"""Every architecture registers itself here under a short name used in config files.

To add a new architecture: write a nn.Module in this package whose forward()
accepts (B, T, C, H, W) and returns (B, num_classes) logits, decorate its class
with @register_model("your_name"), and import the module below.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Model name '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, num_classes: int, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")
    return _REGISTRY[name](num_classes=num_classes, **kwargs)


def available_models() -> list[str]:
    return sorted(_REGISTRY)
