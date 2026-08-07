"""Every person detector registers itself here under a short name used in
tracking configs (`configs/tracking/*.yaml`).

To add a new detector backend: write a class with a
`detect(frame) -> list[Detection]` method (frame: HWC uint8 RGB numpy
array), decorate it with @register_detector("your_name"), and import the
module from `detectors/__init__.py`.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register_detector(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Detector name '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_detector(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Available: {available_detectors()}")
    return _REGISTRY[name](**kwargs)


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)
