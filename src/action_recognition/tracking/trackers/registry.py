"""Every tracker registers itself here under a short name used in tracking
configs (`configs/tracking/*.yaml`).

To add a new tracker backend: write a class with an
`update(frame_idx, detections)` method called once per (possibly
subsampled) frame in increasing `frame_idx` order, and a
`finished_tracks() -> list[Track]` method that flushes and returns every
track once the video is done. Decorate it with @register_tracker("your_name")
and import the module from `trackers/__init__.py`.
"""
from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register_tracker(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"Tracker name '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_tracker(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown tracker '{name}'. Available: {available_trackers()}")
    return _REGISTRY[name](**kwargs)


def available_trackers() -> list[str]:
    return sorted(_REGISTRY)
