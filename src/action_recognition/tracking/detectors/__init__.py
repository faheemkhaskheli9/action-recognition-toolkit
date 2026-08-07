from .registry import available_detectors, build_detector, register_detector

# import registers each backend as a side effect
from . import torchvision_det  # noqa: F401,E402

__all__ = ["build_detector", "available_detectors", "register_detector"]
