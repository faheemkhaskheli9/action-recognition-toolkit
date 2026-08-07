from .registry import available_models, build_model, register_model

# import registers each architecture as a side effect
from . import cnn_lstm, r3d  # noqa: F401,E402

__all__ = ["build_model", "available_models", "register_model"]
