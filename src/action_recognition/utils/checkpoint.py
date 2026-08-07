from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, model, label_map: dict, config: dict, epoch: int, metric: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_map": label_map,
            "config": config,
            "epoch": epoch,
            "metric": metric,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    # weights_only=False: these are checkpoints this project produces itself (contain
    # a label_map/config dict alongside tensors), not third-party weights.
    return torch.load(Path(path), map_location=map_location, weights_only=False)
