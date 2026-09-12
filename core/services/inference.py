from __future__ import annotations

from pathlib import Path

from action_recognition.inference.predict import predict as run_predict
from action_recognition.inference.scene_predict import predict_scene as run_predict_scene

from ..models import TrainingRun


def checkpoint_choices() -> list[tuple[str, str]]:
    """(value, label) pairs for every best.pt/last.pt produced by a known run."""
    choices = []
    for run in TrainingRun.objects.all():
        output_dir = Path(run.output_dir)
        for name in ("best.pt", "last.pt"):
            path = output_dir / name
            if path.exists():
                choices.append((str(path), f"{run.name} / {name}"))
    return choices


def predict_video(checkpoint_path: Path, video_path: Path, top_k: int = 3):
    return run_predict(checkpoint_path, video_path, top_k=top_k)


def predict_scene(checkpoint_path: Path, video_path: Path) -> list[dict]:
    """Per-person, per-time-window predictions for a raw multi-person video —
    see action_recognition.inference.scene_predict for the detect+track+
    classify pipeline this wraps."""
    return run_predict_scene(checkpoint_path, video_path)
