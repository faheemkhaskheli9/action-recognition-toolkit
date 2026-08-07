from pathlib import Path

import cv2
import numpy as np
import pytest

# Django wiring (webapp/ on sys.path, DJANGO_SETTINGS_MODULE) lives in the
# `pythonpath`/`DJANGO_SETTINGS_MODULE` pytest-django ini options in
# pyproject.toml — pytest-django resolves settings before any conftest.py
# runs, so it can't be done here.


def write_synthetic_video(path: Path, num_frames: int = 20, size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (size, size))
    for i in range(num_frames):
        frame = np.full((size, size, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    return write_synthetic_video(tmp_path / "clip.mp4")


@pytest.fixture
def class_folder_dataset(tmp_path) -> Path:
    root = tmp_path / "raw"
    for label, count in [("jump", 3), ("wave", 2)]:
        for i in range(count):
            write_synthetic_video(root / label / f"{label}_{i}.mp4")
    return root
