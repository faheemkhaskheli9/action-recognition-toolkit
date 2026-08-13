"""Manifest = the single source of truth mapping video files to labels (and splits).

A manifest is a CSV with columns: video_path, label, split
`split` is one of {train, val, test} and is added by `stratified_split`;
it's absent right after labeling.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MANIFEST_COLUMNS = ["video_path", "label"]
SPLITS = ("train", "val", "test")
# Optional columns a row may carry to label just a span of a longer source
# video (seconds, source-file-relative) instead of the whole file. Absent or
# blank means "whole video" -- see data.dataset.read_clip_frames.
SPAN_COLUMNS = ["start_time", "end_time"]


def discover_videos(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def discover_class_folders(root: Path) -> pd.DataFrame:
    """Build a manifest from a `<root>/<class_name>/<video>` folder layout."""
    root = Path(root)
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for video in discover_videos(class_dir):
            rows.append({"video_path": str(video.resolve()), "label": class_dir.name})
    if not rows:
        raise ValueError(
            f"No videos found under {root}. Expected '<root>/<class_name>/<video file>'."
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} is missing required column(s): {sorted(missing)}")
    return df


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_label_map(labels: Iterable[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(sorted(set(labels)))}


def save_label_map(label_map: dict[str, int], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_map, indent=2, sort_keys=False))


def load_label_map(path: Path) -> dict[str, int]:
    return json.loads(Path(path).read_text())


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a `split` column, stratified per-label so every class appears in every split
    (when it has enough examples)."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    df = df.copy()
    df["split"] = "train"

    for label, group in df.groupby("label"):
        shuffled = group.sample(frac=1.0, random_state=seed)
        n = len(shuffled)
        n_val = max(1, round(n * val_frac)) if n > 1 else 0
        n_test = max(1, round(n * test_frac)) if n > 1 else 0
        # never take the whole class out of train
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - 1 - n_val))

        val_idx = shuffled.index[:n_val]
        test_idx = shuffled.index[n_val : n_val + n_test]
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    return df
