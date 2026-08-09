"""Thin wrapper around action_recognition.data.manifest for the labeling views."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from action_recognition.data.manifest import (
    MANIFEST_COLUMNS,
    VIDEO_EXTENSIONS,
    discover_videos,
    read_manifest,
    write_manifest,
)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return read_manifest(manifest_path)
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def next_unlabeled(video_dir: Path, manifest: pd.DataFrame, skipped: set[str]) -> Path | None:
    labeled = set(manifest["video_path"]) if not manifest.empty else set()
    for video in discover_videos(video_dir):
        resolved = str(video.resolve())
        if resolved not in labeled and resolved not in skipped:
            return video
    return None


def known_labels(manifest: pd.DataFrame) -> list[str]:
    if manifest.empty:
        return []
    return sorted(manifest["label"].dropna().unique().tolist())


def save_label(manifest_path: Path, manifest: pd.DataFrame, video_path: Path, label: str) -> pd.DataFrame:
    row = pd.DataFrame([{"video_path": str(video_path.resolve()), "label": label}])
    updated = pd.concat([manifest, row], ignore_index=True)
    write_manifest(updated, manifest_path)
    return updated


def set_label(manifest_path: Path, manifest: pd.DataFrame, video_path: Path, label: str) -> pd.DataFrame:
    """Upsert a label for `video_path` — updates the row if it's already in the
    manifest, otherwise appends one. Used by the dataset CRUD page, where a
    video may already be labeled and just needs relabeling."""
    resolved = str(video_path.resolve())
    if not manifest.empty and resolved in set(manifest["video_path"]):
        updated = manifest.copy()
        updated.loc[updated["video_path"] == resolved, "label"] = label
        write_manifest(updated, manifest_path)
        return updated
    return save_label(manifest_path, manifest, video_path, label)


def delete_entry(
    manifest_path: Path, manifest: pd.DataFrame, video_path: Path, delete_file: bool = False
) -> pd.DataFrame:
    """Remove `video_path`'s row from the manifest (a no-op if it has none),
    optionally deleting the underlying video file too."""
    resolved = str(video_path.resolve())
    if not manifest.empty:
        updated = manifest[manifest["video_path"] != resolved].reset_index(drop=True)
    else:
        updated = manifest
    write_manifest(updated, manifest_path)
    if delete_file:
        Path(video_path).unlink(missing_ok=True)
    return updated


def progress(video_dir: Path, manifest: pd.DataFrame) -> tuple[int, int]:
    all_videos = discover_videos(video_dir)
    labeled_paths = set(manifest["video_path"]) if not manifest.empty else set()
    labeled_count = sum(1 for v in all_videos if str(v.resolve()) in labeled_paths)
    return labeled_count, len(all_videos)


def _relative_posix(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def known_video_dirs(data_dir: Path) -> list[str]:
    """Every directory under `data_dir` that directly holds at least one video
    file, as paths relative to `data_dir`'s parent (the repo root) — powers
    the video-folder picker so users choose among folders they've already
    used (data/raw, data/tracks/<name>, ...) instead of typing one from
    memory."""
    if not data_dir.exists():
        return []
    repo_root = data_dir.parent
    dirs = {video.parent for video in discover_videos(data_dir)}
    return sorted({_relative_posix(d, repo_root) for d in dirs})


def known_manifest_paths(data_dir: Path) -> list[str]:
    """Every manifest CSV under `data_dir`, as paths relative to its parent
    (the repo root) — powers the manifest-file picker."""
    if not data_dir.exists():
        return []
    repo_root = data_dir.parent
    return sorted({_relative_posix(p, repo_root) for p in data_dir.rglob("*.csv")})


def known_videos(data_dir: Path) -> list[dict]:
    """Every video file already under `data_dir` (uploaded via the Dataset
    page or produced by track extraction) — lets Inference offer picking one
    directly instead of requiring a fresh upload every time. `path` is
    relative to the repo root (what forms/views elsewhere use for video_dir/
    manifest_path); `name` is relative to `data_dir`, for a shorter display."""
    if not data_dir.exists():
        return []
    repo_root = data_dir.parent
    videos = [
        {"path": _relative_posix(video, repo_root), "name": _relative_posix(video, data_dir)}
        for video in discover_videos(data_dir)
    ]
    return sorted(videos, key=lambda v: v["name"])


def dataset_rows(video_dir: Path, manifest: pd.DataFrame) -> list[dict]:
    """Every video file found under `video_dir`, cross-referenced with its
    manifest label/split if it has one — the listing behind the dataset CRUD
    page, so uploaded-but-unlabeled videos are visible alongside labeled ones."""
    has_split = "split" in manifest.columns
    by_path = {} if manifest.empty else manifest.set_index("video_path").to_dict("index")

    rows = []
    for video in discover_videos(video_dir):
        resolved = str(video.resolve())
        entry = by_path.get(resolved)
        rows.append(
            {
                "path": resolved,
                "name": video.name,
                "label": entry["label"] if entry else "",
                "split": (entry.get("split") or "") if (entry and has_split) else "",
                "labeled": entry is not None,
            }
        )
    return rows
