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


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def _span_mask(manifest: pd.DataFrame, start_time: float | None, end_time: float | None) -> pd.Series:
    """Rows whose start_time/end_time match the given span (both None means
    the "whole video" row -- no span columns set)."""

    def matches(col: str, value: float | None) -> pd.Series:
        if col not in manifest.columns:
            return pd.Series(value is None, index=manifest.index)
        series = manifest[col]
        if value is None:
            return series.isna()
        return (series.astype(float) - float(value)).abs() < 1e-6

    return matches("start_time", start_time) & matches("end_time", end_time)


def save_label(
    manifest_path: Path,
    manifest: pd.DataFrame,
    video_path: Path,
    label: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> pd.DataFrame:
    """Append a labeled row. With `start_time`/`end_time` unset (the common
    case), labels the whole file -- unchanged behavior. With both given,
    labels just that span of `video_path`, so a video can carry several rows
    (one per labeled span) instead of exactly one. The span columns are only
    added to the row dict when used, so a manifest that never labels spans
    keeps its plain video_path,label[,split] shape."""
    row = {"video_path": str(video_path.resolve()), "label": label}
    if start_time is not None:
        row["start_time"] = start_time
    if end_time is not None:
        row["end_time"] = end_time
    updated = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    write_manifest(updated, manifest_path)
    return updated


def save_labels(manifest_path: Path, manifest: pd.DataFrame, entries: list[tuple[Path, str]]) -> pd.DataFrame:
    """Append several whole-file labeled rows in a single write -- used by
    track-level labeling (label_track view), where one label applies to
    every window clip of a track. Equivalent to calling save_label() once
    per entry, but writes the manifest once instead of once per clip."""
    rows = [{"video_path": str(video_path.resolve()), "label": label} for video_path, label in entries]
    updated = pd.concat([manifest, pd.DataFrame(rows)], ignore_index=True) if rows else manifest
    write_manifest(updated, manifest_path)
    return updated


def set_label(manifest_path: Path, manifest: pd.DataFrame, video_path: Path, label: str) -> pd.DataFrame:
    """Upsert the whole-video label for `video_path` — updates that row if
    it's already in the manifest, otherwise appends one. Used by the dataset
    CRUD page's inline edit, which only ever targets the whole-video row;
    labeled spans (added on the Label page) are untouched by this."""
    resolved = str(video_path.resolve())
    if not manifest.empty:
        mask = (manifest["video_path"] == resolved) & _span_mask(manifest, None, None)
        if mask.any():
            updated = manifest.copy()
            updated.loc[mask, "label"] = label
            write_manifest(updated, manifest_path)
            return updated
    return save_label(manifest_path, manifest, video_path, label)


def delete_entry(
    manifest_path: Path,
    manifest: pd.DataFrame,
    video_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
    delete_file: bool = False,
) -> pd.DataFrame:
    """Remove one row for `video_path` from the manifest (a no-op if it has
    none), optionally deleting the underlying video file too. With
    `start_time`/`end_time` unset (the default), targets the whole-video row
    only, leaving any labeled spans for the same video in place; pass a
    span's own start/end to remove just that span instead."""
    resolved = str(video_path.resolve())
    if not manifest.empty:
        target = (manifest["video_path"] == resolved) & _span_mask(manifest, start_time, end_time)
        updated = manifest[~target].reset_index(drop=True)
    else:
        updated = manifest
    write_manifest(updated, manifest_path)
    if delete_file:
        Path(video_path).unlink(missing_ok=True)
    return updated


def spans_for(manifest: pd.DataFrame, video_path: Path) -> list[dict]:
    """Labeled spans already saved for one video, formatted for display —
    used by the Label page while a user is still adding more spans to the
    video currently open, so they can see what's already been added."""
    if manifest.empty or "start_time" not in manifest.columns:
        return []
    resolved = str(video_path.resolve())
    matches = manifest[
        (manifest["video_path"] == resolved) & manifest["start_time"].notna() & manifest["end_time"].notna()
    ]
    return [
        {
            "label": row["label"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "start_display": format_seconds(row["start_time"]),
            "end_display": format_seconds(row["end_time"]),
        }
        for _, row in matches.iterrows()
    ]


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


def known_videos(data_dir: Path, repo_root: Path | None = None) -> list[dict]:
    """Every video file already under `data_dir` (uploaded via the Dataset
    page or produced by track extraction) — lets Inference offer picking one
    directly instead of requiring a fresh upload every time, and lets
    Extraction offer picking a single video from a chosen dataset. `path` is
    relative to `repo_root` (what forms/views elsewhere use for video_dir/
    manifest_path) — defaults to `data_dir`'s parent, which is the repo root
    for the common `data_dir=settings.DATA_DIR` call; pass `repo_root`
    explicitly when `data_dir` is nested deeper, e.g. one dataset's own
    video_dir. `name` is relative to `data_dir`, for a shorter display."""
    if not data_dir.exists():
        return []
    if repo_root is None:
        repo_root = data_dir.parent
    videos = [
        {"path": _relative_posix(video, repo_root), "name": _relative_posix(video, data_dir)}
        for video in discover_videos(data_dir)
    ]
    return sorted(videos, key=lambda v: v["name"])


def dataset_rows(video_dir: Path, manifest: pd.DataFrame) -> list[dict]:
    """Every video file found under `video_dir`, cross-referenced with its
    manifest label/split if it has one — the listing behind the dataset CRUD
    page, so uploaded-but-unlabeled videos are visible alongside labeled ones.

    A video can have a whole-video row, any number of labeled-span rows
    (added on the Label page), or both — `label`/`split` reflect the
    whole-video row only (what the page's inline edit form targets); `spans`
    lists the rest, formatted for display."""
    has_spans = "start_time" in manifest.columns
    has_split = "split" in manifest.columns
    by_video: dict[str, list[dict]] = {}
    if not manifest.empty:
        for _, entry in manifest.iterrows():
            by_video.setdefault(entry["video_path"], []).append(entry.to_dict())

    rows = []
    for video in discover_videos(video_dir):
        resolved = str(video.resolve())
        whole_label = ""
        whole_split = ""
        spans = []
        for entry in by_video.get(resolved, []):
            start = entry.get("start_time") if has_spans else None
            end = entry.get("end_time") if has_spans else None
            if has_spans and pd.notna(start) and pd.notna(end):
                spans.append(
                    {
                        "label": entry["label"],
                        "start_time": start,
                        "end_time": end,
                        "start_display": format_seconds(start),
                        "end_display": format_seconds(end),
                    }
                )
            else:
                whole_label = entry["label"]
                whole_split = (entry.get("split") or "") if has_split else ""

        rows.append(
            {
                "path": resolved,
                "name": video.name,
                "label": whole_label,
                "split": whole_split,
                "spans": spans,
                "labeled": bool(whole_label) or bool(spans),
            }
        )
    return rows
