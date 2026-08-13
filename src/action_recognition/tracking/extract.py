"""Detect + track people in a raw multi-person video, then crop+window each
track into short single-person clip files — the same one-file-per-clip
format the rest of the pipeline (labeling app, VideoClipDataset, ar-train,
ar-predict) already expects, so nothing downstream needs to change to train
on them.

Used by both `ar-extract-tracks` (build a labeling-ready folder of clips)
and `ar-predict-scene` (classify every tracked person in a new video with an
already-trained checkpoint).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from ..utils.logging import get_logger
from .types import Box, Track

logger = get_logger(__name__)

# How often (wall-clock seconds) track_video/extract_tracks_to_clips emit a
# progress line for a video/clip loop that can otherwise run silently for a
# long time (large source videos, many tracks) -- see extract_tracks.py's
# module docstring / DEVELOPMENT.md for why the webapp can only show real
# progress if the underlying pipeline actually logs it.
PROGRESS_LOG_INTERVAL_SECONDS = 15.0


@dataclass
class TrackWindow:
    """One sliding-window slice of a track, ready to be written out as a
    clip (or classified directly, for scene inference)."""

    track_id: int
    window_index: int
    frame_indices: list[int]
    boxes: list[Box]


def track_video(video_path: str, detector, tracker, frame_stride: int = 2) -> list[Track]:
    """Run `detector` + `tracker` over every `frame_stride`-th frame of
    `video_path`, in order, and return every track produced (active tracks
    flushed once the video ends).

    Logs a percent-complete line every ~15s -- this is the phase that can
    dominate wall-clock time for a large source video, and without it
    nothing gets written to the log (or the video's clip folder) until
    detection finishes, which reads as a hang even when it isn't one.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    try:
        frame_idx = 0
        last_log = time.monotonic()
        ok, frame = cap.read()
        while ok:
            if frame_idx % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect(rgb)
                tracker.update(frame_idx, detections)
            frame_idx += 1
            now = time.monotonic()
            if now - last_log >= PROGRESS_LOG_INTERVAL_SECONDS:
                last_log = now
                if total_frames:
                    pct = 100 * frame_idx / total_frames
                    logger.info(f"  detect+track: frame {frame_idx}/{total_frames} ({pct:.0f}%)")
                else:
                    logger.info(f"  detect+track: frame {frame_idx}")
            ok, frame = cap.read()
    finally:
        cap.release()
    tracks = tracker.finished_tracks()
    logger.info(f"  detect+track done: {len(tracks)} track(s) found")
    return tracks


def windows_for_track(
    track: Track,
    window_frames: int = 16,
    stride_frames: int = 8,
    min_track_frames: int = 16,
) -> list[TrackWindow]:
    """Slice a track's (frame_indices, boxes) into overlapping fixed-length
    windows, uniform-sampling spirit matching `_sample_indices` in
    data/dataset.py: the window operates on the track's own detected-frame
    sequence, not raw video frame count, since downstream training/inference
    always re-samples a clip to its own `num_frames` regardless of how many
    frames the source file has.

    A track shorter than `min_track_frames` produces no windows. The final
    window is dropped/kept whole (never zero-padded) — it's only included
    if it's still at least `min_track_frames` long.
    """
    n = len(track)
    if n < min_track_frames:
        return []

    windows = []
    start = 0
    window_index = 0
    while start < n:
        end = min(start + window_frames, n)
        if end - start >= min_track_frames:
            windows.append(
                TrackWindow(
                    track_id=track.track_id,
                    window_index=window_index,
                    frame_indices=track.frame_indices[start:end],
                    boxes=track.boxes[start:end],
                )
            )
            window_index += 1
        if end == n:
            break
        start += stride_frames
    return windows


def _pad_box(box: Box, padding: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - w * padding, x2 + w * padding
    y1, y2 = y1 - h * padding, y2 + h * padding
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(frame_w, int(x2)),
        min(frame_h, int(y2)),
    )


def read_window_frames(video_path: str, window: TrackWindow, crop_padding: float = 0.0) -> list:
    """Re-decode just this window's raw frames from the source video
    (sequential scan and grab-as-seen, same trade-off as
    data.dataset.read_clip_frames — random-access seeking isn't reliable
    across codecs/containers) and return them as HWC uint8 RGB crops, one
    per frame_indices entry, in order. `crop_padding` of 0 disables
    cropping and returns full frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    wanted = set(window.frame_indices)
    box_by_frame = dict(zip(window.frame_indices, window.boxes))
    frames_by_index = {}
    try:
        frame_idx = 0
        ok, frame = cap.read()
        while ok and wanted:
            if frame_idx in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if crop_padding > 0:
                    h, w = rgb.shape[:2]
                    x1, y1, x2, y2 = _pad_box(box_by_frame[frame_idx], crop_padding, w, h)
                    crop = rgb[y1:y2, x1:x2]
                    rgb = crop if crop.size > 0 else rgb
                frames_by_index[frame_idx] = rgb
                wanted.discard(frame_idx)
            frame_idx += 1
            ok, frame = cap.read()
    finally:
        cap.release()
    return [frames_by_index[i] for i in window.frame_indices if i in frames_by_index]


def write_window_clip(
    video_path: str,
    window: TrackWindow,
    output_path: Path,
    crop_padding: float = 0.2,
    output_size: int = 224,
    fps: float = 10.0,
) -> None:
    """Crop + write one track window out as a standalone clip file."""
    frames = read_window_frames(video_path, window, crop_padding=crop_padding)
    if not frames:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_size, output_size)
    )
    try:
        for rgb in frames:
            resized = cv2.resize(rgb, (output_size, output_size))
            writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def extract_tracks_to_clips(
    video_path: Path,
    output_dir: Path,
    detector,
    tracker_factory: Callable[[], object],
    frame_stride: int = 2,
    window_frames: int = 16,
    stride_frames: int = 8,
    min_track_frames: int = 16,
    crop_padding: float = 0.2,
    output_size: int = 224,
) -> list[dict]:
    """Full pipeline for one source video: detect+track, window each track,
    write cropped clips under `output_dir/<video_stem>/`, and return one
    provenance row per clip (for tracks_index.csv — not the training
    manifest, which `ar-build-manifest`/the Label page builds separately
    from the resulting clip folder).
    """
    video_path = Path(video_path)
    tracker = tracker_factory()
    tracks = track_video(str(video_path), detector, tracker, frame_stride=frame_stride)

    stem = video_path.stem
    windows = [
        (track, window)
        for track in tracks
        for window in windows_for_track(
            track,
            window_frames=window_frames,
            stride_frames=stride_frames,
            min_track_frames=min_track_frames,
        )
    ]
    logger.info(f"  {len(windows)} clip(s) to write from {len(tracks)} track(s)")

    rows = []
    last_log = time.monotonic()
    for i, (track, window) in enumerate(windows, start=1):
        clip_path = output_dir / stem / f"track{window.track_id}_win{window.window_index}.mp4"
        write_window_clip(str(video_path), window, clip_path, crop_padding=crop_padding, output_size=output_size)
        rows.append(
            {
                "source_video": str(video_path),
                "track_id": window.track_id,
                "window_index": window.window_index,
                "start_frame": window.frame_indices[0],
                "end_frame": window.frame_indices[-1],
                "num_frames": len(window.frame_indices),
                "clip_path": str(clip_path),
            }
        )
        now = time.monotonic()
        if now - last_log >= PROGRESS_LOG_INTERVAL_SECONDS or i == len(windows):
            last_log = now
            logger.info(f"  wrote {i}/{len(windows)} clip(s)")
    return rows
