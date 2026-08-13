from __future__ import annotations

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_transform


def _sample_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= num_frames:
        return [min(i, total_frames - 1) for i in range(num_frames)]
    step = total_frames / num_frames
    return [int(step * i + step / 2) for i in range(num_frames)]


def _span_to_frames(
    start_time: float | None, end_time: float | None, fps: float | None, total: int | None
) -> tuple[int, int]:
    """Convert an optional (start_time, end_time) span in seconds to a
    (start_frame, end_frame) pair, clamped to [0, total). Falls back to the
    full range when both times are unset, or when `fps` isn't a usable value
    (some codecs/containers report 0/NaN) since seconds can't be converted to
    frame indices without it — same graceful-degradation spirit as the
    unreliable-frame-count branch in `read_clip_frames`."""
    fallback_end = total if total else 1
    if not fps or fps <= 0 or (start_time is None and end_time is None):
        return 0, fallback_end

    start_frame = int(start_time * fps) if start_time is not None else 0
    end_frame = int(end_time * fps) if end_time is not None else fallback_end
    if total:
        end_frame = min(end_frame, total)
    start_frame = max(0, min(start_frame, max(end_frame - 1, 0)))
    if end_frame <= start_frame:
        end_frame = start_frame + 1
    return start_frame, end_frame


def read_clip_frames(
    video_path: str,
    num_frames: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list:
    """Uniformly sample `num_frames` RGB frames (as HWC uint8 arrays) across
    the clip — or, when `start_time`/`end_time` (seconds) are given, across
    just that span of a longer source video. `start_time`/`end_time` default
    to `None` (whole clip), matching every caller that doesn't know about
    spans."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0:
            # some codecs/containers report an unreliable frame count
            all_frames = []
            ok, frame = cap.read()
            while ok:
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ok, frame = cap.read()
            if not all_frames:
                raise IOError(f"No frames could be read from: {video_path}")
            start_frame, end_frame = _span_to_frames(start_time, end_time, fps, len(all_frames))
            span = all_frames[start_frame:end_frame] or all_frames
            return [span[i] for i in _sample_indices(len(span), num_frames)]

        start_frame, end_frame = _span_to_frames(start_time, end_time, fps, total)
        indices = [start_frame + i for i in _sample_indices(end_frame - start_frame, num_frames)]
        wanted = set(indices)
        frames_by_index = {}
        idx = 0
        while idx <= max(wanted):
            if idx < start_frame:
                # frames before the span are skipped without decoding —
                # grab() advances the reader without the cvtColor/decode cost
                # read() pays, a real saving for a span deep in a long video
                ok = cap.grab()
            else:
                ok, frame = cap.read()
                if ok and idx in wanted:
                    frames_by_index[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ok:
                break
            idx += 1

        if not frames_by_index:
            raise IOError(f"No frames could be read from: {video_path}")

        out = []
        last = frames_by_index[min(frames_by_index)]
        for i in indices:
            last = frames_by_index.get(i, last)
            out.append(last)
        return out
    finally:
        cap.release()


class VideoClipDataset(Dataset):
    """Returns (clip, label_idx) with clip shaped (T, C, H, W).

    Model-specific input layouts (e.g. 3D CNNs wanting (C, T, H, W)) are handled
    inside each model's forward() so every architecture shares this one dataset.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        label_map: dict[str, int],
        num_frames: int = 16,
        image_size: int = 112,
        train: bool = False,
    ):
        has_start = "start_time" in manifest.columns
        has_end = "end_time" in manifest.columns
        self.samples = [
            (
                row["video_path"],
                row["label"],
                row["start_time"] if has_start and pd.notna(row.get("start_time")) else None,
                row["end_time"] if has_end and pd.notna(row.get("end_time")) else None,
            )
            for _, row in manifest.iterrows()
        ]
        self.label_map = label_map
        self.num_frames = num_frames
        self.transform = build_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label, start_time, end_time = self.samples[idx]
        frames = read_clip_frames(video_path, self.num_frames, start_time=start_time, end_time=end_time)
        clip = torch.stack([self.transform(frame) for frame in frames], dim=0)
        return clip, self.label_map[label]


def datasets_from_manifest(
    manifest: pd.DataFrame,
    label_map: dict[str, int],
    num_frames: int = 16,
    image_size: int = 112,
) -> dict[str, VideoClipDataset]:
    if "split" not in manifest.columns:
        raise ValueError("Manifest has no 'split' column — run split_dataset first.")
    out = {}
    for split in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split]
        if len(subset) == 0:
            continue
        out[split] = VideoClipDataset(
            subset, label_map, num_frames=num_frames, image_size=image_size, train=(split == "train")
        )
    return out
