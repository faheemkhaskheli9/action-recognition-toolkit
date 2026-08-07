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


def read_clip_frames(video_path: str, num_frames: int) -> list:
    """Uniformly sample `num_frames` RGB frames (as HWC uint8 arrays) across the clip."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # some codecs/containers report an unreliable frame count
            all_frames = []
            ok, frame = cap.read()
            while ok:
                all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ok, frame = cap.read()
            if not all_frames:
                raise IOError(f"No frames could be read from: {video_path}")
            return [all_frames[i] for i in _sample_indices(len(all_frames), num_frames)]

        indices = _sample_indices(total, num_frames)
        wanted = set(indices)
        frames_by_index = {}
        idx = 0
        while idx <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames_by_index[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        self.samples = list(zip(manifest["video_path"], manifest["label"]))
        self.label_map = label_map
        self.num_frames = num_frames
        self.transform = build_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = read_clip_frames(video_path, self.num_frames)
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
