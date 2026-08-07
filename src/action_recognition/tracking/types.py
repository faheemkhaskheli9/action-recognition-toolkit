"""Shared data types connecting detectors, trackers, and the extraction
pipeline — kept dependency-free so any detector/tracker backend can produce
and consume them without importing the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates


@dataclass
class Detection:
    """One detector output for a single frame."""

    box: Box
    score: float


@dataclass
class Track:
    """One tracked person across a video: parallel lists of the (possibly
    subsampled) frame index and box it was seen at. Not necessarily one
    entry per raw video frame — only frames the detector actually ran on
    and matched to this track are included.
    """

    track_id: int
    frame_indices: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame_indices)
