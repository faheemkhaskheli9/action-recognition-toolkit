"""Default tracker: greedy IoU matching between consecutive detected frames
— SORT without the Kalman motion model. Dependency-free (plain Python), good
enough to hold stable per-person IDs across a short clip; swap in a smarter
backend later via the same registry without touching the extraction
pipeline.
"""
from __future__ import annotations

from ..types import Box, Detection, Track
from .registry import register_tracker


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _ActiveTrack:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.frame_indices: list[int] = []
        self.boxes: list[Box] = []
        self.missed = 0

    @property
    def last_box(self) -> Box:
        return self.boxes[-1]

    def to_track(self) -> Track:
        return Track(track_id=self.track_id, frame_indices=list(self.frame_indices), boxes=list(self.boxes))


@register_tracker("iou")
class IoUTracker:
    """Call `update(frame_idx, detections)` once per (possibly subsampled)
    frame in increasing order, then `finished_tracks()` to collect every
    track once the video is done — including ones still active at the end.
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._active: list[_ActiveTrack] = []
        self._done: list[Track] = []
        self._next_id = 0

    def update(self, frame_idx: int, detections: list[Detection]) -> None:
        # Greedy matching: repeatedly take the single best remaining
        # (track, detection) pair above threshold. A true Hungarian
        # assignment would be optimal but isn't worth a new dependency here.
        pairs = []
        for ti, track in enumerate(self._active):
            for di, detection in enumerate(detections):
                iou = _iou(track.last_box, detection.box)
                if iou >= self.iou_thresh:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _iou_score, ti, di in pairs:
            if ti in matched_tracks or di in matched_detections:
                continue
            track = self._active[ti]
            track.frame_indices.append(frame_idx)
            track.boxes.append(detections[di].box)
            track.missed = 0
            matched_tracks.add(ti)
            matched_detections.add(di)

        still_active = []
        for ti, track in enumerate(self._active):
            if ti not in matched_tracks:
                track.missed += 1
            if track.missed > self.max_age:
                self._done.append(track.to_track())
            else:
                still_active.append(track)
        self._active = still_active

        for di, detection in enumerate(detections):
            if di not in matched_detections:
                new_track = _ActiveTrack(self._next_id)
                self._next_id += 1
                new_track.frame_indices.append(frame_idx)
                new_track.boxes.append(detection.box)
                self._active.append(new_track)

    def finished_tracks(self) -> list[Track]:
        for track in self._active:
            self._done.append(track.to_track())
        self._active = []
        return list(self._done)
