"""Motion-model tracker: constant-velocity Kalman filter per track +
Hungarian (optimal) assignment — the SORT algorithm. Unlike `iou_tracker`,
which only matches a detection against a track's *last seen* box, this
predicts where a track should be this frame before matching — so it can
bridge a short gap of missed detections (occlusion, a frame the detector
missed) that would otherwise split one person into two tracks. Same
`update`/`finished_tracks` contract as `iou`, so it drops in via
`tracking.tracker.name: sort` without touching extract.py's call sites.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

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


def _box_to_z(box: Box) -> np.ndarray:
    """(x1, y1, x2, y2) -> (center_x, center_y, area, aspect_ratio)."""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h], dtype=float)


def _state_to_box(x: np.ndarray) -> Box:
    cx, cy, s, r = x[0], x[1], max(x[2], 1e-6), max(x[3], 1e-6)
    w = float(np.sqrt(s * r))
    h = s / max(w, 1e-6)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _KalmanBoxTracker:
    """Constant-velocity filter over one box: state
    `[cx, cy, area, aspect_ratio, v_cx, v_cy, v_area]` (aspect ratio is
    assumed constant, no velocity term for it — the classic SORT
    formulation). `predict()` must be called once per frame (including
    frames with no matching detection) so velocity keeps extrapolating
    position through a gap; `update()` corrects it against a real box."""

    _ndim = 7
    _zdim = 4

    def __init__(self, box: Box):
        self.F = np.eye(self._ndim)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # position/area += velocity each step

        self.H = np.zeros((self._zdim, self._ndim))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(self._ndim)
        self.Q[4:, 4:] *= 0.01  # velocity assumed to change slowly
        self.R = np.eye(self._zdim)
        self.R[2:, 2:] *= 10.0  # area/aspect measurements noisier than position

        self.P = np.eye(self._ndim) * 10.0
        self.P[4:, 4:] *= 1000.0  # velocity starts unknown

        self.x = np.zeros(self._ndim)
        self.x[:4] = _box_to_z(box)

    def predict(self) -> Box:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.x[2] <= 0:  # area can't go non-positive after extrapolation
            self.x[2] = 1e-6
        return _state_to_box(self.x)

    def update(self, box: Box) -> None:
        z = _box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self._ndim) - K @ self.H) @ self.P

    @property
    def box(self) -> Box:
        return _state_to_box(self.x)


class _ActiveTrack:
    def __init__(self, track_id: int, box: Box):
        self.track_id = track_id
        self.frame_indices: list[int] = []
        self.boxes: list[Box] = []
        self.kf = _KalmanBoxTracker(box)
        self.missed = 0

    def to_track(self) -> Track:
        return Track(track_id=self.track_id, frame_indices=list(self.frame_indices), boxes=list(self.boxes))


@register_tracker("sort")
class SortTracker:
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
        predicted = [track.kf.predict() for track in self._active]

        # Hungarian assignment on 1 - IoU cost: optimal, unlike the greedy
        # nearest-pair-first matching `iou_tracker` uses.
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if predicted and detections:
            cost = np.ones((len(predicted), len(detections)))
            for ti, pbox in enumerate(predicted):
                for di, det in enumerate(detections):
                    cost[ti, di] = 1.0 - _iou(pbox, det.box)
            row_idx, col_idx = linear_sum_assignment(cost)
            for ti, di in zip(row_idx, col_idx):
                if 1.0 - cost[ti, di] < self.iou_thresh:
                    continue
                track = self._active[ti]
                track.kf.update(detections[di].box)
                track.frame_indices.append(frame_idx)
                track.boxes.append(track.kf.box)
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
                new_track = _ActiveTrack(self._next_id, detection.box)
                self._next_id += 1
                new_track.frame_indices.append(frame_idx)
                new_track.boxes.append(detection.box)
                self._active.append(new_track)

    def finished_tracks(self) -> list[Track]:
        for track in self._active:
            self._done.append(track.to_track())
        self._active = []
        return list(self._done)
