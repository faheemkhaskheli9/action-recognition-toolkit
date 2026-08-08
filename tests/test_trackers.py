"""Tracker-backend behavior, including a head-to-head comparison showing
what the motion-model (`sort`) tracker buys over the plain `iou` tracker:
surviving a short gap of missed detections without splitting into a new
track ID. A full MOT-metric harness (IDF1, HOTA) needs a real labeled clip
and is future work — these synthetic-detection tests assert the concrete,
observable behavior difference directly.
"""
from action_recognition.tracking.trackers import available_trackers, build_tracker
from action_recognition.tracking.types import Detection


def _box_at(x: float, y: float, size: float = 30.0) -> tuple[float, float, float, float]:
    return (x, y, x + size, y + size)


def test_available_trackers_lists_sort():
    assert "sort" in available_trackers()


def test_sort_tracker_keeps_stable_id_for_smoothly_moving_detections():
    tracker = build_tracker("sort", iou_thresh=0.3, max_age=5)

    for frame_idx, x in enumerate([0, 5, 10, 15, 20]):
        tracker.update(frame_idx, [Detection(box=_box_at(x, 0), score=0.9)])

    tracks = tracker.finished_tracks()
    assert len(tracks) == 1
    assert tracks[0].frame_indices == [0, 1, 2, 3, 4]


def test_sort_tracker_assigns_separate_ids_to_non_overlapping_detections():
    tracker = build_tracker("sort", iou_thresh=0.3, max_age=5)

    tracker.update(0, [Detection(box=_box_at(0, 0), score=0.9), Detection(box=_box_at(200, 200), score=0.9)])
    tracker.update(1, [Detection(box=_box_at(2, 0), score=0.9), Detection(box=_box_at(202, 200), score=0.9)])

    tracks = tracker.finished_tracks()
    assert len(tracks) == 2
    assert {len(t) for t in tracks} == {2}


def test_sort_tracker_closes_track_after_max_age_gap():
    tracker = build_tracker("sort", iou_thresh=0.3, max_age=1)

    tracker.update(0, [Detection(box=_box_at(0, 0), score=0.9)])
    tracker.update(1, [])  # missed once, within max_age
    tracker.update(2, [])  # missed twice, exceeds max_age=1 -> track closes
    tracker.update(3, [Detection(box=_box_at(0, 0), score=0.9)])  # new track, same location

    tracks = tracker.finished_tracks()
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_sort_tracker_bridges_occlusion_gap_that_splits_the_iou_tracker():
    # A person drifting right at 5px/frame, then occluded for 3 frames
    # (detector produces nothing), then re-detected having kept moving at
    # the same speed. `iou` only ever compares against the *last seen* box
    # (frame 3, x=15) so by frame 7 (x=35) the boxes no longer overlap
    # enough to match -> a second track is born. `sort`'s Kalman filter
    # keeps extrapolating position through the gap and predicts a box near
    # the real frame-7 position, so it matches and the identity survives.
    size = 30
    dx = 5
    visible = {0: 0, 1: 5, 2: 10, 3: 15}
    reappear_frame, reappear_x = 7, 35

    def run(tracker_name):
        tracker = build_tracker(tracker_name, iou_thresh=0.3, max_age=10)
        for frame_idx, x in visible.items():
            tracker.update(frame_idx, [Detection(box=_box_at(x, 0, size), score=0.9)])
        for frame_idx in range(max(visible) + 1, reappear_frame):
            tracker.update(frame_idx, [])  # occluded
        tracker.update(reappear_frame, [Detection(box=_box_at(reappear_x, 0, size), score=0.9)])
        return tracker.finished_tracks()

    iou_tracks = run("iou")
    sort_tracks = run("sort")

    assert len(iou_tracks) == 2  # documented limitation: identity lost across the gap
    assert len(sort_tracks) == 1  # motion model bridges it: same identity throughout
    assert sort_tracks[0].frame_indices[0] == 0
    assert sort_tracks[0].frame_indices[-1] == reappear_frame
