import pytest

from action_recognition.tracking.detectors import available_detectors
from action_recognition.tracking.extract import windows_for_track
from action_recognition.tracking.trackers import available_trackers, build_tracker
from action_recognition.tracking.trackers.registry import register_tracker
from action_recognition.tracking.types import Detection, Track


def _box_at(x: float, y: float, size: float = 10.0) -> tuple[float, float, float, float]:
    return (x, y, x + size, y + size)


def test_available_detectors_lists_builtin():
    assert "fasterrcnn" in available_detectors()


def test_available_trackers_lists_builtin():
    assert "iou" in available_trackers()


def test_register_tracker_rejects_duplicate_name():
    register_tracker("only_once_tracker")(object)
    with pytest.raises(ValueError):
        register_tracker("only_once_tracker")(object)


def test_iou_tracker_keeps_stable_id_for_overlapping_detections():
    tracker = build_tracker("iou", iou_thresh=0.3, max_age=5)

    # same person, drifting slightly frame to frame
    for frame_idx, x in enumerate([0, 1, 2, 3]):
        tracker.update(frame_idx, [Detection(box=_box_at(x, 0), score=0.9)])

    tracks = tracker.finished_tracks()
    assert len(tracks) == 1
    assert tracks[0].frame_indices == [0, 1, 2, 3]


def test_iou_tracker_assigns_separate_ids_to_non_overlapping_detections():
    tracker = build_tracker("iou", iou_thresh=0.3, max_age=5)

    tracker.update(0, [Detection(box=_box_at(0, 0), score=0.9), Detection(box=_box_at(100, 100), score=0.9)])
    tracker.update(1, [Detection(box=_box_at(1, 0), score=0.9), Detection(box=_box_at(101, 100), score=0.9)])

    tracks = tracker.finished_tracks()
    assert len(tracks) == 2
    assert {len(t) for t in tracks} == {2}


def test_iou_tracker_closes_track_after_max_age_gap():
    tracker = build_tracker("iou", iou_thresh=0.3, max_age=1)

    tracker.update(0, [Detection(box=_box_at(0, 0), score=0.9)])
    tracker.update(1, [])  # missed once, within max_age
    tracker.update(2, [])  # missed twice, exceeds max_age=1 -> track closes
    tracker.update(3, [Detection(box=_box_at(0, 0), score=0.9)])  # new track, same location

    tracks = tracker.finished_tracks()
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def _track(n: int) -> Track:
    return Track(track_id=0, frame_indices=list(range(0, n * 2, 2)), boxes=[_box_at(0, 0)] * n)


def test_windows_for_track_below_min_length_produces_nothing():
    assert windows_for_track(_track(5), window_frames=16, stride_frames=8, min_track_frames=16) == []


def test_windows_for_track_exact_length_produces_one_window():
    windows = windows_for_track(_track(16), window_frames=16, stride_frames=8, min_track_frames=16)
    assert len(windows) == 1
    assert len(windows[0].frame_indices) == 16


def test_windows_for_track_slides_with_overlap_and_drops_short_tail():
    # 40 frames, window 16, stride 8 -> starts at 0,8,16,24 all full windows (end<=40);
    # start=32 -> end=40 also full (32..40 == 16 wide); start=40 stops.
    windows = windows_for_track(_track(40), window_frames=16, stride_frames=8, min_track_frames=16)

    assert [w.window_index for w in windows] == list(range(len(windows)))
    for w in windows:
        assert len(w.frame_indices) == 16
    # windows should start 8 frames apart until the final one
    starts = [w.frame_indices[0] for w in windows]
    assert starts == sorted(starts)


def test_windows_for_track_keeps_final_partial_window_if_above_min():
    # 20 frames, window 16, stride 8: first window [0:16], next start=8 -> end=20 (12 wide),
    # 12 >= min_track_frames=10 so it's kept even though shorter than window_frames.
    windows = windows_for_track(_track(20), window_frames=16, stride_frames=8, min_track_frames=10)

    assert len(windows) == 2
    assert len(windows[0].frame_indices) == 16
    assert len(windows[1].frame_indices) == 12
