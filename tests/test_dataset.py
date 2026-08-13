import pandas as pd
import torch

from action_recognition.data.dataset import (
    VideoClipDataset,
    _span_to_frames,
    datasets_from_manifest,
    read_clip_frames,
)
from conftest import write_synthetic_video


def test_read_clip_frames_samples_requested_count(synthetic_video):
    frames = read_clip_frames(str(synthetic_video), num_frames=8)

    assert len(frames) == 8
    assert frames[0].shape == (64, 64, 3)


def test_read_clip_frames_handles_more_frames_requested_than_available(synthetic_video):
    frames = read_clip_frames(str(synthetic_video), num_frames=100)

    assert len(frames) == 100


def test_span_to_frames_whole_video_when_unset():
    assert _span_to_frames(None, None, fps=10.0, total=100) == (0, 100)


def test_span_to_frames_converts_seconds_to_frame_indices():
    assert _span_to_frames(1.0, 3.0, fps=10.0, total=100) == (10, 30)


def test_span_to_frames_clamps_end_to_total():
    assert _span_to_frames(9.0, 20.0, fps=10.0, total=100) == (90, 100)


def test_span_to_frames_falls_back_to_whole_video_without_usable_fps():
    assert _span_to_frames(1.0, 3.0, fps=0, total=100) == (0, 100)


def test_read_clip_frames_samples_only_within_the_given_span(tmp_path):
    # 40 frames @ 10fps = 4s; each frame's pixel value is roughly i*10 % 255
    # (see conftest.write_synthetic_video) -- lossy mp4v encoding perturbs it
    # slightly, so allow some tolerance -- letting us check every sampled
    # frame came from an index inside [20, 30), the 2.0s-3.0s span.
    video = write_synthetic_video(tmp_path / "long.mp4", num_frames=40)

    frames = read_clip_frames(str(video), num_frames=4, start_time=2.0, end_time=3.0)

    assert len(frames) == 4
    expected = [(i * 10) % 255 for i in range(20, 30)]
    for frame in frames:
        value = int(frame[0, 0, 0])
        assert any(min(abs(value - e), 255 - abs(value - e)) <= 12 for e in expected)


def test_video_clip_dataset_returns_expected_shapes(synthetic_video):
    manifest = pd.DataFrame({"video_path": [str(synthetic_video)], "label": ["jump"]})
    dataset = VideoClipDataset(manifest, {"jump": 0}, num_frames=6, image_size=32)

    clip, label = dataset[0]

    assert clip.shape == (6, 3, 32, 32)
    assert label == 0
    assert isinstance(clip, torch.Tensor)


def test_video_clip_dataset_reads_span_rows(tmp_path):
    video = write_synthetic_video(tmp_path / "long.mp4", num_frames=40)
    manifest = pd.DataFrame(
        {
            "video_path": [str(video), str(video)],
            "label": ["idle", "jump"],
            "start_time": [None, 2.0],
            "end_time": [None, 3.0],
        }
    )
    dataset = VideoClipDataset(manifest, {"idle": 0, "jump": 1}, num_frames=4, image_size=32)

    whole_clip, whole_label = dataset[0]
    span_clip, span_label = dataset[1]

    assert whole_clip.shape == span_clip.shape == (4, 3, 32, 32)
    assert (whole_label, span_label) == (0, 1)


def test_datasets_from_manifest_splits_by_split_column(class_folder_dataset):
    from action_recognition.data.manifest import build_label_map, discover_class_folders, stratified_split

    manifest = stratified_split(discover_class_folders(class_folder_dataset), val_frac=0.2, test_frac=0.2, seed=0)
    label_map = build_label_map(manifest["label"])

    datasets = datasets_from_manifest(manifest, label_map, num_frames=4, image_size=32)

    assert "train" in datasets
    assert sum(len(ds) for ds in datasets.values()) == len(manifest)
