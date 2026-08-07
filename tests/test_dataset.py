import pandas as pd
import torch

from action_recognition.data.dataset import VideoClipDataset, datasets_from_manifest, read_clip_frames


def test_read_clip_frames_samples_requested_count(synthetic_video):
    frames = read_clip_frames(str(synthetic_video), num_frames=8)

    assert len(frames) == 8
    assert frames[0].shape == (64, 64, 3)


def test_read_clip_frames_handles_more_frames_requested_than_available(synthetic_video):
    frames = read_clip_frames(str(synthetic_video), num_frames=100)

    assert len(frames) == 100


def test_video_clip_dataset_returns_expected_shapes(synthetic_video):
    manifest = pd.DataFrame({"video_path": [str(synthetic_video)], "label": ["jump"]})
    dataset = VideoClipDataset(manifest, {"jump": 0}, num_frames=6, image_size=32)

    clip, label = dataset[0]

    assert clip.shape == (6, 3, 32, 32)
    assert label == 0
    assert isinstance(clip, torch.Tensor)


def test_datasets_from_manifest_splits_by_split_column(class_folder_dataset):
    from action_recognition.data.manifest import build_label_map, discover_class_folders, stratified_split

    manifest = stratified_split(discover_class_folders(class_folder_dataset), val_frac=0.2, test_frac=0.2, seed=0)
    label_map = build_label_map(manifest["label"])

    datasets = datasets_from_manifest(manifest, label_map, num_frames=4, image_size=32)

    assert "train" in datasets
    assert sum(len(ds) for ds in datasets.values()) == len(manifest)
