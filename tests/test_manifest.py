import pandas as pd

from action_recognition.data.manifest import (
    build_label_map,
    discover_class_folders,
    discover_videos,
    load_label_map,
    read_manifest,
    save_label_map,
    stratified_split,
    write_manifest,
)


def test_discover_videos_filters_by_extension(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "clip2.avi").write_bytes(b"x")

    found = discover_videos(tmp_path)

    assert {p.name for p in found} == {"clip.mp4", "clip2.avi"}


def test_discover_class_folders_builds_manifest(class_folder_dataset):
    df = discover_class_folders(class_folder_dataset)

    assert len(df) == 5
    assert set(df["label"]) == {"jump", "wave"}
    assert (df["label"] == "jump").sum() == 3


def test_discover_class_folders_errors_on_empty_root(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        discover_class_folders(empty)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_manifest_roundtrip(tmp_path):
    df = pd.DataFrame({"video_path": ["/a.mp4", "/b.mp4"], "label": ["jump", "wave"]})
    path = tmp_path / "manifest.csv"

    write_manifest(df, path)
    loaded = read_manifest(path)

    assert loaded["video_path"].tolist() == ["/a.mp4", "/b.mp4"]
    assert loaded["label"].tolist() == ["jump", "wave"]


def test_build_label_map_is_sorted_and_dense():
    label_map = build_label_map(["wave", "jump", "jump", "kick"])

    assert label_map == {"jump": 0, "kick": 1, "wave": 2}


def test_label_map_roundtrip(tmp_path):
    label_map = {"jump": 0, "wave": 1}
    path = tmp_path / "label_map.json"

    save_label_map(label_map, path)

    assert load_label_map(path) == label_map


def test_stratified_split_keeps_every_class_in_train_and_covers_all_rows():
    df = pd.DataFrame(
        {
            "video_path": [f"/v{i}.mp4" for i in range(20)],
            "label": ["jump"] * 10 + ["wave"] * 10,
        }
    )

    split_df = stratified_split(df, val_frac=0.2, test_frac=0.2, seed=0)

    assert set(split_df["split"]) <= {"train", "val", "test"}
    assert len(split_df) == len(df)
    for label in ("jump", "wave"):
        counts = split_df[split_df["label"] == label]["split"].value_counts()
        assert counts.get("train", 0) > 0


def test_stratified_split_rejects_fractions_summing_past_one():
    df = pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]})
    try:
        stratified_split(df, val_frac=0.6, test_frac=0.6)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_manifest_roundtrip_preserves_span_columns(tmp_path):
    df = pd.DataFrame(
        {
            "video_path": ["/a.mp4", "/a.mp4"],
            "label": ["idle", "jump"],
            "start_time": [None, 2.0],
            "end_time": [None, 3.0],
        }
    )
    path = tmp_path / "manifest.csv"

    write_manifest(df, path)
    loaded = read_manifest(path)

    assert loaded.loc[1, "start_time"] == 2.0
    assert loaded.loc[1, "end_time"] == 3.0
    assert pd.isna(loaded.loc[0, "start_time"])


def test_stratified_split_preserves_span_columns_untouched():
    df = pd.DataFrame(
        {
            "video_path": [f"/v{i}.mp4" for i in range(10)],
            "label": ["jump"] * 5 + ["wave"] * 5,
            "start_time": [1.0] * 10,
            "end_time": [2.0] * 10,
        }
    )

    split_df = stratified_split(df, val_frac=0.2, test_frac=0.2, seed=0)

    assert (split_df["start_time"] == 1.0).all()
    assert (split_df["end_time"] == 2.0).all()
