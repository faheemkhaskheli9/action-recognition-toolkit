import pandas as pd

from core.services.manifest import (
    dataset_rows,
    delete_entry,
    known_labels,
    load_manifest,
    next_unlabeled,
    progress,
    save_label,
    set_label,
)


def test_load_manifest_returns_empty_frame_when_file_is_missing(tmp_path):
    df = load_manifest(tmp_path / "does_not_exist.csv")

    assert list(df.columns) == ["video_path", "label"]
    assert len(df) == 0


def test_load_manifest_reads_existing_file(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(path, index=False)

    df = load_manifest(path)

    assert df["video_path"].tolist() == ["/a.mp4"]


def test_next_unlabeled_skips_labeled_and_explicitly_skipped_videos(tmp_path):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    a, b, c = video_dir / "a.mp4", video_dir / "b.mp4", video_dir / "c.mp4"
    for v in (a, b, c):
        v.write_bytes(b"x")

    manifest = pd.DataFrame({"video_path": [str(a.resolve())], "label": ["jump"]})
    result = next_unlabeled(video_dir, manifest, skipped={str(b)})

    assert result == c


def test_next_unlabeled_returns_none_when_everything_is_handled(tmp_path):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    a = video_dir / "a.mp4"
    a.write_bytes(b"x")
    manifest = pd.DataFrame({"video_path": [str(a.resolve())], "label": ["jump"]})

    assert next_unlabeled(video_dir, manifest, skipped=set()) is None


def test_known_labels_is_sorted_deduplicated_and_empty_safe():
    assert known_labels(pd.DataFrame(columns=["video_path", "label"])) == []

    manifest = pd.DataFrame({"video_path": ["/a", "/b", "/c"], "label": ["wave", "jump", "wave"]})
    assert known_labels(manifest) == ["jump", "wave"]


def test_save_label_appends_row_and_persists_to_disk(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest = pd.DataFrame(columns=["video_path", "label"])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    updated = save_label(manifest_path, manifest, video, "jump")

    assert len(updated) == 1
    assert updated.iloc[0]["label"] == "jump"
    on_disk = pd.read_csv(manifest_path)
    assert on_disk.iloc[0]["video_path"] == str(video.resolve())


def test_progress_counts_labeled_out_of_total(tmp_path):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    a, b = video_dir / "a.mp4", video_dir / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    manifest = pd.DataFrame({"video_path": [str(a.resolve())], "label": ["jump"]})

    labeled, total = progress(video_dir, manifest)

    assert (labeled, total) == (1, 2)


def test_set_label_appends_when_video_is_new(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest = pd.DataFrame(columns=["video_path", "label"])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    updated = set_label(manifest_path, manifest, video, "jump")

    assert len(updated) == 1
    assert updated.iloc[0]["label"] == "jump"


def test_set_label_updates_existing_row_in_place(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    manifest = pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]})

    updated = set_label(manifest_path, manifest, video, "kick")

    assert len(updated) == 1
    assert updated.iloc[0]["label"] == "kick"
    on_disk = pd.read_csv(manifest_path)
    assert on_disk.iloc[0]["label"] == "kick"


def test_delete_entry_removes_the_row_and_persists(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    manifest = pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]})

    updated = delete_entry(manifest_path, manifest, video)

    assert len(updated) == 0
    assert video.exists()  # file is kept unless delete_file=True
    on_disk = pd.read_csv(manifest_path)
    assert len(on_disk) == 0


def test_delete_entry_can_also_delete_the_file(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    manifest = pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]})

    delete_entry(manifest_path, manifest, video, delete_file=True)

    assert not video.exists()


def test_delete_entry_on_unknown_video_is_a_no_op(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    manifest = pd.DataFrame(columns=["video_path", "label"])

    updated = delete_entry(manifest_path, manifest, video)

    assert len(updated) == 0


def test_dataset_rows_merges_discovered_videos_with_manifest_labels(tmp_path):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    labeled, unlabeled = video_dir / "a.mp4", video_dir / "b.mp4"
    labeled.write_bytes(b"x")
    unlabeled.write_bytes(b"x")
    manifest = pd.DataFrame({"video_path": [str(labeled.resolve())], "label": ["jump"]})

    rows = dataset_rows(video_dir, manifest)

    assert rows == [
        {"path": str(labeled.resolve()), "name": "a.mp4", "label": "jump", "split": "", "labeled": True},
        {"path": str(unlabeled.resolve()), "name": "b.mp4", "label": "", "split": "", "labeled": False},
    ]
