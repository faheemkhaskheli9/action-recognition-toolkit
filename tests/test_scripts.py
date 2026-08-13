import csv

import pandas as pd
import pytest

from action_recognition.scripts import extract_tracks
from action_recognition.scripts.build_manifest import main as build_manifest_main
from action_recognition.scripts.split_dataset import main as split_dataset_main


def test_build_manifest_main_writes_a_manifest_csv(class_folder_dataset, tmp_path, monkeypatch, capsys):
    output = tmp_path / "manifest.csv"
    monkeypatch.setattr("sys.argv", ["ar-build-manifest", str(class_folder_dataset), str(output)])

    build_manifest_main()

    df = pd.read_csv(output)
    assert len(df) == 5
    assert set(df["label"]) == {"jump", "wave"}
    assert "Wrote 5 rows across 2 classes" in capsys.readouterr().out


def test_split_dataset_main_adds_split_column_and_writes_label_map(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "video_path": [f"/v{i}.mp4" for i in range(10)],
            "label": ["jump"] * 5 + ["wave"] * 5,
        }
    ).to_csv(manifest_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["ar-split-dataset", str(manifest_path), "--val-frac", "0.2", "--test-frac", "0.2", "--seed", "0"],
    )

    split_dataset_main()

    df = pd.read_csv(manifest_path)
    assert set(df["split"]) <= {"train", "val", "test"}
    assert len(df) == 10

    label_map_path = manifest_path.with_name("label_map.json")
    assert label_map_path.exists()
    assert "Wrote" in capsys.readouterr().out


def test_split_dataset_main_respects_explicit_output_path(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"video_path": ["/a.mp4", "/b.mp4"], "label": ["jump", "wave"]}).to_csv(manifest_path, index=False)
    output_path = tmp_path / "split_manifest.csv"

    monkeypatch.setattr(
        "sys.argv",
        ["ar-split-dataset", str(manifest_path), "--output", str(output_path), "--val-frac", "0.0", "--test-frac", "0.0"],
    )

    split_dataset_main()

    assert output_path.exists()
    assert (output_path.with_name("label_map.json")).exists()
    # the original manifest (without --output) is left untouched
    assert "split" not in pd.read_csv(manifest_path).columns


def test_extract_tracks_main_device_flag_overrides_config_default(tmp_path, monkeypatch):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    (video_dir / "clip.mp4").write_bytes(b"not a real video, just needs the right extension")
    output_dir = tmp_path / "out"

    captured = {}

    def fake_build_detector(name, **params):
        captured["detector_params"] = params
        return object()  # no .device attribute -- also exercises the hasattr guard in main()

    monkeypatch.setattr(extract_tracks, "build_detector", fake_build_detector)
    monkeypatch.setattr(extract_tracks, "build_tracker", lambda name, **params: object())
    monkeypatch.setattr(extract_tracks, "extract_tracks_to_clips", lambda *a, **k: [])
    monkeypatch.setattr(
        "sys.argv",
        ["ar-extract-tracks", str(video_dir), str(output_dir), "--device", "cuda"],
    )

    extract_tracks.main()

    assert captured["detector_params"]["device"] == "cuda"


def test_extract_tracks_main_accepts_a_single_video_file(tmp_path, monkeypatch):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"fake")
    (video_dir / "b.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "out"

    _stub_out_detector_and_tracker(monkeypatch)

    calls = []

    def fake_extract(video_path, out_dir, *a, **k):
        calls.append(video_path.name)
        return [_fake_row(video_path, str(out_dir / "a" / "track0_win0.mp4"))]

    monkeypatch.setattr(extract_tracks, "extract_tracks_to_clips", fake_extract)
    monkeypatch.setattr("sys.argv", ["ar-extract-tracks", str(video_dir / "a.mp4"), str(output_dir)])

    extract_tracks.main()

    # only the single named file was processed, not the rest of its folder
    assert calls == ["a.mp4"]
    df = pd.read_csv(output_dir / "tracks_index.csv")
    assert df["source_video"].tolist() == [str(video_dir / "a.mp4")]


def test_extract_tracks_main_rejects_a_single_file_with_an_unrecognized_extension(tmp_path, monkeypatch):
    not_a_video = tmp_path / "notes.txt"
    not_a_video.write_text("hello")
    output_dir = tmp_path / "out"

    _stub_out_detector_and_tracker(monkeypatch)
    monkeypatch.setattr("sys.argv", ["ar-extract-tracks", str(not_a_video), str(output_dir)])

    with pytest.raises(SystemExit, match="not a recognized video file"):
        extract_tracks.main()


def _stub_out_detector_and_tracker(monkeypatch):
    monkeypatch.setattr(extract_tracks, "build_detector", lambda name, **params: object())
    monkeypatch.setattr(extract_tracks, "build_tracker", lambda name, **params: object())


def _fake_row(video, clip_path):
    return {
        "source_video": str(video),
        "track_id": 0,
        "window_index": 0,
        "start_frame": 0,
        "end_frame": 15,
        "num_frames": 16,
        "clip_path": clip_path,
    }


def test_extract_tracks_main_writes_index_incrementally_so_a_crash_keeps_prior_progress(tmp_path, monkeypatch):
    # A run interrupted partway through (killed, crashed) must not lose the
    # index rows for videos it already finished -- that's what --resume
    # relies on to know what's already done.
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"fake")
    (video_dir / "b.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "out"

    _stub_out_detector_and_tracker(monkeypatch)

    def fake_extract(video_path, out_dir, *a, **k):
        if video_path.name == "b.mp4":
            raise RuntimeError("simulated crash")
        return [_fake_row(video_path, str(out_dir / "a" / "track0_win0.mp4"))]

    monkeypatch.setattr(extract_tracks, "extract_tracks_to_clips", fake_extract)
    monkeypatch.setattr("sys.argv", ["ar-extract-tracks", str(video_dir), str(output_dir)])

    with pytest.raises(RuntimeError, match="simulated crash"):
        extract_tracks.main()

    df = pd.read_csv(output_dir / "tracks_index.csv")
    assert df["source_video"].tolist() == [str(video_dir / "a.mp4")]


def test_extract_tracks_main_resume_skips_completed_videos_and_cleans_up_partial_ones(tmp_path, monkeypatch):
    video_dir = tmp_path / "raw"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"fake")
    (video_dir / "b.mp4").write_bytes(b"fake")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # a.mp4 already fully extracted by an earlier attempt
    with open(output_dir / "tracks_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=extract_tracks.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(_fake_row(video_dir / "a.mp4", str(output_dir / "a" / "track0_win0.mp4")))

    # b.mp4 was interrupted mid-way: a partial clip folder exists but no index row
    stale_dir = output_dir / "b"
    stale_dir.mkdir()
    (stale_dir / "track0_win0.mp4").write_bytes(b"stale partial clip")

    _stub_out_detector_and_tracker(monkeypatch)

    calls = []

    def fake_extract(video_path, out_dir, *a, **k):
        calls.append(video_path.name)
        return [_fake_row(video_path, str(out_dir / "b" / "track0_win0.mp4"))]

    monkeypatch.setattr(extract_tracks, "extract_tracks_to_clips", fake_extract)
    monkeypatch.setattr("sys.argv", ["ar-extract-tracks", str(video_dir), str(output_dir), "--resume"])

    extract_tracks.main()

    assert calls == ["b.mp4"]  # a.mp4 skipped, already done
    assert not stale_dir.exists()  # partial clips from the interrupted attempt were discarded

    df = pd.read_csv(output_dir / "tracks_index.csv")
    assert sorted(df["source_video"].tolist()) == sorted([str(video_dir / "a.mp4"), str(video_dir / "b.mp4")])
