from pathlib import Path

import pytest

from core.models import Dataset, TrackExtractionRun
from core.services import extraction as extraction_service


def _make_run(**overrides):
    defaults = dict(
        name="scene1",
        config_path="",
        video_dir="data/raw_scenes",
        output_dir="o",
        log_file="l.log",
        status=TrackExtractionRun.Status.FAILED,
        pid=None,
    )
    defaults.update(overrides)
    return TrackExtractionRun.objects.create(**defaults)


# --------------------------------------------------------------------- #
# start_run
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_start_run_over_the_whole_dataset_uses_its_video_dir(settings, tmp_path, monkeypatch):
    settings.REPO_ROOT = tmp_path
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )

    captured = {}

    def fake_launch_detached(argv, **kwargs):
        captured["argv"] = argv
        kwargs["log_file"].touch()
        return 1234

    monkeypatch.setattr(extraction_service._background, "launch_detached", fake_launch_detached)

    run = extraction_service.start_run(name="scene1", dataset=dataset)

    assert run.dataset == dataset
    assert run.video_dir == "data/raw"
    assert "data/raw" in captured["argv"]


@pytest.mark.django_db
def test_start_run_with_a_single_video_uses_that_path_instead(settings, tmp_path, monkeypatch):
    settings.REPO_ROOT = tmp_path
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )

    captured = {}

    def fake_launch_detached(argv, **kwargs):
        captured["argv"] = argv
        kwargs["log_file"].touch()
        return 1234

    monkeypatch.setattr(extraction_service._background, "launch_detached", fake_launch_detached)

    run = extraction_service.start_run(name="scene1", dataset=dataset, video="data/raw/a.mp4")

    assert run.dataset == dataset
    assert run.video_dir == "data/raw/a.mp4"
    assert "data/raw/a.mp4" in captured["argv"]
    assert "data/raw" not in captured["argv"]  # the single video, not the whole dataset folder


# --------------------------------------------------------------------- #
# can_resume
# --------------------------------------------------------------------- #

@pytest.mark.django_db
@pytest.mark.parametrize(
    "status",
    [TrackExtractionRun.Status.FAILED, TrackExtractionRun.Status.SUCCEEDED, TrackExtractionRun.Status.CANCELLED],
)
def test_can_resume_true_when_not_running(status):
    run = _make_run(status=status)
    assert extraction_service.can_resume(run) is True


@pytest.mark.django_db
def test_can_resume_false_while_running():
    run = _make_run(status=TrackExtractionRun.Status.RUNNING)
    assert extraction_service.can_resume(run) is False


# --------------------------------------------------------------------- #
# cancel_run
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_cancel_run_terminates_process_and_marks_cancelled(monkeypatch):
    terminated = {}
    monkeypatch.setattr(
        extraction_service._background, "terminate", lambda pid: terminated.setdefault("pid", pid)
    )
    run = _make_run(status=TrackExtractionRun.Status.RUNNING, pid=4242)

    result = extraction_service.cancel_run(run)

    assert terminated["pid"] == 4242
    assert result.status == TrackExtractionRun.Status.CANCELLED
    assert result.finished_at is not None


@pytest.mark.django_db
def test_cancel_run_is_a_noop_for_a_non_running_run(monkeypatch):
    terminated = {}
    monkeypatch.setattr(
        extraction_service._background, "terminate", lambda pid: terminated.setdefault("pid", pid)
    )
    run = _make_run(status=TrackExtractionRun.Status.SUCCEEDED)

    result = extraction_service.cancel_run(run)

    assert "pid" not in terminated
    assert result.status == TrackExtractionRun.Status.SUCCEEDED


# --------------------------------------------------------------------- #
# resume_run
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_resume_run_relaunches_with_resume_flag_and_updates_the_same_row(settings, tmp_path, monkeypatch):
    settings.REPO_ROOT = tmp_path
    output_dir = tmp_path / "data" / "tracks" / "scene1"
    output_dir.mkdir(parents=True)
    log_file = output_dir / "extract.log"
    log_file.write_text("earlier attempt's log\n")

    run = _make_run(
        output_dir=str(output_dir),
        log_file=str(log_file),
        video_dir="data/raw_scenes",
        config_path="configs/tracking/default.yaml",
        status=TrackExtractionRun.Status.FAILED,
        return_code=1,
        pid=999,
    )

    captured = {}

    def fake_launch_detached(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        kwargs["log_file"].touch()
        return 4321

    monkeypatch.setattr(extraction_service._background, "launch_detached", fake_launch_detached)

    resumed = extraction_service.resume_run(run)

    argv = captured["argv"]
    assert "--resume" in argv
    assert str(output_dir) in argv
    assert "data/raw_scenes" in argv
    assert "--config" in argv and str(Path("configs/tracking/default.yaml")) in argv
    assert captured["kwargs"]["append"] is True
    assert captured["kwargs"]["log_file"] == log_file

    assert resumed.pk == run.pk  # same row, not a new run
    assert resumed.pid == 4321
    assert resumed.status == TrackExtractionRun.Status.RUNNING
    assert resumed.return_code is None
    assert resumed.finished_at is None

    resumed.refresh_from_db()
    assert resumed.pid == 4321


# --------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_progress_none_when_log_missing(tmp_path):
    run = _make_run(log_file=str(tmp_path / "missing.log"))
    assert extraction_service.progress(run) is None


@pytest.mark.django_db
def test_progress_none_before_any_video_is_logged(tmp_path):
    log_file = tmp_path / "extract.log"
    log_file.write_text("Using device: cpu\n")
    run = _make_run(log_file=str(log_file))
    assert extraction_service.progress(run) is None


@pytest.mark.django_db
def test_progress_reads_the_latest_tracking_line(tmp_path):
    log_file = tmp_path / "extract.log"
    log_file.write_text(
        "Tracking data/raw_scenes/a.mp4 (1/3)\n"
        "  2 clip(s) from a.mp4 -- 2 total so far\n"
        "Tracking data/raw_scenes/b.mp4 (2/3)\n"
    )
    run = _make_run(log_file=str(log_file))
    assert extraction_service.progress(run) == {"video": "b.mp4", "current": 2, "total": 3}


@pytest.mark.django_db
def test_progress_counts_skipped_videos_on_resume(tmp_path):
    log_file = tmp_path / "extract.log"
    log_file.write_text("Skipping data/raw_scenes/a.mp4 (1/3) -- already extracted\n")
    run = _make_run(log_file=str(log_file))
    assert extraction_service.progress(run) == {"video": "a.mp4", "current": 1, "total": 3}


@pytest.mark.django_db
def test_progress_matches_the_real_logger_format_with_its_timestamp_prefix(tmp_path):
    # Regression guard: get_logger() (action_recognition.utils.logging)
    # prefixes every real line with "HH:MM:SS LEVEL logger.name: ", so a
    # regex anchored to the start of the line would never match actual
    # subprocess output -- only prefix-free fixtures like the ones above.
    log_file = tmp_path / "extract.log"
    log_file.write_text(
        "14:23:01 INFO action_recognition.scripts.extract_tracks: Tracking data/raw_scenes/a.mp4 (1/3)\n"
    )
    run = _make_run(log_file=str(log_file))
    assert extraction_service.progress(run) == {"video": "a.mp4", "current": 1, "total": 3}


# --------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------- #

def _write_index(output_dir: Path, rows: list[dict]) -> None:
    import csv

    from action_recognition.scripts.extract_tracks import INDEX_FIELDS

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tracks_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.django_db
def test_results_empty_before_the_index_exists(tmp_path):
    run = _make_run(output_dir=str(tmp_path / "o"))
    assert extraction_service.results(run) == []


@pytest.mark.django_db
def test_results_groups_clips_by_video_then_track(tmp_path):
    output_dir = tmp_path / "o"
    _write_index(
        output_dir,
        [
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 1,
                "start_frame": 16, "end_frame": 31, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track0_win1.mp4"),
            },
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track0_win0.mp4"),
            },
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 1, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track1_win0.mp4"),
            },
            {
                "source_video": "data/raw_scenes/b.mp4", "track_id": 0, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16,
                "clip_path": str(output_dir / "b" / "track0_win0.mp4"),
            },
        ],
    )
    run = _make_run(output_dir=str(output_dir))

    results = extraction_service.results(run)

    assert [v["name"] for v in results] == ["a.mp4", "b.mp4"]
    video_a = results[0]
    assert video_a["track_count"] == 2
    assert video_a["clip_count"] == 3
    assert [t["track_id"] for t in video_a["tracks"]] == [0, 1]
    # windows come back in order even though the index rows above didn't
    assert [c["window_index"] for c in video_a["tracks"][0]["clips"]] == [0, 1]
    assert video_a["tracks"][0]["clips"][0]["name"] == "track0_win0.mp4"


# --------------------------------------------------------------------- #
# provenance_for_dataset / group_clips
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_provenance_for_dataset_maps_imported_clips_back_to_their_track(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    (tmp_path / "data" / "raw").mkdir(parents=True)
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )
    output_dir = tmp_path / "data" / "tracks" / "scene1"
    _write_index(
        output_dir,
        [
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track0_win0.mp4"),
            },
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 1,
                "start_frame": 16, "end_frame": 31, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track0_win1.mp4"),
            },
        ],
    )
    run = _make_run(name="scene1", dataset=dataset, output_dir=str(output_dir), status=TrackExtractionRun.Status.SUCCEEDED)
    # Simulate import_clips's own naming/copy so provenance can find them.
    imported = tmp_path / "data" / "raw" / "scene1__a__track0_win0.mp4"
    imported.write_bytes(b"x")
    (tmp_path / "data" / "raw" / "scene1__a__track0_win1.mp4").write_bytes(b"x")

    provenance = extraction_service.provenance_for_dataset(dataset)

    assert provenance[str(imported.resolve())] == {
        "run_id": run.pk,
        "run_name": "scene1",
        "source_video": "a",
        "track_id": 0,
        "window_index": 0,
    }
    assert len(provenance) == 2


@pytest.mark.django_db
def test_provenance_for_dataset_skips_clips_that_were_never_imported(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    (tmp_path / "data" / "raw").mkdir(parents=True)
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )
    output_dir = tmp_path / "data" / "tracks" / "scene1"
    _write_index(
        output_dir,
        [
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track0_win0.mp4"),
            },
        ],
    )
    _make_run(name="scene1", dataset=dataset, output_dir=str(output_dir), status=TrackExtractionRun.Status.SUCCEEDED)
    # Never actually imported into dataset.video_dir.

    assert extraction_service.provenance_for_dataset(dataset) == {}


@pytest.mark.django_db
def test_provenance_for_dataset_ignores_runs_whose_dataset_was_deleted(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    (tmp_path / "data" / "raw").mkdir(parents=True)
    dataset = Dataset.objects.create(
        name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/manifest.csv"
    )
    output_dir = tmp_path / "data" / "tracks" / "scene1"
    _write_index(
        output_dir,
        [
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16,
                "clip_path": str(output_dir / "a" / "track0_win0.mp4"),
            },
        ],
    )
    run = _make_run(name="scene1", dataset=None, output_dir=str(output_dir), status=TrackExtractionRun.Status.SUCCEEDED)
    assert run.dataset is None  # never had this dataset, or it was deleted (SET_NULL) -- same effect
    (tmp_path / "data" / "raw" / "scene1__a__track0_win0.mp4").write_bytes(b"x")

    assert extraction_service.provenance_for_dataset(dataset) == {}


def test_group_clips_nests_by_run_and_source_video_then_track_and_window():
    provenance = {
        "/d/scene1__a__track0_win1.mp4": {"run_id": 1, "run_name": "scene1", "source_video": "a", "track_id": 0, "window_index": 1},
        "/d/scene1__a__track0_win0.mp4": {"run_id": 1, "run_name": "scene1", "source_video": "a", "track_id": 0, "window_index": 0},
        "/d/scene1__a__track1_win0.mp4": {"run_id": 1, "run_name": "scene1", "source_video": "a", "track_id": 1, "window_index": 0},
        "/d/scene1__b__track0_win0.mp4": {"run_id": 1, "run_name": "scene1", "source_video": "b", "track_id": 0, "window_index": 0},
    }

    groups = extraction_service.group_clips(provenance)

    assert [g["source_video"] for g in groups] == ["a", "b"]
    group_a = groups[0]
    assert group_a["track_count"] == 2
    assert group_a["clip_count"] == 3
    assert [t["track_id"] for t in group_a["tracks"]] == [0, 1]
    # windows come back in order even though the input dict didn't
    assert [c["window_index"] for c in group_a["tracks"][0]["clips"]] == [0, 1]


def test_group_clips_of_an_empty_provenance_is_empty():
    assert extraction_service.group_clips({}) == []
