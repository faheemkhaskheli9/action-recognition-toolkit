from pathlib import Path

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core.models import Dataset, TrackExtractionRun, TrainingRun
from core.services import training as training_service

pytestmark = pytest.mark.django_db


@pytest.fixture
def repo(settings, tmp_path):
    """Points every REPO_ROOT-derived setting at an isolated tmp_path tree."""
    settings.REPO_ROOT = tmp_path
    settings.DATA_DIR = tmp_path / "data"
    settings.CONFIGS_DIR = tmp_path / "configs"
    settings.RUNS_DIR = tmp_path / "runs"
    settings.MEDIA_ROOT = tmp_path / "media"
    settings.CONFIGS_DIR.mkdir()
    (settings.CONFIGS_DIR / "default.yaml").write_text("{}")
    # Migration 0003 seeds a "Default" Dataset (name/slug="Default"/"default")
    # for real installs upgrading from before Dataset existed -- that row is
    # part of the test database's migrated baseline too, so clear it here to
    # start each test from a clean slate; tests that want a dataset use the
    # `dataset` fixture below.
    Dataset.objects.all().delete()
    return tmp_path


@pytest.fixture
def dataset(repo):
    """The paths every pre-Dataset test used implicitly (data/raw +
    data/manifest.csv) -- same as migration 0003's auto-created "Default"
    dataset, just built directly rather than via migration state."""
    return Dataset.objects.create(
        name="Default", slug="default", video_dir="data/raw", manifest_path="data/manifest.csv"
    )


def _select(client, dataset):
    """Make `dataset` the session's sticky current dataset -- equivalent to
    a page having already resolved it once via services.datasets.resolve."""
    session = client.session
    session["dataset_slug"] = dataset.slug
    session.save()


# --------------------------------------------------------------------- #
# dataset.dataset_list / upload_videos / update_label / delete_entry
# --------------------------------------------------------------------- #

def test_dataset_list_with_no_datasets_shows_empty_state(client, repo):
    resp = client.get(reverse("core:dataset_list"))

    assert resp.status_code == 200
    assert resp.context["dataset"] is None


def test_dataset_list_reports_missing_directory(client, repo, dataset):
    resp = client.get(reverse("core:dataset_list"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["missing_dir"] is True


def test_dataset_list_shows_labeled_and_unlabeled_videos(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    labeled = video_dir / "a.mp4"
    unlabeled = video_dir / "b.mp4"
    labeled.write_bytes(b"x")
    unlabeled.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame({"video_path": [str(labeled.resolve())], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.get(reverse("core:dataset_list"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["total"] == 2
    assert resp.context["labeled_count"] == 1
    assert resp.context["unlabeled_count"] == 1
    names = {row["name"] for row in resp.context["rows"]}
    assert names == {"a.mp4", "b.mp4"}


def test_dataset_list_ignores_an_unknown_dataset_slug(client, repo, dataset):
    """A ?dataset= for a slug that doesn't match any Dataset row falls back
    to the sticky session selection instead of erroring -- there's no way to
    point video_dir at an arbitrary path via this param, only at an existing
    Dataset's own folder."""
    _select(client, dataset)

    resp = client.get(reverse("core:dataset_list"), {"dataset": "does-not-exist"})

    assert resp.status_code == 200
    assert resp.context["dataset"] == dataset


def test_upload_videos_saves_file_into_video_dir(client, repo, dataset):
    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")

    resp = client.post(reverse("core:upload_videos"), {"videos": [upload], "dataset": dataset.slug})

    assert resp.status_code == 302
    saved = list((repo / "data" / "raw").iterdir())
    assert [p.name for p in saved] == ["clip.mp4"]
    assert not (repo / "data" / "manifest.csv").exists()  # no label given -> not in the manifest yet


def test_upload_videos_without_a_dataset_redirects_to_manage_datasets(client, repo):
    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")

    resp = client.post(reverse("core:upload_videos"), {"videos": [upload]})

    assert resp.status_code == 302
    assert resp.url == reverse("core:manage_datasets")
    assert not (repo / "data" / "raw").exists()


def test_upload_videos_with_a_label_also_writes_the_manifest(client, repo, dataset):
    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")

    resp = client.post(
        reverse("core:upload_videos"), {"videos": [upload], "dataset": dataset.slug, "label": "jump"}
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert manifest.iloc[0]["label"] == "jump"


def test_upload_videos_rejects_non_video_files(client, repo, dataset):
    upload = SimpleUploadedFile("notes.txt", b"hello")

    resp = client.post(reverse("core:upload_videos"), {"videos": [upload], "dataset": dataset.slug})

    assert resp.status_code == 302
    assert not (repo / "data" / "raw").exists() or list((repo / "data" / "raw").iterdir()) == []


def test_upload_videos_avoids_clobbering_an_existing_file(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "clip.mp4").write_bytes(b"original")
    upload = SimpleUploadedFile("clip.mp4", b"new bytes")

    resp = client.post(reverse("core:upload_videos"), {"videos": [upload], "dataset": dataset.slug})

    assert resp.status_code == 302
    saved = sorted(p.name for p in video_dir.iterdir())
    assert len(saved) == 2  # original untouched, new upload saved under a different name
    assert (video_dir / "clip.mp4").read_bytes() == b"original"


def test_update_label_upserts_the_manifest_row(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:update_label"),
        {"video_path": str(video.resolve()), "label": "jump", "dataset": dataset.slug},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert manifest.iloc[0]["label"] == "jump"

    resp = client.post(
        reverse("core:update_label"),
        {"video_path": str(video.resolve()), "label": "kick", "dataset": dataset.slug},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert len(manifest) == 1
    assert manifest.iloc[0]["label"] == "kick"


def test_update_label_rejects_blank_label(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:update_label"),
        {"video_path": str(video.resolve()), "label": "  ", "dataset": dataset.slug},
    )

    assert resp.status_code == 302
    assert not (repo / "data" / "manifest.csv").exists()


def test_delete_entry_removes_manifest_row_but_keeps_file_by_default(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.post(
        reverse("core:delete_entry"), {"video_path": str(video.resolve()), "dataset": dataset.slug}
    )

    assert resp.status_code == 302
    assert video.exists()
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 0


def test_delete_entry_can_delete_the_file_too(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.post(
        reverse("core:delete_entry"),
        {"video_path": str(video.resolve()), "dataset": dataset.slug, "delete_file": "on"},
    )

    assert resp.status_code == 302
    assert not video.exists()


def test_delete_entry_with_a_span_removes_only_that_span(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame(
        {
            "video_path": [str(video.resolve())] * 2,
            "label": ["idle", "jump"],
            "start_time": [None, 2.0],
            "end_time": [None, 3.0],
        }
    ).to_csv(manifest_path, index=False)

    resp = client.post(
        reverse("core:delete_entry"),
        {
            "video_path": str(video.resolve()),
            "dataset": dataset.slug,
            "start_time": "2.0",
            "end_time": "3.0",
        },
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(manifest_path)
    assert manifest["label"].tolist() == ["idle"]


def test_delete_entry_rejects_a_non_numeric_span_instead_of_500ing(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:delete_entry"),
        {"video_path": str(video.resolve()), "dataset": dataset.slug, "start_time": "not-a-number", "end_time": "3.0"},
    )

    assert resp.status_code == 302
    assert resp.url == reverse("core:dataset_list") + f"?dataset={dataset.slug}"


# --------------------------------------------------------------------- #
# datasets.manage_datasets / delete_dataset
# --------------------------------------------------------------------- #

def test_manage_datasets_lists_video_counts(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")
    (video_dir / "b.mp4").write_bytes(b"x")

    resp = client.get(reverse("core:manage_datasets"))

    assert resp.status_code == 200
    row = resp.context["rows"][0]
    assert row["dataset"] == dataset
    assert row["video_count"] == 2


def test_manage_datasets_post_creates_a_dataset_and_selects_it(client, repo):
    resp = client.post(reverse("core:manage_datasets"), {"name": "Restaurant floor"})

    assert resp.status_code == 302
    assert resp.url == reverse("core:dataset_list")
    created = Dataset.objects.get(name="Restaurant floor")
    assert created.slug == "restaurant-floor"
    assert created.video_dir == "data/datasets/restaurant-floor/raw"
    assert client.session["dataset_slug"] == created.slug


def test_manage_datasets_post_rejects_a_blank_name(client, repo):
    resp = client.post(reverse("core:manage_datasets"), {"name": "  "})

    assert resp.status_code == 200
    assert Dataset.objects.count() == 0


def test_manage_datasets_post_rejects_a_colliding_name_instead_of_500ing(client, repo, dataset):
    # dataset fixture already created "Default" -- posting the same name
    # used to hit an unhandled IntegrityError straight from the DB.
    resp = client.post(reverse("core:manage_datasets"), {"name": dataset.name})

    assert resp.status_code == 200
    assert Dataset.objects.filter(name=dataset.name).count() == 1


def test_delete_dataset_removes_the_row_but_keeps_files_by_default(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    resp = client.post(reverse("core:delete_dataset", kwargs={"pk": dataset.pk}))

    assert resp.status_code == 302
    assert not Dataset.objects.filter(pk=dataset.pk).exists()
    assert (video_dir / "a.mp4").exists()


def test_delete_dataset_can_delete_its_files_too(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    resp = client.post(reverse("core:delete_dataset", kwargs={"pk": dataset.pk}), {"delete_files": "on"})

    assert resp.status_code == 302
    assert not video_dir.exists()


# --------------------------------------------------------------------- #
# extraction.start_extraction
# --------------------------------------------------------------------- #

def test_start_extraction_with_no_datasets_shows_empty_state(client, repo):
    resp = client.get(reverse("core:start_extraction"))

    assert resp.status_code == 200
    assert resp.context["dataset"] is None


def test_start_extraction_get_lists_known_videos_for_the_dataset(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    resp = client.get(reverse("core:start_extraction"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["known_videos"] == [{"path": "data/raw/a.mp4", "name": "a.mp4"}]


def test_start_extraction_post_runs_over_the_whole_dataset(client, repo, dataset, monkeypatch):
    from core.services import extraction as extraction_service

    captured = {}

    def fake_launch_detached(argv, **kwargs):
        captured["argv"] = argv
        kwargs["log_file"].touch()
        return 4242

    monkeypatch.setattr(extraction_service._background, "launch_detached", fake_launch_detached)

    resp = client.post(
        reverse("core:start_extraction"), {"name": "scene1", "dataset": dataset.slug, "config": ""}
    )

    assert resp.status_code == 302
    run = TrackExtractionRun.objects.get(name="scene1")
    assert run.dataset == dataset
    assert run.video_dir == "data/raw"
    assert "data/raw" in captured["argv"]


def test_start_extraction_post_with_a_video_runs_over_just_that_one(client, repo, dataset, monkeypatch):
    from core.services import extraction as extraction_service

    monkeypatch.setattr(
        extraction_service._background,
        "launch_detached",
        lambda argv, **kw: (kw["log_file"].touch(), 4242)[1],
    )

    resp = client.post(
        reverse("core:start_extraction"),
        {"name": "scene1", "dataset": dataset.slug, "video": "data/raw/a.mp4", "config": ""},
    )

    assert resp.status_code == 302
    run = TrackExtractionRun.objects.get(name="scene1")
    assert run.dataset == dataset
    assert run.video_dir == "data/raw/a.mp4"


def _extraction_run(repo, dataset=None, **overrides):
    output_dir = repo / "data" / "tracks" / "scene1"
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        name="scene1", config_path="", video_dir="data/raw_scenes", dataset=dataset,
        output_dir=str(output_dir), log_file=str(output_dir / "extract.log"),
        status=TrackExtractionRun.Status.SUCCEEDED,
    )
    defaults.update(overrides)
    return TrackExtractionRun.objects.create(**defaults)


def test_import_extraction_clips_copies_into_the_source_dataset(client, repo, dataset):
    run = _extraction_run(repo, dataset=dataset)
    clip = Path(run.output_dir) / "personA" / "track0_win0.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")

    resp = client.post(reverse("core:import_extraction_clips", kwargs={"pk": run.pk}))

    assert resp.status_code == 302
    imported = list((repo / "data" / "raw").iterdir())
    assert [p.name for p in imported] == ["scene1__personA__track0_win0.mp4"]


def test_import_extraction_clips_without_a_dataset_reports_an_error(client, repo):
    run = _extraction_run(repo)  # no source dataset, and none exist to fall back to
    clip = Path(run.output_dir) / "personA" / "track0_win0.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")

    resp = client.post(reverse("core:import_extraction_clips", kwargs={"pk": run.pk}))

    assert resp.status_code == 302
    assert not (repo / "data" / "raw").exists()


def test_import_extraction_clips_is_idempotent(client, repo, dataset):
    run = _extraction_run(repo, dataset=dataset)
    clip = Path(run.output_dir) / "personA" / "track0_win0.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")

    client.post(reverse("core:import_extraction_clips", kwargs={"pk": run.pk}))
    client.post(reverse("core:import_extraction_clips", kwargs={"pk": run.pk}))

    imported = list((repo / "data" / "raw").iterdir())
    assert len(imported) == 1


def test_import_extraction_clips_rejects_a_running_run(client, repo, dataset):
    import os

    run = _extraction_run(repo, dataset=dataset, status=TrackExtractionRun.Status.RUNNING, pid=os.getpid())
    clip = Path(run.output_dir) / "personA" / "track0_win0.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")

    resp = client.post(reverse("core:import_extraction_clips", kwargs={"pk": run.pk}))

    assert resp.status_code == 302
    assert not (repo / "data" / "raw").exists()


def test_resume_extraction_relaunches_a_failed_run(client, repo, monkeypatch):
    from core.services import extraction as extraction_service

    run = _extraction_run(repo, status=TrackExtractionRun.Status.FAILED, return_code=1)

    def fake_launch_detached(argv, **kwargs):
        kwargs["log_file"].touch()
        return 5555

    monkeypatch.setattr(extraction_service._background, "launch_detached", fake_launch_detached)

    resp = client.post(reverse("core:resume_extraction", kwargs={"pk": run.pk}))

    assert resp.status_code == 302
    run.refresh_from_db()
    assert run.status == TrackExtractionRun.Status.RUNNING
    assert run.pid == 5555
    assert run.return_code is None


def test_resume_extraction_rejects_a_running_run(client, repo):
    import os

    run = _extraction_run(repo, status=TrackExtractionRun.Status.RUNNING, pid=os.getpid())

    resp = client.post(reverse("core:resume_extraction", kwargs={"pk": run.pk}))

    assert resp.status_code == 302
    run.refresh_from_db()
    assert run.status == TrackExtractionRun.Status.RUNNING
    assert run.pid == os.getpid()


def test_extraction_list_renders_with_runs(client, repo):
    _extraction_run(repo)

    resp = client.get(reverse("core:extraction_list"))

    assert resp.status_code == 200
    assert [r.name for r in resp.context["runs"]] == ["scene1"]


def test_extraction_detail_renders_progress_for_a_running_run(client, repo):
    import os

    run = _extraction_run(repo, status=TrackExtractionRun.Status.RUNNING, pid=os.getpid())
    Path(run.log_file).write_text("Tracking data/raw_scenes/a.mp4 (1/2)\n")

    resp = client.get(reverse("core:extraction_detail", kwargs={"pk": run.pk}))

    assert resp.status_code == 200
    assert resp.context["progress"] == {"video": "a.mp4", "current": 1, "total": 2}
    assert b"video 1 of 2" in resp.content


def test_extraction_detail_renders_the_per_video_track_breakdown(client, repo):
    import csv

    from action_recognition.scripts.extract_tracks import INDEX_FIELDS

    run = _extraction_run(repo)
    output_dir = Path(run.output_dir)
    clip_path = output_dir / "a" / "track0_win0.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.write_bytes(b"x")
    with open(output_dir / "tracks_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "source_video": "data/raw_scenes/a.mp4", "track_id": 0, "window_index": 0,
                "start_frame": 0, "end_frame": 15, "num_frames": 16, "clip_path": str(clip_path),
            }
        )

    resp = client.get(reverse("core:extraction_detail", kwargs={"pk": run.pk}))

    assert resp.status_code == 200
    assert resp.context["results"][0]["name"] == "a.mp4"
    assert b"track0_win0.mp4" in resp.content
    assert b"Track 0" in resp.content


def test_extraction_detail_import_copy_names_the_real_target_dataset(client, repo, dataset):
    # Regression guard: the help text used to hardcode "data/raw" and link
    # to Label/Dataset with no dataset slug, both stale since the Dataset
    # model replaced the single fixed data/raw folder.
    run = _extraction_run(repo, dataset=dataset, status=TrackExtractionRun.Status.SUCCEEDED)

    resp = client.get(reverse("core:extraction_detail", kwargs={"pk": run.pk}))

    assert resp.status_code == 200
    assert resp.context["target_dataset"] == dataset
    assert dataset.video_dir.encode() in resp.content
    assert f"?dataset={dataset.slug}".encode() in resp.content
    assert b"data/raw</code> so" not in resp.content


def test_extraction_detail_import_copy_prompts_to_create_a_dataset_when_none_exist(client, repo):
    run = _extraction_run(repo, status=TrackExtractionRun.Status.SUCCEEDED)  # no dataset, none to fall back to

    resp = client.get(reverse("core:extraction_detail", kwargs={"pk": run.pk}))

    assert resp.status_code == 200
    assert resp.context["target_dataset"] is None
    assert b"Create a dataset first" in resp.content


# --------------------------------------------------------------------- #
# labeling.label_videos
# --------------------------------------------------------------------- #

def test_label_videos_with_no_datasets_shows_empty_state(client, repo):
    resp = client.get(reverse("core:label_videos"))

    assert resp.status_code == 200
    assert resp.context["dataset"] is None


def test_label_videos_reports_missing_directory(client, repo, dataset):
    resp = client.get(reverse("core:label_videos"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["missing_dir"] is True


def test_label_videos_shows_first_unlabeled_video(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    resp = client.get(reverse("core:label_videos"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["current_name"] == "a.mp4"
    assert resp.context["labeled_count"] == 0
    assert resp.context["total"] == 1


def test_label_videos_ignores_an_unknown_dataset_slug(client, repo, dataset):
    _select(client, dataset)
    (repo / "data" / "raw").mkdir(parents=True)

    resp = client.get(reverse("core:label_videos"), {"dataset": "does-not-exist"})

    assert resp.context["dataset"] == dataset


# --------------------------------------------------------------------- #
# labeling.save_label / skip_video
# --------------------------------------------------------------------- #

def test_save_label_requires_post(client, repo):
    resp = client.get(reverse("core:save_label"))
    assert resp.status_code == 302
    assert resp.url == reverse("core:label_videos")


def test_save_label_rejects_blank_label(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:save_label"),
        {"label": "  ", "video_path": "data/raw/a.mp4", "dataset": dataset.slug},
    )

    assert resp.status_code == 302
    assert not (repo / "data" / "manifest.csv").exists()


def test_save_label_writes_manifest_row(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:save_label"),
        {"label": "jump", "video_path": "data/raw/a.mp4", "dataset": dataset.slug},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert manifest.iloc[0]["label"] == "jump"


def test_skip_video_hides_it_from_the_next_pick(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    a = video_dir / "a.mp4"
    b = video_dir / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    client.get(reverse("core:label_videos"), {"dataset": dataset.slug})
    skip_resp = client.post(
        reverse("core:skip_video"), {"video_path": str(a.resolve()), "dataset": dataset.slug}
    )
    assert skip_resp.status_code == 302

    resp = client.get(reverse("core:label_videos"))
    assert resp.context["current_name"] == "b.mp4"


def test_skip_video_requires_a_dataset(client, repo, dataset):
    # Regression guard: skip_video used to have no dataset context at all,
    # so its session bookkeeping couldn't be scoped per dataset.
    resp = client.post(reverse("core:skip_video"), {"video_path": "data/raw/a.mp4"})

    assert resp.status_code == 302
    assert resp.url == reverse("core:manage_datasets")


# --------------------------------------------------------------------- #
# labeling.add_span / delete_span / finish_video
# --------------------------------------------------------------------- #

def test_add_span_requires_post(client, repo):
    resp = client.get(reverse("core:add_span"))
    assert resp.status_code == 302
    assert resp.url == reverse("core:label_videos")


def test_add_span_rejects_a_backwards_range(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:add_span"),
        {
            "label": "jump",
            "video_path": "data/raw/a.mp4",
            "dataset": dataset.slug,
            "start_time": "5",
            "end_time": "2",
        },
    )

    assert resp.status_code == 302
    assert not (repo / "data" / "manifest.csv").exists()


def test_add_span_keeps_the_current_video_sticky(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    a, b = video_dir / "a.mp4", video_dir / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    first = client.get(reverse("core:label_videos"), {"dataset": dataset.slug})
    assert first.context["current_name"] == "a.mp4"

    resp = client.post(
        reverse("core:add_span"),
        {
            "label": "jump",
            "video_path": str(a.resolve()),
            "dataset": dataset.slug,
            "start_time": "2.0",
            "end_time": "3.0",
        },
    )
    assert resp.status_code == 302

    # a manifest row now exists for a.mp4, but the page should still show it
    # as "current" -- adding a span isn't "done" with the video.
    again = client.get(reverse("core:label_videos"))
    assert again.context["current_name"] == "a.mp4"
    assert [s["label"] for s in again.context["current_spans"]] == ["jump"]


def test_finish_video_advances_to_the_next_video(client, repo, dataset):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    a, b = video_dir / "a.mp4", video_dir / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    client.get(reverse("core:label_videos"), {"dataset": dataset.slug})
    client.post(
        reverse("core:add_span"),
        {
            "label": "jump",
            "video_path": str(a.resolve()),
            "dataset": dataset.slug,
            "start_time": "2.0",
            "end_time": "3.0",
        },
    )
    finish_resp = client.post(reverse("core:finish_video"), {"dataset": dataset.slug})
    assert finish_resp.status_code == 302

    resp = client.get(reverse("core:label_videos"))
    assert resp.context["current_name"] == "b.mp4"


def test_switching_datasets_mid_label_does_not_leak_the_old_datasets_current_video(client, repo, dataset):
    # Regression guard: current_video/skipped used to be flat, unscoped
    # session keys. Switching datasets via the picker without finishing the
    # video left the old dataset's video "current" under the new dataset,
    # so saving a label would write dataset A's video path into dataset B's
    # manifest.
    other = Dataset.objects.create(name="Other", slug="other", video_dir="data/other", manifest_path="data/other.csv")

    a_dir = repo / "data" / "raw"
    a_dir.mkdir(parents=True)
    (a_dir / "a.mp4").write_bytes(b"x")
    b_dir = repo / "data" / "other"
    b_dir.mkdir(parents=True)
    (b_dir / "b.mp4").write_bytes(b"x")

    first = client.get(reverse("core:label_videos"), {"dataset": dataset.slug})
    assert first.context["current_name"] == "a.mp4"

    # Switch datasets without finishing/skipping -- same as clicking the
    # dataset picker mid-label.
    second = client.get(reverse("core:label_videos"), {"dataset": other.slug})
    assert second.context["current_name"] == "b.mp4"  # not a.mp4 leaking from the old dataset

    resp = client.post(
        reverse("core:save_label"),
        {"label": "jump", "video_path": second.context["current"], "dataset": other.slug},
    )
    assert resp.status_code == 302

    other_manifest = pd.read_csv(repo / "data" / "other.csv")
    assert other_manifest.iloc[0]["video_path"].endswith("b.mp4")
    assert not (repo / "data" / "manifest.csv").exists()  # dataset A's manifest untouched


def test_delete_span_removes_only_that_span(client, repo, dataset):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame(
        {
            "video_path": [str(video.resolve())] * 2,
            "label": ["jump", "kick"],
            "start_time": [2.0, 5.0],
            "end_time": [3.0, 6.0],
        }
    ).to_csv(manifest_path, index=False)

    resp = client.post(
        reverse("core:delete_span"),
        {
            "video_path": str(video.resolve()),
            "dataset": dataset.slug,
            "start_time": "2.0",
            "end_time": "3.0",
        },
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(manifest_path)
    assert manifest["label"].tolist() == ["kick"]


# --------------------------------------------------------------------- #
# labeling.serve_video
# --------------------------------------------------------------------- #

def test_serve_video_returns_file_within_repo(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake video bytes")

    resp = client.get(reverse("core:serve_video"), {"path": "data/raw/a.mp4"})

    assert resp.status_code == 200
    assert b"".join(resp.streaming_content) == b"fake video bytes"


def test_serve_video_404s_outside_repo_root(client, repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "secret.mp4"
    outside.write_bytes(b"x")

    resp = client.get(reverse("core:serve_video"), {"path": str(outside)})

    assert resp.status_code == 404


def test_serve_video_404s_when_missing(client, repo):
    resp = client.get(reverse("core:serve_video"), {"path": "data/raw/does-not-exist.mp4"})
    assert resp.status_code == 404


# --------------------------------------------------------------------- #
# labeling.review_manifest
# --------------------------------------------------------------------- #

def test_review_manifest_prefills_formset_from_existing_rows(client, repo, dataset):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.get(reverse("core:review_manifest"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["formset"].initial[0]["label"] == "jump"


def test_review_manifest_with_no_datasets_shows_empty_state(client, repo):
    resp = client.get(reverse("core:review_manifest"))

    assert resp.status_code == 200
    assert resp.context["dataset"] is None


def test_review_manifest_saves_edited_rows(client, repo, dataset):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(manifest_path, index=False)

    data = {
        "dataset": dataset.slug,
        "form-TOTAL_FORMS": "1",
        "form-INITIAL_FORMS": "1",
        "form-0-video_path": "/a.mp4",
        "form-0-label": "kick",
        "form-0-split": "",
    }
    resp = client.post(reverse("core:review_manifest"), data)

    assert resp.status_code == 302
    updated = pd.read_csv(manifest_path)
    assert updated.iloc[0]["label"] == "kick"


def test_review_manifest_shows_inline_error_on_invalid_submission(client, repo, dataset):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(manifest_path, index=False)

    data = {
        "dataset": dataset.slug,
        "form-TOTAL_FORMS": "1",
        "form-INITIAL_FORMS": "1",
        "form-0-video_path": "/a.mp4",
        "form-0-label": "",  # required -- triggers a formset validation error
        "form-0-split": "",
    }
    resp = client.post(reverse("core:review_manifest"), data)

    assert resp.status_code == 200  # re-renders the form, no redirect
    assert not resp.context["formset"].is_valid()
    assert b"This field is required" in resp.content
    # the manifest on disk is untouched -- the invalid post never got saved
    assert pd.read_csv(manifest_path).iloc[0]["label"] == "jump"


def test_review_manifest_shows_and_roundtrips_span_columns(client, repo, dataset):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"video_path": ["/a.mp4"], "label": ["jump"], "start_time": [2.0], "end_time": [3.0]}
    ).to_csv(manifest_path, index=False)

    resp = client.get(reverse("core:review_manifest"), {"dataset": dataset.slug})

    assert resp.status_code == 200
    assert resp.context["has_spans"] is True
    assert resp.context["formset"].initial[0]["span_display"] == "0:02–0:03"

    data = {
        "dataset": dataset.slug,
        "form-TOTAL_FORMS": "1",
        "form-INITIAL_FORMS": "1",
        "form-0-video_path": "/a.mp4",
        "form-0-label": "kick",
        "form-0-split": "",
        "form-0-start_time": "2.0",
        "form-0-end_time": "3.0",
        "form-0-span_display": "0:02–0:03",
    }
    save_resp = client.post(reverse("core:review_manifest"), data)

    assert save_resp.status_code == 302
    updated = pd.read_csv(manifest_path)
    assert updated.iloc[0]["label"] == "kick"
    assert updated.iloc[0]["start_time"] == 2.0
    assert updated.iloc[0]["end_time"] == 3.0


# --------------------------------------------------------------------- #
# training.start_training
# --------------------------------------------------------------------- #

def test_start_training_get_renders_form(client, repo):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("video_path,label\n")

    resp = client.get(reverse("core:start_training"))

    assert resp.status_code == 200
    assert "form" in resp.context
    assert resp.context["known_manifest_paths"] == ["data/manifest.csv"]


def test_start_training_post_starts_a_run_and_redirects(client, repo, monkeypatch):
    monkeypatch.setattr(training_service._background, "launch_detached", lambda argv, **kw: 999)

    resp = client.post(
        reverse("core:start_training"),
        {"name": "exp1", "config": str(repo / "configs" / "default.yaml")},
    )

    assert resp.status_code == 302
    run = TrainingRun.objects.get(name="exp1")
    assert resp.url == reverse("core:run_detail", kwargs={"pk": run.pk})


def test_start_training_post_rejects_duplicate_run_name(client, repo, monkeypatch):
    monkeypatch.setattr(training_service._background, "launch_detached", lambda argv, **kw: 999)
    (repo / "runs" / "exp1").mkdir(parents=True)

    resp = client.post(
        reverse("core:start_training"),
        {"name": "exp1", "config": str(repo / "configs" / "default.yaml")},
    )

    assert resp.status_code == 200  # re-rendered with a form error, no redirect
    assert "already exists" in str(resp.context["form"].errors)


# --------------------------------------------------------------------- #
# runs.run_list / run_detail / run_log_partial
# --------------------------------------------------------------------- #

def _running_run(repo, **overrides):
    output_dir = repo / "runs" / "exp1"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    log_file.write_text("epoch 1\n")
    defaults = dict(
        name="exp1", model_name="cnn_lstm", config_path="c.yaml", manifest_path="m.csv",
        output_dir=str(output_dir), log_file=str(log_file), status=TrainingRun.Status.RUNNING,
        pid=__import__("os").getpid(),
    )
    defaults.update(overrides)
    return TrainingRun.objects.create(**defaults)


def test_run_list_shows_all_runs(client, repo):
    _running_run(repo)
    resp = client.get(reverse("core:run_list"))

    assert resp.status_code == 200
    assert len(resp.context["runs"]) == 1


def test_run_detail_includes_log_tail_and_checkpoints(client, repo):
    run = _running_run(repo)
    (repo / "runs" / "exp1" / "best.pt").write_bytes(b"x")

    resp = client.get(reverse("core:run_detail", kwargs={"pk": run.pk}))

    assert resp.status_code == 200
    assert "epoch 1" in resp.context["log_tail"]
    assert "best.pt" in resp.context["checkpoints"]


def test_run_detail_404_for_unknown_pk(client, repo):
    resp = client.get(reverse("core:run_detail", kwargs={"pk": 999999}))
    assert resp.status_code == 404


def test_run_log_partial_returns_json_status(client, repo):
    run = _running_run(repo)

    resp = client.get(reverse("core:run_log_partial", kwargs={"pk": run.pk}))

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == TrainingRun.Status.RUNNING
    assert "epoch 1" in payload["log_tail"]


# --------------------------------------------------------------------- #
# inference.inference
# --------------------------------------------------------------------- #

def test_inference_get_reports_no_checkpoints(client, repo):
    resp = client.get(reverse("core:inference"))
    assert resp.status_code == 200
    assert resp.context["no_checkpoints"] is True


def test_inference_get_renders_form_when_checkpoints_exist(client, repo):
    run = _running_run(repo)
    (repo / "runs" / "exp1" / "best.pt").write_bytes(b"x")

    resp = client.get(reverse("core:inference"))

    assert resp.status_code == 200
    assert "form" in resp.context


def test_inference_post_runs_prediction_and_saves_upload_at_a_fixed_path(client, repo, monkeypatch):
    run = _running_run(repo)
    checkpoint = repo / "runs" / "exp1" / "best.pt"
    checkpoint.write_bytes(b"x")

    from core.services import inference as inference_service_module

    monkeypatch.setattr(
        inference_service_module, "run_predict", lambda checkpoint_path, video_path, top_k=3: [("jump", 0.87)]
    )

    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")
    resp = client.post(
        reverse("core:inference"),
        {"checkpoint": str(checkpoint), "video": upload, "top_k": 3},
    )

    assert resp.status_code == 200
    assert resp.context["result"] == [("jump", 0.87)]
    uploads_dir = repo / "media" / "uploads"
    assert [p.name for p in uploads_dir.iterdir()] == ["clip.mp4"]  # kept at a name-derived path

    # Re-uploading the same filename overwrites the same path rather than piling up.
    upload_again = SimpleUploadedFile("clip.mp4", b"different fake bytes")
    resp = client.post(
        reverse("core:inference"),
        {"checkpoint": str(checkpoint), "video": upload_again, "top_k": 3},
    )
    assert resp.status_code == 200
    assert [p.name for p in uploads_dir.iterdir()] == ["clip.mp4"]
    assert (uploads_dir / "clip.mp4").read_bytes() == b"different fake bytes"


def test_inference_get_lists_known_videos_from_the_dataset(client, repo):
    _running_run(repo)
    (repo / "runs" / "exp1" / "best.pt").write_bytes(b"x")
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    resp = client.get(reverse("core:inference"))

    assert resp.context["known_videos"] == [{"path": "data/raw/a.mp4", "name": "raw/a.mp4"}]


def test_inference_post_with_existing_video_skips_upload(client, repo, monkeypatch):
    run = _running_run(repo)
    checkpoint = repo / "runs" / "exp1" / "best.pt"
    checkpoint.write_bytes(b"x")
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    from core.services import inference as inference_service_module

    monkeypatch.setattr(
        inference_service_module, "run_predict", lambda checkpoint_path, video_path, top_k=3: [("jump", 0.87)]
    )

    resp = client.post(
        reverse("core:inference"),
        {"checkpoint": str(checkpoint), "existing_video": "data/raw/a.mp4", "top_k": 3},
    )

    assert resp.status_code == 200
    assert resp.context["result"] == [("jump", 0.87)]
    uploads_dir = repo / "media" / "uploads"
    assert not uploads_dir.exists() or list(uploads_dir.iterdir()) == []


def test_inference_post_rejects_existing_video_outside_the_repo(client, repo, tmp_path_factory):
    run = _running_run(repo)
    checkpoint = repo / "runs" / "exp1" / "best.pt"
    checkpoint.write_bytes(b"x")
    outside = tmp_path_factory.mktemp("outside") / "outside.mp4"
    outside.write_bytes(b"x")
    escaping_rel_path = __import__("os").path.relpath(outside, repo)

    resp = client.post(
        reverse("core:inference"),
        {"checkpoint": str(checkpoint), "existing_video": escaping_rel_path, "top_k": 3},
    )

    assert resp.status_code == 200
    assert resp.context["result"] is None
    assert "existing_video" in resp.context["form"].errors


def test_inference_post_reports_prediction_failure_and_leaves_upload_in_place(client, repo, monkeypatch):
    run = _running_run(repo)
    checkpoint = repo / "runs" / "exp1" / "best.pt"
    checkpoint.write_bytes(b"x")

    from core.services import inference as inference_service_module

    def boom(checkpoint_path, video_path, top_k=3):
        raise RuntimeError("model blew up")

    monkeypatch.setattr(inference_service_module, "run_predict", boom)

    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")
    resp = client.post(
        reverse("core:inference"),
        {"checkpoint": str(checkpoint), "video": upload, "top_k": 3},
    )

    assert resp.status_code == 200
    assert resp.context["result"] is None
    uploads_dir = repo / "media" / "uploads"
    assert [p.name for p in uploads_dir.iterdir()] == ["clip.mp4"]
