"""views.home.home -- the pipeline-overview landing page at `/`."""
import os

from django.test import Client
from django.urls import reverse

import pytest

from core.models import Dataset, TrackExtractionRun, TrainingRun
from core.services.training import EXIT_MARKER_PREFIX as TRAINING_EXIT_MARKER_PREFIX

pytestmark = pytest.mark.django_db


@pytest.fixture
def repo(settings, tmp_path):
    settings.REPO_ROOT = tmp_path
    settings.DATA_DIR = tmp_path / "data"
    settings.CONFIGS_DIR = tmp_path / "configs"
    settings.RUNS_DIR = tmp_path / "runs"
    settings.MEDIA_ROOT = tmp_path / "media"
    Dataset.objects.all().delete()  # clear migration 0003's seeded "Default" dataset
    return tmp_path


def _training_run(repo, **overrides):
    output_dir = repo / "runs" / overrides.get("name", "exp")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"
    log_file.write_text(overrides.pop("log_text", "epoch 1\n"))
    defaults = dict(
        name="exp", model_name="cnn_lstm", config_path="c.yaml", manifest_path="m.csv",
        output_dir=str(output_dir), log_file=str(log_file), status=TrainingRun.Status.SUCCEEDED,
    )
    defaults.update(overrides)
    return TrainingRun.objects.create(**defaults)


def _extraction_run(repo, **overrides):
    output_dir = repo / "data" / "tracks" / overrides.get("name", "scene")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "extract.log"
    log_file.write_text(overrides.pop("log_text", ""))
    defaults = dict(
        name="scene", config_path="", video_dir="data/raw_scenes",
        output_dir=str(output_dir), log_file=str(log_file), status=TrackExtractionRun.Status.SUCCEEDED,
    )
    defaults.update(overrides)
    return TrackExtractionRun.objects.create(**defaults)


def test_home_is_served_at_the_site_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.resolver_match.url_name == "home"


def test_home_empty_state(client, repo):
    resp = client.get(reverse("core:home"))

    assert resp.status_code == 200
    assert list(resp.context["datasets"]) == []
    assert resp.context["recent_extraction_runs"] == []
    assert resp.context["recent_training_runs"] == []
    assert b"create one" in resp.content  # dataset empty-state CTA


def test_home_lists_datasets(client, repo):
    Dataset.objects.create(name="Restaurant", slug="restaurant", video_dir="data/raw", manifest_path="data/m.csv")

    resp = client.get(reverse("core:home"))

    names = [ds.name for ds in resp.context["datasets"]]
    assert names == ["Restaurant"]


def test_home_shows_up_to_five_most_recent_training_runs(client, repo):
    for i in range(7):
        _training_run(repo, name=f"exp{i}")

    resp = client.get(reverse("core:home"))

    runs = resp.context["recent_training_runs"]
    assert len(runs) == 5
    assert runs[0].name == "exp6"  # newest first (model Meta: ordering = ["-created_at"])


def test_home_shows_up_to_five_most_recent_extraction_runs(client, repo):
    for i in range(7):
        _extraction_run(repo, name=f"scene{i}")

    resp = client.get(reverse("core:home"))

    runs = resp.context["recent_extraction_runs"]
    assert len(runs) == 5
    assert runs[0].name == "scene6"


def test_home_refreshes_stale_running_status(client, repo):
    """A run stuck at status=running in the DB but whose process already
    exited (marker written, process gone) should read as its real status on
    Home, same as the list/detail pages already do -- refresh_status() is
    reused, not reimplemented."""
    run = _training_run(
        repo, status=TrainingRun.Status.RUNNING, log_text=f"{TRAINING_EXIT_MARKER_PREFIX}0\n"
    )

    resp = client.get(reverse("core:home"))

    refreshed = next(r for r in resp.context["recent_training_runs"] if r.pk == run.pk)
    assert refreshed.status == TrainingRun.Status.SUCCEEDED
    run.refresh_from_db()
    assert run.status == TrainingRun.Status.SUCCEEDED


def test_home_counts_currently_running_jobs(client, repo):
    _training_run(repo, status=TrainingRun.Status.RUNNING, pid=os.getpid(), log_text="")
    _extraction_run(repo, status=TrackExtractionRun.Status.RUNNING, pid=os.getpid(), log_text="")

    resp = client.get(reverse("core:home"))

    assert resp.context["running_trainings"] == 1
    assert resp.context["running_extractions"] == 1
    assert b"running" in resp.content


def test_home_requires_login():
    resp = Client().get(reverse("core:home"))
    assert resp.status_code == 302
    assert reverse("core:login") in resp.url
