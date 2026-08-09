import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from core.models import TrainingRun
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
    return tmp_path


# --------------------------------------------------------------------- #
# dataset.dataset_list / upload_videos / update_label / delete_entry
# --------------------------------------------------------------------- #

def test_dataset_list_reports_missing_directory(client, repo):
    resp = client.get(reverse("core:dataset_list"), {"video_dir": "data/raw"})

    assert resp.status_code == 200
    assert resp.context["missing_dir"] is True


def test_dataset_list_shows_labeled_and_unlabeled_videos(client, repo):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    labeled = video_dir / "a.mp4"
    unlabeled = video_dir / "b.mp4"
    labeled.write_bytes(b"x")
    unlabeled.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame({"video_path": [str(labeled.resolve())], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.get(reverse("core:dataset_list"), {"video_dir": "data/raw"})

    assert resp.status_code == 200
    assert resp.context["total"] == 2
    assert resp.context["labeled_count"] == 1
    assert resp.context["unlabeled_count"] == 1
    names = {row["name"] for row in resp.context["rows"]}
    assert names == {"a.mp4", "b.mp4"}


def test_upload_videos_saves_file_into_video_dir(client, repo):
    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")

    resp = client.post(
        reverse("core:upload_videos"),
        {"videos": [upload], "video_dir": "data/raw", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    saved = list((repo / "data" / "raw").iterdir())
    assert [p.name for p in saved] == ["clip.mp4"]
    assert not (repo / "data" / "manifest.csv").exists()  # no label given -> not in the manifest yet


def test_upload_videos_with_a_label_also_writes_the_manifest(client, repo):
    upload = SimpleUploadedFile("clip.mp4", b"fake bytes")

    resp = client.post(
        reverse("core:upload_videos"),
        {"videos": [upload], "video_dir": "data/raw", "manifest_path": "data/manifest.csv", "label": "jump"},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert manifest.iloc[0]["label"] == "jump"


def test_upload_videos_rejects_non_video_files(client, repo):
    upload = SimpleUploadedFile("notes.txt", b"hello")

    resp = client.post(
        reverse("core:upload_videos"),
        {"videos": [upload], "video_dir": "data/raw", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    assert not (repo / "data" / "raw").exists() or list((repo / "data" / "raw").iterdir()) == []


def test_upload_videos_avoids_clobbering_an_existing_file(client, repo):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "clip.mp4").write_bytes(b"original")
    upload = SimpleUploadedFile("clip.mp4", b"new bytes")

    resp = client.post(
        reverse("core:upload_videos"),
        {"videos": [upload], "video_dir": "data/raw", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    saved = sorted(p.name for p in video_dir.iterdir())
    assert len(saved) == 2  # original untouched, new upload saved under a different name
    assert (video_dir / "clip.mp4").read_bytes() == b"original"


def test_update_label_upserts_the_manifest_row(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:update_label"),
        {"video_path": str(video.resolve()), "label": "jump", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert manifest.iloc[0]["label"] == "jump"

    resp = client.post(
        reverse("core:update_label"),
        {"video_path": str(video.resolve()), "label": "kick", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert len(manifest) == 1
    assert manifest.iloc[0]["label"] == "kick"


def test_update_label_rejects_blank_label(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:update_label"),
        {"video_path": str(video.resolve()), "label": "  ", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    assert not (repo / "data" / "manifest.csv").exists()


def test_delete_entry_removes_manifest_row_but_keeps_file_by_default(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.post(
        reverse("core:delete_entry"),
        {"video_path": str(video.resolve()), "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    assert video.exists()
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 0


def test_delete_entry_can_delete_the_file_too(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest_path = repo / "data" / "manifest.csv"
    pd.DataFrame({"video_path": [str(video.resolve())], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.post(
        reverse("core:delete_entry"),
        {"video_path": str(video.resolve()), "manifest_path": "data/manifest.csv", "delete_file": "on"},
    )

    assert resp.status_code == 302
    assert not video.exists()


# --------------------------------------------------------------------- #
# labeling.label_videos
# --------------------------------------------------------------------- #

def test_label_videos_reports_missing_directory(client, repo):
    resp = client.get(reverse("core:label_videos"), {"video_dir": "data/raw"})

    assert resp.status_code == 200
    assert resp.context["missing_dir"] is True


def test_label_videos_shows_first_unlabeled_video(client, repo):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    (video_dir / "a.mp4").write_bytes(b"x")

    resp = client.get(reverse("core:label_videos"), {"video_dir": "data/raw"})

    assert resp.status_code == 200
    assert resp.context["current_name"] == "a.mp4"
    assert resp.context["labeled_count"] == 0
    assert resp.context["total"] == 1


def test_label_videos_remembers_folder_choice_in_session(client, repo):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)

    client.get(reverse("core:label_videos"), {"video_dir": "data/raw"})
    resp = client.get(reverse("core:label_videos"))  # no query params this time

    assert resp.context["video_dir"] == "data/raw"


# --------------------------------------------------------------------- #
# labeling.save_label / skip_video
# --------------------------------------------------------------------- #

def test_save_label_requires_post(client, repo):
    resp = client.get(reverse("core:save_label"))
    assert resp.status_code == 302
    assert resp.url == reverse("core:label_videos")


def test_save_label_rejects_blank_label(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:save_label"),
        {"label": "  ", "video_path": "data/raw/a.mp4", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    assert not (repo / "data" / "manifest.csv").exists()


def test_save_label_writes_manifest_row(client, repo):
    video = repo / "data" / "raw" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    resp = client.post(
        reverse("core:save_label"),
        {"label": "jump", "video_path": "data/raw/a.mp4", "manifest_path": "data/manifest.csv"},
    )

    assert resp.status_code == 302
    manifest = pd.read_csv(repo / "data" / "manifest.csv")
    assert manifest.iloc[0]["label"] == "jump"


def test_skip_video_hides_it_from_the_next_pick(client, repo):
    video_dir = repo / "data" / "raw"
    video_dir.mkdir(parents=True)
    a = video_dir / "a.mp4"
    b = video_dir / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    client.get(reverse("core:label_videos"), {"video_dir": "data/raw"})
    skip_resp = client.post(reverse("core:skip_video"), {"video_path": str(a.resolve())})
    assert skip_resp.status_code == 302

    resp = client.get(reverse("core:label_videos"))
    assert resp.context["current_name"] == "b.mp4"


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

def test_review_manifest_prefills_formset_from_existing_rows(client, repo):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(manifest_path, index=False)

    resp = client.get(reverse("core:review_manifest"), {"manifest_path": "data/manifest.csv"})

    assert resp.status_code == 200
    assert resp.context["formset"].initial[0]["label"] == "jump"


def test_review_manifest_saves_edited_rows(client, repo):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(manifest_path, index=False)

    data = {
        "manifest_path": "data/manifest.csv",
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


def test_review_manifest_shows_inline_error_on_invalid_submission(client, repo):
    manifest_path = repo / "data" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame({"video_path": ["/a.mp4"], "label": ["jump"]}).to_csv(manifest_path, index=False)

    data = {
        "manifest_path": "data/manifest.csv",
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


# --------------------------------------------------------------------- #
# training.start_training
# --------------------------------------------------------------------- #

def test_start_training_get_renders_form(client, repo):
    resp = client.get(reverse("core:start_training"))
    assert resp.status_code == 200
    assert "form" in resp.context


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


def test_inference_post_runs_prediction_and_cleans_up_upload(client, repo, monkeypatch):
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
    assert list(uploads_dir.iterdir()) == []  # temp upload was deleted after prediction


def test_inference_post_reports_prediction_failure_and_still_cleans_up(client, repo, monkeypatch):
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
    assert list(uploads_dir.iterdir()) == []
