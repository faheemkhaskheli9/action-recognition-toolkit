from pathlib import Path

import pytest

from core.models import TrainingRun
from core.services import inference as inference_service


@pytest.mark.django_db
def test_checkpoint_choices_lists_existing_best_and_last_per_run(tmp_path):
    run_with_both = TrainingRun.objects.create(
        name="exp-a", model_name="cnn_lstm", config_path="c", manifest_path="m",
        output_dir=str(tmp_path / "a"), log_file="l",
    )
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "best.pt").write_bytes(b"x")
    (tmp_path / "a" / "last.pt").write_bytes(b"x")

    run_with_none = TrainingRun.objects.create(
        name="exp-b", model_name="cnn_lstm", config_path="c", manifest_path="m",
        output_dir=str(tmp_path / "b"), log_file="l",
    )
    # no checkpoint files written for exp-b at all

    choices = inference_service.checkpoint_choices()

    values = {value for value, _label in choices}
    labels = {label for _value, label in choices}
    assert str(tmp_path / "a" / "best.pt") in values
    assert str(tmp_path / "a" / "last.pt") in values
    assert not any(str(tmp_path / "b") in v for v in values)
    assert "exp-a / best.pt" in labels
    assert "exp-a / last.pt" in labels


@pytest.mark.django_db
def test_checkpoint_choices_empty_when_no_runs():
    assert inference_service.checkpoint_choices() == []


def test_predict_video_forwards_arguments_to_the_cli_predict_function(monkeypatch):
    captured = {}

    def fake_predict(checkpoint_path, video_path, top_k=3):
        captured.update(checkpoint_path=checkpoint_path, video_path=video_path, top_k=top_k)
        return [("jump", 0.9)]

    monkeypatch.setattr(inference_service, "run_predict", fake_predict)

    result = inference_service.predict_video(Path("best.pt"), Path("clip.mp4"), top_k=5)

    assert result == [("jump", 0.9)]
    assert captured == {"checkpoint_path": Path("best.pt"), "video_path": Path("clip.mp4"), "top_k": 5}
