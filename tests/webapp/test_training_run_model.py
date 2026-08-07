import pytest

from core.models import TrainingRun


@pytest.mark.django_db
def test_str_combines_name_and_model_name():
    run = TrainingRun.objects.create(
        name="exp1", model_name="cnn_lstm", config_path="c", manifest_path="m",
        output_dir="o", log_file="l",
    )
    assert str(run) == "exp1 (cnn_lstm)"


@pytest.mark.django_db
def test_status_defaults_to_running():
    run = TrainingRun.objects.create(
        name="exp1", model_name="cnn_lstm", config_path="c", manifest_path="m",
        output_dir="o", log_file="l",
    )
    assert run.status == TrainingRun.Status.RUNNING
    assert run.pid is None
    assert run.return_code is None
    assert run.finished_at is None


@pytest.mark.django_db
def test_queryset_orders_newest_first():
    first = TrainingRun.objects.create(
        name="first", model_name="cnn_lstm", config_path="c", manifest_path="m",
        output_dir="o", log_file="l",
    )
    second = TrainingRun.objects.create(
        name="second", model_name="cnn_lstm", config_path="c", manifest_path="m",
        output_dir="o", log_file="l",
    )

    assert list(TrainingRun.objects.all()) == [second, first]
