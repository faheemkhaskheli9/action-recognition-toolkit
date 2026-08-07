import subprocess
from pathlib import Path

import pytest

from core.models import TrainingRun
from core.services import training as training_service


# --------------------------------------------------------------------- #
# available_configs / run_dir
# --------------------------------------------------------------------- #

def test_available_configs_lists_only_yaml_files_sorted(settings, tmp_path):
    settings.CONFIGS_DIR = tmp_path
    (tmp_path / "b.yaml").write_text("{}")
    (tmp_path / "a.yaml").write_text("{}")
    (tmp_path / "notes.txt").write_text("not a config")

    configs = training_service.available_configs()

    assert configs == [tmp_path / "a.yaml", tmp_path / "b.yaml"]


def test_run_dir_joins_name_under_runs_dir(settings, tmp_path):
    settings.RUNS_DIR = tmp_path

    assert training_service.run_dir("exp1") == tmp_path / "exp1"


# --------------------------------------------------------------------- #
# start_run
# --------------------------------------------------------------------- #

class _FakeProcess:
    def __init__(self, pid=4242):
        self.pid = pid


@pytest.mark.django_db
def test_start_run_creates_log_dir_and_training_run_row(settings, tmp_path, monkeypatch):
    settings.RUNS_DIR = tmp_path / "runs"
    settings.REPO_ROOT = tmp_path

    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProcess(pid=1234)

    monkeypatch.setattr(training_service.subprocess, "Popen", fake_popen)

    run = training_service.start_run(
        name="exp1",
        config_path=Path("configs/default.yaml"),
        manifest_path="data/manifest.csv",
        model_name="r3d18",
        epochs=5,
        batch_size=2,
    )

    assert run.pk is not None
    assert run.name == "exp1"
    assert run.model_name == "r3d18"
    assert run.status == TrainingRun.Status.RUNNING
    assert run.pid == 1234
    assert (settings.RUNS_DIR / "exp1").is_dir()
    assert Path(run.log_file).exists()  # log file is created even though Popen is faked

    # the underlying `ar-train` invocation is wrapped in `bash -c "<command>"`
    assert captured["argv"][:2] == ["bash", "-c"]
    command = captured["argv"][2]
    assert "action_recognition.training.train" in command
    assert "--config" in command and "default.yaml" in command
    assert "--manifest" in command and "data/manifest.csv" in command
    assert "--model" in command and "r3d18" in command
    assert "--epochs" in command and "5" in command
    assert "--batch-size" in command and "2" in command
    assert training_service.EXIT_MARKER_PREFIX in command  # exit-code marker appended
    assert captured["kwargs"]["start_new_session"] is True


@pytest.mark.django_db
def test_start_run_defaults_model_name_and_skips_absent_optional_flags(settings, tmp_path, monkeypatch):
    settings.RUNS_DIR = tmp_path / "runs"
    settings.REPO_ROOT = tmp_path
    captured = {}
    monkeypatch.setattr(
        training_service.subprocess, "Popen", lambda argv, **kw: captured.update(argv=argv) or _FakeProcess()
    )

    run = training_service.start_run(name="exp2", config_path=Path("configs/default.yaml"))

    assert run.model_name == "(from config)"
    assert run.manifest_path == ""
    command = captured["argv"][2]
    assert "--manifest" not in command
    assert "--model" not in command
    assert "--epochs" not in command
    assert "--batch-size" not in command


# --------------------------------------------------------------------- #
# is_pid_alive
# --------------------------------------------------------------------- #

def test_is_pid_alive_none_is_false():
    assert training_service.is_pid_alive(None) is False


def test_is_pid_alive_true_for_current_process():
    assert training_service.is_pid_alive(__import__("os").getpid()) is True


def test_is_pid_alive_false_once_process_has_exited():
    proc = subprocess.Popen(["true"])
    proc.wait()

    assert training_service.is_pid_alive(proc.pid) is False


# --------------------------------------------------------------------- #
# refresh_status
# --------------------------------------------------------------------- #

def _make_run(**overrides):
    defaults = dict(
        name="exp",
        model_name="cnn_lstm",
        config_path="c.yaml",
        manifest_path="m.csv",
        output_dir="o",
        log_file="l.log",
        status=TrainingRun.Status.RUNNING,
        pid=None,
    )
    defaults.update(overrides)
    return TrainingRun.objects.create(**defaults)


@pytest.mark.django_db
def test_refresh_status_reads_zero_exit_marker_as_succeeded(tmp_path):
    log_file = tmp_path / "train.log"
    log_file.write_text("epoch 1\nepoch 2\nTRAINING_EXIT_CODE=0\n")
    run = _make_run(log_file=str(log_file), pid=__import__("os").getpid())

    result = training_service.refresh_status(run)

    assert result.status == TrainingRun.Status.SUCCEEDED
    assert result.return_code == 0
    assert result.finished_at is not None


@pytest.mark.django_db
def test_refresh_status_reads_nonzero_exit_marker_as_failed(tmp_path):
    log_file = tmp_path / "train.log"
    log_file.write_text("TRAINING_EXIT_CODE=1\n")
    run = _make_run(log_file=str(log_file))

    result = training_service.refresh_status(run)

    assert result.status == TrainingRun.Status.FAILED
    assert result.return_code == 1


@pytest.mark.django_db
def test_refresh_status_marks_failed_when_process_died_without_marker(tmp_path):
    proc = subprocess.Popen(["true"])
    proc.wait()
    log_file = tmp_path / "train.log"
    log_file.write_text("some partial output, no marker\n")
    run = _make_run(log_file=str(log_file), pid=proc.pid)

    result = training_service.refresh_status(run)

    assert result.status == TrainingRun.Status.FAILED
    assert result.return_code is None


@pytest.mark.django_db
def test_refresh_status_leaves_running_run_alone_while_process_still_alive(tmp_path):
    log_file = tmp_path / "train.log"
    log_file.write_text("still going\n")
    run = _make_run(log_file=str(log_file), pid=__import__("os").getpid())

    result = training_service.refresh_status(run)

    assert result.status == TrainingRun.Status.RUNNING
    assert result.finished_at is None


@pytest.mark.django_db
def test_refresh_status_does_not_recheck_a_finished_run():
    run = _make_run(status=TrainingRun.Status.SUCCEEDED, log_file="/does/not/exist.log", return_code=0)

    result = training_service.refresh_status(run)

    assert result.status == TrainingRun.Status.SUCCEEDED  # unchanged, no crash on missing log file


# --------------------------------------------------------------------- #
# tail_log / checkpoints
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_tail_log_returns_empty_string_when_log_file_is_missing():
    run = _make_run(log_file="/does/not/exist.log")
    assert training_service.tail_log(run) == ""


@pytest.mark.django_db
def test_tail_log_truncates_to_max_lines(tmp_path):
    log_file = tmp_path / "train.log"
    log_file.write_text("\n".join(f"line{i}" for i in range(10)))
    run = _make_run(log_file=str(log_file))

    tail = training_service.tail_log(run, max_lines=3)

    assert tail.splitlines() == ["line7", "line8", "line9"]


@pytest.mark.django_db
def test_checkpoints_only_lists_files_that_exist(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "last.pt").write_bytes(b"x")
    run = _make_run(output_dir=str(output_dir))

    result = training_service.checkpoints(run)

    assert result == {"last.pt": output_dir / "last.pt"}
