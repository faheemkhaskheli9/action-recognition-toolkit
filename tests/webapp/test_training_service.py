import subprocess
import sys
from pathlib import Path

import pytest

from core.models import TrainingRun
from core.services import training as training_service

# `true` is a POSIX coreutil, not a Windows executable -- it only resolves
# here by accident when something upstream put Git for Windows' usr/bin on
# PATH. Spawn a real short-lived process the same way on every platform
# instead: the current interpreter running a no-op script.
_NOOP_COMMAND = [sys.executable, "-c", "pass"]


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

@pytest.mark.django_db
def test_start_run_creates_log_dir_and_training_run_row(settings, tmp_path, monkeypatch):
    settings.RUNS_DIR = tmp_path / "runs"
    settings.REPO_ROOT = tmp_path

    captured = {}

    def fake_launch_detached(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        # launch_detached normally creates the log file itself; the fake
        # stands in for that too so callers can rely on it existing.
        kwargs["log_file"].touch()
        return 1234

    monkeypatch.setattr(training_service._background, "launch_detached", fake_launch_detached)

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
    assert Path(run.log_file).exists()

    # the plain ar-train argv is handed to launch_detached, which owns all
    # the shell-wrapping/platform-specific plumbing (see test__background.py)
    argv = captured["argv"]
    assert "action_recognition.training.train" in argv
    assert "--config" in argv and str(Path("configs/default.yaml")) in argv
    assert "--manifest" in argv and "data/manifest.csv" in argv
    assert "--model" in argv and "r3d18" in argv
    assert "--epochs" in argv and "5" in argv
    assert "--batch-size" in argv and "2" in argv
    assert captured["kwargs"]["exit_marker_prefix"] == training_service.EXIT_MARKER_PREFIX
    assert captured["kwargs"]["cwd"] == settings.REPO_ROOT


@pytest.mark.django_db
def test_start_run_defaults_model_name_and_skips_absent_optional_flags(settings, tmp_path, monkeypatch):
    settings.RUNS_DIR = tmp_path / "runs"
    settings.REPO_ROOT = tmp_path
    captured = {}

    def fake_launch_detached(argv, **kwargs):
        captured["argv"] = argv
        kwargs["log_file"].touch()
        return 4242

    monkeypatch.setattr(training_service._background, "launch_detached", fake_launch_detached)

    run = training_service.start_run(name="exp2", config_path=Path("configs/default.yaml"))

    assert run.model_name == "(from config)"
    assert run.manifest_path == ""
    argv = captured["argv"]
    assert "--manifest" not in argv
    assert "--model" not in argv
    assert "--epochs" not in argv
    assert "--batch-size" not in argv


# --------------------------------------------------------------------- #
# is_pid_alive
# --------------------------------------------------------------------- #

def test_is_pid_alive_none_is_false():
    assert training_service.is_pid_alive(None) is False


def test_is_pid_alive_true_for_current_process():
    assert training_service.is_pid_alive(__import__("os").getpid()) is True


def test_is_pid_alive_false_once_process_has_exited():
    proc = subprocess.Popen(_NOOP_COMMAND)
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
    proc = subprocess.Popen(_NOOP_COMMAND)
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
# cancel_run
# --------------------------------------------------------------------- #

@pytest.mark.django_db
def test_cancel_run_terminates_process_and_marks_cancelled(monkeypatch):
    terminated = {}
    monkeypatch.setattr(
        training_service._background, "terminate", lambda pid: terminated.setdefault("pid", pid)
    )
    run = _make_run(pid=4242)

    result = training_service.cancel_run(run)

    assert terminated["pid"] == 4242
    assert result.status == TrainingRun.Status.CANCELLED
    assert result.finished_at is not None


@pytest.mark.django_db
def test_cancel_run_is_a_noop_for_a_non_running_run(monkeypatch):
    terminated = {}
    monkeypatch.setattr(
        training_service._background, "terminate", lambda pid: terminated.setdefault("pid", pid)
    )
    run = _make_run(status=TrainingRun.Status.SUCCEEDED, return_code=0)

    result = training_service.cancel_run(run)

    assert "pid" not in terminated
    assert result.status == TrainingRun.Status.SUCCEEDED


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
