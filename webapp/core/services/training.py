"""Launches `ar-train` as a detached background process and tracks it.

Liveness/exit status can't rely on subprocess.Popen's own wait(), since the
Django process that started a run isn't guaranteed to still be the one
handling a later request (dev server autoreload, multiple workers). Instead
the shell wrapper appends an explicit exit-code marker line to the log file,
and `refresh_status` reads that back plus a liveness check. See
`_background.py` for the platform-specific mechanics of both.
"""
from __future__ import annotations

import sys
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ..models import TrainingRun
from . import _background

EXIT_MARKER_PREFIX = "TRAINING_EXIT_CODE="


def available_configs() -> list[Path]:
    return sorted(settings.CONFIGS_DIR.glob("*.yaml"))


def run_dir(name: str) -> Path:
    return settings.RUNS_DIR / name


def start_run(
    *,
    name: str,
    config_path: Path,
    manifest_path: str | None = None,
    model_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> TrainingRun:
    output_dir = run_dir(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.log"

    argv = [
        sys.executable,
        "-m",
        "action_recognition.training.train",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if manifest_path:
        argv += ["--manifest", manifest_path]
    if model_name:
        argv += ["--model", model_name]
    if epochs:
        argv += ["--epochs", str(epochs)]
    if batch_size:
        argv += ["--batch-size", str(batch_size)]

    pid = _background.launch_detached(
        argv,
        log_file=log_file,
        exit_marker_prefix=EXIT_MARKER_PREFIX,
        cwd=settings.REPO_ROOT,
    )

    return TrainingRun.objects.create(
        name=name,
        model_name=model_name or "(from config)",
        config_path=str(config_path),
        manifest_path=manifest_path or "",
        output_dir=str(output_dir),
        log_file=str(log_file),
        status=TrainingRun.Status.RUNNING,
        pid=pid,
    )


def is_pid_alive(pid: int | None) -> bool:
    return _background.is_pid_alive(pid)


def _read_exit_code(log_file: Path) -> int | None:
    if not log_file.exists():
        return None
    tail = log_file.read_text(errors="replace").splitlines()[-5:]
    for line in reversed(tail):
        if line.startswith(EXIT_MARKER_PREFIX):
            try:
                return int(line[len(EXIT_MARKER_PREFIX) :].strip())
            except ValueError:
                return None
    return None


def refresh_status(run: TrainingRun) -> TrainingRun:
    if run.status != TrainingRun.Status.RUNNING:
        return run

    exit_code = _read_exit_code(Path(run.log_file))
    if exit_code is not None:
        run.return_code = exit_code
        run.status = TrainingRun.Status.SUCCEEDED if exit_code == 0 else TrainingRun.Status.FAILED
        run.finished_at = timezone.now()
        run.save()
    elif not is_pid_alive(run.pid):
        # Process is gone but never wrote the marker (killed, host restart, crash).
        run.status = TrainingRun.Status.FAILED
        run.finished_at = timezone.now()
        run.save()
    return run


def tail_log(run: TrainingRun, max_lines: int = 200) -> str:
    log_path = Path(run.log_file)
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def checkpoints(run: TrainingRun) -> dict[str, Path]:
    output_dir = Path(run.output_dir)
    return {name: output_dir / name for name in ("best.pt", "last.pt") if (output_dir / name).exists()}
