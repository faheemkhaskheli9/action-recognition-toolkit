"""Launches `ar-extract-tracks` as a detached background process and tracks
it — same pattern as services.training (see that module's docstring for why
liveness/exit status is tracked this way rather than via Popen.wait(), and
`_background.py` for the platform-specific mechanics).
"""
from __future__ import annotations

import sys
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ..models import TrackExtractionRun
from . import _background

EXIT_MARKER_PREFIX = "EXTRACTION_EXIT_CODE="


def available_configs() -> list[Path]:
    return sorted((settings.CONFIGS_DIR / "tracking").glob("*.yaml"))


def run_dir(name: str) -> Path:
    return settings.DATA_DIR / "tracks" / name


def start_run(*, name: str, video_dir: str, config_path: Path | None = None) -> TrackExtractionRun:
    output_dir = run_dir(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "extract.log"

    argv = [
        sys.executable,
        "-m",
        "action_recognition.scripts.extract_tracks",
        video_dir,
        str(output_dir),
    ]
    if config_path:
        argv += ["--config", str(config_path)]

    pid = _background.launch_detached(
        argv,
        log_file=log_file,
        exit_marker_prefix=EXIT_MARKER_PREFIX,
        cwd=settings.REPO_ROOT,
    )

    return TrackExtractionRun.objects.create(
        name=name,
        config_path=str(config_path) if config_path else "",
        video_dir=video_dir,
        output_dir=str(output_dir),
        log_file=str(log_file),
        status=TrackExtractionRun.Status.RUNNING,
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


def refresh_status(run: TrackExtractionRun) -> TrackExtractionRun:
    if run.status != TrackExtractionRun.Status.RUNNING:
        return run

    exit_code = _read_exit_code(Path(run.log_file))
    if exit_code is not None:
        run.return_code = exit_code
        run.status = TrackExtractionRun.Status.SUCCEEDED if exit_code == 0 else TrackExtractionRun.Status.FAILED
        run.finished_at = timezone.now()
        run.save()
    elif not is_pid_alive(run.pid):
        run.status = TrackExtractionRun.Status.FAILED
        run.finished_at = timezone.now()
        run.save()
    return run


def tail_log(run: TrackExtractionRun, max_lines: int = 200) -> str:
    log_path = Path(run.log_file)
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def clip_count(run: TrackExtractionRun) -> int | None:
    index_path = Path(run.output_dir) / "tracks_index.csv"
    if not index_path.exists():
        return None
    # header line + one row per clip
    return max(0, len(index_path.read_text().splitlines()) - 1)
