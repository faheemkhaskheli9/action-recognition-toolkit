"""Launches `ar-extract-tracks` as a detached background process and tracks
it — same pattern as services.training (see that module's docstring for why
liveness/exit status is tracked this way rather than via Popen.wait(), and
`_background.py` for the platform-specific mechanics).
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from action_recognition.data.manifest import discover_videos

from ..models import Dataset, TrackExtractionRun
from ..paths import resolve_repo_path
from . import _background

EXIT_MARKER_PREFIX = "EXTRACTION_EXIT_CODE="


def available_configs() -> list[Path]:
    return sorted((settings.CONFIGS_DIR / "tracking").glob("*.yaml"))


def run_dir(name: str) -> Path:
    return settings.DATA_DIR / "tracks" / name


def _build_argv(*, video_dir: str, output_dir: Path, config_path: Path | None, resume: bool = False) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "action_recognition.scripts.extract_tracks",
        video_dir,
        str(output_dir),
    ]
    if config_path:
        argv += ["--config", str(config_path)]
    if resume:
        argv += ["--resume"]
    return argv


def start_run(
    *, name: str, dataset: Dataset, video: str | None = None, config_path: Path | None = None
) -> TrackExtractionRun:
    """Launch a run against `dataset` -- every video under its video_dir, or
    just `video` (a single video's repo-relative path) if given."""
    source = video if video else dataset.video_dir

    output_dir = run_dir(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "extract.log"

    argv = _build_argv(video_dir=source, output_dir=output_dir, config_path=config_path)
    pid = _background.launch_detached(
        argv,
        log_file=log_file,
        exit_marker_prefix=EXIT_MARKER_PREFIX,
        cwd=settings.REPO_ROOT,
    )

    return TrackExtractionRun.objects.create(
        name=name,
        config_path=str(config_path) if config_path else "",
        video_dir=source,
        dataset=dataset,
        output_dir=str(output_dir),
        log_file=str(log_file),
        status=TrackExtractionRun.Status.RUNNING,
        pid=pid,
    )


def can_resume(run: TrackExtractionRun) -> bool:
    """A run can be resumed once it's known to be stopped (not RUNNING, or
    RUNNING but its pid is actually dead -- refresh_status() should be
    called first so a merely-stale row doesn't get a second process pointed
    at the same output_dir)."""
    return run.status != TrackExtractionRun.Status.RUNNING


def resume_run(run: TrackExtractionRun) -> TrackExtractionRun:
    """Relaunch a stopped run with `--resume`, reusing its own output_dir so
    `ar-extract-tracks` skips whatever videos it already finished before
    being interrupted (killed, crashed, host restart) and only redoes the
    rest -- see extract_tracks.py's --resume flag. Updates `run` in place
    rather than creating a new row: this is a continuation of the same
    logical run, writing into the same output_dir/tracks_index.csv.
    """
    output_dir = Path(run.output_dir)
    log_file = Path(run.log_file)
    config_path = Path(run.config_path) if run.config_path else None

    argv = _build_argv(video_dir=run.video_dir, output_dir=output_dir, config_path=config_path, resume=True)
    pid = _background.launch_detached(
        argv,
        log_file=log_file,
        exit_marker_prefix=EXIT_MARKER_PREFIX,
        cwd=settings.REPO_ROOT,
        append=True,  # keep the previous attempt's log instead of erasing it
    )

    run.pid = pid
    run.status = TrackExtractionRun.Status.RUNNING
    run.return_code = None
    run.finished_at = None
    run.save()
    return run


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


_PROGRESS_RE = re.compile(r"(?:Tracking|Skipping) (.+?) \((\d+)/(\d+)\)")


def progress(run: TrackExtractionRun) -> dict | None:
    """Best-effort per-video progress, parsed from the `Tracking <video>
    (i/N)` / `Skipping <video> (i/N)` lines extract_tracks.py logs for every
    source video. Returns None before the first such line is logged (e.g.
    the detector is still loading) or once there's no log to read.

    Not anchored to the start of the line: the real subprocess log has
    `action_recognition.utils.logging.get_logger`'s `"%(asctime)s
    %(levelname)s %(name)s: "` prefix before each message, so a `^` anchor
    (or requiring re.MULTILINE) would never match real output -- only the
    prefix-free fixtures a naive test would write by hand.
    """
    log_path = Path(run.log_file)
    if not log_path.exists():
        return None
    matches = _PROGRESS_RE.findall(log_path.read_text(errors="replace"))
    if not matches:
        return None
    video, current, total = matches[-1]
    return {"video": Path(video).name, "current": int(current), "total": int(total)}


def clip_count(run: TrackExtractionRun) -> int | None:
    """Clips produced so far. `tracks_index.csv` is only written once the
    whole run finishes (see extract_tracks.py), so while a run is still
    RUNNING it would otherwise show nothing at all -- count clip files
    already on disk instead, which is exactly what's landing in
    output_dir/<video_stem>/ as each clip is written, well before the index
    exists.
    """
    index_path = Path(run.output_dir) / "tracks_index.csv"
    if index_path.exists():
        # header line + one row per clip
        return max(0, len(index_path.read_text().splitlines()) - 1)

    output_dir = Path(run.output_dir)
    if not output_dir.is_dir():
        return None
    count = sum(1 for _ in output_dir.rglob("*.mp4"))
    return count if count else None


def results(run: TrackExtractionRun) -> list[dict]:
    """Per-video, per-track breakdown of what this run actually produced,
    read straight from output_dir/tracks_index.csv -- for the "detailed
    results" section of the extraction detail page. `tracks_index.csv` is
    rewritten after every source video finishes (see extract_tracks.py), so
    this already reflects partial progress for a still-RUNNING run, not just
    a finished one; returns [] before the first video has completed.
    """
    index_path = Path(run.output_dir) / "tracks_index.csv"
    if not index_path.exists():
        return []
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))

    videos: dict[str, dict] = {}
    for row in rows:
        video = videos.setdefault(
            row["source_video"],
            {"source_video": row["source_video"], "name": Path(row["source_video"]).name, "tracks": {}, "clip_count": 0},
        )
        video["clip_count"] += 1
        track_id = int(row["track_id"])
        track = video["tracks"].setdefault(track_id, {"track_id": track_id, "clips": []})
        track["clips"].append(
            {
                "window_index": int(row["window_index"]),
                "start_frame": int(row["start_frame"]),
                "end_frame": int(row["end_frame"]),
                "num_frames": int(row["num_frames"]),
                "clip_path": row["clip_path"],
                "name": Path(row["clip_path"]).name,
            }
        )

    out = list(videos.values())
    for video in out:
        tracks = sorted(video["tracks"].values(), key=lambda t: t["track_id"])
        for track in tracks:
            track["clips"].sort(key=lambda c: c["window_index"])
        video["tracks"] = tracks
        video["track_count"] = len(tracks)
    out.sort(key=lambda v: v["name"])
    return out


def _dest_clip_name(run: TrackExtractionRun, rel: Path) -> str:
    """The filename a clip lands under once imported into a dataset --
    output_dir/<video_stem>/trackN_winM.mp4 (rel) flattens to
    <run.name>__<video_stem>__trackN_winM.mp4, prefixed with the run name
    since two runs can share a video_stem. Shared by import_clips (which
    performs the copy) and provenance_for_dataset (which needs the same
    name to recognize an already-imported clip and trace it back to its
    source video/track/window) so the naming rule only lives in one place.
    """
    return f"{run.name}__{rel.parent.name}__{rel.name}"


def import_clips(run: TrackExtractionRun, dest_dir: Path) -> int:
    """Copy every clip this run produced into dest_dir -- a dataset's own
    video_dir (usually run.dataset's, see views.extraction) -- so the
    Label/Dataset pages, which read/write that dataset's folder, can reach
    multi-person clips without a manual filesystem move. Already-imported
    clips (same destination name and size) are skipped, so re-clicking after
    a run adds more clips only copies what's new. Returns the number copied.
    """
    output_dir = Path(run.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    for clip in discover_videos(output_dir):
        rel = clip.relative_to(output_dir)
        dest = dest_dir / _dest_clip_name(run, rel)
        if dest.exists() and dest.stat().st_size == clip.stat().st_size:
            continue
        shutil.copy2(clip, dest)
        imported += 1
    return imported


def provenance_for_dataset(dataset: Dataset) -> dict[str, dict]:
    """Maps every clip in `dataset` that was imported from a track
    extraction run back to its source video/track/window, so the Label page
    can group multi-person clips instead of showing them as unrelated
    videos. Derived entirely from each run's tracks_index.csv plus the same
    naming rule import_clips used (_dest_clip_name) -- nothing extra is
    persisted. A run whose `dataset` FK has gone null (the dataset it
    imported into was deleted) is skipped, so its clips -- if they still
    exist under some other dataset's video_dir -- fall back to being labeled
    as plain ungrouped videos rather than pointing at a stale run.
    """
    dest_dir = resolve_repo_path(dataset.video_dir)
    provenance: dict[str, dict] = {}
    for run in TrackExtractionRun.objects.filter(dataset=dataset, status=TrackExtractionRun.Status.SUCCEEDED):
        index_path = Path(run.output_dir) / "tracks_index.csv"
        if not index_path.exists():
            continue
        with open(index_path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            # clip_path in the index is the extractor's own absolute path
            # (output_dir/<stem>/trackN_winM.mp4) -- only its last two
            # components (stem folder, filename) feed _dest_clip_name, so
            # rebuild just those rather than assuming clip_path's shape.
            clip_path = Path(row["clip_path"])
            rel = Path(clip_path.parent.name) / clip_path.name
            dest = dest_dir / _dest_clip_name(run, rel)
            if not dest.exists():
                continue
            provenance[str(dest.resolve())] = {
                "run_id": run.pk,
                "run_name": run.name,
                "source_video": Path(row["source_video"]).stem,
                "track_id": int(row["track_id"]),
                "window_index": int(row["window_index"]),
            }
    return provenance


def group_clips(provenance: dict[str, dict]) -> list[dict]:
    """Nest provenance_for_dataset()'s flat per-clip map into one entry per
    (run, source video), each with its tracks (one per tracked person) and
    each track's window clips in window order -- the shape the Label page's
    grouped view renders, and that label_track uses to resolve one track's
    clip paths to label together."""
    groups: dict[tuple[int, str], dict] = {}
    for path, info in provenance.items():
        key = (info["run_id"], info["source_video"])
        group = groups.setdefault(
            key,
            {
                "run_id": info["run_id"],
                "run_name": info["run_name"],
                "source_video": info["source_video"],
                "tracks": {},
            },
        )
        track = group["tracks"].setdefault(info["track_id"], {"track_id": info["track_id"], "clips": []})
        track["clips"].append({"path": path, "window_index": info["window_index"]})

    out = []
    for group in groups.values():
        tracks = sorted(group["tracks"].values(), key=lambda t: t["track_id"])
        for track in tracks:
            track["clips"].sort(key=lambda c: c["window_index"])
        group["tracks"] = tracks
        group["track_count"] = len(tracks)
        group["clip_count"] = sum(len(t["clips"]) for t in tracks)
        out.append(group)
    out.sort(key=lambda g: (g["source_video"], g["run_id"]))
    return out
