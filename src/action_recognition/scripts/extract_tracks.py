"""Detect + track people across a folder of raw multi-person videos (or a
single raw multi-person video file), and crop+window each track into
single-person clip files under `output_dir`.

That output folder is then ready for ar-build-manifest or the web app's
Label page, exactly like any other flat folder of clips — nothing
downstream needs to know these came from tracking instead of manual
trimming, so whatever action taxonomy you label them with (restaurant
staff tasks, warehouse activity, anything else) trains the same way.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml

from action_recognition.data.manifest import VIDEO_EXTENSIONS, discover_videos
from action_recognition.tracking.detectors import build_detector
from action_recognition.tracking.extract import extract_tracks_to_clips
from action_recognition.tracking.trackers import build_tracker
from action_recognition.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULTS = {
    "detector": {"name": "fasterrcnn", "params": {"score_thresh": 0.6, "device": None}},
    "tracker": {"name": "iou", "params": {"iou_thresh": 0.3, "max_age": 15}},
    "extract": {
        "frame_stride": 2,
        "window_frames": 16,
        "stride_frames": 8,
        "min_track_frames": 16,
        "crop_padding": 0.2,
        "output_size": 224,
    },
}

INDEX_FIELDS = ["source_video", "track_id", "window_index", "start_frame", "end_frame", "num_frames", "clip_path"]


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _discover_input_videos(path: Path) -> list[Path]:
    """`path` may be a single video file (process just that one) or a
    directory (searched recursively for every recognized video file) --
    lets ar-extract-tracks run over one video without staging it into its
    own folder first."""
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
            raise SystemExit(f"{path} is not a recognized video file (recognized extensions: {extensions})")
        return [path]
    return discover_videos(path)


def _load_completed_videos(index_path: Path) -> tuple[list[dict], set[str]]:
    """Read a previous, possibly-interrupted run's tracks_index.csv (if any)
    and return its rows plus the set of source videos it already has
    complete entries for. A video only gets an index entry once every one of
    its windows has been written (see main()'s loop below), so anything
    missing here is exactly the work `--resume` still needs to do."""
    if not index_path.exists():
        return [], set()
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {row["source_video"] for row in rows}


def load_config(path: Path | None) -> dict:
    config = {section: dict(values) for section, values in DEFAULTS.items()}
    if path is not None:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        _deep_update(config, user_config)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Folder of raw multi-person videos (searched recursively), or a single video file",
    )
    parser.add_argument("output_dir", type=Path, help="Where cropped per-person clips are written")
    parser.add_argument("--config", type=Path, default=None, help="YAML config, see configs/tracking/default.yaml")
    parser.add_argument(
        "--device", type=str, default=None, help="cpu | cuda | mps, autodetected if omitted (overrides detector.params.device)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Pick up an interrupted run: skip videos already recorded in "
            "output_dir/tracks_index.csv, and discard any partial clips left "
            "under output_dir/<video_stem>/ for a video that didn't finish."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["detector"]["params"]["device"] = args.device

    videos = _discover_input_videos(args.video_dir)
    if not videos:
        resolved = args.video_dir.resolve()
        if not args.video_dir.exists():
            raise SystemExit(f"video_dir does not exist (looked for a file or directory at {resolved})")
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise SystemExit(
            f"No videos found under {resolved} (searched recursively; recognized extensions: {extensions})"
        )

    detector = build_detector(config["detector"]["name"], **config["detector"]["params"])
    if hasattr(detector, "device"):
        logger.info(f"Using device: {detector.device}")

    def tracker_factory():
        # a fresh tracker per video: track IDs are only meaningful within one source video
        return build_tracker(config["tracker"]["name"], **config["tracker"]["params"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "tracks_index.csv"
    all_rows, completed_videos = _load_completed_videos(index_path) if args.resume else ([], set())
    if completed_videos:
        logger.info(f"Resuming: {len(completed_videos)} video(s) already extracted, skipping them")

    def _write_index() -> None:
        # Rewritten after every video (not just once at the end) so a run
        # interrupted partway through -- killed, crashed, host restart --
        # leaves a complete, resumable index of everything finished so far,
        # instead of losing all provenance for the whole run.
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)

    for i, video in enumerate(videos, start=1):
        if str(video) in completed_videos:
            logger.info(f"Skipping {video} ({i}/{len(videos)}) -- already extracted")
            continue
        if args.resume:
            stale_dir = args.output_dir / video.stem
            if stale_dir.exists():
                # No index entry for this video, but its output folder
                # exists -- an earlier attempt started it and got
                # interrupted before finishing. Clear the partial clips so
                # this attempt doesn't mix stale and fresh windows.
                logger.info(f"  discarding partial output from an earlier attempt: {stale_dir}")
                shutil.rmtree(stale_dir)
        logger.info(f"Tracking {video} ({i}/{len(videos)})")
        rows = extract_tracks_to_clips(video, args.output_dir, detector, tracker_factory, **config["extract"])
        logger.info(f"  {len(rows)} clip(s) from {video.name} -- {len(all_rows) + len(rows)} total so far")
        all_rows.extend(rows)
        _write_index()

    print(f"Wrote {len(all_rows)} clip(s) across {len(videos)} video(s) to {args.output_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
