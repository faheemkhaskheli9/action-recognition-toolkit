"""Detect + track people across a folder of raw multi-person videos, and
crop+window each track into single-person clip files under `output_dir`.

That output folder is then ready for ar-build-manifest or the web app's
Label page, exactly like any other flat folder of clips — nothing
downstream needs to know these came from tracking instead of manual
trimming, so whatever action taxonomy you label them with (restaurant
staff tasks, warehouse activity, anything else) trains the same way.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from action_recognition.data.manifest import discover_videos
from action_recognition.tracking.detectors import build_detector
from action_recognition.tracking.extract import extract_tracks_to_clips
from action_recognition.tracking.trackers import build_tracker
from action_recognition.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULTS = {
    "detector": {"name": "fasterrcnn", "params": {"score_thresh": 0.6}},
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


def load_config(path: Path | None) -> dict:
    config = {section: dict(values) for section, values in DEFAULTS.items()}
    if path is not None:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        _deep_update(config, user_config)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="Folder of raw multi-person videos (searched recursively)")
    parser.add_argument("output_dir", type=Path, help="Where cropped per-person clips are written")
    parser.add_argument("--config", type=Path, default=None, help="YAML config, see configs/tracking/default.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)

    videos = discover_videos(args.video_dir)
    if not videos:
        raise SystemExit(f"No videos found under {args.video_dir}")

    detector = build_detector(config["detector"]["name"], **config["detector"]["params"])

    def tracker_factory():
        # a fresh tracker per video: track IDs are only meaningful within one source video
        return build_tracker(config["tracker"]["name"], **config["tracker"]["params"])

    all_rows = []
    for video in videos:
        logger.info(f"Tracking {video}")
        rows = extract_tracks_to_clips(video, args.output_dir, detector, tracker_factory, **config["extract"])
        logger.info(f"  {len(rows)} clip(s)")
        all_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "tracks_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} clip(s) across {len(videos)} video(s) to {args.output_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
