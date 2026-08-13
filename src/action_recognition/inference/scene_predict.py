"""Run a trained checkpoint over every tracked person in a raw multi-person
video: detect+track (the same pipeline `ar-extract-tracks` uses), slide the
checkpoint's own clip length across each track, and classify each window —
producing a per-person, per-time-window action timeline instead of the
single whole-clip prediction `ar-predict` gives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from action_recognition.data.transforms import build_transform
from action_recognition.models import build_model
from action_recognition.tracking.detectors import build_detector
from action_recognition.tracking.extract import read_window_frames, track_video, windows_for_track
from action_recognition.tracking.trackers import build_tracker
from action_recognition.utils.checkpoint import load_checkpoint
from action_recognition.utils.device import resolve_device

DEFAULT_DETECTOR = {"name": "fasterrcnn", "params": {"score_thresh": 0.6}}
DEFAULT_TRACKER = {"name": "iou", "params": {"iou_thresh": 0.3, "max_age": 15}}


def predict_scene(
    checkpoint_path: Path,
    video_path: Path,
    device: str | None = None,
    frame_stride: int = 2,
    stride_frames: int | None = None,
    detector_config: dict | None = None,
    tracker_config: dict | None = None,
) -> list[dict]:
    """Returns a list of window predictions, one per classified track
    window: {track_id, window_index, start_frame, end_frame, box, label,
    confidence}."""
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    label_map = checkpoint["label_map"]
    idx_to_label = {idx: label for label, idx in label_map.items()}
    num_frames = config["data"]["num_frames"]

    resolved_device = resolve_device(device)
    model = build_model(config["model"]["name"], num_classes=len(label_map), **config["model"]["params"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device).eval()
    transform = build_transform(config["data"]["image_size"], train=False)

    detector_config = detector_config or DEFAULT_DETECTOR
    tracker_config = tracker_config or DEFAULT_TRACKER
    detector_params = {"device": str(resolved_device), **detector_config.get("params", {})}
    detector = build_detector(detector_config["name"], **detector_params)
    tracker = build_tracker(tracker_config["name"], **tracker_config.get("params", {}))

    tracks = track_video(str(video_path), detector, tracker, frame_stride=frame_stride)

    results = []
    for track in tracks:
        windows = windows_for_track(
            track,
            window_frames=num_frames,
            stride_frames=stride_frames or num_frames,
            min_track_frames=num_frames,
        )
        for window in windows:
            frames = read_window_frames(str(video_path), window)
            if len(frames) < num_frames:
                # a couple of frames failed to decode near the clip edge; pad by
                # repeating the last good one rather than dropping the window
                frames += [frames[-1]] * (num_frames - len(frames))

            clip = torch.stack([transform(f) for f in frames], dim=0).unsqueeze(0).to(resolved_device)
            with torch.no_grad():
                probs = torch.softmax(model(clip), dim=1).squeeze(0)
            top_prob, top_idx = probs.max(dim=0)
            box = window.boxes[len(window.boxes) // 2]  # representative box for the window

            results.append(
                {
                    "track_id": window.track_id,
                    "window_index": window.window_index,
                    "start_frame": window.frame_indices[0],
                    "end_frame": window.frame_indices[-1],
                    "box": [round(v, 1) for v in box],
                    "label": idx_to_label[top_idx.item()],
                    "confidence": round(top_prob.item(), 4),
                }
            )
    return results


def annotate_video(video_path: Path, results: list[dict], output_path: Path) -> None:
    """Burn per-person boxes + current label onto a copy of the source video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    active_by_frame: dict[int, list[dict]] = {}
    for r in results:
        for f in range(r["start_frame"], r["end_frame"] + 1):
            active_by_frame.setdefault(f, []).append(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        frame_idx = 0
        ok, frame = cap.read()
        while ok:
            for r in active_by_frame.get(frame_idx, []):
                x1, y1, x2, y2 = (int(v) for v in r["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"#{r['track_id']} {r['label']} {r['confidence'] * 100:.0f}%"
                cv2.putText(frame, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.write(frame)
            frame_idx += 1
            ok, frame = cap.read()
    finally:
        writer.release()
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect, track, and classify every person in a scene video")
    parser.add_argument("checkpoint", type=Path, help="Path to best.pt / last.pt")
    parser.add_argument("video", type=Path, help="Raw multi-person video file")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frame-stride", type=int, default=2, help="Run the detector every Nth frame")
    parser.add_argument("--annotate", type=Path, default=None, help="Write an annotated copy of the video here")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON timeline here instead of stdout")
    args = parser.parse_args()

    results = predict_scene(args.checkpoint, args.video, device=args.device, frame_stride=args.frame_stride)
    output_json = json.dumps(results, indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"Wrote {len(results)} window prediction(s) to {args.output}")
    else:
        print(output_json)

    if args.annotate:
        annotate_video(args.video, results, args.annotate)
        print(f"Wrote annotated video to {args.annotate}")


if __name__ == "__main__":
    main()
