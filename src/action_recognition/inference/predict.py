from __future__ import annotations

import argparse
from pathlib import Path

import torch

from action_recognition.data.dataset import read_clip_frames
from action_recognition.data.transforms import build_transform
from action_recognition.models import build_model
from action_recognition.utils.checkpoint import load_checkpoint
from action_recognition.utils.device import resolve_device


def predict(checkpoint_path: Path, video_path: Path, device: str | None = None, top_k: int = 3):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    label_map = checkpoint["label_map"]
    idx_to_label = {idx: label for label, idx in label_map.items()}

    resolved_device = resolve_device(device)
    model = build_model(config["model"]["name"], num_classes=len(label_map), **config["model"]["params"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device).eval()

    frames = read_clip_frames(str(video_path), config["data"]["num_frames"])
    transform = build_transform(config["data"]["image_size"], train=False)
    clip = torch.stack([transform(frame) for frame in frames], dim=0).unsqueeze(0).to(resolved_device)

    with torch.no_grad():
        probs = torch.softmax(model(clip), dim=1).squeeze(0)

    top_k = min(top_k, len(label_map))
    top_probs, top_indices = probs.topk(top_k)
    return [(idx_to_label[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a video with a trained checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to best.pt / last.pt")
    parser.add_argument("video", type=Path, help="Video file to classify")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictions = predict(args.checkpoint, args.video, device=args.device, top_k=args.top_k)
    for label, prob in predictions:
        print(f"{label}: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
