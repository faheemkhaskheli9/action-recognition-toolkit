from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from action_recognition.data.dataset import datasets_from_manifest
from action_recognition.data.manifest import build_label_map, load_label_map, read_manifest
from action_recognition.models import build_model
from action_recognition.training.engine import run_epoch
from action_recognition.utils.checkpoint import load_checkpoint, save_checkpoint
from action_recognition.utils.logging import get_logger
from action_recognition.utils.seed import set_seed

logger = get_logger(__name__)

DEFAULTS = {
    "data": {"num_frames": 16, "image_size": 112, "batch_size": 8, "num_workers": 2},
    "model": {"name": "cnn_lstm", "params": {}},
    "train": {"epochs": 20, "lr": 1e-4, "weight_decay": 1e-4, "seed": 42, "output_dir": "runs/exp1"},
}


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
    parser = argparse.ArgumentParser(description="Train a custom action recognition model")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file, see configs/")
    parser.add_argument("--manifest", type=Path, default=None, help="Overrides data.manifest")
    parser.add_argument("--model", type=str, default=None, help="Overrides model.name (see registry.available_models())")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps, autodetected if omitted")
    return parser


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.manifest is not None:
        config["data"]["manifest"] = str(args.manifest)
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.output_dir is not None:
        config["train"]["output_dir"] = str(args.output_dir)
    if args.device is not None:
        config["train"]["device"] = args.device
    return config


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    manifest_path = config["data"].get("manifest")
    if not manifest_path:
        raise SystemExit("data.manifest must be set, via --manifest or the config file")
    manifest_path = Path(manifest_path)

    set_seed(config["train"]["seed"])
    device = resolve_device(config["train"].get("device"))
    logger.info(f"Using device: {device}")

    manifest = read_manifest(manifest_path)
    label_map_path = manifest_path.with_name("label_map.json")
    label_map = load_label_map(label_map_path) if label_map_path.exists() else build_label_map(manifest["label"])
    logger.info(f"{len(label_map)} classes: {sorted(label_map)}")

    datasets = datasets_from_manifest(
        manifest,
        label_map,
        num_frames=config["data"]["num_frames"],
        image_size=config["data"]["image_size"],
    )
    if "train" not in datasets:
        raise SystemExit("Manifest has no 'train' split rows — run split_dataset first")

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=config["data"]["num_workers"],
        )
        for split, dataset in datasets.items()
    }

    model = build_model(config["model"]["name"], num_classes=len(label_map), **config["model"]["params"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        message = f"epoch {epoch}/{config['train']['epochs']} train_loss={train_loss:.4f} train_acc={train_acc:.4f}"

        if "val" in loaders:
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
            message += f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(output_dir / "best.pt", model, label_map, config, epoch, val_acc)

        logger.info(message)
        save_checkpoint(output_dir / "last.pt", model, label_map, config, epoch, train_acc)

    if "test" in loaders:
        best_ckpt_path = output_dir / "best.pt"
        if best_ckpt_path.exists():
            model.load_state_dict(load_checkpoint(best_ckpt_path, map_location=str(device))["model_state"])
        test_loss, test_acc = run_epoch(model, loaders["test"], criterion, device)
        logger.info(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
