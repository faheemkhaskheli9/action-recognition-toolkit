import argparse

import pytest
import torch
import yaml

from action_recognition.data.manifest import build_label_map, discover_class_folders, stratified_split, write_manifest
from action_recognition.training.train import (
    DEFAULTS,
    _deep_update,
    apply_cli_overrides,
    build_arg_parser,
    load_config,
    main,
    resolve_device,
)


# --------------------------------------------------------------------- #
# _deep_update / load_config
# --------------------------------------------------------------------- #

def test_deep_update_overrides_nested_keys_without_dropping_siblings():
    base = {"data": {"num_frames": 16, "image_size": 112}, "model": {"name": "cnn_lstm"}}
    override = {"data": {"num_frames": 8}}

    result = _deep_update(base, override)

    assert result["data"]["num_frames"] == 8
    assert result["data"]["image_size"] == 112  # untouched sibling survives
    assert result["model"]["name"] == "cnn_lstm"


def test_load_config_with_no_path_returns_defaults():
    config = load_config(None)
    assert config == DEFAULTS
    # must be a copy, not the same nested dicts as DEFAULTS
    config["data"]["num_frames"] = 1
    assert DEFAULTS["data"]["num_frames"] != 1


def test_load_config_merges_yaml_file_over_defaults(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"train": {"epochs": 3}, "model": {"name": "r3d18"}}))

    config = load_config(path)

    assert config["train"]["epochs"] == 3
    assert config["model"]["name"] == "r3d18"
    assert config["data"]["num_frames"] == DEFAULTS["data"]["num_frames"]  # default kept


# --------------------------------------------------------------------- #
# apply_cli_overrides / build_arg_parser
# --------------------------------------------------------------------- #

def test_apply_cli_overrides_only_touches_provided_flags():
    config = load_config(None)
    args = build_arg_parser().parse_args(["--epochs", "5", "--model", "r3d18"])

    result = apply_cli_overrides(config, args)

    assert result["train"]["epochs"] == 5
    assert result["model"]["name"] == "r3d18"
    assert result["data"]["batch_size"] == DEFAULTS["data"]["batch_size"]  # not overridden


def test_apply_cli_overrides_with_no_flags_is_a_no_op():
    config = load_config(None)
    args = build_arg_parser().parse_args([])

    result = apply_cli_overrides(config, args)

    assert result == DEFAULTS


def test_build_arg_parser_returns_argument_parser():
    assert isinstance(build_arg_parser(), argparse.ArgumentParser)


# --------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------- #

def test_resolve_device_honors_explicit_request():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_falls_back_to_cpu_when_nothing_requested(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device(None) == torch.device("cpu")


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device(None) == torch.device("cuda")


# --------------------------------------------------------------------- #
# main() end-to-end smoke test
# --------------------------------------------------------------------- #

def test_main_trains_one_epoch_and_writes_checkpoints(tmp_path, class_folder_dataset, monkeypatch):
    manifest = stratified_split(discover_class_folders(class_folder_dataset), val_frac=0.2, test_frac=0.2, seed=0)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest, manifest_path)
    assert set(manifest["split"]) & {"val", "test"}, "fixture is too small to exercise all splits"

    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "run"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"num_frames": 2, "image_size": 32, "batch_size": 1, "num_workers": 0},
                "model": {"name": "cnn_lstm", "params": {"pretrained_backbone": False, "hidden_size": 8}},
                "train": {"epochs": 1, "lr": 1e-3, "weight_decay": 0.0, "seed": 0, "device": "cpu"},
            }
        )
    )

    monkeypatch.setattr(
        "sys.argv",
        ["ar-train", "--config", str(config_path), "--manifest", str(manifest_path), "--output-dir", str(output_dir)],
    )

    main()

    assert (output_dir / "last.pt").exists()
    assert (output_dir / "best.pt").exists()  # manifest has a val split
