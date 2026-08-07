import logging

import torch
import torch.nn as nn

from action_recognition.utils.checkpoint import load_checkpoint, save_checkpoint
from action_recognition.utils.logging import get_logger
from action_recognition.utils.seed import set_seed


def test_set_seed_makes_torch_rand_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)

    assert torch.equal(a, b)


def test_get_logger_returns_same_configured_logger_for_same_name():
    logger1 = get_logger("action_recognition.test")
    logger2 = get_logger("action_recognition.test")

    assert logger1 is logger2
    assert logger1.level == logging.INFO
    # get_logger() must not stack a new handler on every call.
    assert len(logger1.handlers) == 1


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = nn.Linear(4, 2)
    label_map = {"jump": 0, "wave": 1}
    config = {"model": {"name": "cnn_lstm", "params": {}}, "data": {"num_frames": 8}}
    path = tmp_path / "run" / "best.pt"

    save_checkpoint(path, model, label_map, config, epoch=3, metric=0.75)
    assert path.exists()

    checkpoint = load_checkpoint(path)

    assert checkpoint["label_map"] == label_map
    assert checkpoint["config"] == config
    assert checkpoint["epoch"] == 3
    assert checkpoint["metric"] == 0.75
    for key, value in model.state_dict().items():
        assert torch.equal(checkpoint["model_state"][key], value)


def test_save_checkpoint_creates_parent_directories(tmp_path):
    model = nn.Linear(2, 2)
    path = tmp_path / "a" / "b" / "c" / "last.pt"

    save_checkpoint(path, model, {}, {}, epoch=1, metric=0.0)

    assert path.parent.is_dir()
    assert path.exists()
