import pytest
import torch

from action_recognition.models import available_models, build_model
from action_recognition.models.registry import register_model


def test_available_models_lists_builtins():
    assert {"cnn_lstm", "r3d18"} <= set(available_models())


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("does_not_exist", num_classes=3)


def test_register_model_rejects_duplicate_name():
    register_model("only_once")(object)
    with pytest.raises(ValueError):
        register_model("only_once")(object)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("cnn_lstm", {"pretrained_backbone": False, "hidden_size": 32}),
        ("r3d18", {"pretrained": False}),
    ],
)
def test_model_forward_pass_shape(name, kwargs):
    num_classes = 5
    model = build_model(name, num_classes=num_classes, **kwargs)
    model.eval()

    clip = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        logits = model(clip)

    assert logits.shape == (2, num_classes)
