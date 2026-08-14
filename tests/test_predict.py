import torch

from action_recognition.inference import scene_predict
from action_recognition.inference.predict import predict
from action_recognition.models import build_model
from action_recognition.utils.checkpoint import save_checkpoint


def _write_tiny_checkpoint(path, label_map):
    model = build_model("cnn_lstm", num_classes=len(label_map), pretrained_backbone=False, hidden_size=8)
    config = {
        "model": {"name": "cnn_lstm", "params": {"pretrained_backbone": False, "hidden_size": 8}},
        "data": {"num_frames": 2, "image_size": 32},
    }
    save_checkpoint(path, model, label_map, config, epoch=1, metric=0.5)
    return path


def test_predict_returns_top_k_label_probability_pairs_summing_sensibly(synthetic_video, tmp_path):
    label_map = {"jump": 0, "wave": 1, "kick": 2}
    checkpoint_path = _write_tiny_checkpoint(tmp_path / "best.pt", label_map)

    results = predict(checkpoint_path, synthetic_video, top_k=2)

    assert len(results) == 2
    labels_seen = set()
    probs_seen = []
    for label, prob in results:
        assert label in label_map
        assert 0.0 <= prob <= 1.0
        labels_seen.add(label)
        probs_seen.append(prob)
    assert len(labels_seen) == 2  # no duplicate labels
    assert probs_seen == sorted(probs_seen, reverse=True)  # topk() returns descending order


def test_predict_clamps_top_k_to_number_of_classes(synthetic_video, tmp_path):
    label_map = {"jump": 0, "wave": 1}
    checkpoint_path = _write_tiny_checkpoint(tmp_path / "best.pt", label_map)

    results = predict(checkpoint_path, synthetic_video, top_k=10)

    assert len(results) == 2  # clamped down from 10 to len(label_map)


def test_predict_uses_num_frames_and_image_size_from_the_checkpoints_config(synthetic_video, tmp_path):
    label_map = {"jump": 0, "wave": 1}
    checkpoint_path = _write_tiny_checkpoint(tmp_path / "best.pt", label_map)

    # Should not raise even though synthetic_video's native size (64x64)
    # differs from the checkpoint's configured image_size (32): predict()
    # must resize/crop itself rather than assume a matching input size.
    results = predict(checkpoint_path, synthetic_video, top_k=1)
    assert len(results) == 1


def test_predict_runs_on_explicit_cpu_device(synthetic_video, tmp_path):
    label_map = {"jump": 0, "wave": 1}
    checkpoint_path = _write_tiny_checkpoint(tmp_path / "best.pt", label_map)

    results = predict(checkpoint_path, synthetic_video, device="cpu", top_k=1)
    assert results[0][0] in label_map


def test_predict_uses_resolve_device_instead_of_always_defaulting_to_cpu(synthetic_video, tmp_path, monkeypatch):
    # Regression guard: predict() used to hardcode torch.device("cpu")
    # whenever --device wasn't given, ignoring an available GPU even though
    # train.py/scene_predict.py/the detector all autodetect via
    # resolve_device(). Confirm predict() now goes through the same call.
    label_map = {"jump": 0, "wave": 1}
    checkpoint_path = _write_tiny_checkpoint(tmp_path / "best.pt", label_map)

    from action_recognition.inference import predict as predict_module

    captured = {}
    real_resolve_device = predict_module.resolve_device

    def fake_resolve_device(requested=None):
        captured["requested"] = requested
        return real_resolve_device(requested)

    monkeypatch.setattr(predict_module, "resolve_device", fake_resolve_device)

    results = predict(checkpoint_path, synthetic_video, top_k=1)

    assert "requested" in captured  # predict() went through resolve_device(), not a hardcoded default
    assert captured["requested"] is None  # no --device given
    assert results[0][0] in label_map


def test_predict_scene_passes_resolved_device_to_detector(synthetic_video, tmp_path, monkeypatch):
    # Regression guard: the detector used to be built with no device at all
    # (always CPU) even when the classifier model itself ran on a requested
    # GPU -- confirm the two now agree.
    label_map = {"jump": 0, "wave": 1}
    checkpoint_path = _write_tiny_checkpoint(tmp_path / "best.pt", label_map)

    captured = {}

    class FakeDetector:
        def detect(self, frame):
            return []

    def fake_build_detector(name, **params):
        captured["params"] = params
        return FakeDetector()

    monkeypatch.setattr(scene_predict, "build_detector", fake_build_detector)

    results = scene_predict.predict_scene(checkpoint_path, synthetic_video, device="cpu")

    assert captured["params"]["device"] == "cpu"
    assert results == []  # FakeDetector never finds anyone, so no tracks/windows either
