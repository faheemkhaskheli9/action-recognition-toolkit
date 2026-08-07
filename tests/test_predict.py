import torch

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
