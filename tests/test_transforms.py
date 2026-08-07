import numpy as np
import torch

from action_recognition.data.transforms import build_transform


def _frame(size=200):
    rng = np.random.default_rng(0)
    return (rng.random((size, size, 3)) * 255).astype(np.uint8)


def test_eval_transform_center_crops_to_image_size():
    transform = build_transform(image_size=64, train=False)
    out = transform(_frame())

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32


def test_train_transform_random_resized_crop_to_image_size():
    transform = build_transform(image_size=64, train=True)
    out = transform(_frame())

    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32


def test_transform_output_is_normalized_not_raw_pixel_range():
    # Normalize() shifts by ImageNet mean/std, so a mid-gray frame should not
    # remain close to its raw (frame / 255) scaled value.
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    transform = build_transform(image_size=64, train=False)
    out = transform(frame)

    unnormalized_equivalent = 128 / 255
    assert not torch.allclose(out, torch.full_like(out, unnormalized_equivalent), atol=1e-3)
