"""Default person detector: a COCO-pretrained torchvision detection model,
filtered down to the 'person' class. No extra dependency beyond torchvision,
already required for the r3d18 model — mirrors that same
"pretrained torchvision backbone" pattern.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision

from ...utils.device import resolve_device
from ..types import Detection
from .registry import register_detector

COCO_PERSON_CLASS_ID = 1


@register_detector("fasterrcnn")
class FasterRCNNPersonDetector:
    def __init__(
        self,
        score_thresh: float = 0.6,
        pretrained: bool = True,
        device: str | None = None,
    ):
        weights = (
            torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            if pretrained
            else None
        )
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.model.eval()
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.score_thresh = score_thresh

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """frame: HWC uint8 RGB array, e.g. as produced by
        data.dataset.read_clip_frames / cv2.cvtColor(..., COLOR_BGR2RGB)."""
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        output = self.model(tensor)[0]

        detections = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if label.item() != COCO_PERSON_CLASS_ID or score.item() < self.score_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(Detection(box=(x1, y1, x2, y2), score=score.item()))
        return detections
