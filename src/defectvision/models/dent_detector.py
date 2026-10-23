"""Experimental Dent Detector Module for DefectVision.

Implements offline-trained Faster R-CNN MobileNetV3-Large FPN for Dent detection
on steel surfaces (CR7-DET dataset).

Isolated from the production PatchCore -> ResNet-18 pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image
import torch
import torchvision.models.detection as d_models
import torchvision.transforms.functional as TF

CR7_CLASS_NAMES_WITH_BG = [
    "__background__",
    "Inclusion",
    "Dents",
    "Oil Spots",
    "Pits",
    "Punching",
    "Linear",
    "Macular",
]

DENT_CLASS_ID = 2  # CR7-DET class 1, shifted by +1 for torchvision background
DEFAULT_CONF_THRESHOLD = 0.30
DEFAULT_CHECKPOINT_PATH = Path("experiments/dent_detection_pilot/checkpoints/best_detector.pth")


@dataclass
class DentBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = DENT_CLASS_ID
    class_name: str = "Dents"

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class DentDetectionResult:
    dents_detected: int
    boxes: list[DentBoundingBox]
    confidence_threshold: float
    image_width: int
    image_height: int
    execution_time_ms: float
    model_name: str = "Faster R-CNN MobileNetV3-Large FPN"
    dataset: str = "CR7-DET (Steel Surface)"
    experimental: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dents_detected": self.dents_detected,
            "boxes": [b.to_dict() for b in self.boxes],
            "confidence_threshold": self.confidence_threshold,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "model_name": self.model_name,
            "dataset": self.dataset,
            "experimental": self.experimental,
            "message": self.message,
        }


class DentDetector:
    """Dent detector model wrapper for inference."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: torch.device | str | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            env_path = os.getenv("DEFECTVISION_DENT_CHECKPOINT")
            if env_path:
                self.checkpoint_path = Path(env_path)
            elif DEFAULT_CHECKPOINT_PATH.exists():
                self.checkpoint_path = DEFAULT_CHECKPOINT_PATH
            else:
                # Try finding relative to repo root
                repo_root = Path(__file__).resolve().parent.parent.parent.parent
                self.checkpoint_path = repo_root / DEFAULT_CHECKPOINT_PATH
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Dent detector checkpoint not found at: {self.checkpoint_path}")

        self.model = d_models.fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            num_classes=len(CR7_CLASS_NAMES_WITH_BG),
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = DEFAULT_CONF_THRESHOLD,
    ) -> DentDetectionResult:
        t0 = time.perf_counter()

        image_rgb = image.convert("RGB")
        w_img, h_img = image_rgb.size

        img_tensor = TF.to_tensor(image_rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        pred_boxes = outputs["boxes"].cpu().numpy().tolist()
        pred_labels = outputs["labels"].cpu().numpy().tolist()
        pred_scores = outputs["scores"].cpu().numpy().tolist()

        boxes: list[DentBoundingBox] = []
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            # Target class: Dents (label == 2)
            if label == DENT_CLASS_ID and score >= confidence_threshold:
                x1, y1, x2, y2 = box
                # Clamp coordinates to image boundaries
                x1 = max(0.0, min(float(x1), float(w_img)))
                y1 = max(0.0, min(float(y1), float(h_img)))
                x2 = max(0.0, min(float(x2), float(w_img)))
                y2 = max(0.0, min(float(y2), float(h_img)))
                if x2 > x1 and y2 > y1:
                    boxes.append(
                        DentBoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=float(score),
                            class_id=DENT_CLASS_ID,
                            class_name="Dents",
                        )
                    )

        # Sort by confidence descending
        boxes.sort(key=lambda b: b.confidence, reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dents_count = len(boxes)

        if dents_count > 0:
            msg = f"{dents_count} dent{'s' if dents_count > 1 else ''} detected with confidence >= {confidence_threshold:.2f}."
        else:
            msg = f"No dent detected at confidence threshold {confidence_threshold:.2f}."

        return DentDetectionResult(
            dents_detected=dents_count,
            boxes=boxes,
            confidence_threshold=confidence_threshold,
            image_width=w_img,
            image_height=h_img,
            execution_time_ms=elapsed_ms,
            model_name="Faster R-CNN MobileNetV3-Large FPN",
            dataset="CR7-DET (Steel Surface)",
            experimental=True,
            message=msg,
        )


_detector_instance: DentDetector | None = None


def get_dent_detector() -> DentDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DentDetector()
    return _detector_instance
