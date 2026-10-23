"""Stage 2: Supervised Defect Classification.

Responsibilities:
1. Only execute when Stage 1 (PatchCore + threshold) flags an image as DEFECTIVE.
2. Load and execute the trained ResNet-18 defect classifier (best_model.pt).
3. Classify into the authoritative 4-class taxonomy:
     - Scratch
     - Crack
     - Discoloration
     - UNKNOWN
4. Return the predicted class, confidence score, and complete normalized probability distribution.

Design & Limitations:
- The classifier operates on the full RGB image (resized and normalized to ImageNet statistics).
- TECHNICAL LIMITATION: Heatmap-guided bounding-box cropping is intentionally NOT performed
  because the defect classifier was trained and evaluated on full image frames. Cropping
  would introduce out-of-distribution artifacts without retraining.
- Confidence scores are raw softmax probabilities and NOT calibrated probabilities.
- Production confidence gating is NOT applied by default; it remains an offline diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defectvision.models.defect_classifier import (
    DEFAULT_DEFECT_CLASSES,
    ClassificationPrediction,
    DefectClassifierPipeline,
)


@dataclass
class Stage2Result:
    """Result of Stage 2 defect classification."""

    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": float(self.confidence),
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
            "image_path": self.image_path,
        }


class Stage2DefectClassifier:
    """Stage 2 Defect Classifier wrapping DefectClassifierPipeline."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        pipeline: DefectClassifierPipeline | None = None,
        device: str | None = None,
    ) -> None:
        self.pipeline = pipeline or DefectClassifierPipeline(device=device)
        self.loaded_checkpoint_path: str | None = None

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load trained ResNet-18 defect classifier weights."""
        p = Path(checkpoint_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {p}")

        resolved_str = str(p).replace("\\", "/")
        if self.loaded_checkpoint_path == resolved_str:
            return  # Already loaded

        self.pipeline.load_checkpoint(p)
        self.loaded_checkpoint_path = resolved_str

    def classify(self, image_path: str | Path) -> Stage2Result:
        """Classify a defective image into one of the 4 target defect categories."""
        if self.loaded_checkpoint_path is None:
            raise RuntimeError(
                "Stage 2 Defect Classifier has no checkpoint loaded. Call load_checkpoint() first."
            )

        img_p = Path(image_path).resolve()
        if not img_p.is_file():
            raise FileNotFoundError(f"Image not found for classification: {img_p}")

        pred: ClassificationPrediction = self.pipeline.predict_image(img_p)

        if pred.predicted_class not in DEFAULT_DEFECT_CLASSES:
            raise ValueError(
                f"Unexpected predicted class '{pred.predicted_class}'. "
                f"Must be one of {DEFAULT_DEFECT_CLASSES}"
            )

        return Stage2Result(
            predicted_class=pred.predicted_class,
            confidence=pred.confidence,
            class_probabilities=pred.class_probabilities,
            image_path=str(img_p).replace("\\", "/"),
        )
