"""Unified Inspection Result data structures for DefectVision.

Represents the complete, typed outcome of two-stage inspection:
- Stage 1 anomaly detection result (verdict, raw score, calibrated threshold, margin, heatmap)
- Stage 2 defect classification result (defect class, confidence, probabilities) if executed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

from defectvision.calibration import InspectionStatus


@dataclass
class InspectionResult:
    """Unified result of an industrial inspection run."""

    product_id: str
    image_path: str
    verdict: str  # "NORMAL" or "DEFECTIVE"
    is_defective: bool
    anomaly_score: float
    anomaly_threshold: float
    margin: float
    stage2_executed: bool
    defect_class: str | None = None
    classifier_confidence: float | None = None
    classifier_probabilities: dict[str, float] | None = None
    heatmap_path: str | None = None
    inspection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.anomaly_score = float(self.anomaly_score)
        self.anomaly_threshold = float(self.anomaly_threshold)
        self.margin = float(self.margin)
        if self.classifier_confidence is not None:
            self.classifier_confidence = float(self.classifier_confidence)
        if self.classifier_probabilities is not None:
            self.classifier_probabilities = {
                k: float(v) for k, v in self.classifier_probabilities.items()
            }
        if self.heatmap_path is not None:
            self.heatmap_path = str(self.heatmap_path).replace("\\", "/")

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "inspection_id": self.inspection_id,
            "product_id": self.product_id,
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "is_defective": self.is_defective,
            "anomaly_score": self.anomaly_score,
            "anomaly_threshold": self.anomaly_threshold,
            "margin": self.margin,
            "stage2_executed": self.stage2_executed,
            "defect_class": self.defect_class,
            "classifier_confidence": self.classifier_confidence,
            "classifier_probabilities": self.classifier_probabilities,
            "heatmap_path": self.heatmap_path,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        """Save inspection result to a JSON file."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
        return p
