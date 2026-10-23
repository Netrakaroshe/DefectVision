"""Product Profile specification for DefectVision.

Encapsulates product-specific parameters required for two-stage inspection:
- Product identity
- Reference to fitted PatchCore model / memory bank checkpoint
- Calibrated anomaly threshold
- Associated calibration metadata
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ProductProfile:
    """Product configuration profile holding calibrated model references and thresholds."""

    product_id: str
    patchcore_model_path: str
    threshold: float
    calibration_artifact_path: str | None = None
    calibration_method: str = "unknown"
    calibration_timestamp: str = ""
    image_size: tuple[int, int] = (256, 256)
    backbone: str = "wide_resnet50_2"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.patchcore_model_path = str(self.patchcore_model_path).replace("\\", "/")
        if self.calibration_artifact_path is not None:
            self.calibration_artifact_path = str(self.calibration_artifact_path).replace("\\", "/")
        self.threshold = float(self.threshold)
        if isinstance(self.image_size, list):
            self.image_size = tuple(self.image_size)

    def validate(self, check_paths_exist: bool = False) -> None:
        """Validate profile integrity."""
        if not self.product_id or not str(self.product_id).strip():
            raise ValueError("product_id must be a non-empty string.")

        if not self.patchcore_model_path or not str(self.patchcore_model_path).strip():
            raise ValueError("patchcore_model_path must be a non-empty path.")

        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

        if check_paths_exist:
            model_p = Path(self.patchcore_model_path).resolve()
            if not model_p.is_file():
                raise FileNotFoundError(f"PatchCore model checkpoint not found: {model_p}")
            if self.calibration_artifact_path:
                art_p = Path(self.calibration_artifact_path).resolve()
                if not art_p.is_file():
                    raise FileNotFoundError(f"Calibration artifact not found: {art_p}")

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "product_id": self.product_id,
            "patchcore_model_path": self.patchcore_model_path,
            "threshold": float(self.threshold),
            "calibration_artifact_path": self.calibration_artifact_path,
            "calibration_method": self.calibration_method,
            "calibration_timestamp": self.calibration_timestamp,
            "image_size": list(self.image_size),
            "backbone": self.backbone,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProductProfile:
        """Construct profile from dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")
        for req in ["product_id", "patchcore_model_path", "threshold"]:
            if req not in data:
                raise ValueError(f"Missing required product profile field: '{req}'")

        image_size = tuple(data.get("image_size", (256, 256)))
        return cls(
            product_id=str(data["product_id"]),
            patchcore_model_path=str(data["patchcore_model_path"]),
            threshold=float(data["threshold"]),
            calibration_artifact_path=data.get("calibration_artifact_path"),
            calibration_method=str(data.get("calibration_method", "unknown")),
            calibration_timestamp=str(data.get("calibration_timestamp", "")),
            image_size=image_size,
            backbone=str(data.get("backbone", "wide_resnet50_2")),
            metadata=dict(data.get("metadata", {})),
        )

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        """Save product profile to a JSON file."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> ProductProfile:
        """Load product profile from a JSON file."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Product profile file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_calibration_artifact(
        cls,
        artifact_path: str | Path,
        patchcore_model_path: str | Path,
        product_id: str | None = None,
    ) -> ProductProfile:
        """Construct a ProductProfile directly from a Phase 3 ThresholdCalibrationResult JSON artifact."""
        art_p = Path(artifact_path).resolve()
        if not art_p.is_file():
            raise FileNotFoundError(f"Calibration artifact not found: {art_p}")

        with art_p.open("r", encoding="utf-8") as f:
            art_data = json.load(f)

        pid = product_id or art_data.get("product_name")
        if not pid:
            raise ValueError("product_id could not be determined from artifact or arguments.")

        threshold = float(art_data["threshold"])
        calib_method = art_data.get("calibration_method", "unknown")
        calib_ts = art_data.get("calibration_timestamp", "")
        calib_meta = art_data.get("calibration_metadata", {})
        backbone = calib_meta.get("patchcore_backbone", "wide_resnet50_2")

        return cls(
            product_id=str(pid),
            patchcore_model_path=str(patchcore_model_path),
            threshold=threshold,
            calibration_artifact_path=str(art_p),
            calibration_method=calib_method,
            calibration_timestamp=calib_ts,
            backbone=backbone,
            metadata={
                "is_normal_only": art_data.get("is_normal_only", False),
                "objective_metric_name": art_data.get("objective_metric_name"),
                "optimal_plateau": art_data.get("optimal_plateau"),
                "normal_sample_count": art_data.get("normal_sample_count"),
                "defective_sample_count": art_data.get("defective_sample_count"),
            },
        )
