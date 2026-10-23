"""Pydantic schemas for Product Calibration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CalibrationCreate(BaseModel):
    patchcore_model_path: str = Field(
        ...,
        min_length=1,
        description="Path to the fitted PatchCore checkpoint or model directory",
    )
    artifact_path: str | None = Field(
        default=None,
        description="Path to an existing ThresholdCalibrationResult JSON artifact file",
    )
    threshold: float | None = Field(
        default=None,
        description="Explicit calibrated anomaly threshold (required if neither artifact nor scores are provided)",
    )
    normal_scores: list[float] | None = Field(
        default=None,
        description="Continuous PatchCore anomaly scores on normal reference/calibration samples",
    )
    defective_scores: list[float] | None = Field(
        default=None,
        description="Optional continuous PatchCore anomaly scores on known defective samples",
    )
    calibration_method: str | None = Field(
        default="auto",
        description="Calibration algorithm: auto, youden_j, balanced_accuracy, normal_k_sigma, etc.",
    )
    image_size: tuple[int, int] | list[int] = Field(
        default=(256, 256),
        description="Target image resolution (height, width) for Stage 1 preprocessing",
    )
    backbone: str = Field(
        default="wide_resnet50_2",
        description="Feature extractor backbone used for PatchCore",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional calibration metadata or provenance information",
    )


class CalibrationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: str
    calibration_artifact_path: str | None = None
    patchcore_model_path: str
    threshold: float
    calibration_method: str
    calibration_timestamp: str | None = None
    reference_image_count: int | None = None
    normal_sample_count: int | None = None
    defective_sample_count: int | None = None
    image_width: int
    image_height: int
    backbone: str
    calibration_status: str
    metadata: dict[str, Any] | None = None
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def resolve_orm_metadata(cls, data: Any) -> Any:
        if hasattr(data, "metadata_json"):
            return {
                "id": getattr(data, "id", None),
                "product_id": getattr(data, "product_id", None),
                "calibration_artifact_path": getattr(data, "calibration_artifact_path", None),
                "patchcore_model_path": getattr(data, "patchcore_model_path", None),
                "threshold": getattr(data, "threshold", None),
                "calibration_method": getattr(data, "calibration_method", None),
                "calibration_timestamp": getattr(data, "calibration_timestamp", None),
                "reference_image_count": getattr(data, "reference_image_count", None),
                "normal_sample_count": getattr(data, "normal_sample_count", None),
                "defective_sample_count": getattr(data, "defective_sample_count", None),
                "image_width": getattr(data, "image_width", 256),
                "image_height": getattr(data, "image_height", 256),
                "backbone": getattr(data, "backbone", "wide_resnet50_2"),
                "calibration_status": getattr(data, "calibration_status", "ACTIVE"),
                "metadata": getattr(data, "metadata_json", None),
                "created_at": getattr(data, "created_at", None),
            }
        return data
