"""Pydantic schemas for Inspection operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InspectRequest(BaseModel):
    image_path: str | None = Field(
        default=None,
        description="Path to an existing on-disk image file to inspect (alternative to multipart file upload)",
    )
    save_heatmap: bool = Field(
        default=False,
        description="Whether to generate and save localization heatmap visualization",
    )


class InspectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    inspection_id: str
    product_id: str
    image_path: str
    timestamp: datetime
    verdict: str  # NORMAL or DEFECTIVE
    anomaly_score: float
    anomaly_threshold: float
    margin: float
    stage2_executed: bool
    defect_class: str | None = None
    classifier_confidence: float | None = None
    classifier_probabilities: dict[str, float] | None = None
    heatmap_path: str | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def resolve_orm_metadata(cls, data: Any) -> Any:
        if hasattr(data, "metadata_json"):
            return {
                "id": getattr(data, "id", None),
                "inspection_id": getattr(data, "inspection_id", None),
                "product_id": getattr(data, "product_id", None),
                "image_path": getattr(data, "image_path", None),
                "timestamp": getattr(data, "timestamp", None),
                "verdict": getattr(data, "verdict", None),
                "anomaly_score": getattr(data, "anomaly_score", None),
                "anomaly_threshold": getattr(data, "anomaly_threshold", None),
                "margin": getattr(data, "margin", None),
                "stage2_executed": getattr(data, "stage2_executed", False),
                "defect_class": getattr(data, "defect_class", None),
                "classifier_confidence": getattr(data, "classifier_confidence", None),
                "classifier_probabilities": getattr(data, "classifier_probabilities", None),
                "heatmap_path": getattr(data, "heatmap_path", None),
                "execution_time_ms": getattr(data, "execution_time_ms", None),
                "metadata": getattr(data, "metadata_json", None),
                "created_at": getattr(data, "created_at", None),
            }
        return data


class InspectionListResponse(BaseModel):
    items: list[InspectionResponse]
    total: int
    limit: int
    offset: int
