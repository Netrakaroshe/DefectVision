"""Pydantic schemas for Product entity."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from defectvision.db.models import ProductStatus


class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Human-readable product name")
    description: str | None = Field(default=None, description="Optional product description or notes")
    metadata: dict[str, Any] | None = Field(default=None, description="Arbitrary product metadata")


class ProductCreate(ProductBase):
    product_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique product identifier (e.g., 'bottle', 'screw', 'pcb_board')",
    )


class ActiveCalibrationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    threshold: float
    calibration_method: str
    patchcore_model_path: str
    calibration_artifact_path: str | None = None
    image_width: int = 256
    image_height: int = 256
    backbone: str = "wide_resnet50_2"
    calibration_status: str


class ProductResponse(ProductBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    product_id: str
    status: ProductStatus | str
    created_at: datetime
    updated_at: datetime
    active_calibration: ActiveCalibrationSummary | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_orm_metadata(cls, data: Any) -> Any:
        if hasattr(data, "metadata_json"):
            return {
                "id": getattr(data, "id", None),
                "product_id": getattr(data, "product_id", None),
                "name": getattr(data, "name", None),
                "description": getattr(data, "description", None),
                "status": getattr(data, "status", None),
                "metadata": getattr(data, "metadata_json", None),
                "created_at": getattr(data, "created_at", None),
                "updated_at": getattr(data, "updated_at", None),
                "active_calibration": getattr(data, "active_calibration", None),
            }
        return data
