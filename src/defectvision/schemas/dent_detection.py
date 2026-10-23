"""Pydantic schemas for Experimental Dent Detection operations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DentBoxSchema(BaseModel):
    x1: float = Field(..., description="Left bounding box coordinate")
    y1: float = Field(..., description="Top bounding box coordinate")
    x2: float = Field(..., description="Right bounding box coordinate")
    y2: float = Field(..., description="Bottom bounding box coordinate")
    confidence: float = Field(..., description="Detection confidence score between 0 and 1")
    class_id: int = Field(default=2, description="Class ID (2 for Dents)")
    class_name: str = Field(default="Dents", description="Class name")


class DentDetectionResponse(BaseModel):
    dents_detected: int = Field(..., description="Total number of Dents detected above the threshold")
    boxes: list[DentBoxSchema] = Field(default_factory=list, description="List of detected Dent bounding boxes")
    confidence_threshold: float = Field(..., description="Applied confidence threshold filter")
    image_width: int = Field(..., description="Input image width in pixels")
    image_height: int = Field(..., description="Input image height in pixels")
    execution_time_ms: float = Field(..., description="Inference latency in milliseconds")
    model_name: str = Field(default="Faster R-CNN MobileNetV3-Large FPN", description="Model architecture")
    dataset: str = Field(default="CR7-DET (Steel Surface)", description="Validation dataset")
    experimental: bool = Field(default=True, description="Flag indicating experimental status")
    message: str = Field(default="", description="Human-readable result summary")
