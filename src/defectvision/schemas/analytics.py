"""Pydantic schema for Inspection Analytics."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyticsResponse(BaseModel):
    product_id: str | None = Field(
        default=None,
        description="Filter product ID if scoped to a specific product, or None if global",
    )
    total_inspections: int = Field(
        ...,
        ge=0,
        description="Total number of inspections recorded",
    )
    normal_count: int = Field(
        ...,
        ge=0,
        description="Count of inspections with NORMAL verdict",
    )
    defective_count: int = Field(
        ...,
        ge=0,
        description="Count of inspections with DEFECTIVE verdict",
    )
    defect_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of defective inspections (defective / total)",
    )
    normal_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of normal inspections (normal / total)",
    )
    average_execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Average execution latency in milliseconds",
    )
    defect_class_counts: dict[str, int] = Field(
        ...,
        description="Breakdown of defect occurrences by class taxonomy (Scratch, Crack, Discoloration, UNKNOWN)",
    )
