"""SQLAlchemy ORM models for DefectVision.

Defines schemas for:
- Product (identity, status, lifecycle)
- ProductCalibration (PatchCore checkpoint reference, calibrated threshold, metadata)
- Inspection (persisted two-stage inspection result, anomaly score, stage 2 classification)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from defectvision.db.database import Base


class ProductStatus(str, Enum):
    """Product lifecycle state."""

    REGISTERED = "REGISTERED"
    CALIBRATING = "CALIBRATING"
    READY = "READY"


class CalibrationRecordStatus(str, Enum):
    """Product calibration status."""

    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class Product(Base):
    """Registered industrial product entity."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default=ProductStatus.REGISTERED.value, nullable=False
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    calibrations: Mapped[list[ProductCalibration]] = relationship(
        "ProductCalibration",
        back_populates="product",
        cascade="all, delete-orphan",
        order_by="desc(ProductCalibration.id)",
    )
    inspections: Mapped[list[Inspection]] = relationship(
        "Inspection",
        back_populates="product",
        cascade="all, delete-orphan",
        order_by="desc(Inspection.id)",
    )

    @property
    def active_calibration(self) -> ProductCalibration | None:
        """Return the latest active calibration for this product, if any."""
        for cal in self.calibrations:
            if cal.calibration_status == CalibrationRecordStatus.ACTIVE.value:
                return cal
        return None

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, product_id='{self.product_id}', status='{self.status}')>"


class ProductCalibration(Base):
    """Product-specific threshold calibration and PatchCore model reference."""

    __tablename__ = "product_calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.product_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    calibration_artifact_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    patchcore_model_path: Mapped[str] = mapped_column(String(512), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    calibration_method: Mapped[str] = mapped_column(String(64), nullable=False)
    calibration_timestamp: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reference_image_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    normal_sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    defective_sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_width: Mapped[int] = mapped_column(Integer, default=256, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, default=256, nullable=False)
    backbone: Mapped[str] = mapped_column(
        String(64), default="wide_resnet50_2", nullable=False
    )
    calibration_status: Mapped[str] = mapped_column(
        String(32), default=CalibrationRecordStatus.ACTIVE.value, nullable=False
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    product: Mapped[Product] = relationship("Product", back_populates="calibrations")

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    def __repr__(self) -> str:
        return (
            f"<ProductCalibration(id={self.id}, product_id='{self.product_id}', "
            f"threshold={self.threshold:.4f}, method='{self.calibration_method}', status='{self.calibration_status}')>"
        )


class Inspection(Base):
    """Persisted outcome of a two-stage inspection run."""

    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    product_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("products.product_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)  # NORMAL or DEFECTIVE
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    margin: Mapped[float] = mapped_column(Float, nullable=False)
    stage2_executed: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    defect_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    classifier_probabilities: Mapped[dict[str, float] | None] = mapped_column(
        JSON, nullable=True
    )
    heatmap_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    execution_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    product: Mapped[Product] = relationship("Product", back_populates="inspections")

    def __repr__(self) -> str:
        return (
            f"<Inspection(id={self.id}, inspection_id='{self.inspection_id}', "
            f"product_id='{self.product_id}', verdict='{self.verdict}', score={self.anomaly_score:.4f})>"
        )
