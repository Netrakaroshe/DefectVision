"""Inspection analytics and summary routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from defectvision.db.database import get_db
from defectvision.db.models import Inspection, Product
from defectvision.schemas.analytics import AnalyticsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "",
    response_model=AnalyticsResponse,
    summary="Get aggregated inspection analytics",
)
def get_inspection_analytics(
    product_id: str | None = Query(default=None, description="Optional product ID filter"),
    db: Session = Depends(get_db),
) -> AnalyticsResponse:
    """Retrieve database-aggregated inspection statistics.

    Supports optional product-level scoping via `product_id`. Returns zero-valued
    metrics gracefully when no records exist.
    """
    if product_id is not None:
        product = db.execute(
            select(Product).where(Product.product_id == product_id)
        ).scalar_one_or_none()

        if product is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product '{product_id}' not found.",
            )

    stmt = select(
        func.count(Inspection.id).label("total"),
        func.count(case((Inspection.verdict == "NORMAL", 1))).label("normal_count"),
        func.count(case((Inspection.verdict == "DEFECTIVE", 1))).label("defective_count"),
        func.avg(Inspection.execution_time_ms).label("avg_latency"),
        func.count(case((Inspection.defect_class == "Scratch", 1))).label("scratch_count"),
        func.count(case((Inspection.defect_class == "Crack", 1))).label("crack_count"),
        func.count(case((Inspection.defect_class == "Discoloration", 1))).label("discoloration_count"),
        func.count(case((Inspection.defect_class == "UNKNOWN", 1))).label("unknown_count"),
    )

    if product_id is not None:
        stmt = stmt.where(Inspection.product_id == product_id)

    row = db.execute(stmt).one()

    total = row.total or 0
    normal_count = row.normal_count or 0
    defective_count = row.defective_count or 0

    defect_rate = float(defective_count / total) if total > 0 else 0.0
    normal_rate = float(normal_count / total) if total > 0 else 0.0
    avg_latency = float(row.avg_latency) if row.avg_latency is not None else 0.0

    defect_class_counts = {
        "Scratch": row.scratch_count or 0,
        "Crack": row.crack_count or 0,
        "Discoloration": row.discoloration_count or 0,
        "UNKNOWN": row.unknown_count or 0,
    }

    return AnalyticsResponse(
        product_id=product_id,
        total_inspections=total,
        normal_count=normal_count,
        defective_count=defective_count,
        defect_rate=round(defect_rate, 4),
        normal_rate=round(normal_rate, 4),
        average_execution_time_ms=round(avg_latency, 2),
        defect_class_counts=defect_class_counts,
    )
