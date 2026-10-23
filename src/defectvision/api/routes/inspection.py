"""Inspection execution and retrieval routes."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from defectvision.api.dependencies import get_orchestrator, get_storage
from defectvision.api.storage import StorageManager
from defectvision.db.database import get_db
from defectvision.db.models import (
    CalibrationRecordStatus,
    Inspection,
    Product,
    ProductCalibration,
    ProductStatus,
)
from defectvision.inference.orchestrator import DefectVisionOrchestrator
from defectvision.inference.product_profile import ProductProfile
from defectvision.inference.result import InspectionResult
from defectvision.schemas.inspection import InspectionListResponse, InspectionResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inspection"])


@router.post(
    "/products/{product_id}/inspect",
    response_model=InspectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect an image using the two-stage inference pipeline",
)
async def inspect_product_image(
    product_id: str,
    request: Request,
    db: Session = Depends(get_db),
    orchestrator: DefectVisionOrchestrator = Depends(get_orchestrator),
    storage: StorageManager = Depends(get_storage),
) -> Inspection:
    """Run two-stage inspection for a product image and persist the result.

    Supports:
    - Multipart file upload: provide form file 'image' / 'file' and optional 'save_heatmap' bool.
    - JSON payload: provide 'image_path' and optional 'save_heatmap' bool.

    Pipeline rules:
    - Product must be in READY status with an active calibration.
    - ProductProfile is constructed from the active calibration as the single source of truth.
    - DefectVisionOrchestrator runs Stage 1 PatchCore. If NORMAL, Stage 2 is bypassed.
    - If DEFECTIVE, Stage 2 classifier executes to classify the defect.
    - The complete result is saved to the database.
    """
    product = db.execute(
        select(Product).where(Product.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found.",
        )

    if product.status != ProductStatus.READY.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Product '{product_id}' has status '{product.status}'. "
                "Product must be in READY status with a calibrated threshold before inspection."
            ),
        )

    active_cal = db.execute(
        select(ProductCalibration).where(
            ProductCalibration.product_id == product_id,
            ProductCalibration.calibration_status == CalibrationRecordStatus.ACTIVE.value,
        )
    ).scalar_one_or_none()

    if active_cal is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Product '{product_id}' has no active calibration record.",
        )

    # Parse input from direct image streaming, multipart form, or JSON
    content_type = request.headers.get("content-type", "")
    target_image_path: Path
    save_heatmap: bool = False

    if content_type.startswith("image/") or "octet-stream" in content_type:
        raw_bytes = await request.body()
        if len(raw_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image payload is empty.",
            )
        ext = ".png"
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "webp" in content_type:
            ext = ".webp"
        elif "bmp" in content_type:
            ext = ".bmp"
        elif "tiff" in content_type:
            ext = ".tiff"

        try:
            target_image_path = storage.save_uploaded_image(
                file_bytes=raw_bytes,
                original_filename=f"stream_upload{ext}",
                product_id=product_id,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

        save_heatmap = request.query_params.get("save_heatmap", "false").lower() in ("true", "1", "yes")

    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except (AssertionError, Exception) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Multipart form parsing failed: {e}. "
                    "For binary streaming, send Content-Type 'image/png' or JSON with 'image_path' / 'image_base64'."
                ),
            ) from e

        upload_field = form.get("image") or form.get("file")
        if upload_field is None or not hasattr(upload_field, "filename"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart request must include an 'image' or 'file' field.",
            )
        file_bytes = await upload_field.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        try:
            target_image_path = storage.save_uploaded_image(
                file_bytes=file_bytes,
                original_filename=upload_field.filename,
                product_id=product_id,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

        save_heatmap = str(form.get("save_heatmap") or request.query_params.get("save_heatmap", "false")).lower() in ("true", "1", "yes")

    else:
        # Expect JSON
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request body. Expected JSON, image streaming, or multipart/form-data.",
            ) from e

        save_heatmap = bool(body.get("save_heatmap", False))
        b64_data = body.get("image_base64")

        if b64_data:
            import base64
            try:
                img_bytes = base64.b64decode(b64_data)
                target_image_path = storage.save_uploaded_image(
                    file_bytes=img_bytes,
                    original_filename="base64_upload.png",
                    product_id=product_id,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Base64 image decode failed: {e}",
                ) from e
        else:
            raw_path = body.get("image_path")
            if not raw_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing 'image_path' or 'image_base64' in request body.",
                )

            try:
                target_image_path = storage.validate_image_file(raw_path)
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                ) from e

    # Construct authoritative ProductProfile
    profile = ProductProfile(
        product_id=product.product_id,
        patchcore_model_path=active_cal.patchcore_model_path,
        threshold=active_cal.threshold,
        calibration_artifact_path=active_cal.calibration_artifact_path,
        calibration_method=active_cal.calibration_method,
        calibration_timestamp=active_cal.calibration_timestamp or "",
        image_size=(active_cal.image_width, active_cal.image_height),
        backbone=active_cal.backbone,
        metadata=dict(active_cal.metadata_json or {}),
    )

    heatmap_dir = storage.heatmaps_dir / product_id if save_heatmap else None

    # Execute two-stage inference orchestrator
    try:
        inspection_result: InspectionResult = orchestrator.inspect(
            image_path=target_image_path,
            product_profile=profile,
            save_heatmap=save_heatmap,
            heatmap_dir=heatmap_dir,
        )
    except Exception as e:
        logger.exception(f"Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference pipeline execution failed.",
        ) from e

    # Parse timestamp
    ts_val: datetime
    try:
        ts_val = datetime.fromisoformat(inspection_result.timestamp)
    except Exception:
        ts_val = datetime.now(timezone.utc)

    # Persist inspection in database
    db_inspection = Inspection(
        inspection_id=inspection_result.inspection_id,
        product_id=product.product_id,
        image_path=inspection_result.image_path,
        timestamp=ts_val,
        verdict=inspection_result.verdict,
        anomaly_score=inspection_result.anomaly_score,
        anomaly_threshold=inspection_result.anomaly_threshold,
        margin=inspection_result.margin,
        stage2_executed=inspection_result.stage2_executed,
        defect_class=inspection_result.defect_class,
        classifier_confidence=inspection_result.classifier_confidence,
        classifier_probabilities=inspection_result.classifier_probabilities,
        heatmap_path=inspection_result.heatmap_path,
        execution_time_ms=inspection_result.execution_time_ms,
        metadata_json=inspection_result.metadata,
    )
    db.add(db_inspection)
    db.commit()
    db.refresh(db_inspection)

    logger.info(
        f"Persisted inspection {db_inspection.inspection_id} for '{product_id}': "
        f"verdict={db_inspection.verdict} (score={db_inspection.anomaly_score:.4f}, "
        f"stage2={db_inspection.stage2_executed}, defect={db_inspection.defect_class})"
    )
    return db_inspection


@router.get(
    "/products/{product_id}/inspections",
    response_model=InspectionListResponse,
    summary="Get inspection history for a product",
)
def list_product_inspections(
    product_id: str,
    verdict: str | None = Query(default=None, description="Filter by verdict: NORMAL or DEFECTIVE"),
    limit: int = Query(default=50, ge=1, le=500, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve paginated inspection history for a specific product."""
    product = db.execute(
        select(Product).where(Product.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found.",
        )

    stmt = select(Inspection).where(Inspection.product_id == product_id)
    count_stmt = select(func.count(Inspection.id)).where(Inspection.product_id == product_id)

    if verdict:
        stmt = stmt.where(Inspection.verdict == verdict.upper())
        count_stmt = count_stmt.where(Inspection.verdict == verdict.upper())

    total = db.execute(count_stmt).scalar_one()

    stmt = stmt.order_by(Inspection.created_at.desc()).offset(offset).limit(limit)
    items = db.execute(stmt).scalars().all()

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/inspections/{inspection_id}",
    response_model=InspectionResponse,
    summary="Get inspection result by ID",
)
def get_inspection(
    inspection_id: str,
    db: Session = Depends(get_db),
) -> Inspection:
    """Retrieve a single inspection record by inspection_id (UUID) or database primary key."""
    stmt = select(Inspection).where(Inspection.inspection_id == inspection_id)
    inspection = db.execute(stmt).scalar_one_or_none()

    if inspection is None and inspection_id.isdigit():
        stmt = select(Inspection).where(Inspection.id == int(inspection_id))
        inspection = db.execute(stmt).scalar_one_or_none()

    if inspection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    return inspection


def _resolve_and_validate_media_file(
    raw_path: str | None,
    file_type: str,
    storage_root: Path,
) -> Path:
    """Validate that the file path exists, is within storage root, and is accessible."""
    if not raw_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection has no associated {file_type} file.",
        )

    target_path = Path(raw_path).resolve()
    resolved_root = storage_root.resolve()

    try:
        target_path.relative_to(resolved_root)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: {file_type} path is outside application storage.",
        )

    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{file_type.capitalize()} file not found on disk.",
        )

    return target_path


@router.get(
    "/inspections/{inspection_id}/image",
    summary="Retrieve inspection query image",
    response_class=FileResponse,
)
def get_inspection_image(
    inspection_id: str,
    db: Session = Depends(get_db),
    storage: StorageManager = Depends(get_storage),
) -> FileResponse:
    """Retrieve the original uploaded query image for an inspection."""
    stmt = select(Inspection).where(Inspection.inspection_id == inspection_id)
    inspection = db.execute(stmt).scalar_one_or_none()

    if inspection is None and inspection_id.isdigit():
        stmt = select(Inspection).where(Inspection.id == int(inspection_id))
        inspection = db.execute(stmt).scalar_one_or_none()

    if inspection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    file_path = _resolve_and_validate_media_file(
        raw_path=inspection.image_path,
        file_type="image",
        storage_root=storage.root,
    )

    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return FileResponse(path=str(file_path), media_type=media_type)


@router.get(
    "/inspections/{inspection_id}/heatmap",
    summary="Retrieve inspection localization heatmap",
    response_class=FileResponse,
)
def get_inspection_heatmap(
    inspection_id: str,
    db: Session = Depends(get_db),
    storage: StorageManager = Depends(get_storage),
) -> FileResponse:
    """Retrieve the anomaly localization heatmap for an inspection."""
    stmt = select(Inspection).where(Inspection.inspection_id == inspection_id)
    inspection = db.execute(stmt).scalar_one_or_none()

    if inspection is None and inspection_id.isdigit():
        stmt = select(Inspection).where(Inspection.id == int(inspection_id))
        inspection = db.execute(stmt).scalar_one_or_none()

    if inspection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection '{inspection_id}' not found.",
        )

    file_path = _resolve_and_validate_media_file(
        raw_path=inspection.heatmap_path,
        file_type="heatmap",
        storage_root=storage.root,
    )

    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return FileResponse(path=str(file_path), media_type=media_type)
