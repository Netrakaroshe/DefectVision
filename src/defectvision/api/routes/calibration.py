"""Product calibration routes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from defectvision.api.dependencies import get_storage
from defectvision.api.storage import ALLOWED_IMAGE_EXTENSIONS, StorageManager
from defectvision.calibration.threshold_calibrator import (
    CalibrationMethod,
    ThresholdCalibrationResult,
    calibrate_threshold,
)
from defectvision.db.database import get_db
from defectvision.db.models import (
    CalibrationRecordStatus,
    Product,
    ProductCalibration,
    ProductStatus,
    utc_now,
)
from defectvision.models.patchcore_engine import PatchCorePipeline
from defectvision.schemas.calibration import CalibrationCreate, CalibrationResponse

logger = logging.getLogger(__name__)

_MIN_REFERENCE_IMAGES = 10
_MAX_REFERENCE_IMAGES = 20
_DEFAULT_BACKBONE = "wide_resnet50_2"
_DEFAULT_IMAGE_SIZE = (256, 256)


class RunCalibrationResponse(BaseModel):
    """Response from the dynamic product calibration run endpoint."""

    product_id: str
    calibration_id: int
    threshold: float
    calibration_method: str
    reference_image_count: int
    model_path: str
    artifact_path: str
    normal_score_stats: dict
    warnings: list[str]
    product_status: str

router = APIRouter(prefix="/products/{product_id}/calibration", tags=["Calibration"])


@router.post(
    "/run",
    response_model=RunCalibrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Dynamic product calibration from uploaded reference images",
)
async def run_product_calibration(
    product_id: str,
    files: Annotated[list[UploadFile], File(description="10–50 defect-free reference images")],
    db: Session = Depends(get_db),
    storage: StorageManager = Depends(get_storage),
) -> RunCalibrationResponse:
    """Perform full dynamic PatchCore calibration for a product from uploaded reference images.

    Workflow:
    1. Validate uploaded files (image format, count).
    2. Save reference images to ``data/app_storage/products/{product_id}/reference_images/``.
    3. Fit a product-specific PatchCore memory bank via ``build_feature_memory()``.
    4. Save the fitted checkpoint to ``data/app_storage/products/{product_id}/models/``.
    5. Infer anomaly scores on the reference images (normal-only).
    6. Compute threshold via ``NORMAL_K_SIGMA`` (mean + 3σ).
    7. Persist the calibration artifact to ``data/app_storage/products/{product_id}/artifacts/``.
    8. Create/supersede the ``ProductCalibration`` DB record and set ``Product.status = READY``.

    This endpoint does NOT touch any Phase 2/3 experiment artifacts, manifests, or the
    validated bottle checkpoint.
    """
    # -- Guard: product must exist --
    product = db.execute(
        select(Product).where(Product.product_id == product_id)
    ).scalar_one_or_none()
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found.",
        )

    # -- Guard: file count --
    if len(files) < _MIN_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"At least {_MIN_REFERENCE_IMAGES} reference images are required for calibration; "
                f"got {len(files)}."
            ),
        )
    if len(files) > _MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"At most {_MAX_REFERENCE_IMAGES} reference images are accepted; "
                f"got {len(files)}."
            ),
        )

    # -- Guard: file extensions --
    for uf in files:
        ext = Path(uf.filename or "").suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unsupported image format '{ext}' in file '{uf.filename}'. "
                    f"Supported: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
                ),
            )

    # -- Set up product-specific storage directories --
    product_dir = storage.root / "products" / product_id
    ref_images_dir = product_dir / "reference_images"
    models_dir = product_dir / "models"
    artifacts_dir = product_dir / "artifacts"

    for d in (ref_images_dir, models_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -- Save uploaded reference images --
    saved_image_paths: list[Path] = []
    for uf in files:
        raw_bytes = await uf.read()
        safe_name = f"{len(saved_image_paths):04d}_{Path(uf.filename or 'image').name}"
        dest = ref_images_dir / safe_name
        dest.write_bytes(raw_bytes)
        try:
            storage.validate_image_file(dest)
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Image validation failed for '{uf.filename}': {e}",
            ) from e
        saved_image_paths.append(dest)

    logger.info(
        f"[{product_id}] Saved {len(saved_image_paths)} reference images to {ref_images_dir}"
    )

    # -- Fit PatchCore memory bank --
    # Use the product storage directory as the persistent work_dir so that Anomalib
    # writes its checkpoint under products/{product_id}/models/engine_runs/...
    pipeline = PatchCorePipeline(
        backbone=_DEFAULT_BACKBONE,
        image_size=_DEFAULT_IMAGE_SIZE,
        # Use CPU to avoid GPU availability assumptions in a web-server context.
        # For production with a dedicated GPU, change to accelerator="gpu".
        accelerator="cpu",
        devices=1,
    )

    engine_work_dir = models_dir / "engine_runs"
    try:
        mem_stats = pipeline.build_feature_memory(
            reference_images=saved_image_paths,
            product_name=product_id,
            work_dir=engine_work_dir,
        )
    except Exception as e:
        logger.error(f"[{product_id}] PatchCore fitting failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PatchCore fitting failed: {e}",
        ) from e

    logger.info(
        f"[{product_id}] PatchCore memory bank fitted. Stats: {mem_stats}"
    )

    # -- Locate the checkpoint produced by Anomalib Engine.fit() --
    # Anomalib saves the best checkpoint under:
    #   <default_root_dir>/<model_name>/<dataset_name>/version_*/checkpoints/epoch=*.ckpt
    ckpt_candidates = sorted(engine_work_dir.rglob("*.ckpt"))
    if not ckpt_candidates:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PatchCore fitting completed but no checkpoint file was produced.",
        )
    fitted_ckpt = ckpt_candidates[-1]  # latest checkpoint

    # Copy the checkpoint to the canonical model path
    canonical_ckpt = models_dir / f"{product_id}_patchcore.ckpt"
    shutil.copy2(fitted_ckpt, canonical_ckpt)
    logger.info(f"[{product_id}] Checkpoint saved to {canonical_ckpt}")

    # -- Reload checkpoint to verify provenance and decoupling from fitting state --
    try:
        pipeline.load_checkpoint(canonical_ckpt)
        logger.info(f"[{product_id}] Reloaded fitted checkpoint from {canonical_ckpt}")
    except Exception as e:
        logger.error(f"[{product_id}] Failed to reload checkpoint {canonical_ckpt}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fitted checkpoint reload failed: {e}",
        ) from e

    # -- Infer anomaly scores on reference images (normal-only calibration) --
    try:
        infer_results = pipeline.infer_batch(
            image_paths=saved_image_paths,
            image_size=_DEFAULT_IMAGE_SIZE,
        )
    except Exception as e:
        logger.error(f"[{product_id}] Reference image inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reference image inference failed: {e}",
        ) from e

    normal_scores = [r.anomaly_score for r in infer_results]
    logger.info(
        f"[{product_id}] Extracted {len(normal_scores)} reference anomaly scores. "
        f"Range: [{min(normal_scores):.4f}, {max(normal_scores):.4f}]"
    )

    # -- Calibrate threshold using NORMAL_K_SIGMA (mean + 3.0 * std) --
    try:
        calib_result = calibrate_threshold(
            normal_scores=normal_scores,
            defective_scores=None,
            product_name=product_id,
            method=CalibrationMethod.NORMAL_K_SIGMA,
            k_sigma=3.0,
            calibration_metadata={
                "backbone": _DEFAULT_BACKBONE,
                "image_size": list(_DEFAULT_IMAGE_SIZE),
                "reference_image_count": len(saved_image_paths),
                "patchcore_memory_bank_shape": mem_stats.get("memory_bank_shape"),
            },
        )
    except Exception as e:
        logger.error(f"[{product_id}] Threshold calibration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threshold calibration failed: {e}",
        ) from e

    # -- Persist the calibration artifact JSON --
    artifact_path = artifacts_dir / f"{product_id}_calibration_artifact.json"
    calib_result.save_json(artifact_path)
    logger.info(
        f"[{product_id}] Calibration artifact saved to {artifact_path}. "
        f"Threshold={calib_result.threshold:.4f}"
    )

    # -- Supersede any existing active calibration records --
    db.execute(
        update(ProductCalibration)
        .where(
            ProductCalibration.product_id == product_id,
            ProductCalibration.calibration_status == CalibrationRecordStatus.ACTIVE.value,
        )
        .values(calibration_status=CalibrationRecordStatus.SUPERSEDED.value)
    )

    # -- Create the new active calibration DB record --
    calibration = ProductCalibration(
        product_id=product_id,
        calibration_artifact_path=str(artifact_path).replace("\\", "/"),
        patchcore_model_path=str(canonical_ckpt).replace("\\", "/"),
        threshold=calib_result.threshold,
        calibration_method=calib_result.calibration_method,
        calibration_timestamp=calib_result.calibration_timestamp,
        reference_image_count=len(saved_image_paths),
        normal_sample_count=len(normal_scores),
        defective_sample_count=0,
        image_width=_DEFAULT_IMAGE_SIZE[1],
        image_height=_DEFAULT_IMAGE_SIZE[0],
        backbone=_DEFAULT_BACKBONE,
        calibration_status=CalibrationRecordStatus.ACTIVE.value,
        metadata_json={
            "is_normal_only": True,
            "k_sigma": 3.0,
            "memory_bank_shape": mem_stats.get("memory_bank_shape"),
            "warnings": calib_result.warnings,
        },
    )
    db.add(calibration)

    # -- Transition product to READY --
    product.status = ProductStatus.READY.value
    product.updated_at = utc_now()

    db.commit()
    db.refresh(calibration)

    logger.info(
        f"[{product_id}] Dynamic calibration complete. "
        f"threshold={calib_result.threshold:.4f}, status=READY"
    )

    return RunCalibrationResponse(
        product_id=product_id,
        calibration_id=calibration.id,
        threshold=calib_result.threshold,
        calibration_method=calib_result.calibration_method,
        reference_image_count=len(saved_image_paths),
        model_path=str(canonical_ckpt).replace("\\", "/"),
        artifact_path=str(artifact_path).replace("\\", "/"),
        normal_score_stats=calib_result.normal_score_stats,
        warnings=calib_result.warnings,
        product_status=ProductStatus.READY.value,
    )


@router.post(
    "",
    response_model=CalibrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Calibrate or register threshold calibration for a product",
)
def calibrate_product(
    product_id: str,
    payload: CalibrationCreate,
    db: Session = Depends(get_db),
    storage: StorageManager = Depends(get_storage),
) -> ProductCalibration:
    """Register product calibration and transition product to READY.

    Can be triggered via:
    1. Pre-computed calibration artifact: supply `artifact_path` and `patchcore_model_path`.
    2. Empirical score calibration: supply `normal_scores` (and optional `defective_scores`)
       to run the validated Youden's J / k-sigma calibration algorithm.
    3. Explicit validated threshold: supply `threshold` and `patchcore_model_path`.

    Upon successful calibration, any prior active calibration is superseded, and the
    product lifecycle status transitions to READY.
    """
    product = db.execute(
        select(Product).where(Product.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found.",
        )

    # 1. Validate PatchCore checkpoint path
    model_path = Path(payload.patchcore_model_path).resolve()
    if not model_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PatchCore model checkpoint not found: {payload.patchcore_model_path}",
        )

    threshold: float | None = None
    method: str = payload.calibration_method or "manual"
    artifact_path_str: str | None = None
    calib_timestamp: str | None = None
    normal_count: int | None = None
    defective_count: int | None = None
    meta: dict = dict(payload.metadata or {})

    # Path A: Load from existing calibration artifact JSON
    if payload.artifact_path:
        art_p = Path(payload.artifact_path).resolve()
        if not art_p.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Calibration artifact not found: {payload.artifact_path}",
            )
        try:
            art_result = ThresholdCalibrationResult.load_json(art_p)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse calibration artifact: {e}",
            ) from e

        threshold = art_result.threshold
        method = art_result.calibration_method
        calib_timestamp = art_result.calibration_timestamp
        normal_count = art_result.normal_sample_count
        defective_count = art_result.defective_sample_count
        artifact_path_str = str(art_p).replace("\\", "/")
        meta["metrics"] = art_result.metrics
        meta["optimal_plateau"] = art_result.optimal_plateau
        meta["objective_metric_name"] = art_result.objective_metric_name

    # Path B: Compute threshold dynamically from empirical anomaly scores
    elif payload.normal_scores and len(payload.normal_scores) > 0:
        try:
            calib_res = calibrate_threshold(
                normal_scores=payload.normal_scores,
                defective_scores=payload.defective_scores,
                product_name=product_id,
                method=payload.calibration_method or "auto",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Threshold calibration computation failed: {e}",
            ) from e

        threshold = calib_res.threshold
        method = calib_res.calibration_method
        calib_timestamp = calib_res.calibration_timestamp
        normal_count = calib_res.normal_sample_count
        defective_count = calib_res.defective_sample_count
        meta["metrics"] = calib_res.metrics
        meta["optimal_plateau"] = calib_res.optimal_plateau
        meta["objective_metric_name"] = calib_res.objective_metric_name

        # Persist generated artifact into app storage
        saved_art_p = storage.calibrations_dir / f"{product_id}_calibration_artifact.json"
        calib_res.save_json(saved_art_p)
        artifact_path_str = str(saved_art_p).replace("\\", "/")

    # Path C: Explicit numeric threshold supplied
    elif payload.threshold is not None:
        if payload.threshold <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Threshold must be positive, got {payload.threshold}",
            )
        threshold = float(payload.threshold)
        calib_timestamp = utc_now().isoformat()

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Must provide either 'artifact_path', 'normal_scores', or a positive 'threshold' "
                "along with 'patchcore_model_path'."
            ),
        )

    # Resolve image dimensions
    img_size = payload.image_size
    img_w = int(img_size[1] if len(img_size) > 1 else img_size[0])
    img_h = int(img_size[0])

    # Supersede existing active calibrations for this product
    db.execute(
        update(ProductCalibration)
        .where(
            ProductCalibration.product_id == product_id,
            ProductCalibration.calibration_status == CalibrationRecordStatus.ACTIVE.value,
        )
        .values(calibration_status=CalibrationRecordStatus.SUPERSEDED.value)
    )

    # Create new active calibration record
    calibration = ProductCalibration(
        product_id=product_id,
        calibration_artifact_path=artifact_path_str,
        patchcore_model_path=str(model_path).replace("\\", "/"),
        threshold=threshold,
        calibration_method=method,
        calibration_timestamp=calib_timestamp,
        reference_image_count=normal_count,
        normal_sample_count=normal_count,
        defective_sample_count=defective_count,
        image_width=img_w,
        image_height=img_h,
        backbone=payload.backbone,
        calibration_status=CalibrationRecordStatus.ACTIVE.value,
        metadata_json=meta,
    )
    db.add(calibration)

    # Transition product status to READY
    product.status = ProductStatus.READY.value
    product.updated_at = utc_now()

    db.commit()
    db.refresh(calibration)

    logger.info(
        f"Calibrated product '{product_id}' with threshold={threshold:.4f} "
        f"via {method}. Product is now READY."
    )
    return calibration


@router.get(
    "",
    response_model=CalibrationResponse,
    summary="Get active calibration for a product",
)
def get_product_calibration(
    product_id: str,
    db: Session = Depends(get_db),
) -> ProductCalibration:
    """Retrieve the currently active threshold calibration for a product."""
    product = db.execute(
        select(Product).where(Product.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found.",
        )

    active_cal = db.execute(
        select(ProductCalibration).where(
            ProductCalibration.product_id == product_id,
            ProductCalibration.calibration_status == CalibrationRecordStatus.ACTIVE.value,
        )
    ).scalar_one_or_none()

    if active_cal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' has no active calibration.",
        )

    return active_cal
