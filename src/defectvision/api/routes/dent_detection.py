"""Experimental Dent Detection API routes.

Provides:
- POST /experimental/dent/detect: Run Faster R-CNN dent detection on an image.
- GET /experimental/dent/info: Metadata about model, dataset, and threshold.

Completely isolated from the production two-stage PatchCore -> ResNet18 pipeline.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from PIL import Image

from defectvision.models.dent_detector import get_dent_detector
from defectvision.schemas.dent_detection import DentBoxSchema, DentDetectionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experimental/dent", tags=["Experimental Dent Detection"])


@router.get(
    "/info",
    summary="Get experimental Dent detector metadata",
)
def get_dent_detector_info() -> dict:
    """Return model architecture and validation dataset information."""
    return {
        "status": "active",
        "experimental": True,
        "model_architecture": "Faster R-CNN MobileNetV3-Large FPN",
        "target_class": "Dents (Class ID 2)",
        "default_confidence_threshold": 0.30,
        "validation_dataset": "CR7-DET (Steel Surface)",
        "limitation": (
            "Validated on CR7-DET steel surfaces only. Does not replace or alter the "
            "production PatchCore -> ResNet-18 inspection pipeline."
        ),
    }


@router.post(
    "/detect",
    response_model=DentDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Dents on steel surface images",
)
async def detect_dents(
    request: Request,
    confidence_threshold: float = Query(
        default=0.30,
        ge=0.01,
        le=1.0,
        description="Minimum confidence score threshold for reported Dents",
    ),
) -> DentDetectionResponse:
    """Run dedicated Faster R-CNN Dent detection on an image.

    Supports:
    - Multipart file upload: provide form field 'image' or 'file'.
    - Direct binary image streaming: Content-Type 'image/*' or 'application/octet-stream'.
    - JSON payload: provide 'image_path' to an on-disk image file.

    Returns:
    - Bounding boxes and confidence scores for detected Dents.
    - Zero boxes and a clear message if no Dents meet the threshold.
    """
    content_type = request.headers.get("content-type", "")
    pil_image: Image.Image | None = None

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Multipart form parsing failed: {e}",
            ) from e

        # Extract confidence_threshold from form if provided
        form_thresh = form.get("confidence_threshold")
        if form_thresh is not None:
            try:
                confidence_threshold = float(form_thresh)
            except ValueError:
                pass

        upload_field = form.get("image") or form.get("file")
        if upload_field is None or not hasattr(upload_field, "filename"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart form must contain an 'image' or 'file' field.",
            )

        file_bytes = await upload_field.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty.",
            )

        try:
            pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode image bytes: {e}",
            ) from e

    elif content_type.startswith("image/") or "octet-stream" in content_type:
        raw_bytes = await request.body()
        if len(raw_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Direct binary image payload is empty.",
            )
        try:
            pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode binary image: {e}",
            ) from e

    elif "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON body: {e}",
            ) from e

        image_path_str = body.get("image_path")
        if not image_path_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON body must specify 'image_path'.",
            )

        thresh_val = body.get("confidence_threshold")
        if thresh_val is not None:
            try:
                confidence_threshold = float(thresh_val)
            except ValueError:
                pass

        image_path = Path(image_path_str)
        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image file not found on disk: {image_path_str}",
            )

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read image file from disk: {e}",
            ) from e

    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported Content-Type: '{content_type}'. "
                "Use 'multipart/form-data', direct binary 'image/*', or 'application/json' with 'image_path'."
            ),
        )

    try:
        detector = get_dent_detector()
        result = detector.detect(pil_image, confidence_threshold=confidence_threshold)
    except Exception as e:
        logger.exception("Dent detection inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dent detection inference failed: {e}",
        ) from e

    return DentDetectionResponse(
        dents_detected=result.dents_detected,
        boxes=[DentBoxSchema(**b.to_dict()) for b in result.boxes],
        confidence_threshold=result.confidence_threshold,
        image_width=result.image_width,
        image_height=result.image_height,
        execution_time_ms=result.execution_time_ms,
        model_name=result.model_name,
        dataset=result.dataset,
        experimental=result.experimental,
        message=result.message,
    )
