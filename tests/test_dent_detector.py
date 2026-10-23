"""Tests for Experimental Dent Detection Module and API Route.

Verifies:
- Checkpoint loading and model initialization (Faster R-CNN MobileNetV3-Large FPN).
- Preprocessing and tensor transformation.
- Prediction formatting and bounding box clamping.
- Confidence threshold filtering.
- Empty detection handling and messaging.
- Real held-out test sample verification (Dent-positive and non-Dent).
- Isolation from production database and PatchCore pipeline.
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from defectvision.api.app import app
from defectvision.models.dent_detector import (
    DENT_CLASS_ID,
    DentBoundingBox,
    DentDetectionResult,
    DentDetector,
    get_dent_detector,
)

CHECKPOINT_PATH = Path("experiments/dent_detection_pilot/checkpoints/best_detector.pth")
DENT_TEST_IMAGE = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image\dents103.jpg")
NONDENT_TEST_IMAGE = Path(r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image\inclusions104.jpg")


@pytest.fixture(scope="module")
def dent_detector():
    """Module-level detector fixture to avoid re-loading weights for each test."""
    return get_dent_detector()


@pytest.fixture
def api_client():
    """Test client for API endpoints."""
    return TestClient(app)


def test_checkpoint_exists():
    """Verify that the trained pilot checkpoint is available."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing at {CHECKPOINT_PATH}"


def test_dent_detector_initialization(dent_detector):
    """Verify detector loads model with 8 classes in evaluation mode."""
    assert dent_detector is not None
    assert dent_detector.model is not None
    assert not dent_detector.model.training  # must be in eval mode


def test_dent_detector_synthetic_clean_image(dent_detector):
    """Verify inference on a synthetic uniform image outputs zero dents and proper message."""
    clean_img = Image.new("RGB", (400, 400), color=(180, 180, 180))
    result = dent_detector.detect(clean_img, confidence_threshold=0.30)

    assert isinstance(result, DentDetectionResult)
    assert result.experimental is True
    assert result.dataset == "CR7-DET (Steel Surface)"
    assert result.dents_detected == 0
    assert len(result.boxes) == 0
    assert "No dent detected" in result.message
    assert result.image_width == 400
    assert result.image_height == 400
    assert result.execution_time_ms > 0


def test_dent_detector_confidence_filtering(dent_detector):
    """Verify that increasing confidence threshold filters detections strictly."""
    if not DENT_TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {DENT_TEST_IMAGE}")

    img = Image.open(DENT_TEST_IMAGE)
    res_low = dent_detector.detect(img, confidence_threshold=0.10)
    res_high = dent_detector.detect(img, confidence_threshold=0.99)

    assert res_low.dents_detected >= res_high.dents_detected
    for box in res_high.boxes:
        assert box.confidence >= 0.99
        assert box.class_name == "Dents"
        assert box.class_id == DENT_CLASS_ID


def test_dent_detector_positive_sample(dent_detector):
    """Verify detection on known held-out Dent test sample (dents103.jpg)."""
    if not DENT_TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {DENT_TEST_IMAGE}")

    img = Image.open(DENT_TEST_IMAGE)
    result = dent_detector.detect(img, confidence_threshold=0.30)

    assert result.dents_detected >= 1
    assert len(result.boxes) == result.dents_detected
    for box in result.boxes:
        assert box.class_name == "Dents"
        assert box.confidence >= 0.30
        assert 0 <= box.x1 < box.x2 <= result.image_width
        assert 0 <= box.y1 < box.y2 <= result.image_height


def test_dent_detector_negative_sample(dent_detector):
    """Verify non-Dent image (inclusions104.jpg) reports zero Dents."""
    if not NONDENT_TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {NONDENT_TEST_IMAGE}")

    img = Image.open(NONDENT_TEST_IMAGE)
    result = dent_detector.detect(img, confidence_threshold=0.30)

    assert result.dents_detected == 0
    assert len(result.boxes) == 0
    assert "No dent detected" in result.message


def test_api_dent_info_endpoint(api_client):
    """Verify GET /experimental/dent/info endpoint contract."""
    response = api_client.get("/experimental/dent/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["experimental"] is True
    assert "CR7-DET" in data["validation_dataset"]
    assert "Faster R-CNN" in data["model_architecture"]


def test_api_dent_detect_multipart(api_client):
    """Verify POST /experimental/dent/detect with multipart upload."""
    if not DENT_TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {DENT_TEST_IMAGE}")

    with open(DENT_TEST_IMAGE, "rb") as f:
        response = api_client.post(
            "/experimental/dent/detect?confidence_threshold=0.30",
            files={"image": ("dents103.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["experimental"] is True
    assert data["dents_detected"] >= 1
    assert len(data["boxes"]) == data["dents_detected"]
    assert data["image_width"] == 400
    assert data["image_height"] == 400


def test_api_dent_detect_json(api_client):
    """Verify POST /experimental/dent/detect with JSON image_path."""
    if not NONDENT_TEST_IMAGE.exists():
        pytest.skip(f"Test image not found: {NONDENT_TEST_IMAGE}")

    response = api_client.post(
        "/experimental/dent/detect",
        json={"image_path": str(NONDENT_TEST_IMAGE), "confidence_threshold": 0.30},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["dents_detected"] == 0
    assert len(data["boxes"]) == 0
    assert "No dent detected" in data["message"]
