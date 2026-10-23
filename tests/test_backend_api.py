"""Tests for DefectVision API Foundation and Endpoint Contracts.

Verifies:
- Product registration (POST /products, GET /products, GET /products/{product_id}).
- Product calibration (POST /products/{product_id}/calibration, GET /products/{product_id}/calibration).
- Product status transition to READY upon successful calibration.
- Inspection execution via JSON payload and multipart upload.
- Orchestrator integration boundary and two-stage decision preservation.
- Persisting and querying inspection history (GET /products/{product_id}/inspections, GET /inspections/{inspection_id}).
- Strict failure handling for missing/uncalibrated products and invalid image files.
- Testing is performed using in-memory SQLite and mock/test orchestrators without touching production PostgreSQL.
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from defectvision.api.app import app
from defectvision.api.dependencies import get_db, get_orchestrator, get_storage
from defectvision.api.storage import StorageManager
from defectvision.db.database import Base
from defectvision.db.models import ProductStatus
from defectvision.inference.orchestrator import DefectVisionOrchestrator
from defectvision.inference.product_profile import ProductProfile
from defectvision.inference.result import InspectionResult


from sqlalchemy.pool import StaticPool


@pytest.fixture
def test_db_session():
    """Create a completely isolated in-memory SQLite database with StaticPool for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def test_storage(tmp_path: Path):
    """Provide isolated application storage for testing."""
    return StorageManager(root_dir=tmp_path / "app_storage")


@pytest.fixture
def mock_orchestrator():
    """Create a mock DefectVisionOrchestrator with configurable inspect behavior."""
    mock = MagicMock(spec=DefectVisionOrchestrator)

    def default_inspect(image_path, product_profile, save_heatmap=False, heatmap_dir=None):
        return InspectionResult(
            product_id=product_profile.product_id,
            image_path=str(image_path).replace("\\", "/"),
            verdict="NORMAL",
            is_defective=False,
            anomaly_score=25.4,
            anomaly_threshold=product_profile.threshold,
            margin=25.4 - product_profile.threshold,
            stage2_executed=False,
            defect_class=None,
            classifier_confidence=None,
            classifier_probabilities=None,
            heatmap_path=None,
            execution_time_ms=35.0,
            metadata={"source": "mock_normal"},
        )

    mock.inspect.side_effect = default_inspect
    return mock


@pytest.fixture
def client(test_db_session, test_storage, mock_orchestrator):
    """FastAPI TestClient with overridden DB, Storage, and Orchestrator dependencies."""
    def override_get_db():
        yield test_db_session

    def override_get_storage():
        return test_storage

    def override_get_orchestrator():
        return mock_orchestrator

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage] = override_get_storage
    app.dependency_overrides[get_orchestrator] = override_get_orchestrator

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def dummy_image_file(tmp_path: Path) -> Path:
    """Create a temporary valid PNG image."""
    img_path = tmp_path / "sample_query.png"
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    img.save(img_path)
    return img_path


@pytest.fixture
def dummy_model_checkpoint(tmp_path: Path) -> Path:
    """Create a temporary dummy checkpoint file."""
    ckpt = tmp_path / "patchcore_dummy.pt"
    ckpt.write_bytes(b"dummy_weights_content")
    return ckpt


class TestSystemHealth:
    def test_health_check(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "defectvision-api"}


class TestProductEndpoints:
    def test_register_product_success(self, client: TestClient):
        payload = {
            "product_id": "bottle",
            "name": "MVTec Bottle Inspection",
            "description": "Bottle surface inspection line",
            "metadata": {"line": 1},
        }
        resp = client.post("/products", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["product_id"] == "bottle"
        assert data["name"] == "MVTec Bottle Inspection"
        assert data["status"] == ProductStatus.REGISTERED.value
        assert data["active_calibration"] is None

    def test_register_duplicate_product_rejected(self, client: TestClient):
        payload = {"product_id": "screw", "name": "Screw Inspection"}
        r1 = client.post("/products", json=payload)
        assert r1.status_code == 201
        r2 = client.post("/products", json=payload)
        assert r2.status_code == 409
        assert "already exists" in r2.json()["detail"]

    def test_list_and_get_products(self, client: TestClient):
        client.post("/products", json={"product_id": "p1", "name": "Product 1"})
        client.post("/products", json={"product_id": "p2", "name": "Product 2"})

        # List
        resp = client.get("/products")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        p_ids = [item["product_id"] for item in items]
        assert "p1" in p_ids and "p2" in p_ids

        # Get by id
        resp_single = client.get("/products/p1")
        assert resp_single.status_code == 200
        assert resp_single.json()["product_id"] == "p1"

        # Not found
        resp_none = client.get("/products/unknown_product")
        assert resp_none.status_code == 404


class TestCalibrationEndpoints:
    def test_get_calibration_uncalibrated_returns_404(self, client: TestClient):
        client.post("/products", json={"product_id": "uncalibrated_prod", "name": "Test"})
        resp = client.get("/products/uncalibrated_prod/calibration")
        assert resp.status_code == 404
        assert "has no active calibration" in resp.json()["detail"]

    def test_calibrate_via_artifact_path(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
    ):
        # Register product
        client.post("/products", json={"product_id": "bottle", "name": "Bottle"})

        # Reference existing bottle calibration artifact
        artifact_path = Path("experiments/patchcore_bottle_calibration/artifacts/bottle_threshold_artifact.json")
        assert artifact_path.is_file(), "Authoritative bottle threshold artifact must exist."

        calib_payload = {
            "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
            "artifact_path": str(artifact_path.resolve()).replace("\\", "/"),
        }
        resp = client.post("/products/bottle/calibration", json=calib_payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["product_id"] == "bottle"
        assert data["threshold"] == pytest.approx(41.469597, rel=1e-3)
        assert data["calibration_method"] == "youden_j"
        assert data["normal_sample_count"] == 10
        assert data["defective_sample_count"] == 32
        assert data["calibration_status"] == "ACTIVE"

        # Verify product status is now READY
        p_resp = client.get("/products/bottle")
        assert p_resp.status_code == 200
        p_data = p_resp.json()
        assert p_data["status"] == ProductStatus.READY.value
        assert p_data["active_calibration"] is not None
        assert p_data["active_calibration"]["threshold"] == pytest.approx(41.469597, rel=1e-3)

        # GET /products/bottle/calibration
        cal_get = client.get("/products/bottle/calibration")
        assert cal_get.status_code == 200
        assert cal_get.json()["threshold"] == pytest.approx(41.469597, rel=1e-3)

    def test_calibrate_via_empirical_scores(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
    ):
        client.post("/products", json={"product_id": "scored_prod", "name": "Scored Prod"})

        # Normal scores and defective scores
        normal_scores = [18.5, 19.2, 20.1, 18.9, 19.8, 21.0, 20.5, 19.4]
        defective_scores = [45.2, 52.1, 48.0, 60.3, 55.4]

        calib_payload = {
            "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
            "normal_scores": normal_scores,
            "defective_scores": defective_scores,
            "calibration_method": "youden_j",
        }
        resp = client.post("/products/scored_prod/calibration", json=calib_payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["product_id"] == "scored_prod"
        assert data["threshold"] > 21.0
        assert data["threshold"] <= 45.2
        assert data["calibration_artifact_path"] is not None
        assert Path(data["calibration_artifact_path"]).is_file()

        # Check product is READY
        p_resp = client.get("/products/scored_prod")
        assert p_resp.json()["status"] == ProductStatus.READY.value

    def test_calibrate_missing_model_checkpoint_fails(self, client: TestClient):
        client.post("/products", json={"product_id": "bad_model_prod", "name": "Bad Model"})
        calib_payload = {
            "patchcore_model_path": "nonexistent/patchcore_model.pt",
            "threshold": 35.0,
        }
        resp = client.post("/products/bad_model_prod/calibration", json=calib_payload)
        assert resp.status_code == 400
        assert "checkpoint not found" in resp.json()["detail"]


class TestInspectionEndpoints:
    def test_inspect_unready_product_fails(
        self,
        client: TestClient,
        dummy_image_file: Path,
    ):
        client.post("/products", json={"product_id": "not_ready_prod", "name": "Not Ready"})
        resp = client.post(
            "/products/not_ready_prod/inspect",
            json={"image_path": str(dummy_image_file).replace("\\", "/")},
        )
        assert resp.status_code == 400
        assert "must be in READY status" in resp.json()["detail"]

    def test_inspect_json_normal_flow(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
        dummy_image_file: Path,
        mock_orchestrator: MagicMock,
    ):
        # Register and calibrate
        client.post("/products", json={"product_id": "bottle", "name": "Bottle"})
        client.post(
            "/products/bottle/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "threshold": 41.5,
            },
        )

        # Mock returns NORMAL
        mock_orchestrator.inspect.side_effect = lambda image_path, product_profile, save_heatmap=False, heatmap_dir=None: InspectionResult(
            product_id="bottle",
            image_path=str(image_path).replace("\\", "/"),
            verdict="NORMAL",
            is_defective=False,
            anomaly_score=26.3,
            anomaly_threshold=41.5,
            margin=-15.2,
            stage2_executed=False,
            defect_class=None,
            classifier_confidence=None,
            classifier_probabilities=None,
            execution_time_ms=42.0,
        )

        resp = client.post(
            "/products/bottle/inspect",
            json={"image_path": str(dummy_image_file).replace("\\", "/")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["product_id"] == "bottle"
        assert data["verdict"] == "NORMAL"
        assert data["anomaly_score"] == pytest.approx(26.3)
        assert data["anomaly_threshold"] == pytest.approx(41.5)
        assert data["margin"] < 0
        assert data["stage2_executed"] is False
        assert data["defect_class"] is None
        assert data["inspection_id"] is not None

    def test_inspect_json_defective_flow(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
        dummy_image_file: Path,
        mock_orchestrator: MagicMock,
    ):
        # Register and calibrate
        client.post("/products", json={"product_id": "bottle_def", "name": "Bottle Defective"})
        client.post(
            "/products/bottle_def/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "threshold": 41.5,
            },
        )

        # Mock returns DEFECTIVE with Crack class
        mock_orchestrator.inspect.side_effect = lambda image_path, product_profile, save_heatmap=False, heatmap_dir=None: InspectionResult(
            product_id="bottle_def",
            image_path=str(image_path).replace("\\", "/"),
            verdict="DEFECTIVE",
            is_defective=True,
            anomaly_score=68.9,
            anomaly_threshold=41.5,
            margin=27.4,
            stage2_executed=True,
            defect_class="Crack",
            classifier_confidence=0.982,
            classifier_probabilities={"Crack": 0.982, "Scratch": 0.015, "Discoloration": 0.002, "UNKNOWN": 0.001},
            heatmap_path="data/heatmaps/test_heatmap.png",
            execution_time_ms=68.5,
        )

        resp = client.post(
            "/products/bottle_def/inspect",
            json={"image_path": str(dummy_image_file).replace("\\", "/"), "save_heatmap": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["product_id"] == "bottle_def"
        assert data["verdict"] == "DEFECTIVE"
        assert data["anomaly_score"] == pytest.approx(68.9)
        assert data["stage2_executed"] is True
        assert data["defect_class"] == "Crack"
        assert data["classifier_confidence"] == pytest.approx(0.982)
        assert data["classifier_probabilities"]["Crack"] == pytest.approx(0.982)

    def test_inspect_multipart_upload(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
    ):
        # Register and calibrate
        client.post("/products", json={"product_id": "upload_prod", "name": "Upload Prod"})
        client.post(
            "/products/upload_prod/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "threshold": 40.0,
            },
        )

        # Create in-memory PNG image bytes
        img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/products/upload_prod/inspect",
            content=buf.getvalue(),
            headers={"content-type": "image/png"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["product_id"] == "upload_prod"
        assert "uploads" in data["image_path"]
        assert Path(data["image_path"]).is_file()

    def test_inspect_corrupt_file_rejected(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
    ):
        client.post("/products", json={"product_id": "corrupt_prod", "name": "Corrupt Prod"})
        client.post(
            "/products/corrupt_prod/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "threshold": 40.0,
            },
        )

        corrupt_bytes = b"this is not a valid image stream"
        resp = client.post(
            "/products/corrupt_prod/inspect",
            content=corrupt_bytes,
            headers={"content-type": "image/png"},
        )
        assert resp.status_code == 400
        assert "validation failed" in resp.json()["detail"].lower()


class TestInspectionHistoryAndRetrieval:
    def test_inspection_history_and_single_get(
        self,
        client: TestClient,
        dummy_model_checkpoint: Path,
        dummy_image_file: Path,
        mock_orchestrator: MagicMock,
    ):
        # Setup product
        client.post("/products", json={"product_id": "history_prod", "name": "History Prod"})
        client.post(
            "/products/history_prod/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "threshold": 40.0,
            },
        )

        # Run 2 inspections (one normal, one defective)
        mock_orchestrator.inspect.side_effect = lambda image_path, product_profile, **kwargs: InspectionResult(
            product_id="history_prod",
            image_path=str(image_path).replace("\\", "/"),
            verdict="NORMAL",
            is_defective=False,
            anomaly_score=20.0,
            anomaly_threshold=40.0,
            margin=-20.0,
            stage2_executed=False,
        )
        r1 = client.post(
            "/products/history_prod/inspect",
            json={"image_path": str(dummy_image_file).replace("\\", "/")},
        )
        assert r1.status_code == 200
        insp1_id = r1.json()["inspection_id"]

        mock_orchestrator.inspect.side_effect = lambda image_path, product_profile, **kwargs: InspectionResult(
            product_id="history_prod",
            image_path=str(image_path).replace("\\", "/"),
            verdict="DEFECTIVE",
            is_defective=True,
            anomaly_score=55.0,
            anomaly_threshold=40.0,
            margin=15.0,
            stage2_executed=True,
            defect_class="Scratch",
            classifier_confidence=0.91,
        )
        r2 = client.post(
            "/products/history_prod/inspect",
            json={"image_path": str(dummy_image_file).replace("\\", "/")},
        )
        assert r2.status_code == 200
        insp2_id = r2.json()["inspection_id"]

        # List all
        list_resp = client.get("/products/history_prod/inspections")
        assert list_resp.status_code == 200
        history_data = list_resp.json()
        assert history_data["total"] == 2
        assert len(history_data["items"]) == 2

        # Filter by verdict: DEFECTIVE
        filtered_resp = client.get("/products/history_prod/inspections?verdict=DEFECTIVE")
        assert filtered_resp.status_code == 200
        filtered_data = filtered_resp.json()
        assert filtered_data["total"] == 1
        assert filtered_data["items"][0]["verdict"] == "DEFECTIVE"
        assert filtered_data["items"][0]["defect_class"] == "Scratch"

        # Get single inspection by UUID
        single_resp = client.get(f"/inspections/{insp1_id}")
        assert single_resp.status_code == 200
        assert single_resp.json()["inspection_id"] == insp1_id
        assert single_resp.json()["verdict"] == "NORMAL"

        # Get nonexistent inspection
        none_resp = client.get("/inspections/nonexistent-uuid-1234")
        assert none_resp.status_code == 404


class TestInspectionMediaEndpoints:
    """Tests for GET /inspections/{inspection_id}/image and /heatmap."""

    def test_get_existing_inspection_image(
        self, client: TestClient, test_storage: StorageManager, test_db_session
    ):
        """1. Existing inspection image can be retrieved."""
        prod_upload_dir = test_storage.uploads_dir / "media_prod"
        prod_upload_dir.mkdir(parents=True, exist_ok=True)
        img_p = prod_upload_dir / "sample.png"
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(img_p)

        from defectvision.db.models import Inspection, Product, ProductStatus
        prod = Product(product_id="media_prod", name="Media Prod", status=ProductStatus.READY.value)
        test_db_session.add(prod)
        test_db_session.commit()

        insp = Inspection(
            inspection_id="media-insp-001",
            product_id="media_prod",
            image_path=str(img_p).replace("\\", "/"),
            verdict="NORMAL",
            anomaly_score=10.0,
            anomaly_threshold=20.0,
            margin=-10.0,
        )
        test_db_session.add(insp)
        test_db_session.commit()

        resp = client.get("/inspections/media-insp-001/image")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
        assert len(resp.content) == len(img_p.read_bytes())

    def test_get_existing_inspection_heatmap(
        self, client: TestClient, test_storage: StorageManager, test_db_session
    ):
        """2. Existing heatmap can be retrieved."""
        prod_heatmap_dir = test_storage.heatmaps_dir / "media_prod_2"
        prod_heatmap_dir.mkdir(parents=True, exist_ok=True)
        hm_p = prod_heatmap_dir / "sample_heatmap.png"
        hm_img = Image.new("RGB", (64, 64), color="red")
        hm_img.save(hm_p)

        from defectvision.db.models import Inspection, Product, ProductStatus
        prod = Product(product_id="media_prod_2", name="Media Prod 2", status=ProductStatus.READY.value)
        test_db_session.add(prod)
        test_db_session.commit()

        insp = Inspection(
            inspection_id="media-insp-002",
            product_id="media_prod_2",
            image_path=str(hm_p).replace("\\", "/"),
            heatmap_path=str(hm_p).replace("\\", "/"),
            verdict="DEFECTIVE",
            anomaly_score=35.0,
            anomaly_threshold=20.0,
            margin=15.0,
        )
        test_db_session.add(insp)
        test_db_session.commit()

        resp = client.get("/inspections/media-insp-002/heatmap")
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]
        assert len(resp.content) == len(hm_p.read_bytes())

    def test_nonexistent_inspection_returns_404(self, client: TestClient):
        """3. Nonexistent inspection returns 404 for image and heatmap."""
        r_img = client.get("/inspections/does-not-exist/image")
        assert r_img.status_code == 404
        r_hm = client.get("/inspections/does-not-exist/heatmap")
        assert r_hm.status_code == 404

    def test_missing_image_or_heatmap_returns_404(
        self, client: TestClient, test_storage: StorageManager, test_db_session
    ):
        """4. Missing image or heatmap returns 404."""
        from defectvision.db.models import Inspection, Product, ProductStatus
        prod = Product(product_id="media_prod_3", name="Media Prod 3", status=ProductStatus.READY.value)
        test_db_session.add(prod)
        test_db_session.commit()

        insp_no_hm = Inspection(
            inspection_id="media-insp-no-hm",
            product_id="media_prod_3",
            image_path=str(test_storage.uploads_dir / "nonexistent.png").replace("\\", "/"),
            heatmap_path=None,
            verdict="NORMAL",
            anomaly_score=5.0,
            anomaly_threshold=20.0,
            margin=-15.0,
        )
        test_db_session.add(insp_no_hm)
        test_db_session.commit()

        r_no_hm = client.get("/inspections/media-insp-no-hm/heatmap")
        assert r_no_hm.status_code == 404
        assert "no associated heatmap file" in r_no_hm.json()["detail"].lower()

        r_missing_img = client.get("/inspections/media-insp-no-hm/image")
        assert r_missing_img.status_code == 404
        assert "not found on disk" in r_missing_img.json()["detail"].lower()

    def test_path_traversal_rejected(
        self, client: TestClient, test_storage: StorageManager, test_db_session, tmp_path: Path
    ):
        """5. Path traversal / arbitrary filesystem access is rejected."""
        external_file = tmp_path / "secret.txt"
        external_file.write_text("classified data")

        from defectvision.db.models import Inspection, Product, ProductStatus
        prod = Product(product_id="traversal_prod", name="Traversal", status=ProductStatus.READY.value)
        test_db_session.add(prod)
        test_db_session.commit()

        insp_traversal = Inspection(
            inspection_id="traversal-insp",
            product_id="traversal_prod",
            image_path=str(external_file).replace("\\", "/"),
            heatmap_path=str(external_file).replace("\\", "/"),
            verdict="DEFECTIVE",
            anomaly_score=50.0,
            anomaly_threshold=20.0,
            margin=30.0,
        )
        test_db_session.add(insp_traversal)
        test_db_session.commit()

        r_img = client.get("/inspections/traversal-insp/image")
        assert r_img.status_code == 403
        assert "outside application storage" in r_img.json()["detail"]

        r_hm = client.get("/inspections/traversal-insp/heatmap")
        assert r_hm.status_code == 403
        assert "outside application storage" in r_hm.json()["detail"]


class TestAnalyticsEndpoints:
    """Tests for GET /analytics endpoint."""

    def test_empty_database_returns_valid_zero_metrics(self, client: TestClient):
        """1. Empty database returns valid zero metrics."""
        resp = client.get("/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_inspections"] == 0
        assert data["normal_count"] == 0
        assert data["defective_count"] == 0
        assert data["defect_rate"] == 0.0
        assert data["normal_rate"] == 0.0
        assert data["average_execution_time_ms"] == 0.0
        assert data["defect_class_counts"] == {
            "Scratch": 0,
            "Crack": 0,
            "Discoloration": 0,
            "UNKNOWN": 0,
        }

    def test_mixed_inspections_and_defect_classes(
        self, client: TestClient, test_db_session
    ):
        """2 & 3. Mixed NORMAL/DEFECTIVE inspections produce correct counts and defect classes."""
        from defectvision.db.models import Inspection, Product, ProductStatus
        prod = Product(product_id="analytics_p1", name="Analytics 1", status=ProductStatus.READY.value)
        test_db_session.add(prod)
        test_db_session.commit()

        # Add 3 NORMAL inspections
        for i in range(3):
            test_db_session.add(
                Inspection(
                    inspection_id=f"norm-{i}",
                    product_id="analytics_p1",
                    image_path=f"/dummy/path/norm_{i}.png",
                    verdict="NORMAL",
                    anomaly_score=10.0,
                    anomaly_threshold=20.0,
                    margin=-10.0,
                    execution_time_ms=100.0,
                )
            )

        # Add 2 DEFECTIVE inspections (1 Crack, 1 Scratch)
        test_db_session.add(
            Inspection(
                inspection_id="def-1",
                product_id="analytics_p1",
                image_path="/dummy/path/def_1.png",
                verdict="DEFECTIVE",
                anomaly_score=35.0,
                anomaly_threshold=20.0,
                margin=15.0,
                stage2_executed=True,
                defect_class="Crack",
                execution_time_ms=200.0,
            )
        )
        test_db_session.add(
            Inspection(
                inspection_id="def-2",
                product_id="analytics_p1",
                image_path="/dummy/path/def_2.png",
                verdict="DEFECTIVE",
                anomaly_score=40.0,
                anomaly_threshold=20.0,
                margin=20.0,
                stage2_executed=True,
                defect_class="Scratch",
                execution_time_ms=300.0,
            )
        )
        test_db_session.commit()

        resp = client.get("/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_inspections"] == 5
        assert data["normal_count"] == 3
        assert data["defective_count"] == 2
        assert data["defect_rate"] == pytest.approx(0.4)
        assert data["normal_rate"] == pytest.approx(0.6)
        assert data["average_execution_time_ms"] == pytest.approx(160.0)
        assert data["defect_class_counts"] == {
            "Scratch": 1,
            "Crack": 1,
            "Discoloration": 0,
            "UNKNOWN": 0,
        }

    def test_product_specific_filtering_and_404(
        self, client: TestClient, test_db_session
    ):
        """4 & 5. Product-specific filtering works, nonexistent product returns 404."""
        from defectvision.db.models import Inspection, Product, ProductStatus
        p1 = Product(product_id="filter_p1", name="Product 1", status=ProductStatus.READY.value)
        p2 = Product(product_id="filter_p2", name="Product 2", status=ProductStatus.READY.value)
        test_db_session.add_all([p1, p2])
        test_db_session.commit()

        test_db_session.add(Inspection(
            inspection_id="p1-norm", product_id="filter_p1", image_path="/p1/1.png",
            verdict="NORMAL", anomaly_score=10.0, anomaly_threshold=20.0, margin=-10.0
        ))
        test_db_session.add(Inspection(
            inspection_id="p1-def", product_id="filter_p1", image_path="/p1/2.png",
            verdict="DEFECTIVE", anomaly_score=30.0, anomaly_threshold=20.0, margin=10.0,
            stage2_executed=True, defect_class="Discoloration"
        ))
        test_db_session.add(Inspection(
            inspection_id="p2-norm", product_id="filter_p2", image_path="/p2/1.png",
            verdict="NORMAL", anomaly_score=8.0, anomaly_threshold=20.0, margin=-12.0
        ))
        test_db_session.commit()

        # Filter by p1
        r_p1 = client.get("/analytics?product_id=filter_p1")
        assert r_p1.status_code == 200
        d_p1 = r_p1.json()
        assert d_p1["product_id"] == "filter_p1"
        assert d_p1["total_inspections"] == 2
        assert d_p1["normal_count"] == 1
        assert d_p1["defective_count"] == 1
        assert d_p1["defect_class_counts"]["Discoloration"] == 1

        # Filter by p2
        r_p2 = client.get("/analytics?product_id=filter_p2")
        assert r_p2.status_code == 200
        d_p2 = r_p2.json()
        assert d_p2["product_id"] == "filter_p2"
        assert d_p2["total_inspections"] == 1
        assert d_p2["normal_count"] == 1
        assert d_p2["defective_count"] == 0

        # Filter by nonexistent product
        r_none = client.get("/analytics?product_id=nonexistent_product")
        assert r_none.status_code == 404
        assert "not found" in r_none.json()["detail"].lower()

    def test_average_execution_time_calculation(
        self, client: TestClient, test_db_session
    ):
        """6. Average execution time calculation is accurate."""
        from defectvision.db.models import Inspection, Product, ProductStatus
        prod = Product(product_id="latency_prod", name="Latency", status=ProductStatus.READY.value)
        test_db_session.add(prod)
        test_db_session.commit()

        test_db_session.add_all([
            Inspection(
                inspection_id="lat-1", product_id="latency_prod", image_path="/lat/1.png",
                verdict="NORMAL", anomaly_score=10.0, anomaly_threshold=20.0, margin=-10.0,
                execution_time_ms=120.0,
            ),
            Inspection(
                inspection_id="lat-2", product_id="latency_prod", image_path="/lat/2.png",
                verdict="NORMAL", anomaly_score=10.0, anomaly_threshold=20.0, margin=-10.0,
                execution_time_ms=180.0,
            ),
        ])
        test_db_session.commit()

        resp = client.get("/analytics?product_id=latency_prod")
        assert resp.status_code == 200
        assert resp.json()["average_execution_time_ms"] == pytest.approx(150.0)


class TestRunCalibrationEndpoint:
    """Tests for POST /products/{product_id}/calibration/run (dynamic calibration).

    All tests mock PatchCorePipeline so they run without GPU/Anomalib fitting.
    """

    def _make_png_bytes(self, size: tuple[int, int] = (64, 64)) -> bytes:
        """Return bytes of a minimal valid in-memory PNG."""
        buf = io.BytesIO()
        img = Image.new("RGB", size, color=(100, 150, 200))
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _make_files(self, n: int) -> list[tuple[str, tuple[str, bytes, str]]]:
        """Build a multipart files list with n PNG uploads."""
        png = self._make_png_bytes()
        return [("files", (f"ref_{i:03d}.png", png, "image/png")) for i in range(n)]

    def _inject_fake_ckpt(self, models_dir: Path, product_id: str) -> Path:
        """Create a fake checkpoint file under the Anomalib engine output structure."""
        ckpt_dir = (
            models_dir
            / "engine_runs"
            / "patchcore"
            / product_id
            / "version_0"
            / "checkpoints"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / "epoch=0.ckpt"
        ckpt.write_bytes(b"fake_checkpoint")
        return ckpt

    def _make_mock_pipeline(self, normal_scores: list[float]):
        """Return a mock PatchCorePipeline instance."""
        from unittest.mock import MagicMock
        from defectvision.models.patchcore_engine import AnomalyInferenceResult
        import numpy as np

        mock_inst = MagicMock()
        mock_inst.build_feature_memory.return_value = {
            "product_name": "test",
            "reference_count": len(normal_scores),
            "backbone": "wide_resnet50_2",
            "memory_bank_shape": [100, 512],
            "is_fitted": True,
        }
        mock_inst.infer_batch.return_value = [
            AnomalyInferenceResult(
                image_path=f"/fake/ref_{i}.png",
                anomaly_score=s,
                pred_label=False,
                anomaly_map=np.zeros((8, 8), dtype=np.float32),
            )
            for i, s in enumerate(normal_scores)
        ]
        mock_inst.is_fitted = True
        return mock_inst

    def test_run_calibration_success(
        self,
        client: TestClient,
        test_storage: StorageManager,
    ):
        """Happy-path: valid product + 10 reference images → 201 with correct fields."""
        from unittest.mock import patch

        client.post("/products", json={"product_id": "widget", "name": "Widget"})

        normal_scores = [20.0 + i * 0.4 for i in range(10)]
        mock_pipeline = self._make_mock_pipeline(normal_scores)

        models_dir = test_storage.root / "products" / "widget" / "models"
        self._inject_fake_ckpt(models_dir, "widget")

        with patch(
            "defectvision.api.routes.calibration.PatchCorePipeline",
            return_value=mock_pipeline,
        ):
            resp = client.post(
                "/products/widget/calibration/run",
                files=self._make_files(10),
            )

        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["product_id"] == "widget"
        assert data["calibration_method"] == "normal_k_sigma"
        assert data["reference_image_count"] == 10
        assert data["product_status"] == "READY"
        assert data["threshold"] > 0
        assert isinstance(data["warnings"], list)
        assert Path(data["artifact_path"]).is_file()

        # Product must be READY
        p_resp = client.get("/products/widget")
        assert p_resp.json()["status"] == ProductStatus.READY.value

        # Active calibration must exist and reflect normal_k_sigma
        cal_resp = client.get("/products/widget/calibration")
        assert cal_resp.status_code == 200
        assert cal_resp.json()["calibration_method"] == "normal_k_sigma"

    def test_run_calibration_9_images_rejected(self, client: TestClient):
        """Boundary: 9 images (one below minimum) must be rejected with 422."""
        client.post("/products", json={"product_id": "few_prod_9", "name": "Few9"})
        resp = client.post(
            "/products/few_prod_9/calibration/run",
            files=self._make_files(9),
        )
        assert resp.status_code == 422
        assert "At least 10" in resp.json()["detail"]

    def test_run_calibration_10_images_accepted(
        self,
        client: TestClient,
        test_storage: StorageManager,
    ):
        """Boundary: exactly 10 images (minimum) must be accepted."""
        from unittest.mock import patch

        client.post("/products", json={"product_id": "min_prod_10", "name": "Min10"})
        normal_scores = [20.0 + i * 0.5 for i in range(10)]
        mock_pipeline = self._make_mock_pipeline(normal_scores)
        models_dir = test_storage.root / "products" / "min_prod_10" / "models"
        self._inject_fake_ckpt(models_dir, "min_prod_10")

        with patch(
            "defectvision.api.routes.calibration.PatchCorePipeline",
            return_value=mock_pipeline,
        ):
            resp = client.post(
                "/products/min_prod_10/calibration/run",
                files=self._make_files(10),
            )

        assert resp.status_code == 201, resp.text
        assert resp.json()["reference_image_count"] == 10

    def test_run_calibration_20_images_accepted(
        self,
        client: TestClient,
        test_storage: StorageManager,
    ):
        """Boundary: exactly 20 images (maximum) must be accepted."""
        from unittest.mock import patch

        client.post("/products", json={"product_id": "max_prod_20", "name": "Max20"})
        normal_scores = [20.0 + i * 0.2 for i in range(20)]
        mock_pipeline = self._make_mock_pipeline(normal_scores)
        models_dir = test_storage.root / "products" / "max_prod_20" / "models"
        self._inject_fake_ckpt(models_dir, "max_prod_20")

        with patch(
            "defectvision.api.routes.calibration.PatchCorePipeline",
            return_value=mock_pipeline,
        ):
            resp = client.post(
                "/products/max_prod_20/calibration/run",
                files=self._make_files(20),
            )

        assert resp.status_code == 201, resp.text
        assert resp.json()["reference_image_count"] == 20

    def test_run_calibration_21_images_rejected(self, client: TestClient):
        """Boundary: 21 images (one above maximum) must be rejected with 422."""
        client.post("/products", json={"product_id": "many_prod_21", "name": "Many21"})
        resp = client.post(
            "/products/many_prod_21/calibration/run",
            files=self._make_files(21),
        )
        assert resp.status_code == 422
        assert "At most 20" in resp.json()["detail"]

    def test_run_calibration_invalid_extension(self, client: TestClient):
        """A non-image file extension must be rejected with 422."""
        client.post("/products", json={"product_id": "ext_prod", "name": "Ext"})
        bad_files = [("files", ("ref.txt", b"not an image", "text/plain"))]
        resp = client.post("/products/ext_prod/calibration/run", files=bad_files)
        assert resp.status_code in (422, 400), resp.text

    def test_run_calibration_product_not_found(self, client: TestClient):
        """Calibration run on non-existent product must return 404."""
        resp = client.post(
            "/products/nonexistent_xyz/calibration/run",
            files=self._make_files(10),
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_run_calibration_supersedes_existing(
        self,
        client: TestClient,
        test_storage: StorageManager,
        dummy_model_checkpoint: Path,
    ):
        """A second calibration run must supersede the prior ACTIVE calibration record."""
        from unittest.mock import patch

        client.post("/products", json={"product_id": "recal_prod", "name": "Recal"})

        # First: register via artifact path (pre-computed approach)
        artifact_path = Path(
            "experiments/patchcore_bottle_calibration/artifacts/bottle_threshold_artifact.json"
        )
        if not artifact_path.is_file():
            pytest.skip("Bottle threshold artifact not present on disk.")

        client.post(
            "/products/recal_prod/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "artifact_path": str(artifact_path.resolve()).replace("\\", "/"),
            },
        )
        cal1 = client.get("/products/recal_prod/calibration")
        assert cal1.status_code == 200
        first_threshold = cal1.json()["threshold"]

        # Second: dynamic calibration run (mocked) with different scores
        normal_scores = [30.0 + i * 0.3 for i in range(10)]
        mock_pipeline = self._make_mock_pipeline(normal_scores)

        models_dir = test_storage.root / "products" / "recal_prod" / "models"
        self._inject_fake_ckpt(models_dir, "recal_prod")

        with patch(
            "defectvision.api.routes.calibration.PatchCorePipeline",
            return_value=mock_pipeline,
        ):
            resp = client.post(
                "/products/recal_prod/calibration/run",
                files=self._make_files(10),
            )

        assert resp.status_code == 201, resp.text
        new_threshold = resp.json()["threshold"]

        # Active calibration now reflects the new threshold
        cal2 = client.get("/products/recal_prod/calibration")
        assert cal2.status_code == 200
        assert cal2.json()["threshold"] == pytest.approx(new_threshold, rel=1e-5)
        assert cal2.json()["threshold"] != pytest.approx(first_threshold, rel=1e-2)

    def test_run_calibration_failure_safety_preserves_existing_state(
        self,
        client: TestClient,
        test_storage: StorageManager,
        dummy_model_checkpoint: Path,
    ):
        """A failed calibration run must NOT mark product READY or corrupt active calibration."""
        from unittest.mock import patch

        client.post("/products", json={"product_id": "fail_prod", "name": "FailSafeProd"})

        # Initial valid active calibration
        artifact_path = Path(
            "experiments/patchcore_bottle_calibration/artifacts/bottle_threshold_artifact.json"
        )
        if not artifact_path.is_file():
            pytest.skip("Bottle threshold artifact not present on disk.")

        client.post(
            "/products/fail_prod/calibration",
            json={
                "patchcore_model_path": str(dummy_model_checkpoint).replace("\\", "/"),
                "artifact_path": str(artifact_path.resolve()).replace("\\", "/"),
            },
        )
        cal_before = client.get("/products/fail_prod/calibration").json()
        assert cal_before["threshold"] > 0
        prod_before = client.get("/products/fail_prod").json()
        assert prod_before["status"] == ProductStatus.READY.value

        # Attempt dynamic calibration with pipeline raising RuntimeError
        mock_broken_pipeline = self._make_mock_pipeline([25.0] * 10)
        mock_broken_pipeline.build_feature_memory.side_effect = RuntimeError("GPU OOM / fitting error")

        with patch(
            "defectvision.api.routes.calibration.PatchCorePipeline",
            return_value=mock_broken_pipeline,
        ):
            resp = client.post(
                "/products/fail_prod/calibration/run",
                files=self._make_files(10),
            )

        assert resp.status_code == 500
        assert "PatchCore fitting failed" in resp.json()["detail"]

        # Ensure previous active calibration is STILL active and unchanged
        cal_after = client.get("/products/fail_prod/calibration").json()
        assert cal_after["id"] == cal_before["id"]
        assert cal_after["threshold"] == pytest.approx(cal_before["threshold"])
        assert cal_after["calibration_status"] == "ACTIVE"

        # Ensure product remains READY with previous calibration intact
        prod_after = client.get("/products/fail_prod").json()
        assert prod_after["status"] == ProductStatus.READY.value

