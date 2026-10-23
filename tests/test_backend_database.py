"""Tests for DefectVision Database Models and Schema Constraints.

Verifies:
- Product creation, uniqueness constraints, and lifecycle states.
- ProductCalibration records, relationships, and active calibration resolution.
- Inspection records, foreign keys, cascade deletes, and JSON serialization.
- Fully isolated in-memory SQLite testing without modifying production PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from defectvision.db.database import Base
from defectvision.db.models import (
    CalibrationRecordStatus,
    Inspection,
    Product,
    ProductCalibration,
    ProductStatus,
    utc_now,
)


@pytest.fixture
def db_session():
    """Create a completely isolated in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


class TestProductModel:
    """Test Product entity schema, constraints, and relationships."""

    def test_create_product_defaults(self, db_session: Session):
        product = Product(
            product_id="test_bottle",
            name="Test Bottle Product",
            description="Testing product registration",
        )
        db_session.add(product)
        db_session.commit()
        db_session.refresh(product)

        assert product.id is not None
        assert product.product_id == "test_bottle"
        assert product.status == ProductStatus.REGISTERED.value
        assert product.created_at is not None
        assert product.updated_at is not None
        assert product.active_calibration is None
        assert len(product.calibrations) == 0
        assert len(product.inspections) == 0

    def test_product_id_uniqueness(self, db_session: Session):
        p1 = Product(product_id="unique_prod", name="Product 1")
        db_session.add(p1)
        db_session.commit()

        p2 = Product(product_id="unique_prod", name="Product 2 Duplicate")
        db_session.add(p2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()


class TestProductCalibrationModel:
    """Test ProductCalibration schema, foreign keys, and active status resolution."""

    def test_create_calibration(self, db_session: Session):
        product = Product(product_id="calib_prod", name="Calib Prod")
        db_session.add(product)
        db_session.commit()

        cal = ProductCalibration(
            product_id="calib_prod",
            patchcore_model_path="experiments/patchcore/model.pt",
            threshold=41.469,
            calibration_method="youden_j",
            calibration_timestamp="2026-10-03T12:00:00Z",
            normal_sample_count=10,
            defective_sample_count=32,
            image_width=256,
            image_height=256,
            backbone="wide_resnet50_2",
            calibration_status=CalibrationRecordStatus.ACTIVE.value,
            metadata_json={"metric": "youden_j", "value": 1.0},
        )
        db_session.add(cal)
        db_session.commit()
        db_session.refresh(product)

        assert product.active_calibration is not None
        assert product.active_calibration.threshold == pytest.approx(41.469)
        assert product.active_calibration.image_size == (256, 256)
        assert product.active_calibration.calibration_method == "youden_j"

    def test_superseded_calibration_resolution(self, db_session: Session):
        product = Product(product_id="multi_calib", name="Multi Calib")
        db_session.add(product)
        db_session.commit()

        cal1 = ProductCalibration(
            product_id="multi_calib",
            patchcore_model_path="path/v1.pt",
            threshold=35.0,
            calibration_method="k_sigma",
            calibration_status=CalibrationRecordStatus.SUPERSEDED.value,
        )
        cal2 = ProductCalibration(
            product_id="multi_calib",
            patchcore_model_path="path/v2.pt",
            threshold=42.5,
            calibration_method="youden_j",
            calibration_status=CalibrationRecordStatus.ACTIVE.value,
        )
        db_session.add_all([cal1, cal2])
        db_session.commit()
        db_session.refresh(product)

        assert len(product.calibrations) == 2
        assert product.active_calibration is not None
        assert product.active_calibration.threshold == pytest.approx(42.5)


class TestInspectionModel:
    """Test Inspection schema, cascade delete, and JSON metadata."""

    def test_create_normal_inspection(self, db_session: Session):
        product = Product(product_id="insp_prod", name="Insp Prod")
        db_session.add(product)
        db_session.commit()

        insp = Inspection(
            inspection_id="insp-uuid-001",
            product_id="insp_prod",
            image_path="data/test/normal_001.png",
            verdict="NORMAL",
            anomaly_score=28.5,
            anomaly_threshold=41.47,
            margin=-12.97,
            stage2_executed=False,
            defect_class=None,
            classifier_confidence=None,
            classifier_probabilities=None,
            execution_time_ms=45.2,
            metadata_json={"backbone": "wide_resnet50_2"},
        )
        db_session.add(insp)
        db_session.commit()
        db_session.refresh(insp)

        assert insp.id is not None
        assert insp.verdict == "NORMAL"
        assert insp.stage2_executed is False
        assert insp.margin < 0
        assert insp.defect_class is None

    def test_create_defective_inspection(self, db_session: Session):
        product = Product(product_id="insp_prod_2", name="Insp Prod 2")
        db_session.add(product)
        db_session.commit()

        insp = Inspection(
            inspection_id="insp-uuid-002",
            product_id="insp_prod_2",
            image_path="data/test/defective_001.png",
            verdict="DEFECTIVE",
            anomaly_score=68.2,
            anomaly_threshold=41.47,
            margin=26.73,
            stage2_executed=True,
            defect_class="Crack",
            classifier_confidence=0.965,
            classifier_probabilities={"Crack": 0.965, "Scratch": 0.02, "Discoloration": 0.01, "UNKNOWN": 0.005},
            heatmap_path="data/heatmaps/defective_001_heatmap.png",
            execution_time_ms=62.8,
        )
        db_session.add(insp)
        db_session.commit()
        db_session.refresh(insp)

        assert insp.verdict == "DEFECTIVE"
        assert insp.stage2_executed is True
        assert insp.defect_class == "Crack"
        assert insp.classifier_confidence == pytest.approx(0.965)
        assert insp.classifier_probabilities["Crack"] == pytest.approx(0.965)

    def test_cascade_delete(self, db_session: Session):
        product = Product(product_id="cascade_prod", name="Cascade Prod")
        db_session.add(product)
        db_session.commit()

        cal = ProductCalibration(
            product_id="cascade_prod",
            patchcore_model_path="path.pt",
            threshold=40.0,
            calibration_method="auto",
        )
        insp = Inspection(
            inspection_id="cascade-insp-1",
            product_id="cascade_prod",
            image_path="img.png",
            verdict="NORMAL",
            anomaly_score=10.0,
            anomaly_threshold=40.0,
            margin=-30.0,
            stage2_executed=False,
        )
        db_session.add_all([cal, insp])
        db_session.commit()

        # Delete product
        db_session.delete(product)
        db_session.commit()

        # Verify child records were cascade-deleted
        remaining_cal = db_session.execute(
            select(ProductCalibration).where(ProductCalibration.product_id == "cascade_prod")
        ).scalars().all()
        remaining_insp = db_session.execute(
            select(Inspection).where(Inspection.product_id == "cascade_prod")
        ).scalars().all()

        assert len(remaining_cal) == 0
        assert len(remaining_insp) == 0
