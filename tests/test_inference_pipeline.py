"""Tests for DefectVision Two-Stage Inference Pipeline.

Verifies:
1. NORMAL image: Stage 1 runs, Stage 2 is NOT executed, result is NORMAL.
2. DEFECTIVE image: Stage 1 runs, Stage 2 IS executed, result contains classifier output.
3. Product-specific threshold is actually used in decision logic.
4. Changing product threshold changes Stage 1 verdict appropriately.
5. Continuous raw anomaly score is preserved (not quantized or binarized).
6. Classifier returns exactly the four supported classes (Scratch, Crack, Discoloration, UNKNOWN).
7. Classifier probabilities sum approximately to 1.0.
8. Missing/invalid product profile fails clearly.
9. Missing model/checkpoint artifact fails clearly rather than fabricating results.
10. End-to-end test using real MVTec bottle artifacts and checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import numpy as np
from PIL import Image
import pytest
import torch

from defectvision.calibration import InspectionStatus
from defectvision.inference import (
    DefectVisionOrchestrator,
    InspectionResult,
    ProductProfile,
    Stage1AnomalyDetector,
    Stage1Result,
    Stage2DefectClassifier,
    Stage2Result,
    create_orchestrator,
)
from defectvision.models.defect_classifier import DEFAULT_DEFECT_CLASSES

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PATCHCORE_CKPT = (
    WORKSPACE_ROOT
    / "experiments"
    / "patchcore_bottle"
    / "staging"
    / "engine_runs"
    / "Patchcore"
    / "bottle"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)
CALIB_ARTIFACT = (
    WORKSPACE_ROOT
    / "experiments"
    / "patchcore_bottle_calibration"
    / "artifacts"
    / "bottle_threshold_artifact.json"
)
CLASSIFIER_CKPT = (
    WORKSPACE_ROOT / "experiments" / "defect_classifier_run1" / "best_model.pt"
)
SAMPLE_NORMAL_IMG = (
    Path("D:/datasets/defect_dataset_workspace/sources/MVTecAD/bottle/test/good/001.png")
)
SAMPLE_DEFECT_IMG = (
    Path("D:/datasets/defect_dataset_workspace/sources/MVTecAD/bottle/test/broken_large/001.png")
)


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    """Create a dummy RGB image for fast unit testing."""
    img_path = tmp_path / "test_dummy.png"
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    img.save(img_path)
    return img_path


@pytest.fixture
def valid_profile(dummy_image: Path) -> ProductProfile:
    """Create a valid dummy product profile."""
    return ProductProfile(
        product_id="test_product",
        patchcore_model_path=str(dummy_image),  # points to existing file for path check
        threshold=40.0,
        calibration_method="youden_j",
        calibration_timestamp="2026-10-03T00:00:00Z",
    )


class TestProductProfile:
    """Unit tests for ProductProfile representation and validation."""

    def test_valid_profile_creation(self, valid_profile: ProductProfile):
        assert valid_profile.product_id == "test_product"
        assert valid_profile.threshold == 40.0
        assert valid_profile.calibration_method == "youden_j"

    def test_invalid_profile_empty_id(self, dummy_image: Path):
        profile = ProductProfile(product_id="", patchcore_model_path=str(dummy_image), threshold=10.0)
        with pytest.raises(ValueError, match="product_id"):
            profile.validate()

    def test_invalid_profile_non_positive_threshold(self, dummy_image: Path):
        profile = ProductProfile(product_id="prod", patchcore_model_path=str(dummy_image), threshold=-5.0)
        with pytest.raises(ValueError, match="threshold must be positive"):
            profile.validate()

    def test_profile_missing_model_path_fails(self):
        profile = ProductProfile(
            product_id="prod",
            patchcore_model_path="non_existent/path/model.ckpt",
            threshold=10.0,
        )
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            profile.validate(check_paths_exist=True)

    def test_profile_serialization_round_trip(self, valid_profile: ProductProfile, tmp_path: Path):
        save_p = tmp_path / "profile.json"
        valid_profile.save_json(save_p)
        loaded = ProductProfile.load_json(save_p)

        assert loaded.product_id == valid_profile.product_id
        assert loaded.threshold == valid_profile.threshold
        assert loaded.patchcore_model_path == valid_profile.patchcore_model_path
        assert loaded.calibration_method == valid_profile.calibration_method

    def test_profile_from_calibration_artifact(self):
        if not CALIB_ARTIFACT.is_file():
            pytest.skip("Calibration artifact not present on disk.")

        profile = ProductProfile.from_calibration_artifact(
            artifact_path=CALIB_ARTIFACT,
            patchcore_model_path=PATCHCORE_CKPT,
        )
        assert profile.product_id == "bottle"
        assert profile.threshold == pytest.approx(41.469597, rel=1e-4)
        assert profile.calibration_method == "youden_j"
        assert profile.metadata["normal_sample_count"] == 10
        assert profile.metadata["defective_sample_count"] == 32


class TestInspectionResult:
    """Unit tests for InspectionResult serialization and contracts."""

    def test_result_to_dict_and_json(self):
        res = InspectionResult(
            product_id="bottle",
            image_path="test.png",
            verdict="DEFECTIVE",
            is_defective=True,
            anomaly_score=55.25,
            anomaly_threshold=41.47,
            margin=13.78,
            stage2_executed=True,
            defect_class="Crack",
            classifier_confidence=0.92,
            classifier_probabilities={"Scratch": 0.03, "Crack": 0.92, "Discoloration": 0.01, "UNKNOWN": 0.04},
        )
        d = res.to_dict()
        assert d["verdict"] == "DEFECTIVE"
        assert d["stage2_executed"] is True
        assert d["defect_class"] == "Crack"
        assert d["anomaly_score"] == 55.25
        assert d["margin"] == pytest.approx(13.78, rel=1e-4)

        j_str = res.to_json()
        data = json.loads(j_str)
        assert data["product_id"] == "bottle"
        assert data["defect_class"] == "Crack"


class TestTwoStagePipelineLogic:
    """Unit tests verifying branching, threshold behavior, and Stage 2 gating."""

    def test_normal_image_skips_stage2(self, dummy_image: Path, valid_profile: ProductProfile):
        """Req 1: NORMAL image runs Stage 1, Stage 2 is NOT executed, verdict is NORMAL."""
        mock_stage1 = MagicMock(spec=Stage1AnomalyDetector)
        mock_stage1.detect.return_value = Stage1Result(
            image_path=str(dummy_image),
            verdict=InspectionStatus.NORMAL,
            is_defective=False,
            anomaly_score=32.5,
            threshold=40.0,
            margin=-7.5,
            anomaly_map=None,
        )

        mock_stage2 = MagicMock(spec=Stage2DefectClassifier)

        orchestrator = DefectVisionOrchestrator(
            stage1_detector=mock_stage1,
            stage2_classifier=mock_stage2,
        )

        result = orchestrator.inspect(dummy_image, valid_profile)

        assert mock_stage1.detect.called
        assert not mock_stage2.classify.called, "Stage 2 classify() must NOT be called for NORMAL items"
        assert result.verdict == "NORMAL"
        assert result.is_defective is False
        assert result.stage2_executed is False
        assert result.defect_class is None
        assert result.classifier_confidence is None
        assert result.classifier_probabilities is None
        assert result.anomaly_score == 32.5
        assert result.anomaly_threshold == 40.0

    def test_defective_image_executes_stage2(self, dummy_image: Path, valid_profile: ProductProfile):
        """Req 2: DEFECTIVE image runs Stage 1, Stage 2 IS executed, result has classifier output."""
        mock_stage1 = MagicMock(spec=Stage1AnomalyDetector)
        mock_stage1.detect.return_value = Stage1Result(
            image_path=str(dummy_image),
            verdict=InspectionStatus.DEFECTIVE,
            is_defective=True,
            anomaly_score=52.8,
            threshold=40.0,
            margin=12.8,
            anomaly_map=None,
        )

        mock_stage2 = MagicMock(spec=Stage2DefectClassifier)
        mock_stage2.loaded_checkpoint_path = "mock_ckpt.pt"
        mock_stage2.classify.return_value = Stage2Result(
            predicted_class="Scratch",
            confidence=0.88,
            class_probabilities={"Scratch": 0.88, "Crack": 0.05, "Discoloration": 0.02, "UNKNOWN": 0.05},
        )

        orchestrator = DefectVisionOrchestrator(
            stage1_detector=mock_stage1,
            stage2_classifier=mock_stage2,
        )

        result = orchestrator.inspect(dummy_image, valid_profile)

        assert mock_stage1.detect.called
        assert mock_stage2.classify.called, "Stage 2 classify() MUST be called for DEFECTIVE items"
        assert result.verdict == "DEFECTIVE"
        assert result.is_defective is True
        assert result.stage2_executed is True
        assert result.defect_class == "Scratch"
        assert result.classifier_confidence == 0.88
        assert result.classifier_probabilities["Scratch"] == 0.88
        assert result.anomaly_score == 52.8

    def test_product_specific_threshold_actually_used(self, dummy_image: Path, valid_profile: ProductProfile):
        """Req 3 & 4: Changing threshold changes Stage 1 verdict appropriately."""
        detector = Stage1AnomalyDetector()

        # Score is 42.0
        # If threshold is 45.0 -> score < threshold -> NORMAL
        with patch.object(detector, "load_model"), patch.object(
            detector.pipeline, "infer_batch"
        ) as mock_infer:
            from defectvision.models.patchcore_engine import AnomalyInferenceResult

            mock_infer.return_value = [
                AnomalyInferenceResult(
                    image_path=str(dummy_image),
                    anomaly_score=42.0,
                    pred_label=False,
                    anomaly_map=np.zeros((256, 256), dtype=np.float32),
                )
            ]

            valid_profile.threshold = 45.0
            res_normal = detector.detect(dummy_image, valid_profile)
            assert res_normal.verdict == InspectionStatus.NORMAL
            assert res_normal.is_defective is False
            assert res_normal.margin == -3.0

            # Change threshold to 40.0 -> score >= threshold -> DEFECTIVE
            valid_profile.threshold = 40.0
            res_defect = detector.detect(dummy_image, valid_profile)
            assert res_defect.verdict == InspectionStatus.DEFECTIVE
            assert res_defect.is_defective is True
            assert res_defect.margin == 2.0

    def test_continuous_anomaly_score_preserved(self, dummy_image: Path, valid_profile: ProductProfile):
        """Req 5: Continuous raw anomaly score is preserved without truncation or binarization."""
        mock_stage1 = MagicMock(spec=Stage1AnomalyDetector)
        test_score = 47.123456789
        mock_stage1.detect.return_value = Stage1Result(
            image_path=str(dummy_image),
            verdict=InspectionStatus.DEFECTIVE,
            is_defective=True,
            anomaly_score=test_score,
            threshold=40.0,
            margin=test_score - 40.0,
        )
        mock_stage2 = MagicMock(spec=Stage2DefectClassifier)
        mock_stage2.loaded_checkpoint_path = "dummy"
        mock_stage2.classify.return_value = Stage2Result(
            predicted_class="UNKNOWN",
            confidence=0.5,
            class_probabilities={"Scratch": 0.1, "Crack": 0.1, "Discoloration": 0.3, "UNKNOWN": 0.5},
        )

        orchestrator = DefectVisionOrchestrator(
            stage1_detector=mock_stage1,
            stage2_classifier=mock_stage2,
        )
        result = orchestrator.inspect(dummy_image, valid_profile)
        assert result.anomaly_score == pytest.approx(test_score, abs=1e-6)

    def test_missing_input_image_fails_clearly(self, valid_profile: ProductProfile):
        """Req 8 & 9: Non-existent image raises FileNotFoundError."""
        orchestrator = DefectVisionOrchestrator()
        with pytest.raises(FileNotFoundError, match="Inspection query image not found"):
            orchestrator.inspect("non_existent_image.png", valid_profile)

    def test_image_size_from_profile_propagated_to_infer_batch(
        self, dummy_image: Path, valid_profile: ProductProfile
    ):
        """Changing ProductProfile.image_size changes the image_size used by infer_batch."""
        detector = Stage1AnomalyDetector()

        with patch.object(detector, "load_model"), patch.object(
            detector.pipeline, "infer_batch"
        ) as mock_infer:
            from defectvision.models.patchcore_engine import AnomalyInferenceResult

            mock_infer.return_value = [
                AnomalyInferenceResult(
                    image_path=str(dummy_image),
                    anomaly_score=30.0,
                    pred_label=False,
                    anomaly_map=np.zeros((256, 256), dtype=np.float32),
                )
            ]

            # Default profile image_size is (256, 256)
            valid_profile.image_size = (256, 256)
            detector.detect(dummy_image, valid_profile)
            call_kwargs_256 = mock_infer.call_args
            assert call_kwargs_256.kwargs.get("image_size") == (256, 256), (
                "infer_batch must receive image_size=(256,256) from the profile"
            )

            # Change profile image_size to (224, 224) and verify it propagates
            valid_profile.image_size = (224, 224)
            detector.detect(dummy_image, valid_profile)
            call_kwargs_224 = mock_infer.call_args
            assert call_kwargs_224.kwargs.get("image_size") == (224, 224), (
                "infer_batch must receive image_size=(224,224) from the updated profile"
            )

            # Change profile image_size to (512, 512) and verify it propagates
            valid_profile.image_size = (512, 512)
            detector.detect(dummy_image, valid_profile)
            call_kwargs_512 = mock_infer.call_args
            assert call_kwargs_512.kwargs.get("image_size") == (512, 512), (
                "infer_batch must receive image_size=(512,512) from the updated profile"
            )

    def test_infer_batch_image_size_override_uses_effective_size(self, dummy_image: Path):
        """PatchCorePipeline.infer_batch with explicit image_size uses that size, not self.image_size."""
        from defectvision.models.patchcore_engine import PatchCorePipeline

        pipeline = PatchCorePipeline(image_size=(256, 256))

        with patch.object(pipeline, "is_fitted", True), \
             patch.object(pipeline, "model", MagicMock()):

            # Mock the internals to capture what PredictDataset gets
            captured_sizes = []
            original_init = type(pipeline).infer_batch

            # We'll patch PredictDataset to capture image_size
            import defectvision.models.patchcore_engine as engine_mod
            original_predict_ds = engine_mod.PredictDataset

            class CapturingPredictDataset:
                def __init__(self, *args, **kwargs):
                    captured_sizes.append(kwargs.get("image_size"))
                    self._ds = original_predict_ds(*args, **kwargs)
                    self.image_filenames = self._ds.image_filenames

                def __len__(self):
                    return 0  # Short-circuit: no actual inference

                def __getitem__(self, idx):
                    return self._ds[idx]

            with patch.object(engine_mod, "PredictDataset", CapturingPredictDataset):
                # Call with override (224, 224) — should NOT use pipeline's (256, 256)
                pipeline.infer_batch([dummy_image], image_size=(224, 224))
                assert captured_sizes[-1] == (224, 224)

                # Call with no override — should fall back to pipeline's (256, 256)
                pipeline.infer_batch([dummy_image])
                assert captured_sizes[-1] == (256, 256)

                # Call with override (512, 512)
                pipeline.infer_batch([dummy_image], image_size=(512, 512))
                assert captured_sizes[-1] == (512, 512)


class TestStage2ClassifierContract:
    """Unit tests for Stage 2 Defect Classifier taxonomy and probability sum."""

    def test_classifier_returns_supported_classes_and_sum_to_one(self):
        """Req 6 & 7: Returns exactly the 4 supported classes, probabilities sum to ~1."""
        if not CLASSIFIER_CKPT.is_file():
            pytest.skip("Classifier checkpoint best_model.pt not available.")

        classifier = Stage2DefectClassifier(checkpoint_path=CLASSIFIER_CKPT)
        if not SAMPLE_DEFECT_IMG.is_file():
            pytest.skip(f"Sample defect image not found: {SAMPLE_DEFECT_IMG}")

        res: Stage2Result = classifier.classify(SAMPLE_DEFECT_IMG)

        # Req 6: Must be in exact taxonomy
        assert res.predicted_class in DEFAULT_DEFECT_CLASSES
        assert set(res.class_probabilities.keys()) == set(DEFAULT_DEFECT_CLASSES)

        # Req 7: Probabilities sum to 1.0 within float tolerance
        prob_sum = sum(res.class_probabilities.values())
        assert prob_sum == pytest.approx(1.0, abs=1e-4)

        # Confidence matches predicted class probability
        assert res.confidence == pytest.approx(res.class_probabilities[res.predicted_class], abs=1e-5)


class TestEndToEndWithRealArtifacts:
    """Req 11: End-to-end integration test with real bottle PatchCore + Calibrated Threshold + Classifier."""

    @pytest.mark.skipif(
        not (PATCHCORE_CKPT.is_file() and CALIB_ARTIFACT.is_file() and CLASSIFIER_CKPT.is_file()),
        reason="Required real artifacts (model.ckpt, calibration json, best_model.pt) not found.",
    )
    def test_real_bottle_normal_inspection(self):
        """End-to-end test on real MVTec bottle normal image."""
        if not SAMPLE_NORMAL_IMG.is_file():
            pytest.skip(f"Normal image not found: {SAMPLE_NORMAL_IMG}")

        profile = ProductProfile.from_calibration_artifact(
            artifact_path=CALIB_ARTIFACT,
            patchcore_model_path=PATCHCORE_CKPT,
        )

        orchestrator = create_orchestrator(
            classifier_checkpoint_path=CLASSIFIER_CKPT,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

        result = orchestrator.inspect(
            image_path=SAMPLE_NORMAL_IMG,
            product_profile=profile,
            save_heatmap=False,
        )

        # Normal verification
        assert result.product_id == "bottle"
        assert result.verdict == "NORMAL"
        assert result.is_defective is False
        assert result.anomaly_score < profile.threshold
        assert result.stage2_executed is False
        assert result.defect_class is None
        assert result.classifier_confidence is None
        assert result.execution_time_ms is not None and result.execution_time_ms > 0

    @pytest.mark.skipif(
        not (PATCHCORE_CKPT.is_file() and CALIB_ARTIFACT.is_file() and CLASSIFIER_CKPT.is_file()),
        reason="Required real artifacts (model.ckpt, calibration json, best_model.pt) not found.",
    )
    def test_real_bottle_defective_inspection(self):
        """End-to-end test on real MVTec bottle broken_large image."""
        if not SAMPLE_DEFECT_IMG.is_file():
            pytest.skip(f"Defect image not found: {SAMPLE_DEFECT_IMG}")

        profile = ProductProfile.from_calibration_artifact(
            artifact_path=CALIB_ARTIFACT,
            patchcore_model_path=PATCHCORE_CKPT,
        )

        orchestrator = create_orchestrator(
            classifier_checkpoint_path=CLASSIFIER_CKPT,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = orchestrator.inspect(
                image_path=SAMPLE_DEFECT_IMG,
                product_profile=profile,
                save_heatmap=True,
                heatmap_dir=tmp_dir,
            )

            # Defective verification
            assert result.product_id == "bottle"
            assert result.verdict == "DEFECTIVE"
            assert result.is_defective is True
            assert result.anomaly_score >= profile.threshold
            assert result.stage2_executed is True
            assert result.defect_class in DEFAULT_DEFECT_CLASSES
            assert result.classifier_confidence is not None and 0.0 < result.classifier_confidence <= 1.0
            assert result.classifier_probabilities is not None
            assert sum(result.classifier_probabilities.values()) == pytest.approx(1.0, abs=1e-4)

            # Heatmap verification
            assert result.heatmap_path is not None
            assert Path(result.heatmap_path).is_file()
