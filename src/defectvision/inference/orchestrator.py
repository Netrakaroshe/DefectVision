"""Core Two-Stage Inference Orchestrator for DefectVision.

Connects Stage 1 (PatchCore Anomaly Detection + Product Calibration)
and Stage 2 (Supervised ResNet-18 Defect Classification).

Workflow:
1. Validate inputs (image existence, product profile threshold and model paths).
2. Execute Stage 1 Anomaly Detection:
   - Compute continuous PatchCore anomaly score and localization heatmap.
   - Compare score against the product-specific threshold.
   - If NORMAL: stop pipeline immediately, return clean NORMAL result without running Stage 2.
3. If DEFECTIVE:
   - Execute Stage 2 Defect Classification on the full image.
   - Classify into Scratch, Crack, Discoloration, or UNKNOWN.
   - Return combined DEFECTIVE result containing classification and probabilities.
4. Maintain separation of concerns, explicit model reuse, and strict failure handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

from defectvision.calibration import InspectionStatus
from defectvision.inference.product_profile import ProductProfile
from defectvision.inference.result import InspectionResult
from defectvision.inference.stage1_anomaly import Stage1AnomalyDetector, Stage1Result
from defectvision.inference.stage2_classifier import Stage2DefectClassifier, Stage2Result

logger = logging.getLogger(__name__)


class DefectVisionOrchestrator:
    """Production inference orchestrator connecting PatchCore and Defect Classifier."""

    def __init__(
        self,
        classifier_checkpoint_path: str | Path | None = None,
        stage1_detector: Stage1AnomalyDetector | None = None,
        stage2_classifier: Stage2DefectClassifier | None = None,
        accelerator: str = "gpu",
        devices: int = 1,
    ) -> None:
        self.accelerator = accelerator
        self.devices = devices

        # Stage 1 Detector (reusable across inferences)
        self.stage1 = stage1_detector or Stage1AnomalyDetector(
            accelerator=self.accelerator,
            devices=self.devices,
        )

        # Stage 2 Classifier (lazy loaded or pre-loaded from checkpoint)
        if stage2_classifier is not None:
            self.stage2 = stage2_classifier
        elif classifier_checkpoint_path is not None:
            self.stage2 = Stage2DefectClassifier(checkpoint_path=classifier_checkpoint_path)
        else:
            self.stage2 = Stage2DefectClassifier()

        self.default_classifier_checkpoint = (
            str(Path(classifier_checkpoint_path).resolve()).replace("\\", "/")
            if classifier_checkpoint_path
            else None
        )

    def inspect(
        self,
        image_path: str | Path,
        product_profile: ProductProfile,
        save_heatmap: bool = False,
        heatmap_dir: str | Path | None = None,
        classifier_checkpoint_path: str | Path | None = None,
    ) -> InspectionResult:
        """Execute two-stage inspection on a product image.

        Args:
            image_path: Path to the query image to inspect.
            product_profile: Calibrated product profile holding model path and threshold.
            save_heatmap: Whether to render and save the side-by-side localization heatmap.
            heatmap_dir: Directory where the heatmap will be saved.
            classifier_checkpoint_path: Optional override for the Stage 2 checkpoint path.

        Returns:
            InspectionResult containing Stage 1 verdict, scores, and Stage 2 defect info (if defective).
        """
        t0 = time.perf_counter()

        # 1. Input Validation
        img_p = Path(image_path).resolve()
        if not img_p.is_file():
            raise FileNotFoundError(f"Inspection query image not found: {img_p}")

        if not isinstance(product_profile, ProductProfile):
            raise TypeError(
                f"product_profile must be an instance of ProductProfile, got {type(product_profile)}"
            )

        product_profile.validate(check_paths_exist=True)

        # 2. Stage 1: PatchCore Anomaly Inference
        logger.info(
            f"Stage 1: Running anomaly detection for product '{product_profile.product_id}' on {img_p.name}"
        )
        stage1_result: Stage1Result = self.stage1.detect(
            image_path=img_p,
            product_profile=product_profile,
            save_heatmap=save_heatmap,
            heatmap_dir=heatmap_dir,
        )

        verdict_str = stage1_result.verdict.value
        is_defective = stage1_result.is_defective

        # Base inspection metadata
        meta: dict[str, Any] = {
            "patchcore_model_path": product_profile.patchcore_model_path,
            "calibration_method": product_profile.calibration_method,
            "calibration_timestamp": product_profile.calibration_timestamp,
            "profile_metadata": product_profile.metadata,
        }

        # 3. Decision Branch: NORMAL
        if not is_defective:
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"Inspection NORMAL: score={stage1_result.anomaly_score:.4f} < "
                f"threshold={stage1_result.threshold:.4f} (margin={stage1_result.margin:.4f}). "
                f"Stage 2 skipped."
            )
            return InspectionResult(
                product_id=product_profile.product_id,
                image_path=str(img_p).replace("\\", "/"),
                verdict=InspectionStatus.NORMAL.value,
                is_defective=False,
                anomaly_score=stage1_result.anomaly_score,
                anomaly_threshold=stage1_result.threshold,
                margin=stage1_result.margin,
                stage2_executed=False,
                defect_class=None,
                classifier_confidence=None,
                classifier_probabilities=None,
                heatmap_path=stage1_result.heatmap_path,
                execution_time_ms=t_elapsed,
                metadata=meta,
            )

        # 4. Decision Branch: DEFECTIVE -> Execute Stage 2
        logger.info(
            f"Inspection DEFECTIVE: score={stage1_result.anomaly_score:.4f} >= "
            f"threshold={stage1_result.threshold:.4f}. Executing Stage 2 Defect Classification..."
        )

        ckpt_to_use = (
            classifier_checkpoint_path
            or self.default_classifier_checkpoint
            or product_profile.metadata.get("classifier_checkpoint_path")
        )

        if self.stage2.loaded_checkpoint_path is None:
            if not ckpt_to_use:
                raise RuntimeError(
                    "Stage 2 classifier is not loaded and no classifier_checkpoint_path was provided."
                )
            self.stage2.load_checkpoint(ckpt_to_use)
        elif ckpt_to_use and Path(ckpt_to_use).resolve() != Path(self.stage2.loaded_checkpoint_path).resolve():
            self.stage2.load_checkpoint(ckpt_to_use)

        stage2_result: Stage2Result = self.stage2.classify(img_p)

        meta["classifier_checkpoint"] = self.stage2.loaded_checkpoint_path
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        return InspectionResult(
            product_id=product_profile.product_id,
            image_path=str(img_p).replace("\\", "/"),
            verdict=InspectionStatus.DEFECTIVE.value,
            is_defective=True,
            anomaly_score=stage1_result.anomaly_score,
            anomaly_threshold=stage1_result.threshold,
            margin=stage1_result.margin,
            stage2_executed=True,
            defect_class=stage2_result.predicted_class,
            classifier_confidence=stage2_result.confidence,
            classifier_probabilities=stage2_result.class_probabilities,
            heatmap_path=stage1_result.heatmap_path,
            execution_time_ms=t_elapsed,
            metadata=meta,
        )


def create_orchestrator(
    classifier_checkpoint_path: str | Path | None = None,
    accelerator: str = "gpu",
    devices: int = 1,
) -> DefectVisionOrchestrator:
    """Factory function to instantiate the DefectVision orchestrator."""
    return DefectVisionOrchestrator(
        classifier_checkpoint_path=classifier_checkpoint_path,
        accelerator=accelerator,
        devices=devices,
    )
