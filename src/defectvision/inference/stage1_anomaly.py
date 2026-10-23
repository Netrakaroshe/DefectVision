"""Stage 1: PatchCore Anomaly Detection and Calibrated Threshold Evaluation.

Responsibilities:
1. Load or reuse a fitted PatchCore model / memory bank for a target product.
2. Extract continuous patch embeddings and compute continuous raw anomaly score + anomaly map.
3. Evaluate the continuous score against the product-specific calibrated threshold.
4. Classify outcome as NORMAL or DEFECTIVE strictly according to:
     anomaly_score < threshold  -> NORMAL
     anomaly_score >= threshold -> DEFECTIVE
5. Optionally render and save the side-by-side localization heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from defectvision.calibration import InspectionStatus, classify_score
from defectvision.inference.product_profile import ProductProfile
from defectvision.models.patchcore_engine import AnomalyInferenceResult, PatchCorePipeline


@dataclass
class Stage1Result:
    """Result of Stage 1 anomaly inference."""

    image_path: str
    verdict: InspectionStatus
    is_defective: bool
    anomaly_score: float
    threshold: float
    margin: float
    anomaly_map: np.ndarray | None = None
    heatmap_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "verdict": self.verdict.value,
            "is_defective": self.is_defective,
            "anomaly_score": float(self.anomaly_score),
            "threshold": float(self.threshold),
            "margin": float(self.margin),
            "heatmap_path": self.heatmap_path,
        }


class Stage1AnomalyDetector:
    """Stage 1 Anomaly Detector wrapping PatchCorePipeline and calibrated threshold."""

    def __init__(
        self,
        pipeline: PatchCorePipeline | None = None,
        accelerator: str = "gpu",
        devices: int = 1,
    ) -> None:
        self.accelerator = accelerator
        self.devices = devices
        self.pipeline = pipeline or PatchCorePipeline(
            accelerator=self.accelerator,
            devices=self.devices,
        )
        self.loaded_checkpoint_path: str | None = None

    def load_model(self, checkpoint_path: str | Path) -> None:
        """Load pre-fitted PatchCore model weights and memory bank from disk."""
        resolved = str(Path(checkpoint_path).resolve())
        if self.loaded_checkpoint_path == resolved and self.pipeline.is_fitted:
            return  # Already loaded

        self.pipeline.load_checkpoint(resolved)
        self.loaded_checkpoint_path = resolved

    def detect(
        self,
        image_path: str | Path,
        product_profile: ProductProfile,
        save_heatmap: bool = False,
        heatmap_dir: str | Path | None = None,
    ) -> Stage1Result:
        """Run Stage 1 anomaly detection on a single image against a product profile."""
        img_p = Path(image_path).resolve()
        if not img_p.is_file():
            raise FileNotFoundError(f"Input image not found: {img_p}")

        product_profile.validate(check_paths_exist=True)

        # Ensure PatchCore model is loaded for this product
        self.load_model(product_profile.patchcore_model_path)

        # Run PatchCore raw continuous inference using product-specific image size
        inference_results: list[AnomalyInferenceResult] = self.pipeline.infer_batch(
            [img_p], image_size=product_profile.image_size,
        )
        if not inference_results:
            raise RuntimeError(f"PatchCore failed to produce inference output for {img_p}")

        raw_result = inference_results[0]
        raw_score = float(raw_result.anomaly_score)
        threshold = float(product_profile.threshold)

        # Evaluate against product-specific threshold
        status_str = classify_score(raw_score, threshold)
        verdict = InspectionStatus(status_str)
        is_defective = verdict == InspectionStatus.DEFECTIVE
        margin = raw_score - threshold

        # Render heatmap if requested
        saved_heatmap_path: str | None = None
        if save_heatmap and raw_result.anomaly_map is not None:
            out_dir = Path(heatmap_dir).resolve() if heatmap_dir else img_p.parent / "heatmaps"
            out_dir.mkdir(parents=True, exist_ok=True)
            heatmap_file = out_dir / f"{img_p.stem}_heatmap_{verdict.value.lower()}.png"
            rendered = PatchCorePipeline.render_and_save_heatmap(
                image_path=img_p,
                anomaly_map=raw_result.anomaly_map,
                output_path=heatmap_file,
                score=raw_score,
                ground_truth=verdict.value,
            )
            saved_heatmap_path = str(rendered).replace("\\", "/")

        return Stage1Result(
            image_path=str(img_p).replace("\\", "/"),
            verdict=verdict,
            is_defective=is_defective,
            anomaly_score=raw_score,
            threshold=threshold,
            margin=margin,
            anomaly_map=raw_result.anomaly_map,
            heatmap_path=saved_heatmap_path,
        )
