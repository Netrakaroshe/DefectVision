"""Modular PatchCore anomaly detection pipeline wrapping Anomalib 2.6.2."""

from dataclasses import dataclass
from pathlib import Path
import logging
import shutil
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

logger = logging.getLogger(__name__)


@dataclass
class AnomalyInferenceResult:
    """Structured result from PatchCore anomaly inference."""

    image_path: str
    anomaly_score: float
    pred_label: bool
    anomaly_map: np.ndarray  # 2D float array (H, W)
    heatmap_path: str | None = None


class PatchCorePipeline:
    """Modular PatchCore pipeline for reference calibration and anomaly inference."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        image_size: tuple[int, int] = (256, 256),
        accelerator: str = "gpu",
        devices: int = 1,
    ) -> None:
        self.backbone = backbone
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.image_size = image_size
        self.accelerator = accelerator
        self.devices = devices

        self.model: Patchcore | None = None
        self.engine: Engine | None = None
        self.is_fitted: bool = False
        self.memory_bank_shape: tuple[int, ...] | None = None

    def build_feature_memory(
        self,
        reference_images: Sequence[str | Path],
        product_name: str = "product",
        work_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Construct PatchCore normal feature memory from defect-free reference images.

        Args:
            reference_images: List of paths to defect-free normal reference images.
            product_name: Identifier for the product being calibrated.
            work_dir: Working directory for staging and logging. If None, a temporary
                directory is created.

        Returns:
            Dictionary containing memory bank statistics and metadata.
        """
        if not reference_images:
            raise ValueError("Reference images list cannot be empty.")

        ref_paths = [Path(p).resolve() for p in reference_images]
        for p in ref_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Reference image not found: {p}")

        cleanup_tmp = False
        if work_dir is None:
            temp_dir = tempfile.mkdtemp(prefix=f"patchcore_{product_name}_")
            work_path = Path(temp_dir)
            cleanup_tmp = True
        else:
            work_path = Path(work_dir).resolve()
            work_path.mkdir(parents=True, exist_ok=True)

        try:
            # Stage reference images in a dedicated normal directory
            stage_dir = work_path / "stage_ref" / "good"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)

            for p in ref_paths:
                dest = stage_dir / p.name
                shutil.copy2(p, dest)

            # Initialize Anomalib DataModule
            datamodule = Folder(
                name=product_name,
                normal_dir=stage_dir,
                train_batch_size=min(8, len(ref_paths)),
                eval_batch_size=min(8, len(ref_paths)),
                num_workers=0,
            )

            # Initialize Patchcore model and Engine
            self.model = Patchcore(
                backbone=self.backbone,
                coreset_sampling_ratio=self.coreset_sampling_ratio,
                num_neighbors=self.num_neighbors,
            )

            engine_dir = work_path / "engine_runs"
            engine_dir.mkdir(parents=True, exist_ok=True)

            self.engine = Engine(
                default_root_dir=engine_dir,
                accelerator=self.accelerator,
                devices=self.devices,
            )

            # Fit feature memory bank (coreset subsampling)
            self.engine.fit(model=self.model, datamodule=datamodule)
            self.is_fitted = True

            if hasattr(self.model.model, "memory_bank") and self.model.model.memory_bank is not None:
                self.memory_bank_shape = tuple(self.model.model.memory_bank.shape)
            else:
                self.memory_bank_shape = None

            return {
                "product_name": product_name,
                "reference_count": len(ref_paths),
                "backbone": self.backbone,
                "coreset_sampling_ratio": self.coreset_sampling_ratio,
                "num_neighbors": self.num_neighbors,
                "memory_bank_shape": list(self.memory_bank_shape) if self.memory_bank_shape else None,
                "is_fitted": self.is_fitted,
            }

        finally:
            if cleanup_tmp and work_path.exists():
                shutil.rmtree(work_path, ignore_errors=True)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load pre-fitted PatchCore model and memory bank from checkpoint.

        Args:
            checkpoint_path: Path to the PyTorch Lightning .ckpt checkpoint file.
        """
        ckpt_p = Path(checkpoint_path).resolve()
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"PatchCore checkpoint not found: {ckpt_p}")

        device = torch.device(
            "cuda" if self.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
        )
        self.model = Patchcore.load_from_checkpoint(str(ckpt_p), weights_only=False)
        self.model.to(device)
        self.model.eval()

        if hasattr(self.model.model, "memory_bank") and self.model.model.memory_bank is not None:
            self.memory_bank_shape = tuple(self.model.model.memory_bank.shape)
        else:
            self.memory_bank_shape = None

        self.is_fitted = True

    def infer_batch(
        self,
        image_paths: Sequence[str | Path],
        image_size: tuple[int, int] | None = None,
    ) -> list[AnomalyInferenceResult]:
        """Perform PatchCore inference and return raw continuous anomaly scores.

        Args:
            image_paths: Paths to images for inference.
            image_size: Optional per-call image size override for PredictDataset
                preprocessing. When None, falls back to self.image_size (default 256x256).
        """

        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "PatchCore pipeline has not been calibrated or loaded. "
                "Call build_feature_memory or load_checkpoint first."
            )

        resolved_paths = [Path(p).resolve() for p in image_paths]

        for p in resolved_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Inference image not found: {p}")

        if not resolved_paths:
            return []

        effective_image_size = image_size if image_size is not None else self.image_size
        sample_dir = resolved_paths[0].parent

        pred_dataset = PredictDataset(
            path=sample_dir,
            image_size=effective_image_size,
        )
        pred_dataset.image_filenames = resolved_paths

        results: list[AnomalyInferenceResult] = []

        # Bypass Patchcore.forward() because it applies Anomalib's
        # post-processor to the anomaly score.
        patchcore_model = self.model.model
        patchcore_model.eval()

        # Obtain the actual device from the fitted PatchCore model.
        try:
            device = next(patchcore_model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for dataset_index in range(len(pred_dataset)):
            item = pred_dataset[dataset_index]

            image = item.image
            image_path = Path(item.image_path)

            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            image = image.unsqueeze(0).to(device)

            # Apply model preprocessing, but intentionally skip the
            # PatchCore post-processor.
            if self.model.pre_processor:
                image = self.model.pre_processor(image)

            with torch.inference_mode():
                output = patchcore_model(image)

            # PatchcoreModel.forward() directly returns the continuous
            # PatchCore anomaly score and anomaly map.
            raw_score = output.pred_score.squeeze().detach().cpu()
            anomaly_map = output.anomaly_map.squeeze().detach().cpu()

            score = float(raw_score.item())
            amap = anomaly_map.numpy().astype(np.float32)

            results.append(
                AnomalyInferenceResult(
                    image_path=str(image_path),
                    anomaly_score=score,
                    pred_label=False,
                    anomaly_map=amap,
                )
            )

        return results

    @staticmethod
    def render_and_save_heatmap(
        image_path: str | Path,
        anomaly_map: np.ndarray,
        output_path: str | Path,
        score: float | None = None,
        ground_truth: str | None = None,
    ) -> Path:
        """Render side-by-side [Original | Heatmap | Overlay] image and save to disk."""
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is None:
            raise FileNotFoundError(f"Failed to load image for heatmap rendering: {image_path}")

        h, w = orig_bgr.shape[:2]

        # Resize anomaly map to original image resolution
        map_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Min-max normalize map to 0..255 for color mapping
        map_min, map_max = map_resized.min(), map_resized.max()
        if map_max > map_min:
            norm_map = ((map_resized - map_min) / (map_max - map_min) * 255.0).astype(np.uint8)
        else:
            norm_map = np.zeros((h, w), dtype=np.uint8)

        # Apply Jet colormap
        heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # Create blended overlay (60% original image + 40% heatmap)
        overlay_bgr = cv2.addWeighted(orig_bgr, 0.6, heatmap_bgr, 0.4, 0)

        # Annotate overlay with score and status
        header_text = []
        if score is not None:
            header_text.append(f"Score: {score:.4f}")
        if ground_truth is not None:
            header_text.append(f"GT: {ground_truth}")

        if header_text:
            text_str = " | ".join(header_text)
            cv2.rectangle(overlay_bgr, (0, 0), (w, 35), (0, 0, 0), -1)
            color = (0, 0, 255) if (ground_truth and ground_truth.lower() != "normal") else (0, 255, 0)
            cv2.putText(
                overlay_bgr,
                text_str,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        # Combine into side-by-side composite: [Original | Heatmap | Overlay]
        composite = np.hstack([orig_bgr, heatmap_bgr, overlay_bgr])
        cv2.imwrite(str(out_p), composite)

        return out_p
