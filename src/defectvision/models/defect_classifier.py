"""Modular Defect Classifier Architecture and Inference Pipeline for DefectVision.

Provides:
- DefectClassifierModel (ResNet-18 backbone with 4-class industrial defect head)
- DefectClassifierDataset (JSONL dataset loader with consistent RGB conversion and normalization)
- DefectClassifierPipeline (modular inference interface for single and batch predictions)
- Helper utilities for class weight computation and transform composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights, resnet18

DEFAULT_DEFECT_CLASSES = ["Scratch", "Crack", "Discoloration", "UNKNOWN"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DEFAULT_DEFECT_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(DEFAULT_DEFECT_CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DefectClassifierConfig:
    """Configuration for Defect Classifier architecture and training."""

    backbone: str = "resnet18"
    num_classes: int = 4
    class_names: list[str] = field(default_factory=lambda: list(DEFAULT_DEFECT_CLASSES))
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 30
    early_stopping_patience: int = 7
    loss_type: str = "weighted_ce"  # 'weighted_ce' or 'standard_ce'
    use_pretrained: bool = True
    dropout_rate: float = 0.2
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "image_size": list(self.image_size),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "loss_type": self.loss_type,
            "use_pretrained": self.use_pretrained,
            "dropout_rate": self.dropout_rate,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DefectClassifierConfig:
        image_size = tuple(data.get("image_size", (224, 224)))
        return cls(
            backbone=data.get("backbone", "resnet18"),
            num_classes=int(data.get("num_classes", 4)),
            class_names=data.get("class_names", list(DEFAULT_DEFECT_CLASSES)),
            image_size=image_size,
            batch_size=int(data.get("batch_size", 32)),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            weight_decay=float(data.get("weight_decay", 1e-2)),
            max_epochs=int(data.get("max_epochs", 30)),
            early_stopping_patience=int(data.get("early_stopping_patience", 7)),
            loss_type=data.get("loss_type", "weighted_ce"),
            use_pretrained=bool(data.get("use_pretrained", True)),
            dropout_rate=float(data.get("dropout_rate", 0.2)),
            random_seed=int(data.get("random_seed", 42)),
        )


@dataclass
class ClassificationPrediction:
    """Structured result from defect classifier inference."""

    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "confidence": float(self.confidence),
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
        }


def compute_class_weights(
    records_or_labels: Sequence[dict | str],
    class_names: Sequence[str] = DEFAULT_DEFECT_CLASSES,
) -> torch.FloatTensor:
    """Compute balanced inverse-frequency class weights for cross-entropy loss.

    Formula: w_c = N_total / (N_classes * N_c)
    """
    labels = []
    for item in records_or_labels:
        if isinstance(item, dict):
            lbl = item.get("target_label") or item.get("candidate_target_label")
        else:
            lbl = str(item)
        if lbl is not None:
            labels.append(lbl)

    num_classes = len(class_names)
    total_samples = len(labels)
    counts = {cls_name: 0 for cls_name in class_names}
    for lbl in labels:
        if lbl in counts:
            counts[lbl] += 1

    weights = []
    for cls_name in class_names:
        c_count = counts[cls_name]
        if c_count > 0:
            w = total_samples / (num_classes * c_count)
        else:
            w = 1.0
        weights.append(w)

    return torch.FloatTensor(weights)


def get_classifier_transforms(
    image_size: tuple[int, int] = (224, 224),
    is_training: bool = False,
) -> T.Compose:
    """Construct data preprocessing and augmentation pipeline."""
    if is_training:
        return T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DefectClassifierDataset(Dataset):
    """PyTorch Dataset loading industrial defect images with RGB conversion and labels."""

    def __init__(
        self,
        records_or_jsonl: Sequence[dict] | str | Path,
        transform: T.Compose | None = None,
        class_to_idx: dict[str, int] | None = None,
    ):
        if isinstance(records_or_jsonl, (str, Path)):
            p = Path(records_or_jsonl).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Dataset JSONL not found: {p}")
            self.records = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
        else:
            self.records = list(records_or_jsonl)

        self.transform = transform or get_classifier_transforms(is_training=False)
        self.class_to_idx = class_to_idx or CLASS_TO_IDX

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        img_path = Path(rec["source_image_path"])

        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Explicitly convert all images (L / Grayscale / RGB) consistently to 3-channel RGB
        with Image.open(img_path) as pil_img:
            img_rgb = pil_img.convert("RGB")

        tensor_img = self.transform(img_rgb)
        label_str = rec.get("target_label") or rec.get("candidate_target_label")
        label_idx = self.class_to_idx.get(label_str, 3)  # fallback to UNKNOWN if unmapped

        return tensor_img, label_idx, str(img_path).replace("\\", "/")


class DefectClassifierModel(nn.Module):
    """Supervised defect classifier wrapping ResNet-18 with a 4-class industrial head."""

    def __init__(
        self,
        num_classes: int = 4,
        use_pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        # Replace the final linear head with dropout + 4-class linear layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DefectClassifierPipeline:
    """Modular inference pipeline for DefectVision defect classification."""

    def __init__(
        self,
        config: DefectClassifierConfig | None = None,
        model: DefectClassifierModel | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or DefectClassifierConfig()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = DefectClassifierModel(
                num_classes=self.config.num_classes,
                use_pretrained=self.config.use_pretrained,
                dropout_rate=self.config.dropout_rate,
            ).to(self.device)

        self.model.eval()
        self.transform = get_classifier_transforms(
            image_size=self.config.image_size,
            is_training=False,
        )
        self.class_names = list(self.config.class_names)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load trained model weights from checkpoint."""
        p = Path(checkpoint_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        state_dict = torch.load(p, map_location=self.device)
        if "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_image(
        self,
        image_or_path: str | Path | Image.Image,
    ) -> ClassificationPrediction:
        """Predict defect class for a single input image."""
        if isinstance(image_or_path, (str, Path)):
            img_p = Path(image_or_path).resolve()
            if not img_p.is_file():
                raise FileNotFoundError(f"Image not found: {img_p}")
            with Image.open(img_p) as pil_img:
                img_rgb = pil_img.convert("RGB")
            path_str = str(img_p).replace("\\", "/")
        elif isinstance(image_or_path, Image.Image):
            img_rgb = image_or_path.convert("RGB")
            path_str = None
        else:
            raise TypeError(f"Unsupported image type: {type(image_or_path)}")

        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        class_probs = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }

        return ClassificationPrediction(
            predicted_class=pred_class,
            confidence=confidence,
            class_probabilities=class_probs,
            image_path=path_str,
        )

    def predict_batch(
        self,
        images_or_paths: Sequence[str | Path | Image.Image],
        batch_size: int = 32,
    ) -> list[ClassificationPrediction]:
        """Predict defect classes for a collection of images."""
        predictions = []
        for i in range(0, len(images_or_paths), batch_size):
            chunk = images_or_paths[i : i + batch_size]
            tensors = []
            paths = []
            for item in chunk:
                if isinstance(item, (str, Path)):
                    img_p = Path(item).resolve()
                    with Image.open(img_p) as pil_img:
                        img_rgb = pil_img.convert("RGB")
                    paths.append(str(img_p).replace("\\", "/"))
                else:
                    img_rgb = item.convert("RGB")
                    paths.append(None)
                tensors.append(self.transform(img_rgb))

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch_tensor)
                batch_probs = F.softmax(logits, dim=1).cpu().numpy()

            for j in range(len(chunk)):
                probs = batch_probs[j]
                pred_idx = int(np.argmax(probs))
                pred_class = self.class_names[pred_idx]
                conf = float(probs[pred_idx])
                c_probs = {self.class_names[k]: float(probs[k]) for k in range(len(self.class_names))}
                predictions.append(
                    ClassificationPrediction(
                        predicted_class=pred_class,
                        confidence=conf,
                        class_probabilities=c_probs,
                        image_path=paths[j],
                    )
                )

        return predictions
