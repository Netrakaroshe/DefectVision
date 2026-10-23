"""Unit and architectural tests for Phase 4 Step 3 Defect Classifier."""

from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from defectvision.models.defect_classifier import (
    CLASS_TO_IDX,
    DEFAULT_DEFECT_CLASSES,
    IDX_TO_CLASS,
    ClassificationPrediction,
    DefectClassifierConfig,
    DefectClassifierDataset,
    DefectClassifierModel,
    DefectClassifierPipeline,
    compute_class_weights,
    get_classifier_transforms,
)

FROZEN_TRAIN_JSONL = WORKSPACE_ROOT / "experiments" / "defect_classifier_dataset" / "train.jsonl"


def test_four_output_classes_taxonomy():
    """Verify the classifier taxonomy contains exactly the four approved classes."""
    assert len(DEFAULT_DEFECT_CLASSES) == 4
    assert set(DEFAULT_DEFECT_CLASSES) == {"Scratch", "Crack", "Discoloration", "UNKNOWN"}
    assert CLASS_TO_IDX["Scratch"] == 0
    assert CLASS_TO_IDX["Crack"] == 1
    assert CLASS_TO_IDX["Discoloration"] == 2
    assert CLASS_TO_IDX["UNKNOWN"] == 3


def test_model_instantiation_and_output_shape():
    """Verify model forward pass outputs tensor of shape (batch_size, 4)."""
    # use_pretrained=False for unit testing to avoid network download
    model = DefectClassifierModel(num_classes=4, use_pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 4), f"Expected shape (2, 4), got {output.shape}"
    assert not torch.isnan(output).any(), "NaN found in model output logits"


def test_dataset_loading_and_rgb_conversion():
    """Verify dataset converts both grayscale and RGB images to 3-channel (3, 224, 224) tensors."""
    assert FROZEN_TRAIN_JSONL.is_file(), "Frozen train.jsonl must exist"

    dataset = DefectClassifierDataset(FROZEN_TRAIN_JSONL)
    assert len(dataset) == 1505

    # Check first 5 samples
    for i in range(5):
        img_tensor, label_idx, path_str = dataset[i]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {img_tensor.shape}"
        assert 0 <= label_idx <= 3
        assert Path(path_str).is_file()


def test_class_weight_computation_inversely_proportional():
    """Verify class weights penalize minority classes (Crack highest, Discoloration lowest)."""
    # Sample distribution matching the dataset proportions:
    # Crack: 97, Scratch: 293, UNKNOWN: 271, Discoloration: 844
    sample_records = (
        [{"target_label": "Crack"}] * 97
        + [{"target_label": "Scratch"}] * 293
        + [{"target_label": "UNKNOWN"}] * 271
        + [{"target_label": "Discoloration"}] * 844
    )

    weights = compute_class_weights(sample_records, DEFAULT_DEFECT_CLASSES)
    assert len(weights) == 4

    crack_w = weights[CLASS_TO_IDX["Crack"]].item()
    scratch_w = weights[CLASS_TO_IDX["Scratch"]].item()
    unk_w = weights[CLASS_TO_IDX["UNKNOWN"]].item()
    disc_w = weights[CLASS_TO_IDX["Discoloration"]].item()

    # Minority class Crack must receive highest weight
    assert crack_w > scratch_w
    assert crack_w > unk_w
    assert crack_w > disc_w

    # Majority class Discoloration must receive lowest weight
    assert disc_w < scratch_w
    assert disc_w < unk_w
    assert disc_w < crack_w


def test_probability_normalization_and_inference():
    """Verify inference pipeline returns valid probability distributions summing to 1.0."""
    model = DefectClassifierModel(num_classes=4, use_pretrained=False)
    pipeline = DefectClassifierPipeline(model=model, device="cpu")

    # Create dummy PIL image (grayscale to test conversion)
    dummy_gray = Image.new("L", (100, 100), color=128)
    pred = pipeline.predict_image(dummy_gray)

    assert isinstance(pred, ClassificationPrediction)
    assert pred.predicted_class in DEFAULT_DEFECT_CLASSES
    assert 0.0 <= pred.confidence <= 1.0

    probs = pred.class_probabilities
    assert len(probs) == 4
    prob_sum = sum(probs.values())
    assert np.isclose(prob_sum, 1.0, atol=1e-5), f"Probabilities sum to {prob_sum}, expected 1.0"
    for p_val in probs.values():
        assert p_val >= 0.0


def test_checkpoint_save_and_load():
    """Verify round-trip checkpoint save and load preserves model weights."""
    model_orig = DefectClassifierModel(num_classes=4, use_pretrained=False)
    pipeline_orig = DefectClassifierPipeline(model=model_orig, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        torch.save({"model_state_dict": model_orig.state_dict()}, ckpt_path)

        # Create new model and load checkpoint
        model_new = DefectClassifierModel(num_classes=4, use_pretrained=False)
        pipeline_new = DefectClassifierPipeline(model=model_new, device="cpu")
        pipeline_new.load_checkpoint(ckpt_path)

        dummy_img = Image.new("RGB", (224, 224), color=(200, 100, 50))
        pred_orig = pipeline_orig.predict_image(dummy_img)
        pred_new = pipeline_new.predict_image(dummy_img)

        assert pred_orig.predicted_class == pred_new.predicted_class
        assert np.isclose(pred_orig.confidence, pred_new.confidence, atol=1e-5)


def test_config_serialization():
    """Verify DefectClassifierConfig round-trip dictionary serialization."""
    cfg = DefectClassifierConfig(
        backbone="resnet18",
        num_classes=4,
        batch_size=32,
        learning_rate=1e-4,
        max_epochs=30,
        loss_type="weighted_ce",
    )
    d = cfg.to_dict()
    reconstructed = DefectClassifierConfig.from_dict(d)

    assert reconstructed.backbone == cfg.backbone
    assert reconstructed.num_classes == cfg.num_classes
    assert reconstructed.batch_size == cfg.batch_size
    assert reconstructed.learning_rate == cfg.learning_rate
    assert reconstructed.max_epochs == cfg.max_epochs
    assert reconstructed.loss_type == cfg.loss_type
