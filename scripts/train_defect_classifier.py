"""Phase 4 Step 4 - Train the ResNet-18 Defect Classifier.

Approved configuration:
  - Architecture : ResNet-18 (ImageNet-1K pretrained), 4-class head, Dropout=0.2
  - Input        : 224x224 RGB, ImageNet normalization
  - Batch size   : 32
  - Optimizer    : AdamW, lr=1e-4, weight_decay=0.01
  - Scheduler    : CosineAnnealingLR, T_max=30, eta_min=1e-6
  - Max epochs   : 30
  - Early stop   : patience=7 on val macro-F1 (higher-is-better)
  - Checkpoint   : best val macro-F1 -> experiments/defect_classifier_run1/best_model.pt
  - Seed         : 42
  - Loss         : weighted cross-entropy (inverse-frequency weights from train split)
  - Classes      : Scratch | Crack | Discoloration | UNKNOWN  (in this exact order)
  - Test set     : held-out; evaluated exactly ONCE after training completes

Usage:
  python scripts/train_defect_classifier.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defectvision.models.defect_classifier import (
    DEFAULT_DEFECT_CLASSES,
    CLASS_TO_IDX,
    DefectClassifierConfig,
    DefectClassifierDataset,
    DefectClassifierModel,
    compute_class_weights,
    get_classifier_transforms,
)

DATASET_DIR = PROJECT_ROOT / "experiments" / "defect_classifier_dataset"
TRAIN_JSONL = DATASET_DIR / "train.jsonl"
VAL_JSONL   = DATASET_DIR / "val.jsonl"
TEST_JSONL  = DATASET_DIR / "test.jsonl"

OUTPUT_DIR  = PROJECT_ROOT / "experiments" / "defect_classifier_run1"
CKPT_PATH   = OUTPUT_DIR / "best_model.pt"
LOG_PATH    = OUTPUT_DIR / "training_log.jsonl"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.json"
MD_PATH     = OUTPUT_DIR / "evaluation_report.md"

SEED         = 42
BATCH_SIZE   = 32
LR           = 1e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS   = 30
PATIENCE     = 7
T_MAX        = 30
ETA_MIN      = 1e-6
DROPOUT      = 0.2
IMAGE_SIZE   = (224, 224)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train")


def load_jsonl(path: Path) -> list:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_preds  = []

    if is_train:
        for tensors, labels, _paths in loader:
            tensors = tensors.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(tensors)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tensors.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    else:
        with torch.inference_mode():
            for tensors, labels, _paths in loader:
                tensors = tensors.to(device, non_blocking=True)
                labels  = labels.to(device, non_blocking=True)
                logits  = model(tensors)
                loss    = criterion(logits, labels)
                total_loss += loss.item() * tensors.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

    n         = len(all_labels)
    mean_loss = total_loss / n if n > 0 else float("nan")
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return mean_loss, macro_f1, all_labels, all_preds


def main() -> None:
    log = setup_logging()
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    log.info("Loading dataset splits ...")
    train_records = load_jsonl(TRAIN_JSONL)
    val_records   = load_jsonl(VAL_JSONL)
    test_records  = load_jsonl(TEST_JSONL)
    log.info(f"  train={len(train_records)}  val={len(val_records)}  test={len(test_records)}")

    train_tf = get_classifier_transforms(image_size=IMAGE_SIZE, is_training=True)
    eval_tf  = get_classifier_transforms(image_size=IMAGE_SIZE, is_training=False)

    train_ds = DefectClassifierDataset(train_records, transform=train_tf, class_to_idx=CLASS_TO_IDX)
    val_ds   = DefectClassifierDataset(val_records,   transform=eval_tf,  class_to_idx=CLASS_TO_IDX)
    test_ds  = DefectClassifierDataset(test_records,  transform=eval_tf,  class_to_idx=CLASS_TO_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    class_weights = compute_class_weights(train_records, DEFAULT_DEFECT_CLASSES).to(device)
    log.info("Class weights (from train split):")
    for cls, w in zip(DEFAULT_DEFECT_CLASSES, class_weights.tolist()):
        log.info(f"  {cls:<16s}: {w:.4f}")

    model = DefectClassifierModel(num_classes=4, use_pretrained=True, dropout_rate=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    log.info("=" * 70)
    log.info("Training started")
    log.info("=" * 70)

    best_val_f1   = -1.0
    epochs_no_imp = 0
    epoch_logs    = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_f1, val_labels, val_preds = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        epoch_entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_f1":   round(train_f1,   6),
            "val_loss":   round(val_loss,    6),
            "val_f1":     round(val_f1,      6),
            "lr":         round(scheduler.get_last_lr()[0], 8),
            "elapsed_s":  round(elapsed, 2),
        }
        epoch_logs.append(epoch_entry)

        improved = val_f1 > best_val_f1
        marker   = " <-- BEST" if improved else ""
        log.info(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_f1={train_f1:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}s]{marker}"
        )

        if improved:
            best_val_f1   = val_f1
            epochs_no_imp = 0
            torch.save(
                {
                    "epoch":               epoch,
                    "val_f1":              val_f1,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config":              DefectClassifierConfig(dropout_rate=DROPOUT, random_seed=SEED).to_dict(),
                },
                CKPT_PATH,
            )
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch} (no val_f1 improvement for {PATIENCE} epochs).")
                break

    with LOG_PATH.open("w", encoding="utf-8") as fh:
        for entry in epoch_logs:
            fh.write(json.dumps(entry) + "\n")

    log.info(f"Training complete. Best val macro-F1 = {best_val_f1:.4f}")

    # --- single test evaluation ---
    log.info("=" * 70)
    log.info(f"Loading best checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt["epoch"]
    log.info(f"  Saved from epoch {best_epoch}, val_f1={ckpt['val_f1']:.4f}")

    log.info("Evaluating on test set (one-time) ...")
    test_loss, test_macro_f1, test_labels, test_preds = run_epoch(
        model, test_loader, criterion, None, device, is_train=False
    )

    per_class_f1 = f1_score(
        test_labels, test_preds,
        average=None, labels=list(range(4)), zero_division=0,
    )
    cls_report_str = classification_report(
        test_labels, test_preds,
        target_names=DEFAULT_DEFECT_CLASSES,
        labels=list(range(4)), digits=4, zero_division=0,
    )
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(4)))
    correct  = sum(p == t for p, t in zip(test_preds, test_labels))
    accuracy = correct / len(test_labels) if test_labels else 0.0

    log.info("=" * 70)
    log.info("TEST RESULTS")
    log.info("=" * 70)
    log.info(f"  Test samples  : {len(test_labels)}")
    log.info(f"  Test loss     : {test_loss:.4f}")
    log.info(f"  Accuracy      : {accuracy:.4f}")
    log.info(f"  Macro F1      : {test_macro_f1:.4f}")
    log.info("\nPer-class F1:")
    for cls, f1 in zip(DEFAULT_DEFECT_CLASSES, per_class_f1):
        log.info(f"  {cls:<16s}: {f1:.4f}")
    log.info(f"\nClassification report:\n{cls_report_str}")
    log.info("Confusion matrix (rows=true, cols=pred):")
    log.info(f"  Labels: {DEFAULT_DEFECT_CLASSES}")
    for i, row in enumerate(cm):
        log.info(f"  {DEFAULT_DEFECT_CLASSES[i]:<16s}: {row.tolist()}")

    report = {
        "phase":                   "Phase 4 Step 4",
        "best_epoch":              best_epoch,
        "best_val_macro_f1":       round(best_val_f1,      6),
        "test_loss":               round(test_loss,         6),
        "test_accuracy":           round(accuracy,          6),
        "test_macro_f1":           round(test_macro_f1,     6),
        "test_samples":            len(test_labels),
        "class_order":             DEFAULT_DEFECT_CLASSES,
        "per_class_f1": {
            cls: round(float(f1), 6)
            for cls, f1 in zip(DEFAULT_DEFECT_CLASSES, per_class_f1)
        },
        "confusion_matrix":        cm.tolist(),
        "confusion_matrix_labels": DEFAULT_DEFECT_CLASSES,
        "epoch_logs":              epoch_logs,
        "checkpoint_path":         str(CKPT_PATH),
        "class_weights": {
            cls: round(float(w), 6)
            for cls, w in zip(DEFAULT_DEFECT_CLASSES, class_weights.cpu().tolist())
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(f"Report saved -> {REPORT_PATH}")

    md_lines = [
        "# Phase 4 Step 4 -- Defect Classifier Training Report",
        "",
        "## Configuration",
        f"- Architecture: ResNet-18 (ImageNet-1K pretrained), Dropout={DROPOUT}",
        f"- Classes: {', '.join(DEFAULT_DEFECT_CLASSES)}",
        f"- Seed: {SEED}  |  Batch size: {BATCH_SIZE}",
        f"- Optimizer: AdamW (lr={LR}, weight_decay={WEIGHT_DECAY})",
        f"- Scheduler: CosineAnnealingLR (T_max={T_MAX}, eta_min={ETA_MIN})",
        f"- Max epochs: {MAX_EPOCHS}  |  Early-stop patience: {PATIENCE}",
        "",
        "## Class Weights (train split)",
        "| Class | Weight |",
        "|-------|--------|",
    ]
    for cls, w in zip(DEFAULT_DEFECT_CLASSES, class_weights.cpu().tolist()):
        md_lines.append(f"| {cls} | {w:.4f} |")

    md_lines += [
        "",
        "## Training History",
        "| Epoch | Train Loss | Train F1 | Val Loss | Val F1 | LR |",
        "|-------|-----------|----------|----------|--------|-----|",
    ]
    for e in epoch_logs:
        md_lines.append(
            f"| {e['epoch']:02d} | {e['train_loss']:.4f} | {e['train_f1']:.4f} "
            f"| {e['val_loss']:.4f} | {e['val_f1']:.4f} | {e['lr']:.2e} |"
        )

    md_lines += [
        "",
        "## Test Set Evaluation (one-time hold-out)",
        f"- **Best epoch**: {best_epoch}",
        f"- **Test samples**: {len(test_labels)}",
        f"- **Test loss**: {test_loss:.4f}",
        f"- **Accuracy**: {accuracy:.4f}",
        f"- **Macro F1**: {test_macro_f1:.4f}",
        "",
        "### Per-Class F1",
        "| Class | F1 |",
        "|-------|----|",
    ]
    for cls, f1 in zip(DEFAULT_DEFECT_CLASSES, per_class_f1):
        md_lines.append(f"| {cls} | {f1:.4f} |")

    md_lines += [
        "",
        "### Confusion Matrix (rows=true, cols=predicted)",
        "| True \\ Pred | " + " | ".join(DEFAULT_DEFECT_CLASSES) + " |",
        "|" + "---|" * (len(DEFAULT_DEFECT_CLASSES) + 1),
    ]
    for i, row in enumerate(cm):
        md_lines.append(
            "| **" + DEFAULT_DEFECT_CLASSES[i] + "** | " +
            " | ".join(str(v) for v in row.tolist()) + " |"
        )

    md_lines += [
        "",
        "### Full Classification Report",
        "```",
        cls_report_str.strip(),
        "```",
    ]

    MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    log.info(f"Markdown report saved -> {MD_PATH}")
    log.info("Phase 4 Step 4 complete.")


if __name__ == "__main__":
    main()
