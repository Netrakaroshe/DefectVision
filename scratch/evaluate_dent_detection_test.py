"""Standalone Test Evaluation for Dent Detection Pilot.

Evaluates the existing trained checkpoint on the held-out test split (618 images).
Zero retraining, zero modification to checkpoint or datasets.
Writes results to D:/DefectVision/experiments/dent_detection_pilot/audit/.
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add dent_detection_pilot/src to path
PILOT_SRC_DIR = Path(r"D:\DefectVision\experiments\dent_detection_pilot\src")
sys.path.append(str(PILOT_SRC_DIR))

from dataset import CR7DetectionDataset, collate_fn, CLASS_NAMES_WITH_BG
from train_eval import build_model, evaluate, generate_markdown_report, NUM_CLASSES, CONF_THRESHOLD, IOU_THRESHOLD

BASE_DIR = Path(r"D:\DefectVision\experiments\dent_detection_pilot")
MANIFEST_PATH = BASE_DIR / "manifests" / "detection_dataset_manifest.jsonl"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_detector.pth"
AUDIT_DIR = BASE_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def main():
    print(f"=== Running Held-Out Test Evaluation for Dent Detection Pilot ===")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Audit output dir: {AUDIT_DIR}")
    print(f"Confidence threshold: {CONF_THRESHOLD}, IoU threshold: {IOU_THRESHOLD}")

    # 1. Load Checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print(f"Loaded checkpoint saved from training epoch {best_ckpt.get('epoch')} with validation mAP@50 = {best_ckpt.get('val_map50', 0):.4f}")

    # 2. Build model and load weights
    model = build_model(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    load_res = model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Model load status: {load_res}")
    model.eval()

    # 3. Load Test dataset
    test_ds = CR7DetectionDataset(MANIFEST_PATH, "test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Test dataset size: {len(test_ds)} images ({len(test_loader)} batches)")

    # 4. Run evaluation
    t0 = time.time()
    test_map50, test_class_metrics = evaluate(model, test_loader, DEVICE, conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD)
    eval_duration = time.time() - t0
    print(f"\nEvaluation finished in {eval_duration:.1f}s ({eval_duration/len(test_ds)*1000:.1f} ms/image)!")
    print(f"Overall 7-Class Test mAP@50: {test_map50*100:.2f}%")

    print("\n--- Detailed Per-Class Results on Test Set ---")
    for name, m in test_class_metrics.items():
        is_dent = "***" if name == "Dents" else ""
        print(f"  {is_dent}{name:12s}{is_dent} : AP@50 = {m['AP50']*100:.2f}% | Prec = {m['precision']*100:.2f}% | Rec = {m['recall']*100:.2f}% | F1 = {m['f1_score']*100:.2f}% | TP={m['true_positives']} | FP={m['false_positives']} | FN={m['false_negatives']} | Total GT={m['total_ground_truths']}")

    # 5. Build summary dictionary
    val_map = best_ckpt.get("val_map50", 0.0)
    best_epoch = best_ckpt.get("epoch", 3)
    val_class_metrics = best_ckpt.get("class_metrics", {})
    val_dent = val_class_metrics.get("Dents", {})
    
    # Checkpoint recorded validation state
    history = best_ckpt.get("history", [
        {
            "epoch": best_epoch,
            "train_loss": 0.0,
            "val_map50": round(val_map, 4),
            "dent_ap50": val_dent.get("AP50", 0.0),
            "dent_precision": val_dent.get("precision", 0.0),
            "dent_recall": val_dent.get("recall", 0.0)
        }
    ])

    summary = {
        "dataset": "CR7-DET",
        "num_classes": NUM_CLASSES - 1,
        "classes": CLASS_NAMES_WITH_BG[1:],
        "evaluation_config": {
            "checkpoint_path": str(CHECKPOINT_PATH).replace("\\", "/"),
            "best_epoch": best_epoch,
            "confidence_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "test_images_count": len(test_ds),
            "eval_duration_seconds": round(eval_duration, 2),
            "device": str(DEVICE)
        },
        "history": history,
        "validation_mAP50": round(val_map, 4),
        "test_mAP50": round(test_map50, 4),
        "test_class_metrics": test_class_metrics,
        "val_class_metrics": val_class_metrics
    }

    # 6. Save JSON
    json_path = AUDIT_DIR / "detection_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote JSON metrics to: {json_path}")

    # 7. Generate markdown report
    generate_markdown_report(summary)
    print(f"Wrote Markdown report to: {AUDIT_DIR / 'dent_detection_report.md'}")


if __name__ == "__main__":
    main()
