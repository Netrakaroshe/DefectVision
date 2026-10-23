"""Controlled Phase 3 Product-Specific Anomaly Threshold Calibration Experiment on MVTecAD bottle."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

# Ensure src and scripts are in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from defectvision.calibration import (
    CalibrationMethod,
    ThresholdCalibrationResult,
    calibrate_threshold,
    classify_score,
)
from run_patchcore_experiment import select_controlled_dataset

# Paths
PHASE2_EXP_DIR = WORKSPACE_ROOT / "experiments" / "patchcore_bottle"
PHASE2_RESULTS_PATH = PHASE2_EXP_DIR / "results" / "inference_results.json"

EXP_DIR = WORKSPACE_ROOT / "experiments" / "patchcore_bottle_calibration"
CONFIG_DIR = EXP_DIR / "config"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
RESULTS_DIR = EXP_DIR / "results"
LOGS_DIR = EXP_DIR / "logs"

SOURCES_ROOT = Path(r"D:\datasets\defect_dataset_workspace\sources")
BOTTLE_ROOT = SOURCES_ROOT / "MVTecAD" / "bottle"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "experiment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("threshold_calibration_experiment")


def partition_calibration_and_evaluation_sets():
    """Deterministically partition images into calibration and held-out evaluation sets.

    Guarantees:
    - 20 train/good reference images for feature memory
    - 20 test/good normal images partitioned into:
        - 10 normal calibration images (even indices)
        - 10 normal held-out evaluation images (odd indices)
    - 63 defective images stratified across categories:
        - broken_large (20 images): 10 calibration, 10 evaluation
        - broken_small (22 images): 11 calibration, 11 evaluation
        - contamination (21 images): 11 calibration, 10 evaluation
        - Total: 32 calibration, 31 evaluation
    - Mutual exclusion across reference, calibration, and evaluation sets.
    """
    ref_paths, normal_val_paths, defect_val_paths = select_controlled_dataset()

    # 1. Normal partition (20 images -> 10 calibration, 10 evaluation)
    normal_calib = [p for i, p in enumerate(normal_val_paths) if i % 2 == 0]
    normal_eval = [p for i, p in enumerate(normal_val_paths) if i % 2 != 0]

    # 2. Defective partition stratified by category
    defects_by_cat: dict[str, list[Path]] = {}
    for p in defect_val_paths:
        cat = p.parent.name
        defects_by_cat.setdefault(cat, []).append(p)

    defect_calib: list[Path] = []
    defect_eval: list[Path] = []

    for cat in sorted(defects_by_cat.keys()):
        cat_paths = sorted(defects_by_cat[cat])
        cat_calib = [p for i, p in enumerate(cat_paths) if i % 2 == 0]
        cat_eval = [p for i, p in enumerate(cat_paths) if i % 2 != 0]
        defect_calib.extend(cat_calib)
        defect_eval.extend(cat_eval)

    return {
        "reference": ref_paths,
        "calibration_normal": normal_calib,
        "calibration_defective": defect_calib,
        "evaluation_normal": normal_eval,
        "evaluation_defective": defect_eval,
    }


def load_actual_continuous_scores(partition: dict[str, list[Path]]) -> dict[str, float]:
    """Load actual PatchCore continuous scores from Phase 2 inference results."""
    if not PHASE2_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Phase 2 inference results not found at: {PHASE2_RESULTS_PATH}. "
            "Phase 2 must be run before Phase 3 calibration."
        )

    with PHASE2_RESULTS_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    score_map: dict[str, float] = {}
    for rec in records:
        norm_p = str(Path(rec["image_path"]).resolve()).replace("\\", "/")
        score_map[norm_p] = float(rec["anomaly_score"])

    # Verify all calibration and evaluation paths have scores
    all_needed_paths = (
        partition["calibration_normal"]
        + partition["calibration_defective"]
        + partition["evaluation_normal"]
        + partition["evaluation_defective"]
    )

    missing = []
    for p in all_needed_paths:
        norm_p = str(p.resolve()).replace("\\", "/")
        if norm_p not in score_map:
            missing.append(norm_p)

    if missing:
        raise ValueError(
            f"Could not find continuous scores for {len(missing)} images in Phase 2 results. "
            f"First missing: {missing[0]}"
        )

    return score_map


def main():
    logger = setup_logging()
    logger.info("=== STARTING PHASE 3 PRODUCT-SPECIFIC THRESHOLD CALIBRATION EXPERIMENT ===")
    logger.info(f"Target Product: MVTecAD bottle")
    logger.info(f"Output Directory: {EXP_DIR}")

    # Ensure output directories exist
    for d in [CONFIG_DIR, ARTIFACTS_DIR, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Partition dataset
    partition = partition_calibration_and_evaluation_sets()

    ref_paths = partition["reference"]
    cal_norm_paths = partition["calibration_normal"]
    cal_defect_paths = partition["calibration_defective"]
    eval_norm_paths = partition["evaluation_normal"]
    eval_defect_paths = partition["evaluation_defective"]

    logger.info(f"Reference images (feature memory): {len(ref_paths)} (Target: 20)")
    logger.info(f"Calibration Normal images:       {len(cal_norm_paths)} (Target: 10)")
    logger.info(f"Calibration Defective images:    {len(cal_defect_paths)} (Target: 32)")
    logger.info(f"Held-Out Evaluation Normal:      {len(eval_norm_paths)} (Target: 10)")
    logger.info(f"Held-Out Evaluation Defective:   {len(eval_defect_paths)} (Target: 31)")

    # Assertions for exact partition counts
    assert len(ref_paths) == 20
    assert len(cal_norm_paths) == 10
    assert len(cal_defect_paths) == 32
    assert len(eval_norm_paths) == 10
    assert len(eval_defect_paths) == 31
    assert len(cal_norm_paths) + len(eval_norm_paths) == 20
    assert len(cal_defect_paths) + len(eval_defect_paths) == 63

    # Leakage and Mutual Exclusion verification
    ref_set = set(p.resolve() for p in ref_paths)
    cal_set = set(p.resolve() for p in (cal_norm_paths + cal_defect_paths))
    eval_set = set(p.resolve() for p in (eval_norm_paths + eval_defect_paths))

    assert not (ref_set & cal_set), "Leakage detected between reference and calibration sets!"
    assert not (ref_set & eval_set), "Leakage detected between reference and evaluation sets!"
    assert not (cal_set & eval_set), "Leakage detected between calibration and evaluation sets!"
    logger.info("Partition integrity verified: strictly zero overlap across reference, calibration, and evaluation.")

    # 2. Save experiment configuration
    config_data = {
        "experiment_name": "patchcore_bottle_threshold_calibration",
        "date": datetime.now(timezone.utc).isoformat(),
        "product": "bottle",
        "dataset": "MVTecAD",
        "partition_strategy": "deterministic_even_odd_stratified",
        "reference_count": len(ref_paths),
        "calibration_counts": {
            "normal": len(cal_norm_paths),
            "defective": len(cal_defect_paths),
            "total": len(cal_norm_paths) + len(cal_defect_paths),
        },
        "evaluation_counts": {
            "normal": len(eval_norm_paths),
            "defective": len(eval_defect_paths),
            "total": len(eval_norm_paths) + len(eval_defect_paths),
        },
        "reference_images": [str(p).replace("\\", "/") for p in ref_paths],
        "calibration_normal_images": [str(p).replace("\\", "/") for p in cal_norm_paths],
        "calibration_defective_images": [str(p).replace("\\", "/") for p in cal_defect_paths],
        "evaluation_normal_images": [str(p).replace("\\", "/") for p in eval_norm_paths],
        "evaluation_defective_images": [str(p).replace("\\", "/") for p in eval_defect_paths],
    }

    config_path = CONFIG_DIR / "calibration_experiment_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved experiment configuration to: {config_path}")

    # 3. Load continuous PatchCore anomaly scores
    logger.info("Loading continuous PatchCore scores from Phase 2 inference results...")
    score_map = load_actual_continuous_scores(partition)

    cal_norm_scores = [score_map[str(p.resolve()).replace("\\", "/")] for p in cal_norm_paths]
    cal_defect_scores = [score_map[str(p.resolve()).replace("\\", "/")] for p in cal_defect_paths]
    eval_norm_scores = [score_map[str(p.resolve()).replace("\\", "/")] for p in eval_norm_paths]
    eval_defect_scores = [score_map[str(p.resolve()).replace("\\", "/")] for p in eval_defect_paths]

    # Save raw calibration scores
    cal_records = []
    for p, s in zip(cal_norm_paths, cal_norm_scores):
        cal_records.append({"path": str(p).replace("\\", "/"), "ground_truth": "normal", "score": s})
    for p, s in zip(cal_defect_paths, cal_defect_scores):
        cal_records.append({"path": str(p).replace("\\", "/"), "ground_truth": "defective", "defect_type": p.parent.name, "score": s})

    with (RESULTS_DIR / "calibration_scores.json").open("w", encoding="utf-8") as f:
        json.dump(cal_records, f, indent=2)

    # 4. Perform Primary Product Threshold Calibration (Supervised Youden's J / Balanced Accuracy)
    logger.info("Executing product-specific threshold calibration via Youden's J optimization...")
    t0 = time.time()
    cal_metadata = {
        "dataset": "MVTecAD",
        "product": "bottle",
        "patchcore_backbone": "wide_resnet50_2",
        "memory_bank_shape": [1638, 1536],
        "experiment": "patchcore_bottle_threshold_calibration",
        "calibration_sample_partition": "10_normal_32_defective",
    }

    calibration_artifact = calibrate_threshold(
        normal_scores=cal_norm_scores,
        defective_scores=cal_defect_scores,
        product_name="bottle",
        method=CalibrationMethod.YOUDEN_J,
        calibration_metadata=cal_metadata,
    )
    cal_time = time.time() - t0
    logger.info(f"Calibration completed in {cal_time*1000:.2f} ms!")
    logger.info(f"Calibrated Threshold: {calibration_artifact.threshold:.6f}")
    logger.info(f"Optimal Plateau: {calibration_artifact.optimal_plateau}")
    logger.info(f"Calibration Youden's J: {calibration_artifact.metrics['youden_j']}")
    logger.info(f"Calibration Balanced Accuracy: {calibration_artifact.metrics['balanced_accuracy']}")

    # Save primary calibration artifact
    artifact_path = ARTIFACTS_DIR / "bottle_threshold_artifact.json"
    calibration_artifact.save_json(artifact_path)
    logger.info(f"Saved primary calibration artifact to: {artifact_path}")

    # Also calibrate Normal-Only baseline artifact to verify normal-only architectural support
    logger.info("Calibrating Normal-Only baseline artifact (3-sigma) for architectural verification...")
    normal_only_artifact = calibrate_threshold(
        normal_scores=cal_norm_scores,
        defective_scores=None,
        product_name="bottle",
        method=CalibrationMethod.NORMAL_K_SIGMA,
        k_sigma=3.0,
        calibration_metadata=cal_metadata,
    )
    normal_only_path = ARTIFACTS_DIR / "bottle_threshold_artifact_normal_only.json"
    normal_only_artifact.save_json(normal_only_path)
    logger.info(f"Saved normal-only calibration artifact to: {normal_only_path}")

    # 5. Evaluate Calibrated Threshold on Held-Out Evaluation Set
    logger.info("Evaluating calibrated threshold on HELD-OUT evaluation set (42 images)...")
    eval_items = []
    for p, s in zip(eval_norm_paths, eval_norm_scores):
        eval_items.append({"path": p, "gt_status": "normal", "defect_type": "good", "score": s})
    for p, s in zip(eval_defect_paths, eval_defect_scores):
        eval_items.append({"path": p, "gt_status": "defective", "defect_type": p.parent.name, "score": s})

    eval_records = []
    tn = fp = fn = tp = 0

    for item in eval_items:
        score = item["score"]
        gt = item["gt_status"]
        pred_status = classify_score(score, calibration_artifact.threshold)
        is_defective = pred_status == "DEFECTIVE"
        margin = score - calibration_artifact.threshold

        is_correct = (pred_status == "NORMAL" and gt == "normal") or (
            pred_status == "DEFECTIVE" and gt == "defective"
        )

        if gt == "normal":
            if pred_status == "NORMAL":
                tn += 1
            else:
                fp += 1
        else:
            if pred_status == "DEFECTIVE":
                tp += 1
            else:
                fn += 1

        eval_records.append({
            "image_path": str(item["path"]).replace("\\", "/"),
            "defect_type": item["defect_type"],
            "ground_truth_status": gt,
            "anomaly_score": round(score, 6),
            "calibrated_threshold": round(calibration_artifact.threshold, 6),
            "predicted_status": pred_status,
            "is_correct": is_correct,
            "margin_from_threshold": round(margin, 6),
        })

    # Save evaluation records
    eval_records_path = RESULTS_DIR / "evaluation_records.json"
    with eval_records_path.open("w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)
    logger.info(f"Saved {len(eval_records)} held-out evaluation records to: {eval_records_path}")

    # 6. Compute Held-Out Performance Metrics
    n_eval_normal = len(eval_norm_scores)
    n_eval_defective = len(eval_defect_scores)
    n_eval_total = n_eval_normal + n_eval_defective

    fpr = fp / n_eval_normal if n_eval_normal > 0 else 0.0
    fnr = fn / n_eval_defective if n_eval_defective > 0 else 0.0
    specificity = tn / n_eval_normal if n_eval_normal > 0 else 0.0
    sensitivity = tp / n_eval_defective if n_eval_defective > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / n_eval_total if n_eval_total > 0 else 0.0
    youden_j = sensitivity + specificity - 1.0

    eval_metrics = {
        "product": "bottle",
        "dataset": "MVTecAD",
        "calibrated_threshold": round(calibration_artifact.threshold, 6),
        "held_out_sample_counts": {
            "normal": n_eval_normal,
            "defective": n_eval_defective,
            "total": n_eval_total,
        },
        "held_out_confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "held_out_rates": {
            "false_positive_rate": round(fpr, 6),
            "false_negative_rate": round(fnr, 6),
            "specificity": round(specificity, 6),
            "sensitivity_recall": round(sensitivity, 6),
            "precision": round(precision, 6),
            "balanced_accuracy": round(balanced_acc, 6),
            "f1_score": round(f1, 6),
            "accuracy": round(accuracy, 6),
            "youden_j": round(youden_j, 6),
        },
        "status": "PASS" if balanced_acc >= 0.99 else "FAIL",
    }

    metrics_path = RESULTS_DIR / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    # 7. Print Comprehensive Summary Report
    print("\n" + "=" * 78)
    print("PHASE 3 PRODUCT-SPECIFIC THRESHOLD CALIBRATION REPORT: MVTecAD bottle")
    print("=" * 78)
    print(f"Product:                          bottle")
    print(f"Calibration Method:               {calibration_artifact.calibration_method} (Objective: {calibration_artifact.objective_metric_name})")
    print(f"Optimal Plateau:                  [{calibration_artifact.optimal_plateau[0]:.6f}, {calibration_artifact.optimal_plateau[1]:.6f}]")
    print(f"Selected Calibrated Threshold:    {calibration_artifact.threshold:.6f}")
    print("-" * 78)
    print(f"CALIBRATION DATASET ({len(cal_norm_scores) + len(cal_defect_scores)} images):")
    print(f"  Normal Calibration Count:       {len(cal_norm_scores)}")
    print(f"  Normal Scores:                  min={min(cal_norm_scores):.4f}, max={max(cal_norm_scores):.4f}, mean={np.mean(cal_norm_scores):.4f}")
    print(f"  Defective Calibration Count:    {len(cal_defect_scores)}")
    print(f"  Defective Scores:               min={min(cal_defect_scores):.4f}, max={max(cal_defect_scores):.4f}, mean={np.mean(cal_defect_scores):.4f}")
    print(f"  Calibration Confusion Matrix:   TN={calibration_artifact.confusion_matrix['tn']}, FP={calibration_artifact.confusion_matrix['fp']}, FN={calibration_artifact.confusion_matrix['fn']}, TP={calibration_artifact.confusion_matrix['tp']}")
    print(f"  Calibration Balanced Accuracy:  {calibration_artifact.metrics['balanced_accuracy']:.6f}")
    print("-" * 78)
    print(f"HELD-OUT EVALUATION DATASET ({n_eval_total} images):")
    print(f"  Held-Out Normal Count:          {n_eval_normal}")
    print(f"  Held-Out Normal Scores:         min={min(eval_norm_scores):.4f}, max={max(eval_norm_scores):.4f}, mean={np.mean(eval_norm_scores):.4f}")
    print(f"  Held-Out Defective Count:       {n_eval_defective}")
    print(f"  Held-Out Defective Scores:      min={min(eval_defect_scores):.4f}, max={max(eval_defect_scores):.4f}, mean={np.mean(eval_defect_scores):.4f}")
    print("-" * 78)
    print("HELD-OUT EVALUATION CONFUSION MATRIX:")
    print(f"                Predicted NORMAL    Predicted DEFECTIVE")
    print(f"  Actual NORMAL        TN = {tn:<15} FP = {fp}")
    print(f"  Actual DEFECTIVE     FN = {fn:<15} TP = {tp}")
    print("-" * 78)
    print("HELD-OUT EVALUATION METRICS:")
    print(f"  False Positive Rate (FPR):      {fpr:.6f} ({fpr*100:.2f}%)")
    print(f"  False Negative Rate (FNR):      {fnr:.6f} ({fnr*100:.2f}%)")
    print(f"  Specificity (TNR):              {specificity:.6f} ({specificity*100:.2f}%)")
    print(f"  Sensitivity / Recall (TPR):     {sensitivity:.6f} ({sensitivity*100:.2f}%)")
    print(f"  Precision:                      {precision:.6f} ({precision*100:.2f}%)")
    print(f"  Balanced Accuracy:              {balanced_acc:.6f} ({balanced_acc*100:.2f}%)")
    print(f"  Overall Accuracy:               {accuracy:.6f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score:                       {f1:.6f}")
    print(f"  Youden's J Statistic:           {youden_j:.6f}")
    print("-" * 78)
    print("NORMAL-ONLY BASELINE ARTIFACT COMPARISON:")
    print(f"  Normal-Only Method:             {normal_only_artifact.calibration_method} (k=3.0)")
    print(f"  Normal-Only Threshold:          {normal_only_artifact.threshold:.6f}")
    print(f"  Normal-Only Is Normal Only:     {normal_only_artifact.is_normal_only}")
    print(f"  Normal-Only Warning:            {normal_only_artifact.warnings[1] if len(normal_only_artifact.warnings) > 1 else normal_only_artifact.warnings[0]}")
    print("-" * 78)
    print(f"PRIMARY EXPERIMENT STATUS:        {eval_metrics['status']}")
    print(f"Artifact Location:                {artifact_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
