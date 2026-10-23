"""Controlled Phase 2 PatchCore Anomaly Inference Experiment on MVTecAD bottle."""

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure src is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from defectvision.models.patchcore_engine import PatchCorePipeline

# Experiment paths
EXP_DIR = WORKSPACE_ROOT / "experiments" / "patchcore_bottle"
CONFIG_DIR = EXP_DIR / "config"
RESULTS_DIR = EXP_DIR / "results"
HEATMAPS_DIR = EXP_DIR / "heatmaps"
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
    return logging.getLogger("patchcore_experiment")


def select_controlled_dataset():
    """Select exactly 20 reference, 20 normal validation, and 63 defective validation images."""
    train_good_dir = BOTTLE_ROOT / "train" / "good"
    test_good_dir = BOTTLE_ROOT / "test" / "good"
    test_dir = BOTTLE_ROOT / "test"

    # 1. Exactly 20 reference images from train/good
    all_train_good = sorted([p for p in train_good_dir.glob("*.png") if p.is_file()])
    reference_paths = all_train_good[:20]

    # 2. Exactly 20 normal validation images from test/good
    all_test_good = sorted([p for p in test_good_dir.glob("*.png") if p.is_file()])
    normal_val_paths = all_test_good[:20]

    # 3. Exactly 63 defective validation images from test/<defect>
    defect_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"])
    defective_val_paths = []
    for d in defect_dirs:
        imgs = sorted([p for p in d.glob("*.png") if p.is_file()])
        defective_val_paths.extend(imgs)

    return reference_paths, normal_val_paths, defective_val_paths


def compute_statistics(scores: list[float]) -> dict[str, float]:
    arr = np.array(scores, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)) if len(arr) else 0.0,
        "max": float(np.max(arr)) if len(arr) else 0.0,
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
    }


def main():
    logger = setup_logging()
    logger.info("=== STARTING PHASE 2 PATCHCORE CONTROLLED EXPERIMENT ===")
    logger.info(f"Workspace: {WORKSPACE_ROOT}")
    logger.info(f"Target: MVTecAD bottle")

    # Ensure output directories exist
    for d in [CONFIG_DIR, RESULTS_DIR, HEATMAPS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Dataset selection
    ref_paths, normal_val_paths, defect_val_paths = select_controlled_dataset()

    logger.info(f"Selected reference images: {len(ref_paths)} (Target: 20)")
    logger.info(f"Selected normal validation images: {len(normal_val_paths)} (Target: 20)")
    logger.info(f"Selected defective validation images: {len(defect_val_paths)} (Target: 63)")

    # Assertions for exact requirements
    assert len(ref_paths) == 20, f"Expected exactly 20 reference images, got {len(ref_paths)}"
    assert len(normal_val_paths) == 20, f"Expected exactly 20 normal validation images, got {len(normal_val_paths)}"
    assert len(defect_val_paths) == 63, f"Expected exactly 63 defective validation images, got {len(defect_val_paths)}"

    # Check for leakage
    ref_set = set(p.resolve() for p in ref_paths)
    norm_val_set = set(p.resolve() for p in normal_val_paths)
    defect_val_set = set(p.resolve() for p in defect_val_paths)

    assert not (ref_set & norm_val_set), "Leakage detected between reference and normal validation sets!"
    assert not (ref_set & defect_val_set), "Leakage detected between reference and defective validation sets!"
    assert not (norm_val_set & defect_val_set), "Overlap detected between normal and defective validation sets!"

    logger.info("Dataset selection verified: exactly 20 reference, 20 normal val, 63 defective val. Zero overlap.")

    # Save experiment config
    config_data = {
        "experiment_name": "patchcore_bottle_controlled",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "product": "bottle",
        "dataset": "MVTecAD",
        "backbone": "wide_resnet50_2",
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "image_size": [256, 256],
        "reference_count": len(ref_paths),
        "normal_validation_count": len(normal_val_paths),
        "defective_validation_count": len(defect_val_paths),
        "reference_images": [str(p) for p in ref_paths],
        "normal_validation_images": [str(p) for p in normal_val_paths],
        "defective_validation_images": [str(p) for p in defect_val_paths],
    }

    config_path = CONFIG_DIR / "experiment_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved configuration to: {config_path}")

    # 2. Initialize PatchCore Pipeline
    pipeline = PatchCorePipeline(
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        image_size=(256, 256),
        accelerator="gpu",
        devices=1,
    )

    # 3. Build Feature Memory Bank
    logger.info("Building PatchCore normal feature memory from 20 reference images...")
    t0 = time.time()
    memory_stats = pipeline.build_feature_memory(
        reference_images=ref_paths,
        product_name="bottle",
        work_dir=EXP_DIR / "staging",
    )
    fit_time = time.time() - t0
    logger.info(f"Feature memory built in {fit_time:.2f}s! Memory bank shape: {memory_stats['memory_bank_shape']}")

    # 4. Perform Inference on Validation Sets
    val_items = []
    for p in normal_val_paths:
        val_items.append({"path": p, "split": "test", "gt_status": "normal", "defect_type": "good"})
    for p in defect_val_paths:
        val_items.append({"path": p, "split": "test", "gt_status": "defective", "defect_type": p.parent.name})

    logger.info(f"Running anomaly inference on all {len(val_items)} validation images...")
    all_val_paths = [item["path"] for item in val_items]

    t1 = time.time()
    inference_results = pipeline.infer_batch(all_val_paths)
    infer_time = time.time() - t1
    logger.info(f"Inference completed in {infer_time:.2f}s ({infer_time/len(val_items)*1000:.1f} ms/image)!")

    # 5. Process results and render heatmaps
    normal_heatmaps_dir = HEATMAPS_DIR / "normal"
    defective_heatmaps_dir = HEATMAPS_DIR / "defective"
    normal_heatmaps_dir.mkdir(parents=True, exist_ok=True)
    defective_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    records = []
    normal_scores = []
    defective_scores = []
    y_true = []
    y_scores = []

    for i, res in enumerate(inference_results):
        meta = val_items[i]
        gt_status = meta["gt_status"]
        defect_type = meta["defect_type"]
        score = res.anomaly_score

        if gt_status == "normal":
            normal_scores.append(score)
            y_true.append(0)
        else:
            defective_scores.append(score)
            y_true.append(1)
        y_scores.append(score)

        # Decide whether to render heatmap
        # Render first 5 normal images, and representative defective images across all defect types
        render_heatmap = False
        if gt_status == "normal" and len([r for r in records if r["ground_truth_status"] == "normal"]) < 5:
            render_heatmap = True
            heatmap_dest = normal_heatmaps_dir / f"norm_{Path(res.image_path).name}"
        elif gt_status == "defective":
            # Count how many of this defect type rendered so far
            rendered_type_count = len([r for r in records if r.get("defect_type") == defect_type and r.get("heatmap_path")])
            if rendered_type_count < 4:  # 4 representative heatmaps per defect type
                render_heatmap = True
                heatmap_dest = defective_heatmaps_dir / f"{defect_type}_{Path(res.image_path).name}"

        heatmap_saved_path = None
        if render_heatmap:
            saved_p = pipeline.render_and_save_heatmap(
                image_path=res.image_path,
                anomaly_map=res.anomaly_map,
                output_path=heatmap_dest,
                score=score,
                ground_truth=f"{gt_status} ({defect_type})",
            )
            heatmap_saved_path = str(saved_p).replace("\\", "/")

        record = {
            "image_path": str(meta["path"]).replace("\\", "/"),
            "split": meta["split"],
            "ground_truth_status": gt_status,
            "defect_type": defect_type,
            "anomaly_score": round(score, 6),
            "anomalib_pred_label": res.pred_label,
            "heatmap_path": heatmap_saved_path,
        }
        records.append(record)

    # 6. Evaluation metrics
    norm_stats = compute_statistics(normal_scores)
    defect_stats = compute_statistics(defective_scores)
    auroc = float(roc_auc_score(y_true, y_scores)) if len(set(y_true)) > 1 else None

    # Score margin: min defective score vs max normal score
    separation_margin = defect_stats["min"] - norm_stats["max"]
    separation_achieved = separation_margin > 0 or auroc >= 0.99

    metrics = {
        "dataset": "MVTecAD/bottle",
        "fit_time_seconds": round(fit_time, 2),
        "inference_time_seconds": round(infer_time, 2),
        "inference_ms_per_image": round(infer_time / len(val_items) * 1000, 2),
        "normal_validation": norm_stats,
        "defective_validation": defect_stats,
        "image_auroc": round(auroc, 6) if auroc is not None else None,
        "separation_margin (min_defect - max_normal)": round(separation_margin, 6),
        "perfect_separation": bool(norm_stats["max"] < defect_stats["min"]),
        "experiment_status": "PASS" if auroc and auroc >= 0.99 else "FAIL",
    }

    # Save results
    results_path = RESULTS_DIR / "inference_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info(f"Saved {len(records)} inference records to: {results_path}")

    metrics_path = RESULTS_DIR / "summary_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved summary metrics to: {metrics_path}")

    # Print summary report
    print("\n" + "=" * 70)
    print("PATCHCORE CONTROLLED EXPERIMENT SUMMARY: MVTecAD bottle")
    print("=" * 70)
    print(f"Reference images (train/good):     {len(ref_paths)}")
    print(f"Normal validation (test/good):    {len(normal_val_paths)}")
    print(f"Defective validation (test/*):    {len(defect_val_paths)}")
    print(f"Feature memory shape:             {memory_stats['memory_bank_shape']}")
    print("-" * 70)
    print("NORMAL VALIDATION ANOMALY SCORES:")
    print(f"  Count:  {norm_stats['count']}")
    print(f"  Min:    {norm_stats['min']:.6f}")
    print(f"  Max:    {norm_stats['max']:.6f}")
    print(f"  Mean:   {norm_stats['mean']:.6f}")
    print(f"  Median: {norm_stats['median']:.6f}")
    print(f"  Std:    {norm_stats['std']:.6f}")
    print("-" * 70)
    print("DEFECTIVE VALIDATION ANOMALY SCORES:")
    print(f"  Count:  {defect_stats['count']}")
    print(f"  Min:    {defect_stats['min']:.6f}")
    print(f"  Max:    {defect_stats['max']:.6f}")
    print(f"  Mean:   {defect_stats['mean']:.6f}")
    print(f"  Median: {defect_stats['median']:.6f}")
    print(f"  Std:    {defect_stats['std']:.6f}")
    print("-" * 70)
    print(f"Image AUROC:                      {auroc:.6f}")
    print(f"Score Separation Margin:          {separation_margin:.6f}")
    print(f"Status:                           {metrics['experiment_status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
