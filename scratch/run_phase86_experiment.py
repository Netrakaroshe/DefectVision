"""Phase 8.6: Pill PatchCore Representation Experiment Execution Script.

Evaluates 4 candidate representation variants strictly offline:
- Variant A: Current WideResNet-50-2 with ('layer2', 'layer3') [Frozen Baseline]
- Variant B: WideResNet-50-2 with ('layer1', 'layer2', 'layer3') [Multi-Scale CNN]
- Variant C: DINOv2 vit_small_patch14_dinov2 with ('blocks.8', 'blocks.11') [Dense ViT]
- Variant D: Baseline representation plus Pill alignment normalization

Strict Constraints:
- Zero modifications to production source code, checkpoints, database, or frontend.
- Authoritative MVTec datasets remain read-only.
- All outputs written to experiments/system_evaluation/phase86/.
"""

import sys
import os
import json
import time
import random
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

WORKSPACE_ROOT = Path("D:/DefectVision")
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder
from defectvision.models.patchcore_engine import PatchCorePipeline, AnomalyInferenceResult

OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "system_evaluation" / "phase86"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

REF_DIR = WORKSPACE_ROOT / "data" / "app_storage" / "products" / "pill2" / "reference_images"
MVTEC_PILL_DIR = Path("D:/datasets/defect_dataset_workspace/sources/MVTecAD/pill")
PILL_GT_DIR = MVTEC_PILL_DIR / "ground_truth"

# Load the 24 Phase 8.5 False Negatives for exact tracking
PHASE85_FN_PATH = WORKSPACE_ROOT / "experiments" / "system_evaluation" / "phase85" / "per_sample_analysis.json"
with open(PHASE85_FN_PATH, "r", encoding="utf-8") as f:
    phase85_fns_data = json.load(f)
PHASE85_FN_KEYS = set(f"{x['category']}_{x['filename']}" for x in phase85_fns_data)
print(f"Loaded {len(PHASE85_FN_KEYS)} Phase 8.5 reference false-negative keys.")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_mask_metrics(roi_bbox, mask_path, img_w, img_h, amap_global_peak):
    """Compute quantitative overlap against MVTec pixel ground truth mask."""
    rx, ry, rw, rh = roi_bbox
    px, py = amap_global_peak
    if not mask_path.exists():
        return {"coverage": 0.0, "iou": 0.0, "purity": 0.0, "peak_on_defect": False, "has_gt": False}

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"coverage": 0.0, "iou": 0.0, "purity": 0.0, "peak_on_defect": False, "has_gt": False}

    mask_binary = (mask > 0).astype(np.uint8)
    total_defect_pixels = int(np.sum(mask_binary))

    if total_defect_pixels == 0:
        return {"coverage": 1.0, "iou": 0.0, "purity": 0.0, "peak_on_defect": False, "has_gt": False}

    roi_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    roi_mask[ry:ry+rh, rx:rx+rw] = 1

    overlap_pixels = int(np.sum(mask_binary & roi_mask))
    coverage = float(overlap_pixels / total_defect_pixels)
    purity = float(overlap_pixels / (rw * rh + 1e-8))

    # Minimal enclosing bounding box of ground truth mask
    gt_contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if gt_contours:
        all_pts = np.vstack(gt_contours)
        gx, gy, gw, gh = cv2.boundingRect(all_pts)
        ix1, iy1 = max(rx, gx), max(ry, gy)
        ix2, iy2 = min(rx + rw, gx + gw), min(ry + rh, gy + gh)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union_area = (rw * rh) + (gw * gh) - inter_area
        iou = float(inter_area / (union_area + 1e-8))
        
        # Check if global peak falls within 15px of defect contour
        dist_to_peak = cv2.pointPolygonTest(gt_contours[0], (px, py), True)
        peak_on_defect = bool(dist_to_peak >= -15.0)
    else:
        iou = 0.0
        peak_on_defect = False

    return {
        "coverage": coverage,
        "iou": iou,
        "purity": purity,
        "peak_on_defect": peak_on_defect,
        "total_defect_pixels": total_defect_pixels,
        "overlap_pixels": overlap_pixels,
        "has_gt": True
    }

def extract_deterministic_roi(anomaly_map, img_w, img_h, thresh_calib, padding_ratio=0.25):
    """Deterministic PatchCore ROI extraction."""
    amap_blur = cv2.GaussianBlur(anomaly_map, (15, 15), 4)
    max_val = float(np.max(amap_blur))
    t_rel = max(0.80 * max_val, 0.90 * thresh_calib)

    mask = (amap_blur >= t_rel).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        x, y, bw, bh = int(img_w * 0.25), int(img_h * 0.25), int(img_w * 0.5), int(img_h * 0.5)
        return (x, y, bw, bh), (int(img_w * 0.5), int(img_h * 0.5))

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = contours[0]
    bx, by, bw, bh = cv2.boundingRect(c)

    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)

    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(img_w, bx + bw + pad_w)
    y2 = min(img_h, by + bh + pad_h)

    padded_bbox = (x1, y1, x2 - x1, y2 - y1)
    
    max_loc = np.unravel_index(np.argmax(amap_blur), amap_blur.shape)
    peak_pt = (int(max_loc[1]), int(max_loc[0]))
    return padded_bbox, peak_pt

def align_pill_image(img_bgr, mask_gray=None):
    """Canonical in-plane alignment of pill tablet."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr, mask_gray, 0.0
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return img_bgr, mask_gray, 0.0
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(c)
    major_angle = angle if d1 < d2 else angle + 90.0
    rot_angle = 90.0 - major_angle
    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    aligned_img = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    aligned_mask = cv2.warpAffine(mask_gray, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) if mask_gray is not None else None
    return aligned_img, aligned_mask, float(rot_angle)

def compute_distribution_stats(scores):
    arr = np.array(scores, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    }

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset Manifests
    ref_paths = sorted([str(p.resolve()) for p in REF_DIR.glob("*.png")])
    assert len(ref_paths) == 10, f"Expected 10 reference images, found {len(ref_paths)}"
    print(f"Reference Images (N=10): {[Path(p).name for p in ref_paths]}")

    # Load V2 45-sample training pool to ensure 100% exclusion from external evaluation
    with open(WORKSPACE_ROOT / "experiments" / "pill_roi_classifier_v2" / "roi_manifest.jsonl") as f:
        v2_pool = [json.loads(line) for line in f if line.strip()]
    v2_pool_paths = {Path(r["source_image_path"]).as_posix().lower() for r in v2_pool}
    print(f"Loaded {len(v2_pool_paths)} V2 development pool paths for strict exclusion.")

    # Build 122 test items
    normal_test_paths = sorted(list((MVTEC_PILL_DIR / "test" / "good").glob("*.png")))
    assert len(normal_test_paths) == 26, f"Expected 26 normal test images, found {len(normal_test_paths)}"

    defect_categories = ["faulty_imprint", "contamination", "color", "crack", "combined", "scratch", "pill_type"]
    eval_items = []

    # Normal items
    for p in normal_test_paths:
        eval_items.append({
            "source_image_path": str(p.resolve()),
            "filename": p.name,
            "category": "good",
            "is_defective": False,
            "mask_path": None
        })

    # Defect items
    for cat in defect_categories:
        cat_dir = MVTEC_PILL_DIR / "test" / cat
        cat_paths = sorted(list(cat_dir.glob("*.png")))
        for p in cat_paths:
            p_posix = p.as_posix().lower()
            if p_posix in v2_pool_paths:
                # Exclude V2 development pool
                continue
            mask_p = PILL_GT_DIR / cat / f"{p.stem}_mask.png"
            eval_items.append({
                "source_image_path": str(p.resolve()),
                "filename": p.name,
                "category": cat,
                "is_defective": True,
                "mask_path": str(mask_p.resolve()) if mask_p.exists() else None
            })

    assert len(eval_items) == 122, f"Expected 122 evaluation items, got {len(eval_items)}"
    defect_count = sum(1 for x in eval_items if x["is_defective"])
    normal_count = sum(1 for x in eval_items if not x["is_defective"])
    print(f"Evaluation items total: {len(eval_items)} (Normal: {normal_count}, Defective: {defect_count})")
    cat_counts = Counter(x["category"] for x in eval_items)
    for c, cnt in sorted(cat_counts.items()):
        print(f"  {c:15s}: {cnt}")

    # 2. Variants Specification
    variants = [
        {
            "id": "variant_a",
            "name": "Variant A (Frozen Baseline)",
            "backbone": "wide_resnet50_2",
            "layers": ("layer2", "layer3"),
            "load_existing_ckpt": True,
            "existing_ckpt_path": WORKSPACE_ROOT / "data" / "app_storage" / "products" / "pill2" / "models" / "pill2_patchcore.ckpt",
            "image_size": (256, 256),
            "use_aligned": False
        },
        {
            "id": "variant_b",
            "name": "Variant B (Multi-Scale CNN)",
            "backbone": "wide_resnet50_2",
            "layers": ("layer1", "layer2", "layer3"),
            "load_existing_ckpt": False,
            "image_size": (256, 256),
            "use_aligned": False
        },
        {
            "id": "variant_c",
            "name": "Variant C (DINOv2 Dense ViT)",
            "backbone": "vit_small_patch14_dinov2",
            "layers": ("blocks.8", "blocks.11"),
            "load_existing_ckpt": False,
            "image_size": (224, 224),
            "use_aligned": False
        },
        {
            "id": "variant_d",
            "name": "Variant D (Aligned Baseline)",
            "backbone": "wide_resnet50_2",
            "layers": ("layer2", "layer3"),
            "load_existing_ckpt": False,
            "image_size": (256, 256),
            "use_aligned": True
        }
    ]

    all_variant_results = {}
    per_sample_predictions = defaultdict(dict)
    all_localization_results = defaultdict(list)
    resource_results = {}

    # Pre-generate aligned images for Variant D
    aligned_cache_dir = OUTPUT_DIR / "aligned_cache"
    aligned_ref_dir = aligned_cache_dir / "refs"
    aligned_test_dir = aligned_cache_dir / "test"
    aligned_mask_dir = aligned_cache_dir / "masks"
    aligned_ref_dir.mkdir(parents=True, exist_ok=True)
    aligned_test_dir.mkdir(parents=True, exist_ok=True)
    aligned_mask_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreparing Variant D Pill alignment cache...")
    aligned_ref_paths = []
    for p in ref_paths:
        img_bgr = cv2.imread(p)
        aligned_img, _, _ = align_pill_image(img_bgr)
        dest = aligned_ref_dir / Path(p).name
        cv2.imwrite(str(dest), aligned_img)
        aligned_ref_paths.append(str(dest))

    aligned_eval_items = []
    for item in eval_items:
        img_bgr = cv2.imread(item["source_image_path"])
        mask_gray = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE) if item["mask_path"] else None
        aligned_img, aligned_mask, _ = align_pill_image(img_bgr, mask_gray)
        dest_img = aligned_test_dir / f"{item['category']}_{item['filename']}"
        cv2.imwrite(str(dest_img), aligned_img)
        if aligned_mask is not None:
            dest_mask = aligned_mask_dir / f"{item['category']}_{item['filename']}"
            cv2.imwrite(str(dest_mask), aligned_mask)
            mask_out_p = str(dest_mask)
        else:
            mask_out_p = None
        aligned_eval_items.append({
            "source_image_path": str(dest_img),
            "filename": item["filename"],
            "category": item["category"],
            "is_defective": item["is_defective"],
            "mask_path": mask_out_p
        })
    print("Alignment cache ready.")

    # Execute each variant
    for v_cfg in variants:
        v_id = v_cfg["id"]
        v_name = v_cfg["name"]
        print(f"\n=======================================================")
        print(f"EXECUTING: {v_name}")
        print(f"Backbone: {v_cfg['backbone']} | Layers: {v_cfg['layers']} | ImageSize: {v_cfg['image_size']}")
        print(f"=======================================================")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        current_ref_paths = aligned_ref_paths if v_cfg["use_aligned"] else ref_paths
        current_eval_items = aligned_eval_items if v_cfg["use_aligned"] else eval_items

        pipeline = PatchCorePipeline(
            backbone=v_cfg["backbone"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
            image_size=v_cfg["image_size"],
            accelerator="gpu" if device.type == "cuda" else "cpu"
        )

        work_dir = OUTPUT_DIR / f"work_{v_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        t_fit_start = time.time()
        if v_cfg["load_existing_ckpt"]:
            print(f"Loading existing frozen baseline checkpoint: {v_cfg['existing_ckpt_path'].name}")
            pipeline.load_checkpoint(v_cfg["existing_ckpt_path"])
            t_fit = 0.0
        else:
            print("Fitting feature memory bank on 10 normal calibration images...")
            stage_dir = work_dir / "stage_ref" / "good"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
            for p in current_ref_paths:
                shutil.copy2(p, stage_dir / Path(p).name)

            dm = Folder(
                name=f"pill_{v_id}",
                normal_dir=str(stage_dir),
                train_batch_size=min(8, len(current_ref_paths)),
                eval_batch_size=min(8, len(current_ref_paths)),
                num_workers=0
            )

            # Preprocessor size handling for DINOv2 / ViT
            model_kwargs = {
                "backbone": v_cfg["backbone"],
                "layers": v_cfg["layers"],
                "coreset_sampling_ratio": 0.1,
                "num_neighbors": 9,
                "pre_trained": True
            }
            anomalib_model = Patchcore(**model_kwargs)
            if v_cfg["image_size"] != (256, 256):
                anomalib_model.pre_processor.transform.transforms[0].size = list(v_cfg["image_size"])
                anomalib_model.pre_processor.export_transform.transforms[0].size = list(v_cfg["image_size"])

            engine = Engine(
                default_root_dir=work_dir / "engine_runs",
                accelerator="gpu" if device.type == "cuda" else "cpu",
                devices=1
            )
            engine.fit(model=anomalib_model, datamodule=dm)
            pipeline.model = anomalib_model
            pipeline.is_fitted = True
            if hasattr(anomalib_model.model, "memory_bank") and anomalib_model.model.memory_bank is not None:
                pipeline.memory_bank_shape = tuple(anomalib_model.model.memory_bank.shape)
            t_fit = time.time() - t_fit_start
            print(f"Memory bank fitted in {t_fit:.2f}s. Shape: {pipeline.memory_bank_shape}")

            # Save checkpoint as experiment artifact
            ckpt_save_path = CHECKPOINTS_DIR / f"{v_id}_patchcore.ckpt"
            engine.trainer.save_checkpoint(str(ckpt_save_path))
            print(f"Saved artifact checkpoint: {ckpt_save_path}")

        # Compute calibration threshold on the 10 reference images
        print("Calibrating decision threshold on 10 reference images...")
        ref_infer_res = pipeline.infer_batch(current_ref_paths, image_size=v_cfg["image_size"])
        ref_scores = [r.anomaly_score for r in ref_infer_res]
        mu_ref = float(np.mean(ref_scores))
        std_ref_sample = float(np.std(ref_scores, ddof=1)) # sample standard deviation
        std_ref_pop = float(np.std(ref_scores, ddof=0))
        if v_cfg["load_existing_ckpt"]:
            calib_thresh = 43.5951
            print(f"Variant A using frozen production threshold: {calib_thresh:.4f}")
        else:
            calib_thresh = float(mu_ref + 3.0 * std_ref_sample)
            print(f"Calibration: mu={mu_ref:.4f}, sample_std={std_ref_sample:.4f}, pop_std={std_ref_pop:.4f} -> Calibrated Threshold (mu + 3*s) = {calib_thresh:.4f}")

        # Evaluate 122 test images
        print(f"Evaluating {len(current_eval_items)} test images...")
        test_paths = [x["source_image_path"] for x in current_eval_items]
        t0_eval = time.time()
        test_infer_res = pipeline.infer_batch(test_paths, image_size=v_cfg["image_size"])
        t_eval_total = time.time() - t0_eval
        print(f"Inference completed in {t_eval_total:.2f}s ({t_eval_total/len(test_paths)*1000:.1f} ms/image)")

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # 50 Timed Cycles for Latency Profile (CUDA Events)
        print("Measuring inference latency across 50 timed cycles with CUDA events...")
        sample_img_p = test_paths[0]
        # Warmup 10 cycles
        for _ in range(10):
            _ = pipeline.infer_batch([sample_img_p], image_size=v_cfg["image_size"])

        latencies_ms = []
        for _ in range(50):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = pipeline.infer_batch([sample_img_p], image_size=v_cfg["image_size"])
            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))

        latency_stats = {
            "mean_ms": float(np.mean(latencies_ms)),
            "std_ms": float(np.std(latencies_ms)),
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p90_ms": float(np.percentile(latencies_ms, 90)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "max_ms": float(np.max(latencies_ms))
        }
        print(f"Latency: Mean={latency_stats['mean_ms']:.1f}ms, P50={latency_stats['p50_ms']:.1f}ms, P95={latency_stats['p95_ms']:.1f}ms | Peak VRAM: {peak_vram_mb:.1f} MB")

        # Process per-sample predictions & localization
        tp, tn, fp, fn = 0, 0, 0, 0
        normal_scores, defect_scores = [], []
        cat_scores = defaultdict(list)
        cat_detected = defaultdict(int)
        cat_total = defaultdict(int)
        fn_list = []
        recovered_from_phase85 = []
        still_missed_phase85 = []
        newly_lost_defects = []

        localization_coverages = []
        localization_ious = []
        localization_purities = []
        peak_on_defect_hits = 0
        total_gt_eval = 0

        y_true, y_scores = [], []

        for item, res in zip(current_eval_items, test_infer_res):
            score = float(res.anomaly_score)
            amap = res.anomaly_map
            is_def = item["is_defective"]
            cat = item["category"]
            fn_key = f"{cat}_{item['filename']}"

            y_true.append(1 if is_def else 0)
            y_scores.append(score)

            pred_def = bool(score >= calib_thresh)
            per_sample_predictions[fn_key][v_id] = {
                "score": score,
                "detected": pred_def,
                "threshold": calib_thresh,
                "margin": score - calib_thresh
            }

            if is_def:
                defect_scores.append(score)
                cat_total[cat] += 1
                cat_scores[cat].append(score)
                if pred_def:
                    tp += 1
                    cat_detected[cat] += 1
                    if fn_key in PHASE85_FN_KEYS:
                        recovered_from_phase85.append(fn_key)
                else:
                    fn += 1
                    fn_list.append({
                        "category": cat,
                        "filename": item["filename"],
                        "score": score,
                        "margin": score - calib_thresh
                    })
                    if fn_key in PHASE85_FN_KEYS:
                        still_missed_phase85.append(fn_key)
                    else:
                        newly_lost_defects.append(fn_key)

                # Anomaly localization against ground truth mask
                mask_p = Path(item["mask_path"]) if item["mask_path"] else None
                if mask_p and mask_p.exists():
                    orig_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                    if orig_mask is not None:
                        img_h, img_w = orig_mask.shape[:2]
                        if amap.shape[:2] != (img_h, img_w):
                            amap_resized = cv2.resize(amap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            amap_resized = amap
                        roi_bbox, peak_pt = extract_deterministic_roi(amap_resized, img_w, img_h, calib_thresh)
                        loc_m = compute_mask_metrics(roi_bbox, mask_p, img_w, img_h, peak_pt)
                        if loc_m["has_gt"]:
                            localization_coverages.append(loc_m["coverage"])
                            localization_ious.append(loc_m["iou"])
                            localization_purities.append(loc_m["purity"])
                            if loc_m["peak_on_defect"]:
                                peak_on_defect_hits += 1
                            total_gt_eval += 1
                            all_localization_results[v_id].append({
                                "category": cat,
                                "filename": item["filename"],
                                "coverage": loc_m["coverage"],
                                "iou": loc_m["iou"],
                                "purity": loc_m["purity"],
                                "peak_on_defect": loc_m["peak_on_defect"]
                            })
            else:
                normal_scores.append(score)
                if pred_def:
                    fp += 1
                else:
                    tn += 1

        auroc = float(roc_auc_score(y_true, y_scores))
        fpr_curve, tpr_curve, thresholds_curve = roc_curve(y_true, y_scores)

        # Sensitivity at reference FPR levels
        def get_tpr_at_fpr(target_fpr):
            idx = np.searchsorted(fpr_curve, target_fpr)
            if idx >= len(tpr_curve):
                return float(tpr_curve[-1])
            return float(tpr_curve[idx])

        tpr_at_0fpr = get_tpr_at_fpr(0.0)
        tpr_at_4fpr = get_tpr_at_fpr(0.04)
        tpr_at_10fpr = get_tpr_at_fpr(0.10)

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fpr = fp / (tn + fp)

        score_margin = float(np.min(defect_scores) - np.max(normal_scores))

        cat_sens = {
            c: {
                "total": cat_total[c],
                "detected": cat_detected[c],
                "sensitivity": float(cat_detected[c] / cat_total[c]) if cat_total[c] > 0 else 0.0,
                "score_stats": compute_distribution_stats(cat_scores[c])
            }
            for c in sorted(cat_total.keys())
        }

        loc_summary = {
            "evaluated_count": total_gt_eval,
            "mean_coverage": float(np.mean(localization_coverages)) if localization_coverages else 0.0,
            "mean_iou_box": float(np.mean(localization_ious)) if localization_ious else 0.0,
            "mean_purity": float(np.mean(localization_purities)) if localization_purities else 0.0,
            "peak_on_defect_rate": float(peak_on_defect_hits / total_gt_eval) if total_gt_eval > 0 else 0.0
        }

        memory_bank_shape = list(pipeline.memory_bank_shape) if pipeline.memory_bank_shape else []
        model_param_count = sum(p.numel() for p in pipeline.model.parameters()) if pipeline.model else 0

        variant_summary = {
            "variant_id": v_id,
            "variant_name": v_name,
            "backbone": v_cfg["backbone"],
            "layers": list(v_cfg["layers"]),
            "image_size": list(v_cfg["image_size"]),
            "use_aligned": v_cfg["use_aligned"],
            "model_param_count": model_param_count,
            "memory_bank_shape": memory_bank_shape,
            "calibrated_threshold": calib_thresh,
            "ref_stats": {
                "mean": mu_ref,
                "std_sample": std_ref_sample,
                "std_pop": std_ref_pop,
                "scores": [float(round(s, 4)) for s in ref_scores]
            },
            "metrics": {
                "auroc": auroc,
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "fpr": float(fpr),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "score_margin": score_margin
            },
            "threshold_independent_roc": {
                "auroc": auroc,
                "tpr_at_0_fpr": tpr_at_0fpr,
                "tpr_at_4_fpr": tpr_at_4fpr,
                "tpr_at_10_fpr": tpr_at_10fpr
            },
            "score_distributions": {
                "normals": compute_distribution_stats(normal_scores),
                "all_defects": compute_distribution_stats(defect_scores),
                "false_negatives": compute_distribution_stats([x["score"] for x in fn_list])
            },
            "category_sensitivity": cat_sens,
            "phase85_fn_recovery": {
                "total_phase85_fns": len(PHASE85_FN_KEYS),
                "recovered_count": len(recovered_from_phase85),
                "recovered_samples": sorted(recovered_from_phase85),
                "still_missed_count": len(still_missed_phase85),
                "still_missed_samples": sorted(still_missed_phase85),
                "newly_lost_count": len(newly_lost_defects),
                "newly_lost_samples": sorted(newly_lost_defects)
            },
            "localization_metrics": loc_summary,
            "resources": {
                "latency": latency_stats,
                "peak_vram_mb": float(peak_vram_mb),
                "fitting_time_sec": float(t_fit)
            },
            "false_negative_list": fn_list
        }

        all_variant_results[v_id] = variant_summary
        resource_results[v_id] = {
            "variant_name": v_name,
            "model_param_count": model_param_count,
            "memory_bank_shape": memory_bank_shape,
            "peak_vram_mb": float(peak_vram_mb),
            "fitting_time_sec": float(t_fit),
            "latency": latency_stats
        }

        print(f"\nResults for {v_name}:")
        print(f"  AUROC: {auroc:.4f} | Sens: {sensitivity*100:.1f}% ({tp}/96) | Spec: {specificity*100:.1f}% ({tn}/26) | FPR: {fpr*100:.1f}% ({fp}/26)")
        print(f"  FNs: {fn} (Recovered from Phase 8.5: {len(recovered_from_phase85)}, Newly Lost: {len(newly_lost_defects)})")
        print(f"  Faulty Imprint Sens: {cat_sens['faulty_imprint']['sensitivity']*100:.1f}% ({cat_sens['faulty_imprint']['detected']}/19)")
        print(f"  Crack Sens: {cat_sens['crack']['sensitivity']*100:.1f}% ({cat_sens['crack']['detected']}/10)")
        print(f"  Localization Coverage: {loc_summary['mean_coverage']*100:.1f}% | IoU: {loc_summary['mean_iou_box']:.3f} | Peak on Defect: {loc_summary['peak_on_defect_rate']*100:.1f}%")

    # Save machine-readable JSON artifacts
    with open(OUTPUT_DIR / "variant_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_variant_results, f, indent=2)
    print(f"\nSaved variant metrics to: {OUTPUT_DIR / 'variant_metrics.json'}")

    with open(OUTPUT_DIR / "per_sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(per_sample_predictions, f, indent=2)
    print(f"Saved per-sample predictions to: {OUTPUT_DIR / 'per_sample_predictions.json'}")

    with open(OUTPUT_DIR / "localization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_localization_results, f, indent=2)
    print(f"Saved localization metrics to: {OUTPUT_DIR / 'localization_metrics.json'}")

    with open(OUTPUT_DIR / "resource_metrics.json", "w", encoding="utf-8") as f:
        json.dump(resource_results, f, indent=2)
    print(f"Saved resource metrics to: {OUTPUT_DIR / 'resource_metrics.json'}")

    print("\nPhase 8.6 Execution script complete.")

if __name__ == "__main__":
    main()
