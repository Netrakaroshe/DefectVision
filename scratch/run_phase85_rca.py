"""Phase 8.5: Pill Stage-1 False-Negative Root-Cause Analysis Script.

Analyzes all 24 Pill Stage-1 False Negatives from Phase 8.4 against:
1. Correctly detected Pill True Positives (N=72)
2. Pill Normal test images (N=26)

Investigates whether misses are driven by:
A. Threshold / calibration
B. Weak PatchCore feature representation (layers/backbone)
C. Normal-reference variation
D. Very small / low-contrast defects
E. Imprint / debossing interference

Outputs:
- experiments/system_evaluation/phase85/per_sample_analysis.json
- experiments/system_evaluation/phase85/score_analysis.json
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

WORKSPACE_ROOT = Path("D:/DefectVision")
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from defectvision.models.patchcore_engine import PatchCorePipeline

PHASE84_PREDS_PATH = WORKSPACE_ROOT / "experiments" / "system_evaluation" / "results" / "per_sample_predictions.json"
OUTPUT_DIR = WORKSPACE_ROOT / "experiments" / "system_evaluation" / "phase85"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PILL_CKPT_PATH = WORKSPACE_ROOT / "data" / "app_storage" / "products" / "pill2" / "models" / "pill2_patchcore.ckpt"
PILL_GT_DIR = Path("D:/datasets/defect_dataset_workspace/sources/MVTecAD/pill/ground_truth")
CALIB_THRESHOLD = 43.5951

def compute_contrast_metrics(img_bgr, mask_binary):
    """Compute defect-to-background contrast metrics."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    defect_pixels = gray[mask_binary > 0]
    
    if len(defect_pixels) == 0:
        return {"weber_contrast": 0.0, "delta_intensity": 0.0, "defect_mean_intensity": 0.0, "surround_mean_intensity": 0.0}
    
    # Create surrounding context ring (dilate mask by 15 pixels, subtract mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(mask_binary, kernel)
    surround_mask = (dilated > 0) & (mask_binary == 0)
    surround_pixels = gray[surround_mask]
    
    if len(surround_pixels) == 0:
        surround_pixels = gray[mask_binary == 0]
        
    defect_mean = float(np.mean(defect_pixels))
    surround_mean = float(np.mean(surround_pixels))
    
    delta_i = abs(defect_mean - surround_mean)
    weber = delta_i / (surround_mean + 1e-5)
    
    return {
        "weber_contrast": float(weber),
        "delta_intensity": float(delta_i),
        "defect_mean_intensity": float(defect_mean),
        "surround_mean_intensity": float(surround_mean)
    }

def main():
    print("=================================================================")
    print("Phase 8.5: Pill Stage-1 False-Negative Root-Cause Analysis")
    print("=================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Phase 8.4 predictions
    with open(PHASE84_PREDS_PATH, "r", encoding="utf-8") as f:
        all_preds = json.load(f)
    
    pill_preds = [p for p in all_preds if p["domain"] == "pill"]
    print(f"Total Pill samples loaded: {len(pill_preds)}")
    
    normal_preds = [p for p in pill_preds if not p["is_defective"]]
    defect_preds = [p for p in pill_preds if p["is_defective"]]
    tp_preds = [p for p in defect_preds if p["stage1_detected"]]
    fn_preds = [p for p in defect_preds if not p["stage1_detected"]]
    
    print(f"Normal samples: {len(normal_preds)}")
    print(f"Defect samples: {len(defect_preds)}")
    print(f"True Positives: {len(tp_preds)}")
    print(f"False Negatives: {len(fn_preds)}")
    assert len(fn_preds) == 24, f"Expected 24 False Negatives, got {len(fn_preds)}"
    
    # 2. Load Frozen PatchCore Pipeline
    print(f"Loading frozen PatchCore checkpoint: {PILL_CKPT_PATH}")
    pipeline = PatchCorePipeline(accelerator="gpu" if device.type == "cuda" else "cpu")
    pipeline.load_checkpoint(str(PILL_CKPT_PATH))
    print("PatchCore checkpoint loaded successfully.")
    
    # 3. Comprehensive Analysis of 24 False Negatives
    print("\n--- Analyzing 24 False Negatives ---")
    fn_analysis_list = []
    
    # Rank all defect predictions by anomaly score
    defect_preds_sorted = sorted(defect_preds, key=lambda x: x["stage1_score"])
    score_ranks = {p["filename"] + "_" + p["gt_category"]: rank for rank, p in enumerate(defect_preds_sorted)}
    
    all_pill_sorted = sorted(pill_preds, key=lambda x: x["stage1_score"])
    overall_ranks = {p["filename"] + "_" + p["gt_category"]: rank for rank, p in enumerate(all_pill_sorted)}

    for idx, fn in enumerate(fn_preds, 1):
        fn_file = fn["filename"]
        category = fn["gt_category"]
        img_path = Path(fn["source_image_path"])
        stem = Path(fn_file).stem
        
        # Load image
        img_bgr = cv2.imread(str(img_path))
        h, w = img_bgr.shape[:2]
        total_pixels = h * w
        
        # Load ground truth mask
        mask_path = PILL_GT_DIR / category / f"{stem}_mask.png"
        has_gt = mask_path.exists()
        if has_gt:
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_binary = (mask_raw > 0).astype(np.uint8)
            defect_pixels = int(np.sum(mask_binary))
            coverage_pct = (defect_pixels / total_pixels) * 100.0
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_pts = np.vstack(contours)
                bx, by, bw, bh = cv2.boundingRect(all_pts)
                m = cv2.moments(mask_binary)
                if m["m00"] > 0:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                else:
                    cx, cy = bx + bw // 2, by + bh // 2
            else:
                bx, by, bw, bh = 0, 0, 0, 0
                cx, cy = 0, 0
        else:
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            defect_pixels = 0
            coverage_pct = 0.0
            bx, by, bw, bh = 0, 0, 0, 0
            cx, cy = 0, 0
            
        # Run PatchCore inference to get full 2D anomaly heatmap
        res = pipeline.infer_batch([str(img_path)])[0]
        amap = res.anomaly_map  # 2D float array
        if amap.shape[:2] != (h, w):
            amap_resized = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            amap_resized = amap
            
        global_max_score = float(np.max(amap_resized))
        max_loc_flat = np.argmax(amap_resized)
        peak_y, peak_x = np.unravel_index(max_loc_flat, amap_resized.shape)
        peak_x, peak_y = int(peak_x), int(peak_y)
        
        # Analyze anomaly score inside vs outside mask
        if has_gt and defect_pixels > 0:
            defect_amap_values = amap_resized[mask_binary > 0]
            max_defect_score = float(np.max(defect_amap_values))
            mean_defect_score = float(np.mean(defect_amap_values))
            
            bg_amap_values = amap_resized[mask_binary == 0]
            max_bg_score = float(np.max(bg_amap_values))
            mean_bg_score = float(np.mean(bg_amap_values))
            
            # Check if global peak is inside defect mask (or within 15px radius)
            dist_to_peak = cv2.pointPolygonTest(contours[0], (peak_x, peak_y), True) if contours else -999.0
            # If dist_to_peak >= 0, point is inside; if > -15, within 15px
            is_peak_on_defect = (dist_to_peak >= -15.0)
            
            # Signal to background ratio
            snr = max_defect_score / (max_bg_score + 1e-5)
            
            # Defect localized signal?
            has_localized_signal = (max_defect_score >= 0.85 * global_max_score) and (max_defect_score >= mean_bg_score + 2.0 * np.std(bg_amap_values))
        else:
            max_defect_score = global_max_score
            mean_defect_score = global_max_score
            max_bg_score = global_max_score
            mean_bg_score = global_max_score
            is_peak_on_defect = False
            snr = 1.0
            has_localized_signal = False
            dist_to_peak = -999.0
            
        # Contrast metrics
        contrast_info = compute_contrast_metrics(img_bgr, mask_binary)
        
        # Defect physical characteristics classification
        is_imprint_related = (category == "faulty_imprint") or ("imprint" in category)
        is_tiny = coverage_pct < 0.05  # < 0.05% of image area (less than ~320 pixels in 800x800)
        is_low_contrast = contrast_info["weber_contrast"] < 0.20
        is_micro_crack = (category == "crack") and (bw < 30 or bh < 30 or defect_pixels < 250)
        is_faint_discoloration = (category == "color") and (contrast_info["delta_intensity"] < 25.0)
        
        # Anomaly score metrics
        score = float(fn["stage1_score"])
        margin = score - CALIB_THRESHOLD
        defect_rank = score_ranks.get(fn_file + "_" + category, -1)
        overall_rank = overall_ranks.get(fn_file + "_" + category, -1)
        
        # Primary failure mode attribution for this specific sample
        reasons = []
        if is_imprint_related:
            reasons.append("Debossed Imprint Invariance (PatchCore CNN feature invariance to font/stroke degradation)")
        if is_tiny or is_micro_crack:
            reasons.append(f"Spatial Dilution / Tiny Defect Size ({defect_pixels} px, {coverage_pct:.3f}% area)")
        if is_low_contrast or is_faint_discoloration:
            reasons.append(f"Low Contrast / Faint Intensity Gradient (Weber contrast: {contrast_info['weber_contrast']:.3f}, Delta I: {contrast_info['delta_intensity']:.1f})")
        if not is_peak_on_defect:
            reasons.append(f"Background / Rim Peak Domination (Defect peak {max_defect_score:.2f} < Background peak {max_bg_score:.2f})")
        elif margin > -1.5:
            reasons.append(f"Near-Threshold Calibration Fallout (Margin: {margin:.2f}, Defect localized but score fell below threshold)")
            
        sample_record = {
            "filename": fn_file,
            "category": category,
            "anomaly_score": score,
            "threshold": CALIB_THRESHOLD,
            "margin_to_threshold": margin,
            "rank_among_defects_asc": defect_rank,
            "percentile_among_defects": float((defect_rank / len(defect_preds)) * 100.0),
            "rank_among_all_pill_asc": overall_rank,
            "has_gt_mask": has_gt,
            "defect_pixels": defect_pixels,
            "coverage_pct": coverage_pct,
            "defect_bbox": [bx, by, bw, bh],
            "defect_centroid": [cx, cy],
            "global_peak_loc": [peak_x, peak_y],
            "is_global_peak_on_defect": bool(is_peak_on_defect),
            "dist_peak_to_defect_px": float(dist_to_peak),
            "max_defect_score": max_defect_score,
            "mean_defect_score": mean_defect_score,
            "max_bg_score": max_bg_score,
            "mean_bg_score": mean_bg_score,
            "snr_defect_to_bg": float(snr),
            "shows_localized_defect_signal": bool(has_localized_signal),
            "weber_contrast": contrast_info["weber_contrast"],
            "delta_intensity": contrast_info["delta_intensity"],
            "defect_mean_intensity": contrast_info["defect_mean_intensity"],
            "surround_mean_intensity": contrast_info["surround_mean_intensity"],
            "is_imprint_related": bool(is_imprint_related),
            "is_tiny": bool(is_tiny),
            "is_low_contrast": bool(is_low_contrast),
            "is_micro_crack": bool(is_micro_crack),
            "is_faint_discoloration": bool(is_faint_discoloration),
            "attributed_failure_modes": reasons
        }
        fn_analysis_list.append(sample_record)
        print(f"[{idx:02d}/24] {category:15s} | {fn_file:7s} | Score: {score:.2f} (Margin: {margin:+.2f}) | Px: {defect_pixels:4d} ({coverage_pct:.3f}%) | PeakOnDefect: {str(is_peak_on_defect):5s} | LocalSignal: {str(has_localized_signal):5s}")

    # 4. Analyze True Positives (N=72) for comparison
    print("\n--- Analyzing 72 True Positives for Benchmark Comparison ---")
    tp_metrics_list = []
    for tp in tp_preds:
        tp_file = tp["filename"]
        category = tp["gt_category"]
        img_path = Path(tp["source_image_path"])
        stem = Path(tp_file).stem
        
        mask_path = PILL_GT_DIR / category / f"{stem}_mask.png"
        if mask_path.exists():
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_binary = (mask_raw > 0).astype(np.uint8)
            defect_pixels = int(np.sum(mask_binary))
            coverage_pct = (defect_pixels / (mask_raw.shape[0] * mask_raw.shape[1])) * 100.0
            img_bgr = cv2.imread(str(img_path))
            contrast_info = compute_contrast_metrics(img_bgr, mask_binary)
        else:
            defect_pixels = 0
            coverage_pct = 0.0
            contrast_info = {"weber_contrast": 0.0, "delta_intensity": 0.0}
            
        tp_metrics_list.append({
            "filename": tp_file,
            "category": category,
            "anomaly_score": float(tp["stage1_score"]),
            "defect_pixels": defect_pixels,
            "coverage_pct": coverage_pct,
            "weber_contrast": contrast_info["weber_contrast"],
            "delta_intensity": contrast_info["delta_intensity"]
        })

    # 5. Statistical Distributions & Score Comparisons
    def compute_stats(values):
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return {}
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr))
        }

    normal_scores = [p["stage1_score"] for p in normal_preds]
    defect_scores = [p["stage1_score"] for p in defect_preds]
    tp_scores = [p["stage1_score"] for p in tp_preds]
    fn_scores = [p["stage1_score"] for p in fn_preds]
    
    # Category score distributions
    cat_scores = defaultdict(list)
    cat_tp_counts = defaultdict(int)
    cat_fn_counts = defaultdict(int)
    for p in defect_preds:
        cat = p["gt_category"]
        cat_scores[cat].append(p["stage1_score"])
        if p["stage1_detected"]:
            cat_tp_counts[cat] += 1
        else:
            cat_fn_counts[cat] += 1
            
    cat_score_stats = {cat: compute_stats(scores) for cat, scores in cat_scores.items()}
    
    # Compare defect sizes (area) between TPs and FNs
    tp_areas = [m["coverage_pct"] for m in tp_metrics_list if m["defect_pixels"] > 0]
    fn_areas = [m["coverage_pct"] for m in fn_analysis_list if m["defect_pixels"] > 0]
    
    tp_weber = [m["weber_contrast"] for m in tp_metrics_list if m["defect_pixels"] > 0]
    fn_weber = [m["weber_contrast"] for m in fn_analysis_list if m["defect_pixels"] > 0]
    
    tp_delta_i = [m["delta_intensity"] for m in tp_metrics_list if m["defect_pixels"] > 0]
    fn_delta_i = [m["delta_intensity"] for m in fn_analysis_list if m["defect_pixels"] > 0]
    
    # Score overlap analysis & Threshold sweep simulation
    # Sweeping thresholds to see trade-off between FN recovery and Normal FP fallout
    threshold_sweep = []
    for test_thresh in np.arange(36.0, 46.5, 0.5):
        test_thresh = float(round(test_thresh, 2))
        fp = sum(1 for s in normal_scores if s >= test_thresh)
        fn_count = sum(1 for s in defect_scores if s < test_thresh)
        tp_count = sum(1 for s in defect_scores if s >= test_thresh)
        tn = sum(1 for s in normal_scores if s < test_thresh)
        fpr = fp / len(normal_scores)
        tpr = tp_count / len(defect_scores)
        threshold_sweep.append({
            "threshold": test_thresh,
            "tp": tp_count,
            "fp": fp,
            "tn": tn,
            "fn": fn_count,
            "sensitivity_tpr": float(tpr),
            "specificity_tnr": float(1.0 - fpr),
            "false_positive_rate": float(fpr)
        })

    # Summary of Root Cause Hypothesis Evidence
    # A. Threshold / Calibration
    fn_near_threshold = sum(1 for fn in fn_analysis_list if fn["margin_to_threshold"] >= -1.5)  # within 1.5 pts
    fn_far_threshold = sum(1 for fn in fn_analysis_list if fn["margin_to_threshold"] < -3.0)   # > 3.0 pts below
    
    # B. Imprint debossing invariance
    fn_imprint_count = sum(1 for fn in fn_analysis_list if fn["is_imprint_related"])
    imprint_total = len(cat_scores.get("faulty_imprint", []))
    imprint_fail_rate = fn_imprint_count / imprint_total if imprint_total > 0 else 0.0
    
    # D. Tiny / Low Contrast
    fn_tiny_count = sum(1 for fn in fn_analysis_list if fn["is_tiny"])
    fn_low_contrast_count = sum(1 for fn in fn_analysis_list if fn["is_low_contrast"])
    
    # Peak on defect vs Peak on background
    fn_peak_on_defect_count = sum(1 for fn in fn_analysis_list if fn["is_global_peak_on_defect"])
    fn_peak_on_bg_count = len(fn_analysis_list) - fn_peak_on_defect_count
    
    score_analysis_data = {
        "calibrated_threshold": CALIB_THRESHOLD,
        "score_distributions": {
            "normals": compute_stats(normal_scores),
            "all_defects": compute_stats(defect_scores),
            "true_positives": compute_stats(tp_scores),
            "false_negatives": compute_stats(fn_scores)
        },
        "defect_categories": {
            cat: {
                "total": len(cat_scores[cat]),
                "tp": cat_tp_counts[cat],
                "fn": cat_fn_counts[cat],
                "failure_rate": float(cat_fn_counts[cat] / len(cat_scores[cat])),
                "score_stats": cat_score_stats[cat]
            }
            for cat in sorted(cat_scores.keys())
        },
        "physical_attribute_comparison": {
            "defect_coverage_pct": {
                "true_positives": compute_stats(tp_areas),
                "false_negatives": compute_stats(fn_areas)
            },
            "weber_contrast": {
                "true_positives": compute_stats(tp_weber),
                "false_negatives": compute_stats(fn_weber)
            },
            "delta_intensity": {
                "true_positives": compute_stats(tp_delta_i),
                "false_negatives": compute_stats(fn_delta_i)
            }
        },
        "score_overlap": {
            "normal_score_range": [float(np.min(normal_scores)), float(np.max(normal_scores))],
            "fn_score_range": [float(np.min(fn_scores)), float(np.max(fn_scores))],
            "tp_score_range": [float(np.min(tp_scores)), float(np.max(tp_scores))],
            "overlap_range": [float(np.min(fn_scores)), float(np.max(normal_scores))],
            "normal_samples_above_fn_min": int(sum(1 for s in normal_scores if s >= np.min(fn_scores))),
            "normal_pct_above_fn_min": float(sum(1 for s in normal_scores if s >= np.min(fn_scores)) / len(normal_scores) * 100.0)
        },
        "hypothesis_evidence_metrics": {
            "total_fn": len(fn_analysis_list),
            "fn_near_threshold_margin_gt_minus_1_5": fn_near_threshold,
            "fn_far_threshold_margin_lt_minus_3_0": fn_far_threshold,
            "fn_imprint_count": fn_imprint_count,
            "imprint_total_test": imprint_total,
            "imprint_failure_rate": float(imprint_fail_rate),
            "fn_tiny_count": fn_tiny_count,
            "fn_low_contrast_count": fn_low_contrast_count,
            "fn_peak_on_defect_count": fn_peak_on_defect_count,
            "fn_peak_on_background_count": fn_peak_on_bg_count
        },
        "threshold_sweep_simulation": threshold_sweep
    }
    
    # 6. Save JSON artifacts
    per_sample_out = OUTPUT_DIR / "per_sample_analysis.json"
    with open(per_sample_out, "w", encoding="utf-8") as f:
        json.dump(fn_analysis_list, f, indent=2)
    print(f"\nSaved per-sample analysis to: {per_sample_out}")
    
    score_analysis_out = OUTPUT_DIR / "score_analysis.json"
    with open(score_analysis_out, "w", encoding="utf-8") as f:
        json.dump(score_analysis_data, f, indent=2)
    print(f"Saved score analysis to: {score_analysis_out}")
    
    print("\nRoot-cause script execution complete.")

if __name__ == "__main__":
    main()
