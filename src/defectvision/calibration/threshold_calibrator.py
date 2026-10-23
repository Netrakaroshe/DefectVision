"""Product-specific anomaly threshold calibration and classification for DefectVision.

This module provides statistically defensible, product-specific threshold
calibration derived directly from empirical PatchCore continuous anomaly scores.
It supports:
1. Supervised/Semi-supervised calibration (both normal and known-defective scores available)
   using Youden's J statistic / Balanced Accuracy optimization with maximum-margin plateau resolution.
2. Normal-only calibration (when defective samples are unavailable) using parametric (k-sigma)
   or non-parametric (quantile/IQR) upper bounds, with explicit limitations documented.
3. Decoupled classification: score < threshold -> NORMAL, score >= threshold -> DEFECTIVE.
4. Complete calibration artifact serialization and deserialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class InspectionStatus(str, Enum):
    """Standardized inspection outcome."""

    NORMAL = "NORMAL"
    DEFECTIVE = "DEFECTIVE"


class CalibrationMethod(str, Enum):
    """Supported threshold calibration methods."""

    AUTO = "auto"
    YOUDEN_J = "youden_j"
    BALANCED_ACCURACY = "balanced_accuracy"
    F1_OPTIMAL = "f1_optimal"
    NORMAL_K_SIGMA = "normal_k_sigma"
    NORMAL_PERCENTILE = "normal_percentile"
    NORMAL_TUKEY_IQR = "normal_tukey_iqr"


@dataclass
class ClassificationResult:
    """Detailed result of classifying a single inspection anomaly score."""

    anomaly_score: float
    threshold: float
    status: InspectionStatus
    is_defective: bool
    margin: float  # anomaly_score - threshold (> 0 means defective, < 0 means normal)
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "anomaly_score": float(self.anomaly_score),
            "threshold": float(self.threshold),
            "status": self.status.value,
            "is_defective": self.is_defective,
            "margin": float(self.margin),
        }


def compute_score_statistics(scores: Sequence[float]) -> dict[str, float]:
    """Compute empirical distribution statistics for a sequence of scores."""
    if scores is None or len(scores) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "iqr": 0.0,
        }

    arr = np.array(scores, dtype=np.float64)
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))

    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p25": p25,
        "p75": p75,
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "iqr": float(p75 - p25),
    }


@dataclass
class ThresholdCalibrationResult:
    """Validated product calibration metadata and operating threshold."""

    product_name: str
    threshold: float
    calibration_method: str
    is_normal_only: bool
    normal_sample_count: int
    defective_sample_count: int
    normal_score_stats: dict[str, float]
    defective_score_stats: dict[str, float] | None
    confusion_matrix: dict[str, int]  # TN, FP, FN, TP on calibration data
    metrics: dict[str, float | None]  # FPR, FNR, specificity, sensitivity, balanced_acc, etc.
    objective_metric_name: str
    objective_metric_value: float | None
    optimal_plateau: list[float] | None = None  # [min_threshold, max_threshold] achieving max objective
    calibration_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    calibration_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def classify(
        self,
        anomaly_score: float,
        image_path: str | None = None,
    ) -> ClassificationResult:
        """Classify an anomaly score using this calibrated threshold."""
        score_val = float(anomaly_score)
        threshold_val = float(self.threshold)
        is_defective = score_val >= threshold_val
        status = InspectionStatus.DEFECTIVE if is_defective else InspectionStatus.NORMAL
        margin = score_val - threshold_val

        return ClassificationResult(
            anomaly_score=score_val,
            threshold=threshold_val,
            status=status,
            is_defective=is_defective,
            margin=margin,
            image_path=image_path,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration artifact to serializable dictionary."""
        return {
            "product_name": self.product_name,
            "threshold": float(self.threshold),
            "calibration_method": self.calibration_method,
            "is_normal_only": self.is_normal_only,
            "normal_sample_count": self.normal_sample_count,
            "defective_sample_count": self.defective_sample_count,
            "normal_score_stats": self.normal_score_stats,
            "defective_score_stats": self.defective_score_stats,
            "confusion_matrix": self.confusion_matrix,
            "metrics": self.metrics,
            "objective_metric_name": self.objective_metric_name,
            "objective_metric_value": self.objective_metric_value,
            "optimal_plateau": self.optimal_plateau,
            "calibration_timestamp": self.calibration_timestamp,
            "calibration_metadata": self.calibration_metadata,
            "warnings": self.warnings,
        }

    def save_json(self, path: str | Path, indent: int = 2) -> Path:
        """Save calibrated threshold artifact to a JSON file."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdCalibrationResult:
        """Deserialize from dictionary."""
        return cls(
            product_name=data["product_name"],
            threshold=float(data["threshold"]),
            calibration_method=data["calibration_method"],
            is_normal_only=bool(data["is_normal_only"]),
            normal_sample_count=int(data["normal_sample_count"]),
            defective_sample_count=int(data["defective_sample_count"]),
            normal_score_stats=data["normal_score_stats"],
            defective_score_stats=data.get("defective_score_stats"),
            confusion_matrix=data["confusion_matrix"],
            metrics=data["metrics"],
            objective_metric_name=data.get("objective_metric_name", "unknown"),
            objective_metric_value=data.get("objective_metric_value"),
            optimal_plateau=data.get("optimal_plateau"),
            calibration_timestamp=data.get("calibration_timestamp", ""),
            calibration_metadata=data.get("calibration_metadata", {}),
            warnings=data.get("warnings", []),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> ThresholdCalibrationResult:
        """Load calibrated threshold artifact from a JSON file."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Calibration artifact not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def classify_score(
    anomaly_score: float,
    calibrated_threshold: float | ThresholdCalibrationResult,
) -> str:
    """Classify a continuous PatchCore anomaly score against a calibrated threshold.

    Strictly satisfies:
        score < threshold  -> NORMAL
        score >= threshold -> DEFECTIVE

    Args:
        anomaly_score: Continuous anomaly score from PatchCore.
        calibrated_threshold: Numerical threshold or ThresholdCalibrationResult artifact.

    Returns:
        "NORMAL" or "DEFECTIVE".
    """
    threshold_val = (
        calibrated_threshold.threshold
        if isinstance(calibrated_threshold, ThresholdCalibrationResult)
        else float(calibrated_threshold)
    )

    if float(anomaly_score) < threshold_val:
        return InspectionStatus.NORMAL.value
    return InspectionStatus.DEFECTIVE.value


def _evaluate_candidate_threshold(
    threshold: float,
    normal_scores: np.ndarray,
    defective_scores: np.ndarray,
) -> dict[str, Any]:
    """Compute confusion matrix and performance metrics at a candidate threshold."""
    # Normal: positive means defective, so normal score < threshold is TN, >= threshold is FP
    tn = int(np.sum(normal_scores < threshold))
    fp = int(np.sum(normal_scores >= threshold))
    n_normal = len(normal_scores)

    # Defective: defective score >= threshold is TP, < threshold is FN
    tp = int(np.sum(defective_scores >= threshold))
    fn = int(np.sum(defective_scores < threshold))
    n_defective = len(defective_scores)

    tpr = tp / n_defective if n_defective > 0 else 0.0
    tnr = tn / n_normal if n_normal > 0 else 0.0
    fpr = fp / n_normal if n_normal > 0 else 0.0
    fnr = fn / n_defective if n_defective > 0 else 0.0

    balanced_acc = (tpr + tnr) / 2.0
    youden_j = tpr + tnr - 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2.0 * (precision * tpr) / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (n_normal + n_defective) if (n_normal + n_defective) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "balanced_accuracy": float(balanced_acc),
        "youden_j": float(youden_j),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def calibrate_threshold(
    normal_scores: Sequence[float],
    defective_scores: Sequence[float] | None = None,
    product_name: str = "generic_product",
    method: CalibrationMethod | str = CalibrationMethod.AUTO,
    k_sigma: float = 3.0,
    percentile: float = 99.0,
    tukey_iqr_k: float = 1.5,
    calibration_metadata: dict[str, Any] | None = None,
) -> ThresholdCalibrationResult:
    """Calibrate a product-specific anomaly threshold from empirical score distributions.

    Args:
        normal_scores: Sequence of PatchCore continuous scores on normal calibration samples.
        defective_scores: Optional sequence of PatchCore continuous scores on known-defective samples.
        product_name: Product identifier for calibration provenance.
        method: Calibration algorithm. If 'auto', uses 'youden_j' if defective scores are present,
                or 'normal_k_sigma' if only normal scores are provided.
        k_sigma: Multiplier for standard deviation in normal_k_sigma method (default: 3.0).
        percentile: Percentile value for normal_percentile method (default: 99.0).
        tukey_iqr_k: Multiplier for IQR in normal_tukey_iqr method (default: 1.5).
        calibration_metadata: Additional provenance, hardware, model, and dataset metadata.

    Returns:
        ThresholdCalibrationResult containing the operating threshold and complete statistical diagnostics.
    """
    if normal_scores is None or len(normal_scores) == 0:
        raise ValueError("Cannot calibrate threshold without normal calibration scores.")

    norm_arr = np.array(normal_scores, dtype=np.float64)
    has_defective = defective_scores is not None and len(defective_scores) > 0
    defect_arr = (
        np.array(defective_scores, dtype=np.float64)
        if has_defective
        else np.empty(0, dtype=np.float64)
    )

    if isinstance(method, str):
        method_enum = CalibrationMethod(method.lower())
    else:
        method_enum = method

    meta = dict(calibration_metadata or {})
    warnings_list: list[str] = []

    norm_stats = compute_score_statistics(norm_arr)
    defect_stats = compute_score_statistics(defect_arr) if has_defective else None

    # Handle method resolution under 'auto'
    if method_enum == CalibrationMethod.AUTO:
        if has_defective:
            method_enum = CalibrationMethod.YOUDEN_J
        else:
            method_enum = CalibrationMethod.NORMAL_K_SIGMA

    # Case 1: Defective calibration scores are NOT provided (Normal-Only Calibration)
    if not has_defective:
        if method_enum in (CalibrationMethod.YOUDEN_J, CalibrationMethod.BALANCED_ACCURACY, CalibrationMethod.F1_OPTIMAL):
            warnings_list.append(
                f"Requested method '{method_enum.value}' requires defective calibration samples, "
                "which were not provided. Falling back to 'normal_k_sigma'."
            )
            method_enum = CalibrationMethod.NORMAL_K_SIGMA

        warning_note = (
            "LIMITATION WARNING: Threshold calibrated using normal reference samples only. "
            "No defective samples were available during calibration; false negative rate and "
            "sensitivity cannot be empirically validated until defective samples are observed."
        )
        warnings_list.append(warning_note)

        if method_enum == CalibrationMethod.NORMAL_K_SIGMA:
            chosen_threshold = norm_stats["mean"] + k_sigma * norm_stats["std"]
            obj_name = f"normal_mean_{k_sigma}sigma"
            obj_val = float(chosen_threshold)
        elif method_enum == CalibrationMethod.NORMAL_PERCENTILE:
            chosen_threshold = float(np.percentile(norm_arr, percentile))
            obj_name = f"normal_percentile_{percentile}"
            obj_val = float(chosen_threshold)
        elif method_enum == CalibrationMethod.NORMAL_TUKEY_IQR:
            chosen_threshold = norm_stats["p75"] + tukey_iqr_k * norm_stats["iqr"]
            obj_name = f"normal_tukey_iqr_{tukey_iqr_k}"
            obj_val = float(chosen_threshold)
        else:
            chosen_threshold = norm_stats["mean"] + 3.0 * norm_stats["std"]
            obj_name = "normal_mean_3.0sigma"
            obj_val = float(chosen_threshold)

        # On normal calibration data:
        tn = int(np.sum(norm_arr < chosen_threshold))
        fp = int(np.sum(norm_arr >= chosen_threshold))
        fpr = fp / len(norm_arr) if len(norm_arr) > 0 else 0.0
        specificity = tn / len(norm_arr) if len(norm_arr) > 0 else 0.0

        return ThresholdCalibrationResult(
            product_name=product_name,
            threshold=float(chosen_threshold),
            calibration_method=method_enum.value,
            is_normal_only=True,
            normal_sample_count=len(norm_arr),
            defective_sample_count=0,
            normal_score_stats=norm_stats,
            defective_score_stats=None,
            confusion_matrix={"tn": tn, "fp": fp, "fn": 0, "tp": 0},
            metrics={
                "false_positive_rate": float(fpr),
                "false_negative_rate": None,
                "specificity": float(specificity),
                "sensitivity": None,
                "balanced_accuracy": None,
                "precision": None,
                "f1": None,
                "accuracy": float(specificity),
                "youden_j": None,
            },
            objective_metric_name=obj_name,
            objective_metric_value=obj_val,
            optimal_plateau=None,
            calibration_metadata=meta,
            warnings=warnings_list,
        )

    # Case 2: Both normal and defective calibration scores are available
    # Construct empirical candidate cutpoints from sorted unique scores
    combined_unique = np.unique(np.concatenate([norm_arr, defect_arr]))
    combined_unique.sort()

    # Midpoints between consecutive distinct empirical scores
    midpoints = (combined_unique[:-1] + combined_unique[1:]) / 2.0
    # Include bounds slightly outside min and max to evaluate edge operating points
    eps = 1e-4
    candidates = np.concatenate([
        [combined_unique[0] - eps],
        midpoints,
        [combined_unique[-1] + eps],
    ])

    eval_results = [
        _evaluate_candidate_threshold(cand, norm_arr, defect_arr)
        for cand in candidates
    ]

    # Select candidate based on objective function
    if method_enum in (CalibrationMethod.YOUDEN_J, CalibrationMethod.BALANCED_ACCURACY):
        obj_key = "youden_j" if method_enum == CalibrationMethod.YOUDEN_J else "balanced_accuracy"
        obj_name = obj_key
    elif method_enum == CalibrationMethod.F1_OPTIMAL:
        obj_key = "f1"
        obj_name = "f1"
    else:
        # If user explicitly requested a normal-only method despite having defective samples
        obj_key = "youden_j"
        obj_name = "youden_j"
        warnings_list.append(
            f"Defective scores were provided, but method '{method_enum.value}' was requested. "
            "Using Youden's J optimization instead to exploit labeled defective evidence."
        )
        method_enum = CalibrationMethod.YOUDEN_J

    max_obj_val = max(res[obj_key] for res in eval_results)
    # Find all candidates achieving maximal objective value (the optimal plateau)
    optimal_candidates = [
        res for res in eval_results if np.isclose(res[obj_key], max_obj_val, atol=1e-7)
    ]

    # Extract the plateau bounds
    plateau_thresholds = [res["threshold"] for res in optimal_candidates]
    min_opt_thresh = min(plateau_thresholds)
    max_opt_thresh = max(plateau_thresholds)

    # Statistical Tie-Breaking:
    # Multiple cutpoints frequently achieve the same maximal Youden's J / balanced accuracy
    # (e.g. when there is an empirical separation margin between classes).
    # Selecting the midpoint of the optimal plateau maximizes the geometric margin to the
    # neighboring normal and defective score distributions (analogous to the maximum-margin
    # optimal separating hyperplane in 1D).
    if np.isclose(min_opt_thresh, max_opt_thresh, atol=1e-7):
        selected_threshold = min_opt_thresh
        optimal_plateau = [float(min_opt_thresh), float(max_opt_thresh)]
    else:
        selected_threshold = (min_opt_thresh + max_opt_thresh) / 2.0
        optimal_plateau = [float(min_opt_thresh), float(max_opt_thresh)]

    # Final evaluation at the exact selected operating threshold
    final_eval = _evaluate_candidate_threshold(selected_threshold, norm_arr, defect_arr)

    return ThresholdCalibrationResult(
        product_name=product_name,
        threshold=float(selected_threshold),
        calibration_method=method_enum.value,
        is_normal_only=False,
        normal_sample_count=len(norm_arr),
        defective_sample_count=len(defect_arr),
        normal_score_stats=norm_stats,
        defective_score_stats=defect_stats,
        confusion_matrix={
            "tn": final_eval["tn"],
            "fp": final_eval["fp"],
            "fn": final_eval["fn"],
            "tp": final_eval["tp"],
        },
        metrics={
            "false_positive_rate": final_eval["fpr"],
            "false_negative_rate": final_eval["fnr"],
            "specificity": final_eval["tnr"],
            "sensitivity": final_eval["tpr"],
            "balanced_accuracy": final_eval["balanced_accuracy"],
            "precision": final_eval["precision"],
            "f1": final_eval["f1"],
            "accuracy": final_eval["accuracy"],
            "youden_j": final_eval["youden_j"],
        },
        objective_metric_name=obj_name,
        objective_metric_value=float(max_obj_val),
        optimal_plateau=optimal_plateau,
        calibration_metadata=meta,
        warnings=warnings_list,
    )
