from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from defectvision.calibration import (
    CalibrationMethod,
    ClassificationResult,
    InspectionStatus,
    ThresholdCalibrationResult,
    calibrate_threshold,
    classify_score,
)


def test_classify_below_threshold():
    """Verify score < threshold strictly evaluates to NORMAL."""
    threshold = 40.0
    score = 39.999

    status_str = classify_score(score, threshold)
    assert status_str == InspectionStatus.NORMAL.value
    assert status_str == "NORMAL"


def test_classify_at_threshold():
    """Verify score == threshold strictly evaluates to DEFECTIVE."""
    threshold = 40.0
    score = 40.0

    status_str = classify_score(score, threshold)
    assert status_str == InspectionStatus.DEFECTIVE.value
    assert status_str == "DEFECTIVE"


def test_classify_above_threshold():
    """Verify score > threshold strictly evaluates to DEFECTIVE."""
    threshold = 40.0
    score = 40.001

    status_str = classify_score(score, threshold)
    assert status_str == InspectionStatus.DEFECTIVE.value
    assert status_str == "DEFECTIVE"


def test_no_universal_threshold_assumption():
    """Verify threshold varies adaptively based on product-specific score distributions."""
    # Product A: scores distributed around 30 (normal) and 60 (defective)
    norm_a = [28.0, 29.5, 30.2, 31.0, 32.5]
    defect_a = [55.0, 58.0, 60.5, 62.0, 65.0]
    res_a = calibrate_threshold(norm_a, defect_a, product_name="product_a")

    # Product B: scores distributed around 110 (normal) and 180 (defective)
    norm_b = [105.0, 108.0, 110.5, 112.0, 115.0]
    defect_b = [170.0, 175.0, 180.0, 185.0, 190.0]
    res_b = calibrate_threshold(norm_b, defect_b, product_name="product_b")

    # Product C: scores distributed around 5 (normal) and 12 (defective)
    norm_c = [4.0, 4.5, 5.0, 5.2, 5.8]
    defect_c = [10.0, 11.0, 12.0, 13.0, 14.0]
    res_c = calibrate_threshold(norm_c, defect_c, product_name="product_c")

    # Assert distinct thresholds: no fixed constant (e.g. 0.5) is assumed
    assert res_a.threshold != res_b.threshold
    assert res_b.threshold != res_c.threshold
    assert not np.isclose(res_a.threshold, 0.5)
    assert not np.isclose(res_b.threshold, 0.5)
    assert not np.isclose(res_c.threshold, 0.5)

    # All thresholds must separate their respective distributions
    assert max(norm_a) < res_a.threshold < min(defect_a)
    assert max(norm_b) < res_b.threshold < min(defect_b)
    assert max(norm_c) < res_c.threshold < min(defect_c)


def test_calibration_with_both_normal_and_defective_scores():
    """Verify supervised Youden's J calibration with empirical normal and defective scores."""
    norm_scores = [30.0, 32.0, 34.0, 35.0, 36.0, 38.0]
    defect_scores = [44.0, 46.0, 50.0, 55.0, 60.0, 70.0]

    cal_res = calibrate_threshold(
        normal_scores=norm_scores,
        defective_scores=defect_scores,
        product_name="bottle",
        method=CalibrationMethod.YOUDEN_J,
        calibration_metadata={"environment": "test_env"},
    )

    assert not cal_res.is_normal_only
    assert cal_res.normal_sample_count == 6
    assert cal_res.defective_sample_count == 6
    assert cal_res.threshold > max(norm_scores)
    assert cal_res.threshold <= min(defect_scores)

    # Verify confusion matrix on calibration data
    assert cal_res.confusion_matrix["fp"] == 0
    assert cal_res.confusion_matrix["fn"] == 0
    assert cal_res.confusion_matrix["tn"] == 6
    assert cal_res.confusion_matrix["tp"] == 6

    # Verify metrics
    assert cal_res.metrics["false_positive_rate"] == 0.0
    assert cal_res.metrics["false_negative_rate"] == 0.0
    assert cal_res.metrics["balanced_accuracy"] == 1.0
    assert cal_res.metrics["youden_j"] == 1.0
    assert cal_res.objective_metric_name == "youden_j"
    assert cal_res.objective_metric_value == 1.0


def test_behavior_when_defective_calibration_data_are_unavailable():
    """Verify normal-only calibration behavior and explicit limitation documentation."""
    norm_scores = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0]

    cal_res = calibrate_threshold(
        normal_scores=norm_scores,
        defective_scores=None,
        product_name="bottle",
        k_sigma=3.0,
    )

    assert cal_res.is_normal_only
    assert cal_res.defective_sample_count == 0
    assert cal_res.defective_score_stats is None
    assert cal_res.metrics["false_negative_rate"] is None
    assert cal_res.metrics["sensitivity"] is None

    # Check warning documentation
    assert len(cal_res.warnings) > 0
    assert any("LIMITATION WARNING" in w for w in cal_res.warnings)

    # Threshold must be strictly greater than mean of normal scores
    assert cal_res.threshold > np.mean(norm_scores)


def test_deterministic_reproducibility():
    """Verify repeated calibration with identical inputs yields identical thresholds."""
    norm_scores = [29.46, 33.27, 35.12, 37.89, 39.36]
    defect_scores = [41.74, 45.80, 52.33, 62.40, 79.01]

    res1 = calibrate_threshold(norm_scores, defect_scores, product_name="bottle")
    res2 = calibrate_threshold(norm_scores, defect_scores, product_name="bottle")

    assert res1.threshold == res2.threshold
    assert res1.confusion_matrix == res2.confusion_matrix
    assert res1.metrics == res2.metrics
    assert res1.optimal_plateau == res2.optimal_plateau


def test_calibration_metadata_serialization_deserialization():
    """Verify round-trip JSON serialization and deserialization preserves all fields."""
    norm_scores = [30.5, 32.1, 33.8, 35.2]
    defect_scores = [45.0, 50.2, 55.4, 60.1]

    orig = calibrate_threshold(
        normal_scores=norm_scores,
        defective_scores=defect_scores,
        product_name="test_bottle_widget",
        calibration_metadata={"line_id": "line_42", "operator": "QA_Supervisor"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "calibrated_threshold.json"
        saved_p = orig.save_json(json_path)
        assert saved_p.is_file()

        loaded = ThresholdCalibrationResult.load_json(saved_p)

        assert loaded.product_name == orig.product_name
        assert np.isclose(loaded.threshold, orig.threshold)
        assert loaded.calibration_method == orig.calibration_method
        assert loaded.is_normal_only == orig.is_normal_only
        assert loaded.normal_sample_count == orig.normal_sample_count
        assert loaded.defective_sample_count == orig.defective_sample_count
        assert loaded.confusion_matrix == orig.confusion_matrix
        assert loaded.metrics == orig.metrics
        assert loaded.calibration_metadata == orig.calibration_metadata
        assert loaded.optimal_plateau == orig.optimal_plateau


def test_classification_result_object():
    """Verify detailed ClassificationResult structure and margins."""
    norm_scores = [30.0, 32.0]
    defect_scores = [40.0, 50.0]
    cal = calibrate_threshold(norm_scores, defect_scores, product_name="widget")

    res_norm = cal.classify(31.0, image_path="path/to/normal.png")
    assert isinstance(res_norm, ClassificationResult)
    assert res_norm.status == InspectionStatus.NORMAL
    assert not res_norm.is_defective
    assert res_norm.margin < 0
    assert res_norm.image_path == "path/to/normal.png"

    res_defect = cal.classify(45.0, image_path="path/to/defect.png")
    assert res_defect.status == InspectionStatus.DEFECTIVE
    assert res_defect.is_defective
    assert res_defect.margin > 0


def test_empty_normal_scores_raises_value_error():
    """Verify ValueError is raised if normal scores are empty."""
    with pytest.raises(ValueError, match="Cannot calibrate threshold without normal"):
        calibrate_threshold(normal_scores=[])
