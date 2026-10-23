"""Integration tests for Phase 3 threshold calibration experiment partitioning and execution."""

from pathlib import Path
import sys

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from run_threshold_calibration_experiment import (
    partition_calibration_and_evaluation_sets,
    load_actual_continuous_scores,
)


def test_partition_counts_and_stratification():
    """Verify exact counts and stratified categories for calibration and evaluation sets."""
    part = partition_calibration_and_evaluation_sets()

    ref = part["reference"]
    cal_norm = part["calibration_normal"]
    cal_defect = part["calibration_defective"]
    eval_norm = part["evaluation_normal"]
    eval_defect = part["evaluation_defective"]

    # Check exact counts
    assert len(ref) == 20, f"Expected 20 reference images, got {len(ref)}"
    assert len(cal_norm) == 10, f"Expected 10 calibration normal images, got {len(cal_norm)}"
    assert len(cal_defect) == 32, f"Expected 32 calibration defective images, got {len(cal_defect)}"
    assert len(eval_norm) == 10, f"Expected 10 evaluation normal images, got {len(eval_norm)}"
    assert len(eval_defect) == 31, f"Expected 31 evaluation defective images, got {len(eval_defect)}"

    # Check defect category stratification
    cal_cats = [p.parent.name for p in cal_defect]
    eval_cats = [p.parent.name for p in eval_defect]

    assert cal_cats.count("broken_large") == 10
    assert eval_cats.count("broken_large") == 10

    assert cal_cats.count("broken_small") == 11
    assert eval_cats.count("broken_small") == 11

    assert cal_cats.count("contamination") == 11
    assert eval_cats.count("contamination") == 10


def test_strict_partition_mutual_exclusion():
    """Verify strictly zero data leakage across reference, calibration, and evaluation sets."""
    part = partition_calibration_and_evaluation_sets()

    ref_set = set(p.resolve() for p in part["reference"])
    cal_set = set(p.resolve() for p in (part["calibration_normal"] + part["calibration_defective"]))
    eval_set = set(p.resolve() for p in (part["evaluation_normal"] + part["evaluation_defective"]))

    assert not (ref_set & cal_set), "Leakage between reference and calibration"
    assert not (ref_set & eval_set), "Leakage between reference and evaluation"
    assert not (cal_set & eval_set), "Leakage between calibration and evaluation"


def test_partition_deterministic_reproducibility():
    """Verify partition produces identical lists across multiple invocations."""
    run1 = partition_calibration_and_evaluation_sets()
    run2 = partition_calibration_and_evaluation_sets()

    for k in run1:
        assert [str(p) for p in run1[k]] == [str(p) for p in run2[k]]


def test_load_continuous_scores_from_phase2():
    """Verify loading continuous PatchCore scores from Phase 2 results."""
    part = partition_calibration_and_evaluation_sets()
    score_map = load_actual_continuous_scores(part)

    # All 83 validation images must be present in score_map
    all_paths = (
        part["calibration_normal"]
        + part["calibration_defective"]
        + part["evaluation_normal"]
        + part["evaluation_defective"]
    )
    assert len(all_paths) == 83

    for p in all_paths:
        norm_p = str(p.resolve()).replace("\\", "/")
        assert norm_p in score_map
        score = score_map[norm_p]
        assert isinstance(score, float)
        assert score > 0.0
