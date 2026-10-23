"""Unit tests for Phase 2 PatchCore experiment selection and pipeline logic."""

from pathlib import Path
import sys

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from run_patchcore_experiment import select_controlled_dataset


def test_controlled_dataset_counts():
    """Verify exact counts required for Phase 2 controlled experiment."""
    ref_paths, normal_val_paths, defect_val_paths = select_controlled_dataset()

    assert len(ref_paths) == 20, f"Expected exactly 20 reference images, got {len(ref_paths)}"
    assert len(normal_val_paths) == 20, f"Expected exactly 20 normal validation images, got {len(normal_val_paths)}"
    assert len(defect_val_paths) == 63, f"Expected exactly 63 defective validation images, got {len(defect_val_paths)}"


def test_reference_images_are_defect_free():
    """Verify that all 20 reference images originate exclusively from train/good."""
    ref_paths, _, _ = select_controlled_dataset()

    for p in ref_paths:
        parts = Path(p).parts
        assert "train" in parts, f"Reference image {p} not in train directory"
        assert "good" in parts, f"Reference image {p} not in good directory"


def test_no_leakage_across_partitions():
    """Verify strict mutual exclusion between reference, normal val, and defective val."""
    ref_paths, normal_val_paths, defect_val_paths = select_controlled_dataset()

    ref_set = set(Path(p).resolve() for p in ref_paths)
    norm_val_set = set(Path(p).resolve() for p in normal_val_paths)
    defect_val_set = set(Path(p).resolve() for p in defect_val_paths)

    assert not (ref_set & norm_val_set), "Leakage detected between reference and normal validation sets"
    assert not (ref_set & defect_val_set), "Leakage detected between reference and defective validation sets"
    assert not (norm_val_set & defect_val_set), "Overlap detected between normal and defective validation sets"


def test_deterministic_reproducibility():
    """Verify repeated selection produces identical lists in exact order."""
    run1 = select_controlled_dataset()
    run2 = select_controlled_dataset()

    assert [str(p) for p in run1[0]] == [str(p) for p in run2[0]]
    assert [str(p) for p in run1[1]] == [str(p) for p in run2[1]]
    assert [str(p) for p in run1[2]] == [str(p) for p in run2[2]]


def test_files_exist_on_disk():
    """Verify all selected image files physically exist on disk."""
    ref_paths, normal_val_paths, defect_val_paths = select_controlled_dataset()

    for p in ref_paths + normal_val_paths + defect_val_paths:
        assert Path(p).is_file(), f"File does not exist: {p}"
