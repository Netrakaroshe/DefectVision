"""Comprehensive unit and integrity tests for Phase 4 Step 2 classifier dataset splits."""

import hashlib
import json
from pathlib import Path
import sys
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from create_classifier_splits import (
    APPROVED_CLASSES,
    OUTPUT_DIR,
    RANDOM_SEED,
    REPORT_MD_PATH,
    SUMMARY_JSON_PATH,
    TEST_JSONL_PATH,
    TRAIN_JSONL_PATH,
    VAL_JSONL_PATH,
    build_splits,
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_split_files_exist():
    """Verify all split artifacts exist."""
    assert TRAIN_JSONL_PATH.is_file(), f"Missing train.jsonl at {TRAIN_JSONL_PATH}"
    assert VAL_JSONL_PATH.is_file(), f"Missing val.jsonl at {VAL_JSONL_PATH}"
    assert TEST_JSONL_PATH.is_file(), f"Missing test.jsonl at {TEST_JSONL_PATH}"
    assert SUMMARY_JSON_PATH.is_file(), f"Missing split_summary.json at {SUMMARY_JSON_PATH}"
    assert REPORT_MD_PATH.is_file(), f"Missing split_report.md at {REPORT_MD_PATH}"


def test_split_counts_and_percentages():
    """Verify exact counts and ~70/15/15 ratio."""
    tr = load_jsonl(TRAIN_JSONL_PATH)
    va = load_jsonl(VAL_JSONL_PATH)
    te = load_jsonl(TEST_JSONL_PATH)

    assert len(tr) == 1505, f"Expected 1505 train records, got {len(tr)}"
    assert len(va) == 333, f"Expected 333 val records, got {len(va)}"
    assert len(te) == 345, f"Expected 345 test records, got {len(te)}"

    total = len(tr) + len(va) + len(te)
    assert total == 2183, f"Expected 2183 total records, got {total}"

    tr_pct = len(tr) / total * 100
    va_pct = len(va) / total * 100
    te_pct = len(te) / total * 100

    assert 68.0 <= tr_pct <= 72.0, f"Train percentage {tr_pct:.2f}% out of range"
    assert 14.0 <= va_pct <= 17.0, f"Val percentage {va_pct:.2f}% out of range"
    assert 14.0 <= te_pct <= 17.0, f"Test percentage {te_pct:.2f}% out of range"


def test_strictly_zero_leakage_across_partitions():
    """Verify no path, sample_id, or content-hash overlap between train, val, and test."""
    tr = load_jsonl(TRAIN_JSONL_PATH)
    va = load_jsonl(VAL_JSONL_PATH)
    te = load_jsonl(TEST_JSONL_PATH)

    tr_paths = set(r["source_image_path"] for r in tr)
    va_paths = set(r["source_image_path"] for r in va)
    te_paths = set(r["source_image_path"] for r in te)

    assert not (tr_paths & va_paths), "Path overlap between train and val!"
    assert not (tr_paths & te_paths), "Path overlap between train and test!"
    assert not (va_paths & te_paths), "Path overlap between val and test!"

    tr_ids = set(r["sample_id"] for r in tr)
    va_ids = set(r["sample_id"] for r in va)
    te_ids = set(r["sample_id"] for r in te)

    assert not (tr_ids & va_ids), "Sample ID overlap between train and val!"
    assert not (tr_ids & te_ids), "Sample ID overlap between train and test!"
    assert not (va_ids & te_ids), "Sample ID overlap between val and test!"

    # Verify content hash overlap
    tr_hashes = set(hashlib.md5(Path(r["source_image_path"]).read_bytes()).hexdigest() for r in tr)
    va_hashes = set(hashlib.md5(Path(r["source_image_path"]).read_bytes()).hexdigest() for r in va)
    te_hashes = set(hashlib.md5(Path(r["source_image_path"]).read_bytes()).hexdigest() for r in te)

    assert not (tr_hashes & va_hashes), "Content hash collision between train and val!"
    assert not (tr_hashes & te_hashes), "Content hash collision between train and test!"
    assert not (va_hashes & te_hashes), "Content hash collision between val and test!"


def test_target_labels_restricted_to_approved_taxonomy():
    """Verify target labels across all splits strictly belong to the 4 approved classes."""
    for path in [TRAIN_JSONL_PATH, VAL_JSONL_PATH, TEST_JSONL_PATH]:
        records = load_jsonl(path)
        for r in records:
            lbl = r.get("target_label")
            assert lbl in APPROVED_CLASSES, f"Disallowed target label '{lbl}' in {path.name}"


def test_all_referenced_images_exist_on_disk():
    """Verify 100% of images referenced in train, val, and test physically exist."""
    for path in [TRAIN_JSONL_PATH, VAL_JSONL_PATH, TEST_JSONL_PATH]:
        records = load_jsonl(path)
        for r in records:
            p = Path(r["source_image_path"])
            assert p.is_file(), f"Image file does not exist: {p}"


def test_no_contaminated_sources():
    """Verify no split contains paths from Surface/Magnetic-Tile-Defect or defect-categories."""
    for path in [TRAIN_JSONL_PATH, VAL_JSONL_PATH, TEST_JSONL_PATH]:
        records = load_jsonl(path)
        for r in records:
            p = r["source_image_path"]
            assert "Surface" not in p, f"Contaminated Surface directory in {path.name}: {p}"
            assert "defect-categories" not in p, f"Contaminated defect-categories in {path.name}: {p}"


def test_stratification_representation_across_all_classes():
    """Verify every target class and source dataset is represented in all three splits."""
    tr = load_jsonl(TRAIN_JSONL_PATH)
    va = load_jsonl(VAL_JSONL_PATH)
    te = load_jsonl(TEST_JSONL_PATH)

    for cls in APPROVED_CLASSES:
        n_tr = sum(1 for r in tr if r["target_label"] == cls)
        n_va = sum(1 for r in va if r["target_label"] == cls)
        n_te = sum(1 for r in te if r["target_label"] == cls)

        assert n_tr > 0, f"Class {cls} missing from train split!"
        assert n_va > 0, f"Class {cls} missing from val split!"
        assert n_te > 0, f"Class {cls} missing from test split!"

    for ds in ["GC10-DET", "MT", "MVTecAD", "NEU-CLS"]:
        n_tr = sum(1 for r in tr if r["dataset"] == ds)
        n_va = sum(1 for r in va if r["dataset"] == ds)
        n_te = sum(1 for r in te if r["dataset"] == ds)

        assert n_tr > 0, f"Dataset {ds} missing from train split!"
        assert n_va > 0, f"Dataset {ds} missing from val split!"
        assert n_te > 0, f"Dataset {ds} missing from test split!"


def test_crack_class_properly_represented_in_train():
    """Verify the critical Phase 1 split failure (Crack had 0 in train) is completely fixed."""
    tr = load_jsonl(TRAIN_JSONL_PATH)
    va = load_jsonl(VAL_JSONL_PATH)
    te = load_jsonl(TEST_JSONL_PATH)

    tr_crack = [r for r in tr if r["target_label"] == "Crack"]
    va_crack = [r for r in va if r["target_label"] == "Crack"]
    te_crack = [r for r in te if r["target_label"] == "Crack"]

    assert len(tr_crack) == 97, f"Expected 97 Crack images in train, got {len(tr_crack)}"
    assert len(va_crack) == 22, f"Expected 22 Crack images in val, got {len(va_crack)}"
    assert len(te_crack) == 22, f"Expected 22 Crack images in test, got {len(te_crack)}"
    assert len(tr_crack) + len(va_crack) + len(te_crack) == 141


def test_deterministic_reproducibility():
    """Verify repeated execution with same random seed yields identical splits."""
    run1 = build_splits(RANDOM_SEED)
    run2 = build_splits(RANDOM_SEED)

    tr1 = [r["source_image_path"] for r in run1["train"]]
    tr2 = [r["source_image_path"] for r in run2["train"]]
    assert tr1 == tr2

    va1 = [r["source_image_path"] for r in run1["val"]]
    va2 = [r["source_image_path"] for r in run2["val"]]
    assert va1 == va2

    te1 = [r["source_image_path"] for r in run1["test"]]
    te2 = [r["source_image_path"] for r in run2["test"]]
    assert te1 == te2


def test_unknown_subset_exact_counts():
    """Verify UNKNOWN subset has exactly 419 images matching Scratch count."""
    tr = load_jsonl(TRAIN_JSONL_PATH)
    va = load_jsonl(VAL_JSONL_PATH)
    te = load_jsonl(TEST_JSONL_PATH)

    all_unk = [r for r in (tr + va + te) if r["target_label"] == "UNKNOWN"]
    all_scratch = [r for r in (tr + va + te) if r["target_label"] == "Scratch"]

    assert len(all_unk) == 419, f"Expected 419 UNKNOWN images, got {len(all_unk)}"
    assert len(all_scratch) == 419, f"Expected 419 Scratch images, got {len(all_scratch)}"
    assert len(all_unk) == len(all_scratch)
