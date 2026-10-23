"""Unit and integrity tests for Phase 4 Step 1 defect classifier dataset audit."""

import json
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = WORKSPACE_ROOT / "experiments" / "defect_classifier_audit"
CANDIDATE_MANIFEST_PATH = AUDIT_DIR / "candidate_classifier_manifest.jsonl"
AUDIT_JSON_PATH = AUDIT_DIR / "audit_report.json"
AUDIT_MD_PATH = AUDIT_DIR / "audit_report.md"

ALLOWED_TARGET_LABELS = {"Scratch", "Crack", "Discoloration", "UNKNOWN"}


def test_audit_artifacts_exist():
    """Verify that audit report artifacts were generated."""
    assert AUDIT_JSON_PATH.is_file(), f"Missing audit_report.json at {AUDIT_JSON_PATH}"
    assert AUDIT_MD_PATH.is_file(), f"Missing audit_report.md at {AUDIT_MD_PATH}"
    assert CANDIDATE_MANIFEST_PATH.is_file(), f"Missing candidate manifest at {CANDIDATE_MANIFEST_PATH}"


def test_candidate_manifest_labels_and_counts():
    """Verify candidate manifest contains only approved taxonomy and exact audited counts."""
    records = []
    with CANDIDATE_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == 4634, f"Expected 4634 candidate records, got {len(records)}"

    label_counts = {}
    for r in records:
        lbl = r.get("candidate_target_label")
        assert lbl in ALLOWED_TARGET_LABELS, f"Illegal label {lbl} found in candidate manifest!"
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    assert label_counts["Scratch"] == 419
    assert label_counts["Crack"] == 141
    assert label_counts["Discoloration"] == 1204
    assert label_counts["UNKNOWN"] == 2870


def test_candidate_manifest_no_duplicate_paths_or_sample_ids():
    """Verify candidate manifest has strictly zero duplicate source paths or sample IDs."""
    records = []
    with CANDIDATE_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    paths = [r["source_image_path"] for r in records]
    sample_ids = [r["sample_id"] for r in records]

    assert len(paths) == len(set(paths)), "Duplicate source_image_path detected in candidate manifest!"
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample_id detected in candidate manifest!"


def test_candidate_manifest_no_contaminated_sources():
    """Verify no records use the rejected Surface/Magnetic-Tile-Defect or defect-categories."""
    with CANDIDATE_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r["source_image_path"]
            assert "Surface" not in src, f"Contaminated Surface directory found: {src}"
            assert "defect-categories" not in src, f"Contaminated defect-categories found: {src}"


def test_all_candidate_images_physically_exist():
    """Verify 100% of candidate images exist on the filesystem."""
    with CANDIDATE_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p = Path(r["source_image_path"])
            assert p.is_file(), f"Candidate image not found on disk: {p}"


def test_audit_report_json_structure():
    """Verify audit_report.json conforms to expected schema and findings."""
    with AUDIT_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_manifest_records"] == 9760
    assert data["governance_summary"]["direct_approved_records"] == 1764
    assert data["governance_summary"]["unknown_defect_source_records"] == 2870
    assert data["governance_summary"]["review_ambiguous_records"] == 909
    assert data["governance_summary"]["normal_reference_records"] == 4217

    assert data["target_class_counts"]["Scratch"] == 419
    assert data["target_class_counts"]["Crack"] == 141
    assert data["target_class_counts"]["Discoloration"] == 1204
    assert data["target_class_counts"]["UNKNOWN_available"] == 2870

    assert data["integrity_and_leakage_findings"]["all_files_exist_on_disk"] is True
    assert data["integrity_and_leakage_findings"]["duplicate_source_paths"] == 0
    assert data["integrity_and_leakage_findings"]["duplicate_sample_ids"] == 0
    assert data["integrity_and_leakage_findings"]["surface_mt_contamination_detected"] is False

    assert data["split_assessment"]["is_existing_split_suitable"] is False
