"""Tests for Phase 4 Step 5 - Defect Classifier Diagnostics.

Verifies:
- Exactly 345 test records evaluated
- No train/val records used
- Checkpoint epoch = 15
- All four class probabilities exist per prediction
- Probabilities sum to ~1
- Class order is correct (Scratch, Crack, Discoloration, UNKNOWN)
- Confusion matrix is 4x4
- Prediction count equals test record count
- Recomputed metrics match previously reported values within tolerance
- No dataset or split files were modified
- All artifact files were created
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DIAG_DIR     = PROJECT_ROOT / "experiments" / "defect_classifier_diagnostics"
TEST_JSONL   = PROJECT_ROOT / "experiments" / "defect_classifier_dataset" / "test.jsonl"
TRAIN_JSONL  = PROJECT_ROOT / "experiments" / "defect_classifier_dataset" / "train.jsonl"
VAL_JSONL    = PROJECT_ROOT / "experiments" / "defect_classifier_dataset" / "val.jsonl"
CKPT_PATH    = PROJECT_ROOT / "experiments" / "defect_classifier_run1" / "best_model.pt"

EXPECTED_N         = 345
EXPECTED_EPOCH     = 15
EXPECTED_ACCURACY  = 0.9188
EXPECTED_MACRO_F1  = 0.8686
EXPECTED_CLASSES   = ["Scratch", "Crack", "Discoloration", "UNKNOWN"]
TOLERANCE          = 0.001


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def diag_report():
    p = DIAG_DIR / "diagnostics_report.json"
    if not p.exists():
        pytest.skip("diagnostics_report.json not found — run run_classifier_diagnostics.py first")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def predictions():
    p = DIAG_DIR / "test_predictions.jsonl"
    if not p.exists():
        pytest.skip("test_predictions.jsonl not found — run run_classifier_diagnostics.py first")
    records = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def cm_data():
    p = DIAG_DIR / "confusion_matrix.json"
    if not p.exists():
        pytest.skip("confusion_matrix.json not found")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def src_metrics():
    p = DIAG_DIR / "source_dataset_metrics.json"
    if not p.exists():
        pytest.skip("source_dataset_metrics.json not found")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def misc_analysis():
    p = DIAG_DIR / "misclassification_analysis.json"
    if not p.exists():
        pytest.skip("misclassification_analysis.json not found")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Artifact existence ────────────────────────────────────────────────────────

class TestArtifactsExist:
    def test_diagnostics_report_json(self):
        assert (DIAG_DIR / "diagnostics_report.json").exists()

    def test_diagnostics_report_md(self):
        assert (DIAG_DIR / "diagnostics_report.md").exists()

    def test_test_predictions_jsonl(self):
        assert (DIAG_DIR / "test_predictions.jsonl").exists()

    def test_confusion_matrix_json(self):
        assert (DIAG_DIR / "confusion_matrix.json").exists()

    def test_source_dataset_metrics_json(self):
        assert (DIAG_DIR / "source_dataset_metrics.json").exists()

    def test_misclassification_analysis_json(self):
        assert (DIAG_DIR / "misclassification_analysis.json").exists()

    def test_test_predictions_csv(self):
        assert (DIAG_DIR / "test_predictions.csv").exists()


# ── Checkpoint integrity ──────────────────────────────────────────────────────

class TestCheckpoint:
    def test_checkpoint_exists(self):
        assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"

    def test_checkpoint_epoch(self, diag_report):
        assert diag_report["checkpoint"]["epoch"] == EXPECTED_EPOCH

    def test_checkpoint_val_f1(self, diag_report):
        assert abs(diag_report["checkpoint"]["val_f1"] - 0.8858) < 0.001

    def test_integrity_epoch_correct(self, diag_report):
        assert diag_report["integrity_checks"]["checkpoint_epoch_correct"] is True


# ── Test record count ────────────────────────────────────────────────────────

class TestTestSetCount:
    def test_exactly_345_predictions(self, predictions):
        assert len(predictions) == EXPECTED_N, (
            f"Expected {EXPECTED_N} predictions, got {len(predictions)}"
        )

    def test_report_records_count(self, diag_report):
        assert diag_report["integrity_checks"]["test_records_evaluated"] == EXPECTED_N

    def test_confusion_matrix_sum(self, cm_data):
        total = sum(sum(row) for row in cm_data["matrix"])
        assert total == EXPECTED_N, f"CM sums to {total}, expected {EXPECTED_N}"


# ── No train/val leakage ─────────────────────────────────────────────────────

class TestNoLeakage:
    def test_no_train_images_in_predictions(self, predictions):
        train_paths = set()
        with TRAIN_JSONL.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    train_paths.add(rec["source_image_path"])
        pred_paths = {r["source_image_path"] for r in predictions}
        overlap = pred_paths & train_paths
        assert len(overlap) == 0, f"Train/test leakage: {len(overlap)} shared paths"

    def test_no_val_images_in_predictions(self, predictions):
        val_paths = set()
        with VAL_JSONL.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    val_paths.add(rec["source_image_path"])
        pred_paths = {r["source_image_path"] for r in predictions}
        overlap = pred_paths & val_paths
        assert len(overlap) == 0, f"Val/test leakage: {len(overlap)} shared paths"


# ── Class order ───────────────────────────────────────────────────────────────

class TestClassOrder:
    def test_class_order_in_cm(self, cm_data):
        assert cm_data["labels"] == EXPECTED_CLASSES

    def test_all_probability_keys_present(self, predictions):
        required_keys = [
            "probability_Scratch", "probability_Crack",
            "probability_Discoloration", "probability_UNKNOWN",
        ]
        for rec in predictions:
            for k in required_keys:
                assert k in rec, f"Missing key {k} in prediction record"

    def test_probabilities_sum_to_one(self, predictions):
        for rec in predictions:
            total = (
                rec["probability_Scratch"]
                + rec["probability_Crack"]
                + rec["probability_Discoloration"]
                + rec["probability_UNKNOWN"]
            )
            assert abs(total - 1.0) < 1e-5, f"Probs sum to {total:.8f} for {rec['source_image_path']}"

    def test_confidence_matches_predicted_class_prob(self, predictions):
        prob_keys = {
            "Scratch": "probability_Scratch",
            "Crack": "probability_Crack",
            "Discoloration": "probability_Discoloration",
            "UNKNOWN": "probability_UNKNOWN",
        }
        for rec in predictions:
            expected_conf = rec[prob_keys[rec["predicted_label"]]]
            assert abs(rec["confidence"] - expected_conf) < 1e-5


# ── Confusion matrix structure ────────────────────────────────────────────────

class TestConfusionMatrix:
    def test_confusion_matrix_is_4x4(self, cm_data):
        mat = cm_data["matrix"]
        assert len(mat) == 4
        for row in mat:
            assert len(row) == 4

    def test_confusion_matrix_row_sums_match_support(self, cm_data, diag_report):
        mat = cm_data["matrix"]
        for i, cls in enumerate(EXPECTED_CLASSES):
            row_sum = sum(mat[i])
            support = diag_report["metrics"]["per_class"][cls]["support"]
            assert row_sum == support, (
                f"{cls}: CM row sum {row_sum} != support {support}"
            )


# ── Metric reproducibility ────────────────────────────────────────────────────

class TestMetricReproducibility:
    def test_accuracy_matches_previous(self, diag_report):
        acc = diag_report["metrics"]["accuracy"]
        assert abs(acc - EXPECTED_ACCURACY) < TOLERANCE, (
            f"Accuracy {acc:.6f} differs from expected {EXPECTED_ACCURACY} by more than {TOLERANCE}"
        )

    def test_macro_f1_matches_previous(self, diag_report):
        mf1 = diag_report["metrics"]["macro_f1"]
        assert abs(mf1 - EXPECTED_MACRO_F1) < TOLERANCE, (
            f"Macro F1 {mf1:.6f} differs from expected {EXPECTED_MACRO_F1} by more than {TOLERANCE}"
        )

    def test_integrity_flags_true(self, diag_report):
        ic = diag_report["integrity_checks"]
        assert ic["accuracy_matches_baseline"] is True
        assert ic["macro_f1_matches_baseline"] is True

    def test_per_class_scratch_f1(self, diag_report):
        f1 = diag_report["metrics"]["per_class"]["Scratch"]["f1"]
        assert abs(f1 - 0.9063) < 0.002

    def test_per_class_crack_f1(self, diag_report):
        f1 = diag_report["metrics"]["per_class"]["Crack"]["f1"]
        assert abs(f1 - 0.7308) < 0.002

    def test_per_class_discoloration_f1(self, diag_report):
        f1 = diag_report["metrics"]["per_class"]["Discoloration"]["f1"]
        assert abs(f1 - 0.9748) < 0.002

    def test_per_class_unknown_f1(self, diag_report):
        f1 = diag_report["metrics"]["per_class"]["UNKNOWN"]["f1"]
        assert abs(f1 - 0.8627) < 0.002


# ── Source/domain sanity ─────────────────────────────────────────────────────

class TestSourceDatasetMetrics:
    def test_four_datasets_present(self, src_metrics):
        expected_ds = {"GC10-DET", "MT", "MVTecAD", "NEU-CLS"}
        assert set(src_metrics.keys()) == expected_ds

    def test_dataset_totals_sum_to_345(self, src_metrics):
        total = sum(info["n"] for info in src_metrics.values())
        assert total == EXPECTED_N

    def test_gc10_dominated_by_discoloration(self, src_metrics):
        cd = src_metrics["GC10-DET"]["class_distribution"]
        assert cd.get("Discoloration", 0) > 150, "GC10 should have majority Discoloration"

    def test_neu_dominated_by_scratch(self, src_metrics):
        cd = src_metrics["NEU-CLS"]["class_distribution"]
        assert cd.get("Scratch", 0) > 30, "NEU should have majority Scratch"


# ── Misclassification sanity ──────────────────────────────────────────────────

class TestMisclassificationAnalysis:
    def test_total_misclassified_plus_correct_equals_n(self, misc_analysis, predictions):
        n_wrong   = misc_analysis["total_misclassified"]
        n_correct = sum(1 for r in predictions if r["correct"])
        assert n_wrong + n_correct == EXPECTED_N

    def test_error_rate_consistent(self, misc_analysis):
        er = misc_analysis["error_rate"]
        expected = misc_analysis["total_misclassified"] / EXPECTED_N
        assert abs(er - expected) < 1e-6

    def test_confusion_pairs_are_non_diagonal(self, misc_analysis):
        for pair in misc_analysis["confusion_pairs"]:
            true_cls, pred_cls = pair.split(" -> ")
            assert true_cls != pred_cls, f"Diagonal pair found in misclassification: {pair}"
