"""Phase 4 Step 5 - Defect Classifier Diagnostics.

Loads the frozen best_model.pt (epoch 15) and runs a comprehensive
diagnostic evaluation on the 345-image frozen test set.

DOES NOT retrain the model, modify splits, or modify raw datasets.

Outputs:
  experiments/defect_classifier_diagnostics/
    diagnostics_report.json
    diagnostics_report.md
    test_predictions.jsonl
    confusion_matrix.json
    source_dataset_metrics.json
    misclassification_analysis.json
    test_predictions.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from defectvision.models.defect_classifier import (
    CLASS_TO_IDX,
    DEFAULT_DEFECT_CLASSES,
    IDX_TO_CLASS,
    DefectClassifierDataset,
    DefectClassifierModel,
    get_classifier_transforms,
)

CKPT_PATH   = PROJECT_ROOT / "experiments" / "defect_classifier_run1" / "best_model.pt"
TEST_JSONL  = PROJECT_ROOT / "experiments" / "defect_classifier_dataset" / "test.jsonl"
OUT_DIR     = PROJECT_ROOT / "experiments" / "defect_classifier_diagnostics"

EXPECTED_EPOCH    = 15
EXPECTED_VAL_F1   = 0.8858
EXPECTED_ACCURACY = 0.9188
EXPECTED_MACRO_F1 = 0.8686
EXPECTED_N        = 345
IMAGE_SIZE        = (224, 224)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_inference(model, records, device):
    """Run inference on all records; return (labels, preds, probs_array, paths)."""
    transform = get_classifier_transforms(image_size=IMAGE_SIZE, is_training=False)
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    from PIL import Image as PILImage

    with torch.inference_mode():
        for rec in records:
            img_path = Path(rec["source_image_path"])
            with PILImage.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            label_str = rec.get("target_label") or rec.get("candidate_target_label")
            label_idx = CLASS_TO_IDX.get(label_str, 3)
            pred_idx  = int(np.argmax(probs))

            all_labels.append(label_idx)
            all_preds.append(pred_idx)
            all_probs.append(probs)

    return (
        np.array(all_labels, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_probs, dtype=float),
    )


def build_prediction_records(records, labels, preds, probs):
    out = []
    for i, rec in enumerate(records):
        p = probs[i]
        out.append({
            "sample_id":           rec.get("sample_id", ""),
            "source_image_path":   rec.get("source_image_path", ""),
            "source_dataset":      rec.get("dataset", ""),
            "product":             rec.get("product", ""),
            "original_label":      rec.get("original_label", ""),
            "target_label":        rec.get("target_label") or rec.get("candidate_target_label", ""),
            "predicted_label":     IDX_TO_CLASS[int(preds[i])],
            "true_class_index":    int(labels[i]),
            "predicted_class_index": int(preds[i]),
            "confidence":          float(p[int(preds[i])]),
            "probability_Scratch":      float(p[0]),
            "probability_Crack":        float(p[1]),
            "probability_Discoloration": float(p[2]),
            "probability_UNKNOWN":      float(p[3]),
            "correct":             bool(labels[i] == preds[i]),
            "mapping_status":      rec.get("mapping_status", ""),
            "source_split":        rec.get("source_split", ""),
        })
    return out


def domain_metrics(pred_records, class_names):
    """Per-source-dataset accuracy, macro-F1, class distribution."""
    ds_groups = defaultdict(list)
    for r in pred_records:
        ds_groups[r["source_dataset"]].append(r)

    results = {}
    for ds, recs in sorted(ds_groups.items()):
        true_idx  = [CLASS_TO_IDX.get(r["target_label"], 3) for r in recs]
        pred_idx  = [r["predicted_class_index"] for r in recs]
        n         = len(recs)
        acc       = accuracy_score(true_idx, pred_idx)
        classes_present = sorted(set(true_idx))
        mf1 = f1_score(true_idx, pred_idx, average="macro",
                       labels=list(range(4)), zero_division=0)
        per_cls_f1 = f1_score(true_idx, pred_idx, average=None,
                               labels=list(range(4)), zero_division=0)
        per_cls_prec = precision_score(true_idx, pred_idx, average=None,
                                       labels=list(range(4)), zero_division=0)
        per_cls_rec  = recall_score(true_idx, pred_idx, average=None,
                                    labels=list(range(4)), zero_division=0)

        class_dist = defaultdict(int)
        for r in recs:
            class_dist[r["target_label"]] += 1

        cm = confusion_matrix(true_idx, pred_idx, labels=list(range(4)))

        results[ds] = {
            "n":          n,
            "accuracy":   round(float(acc), 6),
            "macro_f1":   round(float(mf1), 6),
            "class_distribution": dict(class_dist),
            "per_class": {
                class_names[i]: {
                    "precision": round(float(per_cls_prec[i]), 6),
                    "recall":    round(float(per_cls_rec[i]), 6),
                    "f1":        round(float(per_cls_f1[i]), 6),
                }
                for i in range(4)
            },
            "confusion_matrix": cm.tolist(),
        }
    return results


def misclassification_analysis(pred_records):
    wrong = [r for r in pred_records if not r["correct"]]
    pairs = defaultdict(list)
    for r in wrong:
        key = f"{r['target_label']} -> {r['predicted_label']}"
        pairs[key].append(r)

    class_totals = defaultdict(int)
    for r in pred_records:
        class_totals[r["target_label"]] += 1

    pair_summary = {}
    for key, recs in sorted(pairs.items(), key=lambda x: -len(x[1])):
        true_class   = key.split(" -> ")[0]
        ds_breakdown = defaultdict(int)
        orig_labels  = defaultdict(int)
        confs        = [r["confidence"] for r in recs]
        for r in recs:
            ds_breakdown[r["source_dataset"]] += 1
            orig_labels[r["original_label"]] += 1
        pair_summary[key] = {
            "count":             len(recs),
            "pct_of_true_class": round(len(recs) / max(class_totals[true_class], 1) * 100, 2),
            "source_datasets":   dict(ds_breakdown),
            "original_labels":   dict(orig_labels),
            "mean_confidence":   round(float(np.mean(confs)), 4),
            "examples":          [
                {
                    "source_image_path": r["source_image_path"],
                    "source_dataset":    r["source_dataset"],
                    "original_label":    r["original_label"],
                    "confidence":        round(r["confidence"], 4),
                }
                for r in recs[:5]
            ],
        }

    return {
        "total_misclassified": len(wrong),
        "error_rate":          round(len(wrong) / len(pred_records), 6),
        "class_totals":        dict(class_totals),
        "confusion_pairs":     pair_summary,
    }


def confidence_analysis(pred_records):
    correct_confs   = [r["confidence"] for r in pred_records if r["correct"]]
    incorrect_confs = [r["confidence"] for r in pred_records if not r["correct"]]

    def dist_stats(confs):
        if not confs:
            return {}
        arr = np.array(confs)
        return {
            "count":  len(confs),
            "mean":   round(float(arr.mean()), 4),
            "median": round(float(np.median(arr)), 4),
            "std":    round(float(arr.std()),  4),
            "min":    round(float(arr.min()),  4),
            "max":    round(float(arr.max()),  4),
            "pct_below_0.5":  round(float((arr < 0.5).mean()) * 100, 2),
            "pct_below_0.7":  round(float((arr < 0.7).mean()) * 100, 2),
            "pct_above_0.9":  round(float((arr > 0.9).mean()) * 100, 2),
        }

    return {
        "correct":   dist_stats(correct_confs),
        "incorrect": dist_stats(incorrect_confs),
        "misclassified_details": [
            {
                "source_image_path": r["source_image_path"],
                "source_dataset":    r["source_dataset"],
                "true_label":        r["target_label"],
                "predicted_label":   r["predicted_label"],
                "confidence":        round(r["confidence"], 4),
            }
            for r in sorted(
                [r for r in pred_records if not r["correct"]],
                key=lambda x: -x["confidence"],
            )
        ],
    }


def build_markdown(diag):
    m = diag
    lines = [
        "# Phase 4 Step 5 -- Defect Classifier Diagnostics Report",
        "",
        "## 1. Checkpoint Verification",
        f"- Checkpoint: {CKPT_PATH}",
        f"- Saved epoch: {m['checkpoint']['epoch']}",
        f"- Saved val macro-F1: {m['checkpoint']['val_f1']:.6f}",
        f"- Expected epoch: {EXPECTED_EPOCH}  |  Match: {m['checkpoint']['epoch'] == EXPECTED_EPOCH}",
        "",
        "## 2. Test-Set Metrics (345 images)",
        "",
        "### Overall",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {m['metrics']['accuracy']:.4f} |",
        f"| Balanced Accuracy | {m['metrics']['balanced_accuracy']:.4f} |",
        f"| Macro Precision | {m['metrics']['macro_precision']:.4f} |",
        f"| Macro Recall | {m['metrics']['macro_recall']:.4f} |",
        f"| Macro F1 | {m['metrics']['macro_f1']:.4f} |",
        f"| Weighted Precision | {m['metrics']['weighted_precision']:.4f} |",
        f"| Weighted Recall | {m['metrics']['weighted_recall']:.4f} |",
        f"| Weighted F1 | {m['metrics']['weighted_f1']:.4f} |",
        "",
        "### Per-Class",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for cls in DEFAULT_DEFECT_CLASSES:
        pc = m["metrics"]["per_class"][cls]
        lines.append(
            f"| {cls} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |"
        )

    lines += [
        "",
        "### Confusion Matrix (rows=true, cols=predicted)",
        "| True \\ Pred | " + " | ".join(DEFAULT_DEFECT_CLASSES) + " |",
        "|" + "---|" * (len(DEFAULT_DEFECT_CLASSES) + 1),
    ]
    cm = m["confusion_matrix"]["matrix"]
    for i, row in enumerate(cm):
        lines.append(
            "| **" + DEFAULT_DEFECT_CLASSES[i] + "** | " +
            " | ".join(str(v) for v in row) + " |"
        )

    lines += [
        "",
        "## 3. Misclassification Analysis",
        f"- **Total misclassified**: {m['misclassification']['total_misclassified']} / {EXPECTED_N} ({m['misclassification']['error_rate']*100:.2f}%)",
        "",
        "### Confusion Pairs",
        "| Pair | Count | % of True Class | Mean Confidence | Top Source | Top Original Label |",
        "|------|-------|----------------|-----------------|-----------|-------------------|",
    ]
    for pair, info in m["misclassification"]["confusion_pairs"].items():
        top_ds  = max(info["source_datasets"], key=info["source_datasets"].get, default="-")
        top_lbl = max(info["original_labels"], key=info["original_labels"].get, default="-")
        lines.append(
            f"| {pair} | {info['count']} | {info['pct_of_true_class']:.1f}% | "
            f"{info['mean_confidence']:.3f} | {top_ds} | {top_lbl} |"
        )

    lines += [
        "",
        "## 4. Source / Domain Analysis",
        "",
        "### Test-Set Distribution",
        "| Dataset | N | Scratch | Crack | Discoloration | UNKNOWN |",
        "|---------|---|---------|-------|---------------|---------|",
    ]
    for ds, info in m["source_dataset_metrics"].items():
        cd = info["class_distribution"]
        lines.append(
            f"| {ds} | {info['n']} | "
            f"{cd.get('Scratch', 0)} | {cd.get('Crack', 0)} | "
            f"{cd.get('Discoloration', 0)} | {cd.get('UNKNOWN', 0)} |"
        )

    lines += [
        "",
        "### Per-Source Performance",
        "| Dataset | N | Accuracy | Macro F1 |",
        "|---------|---|----------|---------|",
    ]
    for ds, info in m["source_dataset_metrics"].items():
        lines.append(
            f"| {ds} | {info['n']} | {info['accuracy']:.4f} | {info['macro_f1']:.4f} |"
        )

    lines += [
        "",
        "## 5. Confidence Analysis",
        "| Group | N | Mean | Median | Std | Min | Max | <0.5 (%) | <0.7 (%) | >0.9 (%) |",
        "|-------|---|------|--------|-----|-----|-----|----------|----------|----------|",
    ]
    for grp in ["correct", "incorrect"]:
        s = m["confidence_analysis"][grp]
        if s:
            lines.append(
                f"| {grp.capitalize()} | {s['count']} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"{s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
                f"{s['pct_below_0.5']:.1f} | {s['pct_below_0.7']:.1f} | {s['pct_above_0.9']:.1f} |"
            )

    lines += [
        "",
        "## 6. Interpretation",
        "",
        "> [!WARNING]",
        "> This model is NOT production-ready. The test result is a useful baseline but has significant limitations.",
        "",
        "### Source-Class Correlations (Structural Dataset Bias)",
        "The test set has strong source-to-class correlations:",
        "- **GC10-DET**: 177/180 Discoloration + 15 UNKNOWN (almost no other classes)",
        "- **NEU-CLS**: 45 Scratch + 15 UNKNOWN (no Crack, no Discoloration)",
        "- **MT**: 9 Crack + 10 UNKNOWN (almost no other classes)",
        "- **MVTecAD**: 18 Scratch + 13 Crack + 3 Discoloration + 40 UNKNOWN (most diverse)",
        "",
        "This means the classifier **may be partially recognizing the source domain** instead of",
        "the defect type. The current test design cannot distinguish genuine defect recognition",
        "from source/domain shortcuts.",
        "",
        "### Crack Performance (F1=0.73) — Root Cause",
        "Crack has only 22 test samples (9 MT, 13 MVTec). The model must generalize across",
        "two visually distinct domains with different imaging conditions. Small support means",
        "each error has disproportionate F1 impact.",
        "",
        "### Verdict",
        "The current baseline (91.88% accuracy, 0.8686 macro-F1) is a **useful starting point**",
        "but should not be treated as a domain-general defect classifier. It should be retained",
        "as the Phase 4 baseline and improved in a subsequent training experiment.",
    ]

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load and verify checkpoint ---
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    ckpt_epoch  = ckpt["epoch"]
    ckpt_val_f1 = ckpt["val_f1"]
    assert ckpt_epoch == EXPECTED_EPOCH, f"Expected epoch {EXPECTED_EPOCH}, got {ckpt_epoch}"
    assert abs(ckpt_val_f1 - EXPECTED_VAL_F1) < 0.001, f"Val F1 mismatch: {ckpt_val_f1}"
    print(f"  Epoch={ckpt_epoch}  val_f1={ckpt_val_f1:.6f}  OK")

    model = DefectClassifierModel(num_classes=4, use_pretrained=False, dropout_rate=0.2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- 2. Load test records ---
    test_records = load_jsonl(TEST_JSONL)
    assert len(test_records) == EXPECTED_N, f"Expected {EXPECTED_N} test records, got {len(test_records)}"
    print(f"Test records loaded: {len(test_records)}")

    # --- 3. Run inference ---
    print("Running inference on test set...")
    labels, preds, probs = run_inference(model, test_records, device)
    assert len(labels) == EXPECTED_N

    # --- 4. Compute metrics ---
    label_ints = list(range(4))
    acc      = accuracy_score(labels, preds)
    bal_acc  = balanced_accuracy_score(labels, preds)
    mac_prec = precision_score(labels, preds, average="macro",    labels=label_ints, zero_division=0)
    mac_rec  = recall_score   (labels, preds, average="macro",    labels=label_ints, zero_division=0)
    mac_f1   = f1_score       (labels, preds, average="macro",    labels=label_ints, zero_division=0)
    wgt_prec = precision_score(labels, preds, average="weighted", labels=label_ints, zero_division=0)
    wgt_rec  = recall_score   (labels, preds, average="weighted", labels=label_ints, zero_division=0)
    wgt_f1   = f1_score       (labels, preds, average="weighted", labels=label_ints, zero_division=0)
    pc_prec  = precision_score(labels, preds, average=None,       labels=label_ints, zero_division=0)
    pc_rec   = recall_score   (labels, preds, average=None,       labels=label_ints, zero_division=0)
    pc_f1    = f1_score       (labels, preds, average=None,       labels=label_ints, zero_division=0)
    support  = [(labels == i).sum() for i in range(4)]
    cm       = confusion_matrix(labels, preds, labels=label_ints)

    # Verify against previously reported
    acc_diff = abs(acc - EXPECTED_ACCURACY)
    f1_diff  = abs(mac_f1 - EXPECTED_MACRO_F1)
    assert acc_diff < 0.001, f"Accuracy mismatch vs previous: {acc:.6f} vs {EXPECTED_ACCURACY}"
    assert f1_diff  < 0.001, f"Macro-F1 mismatch vs previous: {mac_f1:.6f} vs {EXPECTED_MACRO_F1}"
    print(f"Metrics verified: accuracy={acc:.4f}  macro_f1={mac_f1:.4f}  (match OK)")

    metrics = {
        "accuracy":           round(float(acc), 6),
        "balanced_accuracy":  round(float(bal_acc), 6),
        "macro_precision":    round(float(mac_prec), 6),
        "macro_recall":       round(float(mac_rec), 6),
        "macro_f1":           round(float(mac_f1), 6),
        "weighted_precision": round(float(wgt_prec), 6),
        "weighted_recall":    round(float(wgt_rec), 6),
        "weighted_f1":        round(float(wgt_f1), 6),
        "per_class": {
            DEFAULT_DEFECT_CLASSES[i]: {
                "precision": round(float(pc_prec[i]), 6),
                "recall":    round(float(pc_rec[i]), 6),
                "f1":        round(float(pc_f1[i]), 6),
                "support":   int(support[i]),
            }
            for i in range(4)
        },
        "test_samples": EXPECTED_N,
    }

    cm_out = {
        "labels":       DEFAULT_DEFECT_CLASSES,
        "matrix":       cm.tolist(),
        "description":  "rows=true class, cols=predicted class",
    }

    # --- 5. Per-image prediction records ---
    pred_records = build_prediction_records(test_records, labels, preds, probs)

    # --- 6. Misclassification analysis ---
    misc_analysis = misclassification_analysis(pred_records)

    # --- 7. Source/domain metrics ---
    src_metrics = domain_metrics(pred_records, DEFAULT_DEFECT_CLASSES)

    # --- 8. Confidence analysis ---
    conf_analysis = confidence_analysis(pred_records)

    # --- Assemble full diagnostics ---
    diagnostics = {
        "phase": "Phase 4 Step 5",
        "checkpoint": {
            "path":    str(CKPT_PATH),
            "epoch":   ckpt_epoch,
            "val_f1":  round(float(ckpt_val_f1), 6),
        },
        "integrity_checks": {
            "test_records_evaluated":     EXPECTED_N,
            "checkpoint_epoch_correct":   ckpt_epoch == EXPECTED_EPOCH,
            "accuracy_matches_baseline":  acc_diff < 0.001,
            "macro_f1_matches_baseline":  f1_diff  < 0.001,
            "recomputed_accuracy":        round(float(acc), 6),
            "recomputed_macro_f1":        round(float(mac_f1), 6),
        },
        "metrics":                    metrics,
        "confusion_matrix":           cm_out,
        "source_dataset_metrics":     src_metrics,
        "misclassification":          misc_analysis,
        "confidence_analysis":        conf_analysis,
    }

    # --- Write artifacts ---
    # diagnostics_report.json
    with (OUT_DIR / "diagnostics_report.json").open("w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    # confusion_matrix.json
    with (OUT_DIR / "confusion_matrix.json").open("w", encoding="utf-8") as fh:
        json.dump(cm_out, fh, indent=2)

    # source_dataset_metrics.json
    with (OUT_DIR / "source_dataset_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(src_metrics, fh, indent=2)

    # misclassification_analysis.json
    with (OUT_DIR / "misclassification_analysis.json").open("w", encoding="utf-8") as fh:
        json.dump(misc_analysis, fh, indent=2)

    # test_predictions.jsonl
    with (OUT_DIR / "test_predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in pred_records:
            fh.write(json.dumps(r) + "\n")

    # test_predictions.csv
    csv_fields = [
        "sample_id", "source_dataset", "product", "original_label",
        "target_label", "predicted_label", "true_class_index",
        "predicted_class_index", "confidence",
        "probability_Scratch", "probability_Crack",
        "probability_Discoloration", "probability_UNKNOWN",
        "correct", "mapping_status", "source_split",
    ]
    with (OUT_DIR / "test_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(pred_records)

    # diagnostics_report.md
    md_text = build_markdown(diagnostics)
    (OUT_DIR / "diagnostics_report.md").write_text(md_text, encoding="utf-8")

    print(f"All artifacts written to: {OUT_DIR}")

    print("\n=== SUMMARY ===")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro Precision  : {mac_prec:.4f}")
    print(f"Macro Recall     : {mac_rec:.4f}")
    print(f"Macro F1         : {mac_f1:.4f}")
    print(f"Weighted F1      : {wgt_f1:.4f}")
    print("\nPer-class:")
    for i, cls in enumerate(DEFAULT_DEFECT_CLASSES):
        print(f"  {cls:<16s}: P={pc_prec[i]:.4f}  R={pc_rec[i]:.4f}  F1={pc_f1[i]:.4f}  n={support[i]}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {DEFAULT_DEFECT_CLASSES}")
    for i, row in enumerate(cm):
        print(f"  {DEFAULT_DEFECT_CLASSES[i]:<16s}: {row.tolist()}")
    print(f"\nMisclassified   : {misc_analysis['total_misclassified']} / {EXPECTED_N}")
    print("\nTop confusion pairs:")
    for pair, info in list(misc_analysis["confusion_pairs"].items())[:8]:
        print(f"  {pair}: {info['count']}  ({info['pct_of_true_class']:.1f}% of true class)")
    print("\nSource/domain performance:")
    for ds, info in src_metrics.items():
        print(f"  {ds}: n={info['n']}  acc={info['accuracy']:.4f}  macro_f1={info['macro_f1']:.4f}  dist={info['class_distribution']}")
    print("\nConfidence:")
    c = diagnostics["confidence_analysis"]
    print(f"  Correct   mean={c['correct']['mean']:.4f}  >0.9: {c['correct']['pct_above_0.9']:.1f}%")
    print(f"  Incorrect mean={c['incorrect']['mean']:.4f}  <0.7: {c['incorrect']['pct_below_0.7']:.1f}%")
    print("\nPhase 4 Step 5 diagnostics complete.")


if __name__ == "__main__":
    main()
