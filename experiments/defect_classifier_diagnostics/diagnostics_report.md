# Phase 4 Step 5 -- Defect Classifier Diagnostics Report

## 1. Checkpoint Verification
- Checkpoint: D:\DefectVision\experiments\defect_classifier_run1\best_model.pt
- Saved epoch: 15
- Saved val macro-F1: 0.885796
- Expected epoch: 15  |  Match: True

## 2. Test-Set Metrics (345 images)

### Overall
| Metric | Value |
|--------|-------|
| Accuracy | 0.9188 |
| Balanced Accuracy | 0.8940 |
| Macro Precision | 0.8532 |
| Macro Recall | 0.8940 |
| Macro F1 | 0.8686 |
| Weighted Precision | 0.9259 |
| Weighted Recall | 0.9188 |
| Weighted F1 | 0.9207 |

### Per-Class
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Scratch | 0.8923 | 0.9206 | 0.9062 | 63 |
| Crack | 0.6333 | 0.8636 | 0.7308 | 22 |
| Discoloration | 0.9831 | 0.9667 | 0.9748 | 180 |
| UNKNOWN | 0.9041 | 0.8250 | 0.8627 | 80 |

### Confusion Matrix (rows=true, cols=predicted)
| True \ Pred | Scratch | Crack | Discoloration | UNKNOWN |
|---|---|---|---|---|
| **Scratch** | 58 | 5 | 0 | 0 |
| **Crack** | 2 | 19 | 0 | 1 |
| **Discoloration** | 0 | 0 | 174 | 6 |
| **UNKNOWN** | 5 | 6 | 3 | 66 |

## 3. Misclassification Analysis
- **Total misclassified**: 28 / 345 (8.12%)

### Confusion Pairs
| Pair | Count | % of True Class | Mean Confidence | Top Source | Top Original Label |
|------|-------|----------------|-----------------|-----------|-------------------|
| Discoloration -> UNKNOWN | 6 | 3.3% | 0.663 | GC10-DET | water_spot |
| UNKNOWN -> Crack | 6 | 7.5% | 0.816 | MVTecAD | faulty_imprint |
| UNKNOWN -> Scratch | 5 | 6.2% | 0.723 | MVTecAD | crescent_gap |
| Scratch -> Crack | 5 | 7.9% | 0.623 | MVTecAD | scratch |
| UNKNOWN -> Discoloration | 3 | 3.8% | 0.707 | GC10-DET | inclusion |
| Crack -> Scratch | 2 | 9.1% | 0.548 | MVTecAD | crack |
| Crack -> UNKNOWN | 1 | 4.5% | 0.604 | MT | MT_Crack |

## 4. Source / Domain Analysis

### Test-Set Distribution
| Dataset | N | Scratch | Crack | Discoloration | UNKNOWN |
|---------|---|---------|-------|---------------|---------|
| GC10-DET | 192 | 0 | 0 | 177 | 15 |
| MT | 19 | 0 | 9 | 0 | 10 |
| MVTecAD | 74 | 18 | 13 | 3 | 40 |
| NEU-CLS | 60 | 45 | 0 | 0 | 15 |

### Per-Source Performance
| Dataset | N | Accuracy | Macro F1 |
|---------|---|----------|---------|
| GC10-DET | 192 | 0.9531 | 0.4217 |
| MT | 19 | 0.9474 | 0.4734 |
| MVTecAD | 74 | 0.7568 | 0.7441 |
| NEU-CLS | 60 | 1.0000 | 0.5000 |

## 5. Confidence Analysis
| Group | N | Mean | Median | Std | Min | Max | <0.5 (%) | <0.7 (%) | >0.9 (%) |
|-------|---|------|--------|-----|-----|-----|----------|----------|----------|
| Correct | 317 | 0.9448 | 0.9924 | 0.1130 | 0.3885 | 1.0000 | 1.6 | 6.6 | 84.2 |
| Incorrect | 28 | 0.6937 | 0.6421 | 0.1653 | 0.4704 | 0.9975 | 14.3 | 57.1 | 17.9 |

## 6. Interpretation

> [!WARNING]
> This model is NOT production-ready. The test result is a useful baseline but has significant limitations.

### Source-Class Correlations (Structural Dataset Bias)
The test set has strong source-to-class correlations:
- **GC10-DET**: 177/180 Discoloration + 15 UNKNOWN (almost no other classes)
- **NEU-CLS**: 45 Scratch + 15 UNKNOWN (no Crack, no Discoloration)
- **MT**: 9 Crack + 10 UNKNOWN (almost no other classes)
- **MVTecAD**: 18 Scratch + 13 Crack + 3 Discoloration + 40 UNKNOWN (most diverse)

This means the classifier **may be partially recognizing the source domain** instead of
the defect type. The current test design cannot distinguish genuine defect recognition
from source/domain shortcuts.

### Crack Performance (F1=0.73) — Root Cause
Crack has only 22 test samples (9 MT, 13 MVTec). The model must generalize across
two visually distinct domains with different imaging conditions. Small support means
each error has disproportionate F1 impact.

### Verdict
The current baseline (91.88% accuracy, 0.8686 macro-F1) is a **useful starting point**
but should not be treated as a domain-general defect classifier. It should be retained
as the Phase 4 baseline and improved in a subsequent training experiment.