# Phase 4 Step 4 -- Defect Classifier Training Report

## Configuration
- Architecture: ResNet-18 (ImageNet-1K pretrained), Dropout=0.2
- Classes: Scratch, Crack, Discoloration, UNKNOWN
- Seed: 42  |  Batch size: 32
- Optimizer: AdamW (lr=0.0001, weight_decay=0.01)
- Scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-06)
- Max epochs: 30  |  Early-stop patience: 7

## Class Weights (train split)
| Class | Weight |
|-------|--------|
| Scratch | 1.2841 |
| Crack | 3.8789 |
| Discoloration | 0.4458 |
| UNKNOWN | 1.3884 |

## Training History
| Epoch | Train Loss | Train F1 | Val Loss | Val F1 | LR |
|-------|-----------|----------|----------|--------|-----|
| 01 | 0.7526 | 0.6487 | 0.4758 | 0.7519 | 9.97e-05 |
| 02 | 0.4568 | 0.8041 | 0.3234 | 0.8067 | 9.89e-05 |
| 03 | 0.3635 | 0.8268 | 0.3224 | 0.8290 | 9.76e-05 |
| 04 | 0.3343 | 0.8395 | 0.2326 | 0.8512 | 9.57e-05 |
| 05 | 0.3118 | 0.8539 | 0.2445 | 0.8554 | 9.34e-05 |
| 06 | 0.2674 | 0.8671 | 0.3747 | 0.7893 | 9.06e-05 |
| 07 | 0.2372 | 0.8774 | 0.2489 | 0.8305 | 8.73e-05 |
| 08 | 0.2263 | 0.8848 | 0.2608 | 0.8466 | 8.36e-05 |
| 09 | 0.2216 | 0.8912 | 0.2693 | 0.8273 | 7.96e-05 |
| 10 | 0.2178 | 0.8897 | 0.2604 | 0.8638 | 7.52e-05 |
| 11 | 0.2095 | 0.9053 | 0.2673 | 0.8602 | 7.06e-05 |
| 12 | 0.2107 | 0.8912 | 0.2744 | 0.8304 | 6.58e-05 |
| 13 | 0.1738 | 0.9022 | 0.2115 | 0.8301 | 6.08e-05 |
| 14 | 0.1764 | 0.9134 | 0.2445 | 0.8176 | 5.57e-05 |
| 15 | 0.1748 | 0.9069 | 0.2388 | 0.8858 | 5.05e-05 |
| 16 | 0.1356 | 0.9376 | 0.2460 | 0.8538 | 4.53e-05 |
| 17 | 0.1348 | 0.9273 | 0.2796 | 0.8549 | 4.02e-05 |
| 18 | 0.1342 | 0.9311 | 0.2651 | 0.8486 | 3.52e-05 |
| 19 | 0.1257 | 0.9446 | 0.2377 | 0.8462 | 3.04e-05 |
| 20 | 0.1085 | 0.9374 | 0.2775 | 0.8558 | 2.57e-05 |
| 21 | 0.1144 | 0.9387 | 0.2284 | 0.8458 | 2.14e-05 |
| 22 | 0.0882 | 0.9547 | 0.2716 | 0.8467 | 1.74e-05 |

## Test Set Evaluation (one-time hold-out)
- **Best epoch**: 15
- **Test samples**: 345
- **Test loss**: 0.2734
- **Accuracy**: 0.9188
- **Macro F1**: 0.8686

### Per-Class F1
| Class | F1 |
|-------|----|
| Scratch | 0.9062 |
| Crack | 0.7308 |
| Discoloration | 0.9748 |
| UNKNOWN | 0.8627 |

### Confusion Matrix (rows=true, cols=predicted)
| True \ Pred | Scratch | Crack | Discoloration | UNKNOWN |
|---|---|---|---|---|
| **Scratch** | 58 | 5 | 0 | 0 |
| **Crack** | 2 | 19 | 0 | 1 |
| **Discoloration** | 0 | 0 | 174 | 6 |
| **UNKNOWN** | 5 | 6 | 3 | 66 |

### Full Classification Report
```
precision    recall  f1-score   support

      Scratch     0.8923    0.9206    0.9062        63
        Crack     0.6333    0.8636    0.7308        22
Discoloration     0.9831    0.9667    0.9748       180
      UNKNOWN     0.9041    0.8250    0.8627        80

     accuracy                         0.9188       345
    macro avg     0.8532    0.8940    0.8686       345
 weighted avg     0.9259    0.9188    0.9207       345
```