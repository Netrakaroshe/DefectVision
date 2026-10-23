# Phase 8.3E — Controlled Open-Set Rejection Benchmark

## Overview
This diagnostic experiment benchmarks three post-hoc open-set rejection methods on top of the Phase 8.3D ROI-trained ResNet-18 pill classifier:
1. **Maximum Softmax Probability (MSP)**: Confidence thresholding on standard softmax.
2. **Temperature-Scaled Softmax**: Calibrated confidence scaling with $T = 1.1414$ fitted on the validation set.
3. **Feature-Distance Rejection**: Cosine affinity to training class centroids extracted from the 512-dimensional penultimate layer.

## Datasets
- **Known-Class Benchmark ($N=23$)**: Untouched external evaluation set from Phase 8.3C/8.3D (9 Scratch, 7 Crack, 7 Discoloration).
- **Unsupported Open-Set Benchmark ($N=66$)**: Real MVTecAD pill defects outside the supervised taxonomy (21 contamination, 19 faulty_imprint, 17 combined, 9 pill_type).

## Artifacts in this directory
- `summary_metrics.json`: AUROC, AUPRC, fitted temperature, and operating point statistics.
- `threshold_analysis.json`: Fine-grained threshold sweeps across all three methods.
- `per_sample_results.json`: Full per-image predictions, logits, probabilities, and distances.
- `confusion_matrices.json`: 4-class confusion matrices (Scratch, Crack, Discoloration, UNKNOWN) at calibrated operating points.
- `open_set_roc_pr_curves.png`: Visualization of ROC and Precision-Recall curves.
- `experiment_config.json`: Full experimental metadata.
- `report.md`: Formal diagnostic report.
