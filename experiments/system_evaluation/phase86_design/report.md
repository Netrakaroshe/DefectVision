# Phase 8.6: Pill PatchCore Representation Experiment — Experimental Design

**Design Status:** DESIGN ONLY (Frozen System / Offline Experiment)  
**Author:** AI Pair Programmer (DeepMind Antigravity)  
**Date:** October 7, 2026  
**Target Domain:** Pill (`pill2` product profile)  
**Hardware Environment:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), PyTorch 2.13.0+cu132, CUDA 13.2  
**Governing Context:** Phase 8.5 Root-Cause Analysis ([phase85/report.md](file:///d:/DefectVision/experiments/system_evaluation/phase85/report.md))  
**Output Target:** `experiments/system_evaluation/phase86_design/report.md`

---

## 1. Executive Summary & Experimental Motivation

In Phase 8.5, an offline root-cause analysis of the frozen DefectVision Pill Stage 1 anomaly detection pipeline identified that the **24 false negatives** (sensitivity = 75.0%, AUROC = 0.9451) are primarily driven by **feature representation insensitivity**:
1. **Severe Normal-Defect Score Overlap:** 19 out of 26 normal pill images (73.1%) produce anomaly scores exceeding the minimum false-negative score ($37.49$). Recalibrating the decision threshold $\tau$ alone cannot separate the populations without inducing unacceptable false positives (up to 76.9% FPR).
2. **Semantic Feature Invariance:** The baseline WideResNet-50-2 backbone layers (`layer2`, `layer3`) have large receptive fields (strides 8 and 16) trained for ImageNet classification. They treat subtle debossed lettering alterations (`faulty_imprint`: 9/19 missed) as semantically valid tablet text.
3. **Contrast & Hairline Attenuation:** In low-contrast discolorations (mean Weber contrast 0.127 vs 1.179 for true positives) and hairline micro-cracks ($< 0.33\%$ image area), nearest-neighbor distances in memory bank space plateau between 37.5 and 43.4, falling below the calibrated threshold ($\tau = 43.5951$).

### Primary Research Question:
> *"Can an alternative feature representation improve PatchCore's sensitivity to subtle, low-contrast, and debossed Pill defects without elevating the normal-surface baseline or causing unacceptable false-positive fallout on normal product?"*

This document defines the strict, controlled experimental design for comparing the frozen production baseline against candidate feature representations offline, with zero modifications to production code, databases, models, or configurations.

---

## 2. Hypotheses & Theoretical Rationale

### Primary Hypothesis ($H_1$):
Incorporating high-frequency, fine-grained spatial representations—either through **multi-scale CNN feature pooling** (adding shallow layers) or **dense self-supervised transformer representations** (DINOv2 patch tokens)—will increase the nearest-neighbor Euclidean distance for subtle local alterations (hairline cracks, missing imprint strokes, faint color spots) relative to normal surface patches, thereby widening the separation margin between defective and normal pills.

### Null Hypothesis ($H_0$):
Candidate representations will scale normal surface texture noise and debossing illumination variations at the same rate as defect signals, resulting in no net gain in AUROC or sensitivity at an acceptable false-positive rate ($\text{FPR} \le 5.0\%$).

### Specific Variant Hypotheses:
- **Variant B (Multi-Scale CNN):** Adding `layer1` (stride 4, 256 channels) preserves edge and stroke fidelity before downsampling, specifically boosting sensitivity on `faulty_imprint` and micro-`crack`.
- **Variant C (Self-Supervised ViT / DINOv2):** DINOv2's explicit self-supervised training on patch-level correspondence produces feature representations that do not discard style, stroke, or color variations, yielding higher patch anomaly scores on faint stains and subtle stamp distortions without class-semantic invariance.
- **Variant D (Tablet Pose / Alignment Normalization):** Rotational variance in normal debossing is a primary contributor to normal score spread ($\sigma = 2.27$). Canonical in-plane alignment of the pill tablet before feature extraction will reduce normal memory bank dispersion, lowering the calibration baseline $\tau = \mu + 3\sigma$.

---

## 3. Candidate Representations & Variant Definitions

The experiment evaluates four strictly controlled variants. To ensure isolation of variables, all variants share the identical coreset subsampling ratio ($10\%$), neighborhood parameter ($k=9$), input resolution ($256 \times 256$ or native $224 \times 224$), and reference calibration pool ($N=10$).

| Variant ID | Architecture / Backbone | Extracted Feature Layers | Receptive Field / Spatial Stride | Nominal Feature Dimension ($D$) | Theoretical Mechanism |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Variant A** *(Frozen Baseline)* | `wide_resnet50_2` (ImageNet-1k) | `('layer2', 'layer3')` | Stride 8 & 16 | 1,536 ($512 + 1024$) | Deep semantic features; robust to macro defects, invariant to fine font/texture. |
| **Variant B** *(Multi-Scale CNN)* | `wide_resnet50_2` (ImageNet-1k) | `('layer1', 'layer2', 'layer3')` | Stride 4, 8 & 16 | 1,792 ($256 + 512 + 1024$) | High-frequency edge retention via Layer 1; preserves debossed stroke boundaries. |
| **Variant C** *(Dense DINOv2 ViT)* | `vit_small_patch14_dinov2` (DINOv2) | `('blocks.8', 'blocks.11')` | Patch Size 14 | 768 ($384 + 384$) | Self-supervised dense local patch representations without semantic class invariance. |
| **Variant D** *(Isolated Alignment)* | `wide_resnet50_2` (ImageNet-1k) | `('layer2', 'layer3')` | Stride 8 & 16 (Aligned) | 1,536 ($512 + 1024$) | Pill major-axis / debossing canonical orientation alignment prior to baseline PatchCore. |

### Technical Feasibility & Compatibility Check:
- **Variant A & B:** Natively supported by Anomalib 2.6.2 `Patchcore(backbone="wide_resnet50_2", layers=...)`. Verified via runtime test: `Patchcore(layers=('layer1', 'layer2', 'layer3'))` instantiates and builds feature pyramids cleanly.
- **Variant C:** Verified in project environment. `vit_small_patch14_dinov2` is available via `timm 1.0.15`. Anomalib's `TimmFeatureExtractor` accepts ViT transformer blocks via `layers=('blocks.8', 'blocks.11')`. Verified runtime instantiation on Python 3.13 / CUDA 13.2.
- **Variant D:** Isolated deterministic preprocessing step: moments-based ellipse fitting on binary pill mask to rotate major axis horizontally ($0^\circ / 180^\circ$ canonical alignment).

---

## 4. Dataset Partitions, Reference Sets & Leakage Prevention

### 4.1 Calibration Reference Pool ($N=10$)
All candidate variants will construct their feature memory bank from the **exact same 10 defect-free normal reference images** used to calibrate the frozen production model `pill2_patchcore.ckpt`:
- Directory: `data/app_storage/products/pill2/reference_images/` (or MVTec AD `pill/train/good` source partition).
- Sample IDs: Authoritative 10 calibration images established in Phase 8.3 / Phase 8.4.
- Zero contamination with test sets ($0 / 10$ overlap with evaluation data).

### 4.2 External Evaluation Dataset ($N=122$)
The evaluation set is identical to Phase 8.4 and Phase 8.5:

| Partition | Category | Sample Count ($N$) | Evaluation Role |
| :--- | :--- | :---: | :--- |
| **Normal Test** | `good` | **26** | Specificity, False Positive Rate (FPR), Normal Score Distribution |
| **Defective Test** | `faulty_imprint` | 19 | Primary failure category (9 FNs in Phase 8.5) |
| | `contamination` | 21 | Secondary failure category (6 FNs in Phase 8.5) |
| | `color` | 10 | Low-contrast discoloration (3 FNs in Phase 8.5) |
| | `crack` | 10 | Hairline fissure category (3 FNs in Phase 8.5) |
| | `combined` | 17 | Multi-defect tablet category (2 FNs in Phase 8.5) |
| | `scratch` | 10 | High-contrast abrasion category (1 FN in Phase 8.5) |
| | `pill_type` | 9 | Macro morphological category (0 FNs in Phase 8.5) |
| **Total Evaluation** | | **122** | Complete, auditable, consistent benchmark |

### 4.3 Strict Leakage & Partition Controls:
1. **Exclusion of Development Pool:** The 45-image Phase 8.3D V2 development pool (`experiments/pill_roi_classifier_v2/data/`) remains **100% excluded** from this evaluation.
2. **Authoritative Raw Data Freeze:** `D:/datasets/defect_dataset_workspace/sources/MVTecAD/pill` remains read-only.
3. **No Test-Set Tuning:** Memory banks, coreset selection, and threshold calibration must use **only** the 10 normal calibration images. The 122 test images are evaluated in a single forward pass without threshold adjustments based on test performance.

---

## 5. Memory Bank Construction & Calibration Protocol

### 5.1 Patch Feature Extraction & Coreset Sampling
For each candidate:
1. Feed the 10 reference images through the candidate backbone.
2. Extract patch features at specified layer endpoints.
3. Apply spatial neighborhood average pooling ($3 \times 3$, padding 1) to aggregate context.
4. Downsample all layer patch maps to a uniform spatial grid and concatenate channel-wise.
5. Apply greedy minimax coreset subsampling at fixed ratio $\alpha = 0.10$ ($10\%$ retention).
6. Store memory bank matrix $\mathcal{M} \in \mathbb{R}^{M \times D}$.

### 5.2 Threshold Calibration Protocol
Decision thresholds must be calibrated objectively and reproducibly using the identical formula applied in production:
$$\tau_{\text{variant}} = \mu_{\text{ref}} + 3.0 \cdot \sigma_{\text{ref}}$$
Where $\mu_{\text{ref}}$ and $\sigma_{\text{ref}}$ are the mean and standard deviation of maximum nearest-neighbor anomaly scores evaluated on the **10 normal calibration reference images** in cross-validation / leave-one-out mode.

*Additional Analysis:* In addition to the operational $\mu + 3\sigma$ threshold, full precision-recall and ROC curves across all possible thresholds will be generated to assess fundamental representation separability independent of the calibration multiplier.

---

## 6. Evaluation Metrics & Audit Protocol

Every candidate will be evaluated across eight quantitative dimensions:

### 6.1 Classification & Detection Performance
1. **Image AUROC:** Area under the ROC curve on all 122 evaluation samples (unbiased ranking quality).
2. **Sensitivity (TPR):** Total defects detected divided by 96 defective test images at $\tau_{\text{variant}}$.
3. **Specificity (TNR):** Normal images accepted divided by 26 normal test images at $\tau_{\text{variant}}$.
4. **False Positive Rate (FPR):** Normal images falsely rejected divided by 26 ($\le 5.0\%$ target).
5. **False Negative Count ($N_{\text{FN}}$):** Target reduction from baseline $N_{\text{FN}} = 24$.
6. **False Negative Recovery Rate:** Number of the 24 specific Phase 8.5 false negatives successfully converted to true positives.

### 6.2 Category-Level Sensitivity Breakdown
Report individual sensitivities ($N_{\text{detected}} / N_{\text{total}}$) for:
- `faulty_imprint` (19 samples)
- `contamination` (21 samples)
- `color` (10 samples)
- `crack` (10 samples)
- `scratch` (10 samples)
- `combined` (17 samples)
- `pill_type` (9 samples)

### 6.3 Anomaly Score Distribution & Separation Margin
For both Normal ($N=26$) and Defective ($N=96$) sets:
- Minimum, P10, P25, Median (P50), P75, P90, Maximum, Mean, Std Dev.
- **Score Separation Margin:** $\Delta_{\text{sep}} = \min(\text{Defect Score}) - \max(\text{Normal Score})$.
  *(Baseline was $-8.08$; a less negative or positive margin indicates improved class separability).*

### 6.4 Quantitative Defect Localization (Ground-Truth Masks)
Using authoritative MVTec binary defect masks ($N=96$):
1. **Defect Pixel Coverage:** Proportion of ground-truth defect pixels contained within the PatchCore anomaly bounding box ($\ge 80\%$ standard).
2. **Bounding Box IoU ($\text{IoU}_{\text{box}}$):** Spatial overlap between PatchCore ROI and ground-truth minimal bounding box.
3. **Peak-on-Defect Rate:** Proportion of defective images where the global maximum anomaly score coordinate $(x, y)$ resides inside the defect mask (baseline was 16/24 on FNs; goal is $> 90\%$ overall).

### 6.5 Inference Latency Profile
Measured on NVIDIA RTX 4060 Laptop GPU across 50 warm steady-state cycles:
- Feature extraction latency (ms)
- Nearest-neighbor coreset search latency (ms)
- Total Stage 1 latency: Mean, P50, P90, P95, Max (ms)

---

## 7. Hardware Feasibility & Computational Cost (RTX 4060 8GB)

A critical requirement is that all candidates execute within the physical memory and compute envelope of the local NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM).

| Metric | Variant A (Baseline) | Variant B (Multi-Scale) | Variant C (DINOv2 ViT-S) | Variant D (Alignment + WRN) | Feasibility Margin (8GB VRAM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 68.9 M | 68.9 M | 22.1 M | 68.9 M | Ample capacity (< 275 MB model weights) |
| **Feature Dimension ($D$)** | 1,536 | 1,792 | 768 | 1,536 | Well within CUDA tensor limits |
| **Coreset Memory Size** | ~784 vectors | ~1,200 vectors | ~256 vectors | ~784 vectors | Minimal host RAM (< 10 MB per bank) |
| **Peak VRAM (Fitting)** | ~2.1 GB | ~2.8 GB | ~1.6 GB | ~2.1 GB | **> 5.2 GB Headroom** (65% margin) |
| **Peak VRAM (Inference)** | ~1.8 GB | ~2.2 GB | ~1.4 GB | ~1.8 GB | **> 5.8 GB Headroom** (72% margin) |
| **Fitting Time (10 refs)** | ~18 s | ~25 s | ~12 s | ~22 s | Fast offline fitting (< 30 seconds) |
| **Inference Latency** | ~116 ms | ~135 ms | ~85 ms | ~125 ms | All candidates $\le 140$ ms / frame |

*Conclusion:* All four candidate variants are completely feasible on the RTX 4060 Laptop GPU with zero risk of CUDA Out-Of-Memory (OOM) exceptions.

---

## 8. Comparison Methodology & Decision Matrix

To prevent confirmation bias or premature conclusions, the evaluation will **not** enforce arbitrary pass/fail thresholds. Instead, candidates will be evaluated across a multi-dimensional Pareto frontier comparing defect sensitivity against false-positive rate:

```
                                  EVALUATION PARETO FRONTIER
                  High Sensitivity |
                                   |      [Ideal Target]
                                   |      Sensitivity >= 90%
                                   |      FPR <= 3.8%
                 Baseline Phase 8.5|
                 Sensitivity = 75% |
                 FPR = 3.8%        |
                                   |
                                   +---------------------------------
                                   0%        3.8%        10%       20%
                                            False Positive Rate (FPR)
```

### Important Clarification on Operating Points & Resource Values:
> **NOTE ON EVALUATION CRITERIA & RESOURCES:**  
> The 96% specificity / 4% FPR value is **NOT** a hard acceptance criterion and must **not** be presented as a project pass/fail gate. It is only a reference operating point for comparing candidate trade-offs along the Pareto frontier.  
> Likewise, the stated VRAM and latency values in Section 7 are feasibility estimates only; actual peak VRAM consumption and inference latencies must be measured empirically during execution.

### Empirical Comparison Criteria:

1. **Defect Sensitivity vs. Normal Specificity Trade-Off:**
   - Evaluated as a multi-dimensional trade-off along the ROC / PR curve without imposing single-point pass/fail gates.
   - Candidate is **Marginal** if: Sensitivity increases to $> 85.0\%$ but normal specificity drops below $92.0\%$ ($\text{FPR} > 8.0\%$).
   - Candidate is **Inferior** if: AUROC decreases or sensitivity fails to exceed the baseline ($75.0\%$).
2. **False-Negative Recovery Profile:**
   - How many of the 24 specific Phase 8.5 false negatives are converted to True Positives?
   - Does the candidate specifically fix `faulty_imprint` (baseline missed 9/19) without creating new misses on `scratch` or `pill_type`?
3. **Score Separation Gap:**
   - Does the candidate reduce the overlap interval between normal and defective score distributions?

---

## 9. Expected Artifacts & Directory Structure

All execution artifacts will reside in a dedicated, isolated experiment directory:

```
experiments/system_evaluation/phase86_experiment/
├── configs/
│   ├── variant_a_baseline.json
│   ├── variant_b_multiscale.json
│   ├── variant_c_dinov2.json
│   └── variant_d_aligned.json
├── checkpoints/
│   ├── variant_b_patchcore.ckpt
│   ├── variant_c_dinov2_patchcore.ckpt
│   └── variant_d_aligned_patchcore.ckpt
├── results/
│   ├── comparative_metrics.json          # Consolidated metrics across all variants
│   ├── per_sample_all_variants.json      # Per-sample scores and predictions (122 samples x 4 variants)
│   ├── category_sensitivity.json         # Per-category sensitivity breakdowns
│   ├── fn_recovery_matrix.json           # 24-FN transition status across all variants
│   ├── localization_metrics.json         # Coverage, IoU, purity vs ground truth
│   ├── latency_benchmarks.json           # Timed CUDA event execution times
│   └── report.md                         # Comprehensive final evaluation report
```

---

## 10. Risk Analysis & Failure Mitigation

| Risk ID | Identified Risk | Potential Impact | Mitigation Strategy in Design |
| :---: | :--- | :--- | :--- |
| **R1** | **Multi-scale feature noise amplification:** Adding Layer 1 introduces high-frequency texture noise from normal tablet coating, raising normal scores. | Specificity degrades ($\text{FPR} > 15\%$). | Monitor normal score standard deviation ($\sigma_{\text{ref}}$) independently; evaluate Layer 1 with adaptive spatial pooling. |
| **R2** | **DINOv2 patch boundary artifacts:** Patch size 14 creates a coarse $16 \times 16$ grid for $224 \times 224$ images, potentially blurring tiny micro-cracks. | Fine crack sensitivity remains unimproved. | Evaluate ViT with interpolated patch positions; verify localization on hairline cracks against ground-truth masks. |
| **R3** | **Accidental production overwrite:** Modifying production files or replacing `pill2_patchcore.ckpt`. | Corrupts frozen production state. | Script paths strictly constrained to `experiments/system_evaluation/phase86_experiment/`; read-only access to `data/app_storage/`. |
| **R4** | **Unauthenticated HuggingFace Hub download lock:** DINOv2 weights download may stall if internet access is rate-limited. | Script stalls during execution. | Verified that `timm` can load local cached weights or PyTorch hub weights; fallbacks defined in config. |

---

## 11. Step-by-Step Execution Plan

When Phase 8.6 execution is approved in a subsequent user instruction, execution will follow these exact sequential steps:

1. **Step 1: Staging & Environment Verification**
   - Create isolated directory: `experiments/system_evaluation/phase86_experiment/`.
   - Verify CUDA device availability and GPU memory headroom on RTX 4060.
   - Re-verify frozen production files remain untouched.
2. **Step 2: Baseline Extraction (Variant A)**
   - Re-run frozen `pill2_patchcore.ckpt` on the 122 evaluation samples to reconfirm identical baseline metrics (AUROC = 0.9451, Sens = 75.0%, 24 FNs).
3. **Step 3: Multi-Scale PatchCore Training & Inference (Variant B)**
   - Build feature memory bank on the 10 normal calibration images using WideResNet-50-2 with `layers=('layer1', 'layer2', 'layer3')`.
   - Compute calibrated threshold $\tau_B = \mu_B + 3\sigma_B$.
   - Execute inference on all 122 test samples; record anomaly scores and heatmaps.
4. **Step 4: DINOv2 Dense PatchCore Training & Inference (Variant C)**
   - Build feature memory bank on the 10 normal calibration images using `vit_small_patch14_dinov2` with `layers=('blocks.8', 'blocks.11')`.
   - Compute calibrated threshold $\tau_C = \mu_C + 3\sigma_C$.
   - Execute inference on all 122 test samples; record anomaly scores and heatmaps.
5. **Step 5: Alignment-Normalized PatchCore (Variant D)**
   - Apply ellipse-fitting orientation normalization to the 10 reference images and 122 test images.
   - Build feature memory bank and evaluate baseline PatchCore on aligned inputs.
6. **Step 6: Comparative Analysis & Metric Synthesis**
   - Compute AUROC, sensitivity, specificity, FPR, and score separation margins for all variants.
   - Cross-reference all 24 Phase 8.5 false negatives to determine recovery status.
   - Compute quantitative ROI coverage and $\text{IoU}_{\text{box}}$ against ground-truth masks.
   - Benchmark inference latency across 50 timed cycles per variant.
7. **Step 7: Reporting & Documentation**
   - Write machine-readable JSON artifacts.
   - Compile final comparative report to `experiments/system_evaluation/phase86_experiment/report.md`.
   - Verify zero modifications to production code, checkpoints, database, or frontend.

---

## 12. Recommended Execution Order

When approved for execution, candidate variants should be executed in the following order:

1. **Variant A (Frozen Baseline Verification):** Guarantees zero regression and establishes the identical benchmark reference.
2. **Variant B (Multi-Scale CNN — `layer1` + `layer2` + `layer3`):** Lowest implementation risk; stays within the WideResNet architecture while directly targeting high-frequency edge retention.
3. **Variant C (Dense Self-Supervised ViT — DINOv2):** Highest potential for breaking ImageNet class semantic invariance on debossed text and low-contrast smudges.
4. **Variant D (Pill Alignment Preprocessing):** Evaluated last as an orthogonal factor to determine if mechanical orientation control provides additive benefit.

---

## 13. Design Freeze Statement

- **Design Status:** COMPLETE and FROZEN.
- **Production Files:** 100% frozen; zero files in `src/`, `data/app_storage/`, database, or frontend modified.
- **Execution Status:** **STOPPED.** No models retrained, no experiments executed. Awaiting explicit user approval before execution.
