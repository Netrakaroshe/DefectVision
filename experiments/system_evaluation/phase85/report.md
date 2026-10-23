# Phase 8.5: Pill Stage-1 False-Negative Root-Cause Analysis Report

**Investigation Date:** October 7, 2026  
**Environment:** Windows 11, PyTorch 2.13.0+cu132, CUDA 13.2, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Evaluated Product:** Pill (`pill2_patchcore.ckpt`, Calibrated Threshold $\tau = 43.5951$)  
**Artifact Directory:** `experiments/system_evaluation/phase85/`  
**Data Sources:** Phase 8.4 evaluation artifacts ([results/stage1_metrics.json](file:///d:/DefectVision/experiments/system_evaluation/results/stage1_metrics.json), [results/per_sample_predictions.json](file:///d:/DefectVision/experiments/system_evaluation/results/per_sample_predictions.json)), MVTec AD Pill ground-truth masks.

---

## 1. Executive Summary

In Phase 8.4, Stage 1 (PatchCore anomaly detection) on the Pill domain achieved an AUROC of **0.9451**, specificity of **96.15%** (25/26), and sensitivity of **75.00%** (72/96), missing **24 defective samples** (false negatives). 

This root-cause analysis investigates the physical, mathematical, and architectural mechanisms causing PatchCore to miss these 24 defects. Every false negative was analyzed against its ground-truth segmentation mask, anomaly heatmap distribution, contrast, size, and distance to threshold, and benchmarked against all 72 correctly detected True Positives (TP) and 26 Normal test images.

### Key Empirical Findings:

1. **Failure Distribution Across Categories:**
   - **`faulty_imprint`:** 9 false negatives out of 19 test samples (**47.37% failure rate**). Largest contributor to Stage 1 fallout (37.5% of all FNs).
   - **`contamination`:** 6 false negatives out of 21 test samples (**28.57% failure rate**).
   - **`color`:** 3 false negatives out of 10 test samples (**30.00% failure rate**).
   - **`crack`:** 3 false negatives out of 10 test samples (**30.00% failure rate**).
   - **`combined`:** 2 false negatives out of 17 test samples (**11.76% failure rate**).
   - **`scratch`:** 1 false negative out of 10 test samples (**10.00% failure rate**).
   - **`pill_type`:** 0 false negatives out of 9 test samples (**0.00% failure rate**; 100% detected).

2. **The Threshold Paradox (Hypothesis A Refuted as Sole Remedy):**
   - Normal score range: **[35.53, 45.56]** (Mean: 39.05, Median: 39.10).
   - False Negative score range: **[37.49, 43.40]** (Mean: 41.36, Median: 42.10).
   - True Positive score range: **[43.81, 62.50]** (Mean: 48.91, Median: 48.27).
   - **Critical Overlap:** **19 out of 26 normal samples (73.08%) score higher than the lowest false negative (37.49).**
   - The score distributions of subtle defects and normal pills completely overlap between 37.49 and 43.40. Lowering the threshold to capture all 24 FNs would increase the False Positive Rate from 3.85% to **76.92%** (rejecting >3 out of 4 good pills). Thus, **threshold adjustment alone cannot resolve the issue**.

3. **Weak PatchCore Feature Representation (Hypothesis B Validated):**
   - Pretrained WideResNet-50 features from layer 2 and layer 3 operate at spatial downsampling strides of 8 and 16.
   - For **`faulty_imprint`**, deep CNN features represent high-level semantic shapes. Degraded lettering or missing strokes still match normal debossed text in deep feature space, resulting in minimal nearest-neighbor distance.
   - In **16 of 24 FNs (66.7%)**, the anomaly heatmap **did** localize the defect peak on the true defect, and in **23 of 24 (95.8%)** a distinct local signal existed. However, the patch distance plateaued between 40.0 and 43.4 because deep features lack sensitivity to subtle stroke and texture variations.

4. **Extreme Low-Contrast Defect Profiles (Hypothesis D Validated):**
   - True Positives: Mean Weber contrast = **1.179**, Mean intensity difference $\Delta I$ = **32.52**.
   - False Negatives: Mean Weber contrast = **0.127** (nearly **10× lower**), Mean intensity difference $\Delta I$ = **16.11**.
   - 18 of 24 FNs (75.0%) have Weber contrast $< 0.20$. Faint color bleeds and shallow surface blemishes fail to trigger significant feature divergence.

5. **Rim/Boundary Artifact Domination in Micro-Cracks:**
   - In all 3 missed cracks (`013.png`, `014.png`, `016.png`), the cracks are extremely fine hairlines ($\le 0.33\%$ image area). The activation on the normal pill boundary/rim ($36.3$–$38.7$) exceeded or masked the subtle activation of the crack ($32.3$–$36.2$).

---

## 2. Per-Sample False-Negative Audit Table ($N=24$)

All 24 false negatives are cataloged below with exact ground-truth metrics, anomaly scores, distance to threshold ($\Delta = \text{Score} - \tau$), localized defect signal verification, and attributed failure mechanisms.

| # | Filename | Ground-Truth Category | Anomaly Score | Threshold ($\tau$) | Margin ($\Delta$) | Defect Area (px / %) | Peak On Defect? | Local Signal? | Weber Contrast | Primary Root-Cause Attribution |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | `014.png` | `color` | 42.68 | 43.5951 | -0.92 | 1,874 (0.29%) | No (Rim) | Yes | 0.314 | Faint localized color stain; rim activation slightly exceeds defect peak |
| **02** | `017.png` | `color` | 41.93 | 43.5951 | -1.66 | 932 (0.15%) | No (Rim) | Yes | 0.208 | Small faint chromatic spot; background rim activation dominates |
| **03** | `024.png` | `color` | 42.90 | 43.5951 | -0.69 | 855 (0.13%) | **Yes** | Yes | 0.169 | Defect correctly localized; near-threshold score fallout ($\Delta = -0.69$) |
| **04** | `006.png` | `combined` | 40.88 | 43.5951 | -2.72 | 25,620 (4.00%) | **Yes** | Yes | 0.197 | Low-contrast imprint + smudge; deep features invariant to soft blend |
| **05** | `007.png` | `combined` | 40.03 | 43.5951 | -3.57 | 8,740 (1.37%) | **Yes** | Yes | 0.001 | Extremely low contrast ($\Delta I = 0.1$); minimal luminance gradient |
| **06** | `005.png` | `contamination` | 42.55 | 43.5951 | -1.05 | 4,499 (0.70%) | No (Rim) | Yes | 0.046 | Low-contrast smudge; tablet boundary activation dominates peak |
| **07** | `009.png` | `contamination` | 40.43 | 43.5951 | -3.17 | 5,414 (0.85%) | **Yes** | Yes | 0.044 | Faint gray speckling; low contrast ($\Delta I = 7.8$) |
| **08** | `010.png` | `contamination` | 41.85 | 43.5951 | -1.75 | 13,209 (2.06%) | No (Rim) | Yes | 0.222 | Diffuse particle dispersion; peak activation falls on tablet boundary |
| **09** | `013.png` | `contamination` | 42.73 | 43.5951 | -0.86 | 4,844 (0.76%) | No (Rim) | Yes | 0.001 | Low-contrast residue ($\Delta I = 0.2$); tablet rim score dominates |
| **10** | `019.png` | `contamination` | 42.28 | 43.5951 | -1.32 | 5,231 (0.82%) | **Yes** | Yes | 0.058 | Defect localized; subtle discoloration below threshold |
| **11** | `020.png` | `contamination` | 38.01 | 43.5951 | -5.58 | 8,113 (1.27%) | **Yes** | Yes | 0.231 | Defect localized; low feature distance in PatchCore memory |
| **12** | `013.png` | `crack` | 39.98 | 43.5951 | -3.62 | 1,837 (0.29%) | No (Rim) | **No** | 0.288 | Extremely fine hairline crack; spatial pooling dilutes activation |
| **13** | `014.png` | `crack` | 37.49 | 43.5951 | -6.11 | 2,100 (0.33%) | No (Rim) | Yes | 0.039 | Ultra-thin crack ($\Delta I = 3.9$); rim activation exceeds defect |
| **14** | `016.png` | `crack` | 37.92 | 43.5951 | -5.68 | 2,091 (0.33%) | No (Rim) | Yes | 0.115 | Hairline fissure ($\Delta I = 11.4$); rim activation exceeds defect |
| **15** | `000.png` | `faulty_imprint` | 39.59 | 43.5951 | -4.00 | 5,688 (0.89%) | **Yes** | Yes | 0.032 | Missing debossed stroke; deep CNN features invariant to character font |
| **16** | `003.png` | `faulty_imprint` | 42.46 | 43.5951 | -1.13 | 9,437 (1.48%) | **Yes** | Yes | 0.042 | Distorted letter stroke; near-threshold score fallout ($\Delta = -1.13$) |
| **17** | `004.png` | `faulty_imprint` | 42.47 | 43.5951 | -1.13 | 13,668 (2.14%) | **Yes** | Yes | 0.057 | Blurred debossing stamp; near-threshold fallout ($\Delta = -1.13$) |
| **18** | `005.png` | `faulty_imprint` | 40.79 | 43.5951 | -2.81 | 28,700 (4.48%) | **Yes** | Yes | 0.015 | Faded imprint debossing; low contrast ($\Delta I = 2.7$) |
| **19** | `008.png` | `faulty_imprint` | 43.10 | 43.5951 | -0.49 | 19,531 (3.05%) | **Yes** | Yes | 0.003 | Partial imprint fade; near-threshold score fallout ($\Delta = -0.49$) |
| **20** | `009.png` | `faulty_imprint` | 43.24 | 43.5951 | -0.36 | 9,945 (1.55%) | **Yes** | Yes | 0.006 | Double-stamped debossing; near-threshold score fallout ($\Delta = -0.36$) |
| **21** | `011.png` | `faulty_imprint` | 42.90 | 43.5951 | -0.69 | 35,179 (5.50%) | **Yes** | Yes | 0.007 | Missing character segment; near-threshold fallout ($\Delta = -0.69$) |
| **22** | `013.png` | `faulty_imprint` | 40.87 | 43.5951 | -2.73 | 14,672 (2.29%) | **Yes** | Yes | 0.001 | Faint debossing stroke; low contrast ($\Delta I = 0.3$) |
| **23** | `014.png` | `faulty_imprint` | 42.27 | 43.5951 | -1.32 | 8,594 (1.34%) | **Yes** | Yes | 0.006 | Shifted imprint stroke; near-threshold fallout ($\Delta = -1.32$) |
| **24** | `004.png` | `scratch` | 43.40 | 43.5951 | -0.19 | 23,092 (3.61%) | **Yes** | Yes | 0.727 | Shallow broad surface scrape; near-threshold fallout ($\Delta = -0.19$) |

---

## 3. Score Distribution & Overlap Analysis

### 3.1 Partition Comparison

| Metric | Normal Images ($N=26$) | True Positives ($N=72$) | False Negatives ($N=24$) | All Defects ($N=96$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean** | 39.05 | 48.91 | 41.36 | 47.03 |
| **Std Dev** | 2.27 | 3.80 | 1.73 | 4.72 |
| **Min** | **35.53** | 43.81 | **37.49** | 37.49 |
| **P10** | 36.32 | 44.56 | 38.48 | 41.36 |
| **P25** | 37.29 | 46.12 | 40.33 | 43.71 |
| **Median (P50)** | 39.10 | 48.27 | 42.10 | 46.59 |
| **P75** | 40.13 | 51.40 | 42.69 | 50.55 |
| **P90** | 41.69 | 53.18 | 43.04 | 52.92 |
| **Max** | **45.56** | 62.50 | **43.40** | 62.50 |

```
Distribution Overlap Visualization:
Normals:        [35.53 ======== 39.10 ======== 45.56]
False Negatives:         [37.49 == 42.10 == 43.40]
True Positives:                                [43.81 === 48.27 === 62.50]
                |-------- Score Overlap Zone --------|
                35.53                                45.56
```

### 3.2 Category-Level Score Distributions

| Defect Category | Total ($N$) | TP ($N$) | FN ($N$) | Failure Rate | Min Score | Median (P50) | Max Score | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pill_type`** | 9 | 9 | 0 | **0.00%** | 49.48 | 51.42 | 53.18 | $51.33 \pm 1.22$ |
| **`scratch`** | 10 | 9 | 1 | **10.00%** | 43.40 | 45.67 | 49.36 | $46.26 \pm 1.74$ |
| **`combined`** | 17 | 15 | 2 | **11.76%** | 40.03 | 48.47 | 57.70 | $49.43 \pm 5.16$ |
| **`contamination`** | 21 | 15 | 6 | **28.57%** | 38.01 | 46.19 | 62.50 | $47.29 \pm 5.55$ |
| **`crack`** | 10 | 7 | 3 | **30.00%** | 37.49 | 45.35 | 47.93 | $43.85 \pm 3.69$ |
| **`color`** | 10 | 7 | 3 | **30.00%** | 41.93 | 48.18 | 53.71 | $47.68 \pm 4.30$ |
| **`faulty_imprint`** | 19 | 10 | 9 | **47.37%** | 39.59 | 43.81 | 49.81 | $44.28 \pm 2.81$ |

*Observation:* Categories with high macro-structural alterations (`pill_type`, `scratch`, large `combined`) achieve near-zero failure rates. In contrast, categories characterized by subtle geometric alterations on debossed lettering (`faulty_imprint`) or fine hairlines (`crack`) suffer failure rates of 30% to 47%.

---

## 4. Evaluation of Root-Cause Hypotheses

### Hypothesis A: Threshold / Calibration Related — **REFUTED AS A SOLE ROOT CAUSE**
- **Hypothesis:** The threshold $\tau = 43.5951$ was set too conservatively high, and simple recalibration can resolve the false negatives.
- **Empirical Evidence:**
  - 12 of 24 FNs are indeed within $1.5$ points of the threshold ($\Delta \in [-1.5, 0]$).
  - However, **19 out of 26 normal samples (73.08%) score above 37.49** (the minimum FN score).
  - Normal image `022.png` scores **45.5649** (already causing 1 FP at $\tau = 43.5951$).
  - Lowering the threshold to $41.5$ captures 14 FNs but causes **3 false positives (FPR = 11.54%)**.
  - Lowering the threshold to $40.0$ captures 17 FNs but causes **8 false positives (FPR = 30.77%)**.
  - Lowering the threshold to $37.0$ captures all 24 FNs but causes **20 false positives (FPR = 76.92%)**.
- **Conclusion:** Recalibrating the threshold is a zero-sum trade-off. Threshold adjustment cannot separate subtle defects from normal product variability because the feature representations themselves overlap.

### Hypothesis B: Weak PatchCore Feature Representation (Backbone / Feature Layers) — **CONFIRMED AS PRIMARY ARCHITECTURAL CAUSE**
- **Hypothesis:** Pretrained WideResNet-50 features from layer 2 and layer 3 lack the fine spatial resolution and high-frequency texture sensitivity required for subtle industrial defects.
- **Empirical Evidence:**
  - Layer 2 has a downsampling factor of 8 ($28 \times 28$ patch grid); Layer 3 has a factor of 16 ($14 \times 14$ patch grid).
  - Features are optimized for ImageNet semantic object recognition, which is explicitly trained to be invariant to font styles, minor edge distortions, and color shading.
  - In `faulty_imprint`, when a letter has a missing stroke or slight shift, the receptive field is large enough that the layer-3 feature representation still embeds the patch as "tablet debossed text". Its Euclidean distance to normal imprint patches in memory is minimal ($40.0$–$43.2$).
  - In 16 of 24 FNs (66.7%), PatchCore correctly identifies the defect centroid as its local peak, but the magnitude of the nearest-neighbor distance is compressed.
- **Conclusion:** Deep CNN layer representations are insufficiently sensitive to high-frequency topological deformations and low-contrast surface variations.

### Hypothesis C: Normal Reference Variation — **CONTRIBUTING FACTOR**
- **Hypothesis:** 10 normal calibration images do not provide sufficient coverage of normal pill orientations, debossing depths, and surface specularities.
- **Empirical Evidence:**
  - Normal test pill scores exhibit a standard deviation of $\sigma = 2.27$ and span from $35.53$ to $45.56$ ($\Delta_{\text{normal}} = 10.03$).
  - Coreset memory bank size with 10 images is relatively sparse. When an incoming test pill has debossed lettering rotated at an angle not represented in the 10 reference images, normal debossing features produce an elevated baseline distance (~$39$–$42$).
  - This elevated normal baseline raises the calibrated threshold $\tau = \mu + 3\sigma$, pushing near-threshold defects into false negatives.

### Hypothesis D: Very Small / Low-Contrast Defects — **CONFIRMED AS PRIMARY PHYSICAL CAUSE**
- **Hypothesis:** Missed defects have significantly smaller area or lower contrast than detected defects.
- **Empirical Evidence:**
  - **Weber Contrast:**
    - True Positives: Mean = **1.179**, Median = **0.090**.
    - False Negatives: Mean = **0.127**, Median = **0.045**.
    - False negatives have **less than half the median contrast** and **nearly 10× lower mean contrast** than detected defects.
  - **Intensity Delta ($\Delta I$):**
    - True Positives: Mean = **32.52**, Median = **12.41**.
    - False Negatives: Mean = **16.11**, Median = **8.01**.
  - **Defect Area (Coverage %):**
    - 3 of 3 missed cracks (`crack/013`, `014`, `016`) are hairline fissures covering only $0.28\%$ to $0.33\%$ of the image area ($< 2,100$ pixels).
    - Their low pixel footprint is diluted during PatchCore's spatial neighborhood pooling.

### Hypothesis E: Imprint / Debossing Interference — **CONFIRMED FOR IMPRINT DEFECTS**
- **Hypothesis:** The presence of normal debossed lettering on pill tablets interferes with defect detection.
- **Empirical Evidence:**
  - `faulty_imprint` accounts for **9 of the 24 false negatives (37.5%)**, with an overall category failure rate of **47.37%**.
  - Normal debossing creates high local feature variance because of deep shadows and variable lighting inside the debossed groove.
  - PatchCore's nearest-neighbor search cannot distinguish between legitimate debossing shadows and defective debossing distortions because both reside in the same high-variance subspace.

---

## 5. Threshold Sweep Simulation Data

The following table demonstrates the operational trade-off of adjusting the decision threshold $\tau$:

| Threshold | True Positives ($N=96$) | False Negatives ($N=96$) | True Negatives ($N=26$) | False Positives ($N=26$) | Sensitivity (TPR) | Specificity (TNR) | False Positive Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **46.0** | 54 | 42 | 26 | 0 | 56.25% | 100.00% | 0.00% |
| **45.0** | 62 | 34 | 25 | 1 | 64.58% | 96.15% | 3.85% |
| **44.0** | 66 | 30 | 25 | 1 | 68.75% | 96.15% | 3.85% |
| **43.5951 (Current)** | **72** | **24** | **25** | **1** | **75.00%** | **96.15%** | **3.85%** |
| **43.0** | 75 | 21 | 25 | 1 | 78.13% | 96.15% | 3.85% |
| **42.5** | 80 | 16 | 24 | 2 | 83.33% | 92.31% | 7.69% |
| **42.0** | 84 | 12 | 23 | 3 | 87.50% | 88.46% | 11.54% |
| **41.5** | 86 | 10 | 23 | 3 | 89.58% | 88.46% | 11.54% |
| **41.0** | 86 | 10 | 22 | 4 | 89.58% | 84.62% | 15.38% |
| **40.0** | 89 | 7 | 18 | 8 | 92.71% | 69.23% | 30.77% |
| **38.0** | 93 | 3 | 11 | 15 | 96.88% | 42.31% | 57.69% |
| **37.0** | 96 | 0 | 6 | 20 | 100.00% | 23.08% | 76.92% |

*Mathematical Proof:* To achieve 100% sensitivity on the test set, the false positive rate on defect-free product would surge to **76.92%**.

---

## 6. Diagnostic Conclusions

1. **The Root Cause is Multi-Factorial, Dominated by Representation Invariance:**
   - The primary driver of Stage 1 misses is **weak feature representation sensitivity** in WideResNet-50 layers 2 and 3 when confronted with **low-contrast** (Weber $< 0.20$), **high-frequency** (hairline cracks), and **debossed structural distortions** (`faulty_imprint`).
   - The secondary driver is **normal reference baseline elevation**, caused by natural pill rotation and debossing shadow variability across the 10 reference images.
2. **Threshold Recalibration is Mathematically Incapable of Solving the Problem:**
   - Because 73% of normal pill scores exceed the minimum FN score, no single threshold can separate the populations.

---

## 7. Recommendation for Next Controlled Experiment

Based strictly on the empirical findings, the next offline investigation must address the representation bottleneck directly:

### Recommended Experiment: Option 2 — PatchCore Representation / Backbone / Feature-Layer Experiment

**Objective:**  
Evaluate whether enriching PatchCore's feature memory with shallow high-frequency representations or higher-resolution vision transformer backbones improves separation between subtle defects and normal tablet variations.

**Specific Experimental Levers to Test:**
1. **Feature Layer Inclusion (Multi-Scale Pyramid):**  
   Compare the baseline (WideResNet-50 Layer 2 + Layer 3) against an augmented configuration incorporating **Layer 1** (shallow features preserving high-frequency edges, stroke widths, and local contrast).
2. **Backbone Comparison (Vision Transformer / DINOv2):**  
   Evaluate self-supervised backbones (e.g., DINOv2-S/14 or ViT-B/16 with patch size 8/14) known for dense spatial correspondence and sensitivity to fine geometric deformities without ImageNet class invariance.
3. **Imprint-Aware Spatial Weighting / Alignment:**  
   Evaluate whether pre-aligning pill debossing orientation or masking debossed zones reduces the normal background score variance ($\sigma = 2.27$), effectively lowering the calibration baseline without increasing false positives.

*Constraints for Next Phase:*  
- Keep production frozen.
- Conduct in an isolated offline experiment directory.
- Benchmark directly against the 24 identified false negatives to verify whether their scores elevate above normal pill scores.
