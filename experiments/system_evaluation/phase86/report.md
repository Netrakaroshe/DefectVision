# Phase 8.6: Pill PatchCore Representation Experiment — Evaluation Report

**Evaluation Date:** October 7, 2026  
**Environment:** Windows 11, PyTorch 2.13.0+cu132, CUDA 13.2, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Evaluated Product:** Pill (`pill2` product profile)  
**Execution Script:** `scratch/run_phase86_experiment.py`  
**Artifact Directory:** `experiments/system_evaluation/phase86/`  
**Production Status:** 100% Frozen (Audit Only — Zero Production Changes)

---

## Important Clarification on Operating Points & Measured Resources

> **CLARIFICATION ON EVALUATION CRITERIA & RESOURCES:**  
> 1. The **96% specificity / 4% FPR** operating point is **NOT** a hard acceptance criterion and must **not** be interpreted as a project pass/fail gate. It is strictly a reference operating point along the Pareto frontier for comparing candidate trade-offs.  
> 2. The preliminary VRAM and latency numbers reported in the Phase 8.6 design report were pre-execution feasibility estimates. All VRAM and latency figures presented in this report represent **empirically measured values** captured during steady-state GPU execution with CUDA events and `torch.cuda.max_memory_allocated()`.  
> 3. No candidate in this experiment is declared "production-ready."

---

## 1. Executive Summary & Core Findings

This experiment executes the approved Phase 8.6 experimental design comparing four PatchCore representation variants offline on the exact 10 normal calibration reference images and 122 evaluation images (26 normal, 96 defective across 7 categories):

1. **Variant A (Frozen Baseline):** WideResNet-50-2 with layers `('layer2', 'layer3')`
2. **Variant B (Multi-Scale CNN):** WideResNet-50-2 with layers `('layer1', 'layer2', 'layer3')`
3. **Variant C (Dense Self-Supervised ViT):** DINOv2 `vit_small_patch14_dinov2` with blocks `('blocks.8', 'blocks.11')`
4. **Variant D (Aligned Baseline):** Baseline representation evaluated with Pill canonical in-plane orientation alignment

### Core Empirical Findings:

1. **Variant A (Frozen Baseline) Remains the Most Balanced Overall Representation:**
   - **AUROC = 0.9451**, Sensitivity = **75.00%** (72/96), Specificity = **96.15%** (25/26), FPR = **3.85%** (1/26).
   - Demonstrates strong separation on macro defects (`pill_type`: 100%, `scratch`: 90%, `combined`: 88.2%), but suffers on fine imprint debossing (`faulty_imprint`: 52.6%, 9 FNs).
   - Low resource footprint: **39.3 ms P50 latency**, **138.2 MB peak VRAM**.

2. **Variant B (Multi-Scale CNN) Suffers Calibration Breakdown due to Texture Noise:**
   - Adding Layer 1 preserves high-frequency edge information, yielding a competitive threshold-independent **AUROC = 0.9399** and slightly higher sensitivity at 4% FPR (83.3% vs 80.2%).
   - **Severe Operational Breakdown:** Shallow Layer 1 features dramatically expand normal tablet coating variance ($\sigma_{\text{ref}}$ increased from $4.36$ to $6.86$). The calibrated threshold ($\mu + 3\sigma$) surged from $43.5951$ to **$51.3986$**.
   - At this elevated operating point, sensitivity collapsed to **31.25%** (30/96 TP, 66 FN). Micro-cracks (0/10) and scratches (0/10) were completely missed.
   - Resource overhead was severe: memory bank expanded 4× to **3,276 vectors**, latency slowed 7.5× to **298.0 ms P50**, and peak VRAM increased to **607.0 MB**.

3. **Variant C (DINOv2) Fixes Imprints but Fails on Color Defects & Specificity:**
   - DINOv2 self-supervised patch tokens successfully broke semantic invariance on debossed text: `faulty_imprint` sensitivity jumped from 52.6% to **84.2%** (16/19 detected, recovering 7 Phase 8.5 imprint FNs), and `contamination` jumped to **85.7%** (18/21).
   - **Critical Vulnerability on Color & Normals:** DINOv2 patch features are largely invariant to subtle chromatic tint shifts. Sensitivity on `color` defects collapsed from 70.0% to **10.0%** (only 1 of 10 detected; 9 missed).
   - Specificity on normal tablets dropped to **80.77%** (FPR = **19.23%**, 5 false positives out of 26 normal pills).
   - Overall AUROC dropped substantially from 0.9451 to **0.8718**. At a matched 4% FPR, DINOv2 TPR is only 67.7% (vs 80.2% for baseline).

4. **Variant D (Pill Alignment) Yields Marginal Negative Impact:**
   - Canonical major-axis ellipse rotation did not reduce normal debossing variance ($\sigma_{\text{ref}} = 4.46$ vs $4.36$).
   - AUROC slightly decreased to **0.9347**, sensitivity dropped to **72.92%** (70/96), and latency increased to **126.8 ms** due to affine interpolation blur along tablet contours.

---

## 2. Section A: Full Variant Comparison Table

| Metric | Variant A (Frozen Baseline) | Variant B (Multi-Scale CNN) | Variant C (DINOv2 ViT) | Variant D (Aligned Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone Architecture** | WideResNet-50-2 | WideResNet-50-2 | `vit_small_patch14_dinov2` | WideResNet-50-2 |
| **Feature Layer Endpoints** | `('layer2', 'layer3')` | `('layer1', 'layer2', 'layer3')` | `('blocks.8', 'blocks.11')` | `('layer2', 'layer3')` |
| **Image Input Size** | $256 \times 256$ | $256 \times 256$ | $224 \times 224$ | $256 \times 256$ (Aligned) |
| **Model Parameters** | 24.86 M | 24.86 M | 22.06 M | 24.86 M |
| **Memory Bank Shape** | `[819, 1536]` | `[3276, 1792]` | `[204, 768]` | `[819, 1536]` |
| **Calibrated Threshold ($\tau$)** | **43.5951** | **51.3986** | **35.5992** | **44.1196** |
| **Reference Score $\mu \pm s$** | $30.52 \pm 4.36$ | $30.82 \pm 6.86$ | $30.03 \pm 1.86$ | $30.75 \pm 4.46$ |
| **Image AUROC** | **0.9451** | **0.9399** | **0.8718** | **0.9347** |
| **True Positives (TP / 96)** | 72 | 30 | 76 | 70 |
| **True Negatives (TN / 26)** | 25 | 26 | 21 | 25 |
| **False Positives (FP / 26)** | 1 | 0 | 5 | 1 |
| **False Negatives (FN / 96)** | 24 | 66 | 20 | 26 |
| **Sensitivity (TPR)** | **75.00%** | **31.25%** | **79.17%** | **72.92%** |
| **Specificity (TNR)** | **96.15%** | **100.00%** | **80.77%** | **96.15%** |
| **False Positive Rate (FPR)** | **3.85%** | **0.00%** | **19.23%** | **3.85%** |
| **Score Separation Margin** | $-8.0778$ | $-8.3349$ | $-13.1473$ | $-8.8405$ |
| **Phase 8.5 FNs Recovered** | Baseline (0/24) | 0 / 24 | **13 / 24** | 2 / 24 |
| **Newly Lost Baseline Defects**| 0 | 42 | 9 | 4 |
| **Mean Defect Coverage** | **74.7%** | 63.5% | 73.1% | 71.0% |
| **Mean Bounding Box IoU** | **0.360** | 0.224 | 0.311 | 0.346 |
| **Peak-on-Defect Rate** | **66.7%** | 37.5% | 60.4% | **66.7%** |
| **P50 Inference Latency** | **39.3 ms** | 298.0 ms | 86.0 ms | 126.8 ms |
| **Peak VRAM Allocated** | **138.2 MB** | 607.0 MB | **133.0 MB** | 223.0 MB |

---

## 3. Section B: Per-Category Comparison Table

Sensitivity broken down across all seven MVTec Pill defect categories ($N_{\text{detected}} / N_{\text{total}}$):

| Defect Category | Total Samples ($N$) | Variant A (Baseline) | Variant B (Multi-Scale) | Variant C (DINOv2) | Variant D (Aligned) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`faulty_imprint`** | 19 | 52.6% (10/19) | 5.3% (1/19) | **84.2% (16/19)** *(+31.6%)* | 52.6% (10/19) |
| **`contamination`** | 21 | 71.4% (15/21) | 28.6% (6/21) | **85.7% (18/21)** *(+14.3%)* | 71.4% (15/21) |
| **`combined`** | 17 | 88.2% (15/17) | 47.1% (8/17) | **100.0% (17/17)** *(+11.8%)* | 82.4% (14/17) |
| **`color`** | 10 | **70.0% (7/10)** | 60.0% (6/10) | **10.0% (1/10)** *(-60.0%)* | 60.0% (6/10) |
| **`crack`** | 10 | **70.0% (7/10)** | 0.0% (0/10) | 60.0% (6/10) | **70.0% (7/10)** |
| **`scratch`** | 10 | **90.0% (9/10)** | 0.0% (0/10) | **90.0% (9/10)** | **90.0% (9/10)** |
| **`pill_type`** | 9 | **100.0% (9/9)** | **100.0% (9/9)** | **100.0% (9/9)** | **100.0% (9/9)** |
| **Overall Sensitivity** | **96** | **75.0% (72/96)** | **31.25% (30/96)** | **79.2% (76/96)** | **72.9% (70/96)** |

### Key Category Insights:
- **`faulty_imprint` Transformation in Variant C:** DINOv2 dramatically reduced imprint misses from 9 down to 3. Dense self-supervised patch tokens are sensitive to missing letter strokes and distorted debossing geometry.
- **`color` Catastrophe in Variant C:** DINOv2 missed 9 out of 10 color defects. Self-supervised pretraining encourages invariance to smooth illumination and chromatic shifts, rendering faint pink/brown pill discolorations undetectable.
- **Collapse of Fine Defects in Variant B:** Due to the elevated $51.3986$ threshold, Variant B detected zero cracks (0/10) and zero scratches (0/10).

---

## 4. Section C: Exact Samples Recovered and Lost Relative to Variant A

All transitions relative to the 24 Phase 8.5 baseline False Negatives are tracked below:

### 4.1 Variant B Transitions
- **Recovered from Phase 8.5 FNs ($N=0$):** None.
- **Still Missed ($N=24$):** All 24 Phase 8.5 FNs remained missed.
- **Newly Lost Samples ($N=42$):** 42 defect samples that Variant A successfully detected were lost due to threshold elevation:
  - `crack` (7 lost): `012.png`, `015.png`, `017.png`, `018.png`, `019.png`, `020.png`, `021.png`
  - `scratch` (9 lost): `000.png`, `001.png`, `002.png`, `003.png`, `012.png`, `013.png`, `014.png`, `015.png`, `016.png`
  - `faulty_imprint` (9 lost): `001.png`, `002.png`, `006.png`, `007.png`, `012.png`, `015.png`, `016.png`, `017.png`, `018.png`
  - `contamination` (9 lost): `000.png`, `001.png`, `002.png`, `004.png`, `006.png`, `007.png`, `012.png`, `014.png`, `017.png`
  - `combined` (7 lost): `000.png`, `001.png`, `004.png`, `005.png`, `008.png`, `010.png`, `015.png`
  - `color` (1 lost): `015.png`

### 4.2 Variant C Transitions
- **Recovered from Phase 8.5 FNs ($N=13$):** 13 previously missed defects were successfully detected:
  - `faulty_imprint` (7 recovered): `003.png`, `004.png`, `005.png`, `008.png`, `011.png`, `013.png`, `014.png`
  - `contamination` (3 recovered): `010.png`, `019.png`, `020.png`
  - `combined` (2 recovered): `006.png`, `007.png`
  - `scratch` (1 recovered): `004.png`
- **Still Missed ($N=11$):**
  - `faulty_imprint` (2): `000.png`, `009.png`
  - `contamination` (3): `005.png`, `009.png`, `013.png`
  - `crack` (3): `013.png`, `014.png`, `016.png`
  - `color` (3): `014.png`, `017.png`, `024.png`
- **Newly Lost Samples ($N=9$):**
  - `color` (6 newly lost): `012.png`, `013.png`, `015.png`, `016.png`, `018.png`, `020.png`
  - `crack` (1 newly lost): `015.png`
  - `faulty_imprint` (1 newly lost): `015.png`
  - `scratch` (1 newly lost): `001.png`

### 4.3 Variant D Transitions
- **Recovered from Phase 8.5 FNs ($N=2$):** `contamination/019.png`, `faulty_imprint/008.png`.
- **Newly Lost Samples ($N=4$):** `color/015.png`, `combined/005.png`, `contamination/004.png`, `faulty_imprint/012.png`.

---

## 5. Section D: Threshold-Independent ROC Analysis

To decouple raw representation separability from the specific $\mu + 3\sigma$ calibration heuristic, receiver operating characteristic curves were computed across the entire score range:

| Metric | Variant A (Baseline) | Variant B (Multi-Scale) | Variant C (DINOv2) | Variant D (Aligned) |
| :--- | :---: | :---: | :---: | :---: |
| **Global Image AUROC** | **0.9451** | 0.9399 | 0.8718 | 0.9347 |
| **TPR at 0.0% FPR (Zero False Positives)** | 0.0% | 0.0% | 0.0% | 0.0% |
| **TPR at 4.0% FPR (Reference Operating Point)** | 80.21% (77/96) | **83.33% (80/96)** | 67.71% (65/96) | 76.04% (73/96) |
| **TPR at 10.0% FPR (Moderate Tolerance)** | **87.50% (84/96)** | 83.33% (80/96) | 68.75% (66/96) | 81.25% (78/96) |
| **TPR at 20.0% FPR (High Tolerance)** | **92.71% (89/96)** | 89.58% (86/96) | 79.17% (76/96) | 90.62% (87/96) |

### ROC Curve Insights:
1. **Variant A dominates across the operational curve:** At 10% FPR and 20% FPR, Variant A delivers the highest true-positive rate (87.5% and 92.7%).
2. **Variant B has theoretical potential if calibrated differently:** At 4% FPR, Variant B achieves 83.3% sensitivity (3 defects higher than Variant A). Its catastrophic 31.25% operational sensitivity is solely an artifact of the $\mu + 3\sigma$ calibration heuristic reacting to Layer 1 texture variance.
3. **Variant C is fundamentally unseparated:** At every fixed FPR level (4%, 10%, 20%), Variant C is strictly inferior to Variant A. Its apparent 79.17% sensitivity at calibrated $\tau$ was achieved solely by operating at a high FPR of 19.23%.

---

## 6. Section E: Quantitative Defect Localization Comparison

Evaluated on all defective test samples with ground-truth masks ($N=96$):

| Localization Metric | Variant A (Baseline) | Variant B (Multi-Scale) | Variant C (DINOv2) | Variant D (Aligned) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Defect Pixel Coverage** | **74.67%** | 63.48% | 73.12% | 71.01% |
| **Mean Bounding Box IoU ($\text{IoU}_{\text{box}}$)** | **0.3601** | 0.2243 | 0.3114 | 0.3460 |
| **Mean Defect Purity** | **23.51%** | 16.29% | 21.05% | 22.84% |
| **Peak-on-Defect Localization Rate** | **66.67% (64/96)** | 37.50% (36/96) | 60.42% (58/96) | **66.67% (64/96)** |

*Observation:* Variant A produces the highest spatial fidelity ($\text{IoU}_{\text{box}} = 0.360$, Coverage = $74.7\%$, Peak on Defect = $66.7\%$). In Variant B, Layer 1 texture noise dispersed the anomaly map into multiple fragmented local maxima, dropping IoU to $0.224$ and peak-on-defect rate to $37.5\%$.

---

## 7. Section F: Computational Cost & Hardware Resource Comparison

Measured on local NVIDIA GeForce RTX 4060 Laptop GPU under steady-state CUDA execution:

| Resource Metric | Variant A (Baseline) | Variant B (Multi-Scale) | Variant C (DINOv2) | Variant D (Aligned) |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone Architecture** | WideResNet-50-2 | WideResNet-50-2 | `vit_small_patch14_dinov2` | WideResNet-50-2 |
| **Model Parameters** | 24.86 M | 24.86 M | **22.06 M** | 24.86 M |
| **Memory Bank Tensor Shape** | `[819, 1536]` | `[3276, 1792]` | `[204, 768]` | `[819, 1536]` |
| **Memory Bank Size in RAM** | 5.03 MB | 23.48 MB | **0.63 MB** | 5.03 MB |
| **Offline Fitting Time (10 refs)**| Pre-fitted | 9.2 s | 17.7 s | 4.1 s |
| **Peak VRAM (Inference)** | 138.2 MB | 607.0 MB | **133.0 MB** | 223.0 MB |
| **Mean Latency (50 cycles)** | **40.4 ms** | 296.8 ms | 86.5 ms | 128.5 ms |
| **P50 Latency** | **39.3 ms** | 298.0 ms | 86.0 ms | 126.8 ms |
| **P95 Latency** | **42.1 ms** | 308.2 ms | 91.6 ms | 143.7 ms |
| **Max Latency** | **44.9 ms** | 316.4 ms | 95.8 ms | 151.2 ms |
| **Effective Frame Rate** | **~25 FPS** | ~3.3 FPS | ~11.5 FPS | ~7.8 FPS |

*Hardware Feasibility Verdict:* All variants executed well within the 8GB VRAM capacity of the RTX 4060 (maximum peak VRAM was 607 MB for Variant B). However, Variant B incurs a severe latency penalty ($298$ ms, $3.3$ FPS), whereas Variant A runs at a highly responsive $39.3$ ms ($25$ FPS).

---

## 8. Section G: Evidence-Based Interpretation & Failure Mode Analysis

### 1. Why Did Multi-Scale CNN (Variant B) Fail Operationally?
- **Theoretical Expectation:** Layer 1 features would capture fine debossing edges and micro-cracks before spatial downsampling.
- **Empirical Reality:** Layer 1 features represent raw pixel-level gradient and texture statistics. While defective edges produced higher activations, normal pill surfaces also exhibited high Layer 1 feature distances due to microscopic tablet powder granularity, specular reflection on the coating, and debossed shadow edges.
- **The Variance Explosion Mechanism:** Standard deviation on normal reference samples increased by **57.3%** ($\sigma = 6.86$ vs $4.36$). Because threshold calibration scales with $3\sigma$, $\tau$ rose by **7.80 points** ($51.3986$). This elevation far outstripped the defect signal gain, causing 42 previously detected defects to fall below threshold.

### 2. Why Did DINOv2 (Variant C) Show Polarization (High Imprint / Zero Color)?
- **Debossing Success:** DINOv2 is trained via self-supervised masked image modeling and patch contrastive learning. Patch tokens capture dense spatial correspondence. When debossed characters ("N", "5") have missing or shifted strokes, the patch tokens deviate sharply from normal imprint patterns, yielding high anomaly scores ($84.2\%$ imprint sensitivity).
- **Color Failure:** DINOv2 features are largely invariant to smooth color changes and lighting gradients. Low-contrast discolorations (pastel smudges, faint brown spots) do not disrupt patch self-attention topologies. As a result, 9 out of 10 color defects were completely missed.
- **Normal Surface Instability:** Vision transformers lack the spatial inductive bias of convolutions. Without a large reference set, natural pill surface illumination variations produced patch distances exceeding the reference threshold, causing 5 false positives ($19.23\%$ FPR).

### 3. Why Did Tablet Alignment (Variant D) Provide No Benefit?
- Major-axis rotation effectively standardizes horizontal tablet orientation, but affine interpolation (bilinear resampling) introduces slight smoothing along tablet contours.
- Normal score variance remained essentially identical ($s = 4.46$ vs $4.36$), proving that rotational variation is not the primary driver of normal score spread in WideResNet-50.

---

## 9. Section H: Actionable Recommendations & Next Experiment

Based strictly on empirical evidence:

### 1. Production Decision:
- **Do NOT replace the production model.** Variant A (frozen baseline) remains strictly superior in overall AUROC ($0.9451$), normal specificity ($96.15\%$), color defect detection ($70.0\%$), spatial localization ($\text{IoU} = 0.360$), and throughput ($39.3$ ms P50).
- Neither Variant B nor Variant C represents a drop-in replacement.

### 2. Recommended Next Investigation:
The findings point to an architectural insight:
- **WideResNet-50** is superior for global surface uniformity, color consistency, and texture stability (low FPR, high AUROC).
- **DINOv2** is superior specifically for debossed font, stroke, and geometric deformities (84.2% imprint sensitivity).

**Recommended Direction:**  
Rather than seeking a single monolithic backbone, evaluate a **Dual-Stream / Hybrid Feature Ensemble** offline:
1. Concatenate selected DINOv2 imprint tokens with WideResNet Layer 2+3 features, or
2. Evaluate a dedicated **Imprint Inspection Sub-Head** that applies DINOv2 exclusively to the central debossed lettering ROI, while retaining WideResNet PatchCore for global surface anomaly detection.

---

## 10. Audit Verification Statement

- **Production Source Code (`src/`):** **0 files modified.** Verified via filesystem audit.
- **Production Checkpoints (`data/app_storage/`):** **0 files modified.** Production model `pill2_patchcore.ckpt` remains frozen.
- **Production Database (`defectvision.db`):** Untouched.
- **Frontend Code (`frontend/`):** Untouched.
- **Raw Datasets (`D:/datasets/`):** Read-only and untouched.
- **Artifacts Created:** Stored strictly within `experiments/system_evaluation/phase86/`.
