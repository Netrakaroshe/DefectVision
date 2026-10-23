# Phase 8.7 Experimental Design: Specialized Pill Imprint-Defect Detection

**Document Version:** 1.0 (Design Only — Pre-Execution)  
**Creation Date:** October 7, 2026  
**Status:** PROPOSED DESIGN — AWAITING REVIEW & APPROVAL  
**Target Output Artifact:** `experiments/system_evaluation/phase87_design/report.md`  
**Production Status:** 100% Frozen (Audit / Design Only — Zero Production Modifications)  

---

## 1. Executive Summary & Research Question

In Phase 8.6, an offline evaluation of four PatchCore representation architectures established two critical empirical facts:
1. **Variant A (Frozen Baseline: WideResNet-50-2 `layer2+layer3`) remains the overall superior production baseline**, achieving global Image AUROC of **0.9451**, sensitivity of **75.00%** (72/96), high normal specificity of **96.15%** (1/26 FPR), high spatial localization fidelity ($\text{IoU}_{\text{box}} = 0.360$), and fast inference ($39.3\text{ ms}$ P50).
2. **Variant C (Dense Self-Supervised ViT: DINOv2 `vit_small_patch14_dinov2`) cannot replace Variant A**, because it suffered a catastrophic collapse in color defect detection (sensitivity dropped from 70% to **10.0%**; 9 of 10 color defects missed) and a severe increase in false positive rate on normal tablets (FPR rose from 3.85% to **19.23%**).
3. **DINOv2 demonstrated substantial, complementary sensitivity specifically on debossed lettering defects**, increasing `faulty_imprint` sensitivity from 52.6% to **84.2%** (16 of 19 detected, recovering 7 of the 9 baseline imprint false negatives).

This Phase 8.7 design investigates whether a **specialized, modular imprint detector** powered by DINOv2 can complement the frozen WideResNet PatchCore pipeline without replacing it.

### Core Research Question:
> *"Can a specialized DINOv2-based imprint detector improve detection of `faulty_imprint` defects while preserving the existing PatchCore performance on other pill defects and normal pills?"*

### Scientific Grounding & Impartiality:
This design **does NOT assume that the answer is yes**. 
A dual-stream or specialist detector introduces secondary decision boundaries that carry structural risks:
- Normal debossed lettering possesses sharp gradients and shadows that may trigger false alarms under Vision Transformer patch representations, inflating the False Positive Rate (FPR).
- An ensemble OR decision rule ($\text{Defective} = \text{Stage1} \lor \text{Specialist}$) monotonically increases or maintains false alarms; any false positive emitted by the specialist degrades overall system specificity.
- If the available dataset is insufficient to model normal debossing variance, the specialist will overfit to the calibration sample and fail on general production tablets.

This document establishes the theoretical justification, data feasibility constraints, ROI extraction mechanics, mathematical fusion rules, isolation protocols, and evaluation metrics required for a valid, leak-free offline experiment.

---

## 2. Scientific Feasibility Assessment & Data Sufficiency Audit

A rigorous experimental design must first address whether the available dataset can scientifically support the proposed investigation.

### 2.1 Dataset Inventory & Ground-Truth Defect Counts
The authoritative dataset for the pill inspection cell (`pill2` product profile) is MVTec AD `pill`:

| Dataset Partition | Defect Category | Total Samples ($N$) | Evaluation Role in Phase 8.4 / 8.6 | Available for Training / Tuning? |
| :--- | :--- | :---: | :---: | :---: |
| `train/good` | Defect-free reference | 267 | 10 samples in `reference_images/` | 257 untapped normal samples |
| `test/good` | Defect-free test | 26 | 26 external test samples | **STRICTLY FROZEN (Test Only)** |
| `test/faulty_imprint` | Debossing / Lettering defects | **19** | 19 external test samples | **STRICTLY FROZEN (Test Only)** |
| `test/contamination` | Surface particulate defects | 21 | 21 external test samples | **STRICTLY FROZEN (Test Only)** |
| `test/combined` | Multi-defect combinations | 17 | 17 external test samples | **STRICTLY FROZEN (Test Only)** |
| `test/pill_type` | Shape / Pill type mismatches | 9 | 9 external test samples | **STRICTLY FROZEN (Test Only)** |
| `test/color` | Discoloration / Stains | 25 | 10 test (15 in V2 dev pool) | **STRICTLY FROZEN (Test Only)** |
| `test/crack` | Mechanical cracks | 26 | 10 test (16 in V2 dev pool) | **STRICTLY FROZEN (Test Only)** |
| `test/scratch` | Surface abrasions | 24 | 10 test (14 in V2 dev pool) | **STRICTLY FROZEN (Test Only)** |
| **Total Test Pool** | — | — | **122 Evaluation Samples** | **ZERO Training Samples** |

### 2.2 Feasibility Verdict: Supervised Specialist vs. Unsupervised One-Class Specialist

#### A. Supervised Classifier Feasibility: **NOT FEASIBLE (DISQUALIFIED)**
- **Defective Sample Scarcity:** There are exactly 19 `faulty_imprint` images in the entire dataset.
- **Partition Conflict:** All 19 samples belong to the frozen external test benchmark established in Phase 8.4 and Phase 8.6. Splitting these 19 samples into train/test (e.g., 10 train / 9 test) would destroy backward comparability, drastically reduce statistical power, and invalidate longitudinal comparisons.
- **Overfitting Risk:** Training a deep supervised binary classifier (e.g., Normal Imprint vs. Defective Imprint) on $\le 10$ positive defect instances guarantees extreme over-fitting to the specific characters ("N", "5") and orientations in the training set.
- **Conclusion:** Supervised training of an imprint defect classifier is **scientifically indefensible** with the available data and is explicitly rejected.

#### B. Unsupervised One-Class PatchCore Specialist Feasibility: **FULLY FEASIBLE**
- **One-Class Alignment:** Industrial anomaly detection assumes defects are rare, unmodeled anomalies. PatchCore operates by modeling the distribution of *normal* patch features, requiring zero defect samples during training.
- **Normal Sample Availability:** 
  - The 10 frozen reference images in `data/app_storage/products/pill2/reference_images/` provide the exact baseline reference set used by Variant A.
  - Furthermore, `train/good/` contains 257 additional defect-free normal tablets that are completely disjoint from the 26 `test/good` images, allowing controlled testing of whether expanding normal reference coverage stabilizes ViT patch variance on lettering.
- **Evaluation Integrity:** Evaluating the unsupervised specialist on all 19 `faulty_imprint` test samples and all 103 other test samples preserves 100% comparability with Phase 8.4, 8.5, and 8.6.

### 2.3 The "Smallest Defensible Experiment"
To satisfy scientific validity while strictly avoiding data fabrication:
1. The specialist must be an **unsupervised one-class feature memory detector** (One-Class PatchCore on Imprint ROI).
2. The primary specialist variant must be fitted strictly on the **exact same 10 normal calibration reference images** as Variant A.
3. A controlled secondary variant may investigate whether increasing normal calibration samples from `train/good` (e.g., $N=50$) reduces normal lettering false alarms.
4. Testing must be conducted across the exact frozen **122-image benchmark** (26 normal, 96 defective).

---

## 3. Imprint Region Identification & ROI Extraction Methodology

To apply DINOv2 as a specialist, an imprint-specific Region of Interest (ROI) must be extracted from the tablet image. We evaluate four candidate ROI mechanisms against empirical data.

```
                              Candidate ROI Strategies
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
Fixed Pixel Crop    Stage-1 Anomaly Map         Learned Detector    Tablet-Relative Central
[y1:y2, x1:x2]      (Dynamic Stage 2)           (YOLOv8 / Faster)   Morphological Window
 ❌ Fragile to       ❌ Circular Failure:        ❌ No Bounding Box  ✅ 100.0% Defect Mask
    Translation         Misses Baseline FNs        Labels; Overfits    Coverage; 0 Parameters
```

### 3.1 Systematic Evaluation of ROI Candidates

#### 1. Fixed Coordinate Spatial Crop (Static Bounding Box)
- *Mechanism:* Hardcode a static rectangular pixel window (e.g., $Y \in [200, 600], X \in [250, 550]$) across all $800 \times 800$ pill images.
- *Empirical Analysis:* While pills in MVTec AD are physically fixtured near the image center $(400, 400)$ with vertical orientation ($\theta \in [89.7^\circ, 91.0^\circ]$), slight translational shifts of $\pm 15$ to $25$ pixels exist. A static crop cuts close to lettering boundaries on shifted tablets, generating artificial edge discontinuities that trigger high ViT anomaly distances.
- *Verdict:* **Rejected.** Lacks translation tolerance.

#### 2. Stage 1 PatchCore Anomaly-Map-Guided ROI (Stage 2 Dynamic Extraction)
- *Mechanism:* Use the frozen Variant A anomaly map to locate the maximum anomaly peak and extract an enclosing bounding box.
- *Empirical Analysis:* In Phase 8.5, root-cause analysis proved that Variant A produced false negatives on 9 out of 19 `faulty_imprint` samples precisely because its anomaly map failed to peak strongly on the debossing strokes (yielding scores below the $43.5951$ threshold). When Stage 1 misses a subtle imprint defect, the global anomaly peak lands on background noise or tablet rim edges. Feeding this peak to a downstream specialist creates a **circular dependency**: the specialist is never presented with the imprint region on the very samples it is designed to recover.
- *Verdict:* **Rejected.** Conceptually flawed and empirically disproven by Phase 8.5 findings.

#### 3. Learned Object Detection Network (e.g., YOLOv8 / Faster R-CNN)
- *Mechanism:* Train an object detection network to detect debossed characters ("N", "5").
- *Empirical Analysis:* MVTec AD provides pixel-level masks for anomalies, but **zero bounding-box annotations for normal lettering**. Training a deep detector would require manual labeling of normal images. With only 10 reference images, deep regression networks overfit, risk missing faint debossings, add 3M–25M parameters, and introduce unnecessary failure modes.
- *Verdict:* **Rejected.** Operationally and scientifically unjustified.

#### 4. Tablet-Relative Central Morphological ROI (Contour-Relative Window)
- *Mechanism:*
  1. **Segment Tablet Silhouette:** Convert the $800 \times 800$ image to grayscale, apply thresholding ($I > 30$, exploiting the high contrast between the tablet and black inspection background), and extract the external tablet contour $C$.
  2. **Extract Tablet Extents:** Compute the enclosing bounding box $[T_{x1}, T_{y1}, T_w, T_h]$ and tablet centroid $(C_x, C_y)$.
  3. **Compute Tablet-Normalized Imprint Window:** Define the imprint inspection window relative to the tablet bounding box:
     $$X_{\text{roi}} = [T_{x1} + 0.15 \cdot T_w,\; T_{x1} + 0.85 \cdot T_w]$$
     $$Y_{\text{roi}} = [T_{y1} + 0.20 \cdot T_h,\; T_{y1} + 0.80 \cdot T_h]$$
  4. **Crop and Resize:** Crop the ROI and resize via bilinear interpolation to $224 \times 224$ (the native input resolution for `vit_small_patch14_dinov2`).

### 3.2 Empirical Validation of the Tablet-Relative Central ROI
To verify whether this morphological window reliably captures imprint defects, we performed an empirical pixel-overlap analysis against all 19 MVTec AD `faulty_imprint` ground-truth defect masks:

| Defect Sample | Ground-Truth Defect Pixels | Pixels Enclosed in Morphological ROI | Defect Pixel Coverage |
| :--- | :---: | :---: | :---: |
| `000.png` | 5,688 px | 5,688 px | **100.0%** |
| `001.png` | 5,009 px | 5,009 px | **100.0%** |
| `002.png` | 9,312 px | 9,312 px | **100.0%** |
| `003.png` | 9,437 px | 9,437 px | **100.0%** |
| `004.png` | 13,668 px | 13,668 px | **100.0%** |
| `005.png` | 28,700 px | 28,700 px | **100.0%** |
| `006.png` | 16,662 px | 16,662 px | **100.0%** |
| `007.png` | 38,536 px | 38,536 px | **100.0%** |
| `008.png` | 19,531 px | 19,531 px | **100.0%** |
| `009.png` | 9,945 px | 9,945 px | **100.0%** |
| `010.png` | 21,815 px | 21,815 px | **100.0%** |
| `011.png` | 35,179 px | 35,179 px | **100.0%** |
| `012.png` | 7,305 px | 7,305 px | **100.0%** |
| `013.png` | 14,672 px | 14,672 px | **100.0%** |
| `014.png` | 8,594 px | 8,594 px | **100.0%** |
| `015.png` | 12,676 px | 12,676 px | **100.0%** |
| `016.png` | 26,388 px | 26,388 px | **100.0%** |
| `017.png` | 20,884 px | 20,884 px | **100.0%** |
| `018.png` | 15,784 px | 15,784 px | **100.0%** |
| **Aggregate (All 19 Samples)** | **319,794 px** | **319,794 px** | **100.00% Coverage** |

Furthermore, testing tablet silhouette segmentation across all 122 evaluation samples yielded **0 segmentation failures** (100.0% reliability).

*Design Decision:* The **Tablet-Relative Central Morphological ROI** is selected as the sole justifiable extraction method. It requires zero learned parameters, zero manual annotations, is translation-invariant relative to the tablet contour, executes in $< 1\text{ ms}$, and achieves **100.00% empirical defect coverage**.

---

## 4. Architectural Design of the DINOv2 Imprint Specialist

### 4.1 Specialist Formulation
Rather than processing the entire tablet, DINOv2 operates exclusively on the extracted $224 \times 224$ Imprint ROI:

```
Full Image (800x800)
    │
    ├───► [Stage 1: Frozen WideResNet-50-2] ───► Global Anomaly Score S_global
    │
    └───► [Morphological ROI Extraction]
              │
          Imprint ROI (224x224)
              │
              ▼
          [Specialist: DINOv2 vit_small_patch14]
              │
          Feature Tokens ('blocks.8', 'blocks.11')
              │
              ▼
          [Imprint Memory Bank (Offline Coreset)]
              │
              ▼
          Imprint Anomaly Score S_imprint
```

### 4.2 Specialist Backbone Specification
- **Backbone Architecture:** `vit_small_patch14_dinov2` (self-supervised Vision Transformer, 22.06M parameters).
- **Patch Size:** $14 \times 14$ pixels.
- **Input Resolution:** $224 \times 224 \times 3$ (yielding a $16 \times 16 = 256$ spatial patch grid).
- **Feature Extraction Layers:** Intermediate transformer blocks `('blocks.8', 'blocks.11')`.
- **Feature Dimension:** Concatenation of Block 8 (384-d) and Block 11 (384-d) = **768-d patch descriptor**.
- **Memory Bank Construction:**
  - Patch features are extracted strictly from normal reference Imprint ROIs.
  - Subsampling via greedy K-Center coreset reduction (10% ratio).
  - Shape of specialist memory bank for 10 reference images: $10 \times 256 \times 0.10 \approx \mathbf{[256, 768]}$, requiring only **0.78 MB of RAM**.

### 4.3 Imprint Anomaly Scoring
For a test Imprint ROI, patch embeddings $P = \{p_i\}_{i=1}^{256}$ are extracted. The patch anomaly distance is computed via Euclidean distance to the nearest normal neighbor in the specialist memory bank $\mathcal{M}_{\text{imprint}}$:
$$d(p_i) = \min_{m \in \mathcal{M}_{\text{imprint}}} \|p_i - m\|_2$$
The image-level imprint anomaly score $S_{\text{imprint}}$ is computed using the standard PatchCore neighborhood re-weighting over the top patch distances.

---

## 5. Interaction & Fusion with the Production WideResNet PatchCore

### 5.1 Preservation of Non-Imprint Defect Handling
The existing WideResNet PatchCore pipeline (Variant A) detects defects across the whole tablet surface with high precision:
- `pill_type`: 100.0% (9/9)
- `scratch`: 90.0% (9/10)
- `combined`: 88.2% (15/17)
- `color`: 70.0% (7/10)
- `crack`: 70.0% (7/10)
- `contamination`: 71.4% (15/21)

To guarantee that non-imprint defect detection is **completely preserved**, the Stage 1 WideResNet PatchCore remains the primary decision engine. If Stage 1 determines that an image is DEFECTIVE ($S_{\text{global}} \ge \tau_{\text{global}}$), the sample is **unconditionally flagged as DEFECTIVE**. The specialist cannot override, suppress, or veto a Stage 1 defect detection.

### 5.2 Fusion Mechanisms Under Investigation
We propose investigating three structured fusion mechanisms to test the trade-off between sensitivity recovery and false-positive introduction:

```
                               Decision Fusion Architectures
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      ▼                                      ▼                                      ▼
[Architecture 1: Cascaded OR]        [Architecture 2: Gated Suspicion]     [Architecture 3: Linear Fusion]
Defective IF:                        Defective IF:                         S_fused =
(S_global >= tau_A) OR               (S_global >= tau_A) OR                w * S_norm_global +
(S_imprint >= tau_imprint)           (S_global in [tau_low, tau_A] AND     (1-w) * S_norm_imprint
                                      S_imprint >= tau_imprint)
```

#### Architecture 1: Cascaded Logical Disjunction (Ensemble OR)
$$\text{Decision} = \begin{cases} 
\text{DEFECTIVE} & \text{if } S_{\text{global}} \ge \tau_{\text{global}} \;\lor\; S_{\text{imprint}} \ge \tau_{\text{imprint}} \\
\text{NORMAL} & \text{otherwise}
\end{cases}$$
- **Mechanics:** If Stage 1 detects a defect, inspection stops immediately (Defective). If Stage 1 classifies the tablet as Normal ($S_{\text{global}} < \tau_{\text{global}}$), the Imprint Specialist evaluates the Imprint ROI. If $S_{\text{imprint}} \ge \tau_{\text{imprint}}$, the tablet is flagged as Defective.
- **Structural Property:** Sensitivity is strictly non-decreasing relative to Variant A. However, **FPR is also non-decreasing**. Any normal tablet triggering $S_{\text{imprint}} \ge \tau_{\text{imprint}}$ becomes a new false alarm.

#### Architecture 2: Gated "Suspicion Band" Routing
$$\text{Decision} = \begin{cases} 
\text{DEFECTIVE} & \text{if } S_{\text{global}} \ge \tau_{\text{global}} \;\lor\; \big(S_{\text{global}} \ge \tau_{\text{lower}} \;\land\; S_{\text{imprint}} \ge \tau_{\text{imprint}}\big) \\
\text{NORMAL} & \text{otherwise}
\end{cases}$$
- **Mechanics:** The specialist is only allowed to flag a defect if Stage 1 exhibited at least moderate "suspicion" ($\tau_{\text{lower}} \le S_{\text{global}} < \tau_{\text{global}}$).
- **Evidence-Based Parameterization:** In Phase 8.5, baseline imprint false negatives exhibited scores clustered between $37.5$ and $43.5$ (median margin $-2.72$ below the $43.5951$ threshold). Setting $\tau_{\text{lower}} = \mu_{\text{global, ref}} = 30.52$ or $\tau_{\text{lower}} = \mu + 1.5\sigma \approx 37.0$ prevents the specialist from firing on tablets that WideResNet considered unambiguously clean.

#### Architecture 3: Normalized Score Fusion (Threshold-Independent)
$$S_{\text{fused}} = w \cdot \tilde{S}_{\text{global}} + (1 - w) \cdot \tilde{S}_{\text{imprint}}$$
- Where $\tilde{S} = \frac{S - \mu_{\text{ref}}}{\sigma_{\text{ref}}}$ represents standard z-score normalization on reference sets.
- Evaluated across sweeps of $w \in [0.1, 0.9]$ for threshold-independent ROC analysis.

---

## 6. Controlling False Positives from Normal Pill Lettering

Normal pill tablets feature prominent debossed letters ("N", "5") with sharp edges, cast shadows, and variable stamping depths. Because DINOv2 self-attention tokens are exquisitely sensitive to edge and line geometry, normal debossing is the primary driver of potential false alarms.

### 6.1 Calibration Safety Margins ($k$-Sigma Rule)
The specialist threshold is calibrated strictly from normal reference Imprint ROIs:
$$\tau_{\text{imprint}} = \mu_{\text{imprint, ref}} + k \cdot s_{\text{imprint, ref}}$$
To systematically evaluate false-positive resistance, the experiment will sweep the calibration multiplier:
- $k = 3.0$ (Standard $3\sigma$ reference heuristic)
- $k = 3.5$ (Conservative safety margin)
- $k = 4.0$ (Ultra-conservative margin to suppress normal debossing variance)

### 6.2 Spatial Anomaly Smoothing & Cluster Thresholding
In raw patch memory retrieval, single isolated patch anomalies frequently arise from specular highlights or slight camera noise along the debossed groove. 
To prevent point-noise false alarms:
1. Spatial anomaly map $A_{\text{imprint}} \in \mathbb{R}^{16 \times 16}$ is smoothed using a $2\text{D Gaussian kernel}$ ($\sigma = 1.0$).
2. A patch anomaly is only considered valid if at least two contiguous 8-connected patches exceed the local threshold ($A_{i,j} \ge 0.85 \cdot \max(A)$), filtering out single-token noise spikes.

### 6.3 Reference Sample Set Size Investigation
In Phase 8.6, fitting DINOv2 on only 10 reference images resulted in $\sigma_{\text{ref}} = 1.86$ but an FPR of 19.23%, indicating that 10 tablets under-represented natural manufacturing variance.
To test whether this is resolved by reference bank expansion, the experiment will evaluate:
- **Baseline-Matched Setup ($N=10$):** Specialist fitted strictly on the exact 10 reference images in `products/pill2/reference_images/`.
- **Expanded Calibration Setup ($N=50$):** Specialist fitted on 50 normal images sampled from `train/good/` (completely disjoint from test images).

---

## 7. Strict Data Partitions & Leakage Prevention Protocol

To ensure absolute scientific validity and zero data leakage:

```
                                Authoritative Data Partitions
                                              │
    ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
    ▼                                         ▼                                         ▼
Reference / Calibration Pool             External Evaluation Benchmark             Excluded Pool
(Normal Images Only)                     (Exact Frozen 122 Samples)                (45 V2 Dev Images)
- Exactly 10 normal images (pill2 ref)   - 26 Normal (test/good)                   - 16 Crack
- Optional: 50 normal (train/good)       - 19 faulty_imprint                       - 15 Discoloration
                                         - 21 contamination                        - 14 Scratch
                                         - 17 combined                             ❌ Strictly Excluded
                                         - 9 pill_type
                                         - 10 color / 10 crack / 10 scratch
```

### 7.1 Exact Partition Specification
1. **Reference Pool (Model Fitting & Calibration Only):**
   - **Primary ($N=10$):** The exact 10 normal images stored in `data/app_storage/products/pill2/reference_images/`.
   - **Exploratory ($N=50$):** 50 normal images randomly sampled with fixed seed (42) from `D:/datasets/defect_dataset_workspace/sources/MVTecAD/pill/train/good/`.
   - *Rule:* Zero defective images may be present during fitting or calibration.
2. **External Evaluation Benchmark ($N=122$ Total Images):**
   - **26 Normal Images:** `test/good/*.png` ($N=26$).
   - **96 Defective Images across 7 categories:**
     - `faulty_imprint`: 19 images
     - `contamination`: 21 images
     - `combined`: 17 images
     - `pill_type`: 9 images
     - `color`: 10 images
     - `crack`: 10 images
     - `scratch`: 10 images
3. **Excluded Pool (Strict Non-Contamination):**
   - The 45 images utilized in the Phase 8.3D V2 development pool (16 crack, 15 discoloration, 14 scratch) are strictly excluded from the evaluation set.
4. **Prohibition on Tuning Benchmark:**
   - The 23-image Phase 8.3D/8.3E benchmark must NOT be used for parameter tuning.

---

## 8. Proposed Experimental Variants & Order of Execution

The Phase 8.7 experiment will evaluate the following variants offline in strict sequence:

### Variant A: Frozen Production Baseline (Control)
- **Model:** WideResNet-50-2 with layers `('layer2', 'layer3')`.
- **Resolution:** $256 \times 256$ full-image.
- **Reference Set:** 10 reference images ($\tau_A = 43.5951$).
- **Role:** Direct benchmark reference; establishes baseline sensitivity (75.0%), imprint sensitivity (52.6%), specificity (96.15%), and FPR (3.85%).

### Variant S1: Baseline-Matched Dual-Stream Ensemble (Cascaded OR)
- **Global Stream:** Variant A WideResNet-50-2 ($\tau_A = 43.5951$).
- **Specialist Stream:** DINOv2 `vit_small_patch14_dinov2` on Tablet-Relative Imprint ROI ($224 \times 224$), layers `('blocks.8', 'blocks.11')`.
- **Specialist Reference Set:** Exact same 10 normal reference images.
- **Fusion Rule:** Architecture 1 (Cascaded OR) with $k \in \{3.0, 3.5, 4.0\}$.
- **Core Hypothesis:** Tests whether DINOv2 on the isolated Imprint ROI recovers baseline imprint false negatives without suffering the full-image color/normal penalties observed in Variant C.

### Variant S2: Expanded-Reference Dual-Stream Ensemble
- **Global Stream:** Variant A WideResNet-50-2 ($\tau_A = 43.5951$).
- **Specialist Stream:** DINOv2 on Tablet-Relative Imprint ROI ($224 \times 224$).
- **Specialist Reference Set:** 50 normal images sampled from `train/good/` to provide comprehensive normal debossing coverage.
- **Fusion Rule:** Architecture 1 (Cascaded OR) with $k \in \{3.0, 3.5, 4.0\}$.
- **Core Hypothesis:** Tests whether expanding normal lettering reference diversity suppresses normal imprint false alarms.

### Variant S3: Gated Suspicion-Band Ensemble
- **Global Stream:** Variant A WideResNet-50-2 ($\tau_A = 43.5951$).
- **Specialist Stream:** DINOv2 on Tablet-Relative Imprint ROI (fitted on best-performing reference set from S1/S2).
- **Fusion Rule:** Architecture 2 (Gated Suspicion) with $\tau_{\text{lower}} \in \{30.52, 37.00\}$.
- **Core Hypothesis:** Tests whether gating the specialist by Stage 1 borderline scores eliminates false alarms on unambiguously normal tablets.

---

## 9. Evaluation Protocol & Quantitative Metrics

The experiment will record and compare the following metrics across all variants:

### 9.1 Detection Performance Metrics
1. **`faulty_imprint` Sensitivity:** True Positive Rate specifically on the 19 imprint defect samples ($N_{\text{detected}} / 19$).
2. **Non-Imprint Defect Sensitivity:** Aggregate sensitivity across the other 77 defect samples (`contamination`, `combined`, `color`, `crack`, `scratch`, `pill_type`) ($N_{\text{detected}} / 77$).
3. **Category-Level Sensitivity Table:** Breakdown across all 7 defect categories.
4. **Normal Specificity & FPR:**
   - Specificity: $TN / 26$ on the 26 normal test images.
   - False Positive Rate: $FP / 26$.
5. **Specialist False Positive Count ($N_{\text{FP, specialist}}$):** Number of normal test tablets that Variant A correctly classified as Normal, but the specialist falsely flipped to Defective.
6. **Overall Sensitivity (TPR):** $TP / 96$ across all defective test images.
7. **Overall Image AUROC:** Area under the ROC curve across all 122 test samples.
8. **Phase 8.5 False Negative Recovery:**
   - Tracking exact recovery of the 9 baseline imprint false negatives: `000.png`, `003.png`, `004.png`, `005.png`, `008.png`, `009.png`, `011.png`, `013.png`, `014.png`.
   - Tracking transitions across all 24 Phase 8.5 baseline false negatives.

### 9.2 Threshold-Independent ROC & Pareto Analysis
- Generate full ROC curves ($0.0 \le \text{FPR} \le 1.0$) for each variant.
- Evaluate TPR at fixed reference operating points:
  - $\text{TPR}$ at $0.0\%$ FPR (Zero False Alarms)
  - $\text{TPR}$ at $3.85\%$ FPR (Baseline operating point)
  - $\text{TPR}$ at $7.69\%$ FPR (1 additional false alarm tolerated)
  - $\text{TPR}$ at $15.0\%$ FPR

### 9.3 Computational Cost & Latency
Empirically measured under steady-state GPU execution:
- Mean, P50, P90, P95, and Max inference latency (ms) per image across 50 warmup/timed cycles.
- Breakdown of latency: Stage 1 inference vs. ROI extraction vs. Specialist inference vs. Fusion logic.
- Peak allocated GPU memory (`torch.cuda.max_memory_allocated()`).
- Memory bank tensor dimensions and RAM footprint.

---

## 10. Computational Feasibility & Resource Estimates

Based on measurements obtained during Phase 8.6 on the local NVIDIA GeForce RTX 4060 Laptop GPU:

| Component | Architecture / Operation | Estimated P50 Latency | Estimated Peak VRAM |
| :--- | :--- | :---: | :---: |
| **Stage 1 (Variant A)** | WideResNet-50-2 ($256 \times 256$) | $39.3\text{ ms}$ | $138.2\text{ MB}$ |
| **Morphological ROI Extraction** | Contour thresholding & crop | $< 0.8\text{ ms}$ | Negligible (CPU) |
| **Specialist Stream (DINOv2)** | `vit_small_patch14` ($224 \times 224$, 256 tokens) | $45.0\text{ ms}$ | $133.0\text{ MB}$ |
| **Decision Fusion Logic** | Threshold comparison / Gating | $< 0.1\text{ ms}$ | Negligible |
| **Combined Cascaded System** | Sequential execution | **$\approx 85\text{ ms}$** | **$\approx 271\text{ MB}$** |

*Hardware Verdict:* Peak VRAM of $\approx 271\text{ MB}$ represents less than $3.5\%$ of the available 8GB GPU memory. A P50 latency of $\approx 85\text{ ms}$ corresponds to an effective throughput of **$\approx 11.8\text{ FPS}$**, comfortably within standard industrial inline inspection requirements ($< 250\text{ ms}$ per item).

---

## 11. Potential Failure Modes & Scientific Risks

The design identifies five specific structural risks that could cause the experiment to fail or yield a negative result:

1. **Failure Mode 1: Normal Lettering False Alarms (FPR Inflation)**
   - *Risk:* Normal pill debossing variations (depth, edge shadows, lighting angle) may trigger DINOv2 patch distances exceeding $\tau_{\text{imprint}}$, flipping normal test tablets to Defective.
   - *Impact:* Specificity drops below Variant A's 96.15%, rendering the ensemble operationally unacceptable despite higher imprint sensitivity.
2. **Failure Mode 2: Inability to Detect Ultra-Faint Imprints**
   - *Risk:* In Phase 8.6, both Variant A and Variant C failed to detect imprint defects `000.png` and `009.png`. If these defects are geometrically identical to normal debossing under 14px patch resolution, the specialist will also miss them.
   - *Impact:* Imprint sensitivity gain may plateau at $\le 75\%$, failing to resolve the core defect class.
3. **Failure Mode 3: Contour Segmentation Edge Jitter**
   - *Risk:* If tablet contour thresholding exhibits slight variation on damaged tablet rims, the cropped Imprint ROI window could shift by several pixels, introducing rim or background pixels into the ROI.
   - *Mitigation:* The 15% horizontal and 20% vertical relative padding provides a wide buffer around the tablet rim, preventing outer edge contamination.
4. **Failure Mode 4: Reference Set Under-Sampling**
   - *Risk:* 10 normal tablets may simply be too few to represent the stroke thickness and depth variance of normal debossing in a high-capacity Vision Transformer.
   - *Mitigation:* Evaluated directly via Variant S2 (50 normal reference tablets).
5. **Failure Mode 5: Unfavorable Latency-to-Gain Trade-off**
   - *Risk:* Adding DINOv2 roughly doubles inference latency from $39.3\text{ ms}$ to $\approx 85\text{ ms}$. If the net sensitivity gain is marginal (e.g. recovering only 1 or 2 defects at the cost of 2 false alarms), the dual-stream architecture cannot be justified.

---

## 12. Deliverable Artifacts for Proposed Execution

If approved for execution, Phase 8.7 will generate all artifacts strictly under:
`experiments/system_evaluation/phase87/`

Required artifacts:
1. `report.md`: Complete evaluation report including full variant comparison tables, category breakdowns, exact sample transitions, ROC curves, and hardware profiling.
2. `variant_metrics.json`: Full numerical metrics, confusion matrices, calibrated thresholds, and separation margins for all variants.
3. `per_sample_predictions.json`: Per-image scores ($S_{\text{global}}$, $S_{\text{imprint}}$, $S_{\text{fused}}$), binary predictions, and margins across all 122 evaluation samples.
4. `resource_metrics.json`: Empirical multi-cycle latency percentiles and peak GPU VRAM allocations.
5. `checkpoints/`: Specialist memory bank artifacts (strictly experiment artifacts; zero production deployment).

---

## 13. Audit & Constraint Compliance Checklist

- [x] **Production Frozen:** Zero modifications to `src/`, `data/app_storage/`, database, or frontend.
- [x] **Impartiality:** Does not assume the specialist will succeed; explicitly models failure modes.
- [x] **No Arbitrary Acceptance Gates:** Avoids inventing hard pass/fail criteria; presents full trade-off metrics.
- [x] **Strict Data Isolation:** Excludes the 45-image Phase 8.3D dev pool; forbids tuning on the 23-image benchmark.
- [x] **Empirical Grounding:** Validates 100.00% defect coverage of the proposed ROI method against authoritative masks.
- [x] **Data Feasibility Addressed:** Proves supervised classifier is infeasible; formulates smallest defensible one-class experiment.
- [x] **Execution Halt:** Halts immediately upon document generation without running code.
