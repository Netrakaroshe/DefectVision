# Phase 8.3E — Controlled Open-Set Rejection Benchmark Report

> [!IMPORTANT]
> **OFFLINE DIAGNOSTIC EXPERIMENT ONLY**: No production code, models, weights, checkpoints, calibration thresholds, datasets, database records, or frontend components were modified. All training data and evaluations were executed in an isolated experiment directory (`experiments/pill_open_set_benchmark`).

---

## 1. Dataset Verification

The evaluation benchmark comprised 89 total pill defect inspections:

| Benchmark Partition | Category / Class | Count ($N$) | Ground Truth Origin | Role |
| :--- | :--- | :---: | :--- | :--- |
| **Supported Known-Class Benchmark** | **`Scratch`** | 9 | Phase 8.3C/8.3D external untouched set | Evaluate retention & accuracy |
| | **`Crack`** | 7 | Phase 8.3C/8.3D external untouched set | Evaluate retention & accuracy |
| | **`Discoloration`** | 7 | Phase 8.3C/8.3D external untouched set | Evaluate retention & accuracy |
| **Subtotal Known-Class** | | **23** | | **True In-Distribution** |
| **Unsupported Open-Set Benchmark** | **`contamination`** | 21 | MVTecAD pill non-target defects | Evaluate open-set rejection |
| | **`faulty_imprint`** | 19 | MVTecAD pill non-target defects | Evaluate open-set rejection |
| | **`combined`** | 17 | MVTecAD pill non-target defects | Evaluate open-set rejection |
| | **`pill_type`** | 9 | MVTecAD pill non-target defects | Evaluate open-set rejection |
| **Subtotal Open-Set** | | **66** | | **True Out-of-Distribution** |
| **Total Evaluated** | | **89** | | |

---

## 2. Leakage Checks

- **Zero Overlap**: None of the 23 supported test images or 66 unsupported open-set images were used during the training or hyperparameter optimization of the Phase 8.3D ResNet-18 V2 checkpoint.
- **Reference Integrity**: The PatchCore memory bank was fitted solely on 10 normal defect-free calibration reference images (`0000_253.png` through `0009_262.png`). Zero defective images contributed to feature memory.
- **Isolation of Statistics**:
  - Class centroids for feature-distance rejection were computed strictly from the 31 training samples.
  - Temperature scaling parameter $T$ was fitted strictly on the 6 validation samples.
  - No evaluation sample from the 89 benchmark images was used to fit calibration statistics.

---

## 3. Methodology

We evaluated three standard post-hoc open-set rejection formulations on top of the frozen Phase 8.3D ROI-trained ResNet-18 classifier:

### Method A: Maximum Softmax Probability (MSP)
Standard baseline confidence:
$$\text{Score}_{\text{MSP}}(x) = \max_{c \in \{\text{Scratch}, \text{Crack}, \text{Discoloration}\}} P(c \mid x) = \max_c \frac{\exp(z_c)}{\sum_j \exp(z_j)}$$
Decision rule:
$$\hat{y} = \begin{cases} \arg\max_c P(c \mid x) & \text{if } \text{Score}_{\text{MSP}}(x) \ge \tau \\ \text{UNKNOWN} & \text{if } \text{Score}_{\text{MSP}}(x) < \tau \end{cases}$$

### Method B: Temperature-Scaled Softmax
Mitigates overconfident probability estimation by dividing logits by optimal temperature $T$:
$$P_{\text{temp}}(c \mid x) = \frac{\exp(z_c / T)}{\sum_j \exp(z_j / T)}$$
- Calibration procedure: $T$ was optimized on the 6 validation samples via negative log-likelihood (NLL) minimization.
- Optimal fitted temperature: **$T^* = 1.1414$** ($T > 1.0$ indicates mild overconfidence in the base model).

### Method C: Penultimate Feature-Distance Rejection
Extracts the 512-dimensional feature embedding $f(x) \in \mathbb{R}^{512}$ immediately prior to the linear classification head.
- Centroids $\mu_c = \frac{1}{N_c} \sum_{i \in c} f(x_i)$ were calculated for each class $c$ on the 31 training samples.
- Rejection metric is the cosine affinity to the closest class prototype:
  $$\text{Affinity}(x) = 1 - \min_{c} d_{\cos}(f(x), \mu_c) = \max_{c} \frac{f(x) \cdot \mu_c}{\|f(x)\| \|\mu_c\|}$$
- If $\text{Affinity}(x) < \tau_{\text{dist}}$, the sample is rejected as `UNKNOWN`.

---

## 4. Threshold-Selection Procedure

Rather than picking arbitrary cutoffs (e.g. 0.5 or 0.8), we performed a systematic parameter sweep across thresholds:
- MSP & Temperature-scaled: $\tau \in [0.40, 0.90]$ with step $0.05$.
- Feature Affinity: $\tau \in [0.50, 0.95]$ with step $0.05$.

At each candidate threshold, we recorded:
1. Supported defect acceptance rate ($\%$ of 23 kept).
2. Unsupported defect rejection rate ($\%$ of 66 flagged as UNKNOWN).
3. False UNKNOWN rate ($\%$ of true Scratch/Crack/Discoloration discarded).
4. Classification accuracy among accepted samples.

---

## 5. MSP Threshold Sweep Results

| Threshold ($\tau$) | Supported Accepted % | Supported Correct % | False UNKNOWN % | Unsupported Rejected % | Unsupported Accepted % | Accuracy Among Accepted % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.40** | 100.0% | 91.3% | 0.0% | 1.5% | 98.5% | 91.3% |
| **0.45** | 100.0% | 91.3% | 0.0% | 10.6% | 89.4% | 91.3% |
| **0.50** | **95.7%** | **87.0%** | **4.3%** | **19.7%** | **80.3%** | **90.9%** |
| **0.55** | 82.6% | 73.9% | 17.4% | 36.4% | 63.6% | 89.5% |
| **0.60** | 78.3% | 69.6% | 21.7% | 45.5% | 54.5% | 88.9% |
| **0.65** | 69.6% | 60.9% | 30.4% | 53.0% | 47.0% | 87.5% |
| **0.70** | 52.2% | 47.8% | 47.8% | 63.6% | 36.4% | 91.7% |
| **0.75** | 30.4% | 30.4% | 69.6% | 71.2% | 28.8% | 100.0% |
| **0.80** | 17.4% | 17.4% | 82.6% | 78.8% | 21.2% | 100.0% |
| **0.85** | 8.7% | 8.7% | 91.3% | 90.9% | 9.1% | 100.0% |

- **AUROC (Supported vs Unsupported)**: **0.6298**
- **AUPRC**: **0.3149** (vs random chance prevalence 0.258)

---

## 6. Temperature-Scaled Threshold Sweep Results ($T = 1.1414$)

| Threshold ($\tau$) | Supported Accepted % | Supported Correct % | False UNKNOWN % | Unsupported Rejected % | Unsupported Accepted % | Accuracy Among Accepted % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.35** | 100.0% | 91.3% | 0.0% | 0.0% | 100.0% | 91.3% |
| **0.40** | 100.0% | 91.3% | 0.0% | 4.5% | 95.5% | 91.3% |
| **0.45** | 100.0% | 91.3% | 0.0% | 13.6% | 86.4% | 91.3% |
| **0.50** | **95.7%** | **87.0%** | **4.3%** | **25.8%** | **74.2%** | **90.9%** |
| **0.55** | 78.3% | 69.6% | 21.7% | 43.9% | 56.1% | 88.9% |
| **0.60** | 65.2% | 56.5% | 34.8% | 54.5% | 45.5% | 86.7% |
| **0.65** | 47.8% | 43.5% | 52.2% | 65.2% | 34.8% | 90.9% |
| **0.70** | 30.4% | 30.4% | 69.6% | 72.7% | 27.3% | 100.0% |
| **0.75** | 13.0% | 13.0% | 87.0% | 83.3% | 16.7% | 100.0% |

- **AUROC (Supported vs Unsupported)**: **0.6318**
- **AUPRC**: **0.3152**

---

## 7. Feature-Distance Threshold Sweep Results (Cosine Affinity)

| Threshold ($\tau_{\text{dist}}$) | Supported Accepted % | Supported Correct % | False UNKNOWN % | Unsupported Rejected % | Unsupported Accepted % | Accuracy Among Accepted % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 100.0% | 91.3% | 0.0% | 0.0% | 100.0% | 91.3% |
| **0.60** | 100.0% | 91.3% | 0.0% | 0.0% | 100.0% | 91.3% |
| **0.65** | 100.0% | 91.3% | 0.0% | 1.5% | 98.5% | 91.3% |
| **0.70** | 100.0% | 91.3% | 0.0% | 7.6% | 92.4% | 91.3% |
| **0.75** | **95.7%** | **87.0%** | **4.3%** | **18.2%** | **81.8%** | **90.9%** |
| **0.80** | 60.9% | 52.2% | 39.1% | 57.6% | 42.4% | 85.7% |
| **0.85** | 30.4% | 26.1% | 69.6% | 89.4% | 10.6% | 85.7% |
| **0.90** | 4.3% | 4.3% | 95.7% | 98.5% | 1.5% | 100.0% |

- **AUROC (Supported vs Unsupported)**: **0.6785**
- **AUPRC**: **0.3471**

---

## 8. Per-Category UNKNOWN Rejection Breakdown

The 66 unsupported pill defects exhibited sharp behavioral differences depending on their physical morphology:

| Category | Count ($N$) | Mean MSP | Mean Cosine Affinity | Rejected at MSP $\tau=0.50$ | Rejected at MSP $\tau=0.60$ | Rejected at MSP $\tau=0.70$ | Rejection Difficulty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`contamination`** | 21 | **0.596** | 0.794 | **38.1% (8/21)** | 61.9% (13/21) | 76.2% (16/21) | **Moderate** (dark specks produce lower confidence) |
| **`combined`** | 17 | 0.665 | 0.793 | 11.8% (2/17) | 41.2% (7/17) | 58.8% (10/17) | **Hard** (diffuse mixed features) |
| **`pill_type`** | 9 | 0.693 | **0.725** | 11.1% (1/9) | 33.3% (3/9) | 55.6% (5/9) | **Moderate** (distinct shape, lower feature affinity) |
| **`faulty_imprint`** | 19 | **0.704** | **0.805** | **10.5% (2/19)** | 36.8% (7/19) | 57.9% (11/19) | **Severe Failure** (imprint strokes mimic Scratch) |

> [!WARNING]
> **Imprint Confounding**:
> `faulty_imprint` samples produce an average MSP of **0.704**, virtually indistinguishable from genuine known defects (**0.734**). The convolutional filters recognize the sharp stroke borders of the distorted debossed lettering as high-confidence scratches.

---

## 9. Confusion Matrices (At Calibrated Operating Points)

### Operating Point Comparison (Targeting $\ge 85\%$ Supported Retention)

#### A. MSP Gating ($\tau = 0.50$)
```
True \ Pred          Scratch    Crack    Discoloration    UNKNOWN (Rejected)
Scratch (9)             7         0            1                 1
Crack (7)               0         6            1                 0
Discoloration (7)       0         0            7                 0
combined (17)           4         7            4                 2
contamination (21)      6         7            0                 8
faulty_imprint (19)    10         6            1                 2
pill_type (9)           3         1            4                 1
```

#### B. Temperature-Scaled Gating ($T = 1.1414, \tau = 0.50$)
```
True \ Pred          Scratch    Crack    Discoloration    UNKNOWN (Rejected)
Scratch (9)             7         0            1                 1
Crack (7)               0         6            1                 0
Discoloration (7)       0         0            7                 0
combined (17)           2         7            4                 4
contamination (21)      5         7            0                 9
faulty_imprint (19)    10         6            0                 3
pill_type (9)           3         1            4                 1
```

#### C. Feature-Distance Gating (Cosine Affinity $\tau = 0.75$)
```
True \ Pred          Scratch    Crack    Discoloration    UNKNOWN (Rejected)
Scratch (9)             8         0            1                 0
Crack (7)               0         6            1                 0
Discoloration (7)       0         0            6                 1
combined (17)           3         7            4                 3
contamination (21)      7         8            4                 2
faulty_imprint (19)    10         7            2                 0
pill_type (9)           0         1            1                 7
```

---

## 10. Supported-Defect Retention Analysis

- **Baseline Performance (No Rejection)**: System C retained **100% of supported defects** with **91.30% accuracy (21 / 23 correct)**.
- **Impact of Rejection Gating**:
  - At $\tau = 0.50$: 1 supported defect (`Scratch 015.png`, confidence $0.493$) was discarded as UNKNOWN. Retention = **95.7% (22/23)**.
  - At $\tau = 0.60$: 5 supported defects discarded. Retention drops to **78.3% (18/23)**.
  - At $\tau = 0.70$: 11 supported defects discarded. Retention drops to **52.2% (12/23)**.
- **Trade-Off Verdict**: Any confidence threshold above $\tau = 0.50$ rapidly destroys legitimate defect classification, turning half of genuine scratches and cracks into false UNKNOWN rejections.

---

## 11. Unsupported-Defect Rejection Analysis

- **Evaluation Target Assessment**:
  - Target: $\ge 80\%$ unsupported defects rejected AND $\ge 90\%$ supported defects retained.
  - **Empirical Reality**:
    - When supported retention is $\ge 90\%$ ($\tau = 0.50$), unsupported rejection is **only $19.7\%$ to $25.8\%$**. Over $74\%$ of unsupported defects pass through and are misclassified into Scratch, Crack, or Discoloration.
    - To reject $\ge 80\%$ of unsupported defects ($\tau \ge 0.80$), supported retention collapses to **$17.4\%$** (over $82\%$ false UNKNOWN rate on real defects).
- **Simultaneous Target Feasibility**: **UNACHIEVABLE via post-hoc confidence or distance gating on this model.**

---

## 12. Best Method Comparison

| Method | Operating $\tau$ | Known Retention % | Known Correct % | Unsupported Rejection % | False UNKNOWN % | AUROC | AUPRC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A. MSP** | 0.500 | 95.7% | 87.0% | 19.7% | 4.3% | 0.6298 | 0.3149 | Baseline softmax confidence; high contamination leak. |
| **B. Temperature-Scaled** | 0.500 | 95.7% | 87.0% | **25.8%** | 4.3% | 0.6318 | 0.3152 | Best probability calibration ($T=1.1414$); +6.1% rejection gain. |
| **C. Feature Distance** | 0.750 | 95.7% | 87.0% | 18.2% | 4.3% | **0.6785** | **0.3471** | Highest separation overall; excellent on `pill_type` (77.8% rej). |

### Ranking:
1. **Method C (Feature Distance / Affinity)** achieved the highest separation (AUROC: **0.6785**, AUPRC: **0.3471**), excelling at detecting structural body abnormalities (`pill_type`).
2. **Method B (Temperature Scaling)** yielded the highest unsupported rejection ($25.8\%$) among probability-based methods at the 95% retention operating point.
3. However, **none of the three methods achieves the desired separation threshold**. AUROC values between $0.63$ and $0.68$ represent weak discrimination.

---

## 13. Limitations

1. **Overconfident Feature Encodings**: Standard cross-entropy training forces closed-world feature clustering. Non-target defects (like imprint flaws) fall directly into the feature subspaces of Scratch and Crack.
2. **Small Calibration Sample**: The temperature parameter was fitted on $N=6$ validation samples. While $T=1.1414$ is physically plausible, validation statistical power is low.
3. **Absence of Contrastive OOD Training**: The model was never trained with background rejection losses or Outlier Exposure.

---

## 14. Recommendation for Future Production Implementation

### Final Verdict:
> **Post-hoc open-set rejection is NOT ready for production implementation.**

### Rationale:
1. Setting a confidence threshold high enough to reject unsupported defects ($\ge 80\%$) discards $82.6\%$ of genuine scratches and cracks.
2. Setting a threshold low enough to preserve supported defects ($\ge 90\%$) permits $74.2\%$ of unsupported defects to be misclassified into the 3 classes.
3. Post-hoc thresholding on a standard closed-world classifier cannot compensate for the lack of open-set training signal.

### Recommended Next Experiment (Phase 8.3F):
**Contrastive / Multi-Task Open-Set Representation Learning**:
- Train the ROI feature extractor using **SupCon (Supervised Contrastive Loss) + Outlier Regularization**, utilizing the non-target pill defects as an explicit background/outlier class during representation learning, rather than applying a post-hoc threshold to a closed-world model.

---
*(Phase 8.3E Benchmark Complete — STOP)*
