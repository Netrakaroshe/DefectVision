# PHASE 8.3F — CONTROLLED OPEN-SET REPRESENTATION LEARNING EXPERIMENT REPORT

> **CLASSIFICATION**: OFFLINE ML DIAGNOSTIC EXPERIMENT ONLY  
> **STATUS**: PRODUCTION PIPELINE, CHECKPOINTS, CALIBRATION, API, AND FRONTEND REMAIN STRICTLY FROZEN.  
> **DATE**: 2026-10-05 (Audited & Corrected)  
> **DIRECTORY**: `experiments/pill_open_set_representation_v1/`

---

## 1. EXPERIMENT OVERVIEW & OBJECTIVE

In Phase 8.3E, post-hoc open-set rejection techniques (MSP, temperature scaling, feature-space cosine affinity) evaluated on the frozen Phase 8.3D V2 pill ROI classifier revealed a fundamental limitation: standard cross-entropy models assign arbitrarily high confidence to out-of-distribution/unsupported defects (mean MSP on `faulty_imprint` was 0.704 vs 0.734 on genuine defects). No post-hoc method achieved $\ge 90\%$ supported retention and $\ge 80\%$ unsupported rejection simultaneously.

Phase 8.3F investigated whether **representation-learning techniques** can improve this tradeoff:
1. **Variant A**: Frozen Phase 8.3D ROI classifier V2 baseline (standard cross-entropy on localized defect ROIs).
2. **Variant B**: Supervised Contrastive Learning (SupCon) on supported defect classes (`Scratch`, `Crack`, `Discoloration`).
3. **Variant C**: SupCon + normal-surface feature regularization (repelling normal non-defect pill surface patches from supported class centroids).

---

## 2. METHODOLOGY AUDIT: SAMPLES & LEAKAGE VERIFICATION

### Exact Dataset Counts
- **Total Authoritative Pill Records**: 77 supported defect records + 66 unsupported records + 267 normal calibration pool records.
- **Untouched External Supported Benchmark**: Exact Phase 8.3C/8.3D 23-defect set:
  - `Scratch`: 9
  - `Crack`: 7
  - `Discoloration`: 7
  - **Total**: 23 images (100% isolated, 0 training, 0 validation, 0 threshold fitting, 0 centroid fitting).
- **Available Supported Training/Validation Pool**: 45 records:
  - `Train`: 31 records (Scratch: 11, Crack: 10, Discoloration: 10)
  - `Val`: 6 records (Scratch: 2, Crack: 2, Discoloration: 2)
  - `Internal Test`: 8 held-out records (Scratch: 3, Crack: 3, Discoloration: 2)
- **Untouched Unsupported/Open-Set Evaluation Benchmark**: Exact Phase 8.3E 66-image evaluation set:
  - `contamination`: 21
  - `faulty_imprint`: 19
  - `combined`: 17
  - `pill_type`: 9
  - **Total**: 66 images (100% isolated).

### Audited Sample Identity for Variant C
- **Audit Finding**: In Variant C, training regularization used 15 normal pill reference surface crops sampled from the defect-free calibration pool (`original_label == 'good'`).
- **Correction of Terminology**:
  - These 15 crops are **normal pill surface / non-defect background patches**, NOT unsupported defect outliers.
  - Variant C tested **"SupCon + normal-surface feature regularization"**, NOT "unsupported-defect outlier regularization".
- **Leakage Audit**:
  - Were any of the 66 Phase 8.3E unsupported evaluation samples used during training? **NO (0 / 66). Verified 100% isolated.**
  - Were any of the 23 supported evaluation samples used during training, validation, early stopping, or centroid fitting? **NO (0 / 23). Verified 100% isolated.**
  - **Zero data leakage exists.**

---

## 3. AUDITED REPRESENTATION GEOMETRY METRICS

During the audit, a methodological inconsistency in representation-distance reporting was identified and resolved:
- In the initial draft, "intra-class distance" measured **pairwise sample-to-sample distance** ($1-\cos(x_i, x_j)$), while "inter-class distance" measured **centroid-to-centroid distance** ($1-\cos(\mathbf{c}_1, \mathbf{c}_2)$).
- Because a class centroid averages out individual sample noise, centroid-to-centroid distances ($\sim 0.13$) are naturally smaller than raw sample-to-sample distances ($\sim 0.30 - 0.36$), creating a misleading appearance that samples were closer to other classes.

### Methodologically Consistent Metric Breakdown

| Metric Level | Metric Description | Variant A (V2 Baseline) | Variant B (SupCon Only) | Variant C (SupCon + Normal Reg) |
| :--- | :--- | :---: | :---: | :---: |
| **Pairwise Sample-to-Sample** | **Mean Intra-Class Distance** ($1-\cos$) | **0.3079** | **0.3006** | 0.3318 |
| | - Scratch | 0.3135 | 0.3227 | 0.3514 |
| | - Crack | 0.3135 | 0.3323 | 0.3221 |
| | - Discoloration | 0.2969 | 0.2468 | 0.3221 |
| | **Mean Inter-Class Distance** ($1-\cos$) | **0.3663** | **0.3635** | 0.3662 |
| **Sample-to-Centroid** | **Mean Sample-to-Own-Centroid** | **0.1440** | **0.1406** | 0.1564 |
| | **Mean Sample-to-Other-Centroid** | **0.2595** | **0.2587** | 0.2494 |
| **Centroid-to-Centroid** | **Mean Centroid Inter-Class Distance** | **0.1348** | **0.1369** | 0.1102 |
| | - Scratch vs Crack Centroid Distance | 0.0949 | 0.0712 | 0.0779 |
| | - Scratch vs Discoloration Centroid Distance | 0.1689 | 0.1865 | 0.1147 |
| | - Crack vs Discoloration Centroid Distance | 0.1406 | 0.1530 | 0.1381 |
| **Open-Set Outliers** | **Mean Outlier Distance to Nearest Centroid** | 0.2198 | 0.2148 | 0.2285 |
| | - `contamination` to Nearest Centroid | 0.2039 | 0.1879 | 0.2126 |
| | - `faulty_imprint` to Nearest Centroid | 0.1928 | 0.1828 | 0.2428 |
| | - `combined` to Nearest Centroid | 0.2064 | 0.1869 | 0.2044 |
| | - `pill_type` to Nearest Centroid | 0.2761 | 0.3016 | 0.2541 |

### Key Geometric Findings:
1. **Intra vs Inter Consistency**: At the sample-to-sample level, **inter-class distance (0.3663) is strictly greater than intra-class distance (0.3079)**. Similarly, sample-to-own centroid distance (0.1440) is much smaller than sample-to-other centroid distance (0.2595).
2. **Scratch vs Crack Contraction**: In Variant B, SupCon pulled Discoloration into a tighter cluster (intra-class distance dropped from 0.2969 to 0.2468), but **compressed the distance between Scratch and Crack centroids** from 0.0949 to 0.0712. Because fine cracks and linear scratches share identical edge filter activations, contrastive learning without fine-grained spatial labels pulled both defect types closer together.
3. **Outlier Entanglement**: The distance from unsupported defects (`contamination`: 0.1879, `faulty_imprint`: 0.1828, `combined`: 0.1869) to nearest supported defect centroids in Variant B is **smaller than the sample-to-other-centroid distance (0.2587)** and comparable to sample-to-own-centroid spread. In feature space, unsupported defects lie directly inside the convex hull of supported defects.

---

## 4. INDEPENDENTLY VERIFIED BENCHMARK PERFORMANCE

| Evaluation Benchmark | Variant A (V2 Baseline) | Variant B (SupCon Only) | Variant C (SupCon + Normal Reg) |
| :--- | :---: | :---: | :---: |
| **External Supported Accuracy (N=23)** | **91.30% (21/23)** | 78.26% (18/23) | 34.78% (8/23) |
| **External Supported Macro F1** | **0.9131** | 0.7731 | 0.3161 |
| - Scratch F1 | **0.9412** | 0.7778 | 0.2500 |
| - Crack F1 | **0.9231** | 0.6667 | 0.4762 |
| - Discoloration F1 | **0.8750** | **0.8750** | 0.2222 |
| **Internal Test Accuracy (N=8)** | 0.6250 | 0.5000 | 0.5000 |
| **Internal Test Macro F1 (N=8)** | 0.6571 | 0.5238 | 0.4905 |
| **Open-Set AUROC (MSP)** | **0.6298** | 0.6074 | 0.4980 |
| **Open-Set AUROC (Feature Cosine Affinity)** | 0.6785 | **0.7088** | 0.6173 |
| **Open-Set AUPRC (Feature Cosine Affinity)** | 0.3471 | **0.5016** | 0.3501 |

---

## 5. CONFUSION MATRICES ON EXTERNAL 23-IMAGE BENCHMARK

### Variant A (Phase 8.3D Baseline V2) — 21/23 Correct (91.30%)
```
               Predicted Scratch   Predicted Crack   Predicted Discoloration
True Scratch          8                   0                     1
True Crack            0                   6                     1
True Discoloration    0                   0                     7
```
- Only 2 errors: Scratch `016.png` and Crack `020.png` misclassified as Discoloration.

### Variant B (SupCon) — 18/23 Correct (78.26%)
```
               Predicted Scratch   Predicted Crack   Predicted Discoloration
True Scratch          7                   1                     1
True Crack            2                   4                     1
True Discoloration    0                   0                     7
```
- 5 errors: Scratch `016.png` (Discoloration), Scratch `018.png` (Crack), Crack `015.png` (Scratch), Crack `017.png` (Scratch), Crack `020.png` (Discoloration).

### Variant C (SupCon + Normal Surface Regularization) — 8/23 Correct (34.78%)
```
               Predicted Scratch   Predicted Crack   Predicted Discoloration
True Scratch          2                   6                     1
True Crack            2                   5                     0
True Discoloration    3                   3                     1
```
- Severe collapse: Normal-surface repulsion forces pushed the representations arbitrarily, collapsing the classifier head on the small 31-sample dataset.

---

## 6. OPEN-SET REJECTION & OPERATING POINTS ANALYSIS

Sweeping feature cosine affinity threshold $\tau$ across target supported retention tiers:

| Target Retention | Variant | Threshold $\tau$ | Supported Retention | Supported Correct | Unsupported Rejection | False UNKNOWN Rate | Accuracy Among Retained |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\ge$ 90%** | **Variant A** | 0.75 | 95.65% | 86.96% | 18.18% | 4.35% | 90.91% |
| | **Variant B** | 0.75 | 100.00% | 78.26% | 15.15% | 0.00% | 78.26% |
| | **Variant C** | 0.70 | 91.30% | 30.43% | 13.64% | 8.70% | 33.33% |
| **$\ge$ 85%** | **Variant A** | 0.75 | 95.65% | 86.96% | 18.18% | 4.35% | 90.91% |
| | **Variant B** | 0.75 | 100.00% | 78.26% | 15.15% | 0.00% | 78.26% |
| | **Variant C** | 0.70 | 91.30% | 30.43% | 13.64% | 8.70% | 33.33% |
| **$\ge$ 80%** | **Variant A** | 0.75 | 95.65% | 86.96% | 18.18% | 4.35% | 90.91% |
| | **Variant B** | 0.75 | 100.00% | 78.26% | 15.15% | 0.00% | 78.26% |
| | **Variant C** | 0.75 | 82.61% | 26.09% | 27.27% | 17.39% | 31.58% |

### Category-Specific Rejection Rates at Operating Point ($\tau=0.75$):
- **Variant A ($\tau=0.75$)**: `contamination`: 9.5%, `faulty_imprint`: 0.0%, `combined`: 17.6%, `pill_type`: 77.8% (Total Rejection: 18.18%).
- **Variant B ($\tau=0.75$)**: `contamination`: 4.8%, `faulty_imprint`: 0.0%, `combined`: 0.0%, `pill_type`: 100.0% (Total Rejection: 15.15%).
- **Variant B ($\tau=0.80$)** (Retention drops to 78.26%): `contamination`: 42.9%, `faulty_imprint`: 21.1%, `combined`: 47.1%, `pill_type`: 100.0% (Total Rejection: 45.45%).

Neither representation-learning variant achieved $\ge 90\%$ retention with $\ge 80\%$ rejection.

---

## 7. DETAILED FAILURE ANALYSIS

1. **Why `faulty_imprint` and `contamination` fool the representations**:
   - `faulty_imprint` consists of missing or distorted letter stampings on the pill surface. In high-resolution localized ROIs, broken stampings present sharp contrast discontinuities that ResNet-18 filters interpret as Cracks or Scratches with high confidence (7 of 19 `faulty_imprint` samples produced $\text{MSP} \ge 0.80$ in Variant B).
   - `contamination` consists of foreign dark specks or smudges that project directly onto `Discoloration` (low-contrast smudges) or `Scratch` (elongated dark specks).
2. **Why Normal-Surface Regularization (Variant C) Failed**:
   - Regularizing with non-defect normal patches repels the feature space away from smooth pill texture, but unsupported defects are *not* smooth pill texture; they are anomalies with edges and color shifts. The artificial repulsion gradient destabilized the linear classifier head on the 31 training samples, dropping accuracy from 91.3% to 34.8%.

---

## 8. ANSWERS TO MANDATORY EXPERIMENT QUESTIONS

1. **Does the new representation preserve supported Scratch/Crack/Discoloration classification accuracy?**  
   **NO**. Variant B dropped accuracy from 91.30% to 78.26% (Macro F1 dropped from 0.9131 to 0.7731). Variant C collapsed to 34.78%.
2. **Does it better separate unsupported/open-set pill defects?**  
   **MARGINALLY AT BEST**. Feature affinity AUROC improved from 0.6785 (Variant A) to 0.7088 (Variant B), but this improvement was entirely driven by the non-pill `pill_type` distractor (100% rejection). Rejection of authentic pill defect categories (`faulty_imprint`, `contamination`, `combined`) remained below 5% at $\ge 90\%$ retention.
3. **Does it outperform the frozen Phase 8.3D V2 baseline?**  
   **NO**. The frozen Phase 8.3D V2 baseline remains strictly superior.
4. **Does it preserve performance on the untouched external 23-image benchmark?**  
   **NO**. Variant B made 5 errors on external cracks and scratches (versus only 2 in Variant A).
5. **Is the improvement genuine or caused by leakage/shortcut tuning?**  
   There is zero leakage, but the slight AUROC increase in Variant B is not an improvement in semantic open-set rejection; it reflects sensitivity to global pill shape (`pill_type`) while failing on within-pill defect semantics.

---

## 9. DECISION & CONCLUSION

> **FORMAL CONCLUSION**:  
> **Phase 8.3F does not provide sufficient evidence for production adoption.**

### Production Safety Confirmation
- Production inference pipeline remains **100% FROZEN**:
  `PatchCore -> DEFECTIVE -> Deterministic ROI -> Frozen Phase 8.3D V2 ROI Classifier -> Scratch / Crack / Discoloration`.
- No files in `src/defectvision/`, no production models, no database records, no calibration routines, and no frontend endpoints were altered.
- All experimental models, manifests, and evaluation artifacts reside strictly in `experiments/pill_open_set_representation_v1/`.
