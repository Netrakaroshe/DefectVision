# Dent Pilot Step 3B: Final Pre-Training Validation Report

**Project:** `DefectVision`  
**Phase:** Offline Dent Classification Investigation — Step 3B: Final Pre-Training Validation  
**Target File:** `D:\DefectVision\experiments\dent_classification\background_audit\pretraining_validation_report.md`  
**Audited Artifacts:**
- `D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl`
- `D:\DefectVision\experiments\dent_classification\background_audit\manifests\background_candidates_manifest.jsonl`
- `D:\DefectVision\experiments\dent_classification\background_audit\background_audit_report.md`
- Contact sheets: `background_train_contact_sheet.png`, `background_val_contact_sheet.png`, `background_test_contact_sheet.png`
**Date:** October 19, 2026  
**Auditor / Engineer:** DefectVision AI Pair Programmer  

---

## 1. Executive Summary & Final Verdict

This audit performs a pre-training validation of the candidate background crops against the baseline Dent pilot dataset. All 1,242 negative candidates were programmatically cross-examined against source images, raw YOLO annotations, SHA-256 content hashes, and positive Dent crops.

### Key Audit Findings
* **Pixel Coordinate Fidelity:** **100% Pass** (0 pixel mismatches across all 1,242 crops against raw source image arrays).
* **Buffer & Exclusion Integrity:** **100% Pass** (0 overlaps with annotated defects or their 20-pixel safety buffer; minimum observed distance is $20.04\text{ px}$).
* **Split & Hash Isolation:** **100% Pass** (0 split conflicts; 0 duplicate hashes crossing splits; all crops from any given image strictly share that image's partition).
* **Crop Independence:** **100% Pass** (0 identical crops within background candidates; 0 overlaps/duplicates between background and Dent crops).
* **Sampling Alignment:** Crop size and aspect-ratio distributions match the Dent crops ($0.160 \le \text{AR} \le 8.130$, mean $\text{AR} = 1.228$).
* **Physical Signature:** Mean overall luminance is nearly identical ($122.3$ vs $124.1$), while Dent crops exhibit higher internal pixel standard deviation ($11.3$ vs $7.8$) due to 3D shadow/highlight relief.
* **Manual Review Status:** Inherent to supervised YOLO datasets, unannotated regions cannot be guaranteed defect-free; manual review remains necessary.

---

## 2. Section 1: Candidate Extraction Verification

Every candidate background crop was audited against disk records and raw source images:

| Verification Check | Target Requirement | Measured Result | Status |
| :--- | :--- | :---: | :---: |
| **Pixel Match vs Source Image** | Bit-for-bit exact slice match at `patch_pixels` | **0 mismatches / 1,242** | **PASS** |
| **Safety Buffer Compliance** | Distance to any annotated box $\ge 20.0\text{ px}$ | **0 violations / 1,242** | **PASS** |
| **Minimum Observed Defect Distance** | Strictly $\ge 20.0\text{ px}$ | **$20.04\text{ px}$** | **PASS** |
| **Split Consistency vs Dent Manifest** | Source image split matches Dent manifest split | **0 conflicts / 1,242** | **PASS** |
| **Hash Consistency vs Dent Manifest** | Source image SHA-256 matches Dent manifest | **0 conflicts / 1,242** | **PASS** |
| **Duplicate Hashes Crossing Splits** | Identical images assigned to multiple splits | **0 occurrences** | **PASS** |
| **Background Internal Duplicates** | Duplicate identical background crop files | **0 clusters** (1,242 unique) | **PASS** |
| **Positive vs Negative Duplicates** | Identical crops between Dent and Background | **0 intersections** | **PASS** |
| **Image-Level Split Inheritance** | Background crop split $\equiv$ Source image split | **100% identical** | **PASS** |

---

## 3. Section 2: Label and Sampling Validity

### 2.1 Source Image Origin Breakdown
* **From Exact Same Source Image as Matched Dent Crop:** **864 candidates (69.57%)**
* **From Different Image in Same Split:** **378 candidates (30.43%)**
* **From Different Split:** **0 candidates (0.00%)**

> **Rationale:** For 378 instances, large multi-defect bounding boxes in the origin image left insufficient contiguous defect-free space to fit the target dimensions outside the 20px exclusion zone. In those cases, candidate patches were drawn from another image strictly within the **same split partition**, preserving 100% split integrity.

### 2.2 Unique Contributing Source Images per Split

| Split Partition | Dent Source Images | Background Source Images | Shared Source Images |
| :--- | :---: | :---: | :---: |
| **Train** | 420 | 358 | 358 |
| **Validation** | 90 | 75 | 75 |
| **Test** | 90 | 81 | 81 |
| **Total** | **600** | **514** | **514** |

All 358 Train background source images belong exclusively to Train; all 75 Validation background source images belong exclusively to Validation; all 81 Test background source images belong exclusively to Test.

### 2.3 Systematic Feature Comparison (Dent vs Background)

Quantitative comparison between the 1,242 Dent crops and 1,242 Background candidates:

```text
========================================================================================
Feature Dimension             Positive (Dent)             Negative (Background)
========================================================================================
Crop Area (px^2)              Mean: 15,806.0              Mean: 15,806.0
                              Median: 13,407.0            Median: 13,407.0
                              Min: 1,430 | Max: 67,230    Min: 1,430 | Max: 67,230
----------------------------------------------------------------------------------------
Aspect Ratio (W / H)          Mean: 1.228                 Mean: 1.228
                              Median: 1.006               Median: 1.006
                              Min: 0.160 | Max: 8.130     Min: 0.160 | Max: 8.130
----------------------------------------------------------------------------------------
Center X Coordinate (px)      Mean: 220.8 (std: 119.4)    Mean: 182.7 (std: 84.2)
                              Median: 206.5               Median: 167.2
----------------------------------------------------------------------------------------
Center Y Coordinate (px)      Mean: 210.2 (std: 118.3)    Mean: 197.0 (std: 89.7)
                              Median: 201.8               Median: 202.5
----------------------------------------------------------------------------------------
Mean Pixel Intensity [0-255]  Mean: 122.3 (std: 12.3)     Mean: 124.1 (std: 17.4)
                              Median: 119.3               Median: 120.2
----------------------------------------------------------------------------------------
Internal Texture Std [0-255]  Mean: 11.3 (std: 4.9)       Mean: 7.8 (std: 6.5)
                              Median: 12.3                Median: 7.1
========================================================================================
```

#### Observations on Systematic Differences:
1. **Area & Aspect Ratio (No Confounding):** Distributions are identical by design. A classifier cannot discriminate between classes based on crop scale or aspect ratio.
2. **Spatial Location:** Background patches have slightly smaller standard deviation around image center because defects frequently occur along the rolling strip margins, leaving the complementary space slightly offset. However, both distributions span the full $400 \times 400$ spatial frame.
3. **Global Illumination (No Confounding):** Mean pixel intensity is nearly identical ($122.3$ vs $124.1$), confirming that the background candidates do not suffer from systemic under- or over-exposure relative to defect regions.
4. **Internal Texture Variance (Expected Physical Delta):** Dent crops exhibit significantly higher internal pixel standard deviation ($11.3$ vs $7.8$). This directly reflects the physical 3D indentation morphology (paired highlight and shadow borders along pressed edges) contrasted against smooth rolling steel grain.

### 2.4 Defect-Free Caveat
> [!WARNING]
> **Open-World Dataset Limitation:** Raw YOLO annotations indicate regions where human annotators marked conspicuous defects; **they cannot mathematically guarantee that unannotated regions are 100% defect-free**. Faint roll marks, micro-surface abrasions, or subtle rolling textures falling below annotator sensitivity may exist in unannotated regions. Therefore, manual visual verification of the contact sheets remains mandatory prior to production training.

---

## 3. Section 3: Evaluation Design & Leakage Prevention

### 3.1 Recommended Training & Evaluation Protocol
To guarantee scientifically valid evaluation, the following constraints must be enforced during model training:

1. **Strict Source-Image & Hash Grouping:**
   * All crops (positive or negative) originating from source image $S$ (and any duplicate image sharing $S$'s SHA-256 hash) must remain strictly within one split partition.
   * **Test Partition Quarantine:** The test partition ($N=188$ Dents, $N=188$ Backgrounds) must be evaluated exactly once after all architecture and hyperparameter selections are finalized.
2. **Aspect-Ratio-Preserving Pre-Processing:**
   * Because aspect ratios vary from $0.160$ to $8.130$, square resizing ($224 \times 224$) without aspect-ratio preservation will induce severe geometric distortion.
   * Resizing must utilize **letterbox padding** (preserving aspect ratio with constant or reflection padding) or dynamic RoI pooling.

### 3.2 Intra-Image Positive/Negative Limitation
* **Observed Design Reality:** In 69.6% of negative candidates, positive and negative crops originate from the same physical steel plate image.
* **Resulting Limitation:** Within the training partition, the model is exposed to positive and negative patches that share the same macro-level strip lighting, roller curvature gradient, and grain structure. 
* **Mitigating Safeguard:** Because the **Test partition consists entirely of disjoint source images and disjoint hashes**, test evaluation directly tests the model's ability to classify unseen steel plates without relying on image-specific lighting signatures.

### 3.3 Domain Scope Limitation
* **Material Boundary:** This evaluation is strictly an offline pilot on **cold-rolled steel surfaces under industrial directional illumination**.
* **Non-Generalizability:** It does **not** establish general cross-material Dent detection and must not be characterized as applicable to blow-molded plastic bottles, pharmaceutical powder compacts, or ceramic tiles without separate cross-domain transfer benchmarks.

---

## 4. Verification Verdict

Based on 100% compliance across all data integrity, coordinate slice, safety buffer, and split isolation checks:

```text
========================================================================================
FINAL VERDICT:
READY FOR MANUAL NEGATIVE REVIEW
========================================================================================
```

*(No models were trained, no manifests altered, and no production code modified.)*
