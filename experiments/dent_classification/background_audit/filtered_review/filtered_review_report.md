# Dent Pilot Step 3D: Candidate Filtering & Human Review Preparation Report

**Project:** `DefectVision`  
**Phase:** Offline Dent Classification Investigation — Step 3D: Objective Filtering & Human Review Preparation  
**Target File:** `D:\DefectVision\experiments\dent_classification\background_audit\filtered_review\filtered_review_report.md`  
**Output Location:** `D:\DefectVision\experiments\dent_classification\background_audit\filtered_review\`  
**Date:** October 20, 2026  
**Auditor / Engineer:** DefectVision AI Pair Programmer  

---

## 1. Executive Summary & Objective

In accordance with **Step 3D**, an objective, rule-based filtering protocol was established to purge contaminated and boundary-occluded negative steel patches from the candidate background pool. A structured review set was produced to prepare for manual human review prior to any model training.

> [!IMPORTANT]
> **Explicit Review-Preparation Disclaimer:**
> **This artifact represents a review-preparation step, NOT a completed human review.** No automated heuristic proves that a patch is defect-free. All decisions in the triage template are left strictly blank for independent human evaluation. No models were trained, and no production assets were altered.

### Key Outcomes
* **Objective Filtering:** Evaluated all **1,242 raw background candidates**. Excluded **619 problematic candidates** based strictly on physical coordinate margins and unlit conveyor pixel detection.
* **Filter Rule Compliance:** Per user instructions, **no candidates were automatically excluded based on high contrast or high Laplacian variance**, ensuring natural specular roller lighting and grain textures remain available for human evaluation.
* **Passed Candidate Pool:** **623 defensible candidates** passed all objective filters (**432 Train / 85 Val / 106 Test**).
* **Sampled Review Set:** Extracted **285 candidates** across three split-segregated contact sheets:
  * **Train:** **100 candidates** randomly sampled from 432 available (`seed = 20261020`).
  * **Validation:** **85 candidates** (all available, as $85 \le 100$).
  * **Test:** **100 candidates** randomly sampled from 106 available (`seed = 20261020`).
* **Blank Decision CSV Template:** Generated `blank_reviewer_decisions.csv` containing all 285 review set items with blank `reviewer_decision` (`clean`, `defect_suspected`, `uncertain`) and `reviewer_notes` columns.
* **Duplicate Integrity:** Exact duplicate checks (SHA-256) and near-duplicate checks (mean pixel L1 diff $< 1.0$) yielded **0 duplicates** within the passed candidate pool.

---

## 2. Objective Filtering Rules & Rationales

The following four objective criteria were applied sequentially without subjective metric gating:

```text
==================================================================================================
Filter Rule                   Definition / Threshold             Physical Rationale
==================================================================================================
1. Outer Image Margin         x1 < 25 or y1 < 25 or              Eliminates the outer border of the
   Intersection               x2 > 375 or y2 > 375 (px)          400x400 image where the steel sheet
                                                                 terminates on conveyor rollers.
--------------------------------------------------------------------------------------------------
2. Unlit Conveyor / Black     Minimum pixel intensity < 30       Purges patches capturing black unlit
   Strip Edge Detection       (on 0-255 scale)                   conveyor background or strip drop-off.
--------------------------------------------------------------------------------------------------
3. Defect Buffer Violation    Minimum Euclidean distance to      Guarantees no spatial overlap with the
                              any annotated defect < 20.0 px     20px safety margin around true defects.
--------------------------------------------------------------------------------------------------
4. Exact / Near Duplicates    SHA-256 match or pairwise          Prevents identical image content from
                              mean absolute diff < 1.0 px        redundantly occupying the negative pool.
==================================================================================================
```

> [!NOTE]
> **Adherence to Instruction:** As explicitly mandated, **candidates were NOT filtered by Laplacian edge variance or local contrast range**. Highly contrasting patches from central cylindrical roller curvature remain in the review set for human inspection.

---

## 3. Quantitative Census: Counts Before and After Filtering

### Breakdown by Dataset Partition

| Processing Stage | Train | Validation | Test | Total Candidates |
| :--- | :---: | :---: | :---: | :---: |
| **1. Raw Candidates (Step 3A Pool)** | **866** | **188** | **188** | **1,242** |
| *Exclusion 1: Outer 25px Margin Intersection* | -426 | -103 | -82 | **-611** |
| *Candidates Remaining after Margin Filter* | 440 | 85 | 106 | **631** |
| *Exclusion 2: Defect Buffer Violation ($<20\text{ px}$)* | -0 | -0 | -0 | **-0** |
| *Exclusion 3: Unlit Conveyor ($I_{\min} < 30$)* | -8 | -0 | -0 | **-8** |
| *Exclusion 4: Exact / Near Duplicate Patches* | -0 | -0 | -0 | **-0** |
| **2. Passed Defensible Candidate Pool** | **432** | **85** | **106** | **623** |
| **3. Sampled Review Set for Human Triage** | **100** | **85** | **100** | **285** |

### Split-by-Split Yield Rates
* **Train Split:** 432 passed out of 866 (49.88% pass rate).
* **Validation Split:** 85 passed out of 188 (45.21% pass rate).
* **Test Split:** 106 passed out of 188 (56.38% pass rate).
* **Overall Pass Rate:** 623 passed out of 1,242 (50.16% pass rate).

---

## 4. Duplicate Check Methodology & Results

1. **Exact File Duplication (SHA-256):**
   * Cryptographic SHA-256 hashes were computed for all 623 passed candidate PNG crops.
   * **Result:** **0 duplicate file hashes** (623 unique digests out of 623 files).
2. **Intra-Image Spatial Overlap:**
   * For source images contributing multiple candidates, bounding box coordinates were tested for intersection ($\text{IoU} > 0$).
   * **Result:** **0 overlapping patches** on the same source image.
3. **Near-Duplicate Pixel Content (Pairwise L1 Pixel Delta):**
   * Candidates sharing matching dimensions were evaluated for pairwise pixel identity:
     $$\text{Diff}(A, B) = \frac{1}{W \times H} \sum_{x, y} |A(x, y) - B(x, y)|$$
   * Threshold for near-duplication: $\text{Diff} < 1.0\text{ px}$.
   * **Result:** **0 near-duplicate candidate pairs** found.

---

## 5. Review Artifacts Created

All artifacts are isolated under `D:\DefectVision\experiments\dent_classification\background_audit\filtered_review\`:

```text
D:\DefectVision\experiments\dent_classification\background_audit\filtered_review\
├── contact_sheets\
│   ├── filtered_review_train_100.png          # 10x10 grid: 100 Train candidates with coordinates & metrics
│   ├── filtered_review_val_85.png             # 10x9 grid: 85 Val candidates (all remaining in split)
│   └── filtered_review_test_100.png           # 10x10 grid: 100 Test candidates with coordinates & metrics
├── manifests\
│   ├── filtered_background_candidates.jsonl   # Manifest of all 623 passed candidates
│   └── filtered_review_sample_manifest.jsonl  # Manifest of the 285 sampled review set candidates
├── blank_reviewer_decisions.csv               # Blank decision template for manual human triage
└── filtered_review_report.md                  # This preparation report
```

### 5.1 Contact Sheet Specifications
* **Train Sheet ($N=100$):** [filtered_review_train_100.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/filtered_review/contact_sheets/filtered_review_train_100.png)
* **Validation Sheet ($N=85$):** [filtered_review_val_85.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/filtered_review/contact_sheets/filtered_review_val_85.png)
* **Test Sheet ($N=100$):** [filtered_review_test_100.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/filtered_review/contact_sheets/filtered_review_test_100.png)
* **Tile Content:**
  * Raw crop image rendered preserving aspect ratio without stretching.
  * Source image stem (e.g. `dents100`).
  * Source pixel coordinates $[x_1, y_1, x_2, y_2]$ (e.g. `[74, 87, 205, 202]`).
  * Crop dimensions $w \times h$ (e.g. `131x115`).
  * Minimum distance to nearest defect: $d \ge \text{XX.X px}$.

### 5.2 Blank Decision CSV Schema (`blank_reviewer_decisions.csv`)
Pre-populated with 285 rows. As required, reviewer fields are **strictly empty**:
* `candidate_id`: ID of background patch (e.g., `bg_dents100_p0005`)
* `split`: `train`, `val`, or `test`
* `reviewer_decision`: **` ` (STRICTLY BLANK — to be completed by human reviewer: `clean`, `defect_suspected`, or `uncertain`)**
* `reviewer_notes`: **` ` (STRICTLY BLANK — for reviewer comments)**
* `source_image_stem`: Image name for reverse lookups
* `patch_x1, patch_y1, patch_x2, patch_y2`: Source coordinates
* `patch_width, patch_height, aspect_ratio`: Dimensional metrics
* `min_defect_dist_px`: Nearest defect distance
* `crop_path`: File path to crop PNG

---

## 6. Remaining Limitations & Risk Analysis

1. **Possibility of Unannotated Defects:**
   * In industrial datasets like CR7-DET, annotators label dominant, obvious defects. Unannotated steel regions may harbor subtle roll abrasions, micro-cavities, or roller chatter marks that fell below annotator sensitivity.
   * Objective filtering removes edge occlusions, but only human review of the contact sheets can catch subtle morphological defects.
2. **Specular Roller Curvature Seams:**
   * Because contrast filtering was intentionally omitted, candidates located across the midline of the steel roll exhibit bright vertical reflection bands. Human reviewers must verify whether these specular seams represent healthy reflective steel or surface anomalies.
3. **Substrate Domain Boundary:**
   * All negative candidates depict cold-rolled steel plates under industrial illumination. They provide zero coverage of non-metallic materials (plastic shampoo bottles, pharmaceutical pills, ceramic tiles).
4. **Validation Set Size:**
   * The validation split yielded 85 passed candidates after the 25px margin exclusion. All 85 are included in the review set. If human review disqualifies a portion of these, the validation pool may require supplemental sampling from other defect-free images in that partition.

---

## 7. State & Boundary Verification

* **Models Trained:** 0 (Preparation only).
* **Production Code Modified:** 0 files.
* **Production Manifests / Databases Altered:** 0 files (`dataset_manifest.jsonl` untouched).
* **Original Evidence Preserved:**
  * Original candidates manifest: `background_candidates_manifest.jsonl` (Untouched).
  * Original crops in `background_audit/crops/` (Untouched).
  * Original review artifacts in `background_audit/manual_review/` (Untouched).
  * All new outputs strictly segregated in `background_audit/filtered_review/`.
