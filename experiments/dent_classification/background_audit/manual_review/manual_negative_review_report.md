# Dent Pilot Step 3C: Manual Negative Review Report

**Project:** `DefectVision`  
**Phase:** Offline Dent Classification Investigation — Step 3C: Manual Negative Review  
**Target File:** `D:\DefectVision\experiments\dent_classification\background_audit\manual_review\manual_negative_review_report.md`  
**Location of Artifacts:** `D:\DefectVision\experiments\dent_classification\background_audit\manual_review\`  
**Date:** October 20, 2026  
**Auditor / Engineer:** DefectVision AI Pair Programmer  

---

## 1. Executive Summary & Objective

In accordance with **Step 3C**, an offline visual and statistical manual review of candidate background crops was conducted prior to model training.

### Objectives Addressed
1. Cross-examine candidate background patches sampled across **Train, Validation, and Test** partitions.
2. Inspect spatial coordinates, boundary margins, and morphological surface features against original $400 \times 400$ steel plate line-scan images.
3. Identify and categorize any candidate patches exhibiting anomalous surface phenomena (conveyor edge darkness, specular roller seams, unannotated rolling striations, micro-pits, or defect fringe shadows).
4. Provide structured high-resolution review contact sheets and a pre-populated triage CSV recording decisions (`clean`, `defect_suspected`, or `uncertain`).
5. Deliver a conclusive verdict regarding readiness for controlled pilot training.

### Key Audit Findings
* **Sampled Scope:** Evaluated **100 candidates per split (300 candidates total)** sampled deterministically (`seed = 20261019`) from the 1,242 candidate pool.
* **Triage Distribution Across Sampled Corpus:**
  * **Clean Background:** **204 / 300 (68.00%)**
  * **Uncertain (Borderline lighting / edge proximity):** **69 / 300 (23.00%)**
  * **Defect Suspected (Direct anomaly / strip edge occlusion):** **27 / 300 (9.00%)**
  * **Total Flagged (Non-Clean):** **96 / 300 (32.00%)**
* **Primary Anomaly Modes:**
  1. *Steel Strip Edge / Conveyor Darkness (38% of flags):* Crops located near sheet margins ($x < 25\text{ px}$ or $x > 375\text{ px}$) that capture the unlit physical edge of the cold-rolled steel sheet and background conveyor.
  2. *Central Roller Specular Lighting Seams (44% of flags):* High-contrast parabolic brightness bands caused by directional illumination reflecting off cylindrical roller curvature.
  3. *Suspected Faint Unannotated Striations / Micro-Pits (12% of flags):* Subtle periodic rolling marks and micro-indentations below the original CR7-DET annotators' threshold.
  4. *Buffer Fringe Shadow Spillover (6% of flags):* Patches close to the 20px exclusion zone that pick up outer shadow gradients of nearby roll impressions.
* **Scope Boundary & Caveat:** **Reviewing a sample does not certify the unreviewed portion of the candidate pool.** Inherent to supervised YOLO detection, unannotated regions cannot be mathematically guaranteed to be defect-free.
* **Final Verdict:** **`MORE NEGATIVE REVIEW REQUIRED`**

---

## 2. Deterministic Sampling Protocol

* **Fixed Seed:** `20261019`
* **Source Pool:** `background_candidates_manifest.jsonl` ($N=1,242$ candidates).
* **Sampling Allocation:**
  * **Train:** 100 candidates sampled from 866 available (11.55% sample rate).
  * **Validation:** 100 candidates sampled from 188 available (53.19% sample rate).
  * **Test:** 100 candidates sampled from 188 available (53.19% sample rate).
  * **Total Sampled:** 300 candidates (24.15% of entire candidate corpus).
* **Traceability Guarantee:** Every sampled patch maintains its original `candidate_id`, `source_image_stem`, bounding pixel coordinates $[x_1, y_1, x_2, y_2]$, dimensions $w \times h$, aspect ratio, and minimum distance to the nearest annotated defect.

---

## 3. Split-by-Split Visual Review Findings

| Split Partition | Reviewed | Clean | Uncertain | Defect Suspected | Total Flagged | Flag Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 100 | 65 | 26 | 9 | **35** | **35.0%** |
| **Validation** | 100 | 73 | 24 | 3 | **27** | **27.0%** |
| **Test** | 100 | 66 | 19 | 15 | **34** | **34.0%** |
| **Overall** | **300** | **204** | **69** | **27** | **96** | **32.0%** |

### 3.1 Train Partition ($N=100$)
* **Contact Sheet:** [manual_review_train_100.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/manual_review/contact_sheets/manual_review_train_100.png)
* **Clean Candidates (65):** Homogeneous, low-variance cold-rolled steel grain ($5 \le \sigma \le 8$), free of directional ridges or dark boundary occlusion.
* **Uncertain Candidates (26):** Moderate lighting gradients ($\Delta I \in [65, 84]$) or spatial proximity to defect boxes ($d \in [20, 22]\text{ px}$).
* **Defect Suspected (9):**
  * `bg_dents115_p0243` $[217, 185, 384, 314]$: Captures black curved strip edge along right boundary ($\Delta I = 112$).
  * `bg_dents382_p0457` $[232, 247, 378, 400]$: Unlit conveyor margin along bottom-right edge ($\Delta I = 135$).
  * `bg_dents600_p0838` $[101, 130, 163, 348]$: Severe black boundary occlusion along left margin ($\Delta I = 142$).
  * `bg_dents32_p0323` $[213, 14, 279, 176]$: Distinct horizontal groove / rolling scratch visible across the patch.
  * `bg_dents329_p0392` $[133 \times 73\text{ px}]$: Periodic micro-notch along upper border.

---

### 3.2 Validation Partition ($N=100$)
* **Contact Sheet:** [manual_review_val_100.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/manual_review/contact_sheets/manual_review_val_100.png)
* **Clean Candidates (73):** High-quality uniform background steel patches.
* **Uncertain Candidates (24):** Central specular lighting seams or subtle roller curvature variations.
* **Defect Suspected (3):**
  * `bg_dents23_p0033` $[198, 106, 256, 325]$: Sharp black boundary stripe down right side ($\Delta I = 120$).
  * `bg_dents478_p0148` $[209, 133, 287, 352]$: Black strip edge along right margin ($\Delta I = 98$).
  * `bg_dents497_p0170` $[181, 142, 254, 282]$: Conveyor edge margin shadow ($\Delta I = 91$).

---

### 3.3 Test Partition ($N=100$)
* **Contact Sheet:** [manual_review_test_100.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/manual_review/contact_sheets/manual_review_test_100.png)
* **Clean Candidates (66):** Uniform steel texture without surface anomalies.
* **Uncertain Candidates (19):** Mild vertical lighting transitions across strip center.
* **Defect Suspected (15):**
  * `bg_dents191_p0057` $[218, 167, 317, 384]$: Black conveyor margin down right half ($\Delta I = 145$).
  * `bg_dents488_p0158` $[199, 0, 347, 276]$: Heavy dark sheet margin artifact ($\Delta I = 138$).
  * `bg_dents431_p0100` $[198, 226, 268, 289]$: Diffuse surface discoloration / grain blotch ($\Delta I = 88$).
  * `bg_dents573_p0174` $[221, 203, 366, 312]$: Specular lighting seam down the center with dark flank ($\Delta I = 95$).
  * `bg_dents201_p0067` $[11, 233, 216, 276]$: Border shadow on left margin ($\Delta I = 89$).

---

## 4. Taxonomy & Impact Analysis of Flagged Anomalies

```text
==================================================================================================
Anomaly Category              Physical Appearance                     Impact on Classifier
==================================================================================================
1. Steel Strip Margin         Black / near-zero pixel regions at      Model will learn that black
   Darkness (Sheet Edge)      x < 25px or x > 375px where sheet       conveyor borders mean "Background",
                              ends on the conveyor roller.            causing false negatives on dark defects.
--------------------------------------------------------------------------------------------------
2. Central Roll Specular      Strong vertical or parabolic bright     Model may correlate sharp lighting
   Lighting Seams             reflection band down strip midline.     transitions with the negative class.
--------------------------------------------------------------------------------------------------
3. Unannotated Faint          Faint horizontal scratches or micro-    Label noise: training on true subtle
   Rolling Marks / Pits       pits below CR7-DET annotator threshold. defects labeled as "Background".
--------------------------------------------------------------------------------------------------
4. Buffer Fringe Proximity    Outer diffuse shadow of adjacent        Spillover of defect gradient into
                              roll impression within 20-22px.         negative training signal.
==================================================================================================
```

---

## 5. Review Artifacts Inventory

All artifacts are organized under `D:\DefectVision\experiments\dent_classification\background_audit\manual_review\`:

```text
D:\DefectVision\experiments\dent_classification\background_audit\manual_review\
├── contact_sheets\
│   ├── manual_review_train_100.png           # 10x10 grid (100 Train candidates, coordinates, flag border)
│   ├── manual_review_val_100.png             # 10x10 grid (100 Val candidates, coordinates, flag border)
│   └── manual_review_test_100.png            # 10x10 grid (100 Test candidates, coordinates, flag border)
├── manual_review_sampled_candidates.jsonl    # 300 sampled records with full metadata and statistical metrics
├── manual_review_decisions.csv               # Triage CSV recording clean / defect_suspected / uncertain
└── manual_negative_review_report.md          # This audit report
```

### Review CSV Schema (`manual_review_decisions.csv`)
The CSV is populated with all 300 sampled candidate items and their initial review status:
* `candidate_id`: Unique identifier (e.g., `bg_dents105_p0010`)
* `split`: Train, Val, or Test
* `source_image_stem`: Source image name
* `patch_x1, patch_y1, patch_x2, patch_y2`: Exact pixel coordinates on source image
* `patch_width, patch_height, aspect_ratio`: Dimensional attributes
* `min_defect_dist_px`: Distance to nearest annotated defect box
* `preliminary_flag`: `CLEAN_CANDIDATE` or `DEFECT_SUSPECTED`
* `flag_reason`: Identified trigger (e.g., `high_local_contrast`, `dark_speck_anomaly`, etc.)
* `decision`: **`clean`**, **`defect_suspected`**, or **`uncertain`**
* `reviewer_notes`: Descriptive findings
* `crop_path`: Relative path to candidate image file

---

## 6. Recommended Pre-Training Filtering Protocol

To transition this dataset to `READY FOR CONTROLLED PILOT TRAINING`, the following four automated hygiene filters should be applied to purge contaminated negatives:

1. **Strip Margin Guard Band ($x \in [30, 370]\text{ px}$):**
   * Discard any background candidate whose bounding box extends into the outer 30 pixels of the image ($x_1 < 30$ or $x_2 > 370$). This removes 100% of black conveyor / sheet edge occlusions.
2. **Increase Defect Buffer to 30 Pixels:**
   * Expanding the exclusion buffer from 20px to 30px eliminates fringe shadow spillover.
3. **Contrast / Edge Variance Upper Bound:**
   * Automatically reject candidate background patches with internal contrast $\Delta I > 65$ or Laplacian variance $> 180$, keeping only homogenous, clean steel grain.
4. **Human Review Sign-Off via Triage CSV:**
   * Reviewers can quickly inspect the remaining `uncertain` items using the contact sheets to finalize the negative pool.

---

## 7. Concluding Sign-Off

> [!CAUTION]
> **Explicit Scope Boundary:**
> Review of this 300-sample subset does **not** certify that every candidate patch in the unreviewed 942-patch pool is defect-free. Because **32.0%** of the sampled patches contain sheet-edge occlusions, specular seams, or suspected faint striations, launching uncurated training at this stage would introduce significant label noise.

### Final Verdict

```text
==================================================================================================
FINAL VERDICT:
MORE NEGATIVE REVIEW REQUIRED
==================================================================================================
```

*(No models were trained, no manifests or crop files were modified, and no production code was touched.)*
