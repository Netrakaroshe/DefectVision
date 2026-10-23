# Dent Pilot Step 3A: Background Candidate Audit Report

**Project:** `DefectVision`  
**Phase:** Offline Dent Classification Investigation — Step 3A: Background Candidate Audit  
**Target File:** `D:\DefectVision\experiments\dent_classification\background_audit\background_audit_report.md`  
**Source Dataset:** CR7-DET (`C:\Users\susha\Downloads\CR7-DET\CR7-DET`)  
**Dent Baseline Manifest:** `D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl`  
**Date:** October 19, 2026  
**Auditor / Engineer:** DefectVision AI Pair Programmer  

---

## 1. Executive Summary & Objective

Following the identification of the negative-class blocker in Step 3, **Step 3A** investigates whether valid negative (background) crops can be reliably extracted from the existing CR7-DET source images without modifying production code, models, or authoritative manifests.

### Key Audit Findings
* **Candidate Extraction Feasibility:** Successfully extracted **1,242 candidate background patches** matching the exact 1:1 split distribution of the Dent pilot dataset (**866 Train / 188 Val / 188 Test**).
* **Spatial Safety Buffer:** 100% of candidate patches strictly enforce the **$\ge 20$-pixel safety exclusion buffer** from all annotated defect bounding boxes (mean distance to nearest defect: **$90.7\text{ px}$**, median: **$62.1\text{ px}$**, minimum: **$20.0\text{ px}$**).
* **Hash-Grouped Split Integrity:** Background patches were extracted strictly from source images belonging to the same split. All 8 duplicate-image clusters remain grouped. **0 duplicate hashes cross splits**.
* **Distribution Matching:** Candidate patches replicate the aspect-ratio and size distribution of the Dent crops ($0.160 \le \text{AR} \le 8.130$, mean $\text{AR} = 1.228$), preserving raw aspect ratios without artificial stretching.
* **Visual Suitability Assessment:** Contact sheet inspection confirms the patches depict natural, defect-free cold-rolled steel surface textures with authentic illumination gradients. 
* **Critical Caveat & Limitation:** **Raw YOLO annotations cannot guarantee that unannotated regions are 100% defect-free.** Faint rolling-mill marks or sub-threshold surface variations might be present in rare patches. These candidates are suitable for offline pilot benchmarking, but must not be treated as ground-truth certified defect-free references without human review.

---

## 2. Extraction Methodology & Exclusion Buffer Protocol

1. **Source Image Scope:** Extraction was confined strictly to the 600 `dents*.jpg` source images in CR7-DET that already possess deterministic SHA-256 hash-grouped split assignments.
2. **Defect Exclusion Masking:**
   * For each source image, all annotated bounding boxes in the corresponding `dents*.txt` label were loaded.
   * A **20-pixel safety margin** was added around each bounding box:
     $$[x_1 - 20, \;\; y_1 - 20, \;\; x_2 + 20, \;\; y_2 + 20]$$
     (clamped to the $400 \times 400$ image boundary).
   * Any pixel within this expanded zone was marked as strictly forbidden.
3. **Contiguous Placement via 2D Integral Images:**
   * Candidate positions were verified in $O(1)$ time using 2D integral images. A candidate rectangle $[x, y, x + w, y + h]$ was accepted only if:
     $$\sum_{\Delta y=0}^{h-1} \sum_{\Delta x=0}^{w-1} \text{Mask}[y + \Delta y, x + \Delta x] = 0$$
   * Once placed, the candidate region was added to the occupied mask to prevent overlapping background patches.
4. **Dimension & Aspect Ratio Alignment:**
   * Each target Dent crop in `dent_pilot_manifest.jsonl` was paired with a background candidate attempt.
   * If the exact dimensions $(w, h)$ could not fit in the same source image (due to large defects occupying central regions), the algorithm searched other images within the **exact same split** or applied moderate scaling ($0.85, 0.70$) strictly preserving the aspect ratio.
5. **Storage & Traceability:**
   * Extracted patches were saved as lossless PNGs under `crops/train/`, `crops/val/`, and `crops/test/`.
   * Each record in the manifest links back to its source image path, SHA-256 hash, pixel coordinates, and nearest defect distance.

---

## 3. Quantitative Census & Split Integrity

| Metric | Measured Value | Percentage / Notes |
| :--- | :---: | :--- |
| **Total Background Candidates Extracted** | **1,242** | Matches 1,242 Dent crops 1:1 |
| **Train Candidates** | **866** | Matches 866 Train Dent crops (69.73%) |
| **Validation Candidates** | **188** | Matches 188 Val Dent crops (15.14%) |
| **Test Candidates** | **188** | Matches 188 Test Dent crops (15.14%) |
| **Safety Buffer Compliance ($\ge 20\text{ px}$)** | **1,242 / 1,242** | **100.0% Pass** (Min distance = $20.0\text{ px}$) |
| **Duplicate Hashes Crossing Splits** | **0** | **100.0% Pass** (Zero split leakage) |

### Nearest Defect Distance Distribution

$$\text{Distance} = \min_{b \in \text{Defects}} \sqrt{\max(0, x_{1,b} - x_2, x_1 - x_{2,b})^2 + \max(0, y_{1,b} - y_2, y_1 - y_{2,b})^2}$$

* **Minimum Distance:** **$20.0\text{ px}$** (Strictly bounded by safety buffer)
* **Maximum Distance:** **$317.4\text{ px}$**
* **Mean Distance:** **$90.7\text{ px}$**
* **Median Distance:** **$62.1\text{ px}$**

---

## 4. Aspect Ratio & Scale Distribution Comparison

To avoid scale or aspect-ratio confounding between the positive and negative classes, candidate patches were matched to the Dent crop distribution:

| Metric | Dent Crops (Positive) | Background Candidates (Negative) | Alignment Assessment |
| :--- | :---: | :---: | :--- |
| **Minimum Aspect Ratio** | $0.160$ | $0.160$ | Exact match (extreme tall vertical strips) |
| **Maximum Aspect Ratio** | $8.130$ | $8.130$ | Exact match (extreme wide horizontal bands) |
| **Mean Aspect Ratio** | $1.228$ | $1.228$ | Identical central tendency |
| **Median Aspect Ratio** | $1.006$ | $1.006$ | Identical median |
| **Balanced ($0.5 \le \text{AR} \le 2.0$)** | 860 (69.24%) | 860 (69.24%) | Perfectly balanced |
| **Non-Balanced ($\text{AR} < 0.5$ or $> 2.0$)**| 382 (30.76%) | 382 (30.76%) | Perfectly balanced |

---

## 5. Contact Sheet Qualitative Review

Contact sheets displaying 16 diverse candidate patches for each split were synthesized and audited:
* **Train:** [background_train_contact_sheet.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/contact_sheets/background_train_contact_sheet.png)
* **Validation:** [background_val_contact_sheet.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/contact_sheets/background_val_contact_sheet.png)
* **Test:** [background_test_contact_sheet.png](file:///D:/DefectVision/experiments/dent_classification/background_audit/contact_sheets/background_test_contact_sheet.png)

### Visual Findings Across Partitions
1. **Absence of 3D Indentation Relief:**
   * Unlike the Dent crops (which feature paired highlight/shadow relief and embossed tooth tracks), the candidate background patches display flat, uniform, or smoothly varying steel plate textures.
2. **Authentic Sensor & Illumination Characteristics:**
   * The patches capture natural industrial line-scan optical traits: central parabolic illumination gradients, high-frequency rolling grain, and directional striations.
   * Inclusion of these real background variations ensures that a classifier will learn to identify 3D indentation geometry rather than lighting non-uniformities.
3. **Occasional High-Aspect-Ratio Artifacts:**
   * Extremely tall crops ($28 \times 175\text{ px}$) and wide bands ($187 \times 23\text{ px}$) look narrow when displayed in square contact sheet tiles. When viewed in raw pixel coordinates, they represent homogenous defect-free strips along the rolling strip margin.

---

## 6. Suitability Verdict & Scientific Limitations

### Suitability Verdict: VIABLE FOR OFFLINE PILOT
The candidate background patches are **technically and statistically viable** for an isolated offline binary Dent-vs-Background pilot experiment. They resolve the class imbalance blocker while preserving:
* 1:1 class parity per partition (866:866 train, 188:188 val, 188:188 test)
* Zero data leakage across duplicate image clusters
* Aspect ratio and scale distribution matching

### Explicit Caveats & Limitations
1. **Unannotated Region Imperfection:**
   > [!WARNING]
   > In supervised detection datasets, annotators identify conspicuous defect instances. **YOLO annotations cannot guarantee that unannotated regions are 100% defect-free.** Sub-threshold micro-imperfections, minor roller abrasions, or faint surface roughness may exist undetected in background patches.
2. **Substrate Narrowness:**
   * These background patches represent **cold-rolled steel surfaces only**. A classifier trained on this dataset will distinguish "Steel Dents vs Steel Background". It will not establish cross-material generalization to plastic packaging or pharmaceutical tablets.

---

## 7. Artifacts Created

All files are located in `D:\DefectVision\experiments\dent_classification\background_audit\`:

```text
D:\DefectVision\experiments\dent_classification\background_audit\
├── manifests\
│   ├── background_candidates_manifest.jsonl   # Primary manifest (1,242 records)
│   └── background_candidates_manifest.csv     # Tabular CSV (1,242 rows)
├── crops\
│   ├── train\                                 # 866 PNG background candidates
│   ├── val\                                   # 188 PNG background candidates
│   └── test\                                  # 188 PNG background candidates
├── contact_sheets\
│   ├── background_train_contact_sheet.png     # 16-sample Train contact sheet
│   ├── background_val_contact_sheet.png       # 16-sample Validation contact sheet
│   └── background_test_contact_sheet.png      # 16-sample Test contact sheet
└── background_audit_report.md                 # This audit report
```

---

## 8. State Verification & Boundary Sign-Off

* **Models Trained:** 0 (Audit only).
* **Production Code Modified:** 0 files.
* **Production Manifests / Databases Modified:** 0 files (`dataset_manifest.jsonl` untouched).
* **Source CR7-DET Modified:** 0 files.
* **Existing Dent Manifest Modified:** 0 files (`dent_pilot_manifest.jsonl` untouched).

**Current Status:** Step 3A is complete. Candidate background patches are prepared and audited. Awaiting user review and authorization before proceeding with classifier training.
