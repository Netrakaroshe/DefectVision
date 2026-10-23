# DefectVision Phase 4 Step 1: Defect Classifier Dataset Audit Report
**Date:** 2026-10-03T04:33:18.865105+00:00  **Authoritative Source:** `D:/DefectVision/manifests/dataset_manifest.jsonl`  **Status:** COMPLETE & AUDITED
---
## 1. Executive Summary & Governance Overview
This audit establishes the ground-truth eligibility of all records in `dataset_manifest.jsonl` for the Phase 4 supervised defect classifier (taxonomy: **Scratch**, **Crack**, **Discoloration**, **UNKNOWN**).
| Category / Role | Status | Total Images | Action for Phase 4 |
|---|---|---:|---|
| **Direct Approved Defect Sources** | `DIRECT` | **1764** | **ELIGIBLE** (Core Target Classes) |
| **Unknown Defect Sources** | `NO_MAPPING` | **2870** | **ELIGIBLE** (Target Class: UNKNOWN) |
| **Classifier Review / Ambiguous** | `REVIEW / NOT_EQUIV` | **909** | **EXCLUDED** (Label noise risk) |
| **Normal Reference Images** | `NORMAL_REFERENCE` | **4217** | **EXCLUDED** (Stage 1 PatchCore only) |
| **Total Manifest Records** | All | **9760** | — |

---
## 2. Exact Target Class Counts

| Target Class | Mapping Status | Direct Sources | Approved Count |
|---|---|---|---:|
| **Scratch** | `DIRECT` | NEU-CLS (300) + MVTecAD (119) | **419** |
| **Crack** | `DIRECT` | MT (57) + MVTecAD (84) | **141** |
| **Discoloration** | `DIRECT` | GC10-DET (1,182) + MVTecAD (22) | **1204** |
| **Subtotal (3 Specific Classes)** | `DIRECT` | All Approved | **1764** |
| **UNKNOWN** | `NO_MAPPING` | GC10 (1,044) + MVTec (779) + NEU (900) + MT (147) | **2870** |
| **Grand Total Defective Pool** | `DIRECT` + `UNKNOWN` | Combined Eligible Pool | **4634** |

---
## 3. Granular Breakdown by Class, Dataset, and Original Label

### Target Class: Discoloration (1204 images)

| Source Dataset | Product / Object | Original Dataset Label | Split | Count |
|---|---|---|---|---:|
| `GC10-DET` | `steel_surface` | `oil_spot` | `test` | 35 |
| `GC10-DET` | `steel_surface` | `oil_spot` | `train` | 160 |
| `GC10-DET` | `steel_surface` | `oil_spot` | `val` | 34 |
| `GC10-DET` | `steel_surface` | `silk_spot` | `test` | 100 |
| `GC10-DET` | `steel_surface` | `silk_spot` | `train` | 463 |
| `GC10-DET` | `steel_surface` | `silk_spot` | `val` | 99 |
| `GC10-DET` | `steel_surface` | `water_spot` | `test` | 45 |
| `GC10-DET` | `steel_surface` | `water_spot` | `train` | 203 |
| `GC10-DET` | `steel_surface` | `water_spot` | `val` | 43 |
| `MVTecAD` | `metal_nut` | `color` | `test` | 22 |

### Target Class: Crack (141 images)

| Source Dataset | Product / Object | Original Dataset Label | Split | Count |
|---|---|---|---|---:|
| `MT` | `magnetic_tile` | `MT_Crack` | `test` | 29 |
| `MT` | `magnetic_tile` | `MT_Crack` | `val` | 28 |
| `MVTecAD` | `capsule` | `crack` | `test` | 23 |
| `MVTecAD` | `hazelnut` | `crack` | `test` | 18 |
| `MVTecAD` | `pill` | `crack` | `test` | 26 |
| `MVTecAD` | `tile` | `crack` | `test` | 17 |

### Target Class: Scratch (419 images)

| Source Dataset | Product / Object | Original Dataset Label | Split | Count |
|---|---|---|---|---:|
| `MVTecAD` | `capsule` | `scratch` | `test` | 23 |
| `MVTecAD` | `metal_nut` | `scratch` | `test` | 23 |
| `MVTecAD` | `pill` | `scratch` | `test` | 24 |
| `MVTecAD` | `screw` | `scratch_head` | `test` | 24 |
| `MVTecAD` | `screw` | `scratch_neck` | `test` | 25 |
| `NEU-CLS` | `steel_surface` | `scratches` | `test` | 45 |
| `NEU-CLS` | `steel_surface` | `scratches` | `train` | 210 |
| `NEU-CLS` | `steel_surface` | `scratches` | `val` | 45 |

### Target Class: UNKNOWN (2,870 images across 4 datasets)

| Source Dataset | Product | Original Label | Count |
|---|---|---|---:|
| `GC10-DET` | `steel_surface` | `crescent_gap` | 225 |
| `GC10-DET` | `steel_surface` | `inclusion` | 197 |
| `GC10-DET` | `steel_surface` | `punching_hole` | 209 |
| `GC10-DET` | `steel_surface` | `waist_folding` | 145 |
| `GC10-DET` | `steel_surface` | `welding_line` | 268 |
| `MT` | `magnetic_tile` | `MT_Blowhole` | 115 |
| `MT` | `magnetic_tile` | `MT_Fray` | 32 |
| `MVTecAD` | `bottle` | `broken_large` | 20 |
| `MVTecAD` | `bottle` | `broken_small` | 22 |
| `MVTecAD` | `bottle` | `contamination` | 21 |
| `MVTecAD` | `cable` | `bent_wire` | 13 |
| `MVTecAD` | `cable` | `cable_swap` | 12 |
| `MVTecAD` | `cable` | `combined` | 11 |
| `MVTecAD` | `cable` | `cut_inner_insulation` | 14 |
| `MVTecAD` | `cable` | `cut_outer_insulation` | 10 |
| `MVTecAD` | `cable` | `missing_cable` | 12 |
| `MVTecAD` | `cable` | `missing_wire` | 10 |
| `MVTecAD` | `cable` | `poke_insulation` | 10 |
| `MVTecAD` | `capsule` | `faulty_imprint` | 22 |
| `MVTecAD` | `capsule` | `poke` | 21 |
| `MVTecAD` | `capsule` | `squeeze` | 20 |
| `MVTecAD` | `carpet` | `color` | 19 |
| `MVTecAD` | `carpet` | `cut` | 17 |
| `MVTecAD` | `carpet` | `hole` | 17 |
| `MVTecAD` | `carpet` | `metal_contamination` | 17 |
| `MVTecAD` | `carpet` | `thread` | 19 |
| `MVTecAD` | `grid` | `bent` | 12 |
| `MVTecAD` | `grid` | `broken` | 12 |
| `MVTecAD` | `grid` | `glue` | 11 |
| `MVTecAD` | `grid` | `metal_contamination` | 11 |
| `MVTecAD` | `grid` | `thread` | 11 |
| `MVTecAD` | `hazelnut` | `cut` | 17 |
| `MVTecAD` | `hazelnut` | `hole` | 18 |
| `MVTecAD` | `hazelnut` | `print` | 17 |
| `MVTecAD` | `leather` | `color` | 19 |
| `MVTecAD` | `leather` | `fold` | 17 |
| `MVTecAD` | `leather` | `glue` | 19 |
| `MVTecAD` | `leather` | `poke` | 18 |
| `MVTecAD` | `metal_nut` | `bent` | 25 |
| `MVTecAD` | `metal_nut` | `flip` | 23 |
| `MVTecAD` | `pill` | `color` | 25 |
| `MVTecAD` | `pill` | `combined` | 17 |
| `MVTecAD` | `pill` | `contamination` | 21 |
| `MVTecAD` | `pill` | `faulty_imprint` | 19 |
| `MVTecAD` | `pill` | `pill_type` | 9 |
| `MVTecAD` | `screw` | `manipulated_front` | 24 |
| `MVTecAD` | `screw` | `thread_side` | 23 |
| `MVTecAD` | `screw` | `thread_top` | 23 |
| `MVTecAD` | `tile` | `glue_strip` | 18 |
| `MVTecAD` | `tile` | `oil` | 18 |
| `MVTecAD` | `tile` | `rough` | 15 |
| `MVTecAD` | `toothbrush` | `defective` | 30 |
| `NEU-CLS` | `steel_surface` | `inclusion` | 300 |
| `NEU-CLS` | `steel_surface` | `pitted_surface` | 300 |
| `NEU-CLS` | `steel_surface` | `rolled-in_scale` | 300 |

---
## 4. Excluded Datasets and Images

The following **909 images** are strictly **EXCLUDED** from classifier training due to ambiguous semantics or unverified labels:

| Source Dataset | Product | Original Label | Mapping Status | Count | Reason for Exclusion |
|---|---|---|---|---:|---|
| `GC10-DET` | `steel_surface` | `crease` | `NOT_EQUIVALENT` | 51 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `GC10-DET` | `steel_surface` | `rolled_pit` | `POSSIBLE_BUT_REQUIRES_REVIEW` | 35 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `MT` | `magnetic_tile` | `MT_Break` | `POSSIBLE_BUT_REQUIRES_REVIEW` | 85 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `MT` | `magnetic_tile` | `MT_Uneven` | `POSSIBLE_BUT_REQUIRES_REVIEW` | 103 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `MVTecAD` | `leather` | `cut` | `NOT_EQUIVALENT` | 19 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `MVTecAD` | `tile` | `gray_stroke` | `POSSIBLE_BUT_REQUIRES_REVIEW` | 16 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `NEU-CLS` | `steel_surface` | `crazing` | `POSSIBLE_BUT_REQUIRES_REVIEW` | 300 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |
| `NEU-CLS` | `steel_surface` | `patches` | `POSSIBLE_BUT_REQUIRES_REVIEW` | 300 | Mapping status requires review or is not equivalent; excluded to prevent label contamination |

Additionally, all **4,217 normal reference images** are excluded because the classifier operates strictly downstream of PatchCore defect detection.

---
## 5. Duplicate and Leakage Audit Findings

- **Physical Image Existence**: PASS (100% verified) (0 missing images).
- **Duplicate Source Paths**: 0 (strictly zero duplicate paths).
- **Duplicate Sample IDs**: 0 (strictly zero duplicate IDs).
- **Contaminated Source Check**: Zero paths contain `Surface\Magnetic-Tile-Defect` or `defect-categories`.
- **Image Content MD5 Hash Collisions**: Found **1** collision in the defective pool:
  - Hash `24f39e0f91f95e4c785052ac4d3e49a6`: 2 identical files in NEU-CLS (`patches_101.jpg` and `patches_105.jpg`). Both belong to the `classifier_review` category and are already **EXCLUDED** from training.
  - **Zero content collisions** exist within the approved `DIRECT` or `UNKNOWN` training candidate pools.

---
## 6. Manifest Split Assessment & Critical Finding

> [!CAUTION]
> **EXISTING MANIFEST SPLITS CANNOT BE USED FOR CLASSIFIER TRAINING.**
>
> Existing manifest split was constructed for unsupervised anomaly detection benchmarks (PatchCore). In MVTecAD, all defect images are placed into 'test' split. In MT, defects are in 'val' and 'test'. Consequently, 'Crack' has exactly 0 images in the manifest 'train' split, and MVTecAD has 0 defect images in 'train'. Supervised training on the raw manifest splits would result in catastrophic class absence.

Current distribution of `DIRECT` candidates across manifest splits:

| Target Class | Train Count | Val Count | Test Count | Manifest Split Suitability |
|---|---:|---:|---:|---|
| **Discoloration** | 826 | 176 | 202 | Heavily Skewed |
| **Crack** | 0 | 28 | 113 | UNUSABLE (0 in train) |
| **Scratch** | 210 | 45 | 164 | Heavily Skewed |

**Recommendation**: Implement a dedicated, deterministic supervised classifier partition (e.g. 70% train, 15% val, 15% test) stratified across (Target Class, Source Dataset, Product/Category), with strict leak-free isolation.

---
## 7. Class Imbalance and Domain Risks

### Class Imbalance Analysis
- **Discoloration to Crack Ratio**: `8.54 : 1`
- **Scratch to Crack Ratio**: `2.97 : 1`
- **UNKNOWN Defect Pool**: 2870 images (over 20x the size of Crack).
- **Mitigation**: Stratified sampling, focal loss, or downsampling of the UNKNOWN class during training.

### Key Domain Risks
1. Cross-dataset domain shift: Grayscale sheet metal (NEU/GC10) vs dark ferrite tile (MT) vs RGB objects (MVTecAD).
1. Annotation scale disparity: Full-tile classification (MT, NEU) vs bounding-box extracted crops (GC10) vs localized masks (MVTecAD).
1. Small sample size for Crack (141 images across 5 distinct products).

---
## 8. Recommended Classifier Dataset Policy

### Eligible Sources
- Approved Scratch: NEU-CLS (300) + MVTecAD capsule, metal_nut, pill, screw (119) = 419 images
- Approved Crack: MT MT_Crack (57) + MVTecAD capsule, hazelnut, pill, tile (84) = 141 images
- Approved Discoloration: GC10-DET oil_spot, silk_spot, water_spot (1,182) + MVTecAD metal_nut color (22) = 1,204 images
- Approved UNKNOWN: Stratified subset of 2,870 unknown_defect_source images (curated to ~300-500 images to balance classes)

### Strict Exclusions
- All 909 classifier_review images (crazing, patches, MT_Break, MT_Uneven, rolled_pit, crease, leather/cut, tile/gray_stroke)
- All 4,217 normal reference images (used only in Stage 1 PatchCore)
- Surface/Magnetic-Tile-Defect and defect-categories directories (contaminated)

### Split & Leakage Policy
- **Split Strategy**: Dedicated stratified 70/15/15 partition by (Target Class, Dataset, Category).
- **Leakage Guard**: Stratify by `(Target_Class, Dataset, Product_Category)` using a fixed random seed (42), ensuring identical physical products are not split across train and test.
