# DefectVision Phase 4 Step 2: Supervised Classifier Dataset Splits Report
**Date:** 2026-10-03T04:37:12.117870+00:00  
**Random Seed:** `42` (Deterministic reproducibility)  
**Dataset Root:** `D:/DefectVision/experiments/defect_classifier_dataset`  

---
## 1. Overall Partition Summary

| Split | Sample Count | Percentage | Purpose |
|---|---:|---:|---|
| **Train** | **1505** | **68.94%** | Supervised model parameter optimization |
| **Validation** | **333** | **15.25%** | Hyperparameter tuning, checkpointing, early stopping |
| **Test** | **345** | **15.8%** | Unbiased held-out generalization benchmark |
| **Total** | **2183** | **100.0%** | Defect-type classification pool |

---
## 2. Target Class Counts Across Splits

| Target Class | Train | Val | Test | Total | Split Distribution |
|---|---:|---:|---:|---:|---|
| **Scratch** | 293 | 63 | 63 | **419** | 69.9% / 15.0% / 15.0% |
| **Crack** | 97 | 22 | 22 | **141** | 68.8% / 15.6% / 15.6% |
| **Discoloration** | 844 | 180 | 180 | **1204** | 70.1% / 15.0% / 15.0% |
| **UNKNOWN** | 271 | 68 | 80 | **419** | 64.7% / 16.2% / 19.1% |

---
## 3. Source Dataset Representation Across Splits

| Source Dataset | Train | Val | Test | Total | Split Distribution |
|---|---:|---:|---:|---:|---|
| `GC10-DET` | 898 | 192 | 192 | **1282** | 70.0% / 15.0% / 15.0% |
| `MT` | 83 | 19 | 19 | **121** | 68.6% / 15.7% / 15.7% |
| `MVTecAD` | 239 | 62 | 74 | **375** | 63.7% / 16.5% / 19.7% |
| `NEU-CLS` | 285 | 60 | 60 | **405** | 70.4% / 14.8% / 14.8% |

---
## 4. Multi-Dimensional Stratification Breakdown

### Source Dataset × Target Class × Split

| Source Dataset | Target Class | Train | Val | Test | Total |
|---|---|---:|---:|---:|---:|
| `GC10-DET` | **Discoloration** | 828 | 177 | 177 | 1182 |
| `GC10-DET` | **UNKNOWN** | 70 | 15 | 15 | 100 |
| `MT` | **Crack** | 39 | 9 | 9 | 57 |
| `MT` | **UNKNOWN** | 44 | 10 | 10 | 64 |
| `MVTecAD` | **Crack** | 58 | 13 | 13 | 84 |
| `MVTecAD` | **Discoloration** | 16 | 3 | 3 | 22 |
| `MVTecAD` | **Scratch** | 83 | 18 | 18 | 119 |
| `MVTecAD` | **UNKNOWN** | 82 | 28 | 40 | 150 |
| `NEU-CLS` | **Scratch** | 210 | 45 | 45 | 300 |
| `NEU-CLS` | **UNKNOWN** | 75 | 15 | 15 | 105 |

### Product / Object Category Representation

| Product / Category | Target Class | Train | Val | Test | Total |
|---|---|---:|---:|---:|---:|
| `bottle` | **UNKNOWN** | 8 | 2 | 3 | 13 |
| `cable` | **UNKNOWN** | 7 | 2 | 4 | 13 |
| `capsule` | **Crack** | 17 | 3 | 3 | 23 |
| `capsule` | **Scratch** | 17 | 3 | 3 | 23 |
| `capsule` | **UNKNOWN** | 7 | 3 | 3 | 13 |
| `carpet` | **UNKNOWN** | 6 | 2 | 5 | 13 |
| `grid` | **UNKNOWN** | 7 | 2 | 4 | 13 |
| `hazelnut` | **Crack** | 12 | 3 | 3 | 18 |
| `hazelnut` | **UNKNOWN** | 8 | 2 | 3 | 13 |
| `leather` | **UNKNOWN** | 6 | 3 | 3 | 12 |
| `magnetic_tile` | **Crack** | 39 | 9 | 9 | 57 |
| `magnetic_tile` | **UNKNOWN** | 44 | 10 | 10 | 64 |
| `metal_nut` | **Discoloration** | 16 | 3 | 3 | 22 |
| `metal_nut` | **Scratch** | 17 | 3 | 3 | 23 |
| `metal_nut` | **UNKNOWN** | 8 | 2 | 2 | 12 |
| `pill` | **Crack** | 18 | 4 | 4 | 26 |
| `pill` | **Scratch** | 16 | 4 | 4 | 24 |
| `pill` | **UNKNOWN** | 5 | 2 | 5 | 12 |
| `screw` | **Scratch** | 33 | 8 | 8 | 49 |
| `screw` | **UNKNOWN** | 6 | 3 | 3 | 12 |
| `steel_surface` | **Discoloration** | 828 | 177 | 177 | 1182 |
| `steel_surface` | **Scratch** | 210 | 45 | 45 | 300 |
| `steel_surface` | **UNKNOWN** | 145 | 30 | 30 | 205 |
| `tile` | **Crack** | 11 | 3 | 3 | 17 |
| `tile` | **UNKNOWN** | 6 | 3 | 3 | 12 |
| `toothbrush` | **UNKNOWN** | 8 | 2 | 2 | 12 |

---
## 5. UNKNOWN Class Selection Policy & Category Counts

To prevent the 2,870 available unknown defect images from drowning out `Crack` (141) and `Scratch` (419), a curated subset of **419 images** was selected. This count was intentionally chosen to **match the Scratch class count (419)** exactly.

### Selected Category Breakdown:

| Dataset | Product | Original Label | Selected Count | Selection Rationale |
|---|---|---|---:|---|
| `GC10-DET` | `steel_surface` | `crescent_gap` | 20 | Uniform category sampling |
| `GC10-DET` | `steel_surface` | `inclusion` | 20 | Uniform category sampling |
| `GC10-DET` | `steel_surface` | `punching_hole` | 20 | Uniform category sampling |
| `GC10-DET` | `steel_surface` | `waist_folding` | 20 | Uniform category sampling |
| `GC10-DET` | `steel_surface` | `welding_line` | 20 | Uniform category sampling |
| `MT` | `magnetic_tile` | `MT_Blowhole` | 32 | Uniform category sampling |
| `MT` | `magnetic_tile` | `MT_Fray` | 32 | Uniform category sampling |
| `MVTecAD` | `bottle` | `broken_large` | 7 | Uniform category sampling |
| `MVTecAD` | `bottle` | `broken_small` | 4 | Uniform category sampling |
| `MVTecAD` | `bottle` | `contamination` | 2 | Uniform category sampling |
| `MVTecAD` | `cable` | `bent_wire` | 3 | Uniform category sampling |
| `MVTecAD` | `cable` | `cable_swap` | 1 | Uniform category sampling |
| `MVTecAD` | `cable` | `combined` | 2 | Uniform category sampling |
| `MVTecAD` | `cable` | `cut_inner_insulation` | 1 | Uniform category sampling |
| `MVTecAD` | `cable` | `cut_outer_insulation` | 2 | Uniform category sampling |
| `MVTecAD` | `cable` | `missing_cable` | 3 | Uniform category sampling |
| `MVTecAD` | `cable` | `poke_insulation` | 1 | Uniform category sampling |
| `MVTecAD` | `capsule` | `faulty_imprint` | 3 | Uniform category sampling |
| `MVTecAD` | `capsule` | `poke` | 4 | Uniform category sampling |
| `MVTecAD` | `capsule` | `squeeze` | 6 | Uniform category sampling |
| `MVTecAD` | `carpet` | `color` | 2 | Uniform category sampling |
| `MVTecAD` | `carpet` | `cut` | 3 | Uniform category sampling |
| `MVTecAD` | `carpet` | `hole` | 2 | Uniform category sampling |
| `MVTecAD` | `carpet` | `metal_contamination` | 4 | Uniform category sampling |
| `MVTecAD` | `carpet` | `thread` | 2 | Uniform category sampling |
| `MVTecAD` | `grid` | `bent` | 5 | Uniform category sampling |
| `MVTecAD` | `grid` | `broken` | 1 | Uniform category sampling |
| `MVTecAD` | `grid` | `glue` | 3 | Uniform category sampling |
| `MVTecAD` | `grid` | `metal_contamination` | 2 | Uniform category sampling |
| `MVTecAD` | `grid` | `thread` | 2 | Uniform category sampling |
| `MVTecAD` | `hazelnut` | `cut` | 4 | Uniform category sampling |
| `MVTecAD` | `hazelnut` | `hole` | 2 | Uniform category sampling |
| `MVTecAD` | `hazelnut` | `print` | 7 | Uniform category sampling |
| `MVTecAD` | `leather` | `color` | 1 | Uniform category sampling |
| `MVTecAD` | `leather` | `fold` | 4 | Uniform category sampling |
| `MVTecAD` | `leather` | `glue` | 3 | Uniform category sampling |
| `MVTecAD` | `leather` | `poke` | 4 | Uniform category sampling |
| `MVTecAD` | `metal_nut` | `bent` | 6 | Uniform category sampling |
| `MVTecAD` | `metal_nut` | `flip` | 6 | Uniform category sampling |
| `MVTecAD` | `pill` | `color` | 2 | Uniform category sampling |
| `MVTecAD` | `pill` | `combined` | 3 | Uniform category sampling |
| `MVTecAD` | `pill` | `contamination` | 3 | Uniform category sampling |
| `MVTecAD` | `pill` | `faulty_imprint` | 2 | Uniform category sampling |
| `MVTecAD` | `pill` | `pill_type` | 2 | Uniform category sampling |
| `MVTecAD` | `screw` | `manipulated_front` | 6 | Uniform category sampling |
| `MVTecAD` | `screw` | `thread_side` | 3 | Uniform category sampling |
| `MVTecAD` | `screw` | `thread_top` | 3 | Uniform category sampling |
| `MVTecAD` | `tile` | `glue_strip` | 3 | Uniform category sampling |
| `MVTecAD` | `tile` | `oil` | 4 | Uniform category sampling |
| `MVTecAD` | `tile` | `rough` | 5 | Uniform category sampling |
| `MVTecAD` | `toothbrush` | `defective` | 12 | Uniform category sampling |
| `NEU-CLS` | `steel_surface` | `inclusion` | 35 | Uniform category sampling |
| `NEU-CLS` | `steel_surface` | `pitted_surface` | 35 | Uniform category sampling |
| `NEU-CLS` | `steel_surface` | `rolled-in_scale` | 35 | Uniform category sampling |

---
## 6. Leakage Audit and Integrity Verification

- **Path Overlap (Train ∩ Val)**: 0 (strictly 0)
- **Path Overlap (Train ∩ Test)**: 0 (strictly 0)
- **Path Overlap (Val ∩ Test)**: 0 (strictly 0)
- **Sample ID Overlap**: Strictly 0 across all partitions
- **Content MD5 Hash Leakage**: 0 (strictly 0 hash collisions across partitions)
- **Contaminated Subsets**: Strictly 0 images from `Surface\Magnetic-Tile-Defect` or `defect-categories`
- **Physical File Verification**: 100% of the 2,183 images physically exist on disk
- **Strict Leakage-Free Verdict**: `PASS`

---
## 7. Known Unavoidable Limitations

1. **Crack Class Rarity**: With only 141 total crack images, the train split has 97 images. Data augmentation and transfer learning will be necessary during model training.
2. **MVTec Category Granularity**: In MVTecAD, defects are collected on specific object types (e.g. `metal_nut`, `capsule`, `pill`). While stratified splitting ensures each product appears across splits, physical batch IDs were not provided by the raw dataset author; determinism relies on strict image-level path isolation.
3. **Cross-Domain Material Disparity**: The dataset combines planar sheet metal (NEU, GC10), ferrite tiles (MT), and 3D industrial objects (MVTecAD). Backbones pre-trained on ImageNet are recommended to generalize across these material modalities.
