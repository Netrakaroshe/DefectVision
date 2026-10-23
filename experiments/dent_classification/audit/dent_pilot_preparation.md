# Dent Pilot Dataset Preparation Report

**Project:** `DefectVision`  
**Phase:** Offline Dent Classification Investigation — Step 2: Isolated Dataset Preparation  
**Target File:** `D:\DefectVision\experiments\dent_classification\audit\dent_pilot_preparation.md`  
**Date:** October 19, 2026  
**Auditor / Engineer:** DefectVision AI Pair Programmer  

---

## 1. Executive Summary & Objective

In accordance with **Step 2** of the offline Dent classification investigation, an isolated, leakage-safe **Dent-only pilot dataset** was prepared from the audited CR7-DET download. 

This dataset is strictly isolated under `D:\DefectVision\experiments\dent_classification\` and is intended for offline exploratory classifier benchmarking. **No models were trained, no production code or database was modified, and no production manifests were touched.**

---

## 2. Quantitative Census & Verification Metrics

All metrics below were directly computed from the source dataset and verified on disk:

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Source Images Processed** | **600** | All 600 `dents*.jpg` images from CR7-DET |
| **Unique Image Hashes (SHA-256)** | **592** | 584 singleton images + 8 duplicate clusters |
| **Duplicate Image Groups** | **8** | Comprising 16 source images (8 pairs of identical files) |
| **Total Annotations Evaluated** | **1,242** | All Class `1` (`Dents`) bounding boxes |
| **Invalid Annotations** | **0** | All 1,242 annotations have valid syntax and geometry |
| **Accepted Candidate Crops** | **1,242** | Extracted with 10% modest padding and verified |
| **Rejected Crops** | **0** | No corrupted, out-of-bounds, or sub-5px boxes |
| **Non-Dent Classes Included** | **0** | Classes 0, 2, 3, 4, 5, 6 strictly excluded |

### Split Allocation (Grouped by SHA-256 Hash)

Target ratio was approximately 70% Train / 15% Validation / 15% Test:

| Split | Unique Hashes | Source Images | Candidate Crops | Crop % | Image % | Hash % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 414 | 420 | 866 | 69.73% | 70.00% | 69.93% |
| **Validation** | 89 | 90 | 188 | 15.14% | 15.00% | 15.03% |
| **Test** | 89 | 90 | 188 | 15.14% | 15.00% | 15.03% |
| **Total** | **592** | **600** | **1,242** | **100.0%** | **100.0%** | **100.0%** |

---

## 3. Data Leakage Prevention (Duplicate Grouping)

In the preliminary audit, 8 identical image content pairs were detected within the Dent subset. Because identical images carry slightly jittered manual annotations from separate labeling passes, assigning duplicate copies to different splits would cause severe train-to-test data leakage.

### Deduplication Grouping Verification
Splitting was performed on the **592 unique SHA-256 digests** using a deterministic pseudorandom assignment (`seed=42`). All duplicate copies were guaranteed to land in the same split:

| Duplicate Pair | SHA-256 Prefix | Assigned Split | Leakage Status |
| :--- | :---: | :---: | :---: |
| `dents199.jpg` $\equiv$ `dents202.jpg` | `2ace4ad1e4bc...` | **Train** | **Zero leakage** (Both in Train) |
| `dents203.jpg` $\equiv$ `dents204.jpg` | `1fc099949fed...` | **Train** | **Zero leakage** (Both in Train) |
| `dents218.jpg` $\equiv$ `dents76.jpg`  | `b71a2b7c3eb4...` | **Train** | **Zero leakage** (Both in Train) |
| `dents229.jpg` $\equiv$ `dents30.jpg`  | `52b7972750af...` | **Train** | **Zero leakage** (Both in Train) |
| `dents353.jpg` $\equiv$ `dents364.jpg` | `cce8e2fe2ced...` | **Validation** | **Zero leakage** (Both in Val) |
| `dents427.jpg` $\equiv$ `dents428.jpg` | `85e8d89e023d...` | **Train** | **Zero leakage** (Both in Train) |
| `dents548.jpg` $\equiv$ `dents549.jpg` | `d720b6017b2e...` | **Train** | **Zero leakage** (Both in Train) |
| `dents550.jpg` $\equiv$ `dents551.jpg` | `4c8d5fe5349d...` | **Test** | **Zero leakage** (Both in Test) |

---

## 4. Crop Extraction Protocol

1. **Bounding Box Validation:** Each bounding box was read from the YOLO label file (`1 <xc> <yc> <w> <h>`). All coordinates were checked for positive dimensions ($w > 0, h > 0$) and boundary enclosure. Floating-point rounding at the image perimeter (within $5 \times 10^{-7}$) was accounted for without altering raw annotation references.
2. **Modest Padding:** To provide the classifier with localized contextual margins around defect contours without incorporating excessive background noise, proportional padding of **10%** of box width and height ($\min = 2\text{ px}$) was applied:
   $$\text{crop\_x}_1 = \max(0, \; \lfloor x_1 - \max(2, 0.10 \times w_{\text{box}}) \rfloor)$$
   $$\text{crop\_x}_2 = \min(W, \; \lceil x_2 + \max(2, 0.10 \times w_{\text{box}}) \rceil)$$
3. **Storage & Format:** Crops were written as lossless PNG files into split-segregated directories: `crops/train/`, `crops/val/`, and `crops/test/`.
4. **Traceability:** Every crop file preserves its lineage via its filename `dent_{stem}_box{idx}_{counter}.png` and via full relational fields in the pilot manifest.

---

## 5. Artifacts Created

All artifacts reside strictly under `D:\DefectVision\experiments\dent_classification\`:

```text
D:\DefectVision\experiments\dent_classification\
├── audit\
│   ├── cr7_det_audit.md                    # Initial audit report (Step 1)
│   └── dent_pilot_preparation.md           # Dataset preparation report (Step 2)
├── manifests\
│   ├── dent_pilot_manifest.jsonl           # Primary JSONL research manifest (1,242 records)
│   └── dent_pilot_manifest.csv             # Tabular CSV manifest for rapid inspection (1,242 rows)
├── crops\
│   ├── train\                              # 866 PNG candidate crops
│   ├── val\                                # 188 PNG candidate crops
│   └── test\                               # 188 PNG candidate crops
└── contact_sheets\
    └── dent_crops_contact_sheet.png        # 16-sample diverse contact sheet for visual review
```

### Manifest Schema Reference (`dent_pilot_manifest.jsonl`)
```json
{
  "crop_id": "dent_dents1_box0_0001",
  "source_image_path": "C:/Users/susha/Downloads/CR7-DET/CR7-DET/image/dents1.jpg",
  "source_image_stem": "dents1",
  "sha256": "4cfae...",
  "original_label_id": 1,
  "original_label_name": "Dents",
  "box_index": 0,
  "bbox_norm": [0.745625, 0.8865, 0.41225, 0.227],
  "bbox_pixels": [215.8, 309.2, 380.7, 400.0],
  "crop_pixels": [199, 300, 398, 400],
  "crop_dimensions": [199, 100],
  "aspect_ratio": 1.99,
  "padding_ratio": 0.1,
  "split": "train",
  "validation_status": "ACCEPTED",
  "crop_path": "crops/train/dent_dents1_box0_0001.png",
  "crop_abs_path": "D:/DefectVision/experiments/dent_classification/crops/train/dent_dents1_box0_0001.png",
  "provenance": "CR7-DET cold-rolled steel surface defect dataset; class 1 (Dents)"
}
```

---

## 6. Contact Sheet & Manual Review Protocol

A contact sheet featuring **16 diverse candidate crops** was synthesized and saved to [dent_crops_contact_sheet.png](file:///D:/DefectVision/experiments/dent_classification/contact_sheets/dent_crops_contact_sheet.png).

> [!CAUTION]
> **Semantic Correctness Disclaimer:**
> In accordance with project instructions, **these candidate crops are NOT certified as semantically correct until manual visual review is conducted.** The bounding boxes reflect raw annotator inputs from CR7-DET.

### Selection Strategy for the 16 Review Samples:
* **Area Extremes:** Includes the smallest crops (e.g. `dents71 b1`: $55 \times 26$ px, `dents451 b1`: $56 \times 29$ px) and the largest crops (e.g. `dents338 b1`: $287 \times 232$ px, `dents274 b1`: $270 \times 249$ px).
* **Aspect Ratio Extremes:** Includes extremely tall vertical strip grooves (e.g. `dents457 b1`: $28 \times 175$ px, aspect ratio $0.16$) and wide horizontal grooves (e.g. `dents161 b0`: $187 \times 23$ px, aspect ratio $8.13$).
* **Split Representation:** Balanced across Train, Validation, and Test sets.
* **Duplicate Consistency:** Includes crops from duplicate pair `dents199` and `dents202` for manual comparison of annotation jitter.

---

## 7. Observations & Unresolved Issues for Review

1. **Extreme Aspect Ratio Range:**
   * Measured aspect ratios range from **$0.16$** (narrow vertical lines) to **$8.13$** (wide horizontal bands). Standard square CNN transforms ($224 \times 224$ with naive bilinear stretching) will introduce heavy geometric distortion. An aspect-ratio preserving letterbox transform or RoIAlign feature pooling should be considered for subsequent classification modeling.
2. **Periodic Roll Mill Patterns:**
   * Multiple crops depict recurring indentations spaced regularly along the steel rolling direction. While typical of cold-rolled steel defect physics, the classifier could memorize the repetition frequency rather than localized 3D dent morphology.
3. **Domain Narrowness:**
   * All 1,242 crops depict cold-rolled steel surfaces. They do not represent packaging plastics or pharmaceutical powders. Transferability to DefectVision's production products remains an open research question for downstream evaluation.

---

## 8. Verification & Next Steps

All checks needed to verify dataset preparation were completed:
* 100% of 1,242 crop files were verified readable and uncorrupted on disk.
* 100% of records in `dent_pilot_manifest.jsonl` match the files on disk.
* Zero modifications occurred outside `experiments/dent_classification/`.

**Current Status:** Dataset preparation is complete. Execution has stopped awaiting manual review of the contact sheet and authorization for subsequent offline benchmarking steps.
