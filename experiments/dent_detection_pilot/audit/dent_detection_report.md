# Offline 7-Class Object Detection Pilot Report: Evaluating the Dent Class

**Project:** `DefectVision`  
**Phase:** Offline Object Detection Pilot (CR7-DET 7-Class Baseline)  
**Target Report File:** `D:\DefectVision\experiments\dent_detection_pilot\audit\dent_detection_report.md`  
**Date:** October 20, 2026  
**Detector Architecture:** Faster R-CNN with MobileNetV3-Large FPN Backbone  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 13.2)  

---

## 1. Executive Summary & Core Finding

By switching from patch classification to **end-to-end object detection**, the need for manual background labeling was **completely eliminated**. The detector learns clean steel background implicitly from the full 400×400 images without requiring a single manual clean crop.

> [!IMPORTANT]
> **Dent Class Detection Performance:**
> * **Dent AP@50 (Average Precision):** **89.3%**
> * **Dent Precision:** **71.9%**
> * **Dent Recall:** **99.5%**
> * **Overall 7-Class Test mAP@50:** **64.8%** (Validation mAP@50: **64.2%**)

### Key Achievements
1. **No Manual Background Labeling Required:** Clean steel background is naturally learned by the detector's Region Proposal Network (RPN) from negative regions across all 4,140 images.
2. **Zero Data Leakage:** All identical duplicate image pairs were strictly grouped by SHA-256 hash. The Dent test set (90 images, 188 dent annotations) is 100% isolated and identical to the Step 2 pilot definition.
3. **Zero Impact on Production:** No production manifests, code, database records, or existing checkpoints were altered. Everything is isolated under `experiments/dent_detection_pilot/`.

---

## 2. Quantitative Performance Breakdown (Test Set: 618 Images)

The model was evaluated on the unseen Test set at standard IoU threshold $= 0.50$:

| Class ID | Defect Class Name | Ground Truths | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1 Score | AP@50 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Inclusion | 275 | 157 | 159 | 118 | 49.7% | 57.1% | 53.1% | 43.9% |
| 1 | **Dents** | 188 | 187 | 73 | 1 | 71.9% | 99.5% | 83.5% | **89.3%** |
| 2 | Oil Spots | 269 | 202 | 223 | 67 | 47.5% | 75.1% | 58.2% | 63.7% |
| 3 | Pits | 289 | 161 | 361 | 128 | 30.8% | 55.7% | 39.7% | 32.2% |
| 4 | Punching | 194 | 193 | 30 | 1 | 86.6% | 99.5% | 92.6% | 88.8% |
| 5 | Linear | 220 | 149 | 220 | 71 | 40.4% | 67.7% | 50.6% | 49.2% |
| 6 | Macular | 223 | 208 | 131 | 15 | 61.4% | 93.3% | 74.0% | 86.5% |
| **Overall** | **7-Class Mean (mAP@50)** | **1658** | **1257** | **1197** | **401** | — | — | — | **64.8%** |

---

## 3. Training Dynamics

| Epoch | Train Loss | Val mAP@50 | Dent AP@50 | Dent Precision | Dent Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 0.0000 | 64.2% | 90.3% | 71.1% | 96.8% |

---

## 4. Architectural Comparison: Object Detection vs. Crop Classification

| Dimension | Previous Crop Classifier Approach | Current Object Detection Approach |
| :--- | :--- | :--- |
| **Negative Background Requirement** | Requires manual cropping and manual human review of hundreds of clean patches. | **Zero manual background crops required.** Negative steel is learned implicitly from unannotated image regions. |
| **Human Effort / Fatigue** | High (reviewer must inspect hundreds of candidate patches). | **Zero human review fatigue.** Direct end-to-end training. |
| **Multi-Defect Handling** | Vulnerable when patches contain edge roll seams or adjacent defects. | **Native.** Can localize multiple bounding boxes with different classes in the same image. |
| **Localization Output** | None (only outputs single label for an extracted crop). | **Exact bounding box $[x_1, y_1, x_2, y_2]$** and class confidence. |
| **Inference in DefectVision** | Would require a separate localization step before classification. | **One-stage detection:** takes raw camera image and outputs localized defects directly. |

---

## 5. Artifacts and Reproducibility

* **Dataset Manifest:** [detection_dataset_manifest.jsonl](file:///D:/DefectVision/experiments/dent_detection_pilot/manifests/detection_dataset_manifest.jsonl)
* **Split Lists:** [train.txt](file:///D:/DefectVision/experiments/dent_detection_pilot/splits/train.txt) (2,901 images), [val.txt](file:///D:/DefectVision/experiments/dent_detection_pilot/splits/val.txt) (621 images), [test.txt](file:///D:/DefectVision/experiments/dent_detection_pilot/splits/test.txt) (618 images)
* **Dataset Loader:** [dataset.py](file:///D:/DefectVision/experiments/dent_detection_pilot/src/dataset.py)
* **Training Script:** [train_eval.py](file:///D:/DefectVision/experiments/dent_detection_pilot/src/train_eval.py)
* **Trained Checkpoint:** [best_detector.pth](file:///D:/DefectVision/experiments/dent_detection_pilot/checkpoints/best_detector.pth)
* **Raw Metrics:** [detection_metrics.json](file:///D:/DefectVision/experiments/dent_detection_pilot/audit/detection_metrics.json)

*(Experiment completed offline; no production assets were modified.)*