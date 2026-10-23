# Dent Pilot Classifier (Step 3) — Feasibility Audit & Execution Report

**Project:** `DefectVision`  
**Phase:** Offline Dent Classification Investigation — Step 3: Pilot Classifier Evaluation  
**Target File:** `D:\DefectVision\experiments\dent_classification\pilot_run1\pilot_report.md`  
**Manifest Audited:** `D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl`  
**Date:** October 19, 2026  
**Auditor / Engineer:** DefectVision AI Pair Programmer  

---

## 1. Executive Summary: Execution Blocker Identified

In accordance with the requirements for **Step 3**, an audit of the isolated research pilot dataset was performed to determine if a binary Dent-versus-background classifier could be trained.

### Formal Blocker Assessment
* **Condition Triggered (Requirement 3 & 10):**
  > *"Train a simple binary Dent-versus-background-crop experiment only if suitable background/non-Dent crops are available from the existing isolated dataset. If not, stop and report that a meaningful binary classifier cannot be trained from Dent-positive crops alone."*  
  > *"If a valid negative class cannot be formed, do not invent one; stop and report the blocker."*
* **Census Finding:** The current isolated research dataset contains **1,242 crops**, of which **1,242 (100.0%) belong to the positive class (`Dents`)**. There are **0 background, normal, or non-Dent crops** in the prepared pilot dataset.
* **Integrity Safeguard (Requirement 4):** Treating other Dent crops as pseudo-negatives is strictly prohibited, as it would violate label truth and destroy discriminative learning.
* **Action Taken:** In strict compliance with instructions, **training was halted prior to model execution**. No synthetic, unverified, or out-of-scope negative class was invented.

---

## 2. Dataset Census of Isolated Research Manifest

Inspection of `D:\DefectVision\experiments\dent_classification\manifests\dent_pilot_manifest.jsonl`:

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Total Manifest Records** | **1,242** | Verified bit-for-bit with crops on disk |
| **Class: `Dents` (Positive)** | **1,242** | 100.0% of isolated dataset |
| **Class: `Background` / `Non-Dent` (Negative)** | **0** | **0.0% — Absent from isolated dataset** |
| **Train Partition** | 866 crops | 100% Dent-positive (420 source images, 414 unique hashes) |
| **Validation Partition** | 188 crops | 100% Dent-positive (90 source images, 89 unique hashes) |
| **Test Partition** | 188 crops | 100% Dent-positive (90 source images, 89 unique hashes) |

---

## 3. Scientific & Methodological Rationale

A binary classification model requires two mutually exclusive classes ($y \in \{0, 1\}$) to formulate a valid cross-entropy optimization objective:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

When $y_i = 1$ for all samples in the training, validation, and test partitions:
1. **Mathematical Degeneracy:** The binary loss objective collapses to $\mathcal{L} = -\frac{1}{N} \sum \log(\hat{y}_i)$. The optimal parameters immediately yield a trivial constant predictor $\hat{y} = 1.0$, regardless of input features.
2. **Evaluation Invalidation:** Precision, recall, and F1 scores cannot be measured for a negative class that contains 0 true instances. Specificity and False Positive Rate are mathematically undefined ($\frac{0}{0}$).
3. **Risk of Ad-Hoc Negative Slicing:** Naively extracting arbitrary image patches from the background of the 600 Dent source images without a verified negative sampling specification risks capturing unannotated roller marks, boundary lighting gradients, or stitch artifacts, contaminating the negative baseline.
4. **Domain Confounding:** Borrowing negative crops from external non-steel datasets (e.g. DefectVision's bottle or pill images) would train the network to distinguish plastic/powder background texture from steel plate texture, rather than learning the 3D indentation morphology of Dents.

---

## 4. Substrate & Domain Limitations

As identified in the Phase 1 and Phase 2 audits:
* **Steel-Specific Domain:** All existing Dent crops depict cold-rolled steel sheets under industrial line-scan lighting.
* **Generalization Boundary:** This dataset cannot establish cross-material generalization. Success on CR7-DET steel imprints does not imply detection capability for dents on blow-molded plastic bottles, pharmaceutical tablets, or ceramic tiles.

---

## 5. Recommended Protocols for Resolving the Blocker

To enable rigorous offline classification experiments, one of the following two approved paths should be authorized:

### Option A: Principled Negative Sampling from Defect-Free Steel Regions (Binary Classifier)
1. Using the verified 600 Dent source images, systematically extract candidate background crops from regions where $\text{IoU} = 0.0$ with all annotated defect boxes, maintaining a safety buffer (e.g., $\ge 20\text{ px}$ away from any defect contour).
2. Sample background crops with dimensions and aspect ratios matched to the distribution of the 1,242 Dent crops.
3. Bind background crops strictly to the source image's SHA-256 hash to maintain 100% split isolation across Train (866 negs), Validation (188 negs), and Test (188 negs).
4. Review candidate background crops on a contact sheet to verify absence of faint roll defects.

### Option B: Five-Class Research Benchmark (Multi-Class Classifier)
1. Integrate audited non-dent classes from the source CR7-DET dataset into an isolated multi-class manifest:
   * Class 1 $\rightarrow$ `Dent`
   * Class 5 $\rightarrow$ `Scratch`
   * Class 6 $\rightarrow$ `Discoloration`
   * Classes 0, 2, 3, 4 $\rightarrow$ `UNKNOWN`
2. Evaluate discriminative multi-class classification where other defect categories serve as natural mutual negatives.

---

## 6. Boundary Constraints & System State Verification

* **Models Trained:** 0 (Halted at feasibility check).
* **Production Code Modified:** None.
* **Production Taxonomy Modified:** None (4-class production taxonomy preserved).
* **Production Manifests Modified:** None (`dataset_manifest.jsonl` untouched).
* **Checkpoints / Database:** Untouched.
* **Artifact Location:** All records saved under `D:\DefectVision\experiments\dent_classification\pilot_run1\`.

**Status:** Paused awaiting user decision on negative class definition protocol.
