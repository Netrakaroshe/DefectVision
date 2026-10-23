# DefectVision: Two-Stage Industrial Surface Defect Inspection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)](https://www.typescriptlang.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end full-stack visual inspection platform engineered for automated industrial surface quality control. DefectVision pairs **few-shot out-of-distribution anomaly detection (PatchCore)** with a **conditional supervised defect classifier (ResNet-18)** and an **interactive operator dashboard** with live camera capture, continuous anomaly scoring, and side-by-side localization heatmaps.

---

## 1. Executive Summary & Problem Statement

In industrial manufacturing (electronics, metal fabrication, pharmaceuticals, and packaging), visual defect inspection faces two competing challenges:
1. **Supervised Defect Classifiers Fail on Novel Anomalies:** Standard CNN classifiers require thousands of labeled defect examples. When presented with unseen defect morphologies or healthy items, they frequently hallucinate incorrect defect classes because they lack an open-world defect-free representation.
2. **Unsupervised Anomaly Detectors Cannot Categorize Defects:** Pure anomaly detectors (like PatchCore or Autoencoders) successfully flag that a part is damaged, but cannot inform quality engineers *what kind* of defect occurred (e.g., whether it is a cosmetic scratch, a structural crack, or chemical discoloration) to trigger corrective action.

**DefectVision solves this through an asymmetric two-stage pipeline:**
* **Stage 1 (Anomaly Gate):** PatchCore extracts multi-scale patch embeddings from a handful of defect-free reference images to build a coreset memory bank. If an item's continuous anomaly score is below a product-specific calibrated threshold, the part is declared **NORMAL** and Stage 2 is bypassed entirely (saving compute and eliminating false alarms).
* **Stage 2 (Semantic Categorization):** Only when an anomaly is verified does the supervised ResNet-18 classifier execute, categorizing the anomaly into actionable industrial classes (**Scratch**, **Crack**, **Discoloration**, or **UNKNOWN**).

```text
Incoming Part Image (Webcam / Upload)
                 │
                 ▼
  ┌───────────────────────────────┐
  │   Product Profile Retrieval   │
  │   (Threshold τ & Memory Bank) │
  └──────────────┬────────────────┘
                 │
                 ▼
  ┌───────────────────────────────┐
  │   STAGE 1: PatchCore Engine   │───► Anomaly Heatmap Localization
  │   (Continuous Anomaly Score)  │
  └──────────────┬────────────────┘
                 │
       Is Score >= Threshold (τ)?
        /                  \
      NO                    YES
     /                        \
    ▼                          ▼
┌──────────────┐     ┌───────────────────────────────────┐
│ Verdict:     │     │ STAGE 2: ResNet-18 Classifier     │
│ NORMAL       │     │ (Scratch / Crack / Discoloration) │
└──────────────┘     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Verdict: DEFECTIVE                │
                     │ + Category & Softmax Confidence   │
                     └─────────────────┬─────────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼                                           ▼
   PostgreSQL / SQLite Database                   React Operator Dashboard
   (Audit Trail & Metric Logging)                 (Verdict Badge, Heatmap, Gauges)
```

---

## 2. Key Features

- **Asymmetric Two-Stage Inference:** Normal parts bypass classification in ~110 ms; defective parts trigger secondary semantic classification in ~135–145 ms steady-state on GPU.
- **Product-Specific Threshold Calibration:** Implements supervised Youden's J / Balanced Accuracy optimization and normal-only $k$-sigma / extreme quantile calibration. Thresholds are calibrated per product profile and never hardcoded globally.
- **Pixel-Level Anomaly Heatmaps:** Anomaly localization heatmaps with automatic peak contour detection and bounding-box extraction.
- **Operator Dashboard (React + TypeScript):** Real-time inspection cockpit with browser webcam streaming (`navigator.mediaDevices`), drag-and-drop file inspection, continuous anomaly gauges, side-by-side heatmap overlays, and paginated audit history.
- **Production REST API (FastAPI):** Asynchronous endpoints for product registration, multi-algorithm calibration, streaming inspection execution, and historical analytics.
- **Audit-Tested Codebase:** Over 12 automated test suites covering database transactions, API contracts, mathematical calibration formulas, and pipeline error recovery.
- **Experimental Dent Detection Pilot:** Includes an isolated Faster R-CNN MobileNetV3-Large object detector trained on the CR7-DET steel dataset, achieving **89.3% AP@50** and **99.5% recall** on surface dents.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2, Python-Multipart |
| **Machine Learning** | PyTorch, Torchvision, Anomalib, Timm, PyTorch Lightning |
| **Vision & Math** | OpenCV (cv2), Pillow, NumPy, SciPy, Scikit-learn, Scikit-image |
| **Database & ORM** | PostgreSQL / SQLite, SQLAlchemy 2.0 |
| **Testing & Tooling** | PyTest, HTTPX, Node Test Runner |

---

## 4. Repository Structure

```text
DefectVision/
├── config/                     # Configuration schemas and profiles
├── data/
│   └── app_storage/            # Runtime storage (uploads, heatmaps, SQLite DB)
├── docs/                       # Architectural specifications and pipeline notes
├── experiments/                # Research experiments, training logs, and reports
│   ├── defect_classifier_run1/ # ResNet-18 training logs, metrics, and report
│   ├── defect_classifier_audit/# Dataset eligibility audit and leakage checks
│   ├── defect_classifier_dataset/# 70/15/15 stratified partition manifests
│   ├── defect_classifier_diagnostics/# Confusion matrices and domain shift analysis
│   ├── dent_detection_pilot/   # Experimental Faster R-CNN pilot and evaluation
│   ├── patchcore_bottle/       # PatchCore baseline on MVTec bottle
│   ├── patchcore_bottle_calibration/# Phase 3 threshold calibration artifacts
│   ├── pill_roi_classifier_v2/ # ROI-localized classifier benchmark
│   └── system_evaluation/      # Multi-domain Phase 8.4 system-level benchmarks
├── frontend/                   # React + TypeScript dashboard (Vite + Tailwind)
│   ├── src/
│   │   ├── components/         # Reusable UI components (HeatmapViewer, Header, etc.)
│   │   ├── pages/              # LiveInspectionPage, CalibrationPage, HistoryPage
│   │   └── services/           # Backend API client integration
│   ├── package.json
│   └── vite.config.ts
├── manifests/                  # Dataset indexing and provenance manifests
├── scripts/                    # Training, calibration, and diagnostic automation
│   ├── build_manifest.py
│   ├── create_classifier_splits.py
│   ├── run_classifier_diagnostics.py
│   ├── run_patchcore_experiment.py
│   ├── run_threshold_calibration_experiment.py
│   └── train_defect_classifier.py
├── src/
│   └── defectvision/           # Production Python package
│       ├── api/                # FastAPI application, routes, and storage manager
│       ├── calibration/        # Mathematical calibration engine (Youden's J, k-sigma)
│       ├── db/                 # SQLAlchemy database models and session factory
│       ├── inference/          # DefectVisionOrchestrator and two-stage pipeline
│       ├── models/             # PatchCore, ResNet-18, and Dent detector definitions
│       └── schemas/            # Pydantic request/response schemas
├── tests/                      # Automated unit and integration test suite
├── .gitignore                  # Git exclusions for models, node_modules, and cache
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 5. Verified Empirical Evaluation Results

All figures below are drawn strictly from frozen evaluation reports, logs, and checkpoints in the repository:

### 5.1 Stage 1: PatchCore Anomaly Detection (MVTec AD `bottle`)

* **Reference Images (Memory Bank):** 20 defect-free training images (`bottle/train/good`).
* **Calibration Split ($N=42$):** 10 normal + 32 defective images (derived $\tau = 41.4696$).
* **Held-Out Test Split ($N=41$):** 10 normal + 31 defective images (evaluated blindly).

| Metric | Measured Value | Evaluation Condition | Evidence File |
| :--- | :---: | :--- | :--- |
| **Image AUROC** | **1.0000** | 83 validation samples | [`experiments/patchcore_bottle/results/summary_metrics.json`](experiments/patchcore_bottle/results/summary_metrics.json) |
| **Separation Margin** | **+2.377** | $\min(\text{Defect}) - \max(\text{Normal})$ | [`experiments/patchcore_bottle/results/summary_metrics.json`](experiments/patchcore_bottle/results/summary_metrics.json) |
| **Held-Out Accuracy** | **100.0%** | 41 held-out test images | [`experiments/patchcore_bottle_calibration/results/evaluation_metrics.json`](experiments/patchcore_bottle_calibration/results/evaluation_metrics.json) |
| **Held-Out Sensitivity**| **100.0% (31/31)** | Zero false negatives | [`experiments/patchcore_bottle_calibration/results/evaluation_metrics.json`](experiments/patchcore_bottle_calibration/results/evaluation_metrics.json) |
| **Held-Out Specificity**| **100.0% (10/10)** | Zero false alarms | [`experiments/patchcore_bottle_calibration/results/evaluation_metrics.json`](experiments/patchcore_bottle_calibration/results/evaluation_metrics.json) |

### 5.2 Stage 2: Supervised ResNet-18 Defect Classifier

* **Dataset Pool:** 2,183 images across 4 source datasets (GC10-DET, NEU-CLS, MT, MVTec AD).
* **Partitions:** 1,505 Train / 333 Validation / 345 Held-Out Test (stratified by class & source).
* **Checkpoint:** [`experiments/defect_classifier_run1/best_model.pt`](experiments/defect_classifier_run1/best_model.pt) (Epoch 15).

| Metric | Overall Test Value | Per-Class Breakdown ($N=345$) | Evidence File |
| :--- | :---: | :--- | :--- |
| **Accuracy** | **91.88%** (317/345) | — | [`experiments/defect_classifier_run1/evaluation_report.md`](experiments/defect_classifier_run1/evaluation_report.md) |
| **Macro F1-Score** | **86.86%** | Scratch: **0.9062** \| Crack: **0.7308** \| Discoloration: **0.9748** \| UNKNOWN: **0.8627** | [`experiments/defect_classifier_diagnostics/diagnostics_report.md`](experiments/defect_classifier_diagnostics/diagnostics_report.md) |
| **Macro Precision** | **85.32%** | Scratch: 0.8923 \| Crack: 0.6333 \| Discoloration: 0.9831 \| UNKNOWN: 0.9041 | [`experiments/defect_classifier_diagnostics/diagnostics_report.md`](experiments/defect_classifier_diagnostics/diagnostics_report.md) |
| **Macro Recall** | **89.40%** | Scratch: 0.9206 \| Crack: 0.8636 \| Discoloration: 0.9667 \| UNKNOWN: 0.8250 | [`experiments/defect_classifier_diagnostics/diagnostics_report.md`](experiments/defect_classifier_diagnostics/diagnostics_report.md) |

### 5.3 Experimental Dent Object Detector (CR7-DET Steel Surface)

* **Architecture:** Faster R-CNN with MobileNetV3-Large FPN (quarantined pilot).
* **Test Split:** 618 images (1,658 total defect annotations, 188 Dent annotations).

| Metric | Measured Value | Scope | Evidence File |
| :--- | :---: | :--- | :--- |
| **Dent AP@50** | **89.29%** | Average precision on `Dent` class | [`experiments/dent_detection_pilot/audit/detection_metrics.json`](experiments/dent_detection_pilot/audit/detection_metrics.json) |
| **Dent Recall** | **99.47% (187/188)**| Only 1 ground-truth dent missed | [`experiments/dent_detection_pilot/audit/detection_metrics.json`](experiments/dent_detection_pilot/audit/detection_metrics.json) |
| **Dent Precision** | **71.92%** | $187 \text{ TP} / 260 \text{ detections}$ | [`experiments/dent_detection_pilot/audit/detection_metrics.json`](experiments/dent_detection_pilot/audit/detection_metrics.json) |
| **7-Class mAP@50** | **64.80%** | Mean AP across all 7 defect classes | [`experiments/dent_detection_pilot/audit/detection_metrics.json`](experiments/dent_detection_pilot/audit/detection_metrics.json) |

---

## 6. Engineering Limitations & Academic Integrity

In keeping with rigorous engineering standards, the following limitations are documented explicitly:

1. **End-to-End Application Accuracy is NOT YET MEASURED:**  
   While individual components have been validated on offline benchmark splits, the complete application pipeline (passing a continuous stream of hundreds of live HTTP requests through FastAPI, database transactions, and browser rendering) has been evaluated on unit sanity samples, but has **not yet undergone a full-scale continuous benchmark**. Individual sub-model metrics (e.g., 91.88% or 100%) must **never** be cited as the overall application accuracy.
2. **Physical Real-World & Lighting Testing is NOT YET MEASURED:**  
   Webcam capture is implemented in the browser UI via `getUserMedia` and HTML5 Canvas frame encoding. However, **no physical testing has been conducted under factory conveyor environments or variable physical lighting conditions**.
3. **Cross-Domain Material Disparity:**  
   Forensic analysis revealed that the ResNet-18 classifier learned subtle source-domain shortcuts (e.g., GC10 steel sheet vs MT ferrite tile). When tested on cross-domain materials (such as ceramic tile), classification accuracy dropped significantly. In production, Stage 2 classification heads should be trained per product material rather than treated as universal.

---

## 7. Getting Started & Installation

### 7.1 Prerequisites
* **Python:** 3.10 or 3.11
* **Node.js:** 18.x or 20.x (with npm)
* **GPU (Recommended):** NVIDIA GPU with CUDA 12+ (CPU inference is fully supported via fallback)

---

### 7.2 Backend Setup (FastAPI & PyTorch)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/DefectVision.git
   cd DefectVision
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables (Optional):**
   By default, DefectVision uses an embedded SQLite database in `data/app_storage/defectvision.db`. To use PostgreSQL, set:
   ```bash
   # Optional PostgreSQL connection:
   export DEFECTVISION_DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/defectvision"
   ```

5. **Start the FastAPI backend server:**
   ```bash
   # Ensure PYTHONPATH includes the src/ directory
   # Windows (PowerShell):
   $env:PYTHONPATH="src"
   uvicorn defectvision.api.app:app --reload --host 127.0.0.1 --port 8000

   # Linux / macOS:
   PYTHONPATH=src uvicorn defectvision.api.app:app --reload --host 127.0.0.1 --port 8000
   ```
   The backend service will be live at `http://127.0.0.1:8000`.

---

### 7.3 Interactive API Documentation (`/docs`)

FastAPI automatically generates interactive OpenAPI Swagger documentation. Once the backend is running, open your browser to:
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### Key API Endpoints:
* `POST /products`: Register a new product inspection profile.
* `POST /products/{product_id}/calibration`: Execute threshold calibration from normal/defective scores.
* `POST /products/{product_id}/inspect`: Execute two-stage inspection via multipart image upload or JSON base64.
* `GET /products/{product_id}/inspections`: Retrieve paginated inspection history and metrics.
* `POST /experimental/dent/detect`: Execute the Faster R-CNN steel dent detector pilot.

---

### 7.4 Frontend Setup (React + Vite)

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Configure Environment:**
   ```bash
   # Copy example environment configuration
   cp .env.example .env
   ```
   Ensure `.env` contains:
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8000
   ```

4. **Launch the development server:**
   ```bash
   npm run dev
   ```
   Open your browser at [http://localhost:5173](http://localhost:5173) to access the inspection dashboard.

---

### 7.5 Running Automated Tests

Run the full automated test suite using `pytest`:
```bash
# Windows (PowerShell)
$env:PYTHONPATH="src"
pytest tests/ -v

# Linux / macOS
PYTHONPATH=src pytest tests/ -v
```

---

## 8. Model Weights & Git LFS Notice

Due to GitHub's file-size limits (50 MB warning, 100 MB hard reject), binary model checkpoints (`*.pt`, `*.ckpt`, `*.pth`) are excluded by default in `.gitignore`.

* **Classifier Checkpoint:** `experiments/defect_classifier_run1/best_model.pt` (128 MB)
* **PatchCore Baseline:** `experiments/patchcore_bottle/staging/.../model.ckpt` (104 MB)
* **Dent Detector Checkpoint:** `experiments/dent_detection_pilot/checkpoints/best_detector.pth` (72 MB)

**To track and commit weights with Git LFS:**
```bash
git lfs install
git lfs track "*.pt" "*.ckpt" "*.pth"
git add .gitattributes
```
Alternatively, checkpoints can be hosted externally on [Hugging Face Hub](https://huggingface.co/) or attached as release artifacts under [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github).

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
