# DefectVision Two-Stage Inference Pipeline

## Overview

The DefectVision two-stage inference pipeline connects PatchCore anomaly detection with a supervised ResNet-18 defect classifier.

```text
NEW PRODUCT IMAGE
        |
        v
PRODUCT PROFILE (Threshold + Memory Bank)
        |
        v
STAGE 1: PATCHCORE ANOMALY DETECTION
        |
        +--> Continuous Anomaly Score
        +--> Localization Heatmap
        |
        v
PRODUCT-SPECIFIC CALIBRATED THRESHOLD
        |
        +---- NORMAL (Score < Threshold)
        |       -> Final Verdict: NORMAL
        |       -> Stage 2 Skipped
        |
        +---- DEFECTIVE (Score >= Threshold)
                |
                v
        STAGE 2: DEFECT CLASSIFIER (ResNet-18)
                |
                +--> Scratch
                +--> Crack
                +--> Discoloration
                +--> UNKNOWN
                +--> Softmax Probabilities & Confidence
```

---

## Architecture & Responsibilities

### Stage 1: Anomaly Detection & Calibration
* **Responsibility**: Ingests the query image, queries the product's pre-computed PatchCore feature memory bank, extracts patch embeddings, and calculates continuous raw anomaly scores and 2D anomaly localization maps.
* **Decision Rule**:
  $$\text{score} < \tau_{\text{product}} \implies \text{NORMAL}$$
  $$\text{score} \ge \tau_{\text{product}} \implies \text{DEFECTIVE}$$
* **Threshold Origin**: Derived empirically during product calibration (Phase 3) via Youden's J statistic / Balanced Accuracy optimization (supervised) or $k$-sigma / quantile upper bounds (normal-only). It is stored in the `ProductProfile` and never hardcoded universally.

### Stage 2: Supervised Defect Classification
* **Responsibility**: Invoked **only** when Stage 1 determines that an anomaly is present (`DEFECTIVE`). Passes the image through the trained ResNet-18 industrial classifier to categorize the anomaly into the approved 4-class taxonomy: `Scratch`, `Crack`, `Discoloration`, or `UNKNOWN`.
* **Why Stage 2 Only Runs After DEFECTIVE**:
  1. *Computational Efficiency*: The vast majority of production line parts are normal; invoking a secondary classification network on healthy parts wastes compute cycles and increases latency.
  2. *False Positive Prevention*: Supervised classifiers forced to classify clean, normal images will hallucinate a defect category because standard multi-class heads do not have an open-world "defect-free" training partition. Stage 1 acts as a gate protecting Stage 2.
  3. *Separation of Concerns*: PatchCore excels at out-of-distribution anomaly localization with minimal normal references; the CNN excels at fine-grained semantic categorization among known defect types.

---

## Technical Scope & Intentional Non-Implementations

1. **Full Image Classification (No Heatmap Cropping)**:
   * Stage 2 currently classifies using the full RGB frame. Heatmap bounding-box cropping is intentionally **not** implemented because the defect classifier was trained and validated on full image crops; cropping dynamically would cause domain-shift degradation.
2. **Raw Softmax Confidence (No Production Gate Yet)**:
   * The 0.75 confidence threshold evaluated in diagnostics remains an offline analytical study and is not applied automatically.
3. **No 3D / Depth / Severity Estimation**:
   * Physical dimensions, defect depths, and severity scores are not part of the 2D visual inspection system.
4. **Decoupled from Frameworks**:
   * The inference pipeline does not import FastAPI, databases, or UI libraries, ensuring pure Python embeddability.

---

## Integration with Future FastAPI Layer

The future FastAPI backend can instantiate `DefectVisionOrchestrator` once at application startup (lifespan event) and invoke it concurrently across endpoints:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from defectvision.inference import create_orchestrator, ProductProfile

# Lifespan: load models once in memory
orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = create_orchestrator(
        classifier_checkpoint_path="experiments/defect_classifier_run1/best_model.pt"
    )
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/api/v1/inspect/{product_id}")
async def inspect_item(product_id: str, file: UploadFile):
    # 1. Load product profile from database / file
    profile = ProductProfile.load_json(f"profiles/{product_id}.json")
    
    # 2. Save incoming file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    # 3. Perform two-stage inspection
    result = orchestrator.inspect(
        image_path=temp_path,
        product_profile=profile,
        save_heatmap=True,
    )
    
    return result.to_dict()
```
