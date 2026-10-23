"""Two-Stage Industrial Visual Defect Inspection Inference Package for DefectVision.

Stage 1: Few-shot PatchCore anomaly detection with product-specific threshold calibration.
Stage 2: Supervised ResNet-18 defect classification (Scratch, Crack, Discoloration, UNKNOWN).
"""

from defectvision.calibration import InspectionStatus
from defectvision.inference.orchestrator import (
    DefectVisionOrchestrator,
    create_orchestrator,
)
from defectvision.inference.product_profile import ProductProfile
from defectvision.inference.result import InspectionResult
from defectvision.inference.stage1_anomaly import Stage1AnomalyDetector, Stage1Result
from defectvision.inference.stage2_classifier import Stage2DefectClassifier, Stage2Result

__all__ = [
    "InspectionStatus",
    "ProductProfile",
    "InspectionResult",
    "Stage1Result",
    "Stage1AnomalyDetector",
    "Stage2Result",
    "Stage2DefectClassifier",
    "DefectVisionOrchestrator",
    "create_orchestrator",
]
