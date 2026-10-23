"""DefectVision API routes package."""

from defectvision.api.routes.analytics import router as analytics_router
from defectvision.api.routes.calibration import router as calibration_router
from defectvision.api.routes.dent_detection import router as dent_detection_router
from defectvision.api.routes.inspection import router as inspection_router
from defectvision.api.routes.products import router as products_router

__all__ = [
    "products_router",
    "calibration_router",
    "inspection_router",
    "analytics_router",
    "dent_detection_router",
]

