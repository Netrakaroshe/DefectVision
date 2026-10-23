"""DefectVision FastAPI Application Factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from defectvision.api.routes import (
    analytics_router,
    calibration_router,
    dent_detection_router,
    inspection_router,
    products_router,
)

from defectvision.api.storage import get_storage_manager
from defectvision.db.database import init_db

logger = logging.getLogger("defectvision.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle management."""
    logger.info("Initializing DefectVision backend services...")
    # Ensure storage paths exist
    get_storage_manager()
    # Initialize database tables if connected
    try:
        init_db()
        logger.info("Database schemas verified.")
    except Exception as e:
        logger.warning(
            f"Database auto-migration skipped on startup (DB may not be online yet): {e}"
        )
    yield
    logger.info("DefectVision backend shutdown complete.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DefectVision Industrial Inspection API",
        description=(
            "Backend and Database foundation for DefectVision two-stage industrial "
            "surface inspection (PatchCore Anomaly Detection + Supervised Defect Classification)."
        ),
        version="0.6.0",
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, str]:
        return {"status": "ok", "service": "defectvision-api"}

    # Include routers
    app.include_router(products_router)
    app.include_router(calibration_router)
    app.include_router(inspection_router)
    app.include_router(analytics_router)
    app.include_router(dent_detection_router)

    return app



app = create_app()
