"""FastAPI dependency injection providers."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from fastapi import Depends
from sqlalchemy.orm import Session

from defectvision.api.storage import StorageManager, get_storage_manager
from defectvision.db.database import get_db
from defectvision.inference.orchestrator import DefectVisionOrchestrator

DEFAULT_CLASSIFIER_CHECKPOINT = "experiments/defect_classifier_run1/best_model.pt"

_global_orchestrator: DefectVisionOrchestrator | None = None


def get_orchestrator() -> DefectVisionOrchestrator:
    """Provide singleton DefectVisionOrchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        ckpt = Path(DEFAULT_CLASSIFIER_CHECKPOINT)
        ckpt_path = str(ckpt.resolve()) if ckpt.is_file() else None
        _global_orchestrator = DefectVisionOrchestrator(
            classifier_checkpoint_path=ckpt_path
        )
    return _global_orchestrator


def set_orchestrator(orchestrator: DefectVisionOrchestrator | None) -> None:
    """Set or override the global orchestrator (useful for unit tests)."""
    global _global_orchestrator
    _global_orchestrator = orchestrator


def get_storage() -> StorageManager:
    """Provide application storage manager."""
    return get_storage_manager()
