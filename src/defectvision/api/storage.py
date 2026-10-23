"""Application-managed file storage and validation for DefectVision.

Handles:
- Isolated directory structure for uploaded query images, inspection artifacts, and calibration records.
- Strict image format and integrity verification.
- Ensuring raw dataset directories remain completely read-only and authoritative.
"""

from __future__ import annotations

import os
from pathlib import Path
import uuid

from PIL import Image

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
DEFAULT_STORAGE_ROOT = Path("data/app_storage")


class StorageManager:
    """Manages application-owned directories for incoming images and artifacts."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is not None:
            self.root = Path(root_dir).resolve()
        else:
            env_dir = os.getenv("DEFECTVISION_STORAGE_ROOT")
            self.root = Path(env_dir).resolve() if env_dir else Path(DEFAULT_STORAGE_ROOT).resolve()

        self.uploads_dir = self.root / "uploads"
        self.inspections_dir = self.root / "inspections"
        self.heatmaps_dir = self.root / "heatmaps"
        self.calibrations_dir = self.root / "calibrations"

        # Ensure directory structure exists
        for d in (self.uploads_dir, self.inspections_dir, self.heatmaps_dir, self.calibrations_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_image_file(file_path: str | Path) -> Path:
        """Validate image existence, extension, and decodable data integrity.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file has unsupported extension or cannot be decoded as an image.
        """
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")

        if p.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{p.suffix}'. "
                f"Supported extensions: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            )

        try:
            with Image.open(p) as img:
                img.verify()
            # Double check decode
            with Image.open(p) as img:
                img.load()
        except Exception as e:
            raise ValueError(f"Image integrity validation failed for {p.name}: {e}") from e

        return p

    def save_uploaded_image(
        self,
        file_bytes: bytes,
        original_filename: str,
        product_id: str,
    ) -> Path:
        """Validate and store an uploaded image safely inside the application storage."""
        ext = Path(original_filename).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported upload extension '{ext}'. "
                f"Supported: {sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            )

        prod_dir = self.uploads_dir / product_id
        prod_dir.mkdir(parents=True, exist_ok=True)

        unique_name = f"{uuid.uuid4().hex[:12]}_{Path(original_filename).name}"
        dest_path = prod_dir / unique_name

        dest_path.write_bytes(file_bytes)

        # Validate saved image
        try:
            self.validate_image_file(dest_path)
        except Exception:
            if dest_path.exists():
                dest_path.unlink()
            raise

        return dest_path


_default_storage_manager: StorageManager | None = None


def get_storage_manager() -> StorageManager:
    """Retrieve or initialize singleton StorageManager."""
    global _default_storage_manager
    if _default_storage_manager is None:
        _default_storage_manager = StorageManager()
    return _default_storage_manager
