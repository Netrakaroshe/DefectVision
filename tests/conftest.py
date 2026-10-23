"""Global pytest configuration and fixtures for DefectVision."""

from pathlib import Path
import sys

# Ensure src/ is on sys.path for test discovery
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
