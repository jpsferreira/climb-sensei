"""Shared type aliases and enumerations for climb-sensei.

This module centralizes type definitions used across multiple modules
to avoid repetition and improve semantic clarity.
"""

from enum import StrEnum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Type aliases for landmark data flowing through the pipeline
# ---------------------------------------------------------------------------

LandmarkDict = Dict[str, float]
"""Single landmark with x, y, z, visibility keys."""

LandmarkList = List[LandmarkDict]
"""All 33 landmarks for a single frame."""

LandmarkSequence = List[Optional[LandmarkList]]
"""Landmarks across frames; ``None`` entries indicate frames with no pose detected."""


# ---------------------------------------------------------------------------
# Quality level enumeration (used by video_quality and tracking_quality)
# ---------------------------------------------------------------------------


class QualityLevel(StrEnum):
    """Quality assessment levels for video and tracking checks."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


# ---------------------------------------------------------------------------
# Video processing status (used by database models)
# ---------------------------------------------------------------------------


class VideoStatus(StrEnum):
    """Processing status for uploaded videos."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
