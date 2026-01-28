"""API layer for ClimbingSensei web application.

This module contains separate API routers for each service domain:
- video_quality: Video validation and quality checking
- tracking_quality: Pose tracking reliability assessment
- climbing: Climbing-specific analysis

Each API is independent and can be deployed separately or combined.
"""

from .video_quality import router as video_quality_router
from .tracking_quality import router as tracking_quality_router
from .climbing import router as climbing_router

__all__ = [
    "video_quality_router",
    "tracking_quality_router",
    "climbing_router",
]
