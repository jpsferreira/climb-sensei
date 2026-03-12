"""Tracking Quality API - Pose tracking reliability assessment.

This API analyzes the quality of pose tracking from landmarks.
Can work with pre-extracted landmarks OR process video directly.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from climb_sensei.services import TrackingQualityService

router = APIRouter(
    prefix="/api/v1/tracking-quality",
    tags=["tracking-quality"],
)

# Global service instance
tracking_quality_service = TrackingQualityService()

# Storage
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


class LandmarksRequest(BaseModel):
    """Request model for landmarks-based analysis."""

    landmarks_sequence: List[Optional[List[Dict[str, float]]]]
    video_path: Optional[str] = "unknown"


@router.post("/analyze-from-landmarks", response_model=Dict[str, Any])
async def analyze_from_landmarks(
    request: LandmarksRequest = Body(...),
) -> Dict[str, Any]:
    """Analyze tracking quality from pre-extracted landmarks.

    This is the preferred endpoint when landmarks have already been extracted.
    It's fast and doesn't require video file upload.

    Args:
        request: Request containing landmarks sequence and optional video path

    Returns:
        Tracking quality report

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/tracking-quality/analyze-from-landmarks",
            json={
                "landmarks_sequence": [...],  # Your landmarks
                "video_path": "my_video.mp4"
            }
        )
        ```
    """
    try:
        report = await tracking_quality_service.analyze_from_landmarks_async(
            landmarks_sequence=request.landmarks_sequence,
            video_path=request.video_path,
        )

        return {
            "status": "success",
            "is_trackable": report.is_trackable,
            "quality_level": report.quality_level,
            "metrics": {
                "total_frames": report.total_frames,
                "frames_with_pose": report.frames_with_pose,
                "detection_rate": report.detection_rate,
                "avg_landmark_confidence": report.avg_landmark_confidence,
                "min_landmark_confidence": report.min_landmark_confidence,
                "avg_visibility_score": report.avg_visibility_score,
                "tracking_smoothness": report.tracking_smoothness,
                "tracking_loss_events": report.tracking_loss_events,
            },
            "issues": report.issues,
            "warnings": report.warnings,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-from-video", response_model=Dict[str, Any])
async def analyze_from_video(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Analyze tracking quality by processing video file.

    This extracts landmarks and analyzes tracking quality in one pass.
    Use when you only need tracking quality and don't have pre-extracted landmarks.

    Args:
        file: Video file to analyze

    Returns:
        Tracking quality report

    Example:
        ```bash
        curl -F "file=@video.mp4" \\
            http://localhost:8000/api/v1/tracking-quality/analyze-from-video
        ```
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported: mp4, avi, mov, mkv"
        )

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"tq_{file.filename}"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze tracking quality
        report = await tracking_quality_service.analyze_from_video_async(
            str(upload_path),
        )

        return {
            "status": "success",
            "is_trackable": report.is_trackable,
            "quality_level": report.quality_level,
            "metrics": {
                "total_frames": report.total_frames,
                "frames_with_pose": report.frames_with_pose,
                "detection_rate": report.detection_rate,
                "avg_landmark_confidence": report.avg_landmark_confidence,
                "min_landmark_confidence": report.min_landmark_confidence,
                "avg_visibility_score": report.avg_visibility_score,
                "tracking_smoothness": report.tracking_smoothness,
                "tracking_loss_events": report.tracking_loss_events,
            },
            "issues": report.issues,
            "warnings": report.warnings,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if upload_path.exists():
            upload_path.unlink()


@router.post("/validate", response_model=Dict[str, Any])
async def validate_tracking(
    request: LandmarksRequest = Body(...),
) -> Dict[str, Any]:
    """Validate tracking quality and raise error if poor.

    Args:
        request: Request containing landmarks sequence

    Returns:
        Success message if tracking is acceptable

    Raises:
        HTTPException: If tracking quality is insufficient
    """
    try:
        report = await tracking_quality_service.validate_or_raise_async(
            landmarks_sequence=request.landmarks_sequence,
            video_path=request.video_path,
        )

        return {
            "status": "success",
            "message": "Tracking quality is acceptable",
            "quality_level": report.quality_level,
            "detection_rate": report.detection_rate,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Tracking validation failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Status message
    """
    return {"status": "healthy", "service": "tracking-quality"}
