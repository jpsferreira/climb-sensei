"""Video Quality API - Standalone video validation endpoint.

This API provides video quality assessment without any climbing-specific logic.
Can be used by any application that needs to validate video files.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from climb_sensei.services import VideoQualityService

router = APIRouter(
    prefix="/api/v1/video-quality",
    tags=["video-quality"],
)

# Global service instance (use dependency injection in production)
video_quality_service = VideoQualityService(default_deep_check=False)

# Storage
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_video_quality(
    file: UploadFile = File(...),
    deep_check: bool = Form(False),
) -> Dict[str, Any]:
    """Analyze video quality.

    This endpoint validates video files for:
    - Format compatibility
    - Resolution quality
    - Frame rate
    - Duration
    - Lighting (if deep_check=True)
    - Stability (if deep_check=True)

    Args:
        file: Video file to analyze
        deep_check: If True, perform frame-by-frame analysis (slower)

    Returns:
        Video quality report with validation results

    Example:
        ```bash
        curl -F "file=@video.mp4" -F "deep_check=true" \\
            http://localhost:8000/api/v1/video-quality/analyze
        ```
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported: mp4, avi, mov, mkv"
        )

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"vq_{file.filename}"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze video quality
        report = await video_quality_service.analyze(
            str(upload_path),
            deep_check=deep_check,
        )

        # Build response
        response = {
            "status": "valid" if report.is_valid else "invalid",
            "is_valid": report.is_valid,
            "file_path": file.filename,
            "properties": {
                "width": report.width,
                "height": report.height,
                "fps": report.fps,
                "frame_count": report.frame_count,
                "duration_seconds": report.duration_seconds,
                "codec": report.codec,
                "file_size_mb": report.file_size_mb,
            },
            "quality_assessment": {
                "resolution": report.resolution_quality,
                "fps": report.fps_quality,
                "duration": report.duration_quality,
                "lighting": report.lighting_quality,
                "stability": report.stability_quality,
            },
            "issues": report.issues,
            "warnings": report.warnings,
            "recommendations": report.recommendations,
        }

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if upload_path.exists():
            upload_path.unlink()


@router.post("/validate", response_model=Dict[str, Any])
async def validate_video(
    file: UploadFile = File(...),
    deep_check: bool = Form(False),
) -> Dict[str, Any]:
    """Validate video and return simple pass/fail.

    Similar to analyze but raises error if invalid.
    Useful for upload validation workflows.

    Args:
        file: Video file to validate
        deep_check: If True, perform frame-by-frame analysis

    Returns:
        Success message if valid

    Raises:
        HTTPException: If video fails validation

    Example:
        ```bash
        curl -F "file=@video.mp4" \\
            http://localhost:8000/api/v1/video-quality/validate
        ```
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Supported: mp4, avi, mov, mkv"
        )

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"vq_{file.filename}"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate (raises ValueError if invalid)
        report = await video_quality_service.validate_or_raise_async(
            str(upload_path),
            deep_check=deep_check,
        )

        return {
            "status": "success",
            "message": "Video is valid and ready for processing",
            "properties": {
                "width": report.width,
                "height": report.height,
                "fps": report.fps,
                "duration_seconds": report.duration_seconds,
            },
        }

    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"Video validation failed: {str(e)}"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if upload_path.exists():
            upload_path.unlink()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Status message
    """
    return {"status": "healthy", "service": "video-quality"}
