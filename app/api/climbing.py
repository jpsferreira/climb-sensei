"""Climbing Analysis API - Climbing-specific metrics and analysis.

This API provides comprehensive climbing analysis from landmarks.
Uses the composable metrics calculator system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from climb_sensei.services import ClimbingAnalysisService

router = APIRouter(
    prefix="/api/v1/climbing",
    tags=["climbing"],
)

# Global service instance
climbing_service = ClimbingAnalysisService()


class ClimbingAnalysisRequest(BaseModel):
    """Request model for climbing analysis."""

    landmarks_sequence: List[Optional[List[Dict[str, float]]]]
    fps: float = 30.0
    video_path: Optional[str] = None


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_climbing(
    request: ClimbingAnalysisRequest = Body(...),
) -> Dict[str, Any]:
    """Analyze climbing performance from landmarks.

    This endpoint performs comprehensive climbing analysis including:
    - Vertical progression
    - Movement stability
    - Movement efficiency
    - Climbing technique (lock-offs, rest positions)
    - Joint angles

    Args:
        request: Request containing landmarks sequence and FPS

    Returns:
        Comprehensive climbing analysis

    Example:
        ```python
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/climbing/analyze",
            json={
                "landmarks_sequence": [...],
                "fps": 30.0,
                "video_path": "climb.mp4"
            }
        )
        ```
    """
    try:
        analysis = await climbing_service.analyze_async(
            landmarks_sequence=request.landmarks_sequence,
            fps=request.fps,
            video_path=request.video_path,
        )

        # Build response with organized metrics
        response = {
            "status": "success",
            "video_path": analysis.video_path,
            "summary": {
                "total_frames": analysis.summary.total_frames,
                "progression": {
                    "max_height": analysis.summary.max_height,
                    "total_vertical_progress": analysis.summary.total_vertical_progress,
                },
                "movement": {
                    "avg_velocity": analysis.summary.avg_velocity,
                    "max_velocity": analysis.summary.max_velocity,
                    "avg_sway": analysis.summary.avg_sway,
                    "total_distance": analysis.summary.total_distance,
                },
                "efficiency": {
                    "movement_economy": analysis.summary.movement_economy,
                },
                "technique": {
                    "lock_off_count": analysis.summary.lock_off_count,
                    "rest_position_count": analysis.summary.rest_position_count,
                },
            },
            "history_available": list(analysis.history.keys()),
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/metric/{metric_name}", response_model=Dict[str, Any])
async def analyze_specific_metric(
    metric_name: str,
    request: ClimbingAnalysisRequest = Body(...),
) -> Dict[str, Any]:
    """Analyze and return a specific metric.

    Useful when you only need one metric and want to reduce response size.

    Args:
        metric_name: Name of the metric to compute (e.g., 'com_velocity')
        request: Request containing landmarks sequence and FPS

    Returns:
        Specific metric history and summary

    Example:
        ```bash
        curl -X POST \\
            http://localhost:8000/api/v1/climbing/analyze/metric/com_velocity \\
            -H "Content-Type: application/json" \\
            -d '{"landmarks_sequence": [...], "fps": 30.0}'
        ```
    """
    try:
        analysis = await climbing_service.analyze_async(
            landmarks_sequence=request.landmarks_sequence,
            fps=request.fps,
            video_path=request.video_path,
        )

        if metric_name not in analysis.history:
            available = list(analysis.history.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Metric '{metric_name}' not found. Available: {available}",
            )

        return {
            "status": "success",
            "metric_name": metric_name,
            "history": analysis.history[metric_name],
            "statistics": {
                "count": len(analysis.history[metric_name]),
                "values": analysis.history[metric_name],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/metrics/available")
async def get_available_metrics() -> Dict[str, Any]:
    """Get list of all available metrics.

    Returns:
        List of metric names that can be computed

    Example:
        ```bash
        curl http://localhost:8000/api/v1/climbing/metrics/available
        ```
    """
    try:
        metrics = climbing_service.get_available_metrics()

        return {
            "status": "success",
            "total_metrics": len(metrics),
            "metrics": metrics,
            "categories": {
                "stability": [
                    m for m in metrics if "com" in m or "sway" in m or "jerk" in m
                ],
                "progress": [m for m in metrics if "height" in m or "progress" in m],
                "efficiency": [m for m in metrics if "economy" in m or "distance" in m],
                "technique": [
                    m
                    for m in metrics
                    if "lock" in m or "rest" in m or "angle" in m or "span" in m
                ],
                "joints": [
                    m
                    for m in metrics
                    if any(j in m for j in ["elbow", "shoulder", "knee", "hip"])
                ],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.

    Returns:
        Status message
    """
    return {"status": "healthy", "service": "climbing-analysis"}
