"""Upload pipeline functions.

Stateless functions for the video upload and analysis pipeline.
Each step takes explicit parameters and returns explicit results.
"""

import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from climb_sensei.config import CLIMBING_CONNECTIONS, CLIMBING_LANDMARKS
from climb_sensei.models import ClimbingAnalysis
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.services import (
    ClimbingAnalysisService,
    TrackingQualityService,
    VideoQualityService,
)
from climb_sensei.types import VideoStatus
from climb_sensei.video_io import VideoReader, VideoWriter
from climb_sensei.viz import draw_pose_landmarks

from climb_sensei.database.models import Analysis, ProgressMetric, Video

logger = logging.getLogger(__name__)

# Directories
_APP_DIR = Path(__file__).parent.parent
UPLOAD_DIR = _APP_DIR / "uploads"
OUTPUT_DIR = _APP_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Documentation base URL
BASE_DOC_URL = "https://jpsferreira.github.io/climb-sensei/metrics/"

# Maximum upload size (default 500MB)
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def validate_video_magic_bytes(file_path: Path) -> bool:
    """Validate that a file's magic bytes match a known video format."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
        if len(header) < 12:
            return False
        # MP4/MOV: "ftyp" at offset 4
        if header[4:8] == b"ftyp":
            return True
        # MKV/WebM: EBML header
        if header[0:4] == b"\x1a\x45\xdf\xa3":
            return True
        # AVI: RIFF container with AVI marker at offset 8
        if header[0:4] == b"RIFF" and header[8:12] == b"AVI ":
            return True
    except OSError:
        pass
    return False


def save_upload(file, analysis_id: str) -> Path:
    """Save uploaded file to disk with sanitized filename and size limit.

    Args:
        file: FastAPI UploadFile
        analysis_id: Unique analysis identifier

    Returns:
        Path to the saved file

    Raises:
        ValueError: If file extension is not allowed or path is invalid
        HTTPException: If file exceeds size limit
    """
    from fastapi import HTTPException

    # Sanitize: extract only the extension from the original filename
    original_name = PurePosixPath(file.filename or "video.mp4").name
    suffix = PurePosixPath(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Invalid file type. Please upload a video file.")

    safe_name = f"{analysis_id}{suffix}"
    upload_path = UPLOAD_DIR / safe_name

    # Belt-and-suspenders: verify resolved path is inside UPLOAD_DIR
    if not upload_path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise ValueError("Invalid upload path")

    # Write with size limit enforcement
    bytes_written = 0
    with open(upload_path, "wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_SIZE:
                buffer.close()
                upload_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB.",
                )
            buffer.write(chunk)

    # Validate magic bytes
    if not validate_video_magic_bytes(upload_path):
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid video. Upload a real video file.",
        )

    return upload_path


def create_video_record(
    db: Session, user_id: int, filename: str, upload_path: Path
) -> Video:
    """Create and flush a Video database record.

    Args:
        db: Database session
        user_id: Owner user ID
        filename: Original filename
        upload_path: Path to saved file

    Returns:
        Video record with id assigned
    """
    video_record = Video(
        user_id=user_id,
        filename=filename,
        file_path=str(upload_path),
        status=VideoStatus.PROCESSING,
    )
    db.add(video_record)
    db.flush()
    return video_record


def extract_landmarks(
    upload_path: Path,
) -> Tuple[List, List, int, float]:
    """Extract pose landmarks from video (expensive MediaPipe pass).

    Args:
        upload_path: Path to video file

    Returns:
        Tuple of (landmarks_history, pose_results_history, frame_count, fps)
    """
    landmarks_history = []
    pose_results_history = []
    frame_count = 0
    fps = 30.0

    with PoseEngine() as pose_engine:
        with VideoReader(str(upload_path)) as video:
            fps = video.fps

            while True:
                success, frame = video.read()
                if not success:
                    break

                pose_result = pose_engine.process(frame)

                if pose_result and pose_result.pose_landmarks:
                    landmarks = pose_engine.extract_landmarks(pose_result)
                    landmarks_history.append(landmarks)
                    pose_results_history.append(pose_result)
                    frame_count += 1
                else:
                    landmarks_history.append(None)
                    pose_results_history.append(None)

    logger.info(
        "Extraction complete: %d total frames, %d with pose",
        len(landmarks_history),
        frame_count,
    )
    return landmarks_history, pose_results_history, frame_count, fps


def check_video_quality(upload_path: Path) -> Dict[str, Any]:
    """Run video quality checks.

    Args:
        upload_path: Path to video file

    Returns:
        Video quality report dict

    Raises:
        ValueError: If video quality validation fails
    """
    service = VideoQualityService()
    report = service.analyze_sync(str(upload_path))

    if not report.is_valid:
        error_msg = "Video quality validation failed:\n"
        for issue in report.issues:
            error_msg += f"  - {issue}\n"
        raise ValueError(error_msg.rstrip())

    return {
        "report": report,
        "result": {
            "is_valid": bool(report.is_valid),
            "resolution": f"{report.width}x{report.height}",
            "resolution_quality": str(report.resolution_quality),
            "fps_quality": str(report.fps_quality),
            "duration": f"{float(report.duration_seconds):.1f}s",
            "issues": list(report.issues),
            "warnings": list(report.warnings),
        },
    }


def check_tracking_quality(
    landmarks_history: List, upload_path: Path
) -> Dict[str, Any]:
    """Run tracking quality analysis.

    Args:
        landmarks_history: Extracted landmark sequence
        upload_path: Path to video file

    Returns:
        Tracking quality report dict
    """
    service = TrackingQualityService()
    report = service.analyze_from_landmarks(
        landmarks_history, video_path=str(upload_path)
    )

    return {
        "report": report,
        "result": {
            "is_trackable": bool(report.is_trackable),
            "quality_level": str(report.quality_level),
            "detection_rate": float(report.detection_rate),
            "tracking_smoothness": round(float(report.tracking_smoothness), 4),
            "avg_confidence": round(float(report.avg_landmark_confidence), 4),
            "tracking_loss_events": int(report.tracking_loss_events),
            "warnings": list(report.warnings),
        },
    }


def run_analysis(
    landmarks_history: List,
    fps: float,
    upload_path: Path,
    video_quality_report: Optional[Any] = None,
    tracking_quality_report: Optional[Any] = None,
) -> ClimbingAnalysis:
    """Run climbing metrics analysis.

    Args:
        landmarks_history: Extracted landmark sequence
        fps: Video frames per second
        upload_path: Path to video file
        video_quality_report: Optional quality report
        tracking_quality_report: Optional tracking report

    Returns:
        ClimbingAnalysis result
    """
    service = ClimbingAnalysisService()
    return service.analyze(
        landmarks_history,
        fps=fps,
        video_path=str(upload_path),
        video_quality=video_quality_report,
        tracking_quality=tracking_quality_report,
    )


def build_metrics_response(analysis: ClimbingAnalysis, fps: float) -> Dict[str, Any]:
    """Build categorized metrics response from analysis.

    Args:
        analysis: Completed climbing analysis
        fps: Video frames per second

    Returns:
        Metrics dict with categories for JSON response
    """
    summary = analysis.summary
    history = analysis.history

    def _doc_url(metric_key: str) -> str:
        anchor = metric_key.replace("_", "-")
        return f"{BASE_DOC_URL}#{anchor}"

    def _get_metric_value(key: str) -> float:
        if key not in history:
            return 0.0
        vals = [float(v) for v in history[key] if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    categories = {
        "overview": {
            "title": "Overview",
            "metrics": {
                "total_frames": {
                    "label": "Total Frames",
                    "value": int(summary.total_frames),
                    "unit": "",
                },
                "duration": {
                    "label": "Duration",
                    "value": round(int(summary.total_frames) / float(fps), 1),
                    "unit": "s",
                },
            },
        },
        "movement": {
            "title": "Movement & Velocity",
            "metrics": {
                "vertical_progress": {
                    "label": "Total Vertical Progress",
                    "value": round(float(summary.total_vertical_progress), 3),
                    "unit": "",
                    "doc": _doc_url("vertical_progress"),
                },
                "max_height": {
                    "label": "Maximum Height",
                    "value": round(float(summary.max_height), 3),
                    "unit": "",
                    "doc": _doc_url("max_height"),
                },
                "avg_velocity": {
                    "label": "Average Velocity",
                    "value": round(float(summary.avg_velocity), 4),
                    "unit": "u/s",
                    "doc": _doc_url("com_velocity"),
                },
                "max_velocity": {
                    "label": "Maximum Velocity",
                    "value": round(float(summary.max_velocity), 4),
                    "unit": "u/s",
                    "doc": _doc_url("com_velocity"),
                },
                "total_distance": {
                    "label": "Total Distance Traveled",
                    "value": round(float(summary.total_distance_traveled), 3),
                    "unit": "",
                    "doc": _doc_url("total_distance_traveled"),
                },
            },
        },
        "stability": {
            "title": "Stability & Control",
            "metrics": {
                "avg_sway": {
                    "label": "Average Lateral Sway",
                    "value": round(float(summary.avg_sway), 4),
                    "unit": "",
                    "doc": _doc_url("com_sway"),
                },
                "max_sway": {
                    "label": "Maximum Sway",
                    "value": round(float(summary.max_sway), 4),
                    "unit": "",
                    "doc": _doc_url("com_sway"),
                },
                "avg_jerk": {
                    "label": "Average Jerk (Smoothness)",
                    "value": round(float(summary.avg_jerk), 2),
                    "unit": "",
                    "doc": _doc_url("com_jerk"),
                },
                "max_jerk": {
                    "label": "Maximum Jerk",
                    "value": round(float(summary.max_jerk), 2),
                    "unit": "",
                    "doc": _doc_url("com_jerk"),
                },
            },
        },
        "positioning": {
            "title": "Body Positioning",
            "metrics": {
                "avg_body_angle": {
                    "label": "Average Body Angle",
                    "value": round(float(summary.avg_body_angle), 1),
                    "unit": "\u00b0",
                    "doc": _doc_url("body_angle"),
                },
                "avg_hand_span": {
                    "label": "Average Hand Span",
                    "value": round(float(summary.avg_hand_span), 3),
                    "unit": "",
                    "doc": _doc_url("hand_span"),
                },
                "avg_foot_span": {
                    "label": "Average Foot Span",
                    "value": round(float(summary.avg_foot_span), 3),
                    "unit": "",
                    "doc": _doc_url("foot_span"),
                },
            },
        },
        "efficiency": {
            "title": "Efficiency & Technique",
            "metrics": {
                "movement_economy": {
                    "label": "Movement Economy",
                    "value": round(float(summary.avg_movement_economy), 3),
                    "unit": "",
                    "doc": _doc_url("movement_economy"),
                },
                "lock_off_count": {
                    "label": "Lock-offs Detected",
                    "value": int(summary.lock_off_count),
                    "unit": "",
                    "doc": _doc_url("lock_off_count"),
                },
                "rest_count": {
                    "label": "Rest Positions",
                    "value": int(summary.rest_count),
                    "unit": "",
                    "doc": _doc_url("rest_count"),
                },
            },
        },
        "biomechanics": {
            "title": "Joint Angles & Biomechanics",
            "metrics": {
                "left_elbow_angle": {
                    "label": "Left Elbow Angle",
                    "value": _get_metric_value("left_elbow"),
                    "unit": "\u00b0",
                    "doc": _doc_url("left_elbow_angle"),
                },
                "right_elbow_angle": {
                    "label": "Right Elbow Angle",
                    "value": _get_metric_value("right_elbow"),
                    "unit": "\u00b0",
                    "doc": _doc_url("right_elbow_angle"),
                },
                "left_shoulder_angle": {
                    "label": "Left Shoulder Angle",
                    "value": _get_metric_value("left_shoulder"),
                    "unit": "\u00b0",
                    "doc": _doc_url("left_shoulder_angle"),
                },
                "right_shoulder_angle": {
                    "label": "Right Shoulder Angle",
                    "value": _get_metric_value("right_shoulder"),
                    "unit": "\u00b0",
                    "doc": _doc_url("right_shoulder_angle"),
                },
                "left_knee_angle": {
                    "label": "Left Knee Angle",
                    "value": _get_metric_value("left_knee"),
                    "unit": "\u00b0",
                    "doc": _doc_url("left_knee_angle"),
                },
                "right_knee_angle": {
                    "label": "Right Knee Angle",
                    "value": _get_metric_value("right_knee"),
                    "unit": "\u00b0",
                    "doc": _doc_url("right_knee_angle"),
                },
                "left_hip_angle": {
                    "label": "Left Hip Angle",
                    "value": _get_metric_value("left_hip"),
                    "unit": "\u00b0",
                    "doc": _doc_url("left_hip_angle"),
                },
                "right_hip_angle": {
                    "label": "Right Hip Angle",
                    "value": _get_metric_value("right_hip"),
                    "unit": "\u00b0",
                    "doc": _doc_url("right_hip_angle"),
                },
            },
        },
    }

    return {
        "categories": categories,
        "total_frames": int(summary.total_frames),
    }


def generate_annotated_video(
    upload_path: Path,
    analysis_id: str,
    pose_results_history: List,
    fps: float,
) -> str:
    """Generate annotated video with pose overlay only.

    Metrics/plots are displayed in the web app UI, not composited onto
    the video.  Output uses VP8/WebM for smaller file sizes.

    Args:
        upload_path: Path to original video
        analysis_id: Unique analysis ID
        pose_results_history: Cached pose results
        fps: Video FPS

    Returns:
        Output video URL path
    """
    output_video_path = OUTPUT_DIR / f"{analysis_id}_output.webm"

    output_frames = []
    output_dims = None

    with VideoReader(str(upload_path)) as reader:
        for pose_result in pose_results_history:
            success, frame = reader.read()
            if not success:
                break

            if pose_result is not None:
                annotated = draw_pose_landmarks(
                    frame,
                    pose_result,
                    connections=CLIMBING_CONNECTIONS,
                    landmarks_to_draw=CLIMBING_LANDMARKS,
                )

                if output_dims is None:
                    h, w = annotated.shape[:2]
                    output_dims = (w, h)

                output_frames.append(annotated)

    if output_dims is not None:
        w, h = output_dims
        with VideoWriter(
            str(output_video_path), fps=fps, width=w, height=h, fourcc="VP80"
        ) as writer:
            for output_frame in output_frames:
                writer.write(output_frame)

    return f"/outputs/{analysis_id}_output.webm"


def persist_results(
    db: Session,
    video_record: Video,
    analysis: Optional[ClimbingAnalysis],
    results: Dict[str, Any],
    run_metrics: bool,
    run_video: bool,
    run_quality: bool,
    dashboard_position: str,
    session_id: Optional[int],
    user_id: int,
) -> int:
    """Persist analysis results to database.

    Fixes the latent bug where summary_dict was always empty by reading
    from the ClimbingAnalysis summary object directly.

    Args:
        db: Database session
        video_record: Video record to update
        analysis: Optional ClimbingAnalysis result
        results: Full results dict for JSON storage
        run_metrics: Whether metrics were run
        run_video: Whether video was generated
        run_quality: Whether quality checks were run
        dashboard_position: Dashboard position setting
        session_id: Optional climbing session ID
        user_id: User ID for progress tracking

    Returns:
        Analysis record ID
    """
    video_record.status = VideoStatus.COMPLETED

    # Build denormalized metrics from the analysis summary (not from results dict)
    denorm = {}
    if analysis is not None:
        s = analysis.summary
        denorm = {
            "total_frames": s.total_frames,
            "avg_velocity": s.avg_velocity,
            "max_velocity": s.max_velocity,
            "max_height": s.max_height,
            "total_vertical_progress": s.total_vertical_progress,
            "avg_sway": s.avg_sway,
            "avg_movement_economy": s.avg_movement_economy,
            "lock_off_count": s.lock_off_count,
            "rest_count": s.rest_count,
            "fatigue_score": s.fatigue_score,
        }

    analysis_record = Analysis(
        video_id=video_record.id,
        session_id=session_id,
        summary=results,
        history={
            k: [float(v) if isinstance(v, (int, float)) else v for v in vals]
            for k, vals in (analysis.history if analysis else {}).items()
        },
        video_quality=results.get("video_quality") if run_quality else None,
        tracking_quality=results.get("tracking_quality") if run_quality else None,
        output_video_path=results.get("video_output") if run_video else None,
        run_metrics=run_metrics,
        run_video=run_video,
        run_quality=run_quality,
        dashboard_position=dashboard_position,
        **denorm,
    )
    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    # Auto-record progress metrics
    if run_metrics and analysis is not None:
        _record_progress_metrics(
            db=db,
            user_id=user_id,
            analysis_id=analysis_record.id,
            analysis=analysis,
        )

    return analysis_record.id


def _record_progress_metrics(
    db: Session, user_id: int, analysis_id: int, analysis: ClimbingAnalysis
) -> None:
    """Record progress metrics from analysis summary.

    Args:
        db: Database session
        user_id: User ID
        analysis_id: Analysis record ID
        analysis: ClimbingAnalysis with summary
    """
    s = analysis.summary
    metrics_to_track = {
        "avg_velocity": s.avg_velocity,
        "max_velocity": s.max_velocity,
        "max_height": s.max_height,
        "total_vertical_progress": s.total_vertical_progress,
        "avg_sway": s.avg_sway,
        "max_sway": s.max_sway,
        "avg_movement_economy": s.avg_movement_economy,
        "lock_off_count": float(s.lock_off_count),
        "rest_count": float(s.rest_count),
        "fatigue_score": s.fatigue_score,
        "avg_left_elbow": s.avg_left_elbow,
        "avg_right_elbow": s.avg_right_elbow,
        "avg_left_shoulder": s.avg_left_shoulder,
        "avg_right_shoulder": s.avg_right_shoulder,
        "avg_left_knee": s.avg_left_knee,
        "avg_right_knee": s.avg_right_knee,
        "avg_left_hip": s.avg_left_hip,
        "avg_right_hip": s.avg_right_hip,
    }

    for metric_name, value in metrics_to_track.items():
        if value is not None:
            db.add(
                ProgressMetric(
                    user_id=user_id,
                    analysis_id=analysis_id,
                    metric_name=metric_name,
                    value=float(value),
                )
            )

    db.commit()
