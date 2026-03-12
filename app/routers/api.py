"""JSON API routes: upload, analysis retrieval, video/analysis listing."""

import json
import logging
import threading
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from starlette.requests import Request

from climb_sensei.auth import get_current_active_user
from climb_sensei.database.config import SessionLocal, get_db
from climb_sensei.database.models import Analysis, User, Video
from climb_sensei.types import VideoStatus

from app.services.upload import (
    OUTPUT_DIR,
    build_metrics_response,
    check_tracking_quality,
    check_video_quality,
    extract_landmarks,
    generate_annotated_video,
    persist_results,
    run_analysis,
    save_upload,
    create_video_record,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_analysis_pipeline(
    video_id: int,
    upload_path,
    analysis_id: str,
    filename: str,
    user_id: int,
    run_metrics: bool,
    run_video: bool,
    run_quality: bool,
    dashboard_position: str,
    session_id,
    route_id=None,
):
    """Run the full analysis pipeline in a background thread.

    Uses its own DB session since the request session is already closed.
    """
    db = SessionLocal()
    try:
        video_record = db.query(Video).filter(Video.id == video_id).first()
        if not video_record:
            logger.error("Video record %d not found in background task", video_id)
            return

        results = {"analysis_id": analysis_id, "filename": filename}

        # Phase 1: Video quality check
        video_quality_report = None
        if run_quality:
            vq_data = check_video_quality(upload_path)
            video_quality_report = vq_data["report"]
            results["video_quality"] = vq_data["result"]

        # Phase 2: Extract landmarks (expensive MediaPipe pass)
        landmarks_history, pose_results_history, frame_count, fps = extract_landmarks(
            upload_path
        )
        results["frames_processed"] = frame_count
        results["fps"] = float(fps)

        # Phase 3: Tracking quality
        tracking_quality_report = None
        if run_quality:
            tq_data = check_tracking_quality(landmarks_history, upload_path)
            tracking_quality_report = tq_data["report"]
            results["tracking_quality"] = tq_data["result"]

        # Phase 4: Climbing metrics
        analysis = None
        if run_metrics:
            analysis = run_analysis(
                landmarks_history,
                fps=fps,
                upload_path=upload_path,
                video_quality_report=video_quality_report,
                tracking_quality_report=tracking_quality_report,
            )
            results["metrics"] = build_metrics_response(analysis, fps)

        # Phase 5: Generate annotated video
        if run_video:
            history = analysis.history if run_metrics and analysis else None
            results["video_output"] = generate_annotated_video(
                upload_path,
                analysis_id,
                pose_results_history,
                fps,
                history=history,
                dashboard_position=dashboard_position,
            )

        # Persist to database
        db_analysis_id = persist_results(
            db=db,
            video_record=video_record,
            analysis=analysis,
            results=results,
            run_metrics=run_metrics,
            run_video=run_video,
            run_quality=run_quality,
            dashboard_position=dashboard_position,
            session_id=session_id,
            user_id=user_id,
        )

        # Link attempt to route if route_id was provided
        if route_id:
            from climb_sensei.database.models import Attempt
            from datetime import datetime, timezone

            attempt = Attempt(
                route_id=route_id,
                video_id=video_id,
                session_id=session_id,
                analysis_id=db_analysis_id,
                date=datetime.now(timezone.utc),
            )
            db.add(attempt)
            db.commit()

        logger.info("Background analysis complete for video %d", video_id)

    except Exception:
        logger.exception("Background analysis failed for video %d", video_id)
        db.rollback()
        video_record = db.query(Video).filter(Video.id == video_id).first()
        if video_record:
            video_record.status = VideoStatus.FAILED
            db.commit()

    finally:
        if upload_path.exists():
            upload_path.unlink()
        db.close()


@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    run_metrics: bool = Form(True),
    run_video: bool = Form(False),
    run_quality: bool = Form(True),
    dashboard_position: str = Form("right"),
    session_id: int = Form(None),
    route_id: int = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Upload video and start background analysis.

    Returns immediately with the video ID. The client should poll
    GET /api/videos/{id}/status to track progress.
    """
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a video file."
        )

    analysis_id = str(uuid.uuid4())
    upload_path = save_upload(file, analysis_id)

    video_record = create_video_record(
        db,
        user_id=current_user.id,
        filename=file.filename,
        upload_path=upload_path,
    )
    db.commit()

    # Launch analysis in background thread
    thread = threading.Thread(
        target=_run_analysis_pipeline,
        kwargs={
            "video_id": video_record.id,
            "upload_path": upload_path,
            "analysis_id": analysis_id,
            "filename": file.filename,
            "user_id": current_user.id,
            "run_metrics": run_metrics,
            "run_video": run_video,
            "run_quality": run_quality,
            "dashboard_position": dashboard_position,
            "session_id": session_id,
            "route_id": route_id,
        },
        daemon=True,
    )
    thread.start()

    return JSONResponse(
        status_code=202,
        content={
            "video_id": video_record.id,
            "route_id": route_id,
            "status": VideoStatus.PROCESSING,
            "message": "Upload received. Analysis running in background.",
        },
    )


@router.get("/api/videos/{video_id}/status")
async def get_video_status(
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Poll endpoint for video analysis status.

    Returns current status and, when complete, the analysis ID.
    """
    video = (
        db.query(Video)
        .filter(Video.id == video_id, Video.user_id == current_user.id)
        .first()
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Force a fresh read from DB (background thread may have updated)
    db.refresh(video)

    result = {
        "video_id": video.id,
        "status": video.status,
        "filename": video.filename,
    }

    if video.status == VideoStatus.COMPLETED:
        # Include the analysis ID so the frontend can fetch results
        latest_analysis = (
            db.query(Analysis)
            .filter(Analysis.video_id == video.id)
            .order_by(Analysis.id.desc())
            .first()
        )
        if latest_analysis:
            result["analysis_id"] = latest_analysis.id

    return JSONResponse(content=result)


@router.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get detailed analysis results as JSON (from DB, not in-memory cache)."""
    analysis = (
        db.query(Analysis)
        .join(Video)
        .filter(Analysis.id == analysis_id, Video.user_id == current_user.id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return JSONResponse(
        content={
            "id": analysis.id,
            "video_id": analysis.video_id,
            "summary": analysis.summary,
            "history": analysis.history,
            "video_quality": analysis.video_quality,
            "tracking_quality": analysis.tracking_quality,
        }
    )


@router.get("/download/{analysis_id}")
async def download_json(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Download analysis results as JSON file."""
    analysis = (
        db.query(Analysis)
        .join(Video)
        .filter(Analysis.id == analysis_id, Video.user_id == current_user.id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    json_path = OUTPUT_DIR / f"{analysis_id}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(
            {"summary": analysis.summary, "history": analysis.history},
            f,
            indent=2,
        )

    return FileResponse(
        path=str(json_path),
        filename=f"analysis_{analysis_id}.json",
        media_type="application/json",
    )


# ============================================================================
# Database API Endpoints
# ============================================================================


@router.get("/api/videos")
async def list_videos(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all videos uploaded by the current user."""
    videos = db.query(Video).filter(Video.user_id == current_user.id).all()

    return JSONResponse(
        content=[
            {
                "id": v.id,
                "filename": v.filename,
                "status": v.status,
                "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None,
                "analysis_count": len(v.analyses),
            }
            for v in videos
        ]
    )


@router.get("/api/videos/{video_id}")
async def get_video(
    video_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get details of a specific video."""
    video = (
        db.query(Video)
        .filter(Video.id == video_id, Video.user_id == current_user.id)
        .first()
    )
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return JSONResponse(
        content={
            "id": video.id,
            "filename": video.filename,
            "file_path": video.file_path,
            "status": video.status,
            "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None,
            "analyses": [
                {
                    "id": a.id,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                    "summary": a.summary,
                    "video_quality": a.video_quality,
                    "tracking_quality": a.tracking_quality,
                }
                for a in video.analyses
            ],
        }
    )


@router.get("/api/analyses")
async def list_analyses(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all analyses for the current user's videos."""
    analyses = (
        db.query(Analysis).join(Video).filter(Video.user_id == current_user.id).all()
    )

    return JSONResponse(
        content=[
            {
                "id": a.id,
                "video_id": a.video_id,
                "video_filename": a.video.filename if a.video else None,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "summary": a.summary,
                "has_video_quality": a.video_quality is not None,
                "has_tracking_quality": a.tracking_quality is not None,
            }
            for a in analyses
        ]
    )


@router.get("/api/analyses/{analysis_id}")
async def get_analysis_detail(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get detailed analysis results."""
    analysis = (
        db.query(Analysis)
        .join(Video)
        .filter(Analysis.id == analysis_id, Video.user_id == current_user.id)
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return JSONResponse(
        content={
            "id": analysis.id,
            "video_id": analysis.video_id,
            "video_filename": analysis.video.filename if analysis.video else None,
            "created_at": (
                analysis.created_at.isoformat() if analysis.created_at else None
            ),
            "summary": analysis.summary,
            "history": analysis.history,
            "video_quality": analysis.video_quality,
            "tracking_quality": analysis.tracking_quality,
        }
    )
