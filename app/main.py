"""FastAPI web application for ClimbingSensei video analysis.

Modern service-oriented architecture:
1. Independent services for video quality, tracking quality, climbing analysis
2. Clean separation of concerns
3. Composable, testable, production-ready
"""

import sys
from pathlib import Path
import shutil
import json
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use new service-oriented APIs
from climb_sensei.services import (
    VideoQualityService,
    TrackingQualityService,
    ClimbingAnalysisService,
)
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader, VideoWriter
from climb_sensei.viz import draw_pose_landmarks
from climb_sensei.config import CLIMBING_CONNECTIONS, CLIMBING_LANDMARKS
from climb_sensei.metrics_viz import (
    create_metrics_dashboard,
    compose_frame_with_dashboard,
)

app = FastAPI(
    title="ClimbingSensei Web App",
    description="Upload climbing videos and analyze performance",
    version="1.0.0",
)

# Setup directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Store analysis results in memory (use Redis/DB in production)
analysis_cache = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    run_metrics: bool = Form(True),
    run_video: bool = Form(False),
    run_quality: bool = Form(True),
    dashboard_position: str = Form("right"),
):
    """Upload video and run selected analyses using service-oriented architecture.

    Uses independent services that can be composed as needed:
    - VideoQualityService: Validates video format and quality
    - TrackingQualityService: Assesses pose detection reliability
    - ClimbingAnalysisService: Calculates climbing metrics
    """
    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a video file."
        )

    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize services
    video_quality_service = VideoQualityService()
    tracking_quality_service = TrackingQualityService()
    climbing_service = ClimbingAnalysisService()

    pose_engine = None

    try:
        print(f"\n{'=' * 60}")
        print(f"Analysis ID: {analysis_id}")
        print(f"File: {file.filename}")
        print(
            f"Options: metrics={run_metrics}, video={run_video}, quality={run_quality}"
        )
        print(f"{'=' * 60}\n")

        results = {
            "analysis_id": analysis_id,
            "filename": file.filename,
        }

        # PHASE 1: Video Quality Check (if requested)
        video_quality_report = None
        if run_quality:
            print("Phase 1a: Checking video quality...")
            video_quality_report = video_quality_service.analyze_sync(str(upload_path))

            if not video_quality_report.is_valid:
                error_msg = "Video quality validation failed:\n"
                for issue in video_quality_report.issues:
                    error_msg += f"  - {issue}\n"
                raise ValueError(error_msg.rstrip())

            vq = video_quality_report
            results["video_quality"] = {
                "is_valid": bool(vq.is_valid),
                "resolution": f"{vq.width}x{vq.height}",
                "resolution_quality": str(vq.resolution_quality),
                "fps_quality": str(vq.fps_quality),
                "duration": f"{float(vq.duration_seconds):.1f}s",
                "issues": list(vq.issues),
                "warnings": list(vq.warnings),
            }
            print(
                f"✓ Video Quality: {vq.resolution_quality} resolution, {vq.fps_quality} FPS"
            )

            if vq.warnings:
                print("⚠️  Warnings:")
                for warning in vq.warnings:
                    print(f"  - {warning}")

        # PHASE 2: Extract landmarks once (expensive MediaPipe pass)
        print("\nPhase 1b: Extracting landmarks...")

        landmarks_history = []
        pose_results_history = []
        frame_count = 0
        fps = 30.0

        pose_engine = PoseEngine()

        with VideoReader(str(upload_path)) as video:
            fps = video.fps
            results["fps"] = float(fps)

            while True:
                success, frame = video.read()
                if not success:
                    break

                # Detect pose
                pose_result = pose_engine.process(frame)

                if pose_result and pose_result.pose_landmarks:
                    # Extract landmarks
                    landmarks = pose_engine.extract_landmarks(pose_result)
                    landmarks_history.append(landmarks)
                    pose_results_history.append(pose_result)
                    frame_count += 1

                    if frame_count % 100 == 0:
                        print(f"  Extracted {frame_count} frames...")
                else:
                    # No pose detected
                    landmarks_history.append(None)
                    pose_results_history.append(None)

        results["frames_processed"] = frame_count
        print(
            f"✓ Extraction complete! {len(landmarks_history)} frames total, {frame_count} with pose."
        )

        # PHASE 3: Tracking Quality Analysis (if requested)
        tracking_quality_report = None
        if run_quality:
            print("\nPhase 2a: Analyzing tracking quality...")
            tracking_quality_report = tracking_quality_service.analyze_from_landmarks(
                landmarks_history, video_path=str(upload_path)
            )

            tq = tracking_quality_report
            results["tracking_quality"] = {
                "is_trackable": bool(tq.is_trackable),
                "quality_level": str(tq.quality_level),
                "detection_rate": float(tq.detection_rate),
                "tracking_smoothness": round(float(tq.tracking_smoothness), 4),
                "avg_confidence": round(float(tq.avg_landmark_confidence), 4),
                "tracking_loss_events": int(tq.tracking_loss_events),
                "warnings": list(tq.warnings),
            }

            print(
                f"✓ Tracking Quality: {tq.quality_level} ({tq.detection_rate:.1f}% detection)"
            )

            if not tq.is_trackable:
                print(
                    "⚠️  Warning: Poor tracking quality detected - results may be unreliable"
                )
            elif tq.warnings:
                print("⚠️  Warnings:")
                for warning in tq.warnings:
                    print(f"  - {warning}")

        # PHASE 4: Climbing Metrics Analysis (if requested)
        analysis = None
        if run_metrics:
            print("\nPhase 2b: Analyzing climbing metrics...")
            analysis = climbing_service.analyze(
                landmarks_history,
                fps=fps,
                video_path=str(upload_path),
                video_quality=video_quality_report,
                tracking_quality=tracking_quality_report,
            )

            # Store full analysis
            analysis_cache[analysis_id] = analysis

            # Add summary to results
            summary = analysis.summary
            history = analysis.history

            # Build documentation URLs
            BASE_DOC_URL = "https://jpsferreira.github.io/climb-sensei/metrics/"

            def _doc_url(metric_key: str) -> str:
                anchor = metric_key.replace("_", "-")
                return f"{BASE_DOC_URL}#{anchor}"

            # Categorize metrics for better presentation
            def _get_metric_value(key: str) -> float:
                """Get average value for a metric from history."""
                if key not in history:
                    return 0.0
                vals = [float(v) for v in history[key] if isinstance(v, (int, float))]
                if vals:
                    return round(sum(vals) / len(vals), 4)
                return 0.0

            # Organized metric categories
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
                            "unit": "°",
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
                            "value": _get_metric_value("left_elbow_angle"),
                            "unit": "°",
                            "doc": _doc_url("left_elbow_angle"),
                        },
                        "right_elbow_angle": {
                            "label": "Right Elbow Angle",
                            "value": _get_metric_value("right_elbow_angle"),
                            "unit": "°",
                            "doc": _doc_url("right_elbow_angle"),
                        },
                        "left_shoulder_angle": {
                            "label": "Left Shoulder Angle",
                            "value": _get_metric_value("left_shoulder_angle"),
                            "unit": "°",
                            "doc": _doc_url("left_shoulder_angle"),
                        },
                        "right_shoulder_angle": {
                            "label": "Right Shoulder Angle",
                            "value": _get_metric_value("right_shoulder_angle"),
                            "unit": "°",
                            "doc": _doc_url("right_shoulder_angle"),
                        },
                        "left_knee_angle": {
                            "label": "Left Knee Angle",
                            "value": _get_metric_value("left_knee_angle"),
                            "unit": "°",
                            "doc": _doc_url("left_knee_angle"),
                        },
                        "right_knee_angle": {
                            "label": "Right Knee Angle",
                            "value": _get_metric_value("right_knee_angle"),
                            "unit": "°",
                            "doc": _doc_url("right_knee_angle"),
                        },
                        "left_hip_angle": {
                            "label": "Left Hip Angle",
                            "value": _get_metric_value("left_hip_angle"),
                            "unit": "°",
                            "doc": _doc_url("left_hip_angle"),
                        },
                        "right_hip_angle": {
                            "label": "Right Hip Angle",
                            "value": _get_metric_value("right_hip_angle"),
                            "unit": "°",
                            "doc": _doc_url("right_hip_angle"),
                        },
                    },
                },
            }

            results["metrics"] = {
                "categories": categories,
                "total_frames": int(summary.total_frames),
            }

            print(f"✓ Metrics calculated: {summary.total_frames} frames analyzed")

        # PHASE 5: Generate Annotated Video (if requested)
        if run_video:
            print("\nPhase 3: Generating annotated video...")
            output_video_path = OUTPUT_DIR / f"{analysis_id}_output.mp4"

            with VideoReader(str(upload_path)) as reader:
                writer = None
                frame_num = 0

                try:
                    history = analysis.history if run_metrics and analysis else None

                    for pose_result in pose_results_history:
                        success, frame = reader.read()
                        if not success:
                            break

                        frame_num += 1

                        if pose_result is not None:
                            # Draw pose using cached results (no re-processing!)
                            annotated = draw_pose_landmarks(
                                frame,
                                pose_result,
                                connections=CLIMBING_CONNECTIONS,
                                landmarks_to_draw=CLIMBING_LANDMARKS,
                            )

                            # Add dashboard if metrics available
                            if history:
                                dashboard = create_metrics_dashboard(
                                    history,
                                    current_frame=frame_num - 1,
                                    fps=fps,
                                )

                                output_frame = compose_frame_with_dashboard(
                                    annotated,
                                    dashboard,
                                    position=dashboard_position,
                                )
                            else:
                                output_frame = annotated

                            # Initialize writer on first frame
                            if writer is None:
                                h, w = output_frame.shape[:2]
                                writer = VideoWriter(
                                    str(output_video_path),
                                    fps=fps,
                                    width=w,
                                    height=h,
                                )
                                writer.__enter__()

                            writer.write(output_frame)

                        if frame_num % 100 == 0:
                            print(f"  Processed {frame_num} frames...")

                    if writer:
                        writer.__exit__(None, None, None)

                    results["video_output"] = f"/outputs/{analysis_id}_output.mp4"
                    print(f"✓ Video generated: {frame_num} frames")

                except Exception as e:
                    if writer:
                        writer.__exit__(None, None, None)
                    raise e

        print(f"\n{'=' * 60}")
        print("Analysis complete!")
        print(f"{'=' * 60}\n")

        return JSONResponse(content=results)

    except ValueError as e:
        # Video quality validation failed
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Other errors
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Clean up resources
        if pose_engine:
            pose_engine.close()

        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get detailed analysis results as JSON."""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analysis_cache[analysis_id]
    return JSONResponse(content=analysis.to_dict())


@app.get("/download/{analysis_id}")
async def download_json(analysis_id: str):
    """Download analysis results as JSON file."""
    if analysis_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis = analysis_cache[analysis_id]

    # Save to temp file
    json_path = OUTPUT_DIR / f"{analysis_id}_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis.to_dict(), f, indent=2)

    return FileResponse(
        path=str(json_path),
        filename=f"analysis_{analysis_id}.json",
        media_type="application/json",
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧗 ClimbingSensei Web App")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("Press CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
