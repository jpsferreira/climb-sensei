"""FastAPI web application for ClimbingSensei video analysis.

This app demonstrates the two-phase API in action:
1. Upload video with analysis options
2. Extract landmarks once (Phase 1)
3. Generate selected outputs in parallel (Phase 2)
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

from climb_sensei import ClimbingSensei
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
    """Upload video and run selected analyses using two-phase API.

    This demonstrates the efficiency of the two-phase approach:
    - Extract landmarks once (expensive)
    - Generate multiple outputs from cached data (fast)
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

    try:
        # PHASE 1: Extract landmarks once (expensive MediaPipe pass)
        print(f"\n{'='*60}")
        print(f"Analysis ID: {analysis_id}")
        print(f"File: {file.filename}")
        print(
            f"Options: metrics={run_metrics}, video={run_video}, quality={run_quality}"
        )
        print(f"{'='*60}\n")

        with ClimbingSensei(str(upload_path), validate_quality=run_quality) as sensei:
            print("Phase 1: Extracting landmarks...")
            extracted = sensei.extract_landmarks(
                verbose=True, validate_video_quality=run_quality
            )

            results = {
                "analysis_id": analysis_id,
                "filename": file.filename,
                "frames_processed": int(extracted["frame_count"]),
                "fps": float(extracted["fps"]),
            }

            # PHASE 2: Generate requested outputs (can be parallelized!)

            # 2a. Video Quality Report
            if run_quality and extracted["video_quality"]:
                vq = extracted["video_quality"]
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
                    f"\n✓ Video Quality: {vq.resolution_quality} resolution, {vq.fps_quality} FPS"
                )

            # 2b. Climbing Metrics Analysis
            if run_metrics:
                print("\nPhase 2a: Analyzing metrics from cached landmarks...")
                analysis = sensei.analyze_from_landmarks(
                    landmarks_sequence=extracted["landmarks"],
                    fps=extracted["fps"],
                    validate_tracking_quality=run_quality,
                    verbose=True,
                )

                # Store full analysis
                analysis_cache[analysis_id] = analysis

                # Add summary to results (convert numpy types to Python native types)
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
                    vals = [
                        float(v) for v in history[key] if isinstance(v, (int, float))
                    ]
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
                                "value": round(
                                    int(summary.total_frames) / float(extracted["fps"]),
                                    1,
                                ),
                                "unit": "s",
                            },
                        },
                    },
                    "movement": {
                        "title": "Movement & Velocity",
                        "metrics": {
                            "vertical_progress": {
                                "label": "Total Vertical Progress",
                                "value": round(
                                    float(summary.total_vertical_progress), 3
                                ),
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
                                "value": round(
                                    float(summary.total_distance_traveled), 3
                                ),
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

                # Tracking quality
                if analysis.tracking_quality:
                    tq = analysis.tracking_quality
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
                        f"✓ Tracking Quality: {tq.quality_level} ({tq.detection_rate}% detection)"
                    )

                print(f"✓ Metrics calculated: {summary.total_frames} frames analyzed")

            # 2c. Generate Annotated Video
            if run_video:
                print("\nPhase 2b: Generating video from cached pose results...")
                output_video_path = OUTPUT_DIR / f"{analysis_id}_output.mp4"

                with VideoReader(str(upload_path)) as reader:
                    writer = None
                    frame_num = 0

                    try:
                        history = analysis.history if run_metrics else None

                        for pose_result in extracted["pose_results"]:
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
                                        fps=extracted["fps"],
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
                                        fps=extracted["fps"],
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

        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}\n")

        return JSONResponse(content=results)

    except ValueError as e:
        # Video quality validation failed
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Other errors
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
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
