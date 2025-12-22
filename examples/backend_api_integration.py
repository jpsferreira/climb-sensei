#!/usr/bin/env python3
"""
Example: Backend API Integration for Video Upload Service

This example demonstrates how to integrate climb-sensei's video quality
checking into a backend API service that processes uploaded videos.

The workflow is:
1. User uploads video
2. Validate video quality
3. If valid, process with climb-sensei
4. Return results
"""

import json
from pathlib import Path
from typing import Dict, Tuple

from climb_sensei import (
    ClimbingAnalyzer,
    PoseEngine,
    VideoReader,
    check_video_quality,
)


def validate_upload(filepath: str) -> Tuple[bool, Dict]:
    """
    Validate uploaded video before processing.

    Args:
        filepath: Path to uploaded video file

    Returns:
        Tuple of (is_valid, response_dict)
        - is_valid: True if video passes quality checks
        - response_dict: JSON-serializable response data
    """
    # Check video quality with deep analysis
    report = check_video_quality(filepath, deep_check=True)

    if report.is_valid:
        return True, {
            "status": "valid",
            "message": "Video is ready for processing",
            "properties": {
                "width": report.width,
                "height": report.height,
                "fps": report.fps,
                "duration": round(report.duration, 2),
                "codec": report.codec,
            },
            "quality": {
                "resolution": report.resolution_quality,
                "fps": report.fps_quality,
                "duration": report.duration_quality,
                "lighting": report.lighting_quality,
                "stability": report.stability_quality,
            },
        }
    else:
        return False, {
            "status": "invalid",
            "message": "Video failed quality checks",
            "errors": report.issues,
            "warnings": report.warnings,
            "properties": {
                "width": report.width,
                "height": report.height,
                "fps": report.fps,
                "duration": round(report.duration, 2) if report.duration else None,
            },
            "quality": {
                "resolution": report.resolution_quality,
                "fps": report.fps_quality,
                "duration": report.duration_quality,
                "lighting": report.lighting_quality,
                "stability": report.stability_quality,
            },
        }


def process_video(filepath: str) -> Dict:
    """
    Process validated video with climb-sensei.

    Args:
        filepath: Path to video file

    Returns:
        Analysis results as JSON-serializable dict
    """
    analyzer = ClimbingAnalyzer(window_size=30, fps=30)
    frames_processed = 0
    frames_with_pose = 0

    with PoseEngine() as engine:
        with VideoReader(filepath) as reader:
            # Use actual video FPS
            analyzer = ClimbingAnalyzer(window_size=int(reader.fps), fps=reader.fps)

            while True:
                success, frame = reader.read()
                if not success:
                    break

                frames_processed += 1

                # Detect pose
                results = engine.process(frame)
                if results:
                    landmarks = engine.extract_landmarks(results)
                    if landmarks:
                        analyzer.analyze_frame(landmarks)
                        frames_with_pose += 1

    # Get analysis summary
    summary = analyzer.get_summary()

    # Add processing metadata
    summary["processing"] = {
        "frames_total": frames_processed,
        "frames_with_pose": frames_with_pose,
        "pose_detection_rate": (
            round(frames_with_pose / frames_processed * 100, 1)
            if frames_processed > 0
            else 0
        ),
    }

    return summary


def handle_upload(filepath: str, output_json: str = None) -> Dict:
    """
    Complete upload handling pipeline.

    Args:
        filepath: Path to uploaded video
        output_json: Optional path to save results JSON

    Returns:
        Response dictionary with validation and/or analysis results
    """
    # Step 1: Validate video
    is_valid, validation_response = validate_upload(filepath)

    if not is_valid:
        # Video failed validation
        response = {
            "success": False,
            "validation": validation_response,
        }
    else:
        # Step 2: Process video
        try:
            analysis_results = process_video(filepath)
            response = {
                "success": True,
                "validation": validation_response,
                "analysis": analysis_results,
            }
        except Exception as e:
            response = {
                "success": False,
                "validation": validation_response,
                "error": f"Analysis failed: {str(e)}",
            }

    # Save to file if requested
    if output_json:
        with open(output_json, "w") as f:
            json.dump(response, f, indent=2)

    return response


# Example usage for different API frameworks:


def flask_example():
    """Example Flask API endpoint"""
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route("/upload", methods=["POST"])
    def upload_video():
        # Get uploaded file
        video_file = request.files.get("video")
        if not video_file:
            return jsonify({"error": "No video file provided"}), 400

        # Save to temp location
        temp_path = f"/tmp/{video_file.filename}"
        video_file.save(temp_path)

        try:
            # Process with climb-sensei
            response = handle_upload(temp_path)

            if response["success"]:
                return jsonify(response), 200
            else:
                return jsonify(response), 400

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    return app


def fastapi_example():
    """Example FastAPI endpoint"""
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/upload")
    async def upload_video(video: UploadFile = File(...)):
        # Save to temp location
        temp_path = f"/tmp/{video.filename}"

        try:
            # Save uploaded file
            with open(temp_path, "wb") as f:
                content = await video.read()
                f.write(content)

            # Process with climb-sensei
            response = handle_upload(temp_path)

            if response["success"]:
                return JSONResponse(content=response, status_code=200)
            else:
                return JSONResponse(content=response, status_code=400)

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    return app


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python backend_api_integration.py <video_file> [output.json]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Process video
    result = handle_upload(video_path, output_path)

    # Print results
    print("\n" + "=" * 60)
    print("CLIMB-SENSEI BACKEND API - PROCESSING RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)
