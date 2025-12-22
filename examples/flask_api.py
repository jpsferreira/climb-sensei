#!/usr/bin/env python3
"""
Example: Flask API for Video Upload Service

This example demonstrates how to integrate climb-sensei's video quality
checking into a Flask backend API service that processes uploaded videos.

The workflow is:
1. User uploads video
2. Validate video quality
3. If valid, process with climb-sensei
4. Return results

Installation:
    pip install flask

Usage:
    python examples/flask_api.py

    Then in another terminal:
    curl -F "video=@your_video.mp4" http://localhost:5000/upload
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


def create_app():
    """Create and configure Flask app."""
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        """Health check endpoint."""
        return jsonify(
            {"service": "climb-sensei API", "version": "1.0.0", "status": "running"}
        )

    @app.route("/upload", methods=["POST"])
    def upload_video():
        """
        Process uploaded video.

        Expects:
            - multipart/form-data with 'video' field

        Returns:
            JSON response with validation and analysis results
        """
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


if __name__ == "__main__":
    app = create_app()

    print("\n" + "=" * 60)
    print("Starting climb-sensei Flask API server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /          - Health check")
    print("  POST /upload    - Upload and analyze video")
    print("\nExample usage:")
    print('  curl -F "video=@climb.mp4" http://localhost:5000/upload')
    print("\n" + "=" * 60 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
