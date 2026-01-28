#!/usr/bin/env python3
"""
Example: Analyzing Climbing Video with Service-Oriented Architecture

This example demonstrates the modern service-oriented approach which
provides independent, composable services for production use.

Services used:
1. VideoQualityService - Validates format, resolution, FPS, lighting, stability
2. TrackingQualityService - Assesses detection rate, confidence, smoothness
3. ClimbingAnalysisService - Calculates climbing metrics with pluggable calculators
"""

from climb_sensei.services import (
    VideoQualityService,
    TrackingQualityService,
    ClimbingAnalysisService,
)
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader


def analyze_climb(video_path: str):
    """
    Analyze climbing video using independent services.

    Args:
        video_path: Path to the climbing video file.

    Returns:
        Tuple of (video_quality_report, tracking_quality_report, climbing_analysis)
    """
    print(f"Analyzing: {video_path}")
    print("=" * 70)

    # Initialize services (stateless, reusable)
    video_quality_service = VideoQualityService()
    tracking_quality_service = TrackingQualityService()
    climbing_service = ClimbingAnalysisService()

    # Step 1: Validate video quality
    print("\n1. Validating video quality...")
    video_quality_report = video_quality_service.analyze_sync(video_path)

    if not video_quality_report.is_valid:
        print("❌ Video quality validation failed!")
        for issue in video_quality_report.issues:
            print(f"  - {issue}")
        return None, None, None

    # Step 2: Extract landmarks
    print("\n2. Extracting pose landmarks...")
    landmarks_sequence = []
    fps = 30.0

    pose_engine = PoseEngine()
    with VideoReader(video_path) as reader:
        fps = reader.fps
        frame_count = 0

        while True:
            success, frame = reader.read()
            if not success:
                break

            pose_result = pose_engine.process(frame)
            if pose_result and pose_result.pose_landmarks:
                landmarks = pose_engine.extract_landmarks(pose_result)
                landmarks_sequence.append(landmarks)
                frame_count += 1
            else:
                landmarks_sequence.append(None)

        print(f"   Extracted {frame_count} frames with pose data")

    pose_engine.close()

    # Step 3: Assess tracking quality (uses landmarks, not video!)
    print("\n3. Assessing tracking quality...")
    tracking_quality_report = tracking_quality_service.analyze_from_landmarks(
        landmarks_sequence, video_path=video_path
    )

    # Step 4: Analyze climbing metrics
    print("\n4. Calculating climbing metrics...")
    climbing_analysis = climbing_service.analyze(landmarks_sequence, fps=fps)

    return video_quality_report, tracking_quality_report, climbing_analysis


def print_results(video_quality, tracking, climbing):
    """Print analysis results from independent services."""
    if not video_quality or not tracking or not climbing:
        print("\n❌ Analysis failed - check quality reports above")
        return

    summary = climbing.summary

    print("\n" + "=" * 70)
    print("CLIMBING ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Frames Processed: {summary.total_frames}")
    print(f"Average Velocity: {summary.avg_velocity:.4f}")
    print(f"Total Vertical Progress: {summary.total_vertical_progress:.3f}")
    print(f"Average Movement Economy: {summary.avg_movement_economy:.4f}")
    print(f"Lock-off Count: {summary.lock_off_count}")
    print(f"Rest Position Count: {summary.rest_count}")

    # Video quality information
    print("\n" + "=" * 70)
    print("VIDEO QUALITY REPORT")
    print("=" * 70)
    status = "✅ VALID" if video_quality.is_valid else "❌ INVALID"
    print(f"Status: {status}")
    print(
        f"Resolution: {video_quality.width}x{video_quality.height} ({video_quality.resolution_quality})"
    )
    print(f"FPS: {video_quality.fps} ({video_quality.fps_quality})")
    print(
        f"Duration: {video_quality.duration_seconds:.1f}s ({video_quality.duration_quality})"
    )
    if video_quality.lighting_quality:
        print(f"Lighting: {video_quality.lighting_quality}")
    if video_quality.stability_quality:
        print(f"Stability: {video_quality.stability_quality}")

    # Tracking quality information
    print("\n" + "=" * 70)
    print("TRACKING QUALITY REPORT")
    print("=" * 70)
    status = "✅ TRACKABLE" if tracking.is_trackable else "❌ NOT TRACKABLE"
    print(f"Status: {status}")
    print(f"Quality Level: {tracking.quality_level.upper()}")
    print(
        f"Detection Rate: {tracking.detection_rate}% ({tracking.frames_with_pose}/{tracking.total_frames})"
    )
    print(f"Avg Confidence: {tracking.avg_landmark_confidence:.3f}")
    print(f"Avg Visibility: {tracking.avg_visibility_score:.1f}%")
    print(f"Tracking Smoothness: {tracking.tracking_smoothness:.3f}")
    print(f"Tracking Loss Events: {tracking.tracking_loss_events}")

    if tracking.issues:
        print("\n❌ Tracking Issues:")
        for issue in tracking.issues:
            print(f"  • {issue}")

    if tracking.warnings:
        for warning in tracking.warnings:
            print(f"  • {warning}")

    print("\n" + "=" * 70)

    # Combined recommendations
    print("\nRECOMMENDATIONS")
    print("=" * 70)

    if not tracking.is_trackable:
        print("\u26a0\ufe0f  Poor tracking quality detected:")
        print("  \u2022 Results may be unreliable")
        print("  \u2022 Consider re-recording with:")
        print("    - Better lighting")
        print("    - More stable camera position")
        print("    - Clearer view of the climber")
    elif tracking and tracking.quality_level == "acceptable":
        print("\u26a0\ufe0f  Acceptable but not optimal tracking:")
        print("  \u2022 Results should be interpreted carefully")
        print("  \u2022 Some metrics may have reduced accuracy")
    else:
        print("\u2705 Good quality - results are reliable!")

    if climbing.fatigue_score > 0.5:
        print(f"\n\u26a0\ufe0f  High fatigue score ({climbing.fatigue_score:.2f}):")
        print("  \u2022 Movement quality degraded over time")
        print("  \u2022 Consider shorter attempts or more rest")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_with_tracking.py <video_file> [output.json]")
        print("\nExample:")
        print("  python analyze_with_tracking.py climb.mp4")
        print("  python analyze_with_tracking.py climb.mp4 results.json")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        # Analyze using independent services
        analysis = analyze_climb(video_path)

        # Print comprehensive results
        print_results(analysis)

        # Export to JSON if requested
        if len(sys.argv) > 2:
            import json

            output_file = sys.argv[2]

            with open(output_file, "w") as f:
                json.dump(analysis.to_dict(), f, indent=2)

            print(f"\n\u2705 Results exported to: {output_file}")

    except Exception as e:
        print(f"\n\u274c Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
