#!/usr/bin/env python3
"""
Example: Analyzing Climbing Video with Quality Validation

This example demonstrates the simplified ClimbingSensei facade which
automatically validates video and tracking quality during analysis.

The facade handles:
1. Video quality validation (format, resolution, FPS, lighting, stability)
2. Pose detection and metrics calculation
3. Tracking quality assessment (detection rate, confidence, smoothness)
4. Combined results in a single analysis object
"""

from climb_sensei import ClimbingSensei


def analyze_climb(video_path: str):
    """
    Analyze climbing video with automatic quality validation.

    Args:
        video_path: Path to the climbing video file.

    Returns:
        ClimbingAnalysis with metrics and quality reports.
    """
    print(f"Analyzing: {video_path}")
    print("=" * 70)

    # Use ClimbingSensei facade - handles everything automatically
    with ClimbingSensei(video_path, validate_quality=True) as sensei:
        # This single call:
        # - Validates video quality
        # - Processes all frames
        # - Calculates metrics
        # - Assesses tracking quality
        analysis = sensei.analyze(verbose=True)

    return analysis


def print_results(analysis):
    """Print analysis results with quality information."""
    climbing = analysis.summary
    video_quality = analysis.video_quality
    tracking = analysis.tracking_quality

    print("\n" + "=" * 70)
    print("CLIMBING ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Frames Processed: {climbing.total_frames}")
    print(f"Average Velocity: {climbing.avg_velocity:.4f}")
    print(f"Total Vertical Progress: {climbing.total_vertical_progress:.3f}")
    print(f"Average Movement Economy: {climbing.avg_movement_economy:.4f}")
    print(f"Lock-off Count: {climbing.lock_off_count}")
    print(f"Rest Position Count: {climbing.rest_count}")
    print(f"Fatigue Score: {climbing.fatigue_score:.3f}")

    # Video quality information
    if video_quality:
        print("\n" + "=" * 70)
        print("VIDEO QUALITY REPORT")
        print("=" * 70)
        status = "\u2705 VALID" if video_quality.is_valid else "\u274c INVALID"
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
    if tracking:
        print("\n" + "=" * 70)
        print("TRACKING QUALITY REPORT")
        print("=" * 70)
        status = "\u2705 TRACKABLE" if tracking.is_trackable else "\u274c NOT TRACKABLE"
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
            print("\n\u274c Tracking Issues:")
            for issue in tracking.issues:
                print(f"  \u2022 {issue}")

        if tracking.warnings:
            print("\n\u26a0\ufe0f  Tracking Warnings:")
            for warning in tracking.warnings:
                print(f"  \u2022 {warning}")

    print("\n" + "=" * 70)

    # Combined recommendations
    print("\nRECOMMENDATIONS")
    print("=" * 70)

    if tracking and not tracking.is_trackable:
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
        # Analyze using the facade - simple and automatic!
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
