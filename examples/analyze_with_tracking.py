#!/usr/bin/env python3
"""
Example: Analyzing Tracking Quality During Video Processing

This example demonstrates how to analyze skeleton tracking quality
while processing a climbing video, without re-running pose detection.
"""

from climb_sensei import (
    ClimbingAnalyzer,
    PoseEngine,
    VideoReader,
    analyze_tracking_from_landmarks,
)


def analyze_with_tracking_quality(video_path: str):
    """
    Analyze climbing video and assess tracking quality simultaneously.

    Args:
        video_path: Path to the climbing video file.
    """
    analyzer = ClimbingAnalyzer(window_size=30, fps=30)
    landmarks_history = []

    print(f"Processing video: {video_path}")

    with PoseEngine() as engine:
        with VideoReader(video_path) as reader:
            # Update analyzer with actual FPS
            analyzer = ClimbingAnalyzer(window_size=int(reader.fps), fps=reader.fps)

            frame_count = 0
            while True:
                success, frame = reader.read()
                if not success:
                    break

                frame_count += 1

                # Detect pose
                results = engine.process(frame)

                if results:
                    landmarks = engine.extract_landmarks(results)

                    if landmarks:
                        # Analyze frame metrics
                        analyzer.analyze_frame(landmarks)

                        # Store landmarks for tracking quality analysis
                        landmarks_history.append(landmarks)
                    else:
                        landmarks_history.append(None)
                else:
                    landmarks_history.append(None)

                # Progress feedback
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count} frames...")

    print(f"\nCompleted processing {frame_count} frames")

    # Get climbing analysis summary
    climbing_summary = analyzer.get_summary()

    # Analyze tracking quality from the landmarks we already extracted
    print("\nAnalyzing tracking quality...")
    tracking_report = analyze_tracking_from_landmarks(
        landmarks_history,
        sample_rate=1,  # Already processed all frames
        file_path=video_path,
    )

    return {
        "climbing_analysis": climbing_summary,
        "tracking_quality": tracking_report,
        "frames_processed": frame_count,
    }


def print_results(results):
    """Print combined results."""
    climbing = results["climbing_analysis"]
    tracking = results["tracking_quality"]

    print("\n" + "=" * 70)
    print("CLIMBING ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"Average Velocity: {climbing['avg_velocity']:.4f}")
    print(f"Total Vertical Progress: {climbing['total_vertical_progress']:.3f}")
    print(f"Average Movement Economy: {climbing['avg_movement_economy']:.4f}")
    print(f"Lock-off Count: {climbing['lock_off_count']}")
    print(f"Rest Position Count: {climbing['rest_count']}")
    print(f"Fatigue Score: {climbing['fatigue_score']:.3f}")

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
        print("\n⚠️  Tracking Warnings:")
        for warning in tracking.warnings:
            print(f"  • {warning}")

    print("\n" + "=" * 70)

    # Combined recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 70)

    if not tracking.is_trackable:
        print("⚠️  Poor tracking quality detected:")
        print("  • Results may be unreliable")
        print("  • Consider re-recording with:")
        print("    - Better lighting")
        print("    - More stable camera position")
        print("    - Clearer view of the climber")
    elif tracking.quality_level == "acceptable":
        print("⚠️  Acceptable but not optimal tracking:")
        print("  • Results should be interpreted carefully")
        print("  • Some metrics may have reduced accuracy")
    else:
        print("✅ Good tracking quality - results are reliable!")

    if climbing["fatigue_score"] > 0.5:
        print(f"\n⚠️  High fatigue score ({climbing['fatigue_score']:.2f}):")
        print("  • Movement quality degraded over time")
        print("  • Consider shorter attempts or more rest")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_with_tracking.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    try:
        results = analyze_with_tracking_quality(video_path)
        print_results(results)

        # Export to JSON if needed
        if len(sys.argv) > 2:
            import json

            output_file = sys.argv[2]

            # Convert tracking report to dict
            tracking_dict = {
                "is_trackable": results["tracking_quality"].is_trackable,
                "quality_level": results["tracking_quality"].quality_level,
                "detection_rate": results["tracking_quality"].detection_rate,
                "avg_confidence": results["tracking_quality"].avg_landmark_confidence,
                "avg_visibility": results["tracking_quality"].avg_visibility_score,
                "tracking_smoothness": results["tracking_quality"].tracking_smoothness,
                "tracking_loss_events": results[
                    "tracking_quality"
                ].tracking_loss_events,
                "issues": results["tracking_quality"].issues,
                "warnings": results["tracking_quality"].warnings,
            }

            export_data = {
                "climbing_analysis": results["climbing_analysis"],
                "tracking_quality": tracking_dict,
                "frames_processed": results["frames_processed"],
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

            print(f"\n✅ Results exported to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
