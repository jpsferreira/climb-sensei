"""Example: Using the new service-oriented architecture.

This example demonstrates how the new services provide better modularity,
testability, and separation of concerns.

Run with:
    python examples/use_new_services.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.services import (
    VideoQualityService,
    TrackingQualityService,
    ClimbingAnalysisService,
)
from climb_sensei.domain.calculators import (
    StabilityCalculator,
    ProgressCalculator,
    EfficiencyCalculator,
)
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader


def example_1_video_quality_only():
    """Example 1: Use video quality service standalone."""
    print("\n" + "=" * 60)
    print("Example 1: Standalone Video Quality Check")
    print("=" * 60)

    # Create service (no dependencies on climbing analysis!)
    service = VideoQualityService(default_deep_check=False)

    # Analyze video
    video_path = "path/to/video.mp4"

    try:
        report = service.analyze_sync(video_path, deep_check=True)

        print(f"\n✓ Video Quality: {report.resolution_quality}")
        print(f"  Resolution: {report.width}x{report.height}")
        print(f"  FPS: {report.fps} ({report.fps_quality})")
        print(f"  Duration: {report.duration_seconds:.1f}s")

        if not report.is_valid:
            print("\n✗ Issues found:")
            for issue in report.issues:
                print(f"  - {issue}")

    except FileNotFoundError:
        print(f"Video not found: {video_path}")
        print("This example requires a video file.")


def example_2_custom_calculators():
    """Example 2: Use climbing service with custom calculators."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Calculator Configuration")
    print("=" * 60)

    # Create custom calculator suite (only stability and progress)
    custom_calculators = [
        StabilityCalculator(window_size=60, fps=30.0),  # 2-second window
        ProgressCalculator(fps=30.0),
    ]

    # Create service with custom calculators
    service = ClimbingAnalysisService(calculators=custom_calculators)

    # Show available metrics
    metrics = service.get_available_metrics()
    print("\n✓ Available metrics with custom calculators:")
    for metric in metrics:
        print(f"  - {metric}")

    print("\n💡 You only get metrics from the calculators you configured!")
    print("   This reduces computation and response size.")


def example_3_add_calculator_dynamically():
    """Example 3: Add calculators dynamically."""
    print("\n" + "=" * 60)
    print("Example 3: Dynamic Calculator Addition")
    print("=" * 60)

    # Start with minimal service
    service = ClimbingAnalysisService(calculators=[])

    print("\n✓ Starting with no calculators")
    print(f"  Available metrics: {len(service.get_available_metrics())}")

    # Add calculators as needed
    service.add_calculator(StabilityCalculator())
    print("\n✓ Added StabilityCalculator")
    print(f"  Available metrics: {len(service.get_available_metrics())}")

    service.add_calculator(EfficiencyCalculator())
    print("\n✓ Added EfficiencyCalculator")
    print(f"  Available metrics: {len(service.get_available_metrics())}")

    print("\n💡 Final metrics:")
    for metric in service.get_available_metrics():
        print(f"  - {metric}")


def example_4_full_pipeline_with_services():
    """Example 4: Full analysis pipeline using all services."""
    print("\n" + "=" * 60)
    print("Example 4: Full Pipeline with Service Composition")
    print("=" * 60)

    video_path = "path/to/video.mp4"

    try:
        # Step 1: Video Quality (independent)
        print("\n[1/4] Checking video quality...")
        vq_service = VideoQualityService()
        vq_report = vq_service.validate_or_raise(video_path)
        print(f"✓ Video is valid: {vq_report.resolution_quality} quality")

        # Step 2: Extract landmarks (infrastructure)
        print("\n[2/4] Extracting landmarks...")
        landmarks_sequence = []
        with PoseEngine() as engine:
            with VideoReader(video_path) as video:
                frame_count = 0
                while True:
                    success, frame = video.read()
                    if not success:
                        break

                    result = engine.process(frame)
                    if result and result.pose_landmarks:
                        landmarks = engine.extract_landmarks(result)
                        landmarks_sequence.append(landmarks)
                    else:
                        landmarks_sequence.append(None)

                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"  Processed {frame_count} frames...")

        print(f"✓ Extracted {len(landmarks_sequence)} frames")

        # Step 3: Tracking Quality (independent)
        print("\n[3/4] Analyzing tracking quality...")
        tq_service = TrackingQualityService()
        tq_report = tq_service.validate_or_raise(landmarks_sequence, video_path)
        print(f"✓ Tracking is {tq_report.quality_level}")
        print(f"  Detection rate: {tq_report.detection_rate:.1f}%")

        # Step 4: Climbing Analysis (independent)
        print("\n[4/4] Analyzing climbing metrics...")
        ca_service = ClimbingAnalysisService()
        analysis = ca_service.analyze(landmarks_sequence, fps=30.0)

        print("✓ Analysis complete!")
        print("\n📊 Summary:")
        print(f"  Total frames: {analysis.summary.total_frames}")
        print(f"  Vertical progress: {analysis.summary.total_vertical_progress:.3f}")
        print(f"  Avg velocity: {analysis.summary.avg_velocity:.3f}")
        print(f"  Movement economy: {analysis.summary.movement_economy:.3f}")
        print(f"  Lock-offs: {analysis.summary.lock_off_count}")

        print("\n💡 Each service was independent!")
        print("   - Video quality didn't need pose detection")
        print("   - Tracking quality worked from landmarks")
        print("   - Climbing analysis is composable")

    except FileNotFoundError:
        print(f"Video not found: {video_path}")
        print("This example requires a video file.")
    except ValueError as e:
        print(f"Validation failed: {e}")


def example_5_async_usage():
    """Example 5: Async service usage."""
    print("\n" + "=" * 60)
    print("Example 5: Async Service Usage")
    print("=" * 60)

    import asyncio

    async def analyze_async():
        # All services support async!
        vq_service = VideoQualityService()

        video_path = "path/to/video.mp4"

        try:
            # Async video quality check
            report = await vq_service.analyze(video_path, deep_check=True)

            print("\n✓ Async analysis complete!")
            print(f"  Resolution: {report.width}x{report.height}")
            print(f"  Quality: {report.resolution_quality}")

        except FileNotFoundError:
            print(f"Video not found: {video_path}")

    # Run async example
    print("\n💡 Running async analysis...")
    asyncio.run(analyze_async())
    print("\n✓ Async pattern allows concurrent processing of multiple videos!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ClimbingSensei - New Service Architecture Examples")
    print("=" * 60)

    # Run examples
    example_1_video_quality_only()
    example_2_custom_calculators()
    example_3_add_calculator_dynamically()
    example_4_full_pipeline_with_services()
    example_5_async_usage()

    print("\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Services are independent and reusable")
    print("2. Calculators are composable and extensible")
    print("3. No tight coupling between components")
    print("4. Async support for better concurrency")
    print("5. Easy to test each component in isolation")
    print("\n")
