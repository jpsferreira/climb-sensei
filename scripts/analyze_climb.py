#!/usr/bin/env python3
"""Analyze climbing performance from a video.

This script processes a video and generates climbing analysis metrics
including stability, speed, and vertical progression.

Usage:
    python scripts/analyze_climb.py input_video.mp4
    python scripts/analyze_climb.py input_video.mp4 --output analysis.json
"""

import sys
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader
from climb_sensei.metrics import ClimbingAnalyzer


def analyze_climb(
    input_path: str,
    output_path: str = None,
    show_progress: bool = True,
) -> None:
    """Analyze climbing performance from video.

    Args:
        input_path: Path to input video file
        output_path: Path to save analysis JSON (optional)
        show_progress: Whether to show progress bar
    """
    print(f"Analyzing climb: {input_path}")

    # Initialize components
    with VideoReader(input_path) as reader:
        print(
            f"Video: {reader.width}x{reader.height} @ {reader.fps} fps, {reader.frame_count} frames"
        )

        analyzer = ClimbingAnalyzer(window_size=30, fps=reader.fps)

        with PoseEngine() as engine:
            frame_metrics = []
            detected_frames = 0

            # Progress bar
            iterator = tqdm(
                total=reader.frame_count,
                desc="Analyzing",
                unit="frame",
                disable=not show_progress,
            )

            frame_num = 0
            while True:
                success, frame = reader.read()
                if not success:
                    break

                frame_num += 1

                # Detect pose
                results = engine.process(frame)

                if results and results.pose_landmarks:
                    detected_frames += 1
                    landmarks = engine.extract_landmarks(results)

                    # Analyze frame
                    metrics = analyzer.analyze_frame(landmarks)
                    metrics["frame"] = frame_num
                    metrics["timestamp"] = frame_num / reader.fps
                    frame_metrics.append(metrics)

                iterator.update(1)

            iterator.close()

    # Get summary statistics
    summary = analyzer.get_summary()

    print("\n" + "=" * 60)
    print("CLIMBING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total frames analyzed: {summary['total_frames']}")
    print(f"Detection rate: {100*detected_frames/frame_num:.1f}%")
    print("\nVertical Progression:")
    print(
        f"  Total height gained: {summary['total_vertical_progress']:.3f} (normalized)"
    )
    print(f"  Maximum height: {summary['max_height']:.3f}")
    print("\nMovement Speed:")
    print(f"  Average velocity: {summary['avg_velocity']:.4f} units/sec")
    print(f"  Maximum velocity: {summary['max_velocity']:.4f} units/sec")
    print("\nStability:")
    print(f"  Average lateral sway: {summary['avg_sway']:.4f} (lower = more stable)")
    print(f"  Maximum sway: {summary['max_sway']:.4f}")
    print("\nSmoothness:")
    print(f"  Average jerk: {summary['avg_jerk']:.2f} (lower = smoother)")
    print(f"  Maximum jerk: {summary['max_jerk']:.2f}")
    print("\nBody Positioning:")
    print(
        f"  Average body angle: {summary['avg_body_angle']:.1f}° (lean from vertical)"
    )
    print(f"  Average hand span: {summary['avg_hand_span']:.3f}")
    print(f"  Average foot span: {summary['avg_foot_span']:.3f}")
    print("\nEfficiency & Technique:")
    print(f"  Total distance traveled: {summary['total_distance_traveled']:.3f}")
    print(
        f"  Movement economy: {summary['avg_movement_economy']:.3f} (higher = more efficient)"
    )
    print("\nStrength & Technique:")
    print(
        f"  Lock-offs detected: {summary['lock_off_count']} ({summary['lock_off_percentage']:.1f}% of frames)"
    )
    print(
        f"  Rest positions: {summary['rest_count']} ({summary['rest_percentage']:.1f}% of frames)"
    )
    print("\nFatigue & Endurance:")
    print(
        f"  Fatigue score: {summary['fatigue_score']:.3f} (0=no fatigue, 1=high fatigue)"
    )
    print("\nJoint Angles (average):")
    print(
        f"  Left elbow: {summary['avg_left_elbow']:.1f}°, Right elbow: {summary['avg_right_elbow']:.1f}°"
    )
    print(
        f"  Left shoulder: {summary['avg_left_shoulder']:.1f}°, Right shoulder: {summary['avg_right_shoulder']:.1f}°"
    )
    print(
        f"  Left knee: {summary['avg_left_knee']:.1f}°, Right knee: {summary['avg_right_knee']:.1f}°"
    )
    print(
        f"  Left hip: {summary['avg_left_hip']:.1f}°, Right hip: {summary['avg_right_hip']:.1f}°"
    )
    print("=" * 60)

    # Save to file if requested
    if output_path:
        output_data = {
            "video": input_path,
            "summary": summary,
            "frame_metrics": frame_metrics,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed analysis saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze climbing performance from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to input video file")

    parser.add_argument(
        "--output",
        "-o",
        help="Path to save analysis JSON (optional)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress updates",
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        analyze_climb(
            args.input,
            output_path=args.output,
            show_progress=not args.quiet,
        )
    except Exception as e:
        print(f"Error analyzing climb: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
