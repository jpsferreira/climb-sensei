#!/usr/bin/env python3
"""Analyze climbing performance from a video.

This script processes climbing videos to extract performance metrics.
It can generate statistical summaries, export JSON data, and/or create
annotated videos with metrics dashboards positioned side-by-side or overlaid.

Usage:
    # Text summary only (fast)
    python scripts/analyze_climb.py input.mp4

    # Export detailed JSON
    python scripts/analyze_climb.py input.mp4 --json analysis.json

    # Create annotated video with dashboard on right (side-by-side)
    python scripts/analyze_climb.py input.mp4 --video output.mp4

    # Dashboard on left side
    python scripts/analyze_climb.py input.mp4 --video output.mp4 --position left

    # Use overlay mode instead of side-by-side
    python scripts/analyze_climb.py input.mp4 --video output.mp4 --overlay

    # Both JSON and video
    python scripts/analyze_climb.py input.mp4 --json data.json --video output.mp4
"""

import sys
import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader, VideoWriter
from climb_sensei.viz import draw_pose_landmarks
from climb_sensei.config import CLIMBING_CONNECTIONS, CLIMBING_LANDMARKS
from climb_sensei.metrics import ClimbingAnalyzer
from climb_sensei.metrics_viz import (
    create_metrics_dashboard,
    compose_frame_with_dashboard,
    overlay_metrics_on_frame,
    draw_metric_text_overlay,
)


def analyze_climb(
    input_path: str,
    json_path: str = None,
    video_path: str = None,
    dashboard_position: str = "right",
    show_text: bool = False,
    show_progress: bool = True,
    use_overlay: bool = False,
) -> None:
    """Analyze climbing performance from video.

    Args:
        input_path: Path to input video file
        json_path: Path to save analysis JSON (optional)
        video_path: Path to save output video with dashboard (optional)
        dashboard_position: Position of dashboard ("right" or "left")
        show_text: Whether to show text overlay on video
        show_progress: Whether to show progress bar
        use_overlay: Whether to overlay dashboard on video (True) or place side-by-side (False)
    """
    print(f"Analyzing climb: {input_path}")

    if video_path:
        print(f"Will create video: {video_path}")
        print(f"Dashboard position: {dashboard_position}")
    if json_path:
        print(f"Will export JSON: {json_path}")

    # Initialize components
    with VideoReader(input_path) as reader:
        print(
            f"Video: {reader.width}x{reader.height} @ {reader.fps} fps, {reader.frame_count} frames"
        )

        analyzer = ClimbingAnalyzer(window_size=30, fps=reader.fps)

        # Open video writer if needed
        writer = None

        if video_path and not use_overlay:
            # For side-by-side, we need to calculate output dimensions after first frame
            # We'll initialize writer lazily after creating first dashboard
            pass
        elif video_path:
            # For overlay mode, dimensions match input
            writer = VideoWriter(
                video_path, fps=reader.fps, width=reader.width, height=reader.height
            )
            writer.__enter__()

        try:
            with PoseEngine() as engine:
                frame_metrics = []
                detected_frames = 0
                frame_num = 0

                # Progress bar
                iterator = tqdm(
                    total=reader.frame_count,
                    desc="Processing" if video_path else "Analyzing",
                    unit="frame",
                    disable=not show_progress,
                )

                # Process frames
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

                        # Store metrics if JSON export requested
                        if json_path:
                            metrics["frame"] = frame_num
                            metrics["timestamp"] = frame_num / reader.fps
                            frame_metrics.append(metrics)

                        # Create annotated frame if video output requested
                        if video_path:
                            # Draw pose
                            annotated_frame = draw_pose_landmarks(
                                frame,
                                results,
                                connections=CLIMBING_CONNECTIONS,
                                landmarks_to_draw=CLIMBING_LANDMARKS,
                            )

                            # Get metrics history for plotting
                            history = analyzer.get_history()

                            # Create dashboard
                            dashboard = create_metrics_dashboard(
                                history,
                                current_frame=frame_num - 1,  # 0-indexed
                                fps=reader.fps,
                            )

                            # Compose frame with dashboard
                            if use_overlay:
                                # Overlay mode: dashboard on top of video
                                output_frame = overlay_metrics_on_frame(
                                    annotated_frame,
                                    dashboard,
                                    position=dashboard_position,
                                    alpha=0.85,
                                )
                            else:
                                # Side-by-side mode: no overlay
                                output_frame = compose_frame_with_dashboard(
                                    annotated_frame,
                                    dashboard,
                                    position=dashboard_position,
                                    spacing=0,
                                )

                                # Initialize writer on first frame (now we know output dimensions)
                                if writer is None:
                                    out_h, out_w = output_frame.shape[:2]
                                    print(
                                        f"Output video: {out_w}x{out_h} (video + dashboard side-by-side)"
                                    )
                                    writer = VideoWriter(
                                        video_path,
                                        fps=reader.fps,
                                        width=out_w,
                                        height=out_h,
                                    )
                                    writer.__enter__()

                            # Optionally add text overlay
                            if show_text:
                                output_frame = draw_metric_text_overlay(
                                    output_frame,
                                    metrics,
                                    position=(10, 30),
                                )

                            writer.write(output_frame)
                    else:
                        # No pose detected
                        if video_path:
                            if use_overlay or writer is None:
                                # For overlay mode or before writer initialized, write original frame
                                if writer:
                                    writer.write(frame)
                            else:
                                # For side-by-side, create blank dashboard
                                blank_dashboard = np.zeros(
                                    (reader.height, 500, 3), dtype=np.uint8
                                )
                                output_frame = compose_frame_with_dashboard(
                                    frame, blank_dashboard, position=dashboard_position
                                )
                                writer.write(output_frame)

                    iterator.update(1)
                    if video_path:
                        iterator.set_postfix({"detected": detected_frames})

                iterator.close()

        finally:
            # Close video writer if it was opened
            if writer:
                writer.__exit__(None, None, None)

    # Get summary statistics
    summary = analyzer.get_summary()

    # Print comprehensive summary
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

    # Save JSON if requested
    if json_path:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            """Convert numpy types to native Python types."""
            import numpy as np

            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        output_data = {
            "video": input_path,
            "summary": convert_numpy(summary),
            "frame_metrics": convert_numpy(frame_metrics),
        }

        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed analysis saved to: {json_path}")

    # Confirm video output
    if video_path:
        print(f"Annotated video saved to: {video_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze climbing performance from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text summary only
  python scripts/analyze_climb.py climbing.mp4

  # Export JSON data
  python scripts/analyze_climb.py climbing.mp4 --json analysis.json

  # Create annotated video
  python scripts/analyze_climb.py climbing.mp4 --video output.mp4

  # Both outputs with custom dashboard
  python scripts/analyze_climb.py climbing.mp4 --json data.json --video output.mp4 --position left
        """,
    )

    parser.add_argument("input", help="Path to input video file")

    parser.add_argument(
        "--json",
        "-j",
        metavar="PATH",
        help="Export detailed analysis to JSON file",
    )

    parser.add_argument(
        "--video",
        "-v",
        metavar="PATH",
        help="Create annotated video with metrics dashboard",
    )

    parser.add_argument(
        "--position",
        choices=["right", "left"],
        default="right",
        help="Dashboard position for video output (default: right)",
    )

    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay dashboard on video instead of side-by-side (default: side-by-side)",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Add text overlay with current metric values (video only)",
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

    # Check if at least we're doing something useful
    if not args.json and not args.video:
        # Default behavior: just show summary (no file output)
        pass

    try:
        analyze_climb(
            args.input,
            json_path=args.json,
            video_path=args.video,
            dashboard_position=args.position,
            show_text=args.show_text,
            show_progress=not args.quiet,
            use_overlay=args.overlay,
        )
    except Exception as e:
        print(f"Error analyzing climb: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
