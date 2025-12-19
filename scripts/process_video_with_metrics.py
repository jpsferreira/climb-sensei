#!/usr/bin/env python3
"""Process video with animated metrics dashboard overlay.

This script analyzes climbing video and overlays an animated metrics
dashboard showing how various metrics evolve over time.

Usage:
    python scripts/process_video_with_metrics.py input.mp4 output.mp4
    python scripts/process_video_with_metrics.py input.mp4 output.mp4 --position left
"""

import sys
import argparse
from pathlib import Path
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
    overlay_metrics_on_frame,
    draw_metric_text_overlay,
)


def process_video_with_metrics(
    input_path: str,
    output_path: str,
    dashboard_position: str = "right",
    show_text: bool = False,
    show_progress: bool = True,
) -> None:
    """Process video with animated metrics dashboard.

    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        dashboard_position: Position of dashboard ("right", "left", "bottom")
        show_text: Whether to also show text overlay
        show_progress: Whether to show progress bar
    """
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Dashboard position: {dashboard_position}")

    # Initialize components
    with VideoReader(input_path) as reader:
        print(
            f"Video: {reader.width}x{reader.height} @ {reader.fps} fps, {reader.frame_count} frames"
        )

        analyzer = ClimbingAnalyzer(window_size=30, fps=reader.fps)

        with VideoWriter(
            output_path, fps=reader.fps, width=reader.width, height=reader.height
        ) as writer:

            with PoseEngine() as engine:
                detected_frames = 0
                frame_num = 0

                # Progress bar
                iterator = tqdm(
                    total=reader.frame_count,
                    desc="Processing",
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

                    # Draw pose landmarks
                    if results and results.pose_landmarks:
                        detected_frames += 1
                        landmarks = engine.extract_landmarks(results)

                        # Analyze frame
                        metrics = analyzer.analyze_frame(landmarks)

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

                        # Overlay dashboard
                        annotated_frame = overlay_metrics_on_frame(
                            annotated_frame,
                            dashboard,
                            position=dashboard_position,
                            alpha=0.85,
                        )

                        # Optionally add text overlay
                        if show_text:
                            annotated_frame = draw_metric_text_overlay(
                                annotated_frame,
                                metrics,
                                position=(10, 30),
                            )
                    else:
                        annotated_frame = frame

                    # Write frame
                    writer.write(annotated_frame)
                    iterator.update(1)
                    iterator.set_postfix({"detected": detected_frames})

                iterator.close()

                # Print summary
                summary = analyzer.get_summary()
                print("\n" + "=" * 60)
                print("PROCESSING COMPLETE")
                print("=" * 60)
                print(f"Total frames: {frame_num}")
                print(
                    f"Detected frames: {detected_frames} ({100*detected_frames/frame_num:.1f}%)"
                )
                print(f"\nMetrics Summary:")
                print(f"  Vertical progress: {summary['total_vertical_progress']:.3f}")
                print(f"  Average speed: {summary['avg_velocity']:.4f}")
                print(f"  Average stability (sway): {summary['avg_sway']:.4f}")
                print(f"  Average smoothness (jerk): {summary['avg_jerk']:.4f}")
                print(f"  Average body angle: {summary['avg_body_angle']:.1f}°")
                print("=" * 60)
                print(f"Output saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process climbing video with animated metrics dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path to save output video file")

    parser.add_argument(
        "--position",
        choices=["right", "left", "bottom"],
        default="right",
        help="Position of metrics dashboard (default: right)",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Also show text overlay with current values",
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
        process_video_with_metrics(
            args.input,
            args.output,
            dashboard_position=args.position,
            show_text=args.show_text,
            show_progress=not args.quiet,
        )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
