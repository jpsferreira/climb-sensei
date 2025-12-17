#!/usr/bin/env python3
"""Script to process a video through the pose estimator.

This script reads a video file, runs pose estimation on each frame,
and outputs a new video with pose landmarks drawn on it.

Usage:
    python scripts/process_video.py input_video.mp4 output_video.mp4

    Or with optional confidence thresholds:
    python scripts/process_video.py input.mp4 output.mp4 --detection-conf 0.5 --tracking-conf 0.5
    
    With metrics calculation:
    python scripts/process_video.py input.mp4 output.mp4 --metrics
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional


# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader, VideoWriter
from climb_sensei.viz import draw_pose_landmarks, draw_metrics_overlay
from climb_sensei.config import CLIMBING_CONNECTIONS, CLIMBING_LANDMARKS
from climb_sensei.smoothing import LandmarkSmoother
from climb_sensei.metrics import ClimbingMetrics


def process_video(
    input_path: str,
    output_path: str,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    show_progress: bool = True,
    include_head: bool = False,
    enable_smoothing: bool = True,
    smoothing_min_cutoff: float = 1.0,
    smoothing_beta: float = 0.007,
    calculate_metrics: bool = False,
    metrics_output: Optional[str] = None,
    show_metrics_overlay: bool = False,
) -> None:
    """Process video through pose estimator and save results.

    Args:
        input_path: Path to input video file.
        output_path: Path to save output video.
        min_detection_confidence: Minimum confidence for pose detection.
        min_tracking_confidence: Minimum confidence for pose tracking.
        show_progress: Whether to print progress updates.
        include_head: Whether to include head keypoints (default: False for climbing).
        enable_smoothing: Whether to apply One Euro Filter smoothing (default: True).
        smoothing_min_cutoff: Minimum cutoff frequency for smoothing (default: 1.0).
        smoothing_beta: Speed coefficient for smoothing (default: 0.007).
        calculate_metrics: Whether to calculate climbing metrics (default: False).
        metrics_output: Path to save metrics JSON file (optional).
        show_metrics_overlay: Whether to overlay metrics on video (default: False).
    """
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Choose connections and landmarks based on include_head flag
    if include_head:
        from climb_sensei.config import FULL_POSE_CONNECTIONS

        connections = FULL_POSE_CONNECTIONS
        landmarks = None  # Draw all landmarks
        print("Using full pose (including head keypoints)")
    else:
        connections = CLIMBING_CONNECTIONS
        landmarks = CLIMBING_LANDMARKS
        print("Using climbing pose (body only, no head keypoints)")

    # Initialize smoother if enabled
    if enable_smoothing:
        smoother = LandmarkSmoother(
            filter_type="one_euro", min_cutoff=smoothing_min_cutoff, beta=smoothing_beta
        )
        print(
            f"Smoothing enabled (One Euro Filter: min_cutoff={smoothing_min_cutoff}, beta={smoothing_beta})"
        )
    else:
        smoother = None
        print("Smoothing disabled")
    
    # Metrics collection
    if calculate_metrics or show_metrics_overlay:
        print("Metrics calculation enabled")
        frame_metrics = []
        # Running totals for cumulative metrics
        cumulative_sums = {
            'left_elbow': 0.0,
            'right_elbow': 0.0,
            'max_reach': 0.0,
            'extension': 0.0,
        }
        metrics_count = 0
    else:
        frame_metrics = None
        cumulative_sums = None
        metrics_count = 0

    # Initialize video reader
    with VideoReader(input_path) as reader:
        print(
            f"Video info: {reader.width}x{reader.height} @ {reader.fps} fps, {reader.frame_count} frames"
        )

        # Initialize video writer
        with VideoWriter(
            output_path, fps=reader.fps, width=reader.width, height=reader.height
        ) as writer:

            # Initialize pose engine
            with PoseEngine(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ) as engine:

                detected_frames = 0
                frame_num = 0

                # Create progress bar with tqdm
                iterator = tqdm(
                    total=reader.frame_count,
                    desc="Processing",
                    unit="frame",
                    disable=not show_progress,
                )

                # Process each frame
                while True:
                    success, frame = reader.read()
                    if not success:
                        break

                    frame_num += 1

                    # Run pose detection
                    results = engine.process(frame)

                    # Apply smoothing if enabled and extract landmarks for metrics
                    if results and results.pose_landmarks and smoother:
                        # Extract landmarks
                        raw_landmarks = engine.extract_landmarks(results)
                        
                        # Smooth landmarks
                        smoothed_landmarks = smoother.smooth(raw_landmarks)
                        
                        # Use smoothed landmarks for metrics if calculating
                        landmarks_for_metrics = smoothed_landmarks
                    elif results and results.pose_landmarks:
                        landmarks_for_metrics = engine.extract_landmarks(results)
                    else:
                        landmarks_for_metrics = None
                    
                    # Calculate metrics if enabled
                    if (calculate_metrics or show_metrics_overlay) and landmarks_for_metrics:
                        metrics = ClimbingMetrics.calculate_all_metrics(landmarks_for_metrics)
                        metrics["frame"] = frame_num
                        metrics["timestamp"] = frame_num / reader.fps
                        
                        if calculate_metrics:
                            frame_metrics.append(metrics)
                        
                        # Update cumulative metrics
                        if cumulative_sums is not None:
                            cumulative_sums['left_elbow'] += metrics['left_elbow_angle']
                            cumulative_sums['right_elbow'] += metrics['right_elbow_angle']
                            cumulative_sums['max_reach'] += metrics['max_reach']
                            cumulative_sums['extension'] += metrics['body_extension']
                            metrics_count += 1
                    else:
                        metrics = None

                    # Draw pose landmarks on frame (body only, no head)
                    if results and results.pose_landmarks:
                        annotated_frame = draw_pose_landmarks(
                            frame,
                            results,
                            connections=connections,
                            landmarks_to_draw=landmarks,
                        )
                        detected_frames += 1
                    else:
                        # No pose detected, use original frame
                        annotated_frame = frame
                    
                    # Add metrics overlay if enabled
                    if show_metrics_overlay and metrics:
                        # Calculate current averages
                        cumulative_metrics = None
                        if metrics_count > 0:
                            cumulative_metrics = {
                                'avg_left_elbow': cumulative_sums['left_elbow'] / metrics_count,
                                'avg_right_elbow': cumulative_sums['right_elbow'] / metrics_count,
                                'avg_max_reach': cumulative_sums['max_reach'] / metrics_count,
                                'avg_extension': cumulative_sums['extension'] / metrics_count,
                            }
                        
                        annotated_frame = draw_metrics_overlay(
                            annotated_frame,
                            current_metrics=metrics,
                            cumulative_metrics=cumulative_metrics
                        )

                    # Write to output video
                    writer.write(annotated_frame)

                    # Update progress bar
                    iterator.update(1)
                    iterator.set_postfix({"detected": detected_frames})

                iterator.close()
                
                # Save metrics to file if requested
                if calculate_metrics and frame_metrics:
                    if metrics_output is None:
                        # Default: save next to output video
                        metrics_output = Path(output_path).with_suffix('.json')
                    
                    print(f"\nSaving metrics to: {metrics_output}")
                    with open(metrics_output, 'w') as f:
                        json.dump({
                            "video": input_path,
                            "fps": reader.fps,
                            "total_frames": frame_num,
                            "detected_frames": detected_frames,
                            "metrics": frame_metrics
                        }, f, indent=2)
                    
                    # Print summary statistics
                    if frame_metrics:
                        print("\nMetrics Summary:")
                        avg_left_elbow = sum(m["left_elbow_angle"] for m in frame_metrics) / len(frame_metrics)
                        avg_right_elbow = sum(m["right_elbow_angle"] for m in frame_metrics) / len(frame_metrics)
                        avg_max_reach = sum(m["max_reach"] for m in frame_metrics) / len(frame_metrics)
                        print(f"  Average left elbow angle: {avg_left_elbow:.1f}°")
                        print(f"  Average right elbow angle: {avg_right_elbow:.1f}°")
                        print(f"  Average max reach: {avg_max_reach:.3f}")
                
                # Final summary
                print(f"\nProcessing complete!")
                print(f"Total frames processed: {frame_num}")
                print(f"Frames with pose detected: {detected_frames}")
                if frame_num > 0:
                    detection_rate = (detected_frames / frame_num) * 100
                    print(f"Detection rate: {detection_rate:.1f}%")
                print(f"Output saved to: {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process a video through the climb-sensei pose estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/process_video.py input.mp4 output.mp4
  
  # Adjust confidence thresholds
  python scripts/process_video.py input.mp4 output.mp4 --detection-conf 0.7 --tracking-conf 0.6
  
  # Quiet mode (no progress updates)
  python scripts/process_video.py input.mp4 output.mp4 --quiet
        """,
    )

    parser.add_argument("input", help="Path to input video file")

    parser.add_argument("output", help="Path to save output video file")

    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.5,
        help="Minimum detection confidence (0.0-1.0, default: 0.5)",
    )

    parser.add_argument(
        "--tracking-conf",
        type=float,
        default=0.5,
        help="Minimum tracking confidence (0.0-1.0, default: 0.5)",
    )

    parser.add_argument(
        "--no-smoothing", action="store_true", help="Disable One Euro Filter smoothing"
    )

    parser.add_argument(
        "--smoothing-cutoff",
        type=float,
        default=1.0,
        help="Smoothing min cutoff frequency (default: 1.0, lower = more smoothing)",
    )

    parser.add_argument(
        "--smoothing-beta",
        type=float,
        default=0.007,
        help="Smoothing speed coefficient (default: 0.007, higher = more adaptive)",
    )
    
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Overlay metrics on video (shows current and average values)",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress updates"
    )

    parser.add_argument(
        "--include-head",
        action="store_true",
        help="Include head keypoints (default: False, only body keypoints for climbing)",
    )
    
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Calculate climbing metrics (joint angles, reach, etc.)",
    )
    
    parser.add_argument(
        "--metrics-output",
        type=str,
        help="Path to save metrics JSON file (default: same as output video with .json extension)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate confidence values
    if not 0.0 <= args.detection_conf <= 1.0:
        print("Error: detection-conf must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.tracking_conf <= 1.0:
        print("Error: tracking-conf must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    try:
        process_video(
            args.input,
            args.output,
            min_detection_confidence=args.detection_conf,
            min_tracking_confidence=args.tracking_conf,
            show_progress=not args.quiet,
            include_head=args.include_head,
            enable_smoothing=not args.no_smoothing,
            smoothing_min_cutoff=args.smoothing_cutoff,
            smoothing_beta=args.smoothing_beta,
            calculate_metrics=args.metrics,
            metrics_output=args.metrics_output,
            show_metrics_overlay=args.show_metrics,
        )
    except Exception as e:
        print(f"Error processing video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
