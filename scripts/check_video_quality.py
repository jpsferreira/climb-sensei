#!/usr/bin/env python3
"""CLI tool for checking climbing video quality.

This script validates video files for climbing analysis, checking:
- Format compatibility
- Resolution and frame rate
- Duration
- Lighting conditions (with --deep flag)
- Camera stability (with --deep flag)

Usage:
    python scripts/check_video_quality.py video.mp4
    python scripts/check_video_quality.py video.mp4 --deep
    python scripts/check_video_quality.py video.mp4 --json output.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.video_quality import check_video_quality, VideoQualityReport


def format_report_text(report: VideoQualityReport) -> str:
    """Format quality report as readable text."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("VIDEO QUALITY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Overall status
    status = "✅ VALID" if report.is_valid else "❌ INVALID"
    lines.append(f"Status: {status}")
    lines.append("")

    # Basic properties
    lines.append("📹 Video Properties")
    lines.append("-" * 70)
    lines.append(f"  File: {Path(report.file_path).name}")
    lines.append(f"  Size: {report.file_size_mb} MB")
    lines.append(
        f"  Resolution: {report.width}x{report.height} ({report.resolution_quality})"
    )
    lines.append(f"  Frame Rate: {report.fps} fps ({report.fps_quality})")
    lines.append(f"  Duration: {report.duration_seconds}s ({report.duration_quality})")
    lines.append(f"  Codec: {report.codec}")
    lines.append(f"  Frames: {report.frame_count}")
    lines.append("")

    # Deep analysis results
    if report.lighting_quality or report.stability_quality:
        lines.append("🔍 Deep Analysis")
        lines.append("-" * 70)
        if report.lighting_quality:
            lines.append(f"  Lighting: {report.lighting_quality}")
        if report.stability_quality:
            lines.append(f"  Stability: {report.stability_quality}")
        lines.append("")

    # Issues
    if report.issues:
        lines.append("❌ Issues")
        lines.append("-" * 70)
        for issue in report.issues:
            lines.append(f"  • {issue}")
        lines.append("")

    # Warnings
    if report.warnings:
        lines.append("⚠️  Warnings")
        lines.append("-" * 70)
        for warning in report.warnings:
            lines.append(f"  • {warning}")
        lines.append("")

    # Recommendations
    if report.recommendations:
        lines.append("💡 Recommendations")
        lines.append("-" * 70)
        for rec in report.recommendations:
            lines.append(f"  • {rec}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def report_to_dict(report: VideoQualityReport) -> Dict[str, Any]:
    """Convert report to dictionary for JSON export."""
    return {
        "is_valid": report.is_valid,
        "file_path": report.file_path,
        "file_size_mb": report.file_size_mb,
        "properties": {
            "width": report.width,
            "height": report.height,
            "fps": report.fps,
            "frame_count": report.frame_count,
            "duration_seconds": report.duration_seconds,
            "codec": report.codec,
        },
        "quality_assessment": {
            "format_compatible": report.format_compatible,
            "resolution_quality": report.resolution_quality,
            "fps_quality": report.fps_quality,
            "duration_quality": report.duration_quality,
            "lighting_quality": report.lighting_quality,
            "stability_quality": report.stability_quality,
        },
        "issues": report.issues,
        "warnings": report.warnings,
        "recommendations": report.recommendations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check video quality for climbing analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick check
  python scripts/check_video_quality.py climbing.mp4

  # Deep analysis (slower, checks lighting and stability)
  python scripts/check_video_quality.py climbing.mp4 --deep

  # Export to JSON
  python scripts/check_video_quality.py climbing.mp4 --json quality.json

  # Deep analysis with JSON export
  python scripts/check_video_quality.py climbing.mp4 --deep --json quality.json
        """,
    )

    parser.add_argument("video", type=str, help="Path to video file")

    parser.add_argument(
        "--deep",
        action="store_true",
        help="Perform deep frame-by-frame analysis (slower)",
    )

    parser.add_argument(
        "--json",
        type=str,
        metavar="OUTPUT",
        help="Export report as JSON to specified file",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Suppress text output (useful with --json)"
    )

    args = parser.parse_args()

    try:
        # Check video quality
        if not args.quiet:
            print(f"Analyzing video: {args.video}")
            if args.deep:
                print("Performing deep analysis (this may take a moment)...")
            print()

        report = check_video_quality(args.video, deep_check=args.deep)

        # Export JSON if requested
        if args.json:
            report_dict = report_to_dict(report)
            with open(args.json, "w") as f:
                json.dump(report_dict, f, indent=2)
            if not args.quiet:
                print(f"Report saved to: {args.json}\n")

        # Print text report
        if not args.quiet:
            print(format_report_text(report))

        # Exit with appropriate code
        sys.exit(0 if report.is_valid else 1)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
