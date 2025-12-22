"""climb-sensei: A Python pose estimation tool for climbing.

This package provides computer vision tools for analyzing climbing footage,
extracting pose data, calculating biomechanical metrics, and visualizing results.

Quick Start:
    >>> from climb_sensei import ClimbingSensei
    >>> sensei = ClimbingSensei("climbing_video.mp4")
    >>> analysis = sensei.analyze()
"""

# Core I/O
from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader, VideoWriter

# Biomechanics utilities
from climb_sensei.biomechanics import (
    calculate_joint_angle,
    calculate_reach_distance,
    calculate_limb_angles,
    calculate_total_distance_traveled,
)

# Configuration and constants
from climb_sensei.config import (
    CLIMBING_LANDMARKS,
    CLIMBING_CONNECTIONS,
    get_landmark_name,
    LandmarkIndex,
    MetricsConfig,
)

# Primary API (facade pattern - recommended)
from climb_sensei.facade import ClimbingSensei

# Core analysis
from climb_sensei.metrics import ClimbingAnalyzer

# Repository patterns
from climb_sensei.repository import JSONRepository, CSVRepository

# Quality checking
from climb_sensei.video_quality import (
    VideoQualityReport,
    check_video_quality,
)
from climb_sensei.tracking_quality import (
    TrackingQualityReport,
    analyze_tracking_quality,
    analyze_tracking_from_landmarks,
)

# Data models
from climb_sensei.models import (
    Landmark,
    FrameMetrics,
    ClimbingSummary,
    ClimbingAnalysis,
)

# Visualization
from climb_sensei.viz import draw_pose_landmarks
from climb_sensei.metrics_viz import (
    create_metrics_dashboard,
    overlay_metrics_on_frame,
)

__version__ = "0.3.0"

# Public API - organized by category
__all__ = [
    # === PRIMARY API (Recommended) ===
    "ClimbingSensei",  # Main facade - easiest way to use climb-sensei
    # === REPOSITORIES ===
    "JSONRepository",  # Save/load analysis as JSON
    "CSVRepository",  # Save/load analysis as CSV
    # === QUALITY CHECKING ===
    "check_video_quality",  # Validate video before processing
    "analyze_tracking_quality",  # Check pose tracking quality
    "analyze_tracking_from_landmarks",  # Quality check from existing data
    "VideoQualityReport",  # Video quality results
    "TrackingQualityReport",  # Tracking quality results
    # === CORE ANALYSIS ===
    "ClimbingAnalyzer",  # Main metrics analyzer
    "PoseEngine",  # Pose detection engine
    # === DATA MODELS ===
    "Landmark",  # Single pose landmark
    "FrameMetrics",  # Per-frame metrics
    "ClimbingSummary",  # Aggregate summary
    "ClimbingAnalysis",  # Complete analysis
    # === VIDEO I/O ===
    "VideoReader",  # Read video frames
    "VideoWriter",  # Write video output
    # === BIOMECHANICS ===
    "calculate_joint_angle",  # Angle between 3 points
    "calculate_reach_distance",  # Distance between landmarks
    "calculate_limb_angles",  # All limb angles
    "calculate_total_distance_traveled",  # Movement distance
    # === VISUALIZATION ===
    "draw_pose_landmarks",  # Draw skeleton on frame
    "create_metrics_dashboard",  # Create metrics charts
    "overlay_metrics_on_frame",  # Overlay dashboard on video
    # === CONFIGURATION ===
    "CLIMBING_LANDMARKS",  # Landmark indices
    "CLIMBING_CONNECTIONS",  # Skeleton connections
    "get_landmark_name",  # Get landmark name by index
    "LandmarkIndex",  # Landmark enum
    "MetricsConfig",  # Configure metrics thresholds
]
