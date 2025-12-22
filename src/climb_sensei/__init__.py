"""climb-sensei: A Python pose estimation tool for climbing.

This package provides computer vision tools for analyzing climbing footage,
extracting pose data, calculating biomechanical metrics, and visualizing results.
"""

from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader, VideoWriter
from climb_sensei.biomechanics import (
    calculate_joint_angle,
    calculate_reach_distance,
    calculate_limb_angles,
    calculate_total_distance_traveled,
)
from climb_sensei.viz import draw_pose_landmarks
from climb_sensei.config import (
    CLIMBING_LANDMARKS,
    CLIMBING_CONNECTIONS,
    FULL_POSE_CONNECTIONS,
    get_landmark_name,
    LandmarkIndex,
    PoseConfig,
    MetricsConfig,
    VisualizationConfig,
)
from climb_sensei.metrics import ClimbingAnalyzer, AdvancedClimbingMetrics
from climb_sensei.metrics_viz import (
    create_metric_plot,
    create_metrics_dashboard,
    compose_frame_with_dashboard,
    overlay_metrics_on_frame,
    draw_metric_text_overlay,
)
from climb_sensei.models import (
    Landmark,
    FrameMetrics,
    ClimbingSummary,
    ClimbingAnalysis,
)
from climb_sensei.protocols import (
    PoseDetector,
    MetricsAnalyzer,
    AnalysisRepository,
)
from climb_sensei.facade import ClimbingSensei
from climb_sensei.builder import ClimbingAnalyzerBuilder
from climb_sensei.repository import JSONRepository, CSVRepository
from climb_sensei.video_quality import (
    VideoQualityChecker,
    VideoQualityReport,
    check_video_quality,
)
from climb_sensei.tracking_quality import (
    TrackingQualityAnalyzer,
    TrackingQualityReport,
    analyze_tracking_quality,
    analyze_tracking_from_landmarks,
)

__version__ = "0.3.0"

__all__ = [
    "PoseEngine",
    "VideoReader",
    "VideoWriter",
    "calculate_joint_angle",
    "calculate_reach_distance",
    "calculate_limb_angles",
    "calculate_total_distance_traveled",
    "draw_pose_landmarks",
    "CLIMBING_LANDMARKS",
    "CLIMBING_CONNECTIONS",
    "FULL_POSE_CONNECTIONS",
    "get_landmark_name",
    "LandmarkIndex",
    "PoseConfig",
    "MetricsConfig",
    "VisualizationConfig",
    "ClimbingAnalyzer",
    "AdvancedClimbingMetrics",
    "create_metric_plot",
    "create_metrics_dashboard",
    "compose_frame_with_dashboard",
    "overlay_metrics_on_frame",
    "draw_metric_text_overlay",
    # New typed models
    "Landmark",
    "FrameMetrics",
    "ClimbingSummary",
    "ClimbingAnalysis",
    # Protocols for extensibility
    "PoseDetector",
    "MetricsAnalyzer",
    "AnalysisRepository",
    # Phase 2: Simplified API patterns
    "ClimbingSensei",
    "ClimbingAnalyzerBuilder",
    "JSONRepository",
    "CSVRepository",
    # Video quality checking
    "VideoQualityChecker",
    "VideoQualityReport",
    "check_video_quality",
    # Tracking quality analysis
    "TrackingQualityAnalyzer",
    "TrackingQualityReport",
    "analyze_tracking_quality",
    "analyze_tracking_from_landmarks",
]
