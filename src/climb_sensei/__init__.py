"""climb-sensei: A Python pose estimation tool for climbing.

This package provides computer vision tools for analyzing climbing footage,
extracting pose data, calculating biomechanical metrics, and visualizing results.

Service-Oriented Architecture:
    >>> from climb_sensei.services import (
    ...     VideoQualityService,
    ...     TrackingQualityService,
    ...     ClimbingAnalysisService,
    ... )
    >>> from climb_sensei.pose_engine import PoseEngine
    >>>
    >>> # Independent, composable services
    >>> video_quality = VideoQualityService()
    >>> tracking = TrackingQualityService()
    >>> climbing = ClimbingAnalysisService()
    >>>
    >>> # Extract landmarks
    >>> pose_engine = PoseEngine()
    >>> # ... extract landmarks from video ...
    >>>
    >>> # Use services independently
    >>> analysis = climbing.analyze(landmarks, fps=30.0)
    >>> pose_engine.close()
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

# Core analysis (internal use by services)
from climb_sensei.metrics import ClimbingAnalyzer

# Repository patterns
from climb_sensei.repository import JSONRepository, CSVRepository

# Quality checking
from climb_sensei.video_quality import VideoQualityReport
from climb_sensei.tracking_quality import TrackingQualityReport

# Service Layer (PRIMARY API)
from climb_sensei.services import (
    VideoQualityService,
    TrackingQualityService,
    ClimbingAnalysisService,
)

# Domain Layer - Composable Calculators
from climb_sensei.domain.calculators import (
    MetricsCalculator,
    StabilityCalculator,
    ProgressCalculator,
    EfficiencyCalculator,
    TechniqueCalculator,
    JointAngleCalculator,
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
    # === PRIMARY API (Service-Oriented) ===
    "VideoQualityService",
    "TrackingQualityService",
    "ClimbingAnalysisService",
    # === DOMAIN LAYER (Calculators) ===
    "MetricsCalculator",
    "StabilityCalculator",
    "ProgressCalculator",
    "EfficiencyCalculator",
    "TechniqueCalculator",
    "JointAngleCalculator",
    # === REPOSITORIES ===
    "JSONRepository",
    "CSVRepository",
    # === QUALITY CHECKING ===
    "VideoQualityReport",
    "TrackingQualityReport",
    # === CORE COMPONENTS ===
    "PoseEngine",
    "VideoReader",
    "VideoWriter",
    "ClimbingAnalyzer",
    # === DATA MODELS ===
    "Landmark",
    "FrameMetrics",
    "ClimbingSummary",
    "ClimbingAnalysis",
    # === BIOMECHANICS UTILITIES ===
    "calculate_joint_angle",
    "calculate_reach_distance",
    "calculate_limb_angles",
    "calculate_total_distance_traveled",
    # === VISUALIZATION ===
    "draw_pose_landmarks",
    "create_metrics_dashboard",
    "overlay_metrics_on_frame",
    # === CONFIGURATION ===
    "CLIMBING_LANDMARKS",
    "CLIMBING_CONNECTIONS",
    "get_landmark_name",
    "LandmarkIndex",
    "MetricsConfig",
]
