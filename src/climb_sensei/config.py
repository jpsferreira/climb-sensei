"""Configuration for pose visualization and analysis.

This module defines application-wide configuration including:
- Pose landmarks and connections for climbing analysis
- Model configuration (confidence thresholds, etc.)
- Metrics calculation parameters
- Visualization styling (colors, dimensions, etc.)
"""

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import FrozenSet, Tuple


# ==============================================================================
# LANDMARK DEFINITIONS
# ==============================================================================

# MediaPipe Pose Landmarks (0-32):
# 0-10: Face/Head (nose, eyes, ears, mouth) - NOT RELEVANT FOR CLIMBING
# 11-32: Body landmarks - RELEVANT FOR CLIMBING
#
# Body Landmarks:
# 11, 12: Shoulders (left, right)
# 13, 14: Elbows (left, right)
# 15, 16: Wrists (left, right)
# 17, 18: Pinky fingers (left, right)
# 19, 20: Index fingers (left, right)
# 21, 22: Thumbs (left, right)
# 23, 24: Hips (left, right)
# 25, 26: Knees (left, right)
# 27, 28: Ankles (left, right)
# 29, 30: Heels (left, right)
# 31, 32: Foot indices (left, right)


class LandmarkIndex(IntEnum):
    """MediaPipe Pose landmark indices.

    Centralized definition of landmark indices for consistency across modules.
    """

    # Face/Head (0-10)
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10

    # Upper Body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Hands
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22

    # Lower Body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Landmark groups for visualization color-coding
FACE_LANDMARKS = frozenset(range(0, 11))  # 0-10
TORSO_LANDMARKS = frozenset(
    [
        LandmarkIndex.LEFT_SHOULDER,
        LandmarkIndex.RIGHT_SHOULDER,
        LandmarkIndex.LEFT_HIP,
        LandmarkIndex.RIGHT_HIP,
    ]
)
LEFT_ARM_LANDMARKS = frozenset(
    [
        LandmarkIndex.LEFT_SHOULDER,
        LandmarkIndex.LEFT_ELBOW,
        LandmarkIndex.LEFT_WRIST,
        LandmarkIndex.LEFT_PINKY,
        LandmarkIndex.LEFT_INDEX,
        LandmarkIndex.LEFT_THUMB,
    ]
)
RIGHT_ARM_LANDMARKS = frozenset(
    [
        LandmarkIndex.RIGHT_SHOULDER,
        LandmarkIndex.RIGHT_ELBOW,
        LandmarkIndex.RIGHT_WRIST,
        LandmarkIndex.RIGHT_PINKY,
        LandmarkIndex.RIGHT_INDEX,
        LandmarkIndex.RIGHT_THUMB,
    ]
)
LEFT_LEG_LANDMARKS = frozenset(
    [
        LandmarkIndex.LEFT_HIP,
        LandmarkIndex.LEFT_KNEE,
        LandmarkIndex.LEFT_ANKLE,
        LandmarkIndex.LEFT_HEEL,
        LandmarkIndex.LEFT_FOOT_INDEX,
    ]
)
RIGHT_LEG_LANDMARKS = frozenset(
    [
        LandmarkIndex.RIGHT_HIP,
        LandmarkIndex.RIGHT_KNEE,
        LandmarkIndex.RIGHT_ANKLE,
        LandmarkIndex.RIGHT_HEEL,
        LandmarkIndex.RIGHT_FOOT_INDEX,
    ]
)

# Landmarks to include for climbing analysis (body only, no head)
CLIMBING_LANDMARKS = frozenset(range(11, 33))  # Landmarks 11-32


# ==============================================================================
# POSE CONNECTIONS
# ==============================================================================

# Pose connections relevant for climbing movements
# Excludes all head/face connections
CLIMBING_CONNECTIONS: FrozenSet[Tuple[int, int]] = frozenset(
    [
        # Torso
        (11, 12),  # Shoulders
        (11, 23),  # Left shoulder to left hip
        (12, 24),  # Right shoulder to right hip
        (23, 24),  # Hips
        # Left arm
        (11, 13),  # Left shoulder to left elbow
        (13, 15),  # Left elbow to left wrist
        (15, 17),  # Left wrist to left pinky
        (15, 19),  # Left wrist to left index
        (15, 21),  # Left wrist to left thumb
        (17, 19),  # Left pinky to left index
        # Right arm
        (12, 14),  # Right shoulder to right elbow
        (14, 16),  # Right elbow to right wrist
        (16, 18),  # Right wrist to right pinky
        (16, 20),  # Right wrist to right index
        (16, 22),  # Right wrist to right thumb
        (18, 20),  # Right pinky to right index
        # Left leg
        (23, 25),  # Left hip to left knee
        (25, 27),  # Left knee to left ankle
        (27, 29),  # Left ankle to left heel
        (27, 31),  # Left ankle to left foot index
        (29, 31),  # Left heel to left foot index
        # Right leg
        (24, 26),  # Right hip to right knee
        (26, 28),  # Right knee to right ankle
        (28, 30),  # Right ankle to right heel
        (28, 32),  # Right ankle to right foot index
        (30, 32),  # Right heel to right foot index
    ]
)

# Full MediaPipe connections (including head) for reference
FULL_POSE_CONNECTIONS: FrozenSet[Tuple[int, int]] = frozenset(
    [
        (0, 1),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 7),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (11, 23),
        (12, 14),
        (12, 24),
        (13, 15),
        (14, 16),
        (15, 17),
        (15, 19),
        (15, 21),
        (16, 18),
        (16, 20),
        (16, 22),
        (17, 19),
        (18, 20),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (27, 31),
        (28, 30),
        (28, 32),
        (29, 31),
        (30, 32),
    ]
)


# ==============================================================================
# APPLICATION CONFIGURATION
# ==============================================================================


@dataclass(frozen=True)
class PoseConfig:
    """Immutable pose detection and tracking configuration.

    Attributes:
        min_detection_confidence: Minimum confidence for pose detection (0.0-1.0)
        min_tracking_confidence: Minimum confidence for pose tracking (0.0-1.0)
        timestamp_increment_ms: Milliseconds per frame for temporal smoothing
    """

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    timestamp_increment_ms: int = 33  # ~30fps

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class MetricsConfig:
    """Immutable metrics calculation configuration.

    Attributes:
        lock_off_threshold_degrees: Elbow angle threshold for lock-off detection
        rest_velocity_threshold: Max velocity for rest position detection
        rest_body_angle_threshold: Max body angle for rest position (degrees)
        efficient_economy_ratio: Threshold for efficient movement economy
        fatigue_window_size: Number of frames for fatigue analysis
        com_body_weight: Weight factor for center of mass calculation
    """

    lock_off_threshold_degrees: float = 90.0
    lock_off_velocity_threshold: float = 0.002
    rest_velocity_threshold: float = 0.01
    rest_body_angle_threshold: float = 15.0
    efficient_economy_ratio: float = 0.8
    fatigue_window_size: int = 90
    com_body_weight: float = 1.0


# Immutable color mapping for visualization (module-level constant)
COLORS: MappingProxyType[str, Tuple[int, int, int]] = MappingProxyType(
    {
        "face": (255, 255, 255),  # White
        "torso": (0, 255, 255),  # Yellow
        "left_arm": (0, 255, 0),  # Green
        "right_arm": (0, 0, 255),  # Red
        "left_leg": (255, 0, 255),  # Magenta
        "right_leg": (255, 128, 0),  # Orange
        "default": (200, 200, 200),  # Gray
        "connection": (255, 255, 255),  # White
    }
)


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization styling configuration (immutable)."""

    # Drawing dimensions
    line_thickness: int = 3
    circle_radius: int = 7
    landmark_border_thickness: int = 2

    # Text styling
    font_scale: float = 3.0
    font_thickness: int = 2

    # Metrics overlay layout
    metrics_overlay_padding: int = 15
    metrics_line_height: int = 50
    metrics_overlay_bg_alpha: float = 0.7

    # Angle annotation
    angle_annotation_padding: int = 5

    # Plot settings for metrics dashboard
    plot_width: int = 500
    plot_height: int = 150
    plot_background_color: Tuple[int, int, int] = (40, 40, 40)
    plot_margin_left: int = 70
    plot_margin_right: int = 15
    plot_margin_top: int = 40
    plot_margin_bottom: int = 25

    # Plot text styling
    plot_title_font_scale: float = 0.8
    plot_title_thickness: int = 2
    plot_title_color: Tuple[int, int, int] = (200, 200, 200)
    plot_label_font_scale: float = 0.5
    plot_label_thickness: int = 2
    plot_label_color: Tuple[int, int, int] = (150, 150, 150)

    # Plot elements
    plot_grid_color: Tuple[int, int, int] = (60, 60, 60)
    plot_line_thickness: int = 2
    plot_current_marker_inner_color: Tuple[int, int, int] = (255, 255, 255)
    plot_current_marker_inner_radius: int = 4
    plot_current_marker_outer_radius: int = 6
    plot_current_marker_outer_thickness: int = 2
    plot_current_line_color: Tuple[int, int, int] = (100, 100, 100)


VIZ = VisualizationConfig()


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_landmark_name(index: int) -> str:
    """Get the human-readable name for a landmark index.

    Args:
        index: Landmark index (0-32).

    Returns:
        Human-readable landmark name.
    """
    try:
        return LandmarkIndex(index).name.lower()
    except ValueError:
        return f"unknown_{index}"
