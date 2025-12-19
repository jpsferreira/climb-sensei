"""Configuration for pose visualization and analysis.

This module defines application-wide configuration including:
- Pose landmarks and connections for climbing analysis
- Model configuration (confidence thresholds, etc.)
- Metrics calculation parameters
- Visualization styling (colors, dimensions, etc.)
"""

from typing import FrozenSet, Tuple, Dict


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


class LandmarkIndex:
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


class PoseConfig:
    """Pose detection and tracking configuration."""

    # Model confidence thresholds
    DEFAULT_DETECTION_CONFIDENCE = 0.5
    DEFAULT_TRACKING_CONFIDENCE = 0.5

    # Video mode temporal smoothing
    TIMESTAMP_INCREMENT_MS = 33  # ~30fps (33ms per frame)


class MetricsConfig:
    """Metrics calculation configuration."""

    # Lock-off detection threshold (degrees)
    # An elbow angle less than this indicates a lock-off position
    LOCK_OFF_THRESHOLD_DEGREES = 90

    # Rest position thresholds
    REST_VELOCITY_THRESHOLD = 0.01  # COM velocity threshold for static positions
    REST_BODY_ANGLE_THRESHOLD = 15  # Max body angle (degrees) for rest position

    # Movement economy thresholds
    EFFICIENT_ECONOMY_RATIO = (
        0.8  # vertical_progress / total_distance (higher = better)
    )

    # Fatigue detection window (frames)
    FATIGUE_WINDOW_SIZE = 90  # 3 seconds at 30fps

    # Center of mass calculation - body part weights (normalized)
    COM_BODY_WEIGHT = 1.0


class VisualizationConfig:
    """Visualization styling configuration."""

    # Body part colors (BGR format for OpenCV)
    COLORS: Dict[str, Tuple[int, int, int]] = {
        "face": (255, 255, 255),  # White
        "torso": (0, 255, 255),  # Yellow
        "left_arm": (0, 255, 0),  # Green
        "right_arm": (0, 0, 255),  # Red
        "left_leg": (255, 0, 255),  # Magenta
        "right_leg": (255, 128, 0),  # Orange
        "default": (200, 200, 200),  # Gray
        "connection": (255, 255, 255),  # White
    }

    # Drawing dimensions
    DEFAULT_LINE_THICKNESS = 2
    DEFAULT_CIRCLE_RADIUS = 5
    LANDMARK_BORDER_THICKNESS = 1

    # Text styling
    DEFAULT_FONT_SCALE = 0.6
    DEFAULT_FONT_THICKNESS = 2

    # Metrics overlay layout
    METRICS_OVERLAY_PADDING = 10
    METRICS_LINE_HEIGHT = 25
    METRICS_OVERLAY_BG_ALPHA = 0.7

    # Angle annotation
    ANGLE_ANNOTATION_PADDING = 5


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
    landmark_names = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
        29: "left_heel",
        30: "right_heel",
        31: "left_foot_index",
        32: "right_foot_index",
    }
    return landmark_names.get(index, f"unknown_{index}")
