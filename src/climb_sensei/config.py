"""Configuration for pose visualization and analysis.

This module defines which landmarks and connections are relevant
for climbing movement analysis, excluding head/face keypoints.
"""

from typing import FrozenSet, Tuple

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

# Landmarks to include for climbing analysis (body only, no head)
CLIMBING_LANDMARKS = frozenset(range(11, 33))  # Landmarks 11-32

# Pose connections relevant for climbing movements
# Excludes all head/face connections
CLIMBING_CONNECTIONS: FrozenSet[Tuple[int, int]] = frozenset([
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
])

# Full MediaPipe connections (including head) for reference
FULL_POSE_CONNECTIONS: FrozenSet[Tuple[int, int]] = frozenset([
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (11, 23), (12, 14), (12, 24), (13, 15), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20),
    (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (27, 31),
    (28, 30), (28, 32), (29, 31), (30, 32)
])


def get_landmark_name(index: int) -> str:
    """Get the human-readable name for a landmark index.
    
    Args:
        index: Landmark index (0-32).
    
    Returns:
        Human-readable landmark name.
    """
    landmark_names = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        17: "left_pinky", 18: "right_pinky",
        19: "left_index", 20: "right_index",
        21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle",
        29: "left_heel", 30: "right_heel",
        31: "left_foot_index", 32: "right_foot_index",
    }
    return landmark_names.get(index, f"unknown_{index}")
