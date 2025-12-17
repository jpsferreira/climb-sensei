"""climb-sensei: A professional Python pose estimation tool for climbing.

This package provides computer vision tools for analyzing climbing footage,
extracting pose data, calculating biomechanical metrics, and visualizing results.
"""

from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader, VideoWriter
from climb_sensei.biomechanics import calculate_joint_angle, calculate_reach_distance
from climb_sensei.viz import draw_pose_landmarks

__version__ = "0.1.0"

__all__ = [
    "PoseEngine",
    "VideoReader",
    "VideoWriter",
    "calculate_joint_angle",
    "calculate_reach_distance",
    "draw_pose_landmarks",
]
