"""Joint Angle Calculator - Joint angle measurements.

Calculates angles for all major joints:
- Elbows (left, right)
- Shoulders (left, right)
- Knees (left, right)
- Hips (left, right)
"""

from typing import List, Dict, Any
import numpy as np

from .base import BaseCalculator
from ...config import LandmarkIndex
from ...biomechanics import calculate_joint_angle


class JointAngleCalculator(BaseCalculator):
    """Calculator for joint angle measurements.

    Tracks all major joint angles throughout the climb.
    Useful for biomechanical analysis and injury prevention.

    Usage:
        >>> calc = JointAngleCalculator(fps=30.0)
        >>> for landmarks in sequence:
        ...     metrics = calc.calculate(landmarks)
        ...     print(f"Left elbow: {metrics['left_elbow']:.1f}°")
        >>> summary = calc.get_summary()
        >>> print(f"Min left elbow: {summary['min_left_elbow']:.1f}°")
    """

    def __init__(self, window_size: int = 30, fps: float = 30.0):
        """Initialize joint angle calculator.

        Args:
            window_size: Number of frames for moving window (unused)
            fps: Frames per second
        """
        super().__init__(window_size, fps)

    def calculate(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate joint angles for one frame.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Dictionary with all joint angles in degrees
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1

        metrics = {}

        # Elbow angles
        metrics["left_elbow"] = self._calculate_elbow_angle(landmarks, left=True)
        metrics["right_elbow"] = self._calculate_elbow_angle(landmarks, left=False)

        # Shoulder angles
        metrics["left_shoulder"] = self._calculate_shoulder_angle(landmarks, left=True)
        metrics["right_shoulder"] = self._calculate_shoulder_angle(
            landmarks, left=False
        )

        # Knee angles
        metrics["left_knee"] = self._calculate_knee_angle(landmarks, left=True)
        metrics["right_knee"] = self._calculate_knee_angle(landmarks, left=False)

        # Hip angles
        metrics["left_hip"] = self._calculate_hip_angle(landmarks, left=True)
        metrics["right_hip"] = self._calculate_hip_angle(landmarks, left=False)

        # Track history
        for key, value in metrics.items():
            self._append_to_history(key, value)

        return metrics

    def _calculate_elbow_angle(
        self, landmarks: List[Dict[str, float]], left: bool
    ) -> float:
        """Calculate elbow angle.

        Args:
            landmarks: List of landmark dictionaries
            left: True for left elbow, False for right

        Returns:
            Elbow angle in degrees
        """
        if left:
            shoulder_idx = LandmarkIndex.LEFT_SHOULDER
            elbow_idx = LandmarkIndex.LEFT_ELBOW
            wrist_idx = LandmarkIndex.LEFT_WRIST
        else:
            shoulder_idx = LandmarkIndex.RIGHT_SHOULDER
            elbow_idx = LandmarkIndex.RIGHT_ELBOW
            wrist_idx = LandmarkIndex.RIGHT_WRIST

        shoulder = (landmarks[shoulder_idx]["x"], landmarks[shoulder_idx]["y"])
        elbow = (landmarks[elbow_idx]["x"], landmarks[elbow_idx]["y"])
        wrist = (landmarks[wrist_idx]["x"], landmarks[wrist_idx]["y"])

        return calculate_joint_angle(shoulder, elbow, wrist)

    def _calculate_shoulder_angle(
        self, landmarks: List[Dict[str, float]], left: bool
    ) -> float:
        """Calculate shoulder angle.

        Args:
            landmarks: List of landmark dictionaries
            left: True for left shoulder, False for right

        Returns:
            Shoulder angle in degrees
        """
        if left:
            hip_idx = LandmarkIndex.LEFT_HIP
            shoulder_idx = LandmarkIndex.LEFT_SHOULDER
            elbow_idx = LandmarkIndex.LEFT_ELBOW
        else:
            hip_idx = LandmarkIndex.RIGHT_HIP
            shoulder_idx = LandmarkIndex.RIGHT_SHOULDER
            elbow_idx = LandmarkIndex.RIGHT_ELBOW

        hip = (landmarks[hip_idx]["x"], landmarks[hip_idx]["y"])
        shoulder = (landmarks[shoulder_idx]["x"], landmarks[shoulder_idx]["y"])
        elbow = (landmarks[elbow_idx]["x"], landmarks[elbow_idx]["y"])

        return calculate_joint_angle(hip, shoulder, elbow)

    def _calculate_knee_angle(
        self, landmarks: List[Dict[str, float]], left: bool
    ) -> float:
        """Calculate knee angle.

        Args:
            landmarks: List of landmark dictionaries
            left: True for left knee, False for right

        Returns:
            Knee angle in degrees
        """
        if left:
            hip_idx = LandmarkIndex.LEFT_HIP
            knee_idx = LandmarkIndex.LEFT_KNEE
            ankle_idx = LandmarkIndex.LEFT_ANKLE
        else:
            hip_idx = LandmarkIndex.RIGHT_HIP
            knee_idx = LandmarkIndex.RIGHT_KNEE
            ankle_idx = LandmarkIndex.RIGHT_ANKLE

        hip = (landmarks[hip_idx]["x"], landmarks[hip_idx]["y"])
        knee = (landmarks[knee_idx]["x"], landmarks[knee_idx]["y"])
        ankle = (landmarks[ankle_idx]["x"], landmarks[ankle_idx]["y"])

        return calculate_joint_angle(hip, knee, ankle)

    def _calculate_hip_angle(
        self, landmarks: List[Dict[str, float]], left: bool
    ) -> float:
        """Calculate hip angle.

        Args:
            landmarks: List of landmark dictionaries
            left: True for left hip, False for right

        Returns:
            Hip angle in degrees
        """
        if left:
            shoulder_idx = LandmarkIndex.LEFT_SHOULDER
            hip_idx = LandmarkIndex.LEFT_HIP
            knee_idx = LandmarkIndex.LEFT_KNEE
        else:
            shoulder_idx = LandmarkIndex.RIGHT_SHOULDER
            hip_idx = LandmarkIndex.RIGHT_HIP
            knee_idx = LandmarkIndex.RIGHT_KNEE

        shoulder = (landmarks[shoulder_idx]["x"], landmarks[shoulder_idx]["y"])
        hip = (landmarks[hip_idx]["x"], landmarks[hip_idx]["y"])
        knee = (landmarks[knee_idx]["x"], landmarks[knee_idx]["y"])

        return calculate_joint_angle(shoulder, hip, knee)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for joint angles.

        Returns:
            Dictionary with min, max, avg for each joint
        """
        summary = {}

        joint_names = [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]

        for joint in joint_names:
            if joint in self._history and self._history[joint]:
                values = [
                    v for v in self._history[joint] if isinstance(v, (int, float))
                ]
                if values:
                    summary[f"min_{joint}"] = float(np.min(values))
                    summary[f"max_{joint}"] = float(np.max(values))
                    summary[f"avg_{joint}"] = float(np.mean(values))

        return summary
