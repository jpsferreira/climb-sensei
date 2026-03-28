"""Technique Calculator - Climbing technique metrics.

Calculates:
- Lock-off detection (static strength positions)
- Rest position detection (recovery positions)
- Body angle (lean from vertical)
- Hand span (distance between hands)
- Foot span (distance between feet)
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseCalculator, FrameContext
from ...config import LandmarkIndex
from ...biomechanics import calculate_reach_distance


class TechniqueCalculator(BaseCalculator):
    """Calculator for climbing technique metrics.

    Analyzes technique-specific aspects:
    - Lock-offs (one-arm static holds)
    - Rest positions (low-stress recovery)
    - Body positioning (lean angle, stance width)

    Usage:
        >>> calc = TechniqueCalculator(fps=30.0)
        >>> for landmarks in sequence:
        ...     metrics = calc.calculate(landmarks)
        ...     if metrics.get('is_lock_off'):
        ...         print("Lock-off detected!")
        >>> summary = calc.get_summary()
        >>> print(f"Lock-offs: {summary['total_lock_offs']}")
    """

    def __init__(
        self,
        window_size: int = 30,
        fps: float = 30.0,
        lock_off_elbow_threshold: float = 90.0,
        rest_position_angle_threshold: float = 150.0,
    ):
        """Initialize technique calculator.

        Args:
            window_size: Number of frames for moving window
            fps: Frames per second
            lock_off_elbow_threshold: Max elbow angle for lock-off (degrees)
            rest_position_angle_threshold: Min elbow angle for rest (degrees)
        """
        super().__init__(window_size, fps)
        self.lock_off_threshold = lock_off_elbow_threshold
        self.rest_threshold = rest_position_angle_threshold
        self._total_lock_offs = 0
        self._total_rest_positions = 0

    def calculate(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> Dict[str, Any]:
        """Calculate technique metrics for one frame.

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context with joint angles

        Returns:
            Dictionary with technique metrics
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1

        metrics = {}

        # Calculate body angle (lean from vertical)
        body_angle = self._calculate_body_angle(landmarks)
        metrics["body_angle"] = body_angle

        # Calculate hand and foot spans
        hand_span = self._calculate_hand_span(landmarks)
        foot_span = self._calculate_foot_span(landmarks)
        metrics["hand_span"] = hand_span
        metrics["foot_span"] = foot_span

        # Detect lock-offs using pre-computed angles if available
        left_lock, right_lock, is_lock_off = self._detect_lock_off(landmarks, context)
        metrics["is_lock_off"] = is_lock_off
        metrics["left_lock_off"] = left_lock
        metrics["right_lock_off"] = right_lock

        if is_lock_off:
            self._total_lock_offs += 1

        # Detect rest positions using pre-computed angles if available
        is_rest = self._detect_rest_position(landmarks, context)
        metrics["is_rest_position"] = is_rest

        if is_rest:
            self._total_rest_positions += 1

        # Track history
        for key, value in metrics.items():
            self._append_to_history(key, value)

        return metrics

    def _calculate_body_angle(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate body lean angle from vertical.

        Uses atan2 to preserve lean direction:
        - Positive = leaning right
        - Negative = leaning left
        - 0 = perfectly vertical
        - ±90 = horizontal

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Signed angle in degrees from vertical (-90 to +90)
        """
        # Use shoulder midpoint to hip midpoint vector
        shoulder_x = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["x"]
            + landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"]
        ) / 2
        shoulder_y = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["y"]
            + landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"]
        ) / 2

        hip_x = (
            landmarks[LandmarkIndex.LEFT_HIP]["x"]
            + landmarks[LandmarkIndex.RIGHT_HIP]["x"]
        ) / 2
        hip_y = (
            landmarks[LandmarkIndex.LEFT_HIP]["y"]
            + landmarks[LandmarkIndex.RIGHT_HIP]["y"]
        ) / 2

        # dx = lateral displacement, dy = vertical displacement (Y inverted in image)
        dx = shoulder_x - hip_x
        dy = hip_y - shoulder_y  # Positive = shoulders above hips (normal)

        # atan2(dx, dy) gives angle from vertical, preserving sign
        angle = np.degrees(np.arctan2(dx, dy))
        return float(angle)

    def _calculate_hand_span(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate distance between hands.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Distance between wrists (normalized)
        """
        left_wrist = (
            landmarks[LandmarkIndex.LEFT_WRIST]["x"],
            landmarks[LandmarkIndex.LEFT_WRIST]["y"],
        )
        right_wrist = (
            landmarks[LandmarkIndex.RIGHT_WRIST]["x"],
            landmarks[LandmarkIndex.RIGHT_WRIST]["y"],
        )

        return float(calculate_reach_distance(left_wrist, right_wrist))

    def _calculate_foot_span(self, landmarks: List[Dict[str, float]]) -> float:
        """Calculate distance between feet.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Distance between ankles (normalized)
        """
        left_ankle = (
            landmarks[LandmarkIndex.LEFT_ANKLE]["x"],
            landmarks[LandmarkIndex.LEFT_ANKLE]["y"],
        )
        right_ankle = (
            landmarks[LandmarkIndex.RIGHT_ANKLE]["x"],
            landmarks[LandmarkIndex.RIGHT_ANKLE]["y"],
        )

        return float(calculate_reach_distance(left_ankle, right_ankle))

    def _detect_lock_off(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> tuple:
        """Detect lock-off position (one arm bent, supporting weight).

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context with joint angles

        Returns:
            Tuple of (left_lock_off, right_lock_off, is_lock_off)
        """
        left_elbow_angle, right_elbow_angle = self._get_elbow_angles(landmarks, context)

        # Lock-off = elbow bent at < 90 degrees
        left_lock = left_elbow_angle < self.lock_off_threshold
        right_lock = right_elbow_angle < self.lock_off_threshold

        # At least one arm locked off
        is_lock_off = left_lock or right_lock

        return left_lock, right_lock, is_lock_off

    def _detect_rest_position(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> bool:
        """Detect rest position (straight arms, minimal effort).

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context with joint angles

        Returns:
            True if in rest position
        """
        left_elbow_angle, right_elbow_angle = self._get_elbow_angles(landmarks, context)

        # Rest position = both arms relatively straight
        return (
            left_elbow_angle > self.rest_threshold
            and right_elbow_angle > self.rest_threshold
        )

    def _get_elbow_angles(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> tuple:
        """Get elbow angles from context or compute them.

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context

        Returns:
            Tuple of (left_elbow_angle, right_elbow_angle)
        """
        if context is not None:
            return (
                context.joint_angles["left_elbow"],
                context.joint_angles["right_elbow"],
            )

        from ...biomechanics import calculate_joint_angle

        left_elbow_angle = calculate_joint_angle(
            (
                landmarks[LandmarkIndex.LEFT_SHOULDER]["x"],
                landmarks[LandmarkIndex.LEFT_SHOULDER]["y"],
            ),
            (
                landmarks[LandmarkIndex.LEFT_ELBOW]["x"],
                landmarks[LandmarkIndex.LEFT_ELBOW]["y"],
            ),
            (
                landmarks[LandmarkIndex.LEFT_WRIST]["x"],
                landmarks[LandmarkIndex.LEFT_WRIST]["y"],
            ),
        )

        right_elbow_angle = calculate_joint_angle(
            (
                landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"],
                landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"],
            ),
            (
                landmarks[LandmarkIndex.RIGHT_ELBOW]["x"],
                landmarks[LandmarkIndex.RIGHT_ELBOW]["y"],
            ),
            (
                landmarks[LandmarkIndex.RIGHT_WRIST]["x"],
                landmarks[LandmarkIndex.RIGHT_WRIST]["y"],
            ),
        )

        return left_elbow_angle, right_elbow_angle

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for technique metrics.

        Returns:
            Dictionary with counts and averages
        """
        summary = {
            "total_lock_offs": self._total_lock_offs,
            "total_rest_positions": self._total_rest_positions,
        }

        # Average body angle
        if "body_angle" in self._history and self._history["body_angle"]:
            values = [
                v for v in self._history["body_angle"] if isinstance(v, (int, float))
            ]
            if values:
                summary["avg_body_angle"] = float(np.mean(values))

        # Average hand/foot spans
        for metric in ["hand_span", "foot_span"]:
            if metric in self._history and self._history[metric]:
                values = [
                    v for v in self._history[metric] if isinstance(v, (int, float))
                ]
                if values:
                    summary[f"avg_{metric}"] = float(np.mean(values))

        return summary

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._total_lock_offs = 0
        self._total_rest_positions = 0
