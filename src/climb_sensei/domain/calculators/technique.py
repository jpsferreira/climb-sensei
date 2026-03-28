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
        lock_off_velocity_threshold: float = 0.002,
        rest_position_angle_threshold: float = 150.0,
        rest_velocity_threshold: float = 0.01,
    ):
        """Initialize technique calculator.

        Args:
            window_size: Number of frames for moving window
            fps: Frames per second
            lock_off_elbow_threshold: Max elbow angle for lock-off (degrees)
            lock_off_velocity_threshold: Max wrist velocity for lock-off
                (normalized coords/frame). Arm must be stationary.
            rest_position_angle_threshold: Min elbow angle for rest (degrees)
            rest_velocity_threshold: Max COM velocity for rest detection
                (normalized coords/frame). Body must be nearly still.
        """
        super().__init__(window_size, fps)
        self.lock_off_threshold = lock_off_elbow_threshold
        self.lock_off_velocity_threshold = lock_off_velocity_threshold
        self.rest_threshold = rest_position_angle_threshold
        self.rest_velocity_threshold = rest_velocity_threshold
        self._total_lock_offs = 0
        self._total_rest_positions = 0
        self._prev_wrists: Optional[tuple] = None
        self._prev_com: Optional[tuple] = None

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

        # Use |dy| so angle is always measured from the upward vertical direction.
        # This keeps the signed angle in the documented -90..+90 range even when
        # shoulders are detected below hips (dy <= 0).
        dy_abs = abs(dy)

        # atan2(dx, dy_abs) gives angle from vertical, preserving left/right sign
        angle = np.degrees(np.arctan2(dx, dy_abs))
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
        """Detect lock-off position (one arm bent AND stationary).

        A true lock-off requires both:
        1. Elbow angle below threshold (arm bent under load)
        2. Wrist nearly stationary (not mid-movement)

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context with joint angles

        Returns:
            Tuple of (left_lock_off, right_lock_off, is_lock_off)
        """
        left_elbow_angle, right_elbow_angle = self._get_elbow_angles(landmarks, context)

        # Current wrist positions
        lw = (
            landmarks[LandmarkIndex.LEFT_WRIST]["x"],
            landmarks[LandmarkIndex.LEFT_WRIST]["y"],
        )
        rw = (
            landmarks[LandmarkIndex.RIGHT_WRIST]["x"],
            landmarks[LandmarkIndex.RIGHT_WRIST]["y"],
        )

        # Check wrist velocity (stationary = true lock-off, not mid-pull)
        left_still = True
        right_still = True
        if self._prev_wrists is not None:
            prev_lw, prev_rw = self._prev_wrists
            left_vel = np.sqrt((lw[0] - prev_lw[0]) ** 2 + (lw[1] - prev_lw[1]) ** 2)
            right_vel = np.sqrt((rw[0] - prev_rw[0]) ** 2 + (rw[1] - prev_rw[1]) ** 2)
            left_still = left_vel < self.lock_off_velocity_threshold
            right_still = right_vel < self.lock_off_velocity_threshold

        self._prev_wrists = (lw, rw)

        # Lock-off = elbow bent AND wrist stationary
        left_lock = left_elbow_angle < self.lock_off_threshold and left_still
        right_lock = right_elbow_angle < self.lock_off_threshold and right_still

        is_lock_off = left_lock or right_lock
        return left_lock, right_lock, is_lock_off

    def _detect_rest_position(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> bool:
        """Detect rest position (straight arms AND body nearly still).

        A true rest requires both:
        1. Both elbows extended (straight arms, hanging on skeleton)
        2. Body center of mass nearly stationary (not moving between holds)

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context with joint angles

        Returns:
            True if in rest position
        """
        left_elbow_angle, right_elbow_angle = self._get_elbow_angles(landmarks, context)

        # Compute COM every frame so velocity is always frame-to-frame
        if context is not None:
            com = context.com
        else:
            com = (
                (
                    landmarks[LandmarkIndex.LEFT_HIP]["x"]
                    + landmarks[LandmarkIndex.RIGHT_HIP]["x"]
                )
                / 2,
                (
                    landmarks[LandmarkIndex.LEFT_HIP]["y"]
                    + landmarks[LandmarkIndex.RIGHT_HIP]["y"]
                )
                / 2,
            )

        body_still = True
        if self._prev_com is not None:
            com_vel = np.sqrt(
                (com[0] - self._prev_com[0]) ** 2 + (com[1] - self._prev_com[1]) ** 2
            )
            body_still = com_vel < self.rest_velocity_threshold

        self._prev_com = com

        arms_straight = (
            left_elbow_angle > self.rest_threshold
            and right_elbow_angle > self.rest_threshold
        )

        return arms_straight and body_still

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
        values = self._history.get("body_angle")
        if values:
            summary["avg_body_angle"] = float(np.mean(values))

        # Average hand/foot spans
        for metric in ["hand_span", "foot_span"]:
            values = self._history.get(metric)
            if values:
                summary[f"avg_{metric}"] = float(np.mean(values))

        return summary

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._total_lock_offs = 0
        self._total_rest_positions = 0
        self._prev_wrists = None
        self._prev_com = None
