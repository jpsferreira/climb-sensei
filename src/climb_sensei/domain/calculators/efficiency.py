"""Efficiency Calculator - Movement economy metrics.

Calculates:
- Movement economy (vertical progress / total distance)
- Total distance traveled
- Efficiency ratio
"""

from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np

from .base import BaseCalculator, FrameContext
from ...config import LandmarkIndex


class EfficiencyCalculator(BaseCalculator):
    """Calculator for movement efficiency metrics.

    Assesses how efficiently the climber moves:
    - Higher economy = more vertical gain per unit distance
    - Lower total distance = more direct movement

    Usage:
        >>> calc = EfficiencyCalculator(window_size=30, fps=30.0)
        >>> for landmarks in sequence:
        ...     metrics = calc.calculate(landmarks)
        ...     print(f"Economy: {metrics['movement_economy']:.3f}")
        >>> summary = calc.get_summary()
        >>> print(f"Total distance: {summary['total_distance']:.3f}")
    """

    def __init__(self, window_size: int = 30, fps: float = 30.0):
        """Initialize efficiency calculator.

        Args:
            window_size: Number of frames for moving window
            fps: Frames per second
        """
        super().__init__(window_size, fps)
        self._com_positions = deque(maxlen=window_size)
        self._total_distance = 0.0
        self._initial_hip_height: Optional[float] = None

    def calculate(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics for one frame.

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context

        Returns:
            Dictionary with movement_economy, total_distance
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1

        # Use pre-computed values from context, or calculate if not available
        if context is not None:
            com = context.com
            hip_height = context.hip_height
        else:
            from ...biomechanics import calculate_center_of_mass

            core_points = [
                (
                    landmarks[LandmarkIndex.LEFT_SHOULDER]["x"],
                    landmarks[LandmarkIndex.LEFT_SHOULDER]["y"],
                ),
                (
                    landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"],
                    landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"],
                ),
                (
                    landmarks[LandmarkIndex.LEFT_HIP]["x"],
                    landmarks[LandmarkIndex.LEFT_HIP]["y"],
                ),
                (
                    landmarks[LandmarkIndex.RIGHT_HIP]["x"],
                    landmarks[LandmarkIndex.RIGHT_HIP]["y"],
                ),
            ]
            com = calculate_center_of_mass(core_points)
            left_hip_y = landmarks[LandmarkIndex.LEFT_HIP]["y"]
            right_hip_y = landmarks[LandmarkIndex.RIGHT_HIP]["y"]
            hip_height = (left_hip_y + right_hip_y) / 2.0

        if self._initial_hip_height is None:
            self._initial_hip_height = hip_height

        # Calculate distance traveled (with tracking-loss detection)
        if len(self._com_positions) > 0:
            prev_com = self._com_positions[-1]
            dx = com[0] - prev_com[0]
            dy = com[1] - prev_com[1]
            distance = np.sqrt(dx**2 + dy**2)

            # Skip large jumps that indicate tracking loss / re-detection
            # (COM shouldn't move more than ~10% of frame in one step)
            if distance <= 0.1:
                self._total_distance += distance

        self._com_positions.append(com)

        # Calculate movement economy, clamped to [0, 1]
        vertical_progress = self._initial_hip_height - hip_height

        if self._total_distance > 1e-6:
            movement_economy = float(
                np.clip(vertical_progress / self._total_distance, 0.0, 1.0)
            )
        else:
            movement_economy = 0.0

        metrics = {
            "movement_economy": movement_economy,
            "total_distance": float(self._total_distance),
        }

        # Track history
        for key, value in metrics.items():
            self._append_to_history(key, value)

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for efficiency metrics.

        Returns:
            Dictionary with total distance and final economy
        """
        summary = {
            "total_distance_traveled": float(self._total_distance),
        }

        # Final movement economy
        if "movement_economy" in self._history and self._history["movement_economy"]:
            values = self._history["movement_economy"]
            if values and len(values) > 0:
                summary["final_movement_economy"] = float(values[-1])
                summary["avg_movement_economy"] = float(np.mean(values))

        return summary

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._com_positions.clear()
        self._total_distance = 0.0
        self._initial_hip_height = None
