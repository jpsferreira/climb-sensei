"""Stability Calculator - Movement stability and smoothness metrics.

Calculates:
- Center of mass (COM) position
- COM velocity (movement speed)
- COM sway (lateral stability)
- Movement jerk (smoothness)
"""

from typing import List, Dict, Any
from collections import deque
import numpy as np

from .base import BaseCalculator
from ...config import LandmarkIndex
from ...biomechanics import calculate_center_of_mass


class StabilityCalculator(BaseCalculator):
    """Calculator for movement stability metrics.

    Tracks center of mass movement to assess:
    - How stable the climber is (sway)
    - How smoothly they move (jerk)
    - How fast they're moving (velocity)

    Usage:
        >>> calc = StabilityCalculator(window_size=30, fps=30.0)
        >>> for landmarks in sequence:
        ...     metrics = calc.calculate(landmarks)
        ...     print(f"Sway: {metrics['com_sway']:.3f}")
        >>> summary = calc.get_summary()
        >>> print(f"Average velocity: {summary['avg_velocity']:.3f}")
    """

    def __init__(self, window_size: int = 30, fps: float = 30.0):
        """Initialize stability calculator.

        Args:
            window_size: Number of frames for moving window calculations
            fps: Frames per second for velocity calculations
        """
        super().__init__(window_size, fps)
        self._com_positions = deque(maxlen=window_size)

    def calculate(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate stability metrics for one frame.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Dictionary with com_x, com_y, com_velocity, com_sway, jerk
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1

        # Calculate center of mass using core body points
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
        weights = np.ones(len(core_points))
        com = calculate_center_of_mass(core_points, weights)

        # Update temporal buffer
        self._com_positions.append(com)

        # Base metrics
        metrics = {
            "com_x": com[0],
            "com_y": com[1],
        }

        # Velocity (requires at least 2 frames)
        if len(self._com_positions) >= 2:
            prev_com = self._com_positions[-2]
            curr_com = self._com_positions[-1]
            dx = curr_com[0] - prev_com[0]
            dy = curr_com[1] - prev_com[1]
            distance = np.sqrt(dx**2 + dy**2)
            velocity = distance / self.dt
            metrics["com_velocity"] = float(velocity)
        else:
            metrics["com_velocity"] = 0.0

        # Sway - lateral stability (std dev of x position over window)
        if len(self._com_positions) >= 3:
            com_x_values = [pos[0] for pos in self._com_positions]
            sway = float(np.std(com_x_values))
            metrics["com_sway"] = sway
        else:
            metrics["com_sway"] = 0.0

        # Jerk - smoothness (3rd derivative of position)
        if len(self._com_positions) >= 4:
            jerk = self._calculate_jerk()
            metrics["jerk"] = jerk
        else:
            metrics["jerk"] = 0.0

        # Track history
        for key, value in metrics.items():
            self._append_to_history(key, value)

        return metrics

    def _calculate_jerk(self) -> float:
        """Calculate movement jerk (smoothness indicator).

        Jerk is the rate of change of acceleration.
        Lower jerk = smoother movement.

        Returns:
            Jerk value (lower is better)
        """
        if len(self._com_positions) < 4:
            return 0.0

        positions = list(self._com_positions)[-4:]

        # Calculate velocities
        velocities = []
        for i in range(len(positions) - 1):
            dx = positions[i + 1][0] - positions[i][0]
            dy = positions[i + 1][1] - positions[i][1]
            v = np.sqrt(dx**2 + dy**2) / self.dt
            velocities.append(v)

        # Calculate accelerations
        accelerations = []
        for i in range(len(velocities) - 1):
            a = (velocities[i + 1] - velocities[i]) / self.dt
            accelerations.append(a)

        # Calculate jerk
        if len(accelerations) >= 2:
            jerk = (accelerations[1] - accelerations[0]) / self.dt
            return abs(jerk)

        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for stability metrics.

        Returns:
            Dictionary with max, min, avg for each metric
        """
        summary = {}

        for metric_name in ["com_velocity", "com_sway", "jerk"]:
            if metric_name in self._history and self._history[metric_name]:
                values = [
                    v for v in self._history[metric_name] if isinstance(v, (int, float))
                ]
                if values:
                    summary[f"max_{metric_name}"] = float(np.max(values))
                    summary[f"min_{metric_name}"] = float(np.min(values))
                    summary[f"avg_{metric_name}"] = float(np.mean(values))

        return summary

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._com_positions.clear()
