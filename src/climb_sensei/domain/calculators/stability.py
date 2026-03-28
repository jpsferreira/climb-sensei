"""Stability Calculator - Movement stability and smoothness metrics.

Calculates:
- Center of mass (COM) position
- COM velocity (movement speed)
- COM sway (lateral stability)
- Movement jerk (smoothness)
"""

from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np

from .base import BaseCalculator, FrameContext
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

    def calculate(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> Dict[str, Any]:
        """Calculate stability metrics for one frame.

        Args:
            landmarks: List of landmark dictionaries
            context: Optional pre-computed frame context

        Returns:
            Dictionary with com_x, com_y, com_velocity, com_sway, jerk
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1

        # Use pre-computed COM from context, or calculate if not available
        if context is not None:
            com = context.com
        else:
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

        Jerk is the rate of change of acceleration (3rd derivative of position).
        Lower jerk = smoother movement. Uses a sliding window for noise
        reduction instead of the minimum 4-frame window.

        Returns:
            Jerk value (lower is better), averaged over the window
        """
        n = len(self._com_positions)
        if n < 4:
            return 0.0

        # Use up to window_size frames for stable signal (minimum 4)
        window = min(self.window_size, n)
        positions = list(self._com_positions)[-window:]

        # Vectorized: positions → velocities → accelerations → jerks
        pos_array = np.array(positions)  # (window, 2)
        displacements = np.diff(pos_array, axis=0)  # (window-1, 2)
        speeds = np.linalg.norm(displacements, axis=1) / self.dt  # (window-1,)

        if len(speeds) < 2:
            return 0.0

        accelerations = np.diff(speeds) / self.dt  # (window-2,)

        if len(accelerations) < 2:
            return 0.0

        jerks = np.abs(np.diff(accelerations) / self.dt)  # (window-3,)

        if len(jerks) == 0:
            return 0.0

        # Return mean jerk over window (more stable than single-frame value)
        return float(np.mean(jerks))

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
