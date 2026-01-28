"""Progress Calculator - Vertical progression metrics.

Calculates:
- Hip height (current position)
- Vertical progress (height gained from start)
- Max/min heights reached
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseCalculator
from ...config import LandmarkIndex


class ProgressCalculator(BaseCalculator):
    """Calculator for vertical progression metrics.

    Tracks the climber's vertical progress up the wall.

    Usage:
        >>> calc = ProgressCalculator(fps=30.0)
        >>> for landmarks in sequence:
        ...     metrics = calc.calculate(landmarks)
        ...     print(f"Progress: {metrics['vertical_progress']:.3f}m")
        >>> summary = calc.get_summary()
        >>> print(f"Total climb: {summary['total_vertical_progress']:.3f}m")
    """

    def __init__(self, window_size: int = 30, fps: float = 30.0):
        """Initialize progress calculator.

        Args:
            window_size: Number of frames for moving window (unused for progress)
            fps: Frames per second
        """
        super().__init__(window_size, fps)
        self._initial_hip_height: Optional[float] = None
        self._max_height: float = 0.0
        self._min_height: float = 1.0

    def calculate(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate progress metrics for one frame.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Dictionary with hip_height, vertical_progress
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1

        # Calculate hip height (average of left and right hip y-coordinates)
        left_hip_y = landmarks[LandmarkIndex.LEFT_HIP]["y"]
        right_hip_y = landmarks[LandmarkIndex.RIGHT_HIP]["y"]
        hip_height = (left_hip_y + right_hip_y) / 2.0

        # Store initial height for progress tracking
        if self._initial_hip_height is None:
            self._initial_hip_height = hip_height

        # Track max/min
        self._max_height = max(self._max_height, hip_height)
        self._min_height = min(self._min_height, hip_height)

        # Calculate vertical progress from start
        # Note: In image coordinates, y increases downward, so we subtract
        vertical_progress = self._initial_hip_height - hip_height

        metrics = {
            "hip_height": hip_height,
            "vertical_progress": vertical_progress,
        }

        # Track history
        for key, value in metrics.items():
            self._append_to_history(key, value)

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for progress metrics.

        Returns:
            Dictionary with total progress, max/min heights
        """
        summary = {}

        if self._initial_hip_height is not None:
            summary["initial_height"] = float(self._initial_hip_height)
            summary["max_height"] = float(self._max_height)
            summary["min_height"] = float(self._min_height)
            summary["total_vertical_progress"] = float(
                self._initial_hip_height - self._min_height
            )

        # Average hip height
        if "hip_height" in self._history and self._history["hip_height"]:
            values = [
                v for v in self._history["hip_height"] if isinstance(v, (int, float))
            ]
            if values:
                summary["avg_hip_height"] = float(np.mean(values))

        return summary

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._initial_hip_height = None
        self._max_height = 0.0
        self._min_height = 1.0
