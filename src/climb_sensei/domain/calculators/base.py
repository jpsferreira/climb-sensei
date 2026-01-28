"""Base protocol for metrics calculators.

This module defines the interface that all metrics calculators must implement.
"""

from typing import Protocol, List, Dict, Any


class MetricsCalculator(Protocol):
    """Protocol for climbing metrics calculators.

    Each calculator is responsible for computing a specific set of related metrics.
    Calculators maintain their own state and temporal buffers as needed.

    Usage:
        >>> calculator = StabilityCalculator(window_size=30, fps=30.0)
        >>> for landmarks in landmarks_sequence:
        ...     metrics = calculator.calculate(landmarks)
        >>> summary = calculator.get_summary()
        >>> history = calculator.get_history()
    """

    def calculate(
        self,
        landmarks: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Calculate metrics for a single frame.

        Args:
            landmarks: List of landmark dictionaries with x, y, z, visibility

        Returns:
            Dictionary of metric_name -> value for this frame
        """
        ...

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all processed frames.

        Returns:
            Dictionary of summary statistics (e.g., max, min, average)
        """
        ...

    def get_history(self) -> Dict[str, List]:
        """Get time-series history of all metrics.

        Returns:
            Dictionary mapping metric names to value lists
        """
        ...

    def reset(self) -> None:
        """Reset calculator state for new analysis."""
        ...


class BaseCalculator:
    """Base class providing common functionality for calculators.

    This is a concrete base class that calculators can inherit from
    to get common functionality like history tracking.
    """

    def __init__(self, window_size: int = 30, fps: float = 30.0):
        """Initialize base calculator.

        Args:
            window_size: Number of frames for moving window calculations
            fps: Frames per second for time-based metrics
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps
        self.total_frames = 0
        self._history: Dict[str, List] = {}

    def _init_history(self, metric_name: str) -> None:
        """Initialize history tracking for a metric.

        Args:
            metric_name: Name of the metric to track
        """
        if metric_name not in self._history:
            self._history[metric_name] = []

    def _append_to_history(self, metric_name: str, value: Any) -> None:
        """Append a value to metric history.

        Args:
            metric_name: Name of the metric
            value: Value to append
        """
        self._init_history(metric_name)
        self._history[metric_name].append(value)

    def get_history(self) -> Dict[str, List]:
        """Get time-series history of all metrics.

        Returns:
            Dictionary mapping metric names to value lists
        """
        return self._history.copy()

    def reset(self) -> None:
        """Reset calculator state."""
        self.total_frames = 0
        self._history.clear()
