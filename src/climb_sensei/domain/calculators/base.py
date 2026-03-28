"""Base protocol for metrics calculators.

This module defines the interface that all metrics calculators must implement,
and the FrameContext dataclass for sharing pre-computed values across calculators.
"""

import math
import numbers
from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class FrameContext:
    """Pre-computed per-frame values shared across calculators.

    Built once per frame by the analysis service to avoid redundant
    computation of COM, hip height, and joint angles across calculators.
    """

    com: Tuple[float, float]
    """Center of mass (x, y) from shoulder/hip midpoints."""

    hip_height: float
    """Average of left and right hip y-coordinates."""

    joint_angles: Dict[str, float]
    """Pre-computed joint angles (left_elbow, right_elbow, etc.)."""


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
        context: Optional[FrameContext] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single frame.

        Args:
            landmarks: List of landmark dictionaries with x, y, z, visibility
            context: Optional pre-computed frame context for shared values

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
        """Append a numeric value to metric history.

        Only appends finite numeric values. Non-numeric or NaN/Inf values
        are silently dropped to prevent downstream aggregation issues.

        Args:
            metric_name: Name of the metric
            value: Value to append (must be numeric and finite)
        """
        self._init_history(metric_name)
        if (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            self._history[metric_name].append(float(value))

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
