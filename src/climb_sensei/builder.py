"""Builder pattern for fluent analyzer construction.

This module provides a builder pattern for constructing ClimbingAnalyzer
instances with a fluent, readable API.
"""

from typing import Optional

from .metrics import ClimbingAnalyzer
from .config import MetricsConfig


class ClimbingAnalyzerBuilder:
    """Fluent builder for constructing ClimbingAnalyzer instances.

    This builder provides a readable, chainable API for configuring
    and creating ClimbingAnalyzer instances.

    Example:
        >>> analyzer = (ClimbingAnalyzerBuilder()
        ...     .with_window_size(60)
        ...     .with_fps(60.0)
        ...     .with_velocity_threshold(0.15)
        ...     .build())

        >>> # Or step by step:
        >>> builder = ClimbingAnalyzerBuilder()
        >>> builder.with_window_size(45)
        >>> builder.with_fps(30.0)
        >>> analyzer = builder.build()
    """

    def __init__(self):
        """Initialize builder with default values."""
        self._window_size: int = 30
        self._fps: float = 30.0
        self._config: Optional[MetricsConfig] = None

        # Individual config overrides (optional)
        self._velocity_threshold: Optional[float] = None
        self._sway_threshold: Optional[float] = None
        self._lock_off_threshold: Optional[float] = None
        self._rest_velocity_threshold: Optional[float] = None
        self._elbow_lock_threshold: Optional[float] = None

    def with_window_size(self, size: int) -> "ClimbingAnalyzerBuilder":
        """Set the moving window size for calculations.

        Args:
            size: Number of frames in the moving window

        Returns:
            Self for method chaining

        Raises:
            ValueError: If size is not positive
        """
        if size <= 0:
            raise ValueError(f"Window size must be positive, got {size}")
        self._window_size = size
        return self

    def with_fps(self, fps: float) -> "ClimbingAnalyzerBuilder":
        """Set the frames per second.

        Args:
            fps: Frames per second of the video

        Returns:
            Self for method chaining

        Raises:
            ValueError: If fps is not positive
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        self._fps = fps
        return self

    def with_config(self, config: MetricsConfig) -> "ClimbingAnalyzerBuilder":
        """Set a complete metrics configuration.

        Args:
            config: Metrics configuration object

        Returns:
            Self for method chaining

        Note:
            This overrides any individual threshold settings
        """
        self._config = config
        return self

    def with_velocity_threshold(self, threshold: float) -> "ClimbingAnalyzerBuilder":
        """Set the velocity threshold for movement detection.

        Args:
            threshold: Velocity threshold in m/s

        Returns:
            Self for method chaining
        """
        self._velocity_threshold = threshold
        return self

    def with_sway_threshold(self, threshold: float) -> "ClimbingAnalyzerBuilder":
        """Set the sway threshold for lateral movement.

        Args:
            threshold: Sway threshold in meters

        Returns:
            Self for method chaining
        """
        self._sway_threshold = threshold
        return self

    def with_lock_off_threshold(self, threshold: float) -> "ClimbingAnalyzerBuilder":
        """Set the lock-off detection threshold.

        Args:
            threshold: Lock-off velocity threshold

        Returns:
            Self for method chaining
        """
        self._lock_off_threshold = threshold
        return self

    def with_rest_velocity_threshold(
        self, threshold: float
    ) -> "ClimbingAnalyzerBuilder":
        """Set the rest position detection threshold.

        Args:
            threshold: Rest velocity threshold

        Returns:
            Self for method chaining
        """
        self._rest_velocity_threshold = threshold
        return self

    def with_elbow_lock_threshold(self, angle: float) -> "ClimbingAnalyzerBuilder":
        """Set the elbow lock angle threshold.

        Args:
            angle: Elbow angle threshold in degrees

        Returns:
            Self for method chaining
        """
        self._elbow_lock_threshold = angle
        return self

    def build(self) -> ClimbingAnalyzer:
        """Build the ClimbingAnalyzer with configured parameters.

        Returns:
            Configured ClimbingAnalyzer instance

        Note:
            Config and individual threshold settings are stored but not yet
            used by ClimbingAnalyzer. They are prepared for future integration
            when ClimbingAnalyzer supports custom configuration.
        """
        # Note: ClimbingAnalyzer doesn't accept config parameter yet
        # For now, just use window_size and fps
        # Config support will be added in future phase
        return ClimbingAnalyzer(
            window_size=self._window_size,
            fps=self._fps,
        )

    def reset(self) -> "ClimbingAnalyzerBuilder":
        """Reset builder to default values.

        Returns:
            Self for method chaining
        """
        self.__init__()
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ClimbingAnalyzerBuilder("
            f"window_size={self._window_size}, "
            f"fps={self._fps}, "
            f"config={'custom' if self._config else 'default'}"
            f")"
        )
