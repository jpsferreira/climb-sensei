"""Fatigue Calculator - Movement quality degradation over time.

Calculates:
- Fatigue score (0.0 = no degradation, 1.0+ = significant)

Compares jerk and sway in the first third vs last third of the climb
to detect movement quality deterioration.
"""

from typing import List, Dict, Any, Optional

import numpy as np

from .base import BaseCalculator, FrameContext


class FatigueCalculator(BaseCalculator):
    """Calculator for fatigue detection via movement quality degradation.

    Compares movement smoothness (jerk) and stability (sway) between
    the first and last thirds of the climb. A rising fatigue score
    indicates the climber's movement quality is deteriorating.

    Usage:
        >>> calc = FatigueCalculator(min_frames=90)
        >>> for landmarks in sequence:
        ...     metrics = calc.calculate(landmarks)
        >>> summary = calc.get_summary()
        >>> print(f"Fatigue: {summary['fatigue_score']:.2f}")
    """

    def __init__(
        self,
        window_size: int = 30,
        fps: float = 30.0,
        min_frames: int = 90,
    ):
        """Initialize fatigue calculator.

        Args:
            window_size: Number of frames for moving window (unused)
            fps: Frames per second
            min_frames: Minimum frames required before computing fatigue
        """
        super().__init__(window_size, fps)
        self.min_frames = min_frames
        self._jerk_history: List[float] = []
        self._sway_history: List[float] = []

    def calculate(
        self,
        landmarks: List[Dict[str, float]],
        context: Optional[FrameContext] = None,
    ) -> Dict[str, Any]:
        """Record jerk/sway for this frame (no per-frame fatigue output).

        Fatigue is only meaningful as a summary metric computed over the
        full climb. This method collects the required jerk and sway values
        from the StabilityCalculator's output passed via the service.

        Args:
            landmarks: List of landmark dictionaries (unused directly)
            context: Optional pre-computed frame context (unused directly)

        Returns:
            Empty dict — fatigue is a summary-only metric
        """
        if len(landmarks) < 33:
            return {}

        self.total_frames += 1
        return {}

    def record_stability_metrics(self, jerk: float, sway: float) -> None:
        """Record stability metrics from StabilityCalculator output.

        Called by the analysis service after each frame to feed jerk/sway
        values into the fatigue calculator.

        Args:
            jerk: Movement jerk value for this frame
            sway: COM sway value for this frame
        """
        self._jerk_history.append(jerk)
        self._sway_history.append(sway)

    def get_summary(self) -> Dict[str, Any]:
        """Get fatigue score computed over the full climb.

        Returns:
            Dictionary with fatigue_score (0.0 if insufficient data)
        """
        return {
            "fatigue_score": self._calculate_fatigue_score(),
        }

    def _calculate_fatigue_score(self) -> float:
        """Calculate fatigue score based on quality degradation.

        Compares movement quality (jerk and sway) in first third vs last third.
        Higher score = more fatigued (0.0 = no change, 1.0 = maximum degradation).

        Returns:
            Fatigue score clamped to [0.0, 1.0]
        """
        if len(self._jerk_history) < self.min_frames:
            return 0.0

        third = len(self._jerk_history) // 3
        if third < 10:
            return 0.0

        early_jerks = self._jerk_history[:third]
        late_jerks = self._jerk_history[-third:]
        early_sways = self._sway_history[:third]
        late_sways = self._sway_history[-third:]

        early_jerk_avg = float(np.mean(early_jerks)) if early_jerks else 0.0
        late_jerk_avg = float(np.mean(late_jerks)) if late_jerks else 0.0
        early_sway_avg = float(np.mean(early_sways)) if early_sways else 0.0
        late_sway_avg = float(np.mean(late_sways)) if late_sways else 0.0

        jerk_degradation = 0.0
        if early_jerk_avg > 0:
            jerk_degradation = (late_jerk_avg - early_jerk_avg) / early_jerk_avg

        sway_degradation = 0.0
        if early_sway_avg > 0:
            sway_degradation = (late_sway_avg - early_sway_avg) / early_sway_avg

        fatigue_score = float(
            np.clip((jerk_degradation + sway_degradation) / 2.0, 0.0, 1.0)
        )
        return fatigue_score

    def reset(self) -> None:
        """Reset calculator state."""
        super().reset()
        self._jerk_history.clear()
        self._sway_history.clear()
