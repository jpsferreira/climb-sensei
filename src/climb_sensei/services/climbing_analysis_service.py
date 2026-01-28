"""Climbing Analysis Service - Composable climbing metrics analysis.

This service uses the plugin-based metrics calculators to perform
comprehensive climbing analysis. Calculators can be added or removed
based on requirements.
"""

import asyncio
from typing import List, Dict, Optional, Any

from ..domain.calculators import (
    MetricsCalculator,
    StabilityCalculator,
    ProgressCalculator,
    EfficiencyCalculator,
    TechniqueCalculator,
    JointAngleCalculator,
)
from ..models import ClimbingAnalysis, ClimbingSummary


class ClimbingAnalysisService:
    """Service for comprehensive climbing analysis using composable calculators.

    This service coordinates multiple metrics calculators to perform
    full climbing analysis. Calculators can be customized or extended
    without modifying this service.

    Usage:
        >>> # Use default calculators
        >>> service = ClimbingAnalysisService()
        >>> analysis = service.analyze(landmarks_sequence, fps=30.0)

        >>> # Use custom calculators
        >>> custom_calcs = [
        ...     StabilityCalculator(window_size=60),
        ...     ProgressCalculator(),
        ... ]
        >>> service = ClimbingAnalysisService(calculators=custom_calcs)
        >>> analysis = service.analyze(landmarks_sequence, fps=30.0)

        >>> # Async usage
        >>> analysis = await service.analyze_async(landmarks_sequence, fps=30.0)
    """

    def __init__(
        self,
        calculators: Optional[List[MetricsCalculator]] = None,
        window_size: int = 30,
        fps: float = 30.0,
    ):
        """Initialize climbing analysis service.

        Args:
            calculators: List of metrics calculators to use. If None, uses default set.
            window_size: Number of frames for moving window calculations
            fps: Frames per second for time-based metrics
        """
        self.window_size = window_size
        self.fps = fps

        if calculators is None:
            # Default calculator suite
            self.calculators = [
                StabilityCalculator(window_size=window_size, fps=fps),
                ProgressCalculator(window_size=window_size, fps=fps),
                EfficiencyCalculator(window_size=window_size, fps=fps),
                TechniqueCalculator(window_size=window_size, fps=fps),
                JointAngleCalculator(window_size=window_size, fps=fps),
            ]
        else:
            self.calculators = calculators

    def analyze(
        self,
        landmarks_sequence: List[Optional[List[Dict[str, float]]]],
        fps: Optional[float] = None,
        video_path: Optional[str] = None,
        video_quality: Optional[Any] = None,
        tracking_quality: Optional[Any] = None,
    ) -> ClimbingAnalysis:
        """Analyze climbing performance from landmark sequence.

        Args:
            landmarks_sequence: List of landmark lists (None for frames without pose)
            fps: Override FPS (uses service default if not provided)
            video_path: Optional path to original video (for metadata)
            video_quality: Optional VideoQualityReport
            tracking_quality: Optional TrackingQualityReport

        Returns:
            ClimbingAnalysis with comprehensive metrics
        """
        # Reset all calculators
        for calc in self.calculators:
            calc.reset()

        # Process each frame through all calculators
        frame_history = []
        frame_count = 0

        for landmarks in landmarks_sequence:
            if landmarks is None:
                continue

            frame_count += 1
            frame_metrics = {}

            # Run all calculators on this frame
            for calc in self.calculators:
                metrics = calc.calculate(landmarks)
                frame_metrics.update(metrics)

            frame_history.append(frame_metrics)

        # Collect summaries from all calculators
        summary_data = {
            "total_frames": frame_count,
        }

        for calc in self.calculators:
            summary = calc.get_summary()
            summary_data.update(summary)

        # Collect history from all calculators
        history_data = {}
        for calc in self.calculators:
            history = calc.get_history()
            history_data.update(history)

        # Build summary object
        summary = self._build_summary(summary_data)

        # Build analysis
        analysis = ClimbingAnalysis(
            summary=summary,
            history=history_data,
            video_path=video_path,
            video_quality=video_quality,
            tracking_quality=tracking_quality,
        )

        return analysis

    async def analyze_async(
        self,
        landmarks_sequence: List[Optional[List[Dict[str, float]]]],
        fps: Optional[float] = None,
        video_path: Optional[str] = None,
        video_quality: Optional[Any] = None,
        tracking_quality: Optional[Any] = None,
    ) -> ClimbingAnalysis:
        """Asynchronously analyze climbing performance.

        Args:
            landmarks_sequence: List of landmark lists
            fps: Override FPS
            video_path: Optional path to original video
            video_quality: Optional VideoQualityReport
            tracking_quality: Optional TrackingQualityReport

        Returns:
            ClimbingAnalysis with comprehensive metrics
        """
        return await asyncio.to_thread(
            self.analyze,
            landmarks_sequence,
            fps,
            video_path,
            video_quality,
            tracking_quality,
        )

    def _build_summary(self, summary_data: Dict[str, Any]) -> ClimbingSummary:
        """Build ClimbingSummary from aggregated data.

        Args:
            summary_data: Dictionary of summary statistics

        Returns:
            ClimbingSummary object
        """
        # Extract or compute required fields
        total_frames = summary_data.get("total_frames", 0)

        return ClimbingSummary(
            total_frames=total_frames,
            max_height=summary_data.get("max_height", 0.0),
            total_vertical_progress=summary_data.get("total_vertical_progress", 0.0),
            avg_velocity=summary_data.get("avg_com_velocity", 0.0),
            max_velocity=summary_data.get("max_com_velocity", 0.0),
            avg_sway=summary_data.get("avg_com_sway", 0.0),
            max_sway=summary_data.get("max_com_sway", 0.0),
            avg_jerk=summary_data.get("avg_jerk", 0.0),
            max_jerk=summary_data.get("max_jerk", 0.0),
            avg_body_angle=summary_data.get("avg_body_angle", 0.0),
            avg_hand_span=summary_data.get("avg_hand_span", 0.0),
            avg_foot_span=summary_data.get("avg_foot_span", 0.0),
            total_distance_traveled=summary_data.get("total_distance_traveled", 0.0),
            avg_movement_economy=summary_data.get("final_movement_economy", 0.0),
            lock_off_count=summary_data.get("total_lock_offs", 0),
            lock_off_percentage=0.0,
            rest_count=summary_data.get("total_rest_positions", 0),
            rest_percentage=0.0,
            fatigue_score=0.0,
            avg_left_elbow=summary_data.get("avg_left_elbow", 0.0),
            avg_right_elbow=summary_data.get("avg_right_elbow", 0.0),
            avg_left_shoulder=summary_data.get("avg_left_shoulder", 0.0),
            avg_right_shoulder=summary_data.get("avg_right_shoulder", 0.0),
            avg_left_knee=summary_data.get("avg_left_knee", 0.0),
            avg_right_knee=summary_data.get("avg_right_knee", 0.0),
            avg_left_hip=summary_data.get("avg_left_hip", 0.0),
            avg_right_hip=summary_data.get("avg_right_hip", 0.0),
        )

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metrics from configured calculators.

        Returns:
            List of metric names
        """
        all_metrics = set()

        # Create dummy data to see what metrics each calculator produces
        dummy_landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}] * 33

        for calc in self.calculators:
            calc.reset()
            metrics = calc.calculate(dummy_landmarks)
            all_metrics.update(metrics.keys())
            calc.reset()

        return sorted(list(all_metrics))

    def add_calculator(self, calculator: MetricsCalculator) -> None:
        """Add a new metrics calculator to the service.

        Args:
            calculator: Metrics calculator to add
        """
        self.calculators.append(calculator)

    def remove_calculator(self, calculator_type: type) -> None:
        """Remove a metrics calculator by type.

        Args:
            calculator_type: Type of calculator to remove
        """
        self.calculators = [
            calc for calc in self.calculators if not isinstance(calc, calculator_type)
        ]
