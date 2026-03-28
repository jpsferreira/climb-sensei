"""Climbing Analysis Service - Composable climbing metrics analysis.

This service uses the plugin-based metrics calculators to perform
comprehensive climbing analysis. Calculators can be added or removed
based on requirements.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any

from ..biomechanics import calculate_center_of_mass, calculate_joint_angles_batch
from ..config import LandmarkIndex
from ..domain.calculators import (
    FrameContext,
    MetricsCalculator,
    StabilityCalculator,
    ProgressCalculator,
    EfficiencyCalculator,
    TechniqueCalculator,
    JointAngleCalculator,
    FatigueCalculator,
)
from ..domain.calculators.fatigue import FatigueCalculator as _FatigueCalcType
from ..models import ClimbingAnalysis, ClimbingSummary

logger = logging.getLogger(__name__)


class ClimbingAnalysisService:
    """Service for comprehensive climbing analysis using composable calculators.

    This service coordinates multiple metrics calculators to perform
    full climbing analysis. It builds a shared FrameContext per frame
    to avoid redundant computation across calculators.

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
            self.calculators: List[MetricsCalculator] = [
                StabilityCalculator(window_size=window_size, fps=fps),
                ProgressCalculator(window_size=window_size, fps=fps),
                EfficiencyCalculator(window_size=window_size, fps=fps),
                TechniqueCalculator(window_size=window_size, fps=fps),
                JointAngleCalculator(window_size=window_size, fps=fps),
                FatigueCalculator(window_size=window_size, fps=fps),
            ]
        else:
            self.calculators = calculators

    def _build_frame_context(self, landmarks: List[Dict[str, float]]) -> FrameContext:
        """Build shared FrameContext from landmarks.

        Computes COM, hip height, and all joint angles once per frame.

        Args:
            landmarks: List of 33 landmark dictionaries

        Returns:
            FrameContext with pre-computed shared values
        """
        # Center of mass from shoulder/hip midpoints
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

        # Hip height
        hip_height = (
            landmarks[LandmarkIndex.LEFT_HIP]["y"]
            + landmarks[LandmarkIndex.RIGHT_HIP]["y"]
        ) / 2.0

        # All 8 joint angles
        joint_angles = self._compute_joint_angles(landmarks)

        return FrameContext(com=com, hip_height=hip_height, joint_angles=joint_angles)

    # Joint triplets reused from JointAngleCalculator for consistency
    _JOINT_TRIPLETS = [
        (
            LandmarkIndex.LEFT_SHOULDER,
            LandmarkIndex.LEFT_ELBOW,
            LandmarkIndex.LEFT_WRIST,
        ),
        (
            LandmarkIndex.RIGHT_SHOULDER,
            LandmarkIndex.RIGHT_ELBOW,
            LandmarkIndex.RIGHT_WRIST,
        ),
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW),
        (
            LandmarkIndex.RIGHT_HIP,
            LandmarkIndex.RIGHT_SHOULDER,
            LandmarkIndex.RIGHT_ELBOW,
        ),
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
        (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
        (
            LandmarkIndex.RIGHT_SHOULDER,
            LandmarkIndex.RIGHT_HIP,
            LandmarkIndex.RIGHT_KNEE,
        ),
    ]
    _JOINT_NAMES = [
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
        "left_knee",
        "right_knee",
        "left_hip",
        "right_hip",
    ]

    def _compute_joint_angles(
        self, landmarks: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute all 8 joint angles from landmarks using vectorized batch.

        Args:
            landmarks: List of 33 landmark dictionaries

        Returns:
            Dictionary mapping joint name to angle in degrees
        """
        angles = calculate_joint_angles_batch(landmarks, self._JOINT_TRIPLETS)
        return dict(zip(self._JOINT_NAMES, angles))

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

        # Find FatigueCalculator for feeding stability metrics
        fatigue_calc = self._find_fatigue_calculator()

        # Process each frame through all calculators
        frame_history = []
        frame_count = 0

        for landmarks in landmarks_sequence:
            if landmarks is None:
                continue

            if len(landmarks) < 33:
                continue

            frame_count += 1

            # Build shared context once per frame
            context = self._build_frame_context(landmarks)

            frame_metrics: Dict[str, Any] = {}

            # Run all calculators on this frame with shared context
            for calc in self.calculators:
                metrics = calc.calculate(landmarks, context)
                frame_metrics.update(metrics)

            # Feed jerk/sway to FatigueCalculator after stability metrics are computed
            if fatigue_calc is not None:
                fatigue_calc.record_stability_metrics(
                    jerk=frame_metrics.get("jerk", 0.0),
                    sway=frame_metrics.get("com_sway", 0.0),
                )

            frame_history.append(frame_metrics)

        # Collect summaries from all calculators
        summary_data: Dict[str, Any] = {
            "total_frames": frame_count,
        }

        for calc in self.calculators:
            summary = calc.get_summary()
            summary_data.update(summary)

        # Warn about unmapped summary keys for debugging
        self._check_summary_coverage(summary_data)

        # Collect history from all calculators
        history_data: Dict[str, List] = {}
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

    def _find_fatigue_calculator(self) -> Optional[_FatigueCalcType]:
        """Find FatigueCalculator instance in calculators list."""
        for calc in self.calculators:
            if isinstance(calc, _FatigueCalcType):
                return calc
        return None

    def _check_summary_coverage(self, summary_data: Dict[str, Any]) -> None:
        """Log warnings for summary keys not mapped to ClimbingSummary fields."""
        mapped_keys = {
            "total_frames",
            "max_height",
            "total_vertical_progress",
            "avg_com_velocity",
            "max_com_velocity",
            "avg_com_sway",
            "max_com_sway",
            "avg_jerk",
            "max_jerk",
            "avg_body_angle",
            "avg_hand_span",
            "avg_foot_span",
            "total_distance_traveled",
            "final_movement_economy",
            "total_lock_offs",
            "total_rest_positions",
            "fatigue_score",
            "avg_left_elbow",
            "avg_right_elbow",
            "avg_left_shoulder",
            "avg_right_shoulder",
            "avg_left_knee",
            "avg_right_knee",
            "avg_left_hip",
            "avg_right_hip",
            # Known unmapped keys (informational, not warnings)
            "initial_height",
            "min_height",
            "avg_hip_height",
            "min_com_velocity",
            "min_com_sway",
            "min_jerk",
            "avg_movement_economy",
            "min_left_elbow",
            "max_left_elbow",
            "min_right_elbow",
            "max_right_elbow",
            "min_left_shoulder",
            "max_left_shoulder",
            "min_right_shoulder",
            "max_right_shoulder",
            "min_left_knee",
            "max_left_knee",
            "min_right_knee",
            "max_right_knee",
            "min_left_hip",
            "max_left_hip",
            "min_right_hip",
            "max_right_hip",
            "max_com_velocity",
            "max_com_sway",
            "max_jerk",
        }
        unmapped = set(summary_data.keys()) - mapped_keys
        if unmapped:
            logger.debug("Unmapped summary keys: %s", unmapped)

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
            lock_off_percentage=self._calc_percentage(
                summary_data.get("total_lock_offs", 0),
                summary_data.get("total_frames", 0),
            ),
            rest_count=summary_data.get("total_rest_positions", 0),
            rest_percentage=self._calc_percentage(
                summary_data.get("total_rest_positions", 0),
                summary_data.get("total_frames", 0),
            ),
            fatigue_score=summary_data.get("fatigue_score", 0.0),
            avg_left_elbow=summary_data.get("avg_left_elbow", 0.0),
            avg_right_elbow=summary_data.get("avg_right_elbow", 0.0),
            avg_left_shoulder=summary_data.get("avg_left_shoulder", 0.0),
            avg_right_shoulder=summary_data.get("avg_right_shoulder", 0.0),
            avg_left_knee=summary_data.get("avg_left_knee", 0.0),
            avg_right_knee=summary_data.get("avg_right_knee", 0.0),
            avg_left_hip=summary_data.get("avg_left_hip", 0.0),
            avg_right_hip=summary_data.get("avg_right_hip", 0.0),
        )

    @staticmethod
    def _calc_percentage(count: int, total: int) -> float:
        """Calculate percentage, returning 0.0 if total is zero."""
        if total == 0:
            return 0.0
        return (count / total) * 100.0

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
