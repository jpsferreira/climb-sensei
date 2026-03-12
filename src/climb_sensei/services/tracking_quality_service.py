"""Tracking Quality Service - Pose detection reliability analysis.

This service analyzes the quality and reliability of pose tracking.
Can work with pre-extracted landmarks OR process video directly.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Optional

from ..tracking_quality import (
    TrackingQualityAnalyzer,
    TrackingQualityReport,
)


class TrackingQualityService:
    """Service for pose tracking quality assessment.

    This service can operate in two modes:
    1. From landmarks: Analyze pre-extracted landmarks (fast, reusable)
    2. From video: Extract landmarks and analyze (integrated workflow)

    Usage:
        >>> # Mode 1: From pre-extracted landmarks
        >>> service = TrackingQualityService()
        >>> report = service.analyze_from_landmarks(landmarks_sequence)

        >>> # Mode 2: From video file
        >>> report = service.analyze_from_video("video.mp4")

        >>> # Async usage
        >>> report = await service.analyze_from_video_async("video.mp4")
    """

    def __init__(
        self,
        min_detection_rate: float = 70.0,
        min_avg_confidence: float = 0.5,
        min_visibility: float = 60.0,
        min_smoothness: float = 0.6,
        max_tracking_losses: int = 5,
        sample_rate: int = 1,
    ):
        """Initialize tracking quality service.

        Args:
            min_detection_rate: Minimum acceptable detection rate (0-100)
            min_avg_confidence: Minimum acceptable average confidence (0-1)
            min_visibility: Minimum acceptable visibility percentage (0-100)
            min_smoothness: Minimum acceptable smoothness score (0-1)
            max_tracking_losses: Maximum acceptable tracking loss events
            sample_rate: Analyze every Nth frame (1 = every frame)
        """
        self.min_detection_rate = min_detection_rate
        self.min_avg_confidence = min_avg_confidence
        self.min_visibility = min_visibility
        self.min_smoothness = min_smoothness
        self.max_tracking_losses = max_tracking_losses
        self.sample_rate = sample_rate

    def analyze_from_landmarks(
        self,
        landmarks_sequence: List[Optional[List[Dict[str, float]]]],
        video_path: str = "unknown",
    ) -> TrackingQualityReport:
        """Analyze tracking quality from pre-extracted landmarks.

        This is the preferred method when landmarks have already been extracted.
        It's fast and doesn't require video access.

        Args:
            landmarks_sequence: List of landmark lists (None for frames without pose)
            video_path: Path to original video (for reporting only)

        Returns:
            TrackingQualityReport with quality metrics
        """
        # Create analyzer with custom thresholds
        analyzer = TrackingQualityAnalyzer(
            min_detection_rate=self.min_detection_rate,
            min_avg_confidence=self.min_avg_confidence,
            min_visibility=self.min_visibility,
            min_smoothness=self.min_smoothness,
            max_tracking_losses=self.max_tracking_losses,
            sample_rate=self.sample_rate,
        )

        # Convert landmark format from dicts to tuples (x, y)
        # The analyzer expects List[Optional[List[Tuple[float, float]]]]
        converted_landmarks = []
        for frame_landmarks in landmarks_sequence:
            if frame_landmarks is None:
                converted_landmarks.append(None)
            else:
                # Convert from list of dicts to list of (x, y) tuples
                converted = [(lm["x"], lm["y"]) for lm in frame_landmarks]
                converted_landmarks.append(converted)

        # Analyze from landmarks
        return analyzer.analyze_from_landmarks(converted_landmarks, video_path)

    async def analyze_from_landmarks_async(
        self,
        landmarks_sequence: List[Optional[List[Dict[str, float]]]],
        video_path: str = "unknown",
    ) -> TrackingQualityReport:
        """Asynchronously analyze tracking quality from landmarks.

        Args:
            landmarks_sequence: List of landmark lists (None for frames without pose)
            video_path: Path to original video (for reporting only)

        Returns:
            TrackingQualityReport with quality metrics
        """
        return await asyncio.to_thread(
            self.analyze_from_landmarks,
            landmarks_sequence,
            video_path,
        )

    def analyze_from_video(
        self,
        video_path: str,
        pose_detection_confidence: float = 0.5,
        pose_tracking_confidence: float = 0.5,
    ) -> TrackingQualityReport:
        """Analyze tracking quality by processing video file.

        This extracts landmarks and analyzes them in one pass.
        Use this when you only need tracking quality and don't have landmarks.

        Args:
            video_path: Path to video file
            pose_detection_confidence: Confidence threshold for pose detection
            pose_tracking_confidence: Confidence threshold for pose tracking

        Returns:
            TrackingQualityReport with quality metrics

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        analyzer = TrackingQualityAnalyzer(
            min_detection_rate=self.min_detection_rate,
            min_avg_confidence=self.min_avg_confidence,
            min_visibility=self.min_visibility,
            min_smoothness=self.min_smoothness,
            max_tracking_losses=self.max_tracking_losses,
            sample_rate=self.sample_rate,
            pose_detection_confidence=pose_detection_confidence,
            pose_tracking_confidence=pose_tracking_confidence,
        )

        return analyzer.analyze_video(video_path)

    async def analyze_from_video_async(
        self,
        video_path: str,
        pose_detection_confidence: float = 0.5,
        pose_tracking_confidence: float = 0.5,
    ) -> TrackingQualityReport:
        """Asynchronously analyze tracking quality from video.

        Args:
            video_path: Path to video file
            pose_detection_confidence: Confidence threshold for pose detection
            pose_tracking_confidence: Confidence threshold for pose tracking

        Returns:
            TrackingQualityReport with quality metrics

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        return await asyncio.to_thread(
            self.analyze_from_video,
            video_path,
            pose_detection_confidence,
            pose_tracking_confidence,
        )

    def validate_or_raise(
        self,
        landmarks_sequence: List[Optional[List[Dict[str, float]]]],
        video_path: str = "unknown",
    ) -> TrackingQualityReport:
        """Validate tracking quality and raise exception if poor.

        Args:
            landmarks_sequence: List of landmark lists
            video_path: Path to original video (for reporting)

        Returns:
            TrackingQualityReport if tracking is acceptable

        Raises:
            ValueError: If tracking quality is insufficient
        """
        report = self.analyze_from_landmarks(landmarks_sequence, video_path)

        if not report.is_trackable:
            error_msg = "Tracking quality validation failed:\n"
            for issue in report.issues:
                error_msg += f"  - {issue}\n"
            raise ValueError(error_msg.rstrip())

        return report

    async def validate_or_raise_async(
        self,
        landmarks_sequence: List[Optional[List[Dict[str, float]]]],
        video_path: str = "unknown",
    ) -> TrackingQualityReport:
        """Asynchronously validate tracking quality and raise exception if poor.

        Args:
            landmarks_sequence: List of landmark lists
            video_path: Path to original video (for reporting)

        Returns:
            TrackingQualityReport if tracking is acceptable

        Raises:
            ValueError: If tracking quality is insufficient
        """
        return await asyncio.to_thread(
            self.validate_or_raise,
            landmarks_sequence,
            video_path,
        )
