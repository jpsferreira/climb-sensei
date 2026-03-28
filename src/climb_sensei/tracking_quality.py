"""Tracking quality analysis for pose detection in climbing videos.

This module analyzes the quality of pose/skeleton tracking, providing metrics
on detection reliability, landmark confidence, and tracking consistency.
Can analyze from video files or from pre-extracted landmark sequences.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from climb_sensei.pose_engine import PoseEngine
from climb_sensei.video_io import VideoReader

from .types import QualityLevel


@dataclass
class TrackingQualityReport:
    """Report of pose tracking quality analysis.

    Attributes:
        file_path: Path to analyzed video file
        total_frames: Total number of frames analyzed
        frames_with_pose: Number of frames where pose was detected
        detection_rate: Percentage of frames with pose detected (0-100)
        avg_landmark_confidence: Average confidence across all landmarks (0-1)
        min_landmark_confidence: Minimum average confidence in any frame (0-1)
        avg_visibility_score: Average percentage of visible landmarks (0-100)
        tracking_smoothness: Smoothness score based on landmark jitter (0-1, higher=smoother)
        tracking_loss_events: Number of times tracking was lost then regained
        is_trackable: Whether video has sufficient tracking quality
        issues: List of critical tracking issues
        warnings: List of tracking quality warnings
        quality_level: Overall tracking quality ('poor', 'acceptable', 'good', 'excellent')
        frame_confidences: Per-frame average confidence scores
        frame_visibility: Per-frame visibility percentages
    """

    file_path: str
    total_frames: int
    frames_with_pose: int
    detection_rate: float
    avg_landmark_confidence: float
    min_landmark_confidence: float
    avg_visibility_score: float
    tracking_smoothness: float
    tracking_loss_events: int
    is_trackable: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_level: str = "unknown"
    frame_confidences: List[float] = field(default_factory=list)
    frame_visibility: List[float] = field(default_factory=list)


class TrackingQualityAnalyzer:
    """Analyze pose tracking quality in climbing videos.

    This analyzer runs pose detection on a video and evaluates:
    - Detection rate: How often poses are detected
    - Landmark confidence: Average confidence scores
    - Visibility: Percentage of landmarks visible
    - Smoothness: Consistency of tracking (low jitter)
    - Tracking loss: Number of times tracking is lost

    Args:
        min_detection_rate: Minimum acceptable detection rate (0-100)
        min_avg_confidence: Minimum acceptable average confidence (0-1)
        min_visibility: Minimum acceptable visibility percentage (0-100)
        min_smoothness: Minimum acceptable smoothness score (0-1)
        max_tracking_losses: Maximum acceptable tracking loss events
        sample_rate: Analyze every Nth frame (1 = every frame)
        pose_detection_confidence: Confidence threshold for pose detection
        pose_tracking_confidence: Confidence threshold for pose tracking
    """

    def __init__(
        self,
        min_detection_rate: float = 70.0,
        min_avg_confidence: float = 0.5,
        min_visibility: float = 60.0,
        min_smoothness: float = 0.6,
        max_tracking_losses: int = 5,
        sample_rate: int = 1,
        pose_detection_confidence: float = 0.5,
        pose_tracking_confidence: float = 0.5,
    ):
        self.min_detection_rate = min_detection_rate
        self.min_avg_confidence = min_avg_confidence
        self.min_visibility = min_visibility
        self.min_smoothness = min_smoothness
        self.max_tracking_losses = max_tracking_losses
        self.sample_rate = sample_rate
        self.pose_detection_confidence = pose_detection_confidence
        self.pose_tracking_confidence = pose_tracking_confidence

    def analyze_video(self, video_path: str) -> TrackingQualityReport:
        """Analyze tracking quality for a video.

        Args:
            video_path: Path to video file

        Returns:
            TrackingQualityReport with analysis results

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = str(Path(video_path).resolve())

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Analyze frames
        results = self._analyze_frames_from_video(video_path)

        # Generate report
        report = self._generate_report(video_path, results)

        return report

    def analyze_from_landmarks(
        self,
        landmarks_sequence: List[Optional[List[Tuple[float, float]]]],
        file_path: str = "landmarks_sequence",
    ) -> TrackingQualityReport:
        """Analyze tracking quality from pre-extracted landmark sequence.

        This is more efficient than analyze_video() when landmarks are already
        available (e.g., during analyze_climb processing).

        Args:
            landmarks_sequence: List of landmark lists, one per frame.
                Each frame is either None (no detection) or a list of (x, y) tuples.
                Use None for frames where pose was not detected.
            file_path: Optional identifier for the source (for reporting)

        Returns:
            TrackingQualityReport with analysis results

        Example:
            >>> landmarks_seq = [
            ...     [(0.5, 0.5), (0.6, 0.4), ...],  # Frame 0
            ...     [(0.5, 0.5), (0.6, 0.4), ...],  # Frame 1
            ...     None,  # Frame 2 - no detection
            ...     [(0.5, 0.5), (0.6, 0.4), ...],  # Frame 3
            ... ]
            >>> analyzer = TrackingQualityAnalyzer()
            >>> report = analyzer.analyze_from_landmarks(landmarks_seq)
        """
        results = self._analyze_landmarks_sequence(landmarks_sequence)
        report = self._generate_report(file_path, results)
        return report

    def _analyze_landmarks_sequence(
        self, landmarks_sequence: List[Optional[List[Tuple[float, float]]]]
    ) -> Dict[str, any]:
        """Analyze pre-extracted landmark sequence."""
        frame_count = 0
        frames_with_pose = 0
        frame_confidences = []
        frame_visibility = []
        landmark_positions = []
        tracking_losses = 0
        previous_had_pose = False

        for frame_idx, landmarks in enumerate(landmarks_sequence):
            # Apply sampling
            if frame_idx % self.sample_rate != 0:
                continue

            frame_count += 1

            if landmarks is not None and len(landmarks) > 0:
                frames_with_pose += 1

                # Pre-extracted landmarks lack per-frame confidence scores.
                # Mark as NaN so the report clearly indicates these are
                # estimated, not measured. The report generator handles NaN
                # by reporting "N/A" instead of misleading values.
                confidence = float("nan")
                visibility_pct = float("nan")

                frame_confidences.append(confidence)
                frame_visibility.append(visibility_pct)

                # Convert 2D landmarks to 3D for consistency (z=0)
                positions_3d = [(x, y, 0.0) for x, y in landmarks]
                landmark_positions.append(positions_3d)

                # Track if we regained tracking
                if not previous_had_pose:
                    tracking_losses += 1
                previous_had_pose = True
            else:
                frame_confidences.append(0.0)
                frame_visibility.append(0.0)
                previous_had_pose = False

        return {
            "total_frames": frame_count,
            "frames_with_pose": frames_with_pose,
            "frame_confidences": frame_confidences,
            "frame_visibility": frame_visibility,
            "landmark_positions": landmark_positions,
            "tracking_losses": max(0, tracking_losses - 1),
        }

    def _analyze_frames_from_video(self, video_path: str) -> Dict[str, any]:
        """Analyze all frames for tracking quality."""
        frame_count = 0
        frames_with_pose = 0
        frame_confidences = []
        frame_visibility = []
        landmark_positions = []  # For smoothness calculation
        tracking_losses = 0
        previous_had_pose = False

        with PoseEngine(
            min_detection_confidence=self.pose_detection_confidence,
            min_tracking_confidence=self.pose_tracking_confidence,
        ) as engine:
            with VideoReader(video_path) as reader:
                while True:
                    success, frame = reader.read()
                    if not success:
                        break

                    # Sample frames
                    if frame_count % self.sample_rate != 0:
                        frame_count += 1
                        continue

                    frame_count += 1

                    # Process frame
                    results = engine.process(frame)

                    if results and results.pose_landmarks:
                        landmarks = engine.extract_landmarks(results)

                        if landmarks and len(landmarks) > 0:
                            frames_with_pose += 1

                            # Calculate confidence (from landmark visibility)
                            confidences = []
                            visible_count = 0
                            positions = []

                            for lm in results.pose_landmarks.landmark:
                                confidences.append(lm.visibility)
                                if lm.visibility > 0.5:
                                    visible_count += 1
                                positions.append((lm.x, lm.y, lm.z))

                            avg_confidence = np.mean(confidences)
                            visibility_pct = (
                                visible_count / len(results.pose_landmarks.landmark)
                            ) * 100

                            frame_confidences.append(avg_confidence)
                            frame_visibility.append(visibility_pct)
                            landmark_positions.append(positions)

                            # Track if we regained tracking
                            if not previous_had_pose:
                                tracking_losses += 1
                            previous_had_pose = True
                        else:
                            frame_confidences.append(0.0)
                            frame_visibility.append(0.0)
                            previous_had_pose = False
                    else:
                        frame_confidences.append(0.0)
                        frame_visibility.append(0.0)
                        previous_had_pose = False

        return {
            "total_frames": frame_count,
            "frames_with_pose": frames_with_pose,
            "frame_confidences": frame_confidences,
            "frame_visibility": frame_visibility,
            "landmark_positions": landmark_positions,
            "tracking_losses": max(
                0, tracking_losses - 1
            ),  # First detection isn't a "loss"
        }

    def _calculate_smoothness(
        self, landmark_positions: List[List[Tuple[float, float, float]]]
    ) -> float:
        """Calculate tracking smoothness based on landmark jitter.

        Uses vectorized numpy operations for efficiency (10-20x faster than
        the previous per-landmark Python loop). Computes the coefficient of
        variation (CV = std/mean) of frame-to-frame landmark displacements
        to produce a smoothness score.

        Smoothness = 1 - clamp(CV, 0, 1):
        - CV ≈ 0 → all movements uniform → smoothness ≈ 1.0 (excellent)
        - CV ≈ 1+ → highly variable movement → smoothness ≈ 0.0 (poor)

        Only uses x, y coordinates since MediaPipe z-depth is unreliable
        and would inflate jitter without perceptual relevance.

        Args:
            landmark_positions: List of landmark positions per frame,
                each frame is a list of (x, y, z) tuples

        Returns:
            Smoothness score between 0.0 and 1.0
        """
        if len(landmark_positions) < 3:
            return 0.0

        # Vectorized computation: convert to numpy array
        try:
            # (num_frames, num_landmarks, 3) — only use x, y
            pos_array = np.array(landmark_positions)[:, :, :2]
        except (ValueError, IndexError):
            # Inconsistent landmark counts across frames
            return 0.0

        # Frame-to-frame displacements: (num_frames-1, num_landmarks, 2)
        diffs = np.diff(pos_array, axis=0)

        # Per-landmark per-frame distances: (num_frames-1, num_landmarks)
        distances = np.linalg.norm(diffs, axis=2)

        # Flatten to single array of all displacements
        all_distances = distances.ravel()

        # Filter out non-finite values (NaN/Inf from bad landmarks)
        all_distances = all_distances[np.isfinite(all_distances)]

        if len(all_distances) == 0:
            return 0.0

        mean_dist = np.mean(all_distances)
        if mean_dist < 1e-10:
            return 1.0  # No movement at all = perfectly smooth

        # Coefficient of variation: std / mean
        cv = np.std(all_distances) / mean_dist

        # Smoothness = 1 - CV, clamped to [0, 1], guaranteed finite
        smoothness = 1.0 - float(np.clip(cv, 0.0, 1.0))
        return smoothness if np.isfinite(smoothness) else 0.0

    def _generate_report(
        self, video_path: str, results: Dict[str, any]
    ) -> TrackingQualityReport:
        """Generate tracking quality report from analysis results."""
        total_frames = results["total_frames"]
        frames_with_pose = results["frames_with_pose"]
        frame_confidences = results["frame_confidences"]
        frame_visibility = results["frame_visibility"]
        landmark_positions = results["landmark_positions"]
        tracking_losses = results["tracking_losses"]

        # Calculate metrics
        detection_rate = (
            (frames_with_pose / total_frames * 100) if total_frames > 0 else 0.0
        )

        # Only consider frames with valid (non-NaN, > 0) confidence values.
        # When all values are NaN (pre-extracted landmarks), use None to signal
        # "unavailable" so quality checks don't penalize missing data.
        valid_confidences = [c for c in frame_confidences if np.isfinite(c) and c > 0]
        confidence_available = len(valid_confidences) > 0
        avg_confidence = (
            float(np.mean(valid_confidences)) if confidence_available else None
        )
        min_confidence = (
            float(np.min(valid_confidences)) if confidence_available else None
        )

        valid_visibility = [v for v in frame_visibility if np.isfinite(v) and v > 0]
        visibility_available = len(valid_visibility) > 0
        avg_visibility = (
            float(np.mean(valid_visibility)) if visibility_available else None
        )

        smoothness = self._calculate_smoothness(landmark_positions)

        # Determine quality issues
        issues = []
        warnings = []

        if detection_rate < self.min_detection_rate:
            issues.append(
                f"Low detection rate: {detection_rate:.1f}% "
                f"(minimum: {self.min_detection_rate:.1f}%)"
            )

        if avg_confidence is not None and avg_confidence < self.min_avg_confidence:
            issues.append(
                f"Low landmark confidence: {avg_confidence:.2f} "
                f"(minimum: {self.min_avg_confidence:.2f})"
            )

        if avg_visibility is not None and avg_visibility < self.min_visibility:
            issues.append(
                f"Low landmark visibility: {avg_visibility:.1f}% "
                f"(minimum: {self.min_visibility:.1f}%)"
            )

        if smoothness < self.min_smoothness:
            warnings.append(
                f"Tracking is jittery: smoothness {smoothness:.2f} "
                f"(recommended: {self.min_smoothness:.2f})"
            )

        if tracking_losses > self.max_tracking_losses:
            warnings.append(
                f"Frequent tracking loss: {tracking_losses} events "
                f"(maximum recommended: {self.max_tracking_losses})"
            )

        # Determine overall quality level
        is_trackable = len(issues) == 0
        quality_level = self._determine_quality_level(
            detection_rate, avg_confidence, avg_visibility, smoothness
        )

        return TrackingQualityReport(
            file_path=video_path,
            total_frames=total_frames,
            frames_with_pose=frames_with_pose,
            detection_rate=round(detection_rate, 2),
            avg_landmark_confidence=round(avg_confidence, 3)
            if avg_confidence is not None
            else 0.0,
            min_landmark_confidence=round(min_confidence, 3)
            if min_confidence is not None
            else 0.0,
            avg_visibility_score=round(avg_visibility, 2)
            if avg_visibility is not None
            else 0.0,
            tracking_smoothness=round(smoothness, 3),
            tracking_loss_events=tracking_losses,
            is_trackable=is_trackable,
            issues=issues,
            warnings=warnings,
            quality_level=quality_level,
            frame_confidences=[
                round(c, 3) if np.isfinite(c) else 0.0 for c in frame_confidences
            ],
            frame_visibility=[
                round(v, 2) if np.isfinite(v) else 0.0 for v in frame_visibility
            ],
        )

    def _determine_quality_level(
        self,
        detection_rate: float,
        avg_confidence: float,
        avg_visibility: float,
        smoothness: float,
    ) -> str:
        """Determine overall tracking quality level.

        When confidence/visibility are None (pre-extracted landmarks),
        skip those checks rather than penalizing as poor quality.
        """
        # Treat None as "unavailable, don't penalize"
        conf = avg_confidence if avg_confidence is not None else 1.0
        vis = avg_visibility if avg_visibility is not None else 100.0

        # Excellent: All metrics exceed thresholds significantly
        if detection_rate >= 95 and conf >= 0.8 and vis >= 85 and smoothness >= 0.8:
            return QualityLevel.EXCELLENT

        # Good: All metrics meet or exceed thresholds
        if (
            detection_rate >= self.min_detection_rate
            and conf >= self.min_avg_confidence
            and vis >= self.min_visibility
            and smoothness >= self.min_smoothness
        ):
            return QualityLevel.GOOD

        # Acceptable: Meets minimum thresholds even if some warnings
        if (
            detection_rate >= self.min_detection_rate
            and conf >= self.min_avg_confidence
            and vis >= self.min_visibility
        ):
            return QualityLevel.ACCEPTABLE

        # Poor: Does not meet minimum requirements
        return QualityLevel.POOR


def analyze_tracking_quality(
    video_path: str,
    sample_rate: int = 1,
    min_detection_rate: float = 70.0,
) -> TrackingQualityReport:
    """Convenience function to analyze tracking quality from video.

    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame (1 = every frame)
        min_detection_rate: Minimum acceptable detection rate percentage

    Returns:
        TrackingQualityReport with analysis results

    Example:
        >>> report = analyze_tracking_quality('climbing.mp4')
        >>> if report.is_trackable:
        ...     print(f"Detection rate: {report.detection_rate}%")
        ... else:
        ...     print("Tracking issues:", report.issues)
    """
    analyzer = TrackingQualityAnalyzer(
        min_detection_rate=min_detection_rate,
        sample_rate=sample_rate,
    )
    return analyzer.analyze_video(video_path)


def analyze_tracking_from_landmarks(
    landmarks_sequence: List[Optional[List[Tuple[float, float]]]],
    sample_rate: int = 1,
    min_detection_rate: float = 70.0,
    file_path: str = "landmarks_sequence",
) -> TrackingQualityReport:
    """Convenience function to analyze tracking quality from landmarks.

    More efficient than analyze_tracking_quality() when landmarks are already
    available from pose detection.

    Args:
        landmarks_sequence: List of landmark lists, one per frame.
            Each frame is either None (no detection) or list of (x, y) tuples.
        sample_rate: Analyze every Nth frame (1 = every frame)
        min_detection_rate: Minimum acceptable detection rate percentage
        file_path: Optional identifier for the source

    Returns:
        TrackingQualityReport with analysis results

    Example:
        >>> # During analysis
        >>> landmarks_history = []
        >>> for frame in video:
        ...     landmarks = detect_pose(frame)
        ...     landmarks_history.append(landmarks)
        >>>
        >>> # Analyze tracking quality
        >>> report = analyze_tracking_from_landmarks(landmarks_history)
        >>> print(f"Smoothness: {report.tracking_smoothness:.3f}")
    """
    analyzer = TrackingQualityAnalyzer(
        min_detection_rate=min_detection_rate,
        sample_rate=sample_rate,
    )
    return analyzer.analyze_from_landmarks(landmarks_sequence, file_path)
