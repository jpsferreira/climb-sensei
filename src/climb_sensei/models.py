"""Data models for climbing analysis.

This module provides immutable data classes for type-safe representation
of landmarks, metrics, and analysis results.
"""

from dataclasses import dataclass, asdict
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from climb_sensei.video_quality import VideoQualityReport
    from climb_sensei.tracking_quality import TrackingQualityReport


@dataclass(frozen=True)
class Landmark:
    """Immutable 3D landmark with visibility.

    Attributes:
        x: Normalized x-coordinate (0.0-1.0)
        y: Normalized y-coordinate (0.0-1.0)
        z: Depth coordinate
        visibility: Confidence score (0.0-1.0)
    """

    x: float
    y: float
    z: float
    visibility: float

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Landmark":
        """Create landmark from dictionary.

        Args:
            d: Dictionary with x, y, z, visibility keys

        Returns:
            Landmark instance
        """
        return cls(x=d["x"], y=d["y"], z=d["z"], visibility=d["visibility"])

    def to_tuple_2d(self) -> tuple[float, float]:
        """Get 2D coordinates as tuple.

        Returns:
            (x, y) coordinates
        """
        return (self.x, self.y)

    def to_tuple_3d(self) -> tuple[float, float, float]:
        """Get 3D coordinates as tuple.

        Returns:
            (x, y, z) coordinates
        """
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary with x, y, z, visibility keys
        """
        return asdict(self)


@dataclass(frozen=True)
class FrameMetrics:
    """Immutable metrics for a single frame.

    All metrics are calculated for one analyzed frame of video.
    Immutability ensures thread-safety and prevents accidental modification.
    """

    # Core movement metrics
    hip_height: float
    com_velocity: float
    com_sway: float
    jerk: float
    vertical_progress: float

    # Efficiency & technique metrics
    movement_economy: float
    is_lock_off: bool
    left_lock_off: bool
    right_lock_off: bool
    is_rest_position: bool

    # Body positioning metrics
    body_angle: float
    hand_span: float
    foot_span: float

    # Joint angle metrics (degrees)
    left_elbow: float
    right_elbow: float
    left_shoulder: float
    right_shoulder: float
    left_knee: float
    right_knee: float
    left_hip: float
    right_hip: float

    def to_dict(self) -> Dict[str, float | bool]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary with all metric fields
        """
        return asdict(self)


@dataclass(frozen=True)
class ClimbingSummary:
    """Immutable summary statistics for entire climb.

    Aggregated metrics calculated across all analyzed frames.
    """

    # Frame statistics
    total_frames: int

    # Vertical progression
    total_vertical_progress: float
    max_height: float

    # Movement speed
    avg_velocity: float
    max_velocity: float

    # Stability
    avg_sway: float
    max_sway: float

    # Smoothness
    avg_jerk: float
    max_jerk: float

    # Body positioning
    avg_body_angle: float
    avg_hand_span: float
    avg_foot_span: float

    # Efficiency & technique
    total_distance_traveled: float
    avg_movement_economy: float

    # Strength & technique
    lock_off_count: int
    lock_off_percentage: float
    rest_count: int
    rest_percentage: float

    # Fatigue & endurance
    fatigue_score: float

    # Joint angles (averages)
    avg_left_elbow: float
    avg_right_elbow: float
    avg_left_shoulder: float
    avg_right_shoulder: float
    avg_left_knee: float
    avg_right_knee: float
    avg_left_hip: float
    avg_right_hip: float

    def to_dict(self) -> Dict[str, float | int]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary with all summary fields
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float | int]) -> "ClimbingSummary":
        """Create ClimbingSummary from dictionary.

        Args:
            d: Dictionary with summary fields

        Returns:
            ClimbingSummary instance
        """
        return cls(**d)


@dataclass(frozen=True)
class ClimbingAnalysis:
    """Complete analysis result with summary and detailed history.

    Attributes:
        summary: Aggregated statistics
        history: Frame-by-frame metric history
        video_path: Path to analyzed video
        video_quality: Video quality report (if validation was run)
        tracking_quality: Pose tracking quality report (if validation was run)
    """

    summary: ClimbingSummary
    history: Dict[str, list]
    video_path: str | None = None
    video_quality: "VideoQualityReport | None" = None
    tracking_quality: "TrackingQualityReport | None" = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with summary, history, video_path, and quality reports
        """
        result = {
            "summary": self.summary.to_dict(),
            "history": self.history,
            "video_path": self.video_path,
        }

        # Add quality reports if present
        if self.video_quality:
            result["video_quality"] = {
                "is_valid": self.video_quality.is_valid,
                "resolution_quality": self.video_quality.resolution_quality,
                "fps_quality": self.video_quality.fps_quality,
                "duration_quality": self.video_quality.duration_quality,
                "issues": self.video_quality.issues,
                "warnings": self.video_quality.warnings,
            }

        if self.tracking_quality:
            result["tracking_quality"] = {
                "is_trackable": self.tracking_quality.is_trackable,
                "quality_level": self.tracking_quality.quality_level,
                "detection_rate": self.tracking_quality.detection_rate,
                "avg_confidence": self.tracking_quality.avg_landmark_confidence,
                "tracking_smoothness": self.tracking_quality.tracking_smoothness,
                "issues": self.tracking_quality.issues,
                "warnings": self.tracking_quality.warnings,
            }

        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "ClimbingAnalysis":
        """Create ClimbingAnalysis from dictionary.

        Args:
            d: Dictionary with summary, history, and video_path

        Returns:
            ClimbingAnalysis instance
        """
        summary = ClimbingSummary.from_dict(d["summary"])
        return cls(
            summary=summary,
            history=d.get("history", {}),
            video_path=d.get("video_path"),
        )
