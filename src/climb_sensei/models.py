"""Data models for climbing analysis.

This module provides immutable data classes for type-safe representation
of landmarks, metrics, and analysis results.
"""

from dataclasses import dataclass, asdict
from typing import Dict


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
    """

    summary: ClimbingSummary
    history: Dict[str, list]
    video_path: str | None = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with summary, history, and video_path
        """
        return {
            "summary": self.summary.to_dict(),
            "history": self.history,
            "video_path": self.video_path,
        }

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
