"""Simplified facade for climbing analysis.

This module provides a high-level interface for the entire climbing analysis
pipeline, hiding complexity and making common use cases simple.
"""

from pathlib import Path
from typing import Optional

from .models import ClimbingAnalysis, ClimbingSummary
from .metrics import ClimbingAnalyzer
from .pose_engine import PoseEngine
from .video_io import VideoReader
from .config import PoseConfig, MetricsConfig


class ClimbingSensei:
    """Facade for the entire climbing analysis pipeline.

    This class provides a simplified interface for analyzing climbing videos,
    wrapping the complexity of pose detection, metrics calculation, and
    result management.

    Example:
        >>> sensei = ClimbingSensei("climb.mp4")
        >>> analysis = sensei.analyze()
        >>> print(f"Max height: {analysis.summary.max_height:.2f}m")
        >>> sensei.close()

        # Or use as context manager:
        >>> with ClimbingSensei("climb.mp4") as sensei:
        ...     analysis = sensei.analyze()
    """

    def __init__(
        self,
        video_path: str,
        pose_config: Optional[PoseConfig] = None,
        metrics_config: Optional[MetricsConfig] = None,
        window_size: int = 30,
        fps: float = 30.0,
    ):
        """Initialize the climbing analysis facade.

        Args:
            video_path: Path to the climbing video file
            pose_config: Optional pose detection configuration
            metrics_config: Optional metrics calculation configuration
            window_size: Number of frames for moving window calculations
            fps: Frames per second of the video
        """
        self.video_path = Path(video_path)
        self.pose_config = pose_config or PoseConfig()
        self.metrics_config = metrics_config or MetricsConfig()
        self.window_size = window_size
        self.fps = fps

        self._pose_engine: Optional[PoseEngine] = None
        self._analyzer: Optional[ClimbingAnalyzer] = None
        self._analysis: Optional[ClimbingAnalysis] = None

    def __enter__(self) -> "ClimbingSensei":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure resources are cleaned up."""
        self.close()

    @property
    def pose_engine(self) -> PoseEngine:
        """Lazy-loaded pose engine."""
        if self._pose_engine is None:
            # Use config values directly
            self._pose_engine = PoseEngine(
                min_detection_confidence=self.pose_config.min_detection_confidence,
                min_tracking_confidence=self.pose_config.min_tracking_confidence,
            )
        return self._pose_engine

    @property
    def analyzer(self) -> ClimbingAnalyzer:
        """Lazy-loaded climbing analyzer."""
        if self._analyzer is None:
            # Note: ClimbingAnalyzer doesn't accept config parameter yet
            self._analyzer = ClimbingAnalyzer(
                window_size=self.window_size,
                fps=self.fps,
            )
        return self._analyzer

    def analyze(self, verbose: bool = True) -> ClimbingAnalysis:
        """Run complete analysis on the climbing video.

        This method processes the entire video, detecting poses and calculating
        metrics for each frame, then returns a comprehensive analysis.

        Args:
            verbose: If True, print progress information

        Returns:
            ClimbingAnalysis with summary statistics and frame-by-frame history

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be read
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        frame_count = 0

        with VideoReader(str(self.video_path)) as video:
            for frame in video:
                # Detect pose
                pose_result = self.pose_engine.process(frame)

                if pose_result and pose_result.pose_landmarks:
                    # Extract landmarks
                    landmarks = self.pose_engine.extract_landmarks(pose_result)

                    # Analyze frame
                    self.analyzer.analyze_frame(landmarks)
                    frame_count += 1

                    if verbose and frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames...")

        if verbose:
            print(f"Analysis complete! Processed {frame_count} frames total.")

        # Get results
        summary = self.analyzer.get_summary_typed()
        history = self.analyzer.get_history()

        # Create analysis object
        self._analysis = ClimbingAnalysis(
            summary=summary,
            history=history,
            video_path=str(self.video_path),
        )

        return self._analysis

    def get_analysis(self) -> Optional[ClimbingAnalysis]:
        """Get the cached analysis result.

        Returns:
            ClimbingAnalysis if analyze() has been called, None otherwise
        """
        return self._analysis

    def get_summary(self) -> Optional[ClimbingSummary]:
        """Get summary statistics from the analysis.

        Returns:
            ClimbingSummary if analyze() has been called, None otherwise
        """
        if self._analysis:
            return self._analysis.summary
        return None

    def reset(self) -> None:
        """Reset the analyzer state.

        This clears all accumulated metrics and frame history, allowing
        you to run a new analysis.
        """
        if self._analyzer:
            self._analyzer.reset()
        self._analysis = None

    def close(self) -> None:
        """Release resources (pose detection models, etc.)."""
        if self._pose_engine:
            self._pose_engine.close()
            self._pose_engine = None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        status = "analyzed" if self._analysis else "not analyzed"
        return f"ClimbingSensei(video={self.video_path.name}, status={status})"
