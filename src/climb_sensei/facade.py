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
from .video_quality import check_video_quality, VideoQualityReport
from .tracking_quality import analyze_tracking_from_landmarks, TrackingQualityReport


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
        validate_quality: bool = True,
    ):
        """Initialize the climbing analysis facade.

        Args:
            video_path: Path to the climbing video file
            pose_config: Optional pose detection configuration
            metrics_config: Optional metrics calculation configuration
            window_size: Number of frames for moving window calculations
            fps: Frames per second of the video
            validate_quality: If True, automatically check video and tracking quality
        """
        self.video_path = Path(video_path)
        self.pose_config = pose_config or PoseConfig()
        self.metrics_config = metrics_config or MetricsConfig()
        self.window_size = window_size
        self.fps = fps
        self.validate_quality = validate_quality

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

        If validate_quality=True (default), automatically checks:
        - Video quality (format, resolution, FPS, lighting, stability)
        - Tracking quality (pose detection reliability, smoothness)

        Args:
            verbose: If True, print progress information

        Returns:
            ClimbingAnalysis with summary statistics, frame history, and quality reports

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be read or quality validation fails
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        video_quality_report: Optional[VideoQualityReport] = None
        tracking_quality_report: Optional[TrackingQualityReport] = None

        # Step 1: Validate video quality (if enabled)
        if self.validate_quality:
            if verbose:
                print("Checking video quality...")

            video_quality_report = check_video_quality(
                str(self.video_path), deep_check=True
            )

            if not video_quality_report.is_valid:
                error_msg = "Video quality validation failed:\n"
                for issue in video_quality_report.issues:
                    error_msg += f"  - {issue}\n"
                raise ValueError(error_msg.rstrip())

            if verbose and video_quality_report.warnings:
                print("⚠️  Video quality warnings:")
                for warning in video_quality_report.warnings:
                    print(f"  - {warning}")

        # Step 2: Process video and extract landmarks
        frame_count = 0
        landmarks_history = []

        with VideoReader(str(self.video_path)) as video:
            while True:
                success, frame = video.read()
                if not success:
                    break

                # Detect pose
                pose_result = self.pose_engine.process(frame)

                if pose_result and pose_result.pose_landmarks:
                    # Extract landmarks
                    landmarks = self.pose_engine.extract_landmarks(pose_result)

                    # Analyze frame
                    self.analyzer.analyze_frame(landmarks)

                    # Store for tracking quality analysis
                    if self.validate_quality:
                        landmarks_history.append(landmarks)

                    frame_count += 1

                    if verbose and frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames...")
                else:
                    # No pose detected
                    if self.validate_quality:
                        landmarks_history.append(None)

        if verbose:
            print(f"Analysis complete! Processed {frame_count} frames total.")

        # Step 3: Validate tracking quality (if enabled)
        if self.validate_quality and landmarks_history:
            if verbose:
                print("Analyzing tracking quality...")

            # Convert landmarks from dict format to (x, y) tuple format
            # for tracking quality analysis
            converted_landmarks = []
            for frame_landmarks in landmarks_history:
                if frame_landmarks is None:
                    converted_landmarks.append(None)
                else:
                    # Convert list of dicts to list of (x, y) tuples
                    tuples = [(lm["x"], lm["y"]) for lm in frame_landmarks]
                    converted_landmarks.append(tuples)

            tracking_quality_report = analyze_tracking_from_landmarks(
                converted_landmarks, sample_rate=1, file_path=str(self.video_path)
            )

            if not tracking_quality_report.is_trackable:
                if verbose:
                    print("⚠️  Warning: Poor tracking quality detected.")
                    print("    Results may be unreliable. Consider:")
                    print("    - Better lighting conditions")
                    print("    - More stable camera position")
                    print("    - Clearer view of climber")
            elif verbose and tracking_quality_report.warnings:
                print("⚠️  Tracking quality warnings:")
                for warning in tracking_quality_report.warnings:
                    print(f"  - {warning}")

        # Step 4: Get results
        summary = self.analyzer.get_summary_typed()
        history = self.analyzer.get_history()

        # Create analysis object with quality reports
        self._analysis = ClimbingAnalysis(
            summary=summary,
            history=history,
            video_path=str(self.video_path),
            video_quality=video_quality_report,
            tracking_quality=tracking_quality_report,
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

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        status = "analyzed" if self._analysis else "not analyzed"
        return f"ClimbingSensei(video={self.video_path.name}, status={status})"
