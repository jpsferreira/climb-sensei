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

        This is a convenience method that combines extract_landmarks() and
        analyze_from_landmarks() in a single call.

        For more control (e.g., parallel processing, video generation),
        use the two-phase approach:
        1. extract_landmarks() - Extract poses once
        2. analyze_from_landmarks() - Analyze metrics (can run in parallel)

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
        # Phase 1: Extract landmarks and validate video quality
        extracted = self.extract_landmarks(
            verbose=verbose,
            validate_video_quality=self.validate_quality,
        )

        # Phase 2: Analyze from landmarks
        analysis = self.analyze_from_landmarks(
            landmarks_sequence=extracted["landmarks"],
            fps=extracted["fps"],
            validate_tracking_quality=self.validate_quality,
            verbose=verbose,
        )

        # Add video quality report from extraction phase
        if extracted["video_quality"] is not None:
            # Create new analysis with video quality included
            self._analysis = ClimbingAnalysis(
                summary=analysis.summary,
                history=analysis.history,
                video_path=analysis.video_path,
                video_quality=extracted["video_quality"],
                tracking_quality=analysis.tracking_quality,
            )
        else:
            self._analysis = analysis

        return self._analysis

    def extract_landmarks(
        self, verbose: bool = True, validate_video_quality: bool = True
    ) -> dict:
        """Phase 1: Extract landmarks from video (single pass).

        This method performs video quality validation and pose detection,
        returning all landmarks for efficient downstream processing.

        Use this when you need to:
        - Process landmarks multiple times (analysis + video generation)
        - Run multiple analyses in parallel
        - Separate extraction from analysis in backend APIs

        Args:
            verbose: If True, print progress information
            validate_video_quality: If True, check video quality first

        Returns:
            Dictionary with:
            - 'landmarks': List of landmark lists (one per frame)
            - 'pose_results': List of MediaPipe pose results (for drawing)
            - 'video_quality': VideoQualityReport (if validation enabled)
            - 'fps': Actual video FPS
            - 'frame_count': Number of frames processed

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video quality validation fails

        Example:
            >>> sensei = ClimbingSensei("climb.mp4")
            >>> extracted = sensei.extract_landmarks()
            >>> # Now use landmarks for multiple purposes
            >>> analysis = sensei.analyze_from_landmarks(extracted['landmarks'])
            >>> video = generate_video(extracted['pose_results'])  # Parallel!
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        video_quality_report: Optional[VideoQualityReport] = None

        # Step 1: Validate video quality (if enabled)
        if validate_video_quality:
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

        # Step 2: Extract all landmarks (single pass)
        landmarks_history = []
        pose_results_history = []
        frame_count = 0
        actual_fps = self.fps

        with VideoReader(str(self.video_path)) as video:
            actual_fps = video.fps  # Get actual FPS from video

            while True:
                success, frame = video.read()
                if not success:
                    break

                # Detect pose
                pose_result = self.pose_engine.process(frame)

                if pose_result and pose_result.pose_landmarks:
                    # Extract landmarks
                    landmarks = self.pose_engine.extract_landmarks(pose_result)
                    landmarks_history.append(landmarks)
                    pose_results_history.append(pose_result)
                    frame_count += 1

                    if verbose and frame_count % 100 == 0:
                        print(f"Extracted {frame_count} frames...")
                else:
                    # No pose detected
                    landmarks_history.append(None)
                    pose_results_history.append(None)

        if verbose:
            print(
                f"Extraction complete! Processed {len(landmarks_history)} frames total."
            )

        return {
            "landmarks": landmarks_history,
            "pose_results": pose_results_history,
            "video_quality": video_quality_report,
            "fps": actual_fps,
            "frame_count": frame_count,
        }

    def analyze_from_landmarks(
        self,
        landmarks_sequence: list,
        fps: Optional[float] = None,
        validate_tracking_quality: bool = True,
        verbose: bool = True,
    ) -> ClimbingAnalysis:
        """Phase 2: Analyze pre-extracted landmarks.

        This method takes landmarks from extract_landmarks() and performs
        climbing analysis and tracking quality assessment.

        This enables:
        - Parallel processing of different analyses
        - Reusing landmarks for video generation
        - Efficient backend API workflows

        Args:
            landmarks_sequence: List of landmark lists from extract_landmarks()
            fps: Override FPS (uses video FPS if not provided)
            validate_tracking_quality: If True, assess tracking quality
            verbose: If True, print progress information

        Returns:
            ClimbingAnalysis with summary, history, and quality reports

        Example:
            >>> extracted = sensei.extract_landmarks()
            >>> analysis = sensei.analyze_from_landmarks(extracted['landmarks'])
        """
        # Use provided FPS or default
        analysis_fps = fps or self.fps

        # Reset analyzer with correct FPS
        if self._analyzer is None or self._analyzer.fps != analysis_fps:
            self._analyzer = ClimbingAnalyzer(
                window_size=self.window_size,
                fps=analysis_fps,
            )
        else:
            self._analyzer.reset()

        # Process landmarks for climbing analysis
        frame_count = 0
        for landmarks in landmarks_sequence:
            if landmarks is not None:
                self.analyzer.analyze_frame(landmarks)
                frame_count += 1

        if verbose:
            print(f"Analyzed {frame_count} frames with pose data.")

        # Validate tracking quality (if enabled)
        tracking_quality_report: Optional[TrackingQualityReport] = None
        if validate_tracking_quality:
            if verbose:
                print("Analyzing tracking quality...")

            # Convert landmarks from dict format to (x, y) tuple format
            converted_landmarks = []
            for frame_landmarks in landmarks_sequence:
                if frame_landmarks is None:
                    converted_landmarks.append(None)
                else:
                    tuples = [(lm["x"], lm["y"]) for lm in frame_landmarks]
                    converted_landmarks.append(tuples)

            tracking_quality_report = analyze_tracking_from_landmarks(
                converted_landmarks, sample_rate=1, file_path=str(self.video_path)
            )

            if not tracking_quality_report.is_trackable and verbose:
                print("⚠️  Warning: Poor tracking quality detected.")
                print("    Results may be unreliable.")
            elif verbose and tracking_quality_report.warnings:
                print("⚠️  Tracking quality warnings:")
                for warning in tracking_quality_report.warnings:
                    print(f"  - {warning}")

        # Get results
        summary = self.analyzer.get_summary_typed()
        history = self.analyzer.get_history()

        # Create analysis object
        self._analysis = ClimbingAnalysis(
            summary=summary,
            history=history,
            video_path=str(self.video_path),
            video_quality=None,  # Only available from extract_landmarks
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
