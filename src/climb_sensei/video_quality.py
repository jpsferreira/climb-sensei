"""Video quality assessment module.

This module provides functions to validate video files for climbing analysis.
Checks include format compatibility, resolution, frame rate, duration,
lighting conditions, and camera stability.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
import numpy as np
from pathlib import Path


@dataclass
class VideoQualityReport:
    """Comprehensive video quality assessment report.

    Attributes:
        is_valid: Overall validity for climbing analysis
        file_path: Path to video file
        file_size_mb: File size in megabytes
        width: Video width in pixels
        height: Video height in pixels
        fps: Frames per second
        frame_count: Total number of frames
        duration_seconds: Video duration in seconds
        codec: Video codec (fourcc code)
        format_compatible: Whether format is supported
        resolution_quality: Resolution quality assessment
        fps_quality: Frame rate quality assessment
        duration_quality: Duration quality assessment
        lighting_quality: Lighting conditions assessment
        stability_quality: Camera stability assessment
        issues: List of detected issues
        warnings: List of warnings
        recommendations: List of recommendations
    """

    is_valid: bool
    file_path: str
    file_size_mb: float
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    format_compatible: bool
    resolution_quality: str  # "excellent", "good", "acceptable", "poor"
    fps_quality: str
    duration_quality: str
    lighting_quality: Optional[str]  # None if not analyzed
    stability_quality: Optional[str]  # None if not analyzed
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class VideoQualityChecker:
    """Validates video quality for climbing analysis."""

    # Quality thresholds
    MIN_RESOLUTION = (640, 480)  # Minimum acceptable resolution
    RECOMMENDED_RESOLUTION = (1280, 720)  # HD recommended
    OPTIMAL_RESOLUTION = (1920, 1080)  # Full HD optimal

    MIN_FPS = 15  # Minimum for temporal analysis
    RECOMMENDED_FPS = 30  # Standard video
    OPTIMAL_FPS = 60  # High quality

    MIN_DURATION = 5.0  # Seconds - minimum meaningful climb
    MAX_DURATION = 600.0  # Seconds - 10 minutes max for processing
    RECOMMENDED_DURATION = (10.0, 180.0)  # 10s - 3min ideal

    MIN_BRIGHTNESS = 40  # 0-255 scale
    MAX_BRIGHTNESS = 215
    OPTIMAL_BRIGHTNESS = (80, 180)

    MAX_MOTION_BLUR_THRESHOLD = 100  # Laplacian variance threshold

    def __init__(self, deep_check: bool = False):
        """Initialize quality checker.

        Args:
            deep_check: If True, perform frame-by-frame analysis (slower)
        """
        self.deep_check = deep_check

    def check_video(self, video_path: str) -> VideoQualityReport:
        """Perform comprehensive video quality check.

        Args:
            video_path: Path to video file

        Returns:
            VideoQualityReport with detailed assessment

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If file cannot be opened as video
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        issues = []
        warnings = []
        recommendations = []

        # Get file size
        file_size_mb = path.stat().st_size / (1024 * 1024)

        # Open video
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            # Extract basic properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = self._fourcc_to_string(fourcc)

            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0.0

            # Check format compatibility
            format_compatible = self._check_format_compatibility(codec)
            if not format_compatible:
                issues.append(f"Codec {codec} may not be fully supported")
                recommendations.append("Consider converting to H.264 (MP4)")

            # Check resolution
            resolution_quality = self._assess_resolution(width, height)
            if resolution_quality == "poor":
                issues.append(f"Resolution {width}x{height} is below minimum")
            elif resolution_quality == "acceptable":
                warnings.append(f"Resolution {width}x{height} is below recommended")
                recommendations.append(
                    f"Recommended minimum: {self.RECOMMENDED_RESOLUTION[0]}x{self.RECOMMENDED_RESOLUTION[1]}"
                )

            # Check FPS
            fps_quality = self._assess_fps(fps)
            if fps_quality == "poor":
                issues.append(
                    f"Frame rate {fps} is below minimum for temporal analysis"
                )
            elif fps_quality == "acceptable":
                warnings.append(f"Frame rate {fps} is below recommended")
                recommendations.append(
                    f"Recommended: {self.RECOMMENDED_FPS} fps or higher"
                )

            # Check duration
            duration_quality = self._assess_duration(duration)
            if duration_quality == "poor":
                if duration < self.MIN_DURATION:
                    issues.append(f"Video duration {duration:.1f}s is too short")
                else:
                    warnings.append(
                        f"Video duration {duration:.1f}s is very long - may be slow to process"
                    )
            elif duration_quality == "acceptable":
                warnings.append(
                    f"Video duration {duration:.1f}s is outside optimal range"
                )

            # Deep analysis (frame sampling)
            lighting_quality = None
            stability_quality = None

            if self.deep_check and frame_count > 0:
                lighting_quality, stability_quality = self._analyze_frames(
                    cap, frame_count, issues, warnings, recommendations
                )

            # Determine overall validity
            is_valid = (
                len(issues) == 0
                and format_compatible
                and resolution_quality != "poor"
                and fps_quality != "poor"
                and duration_quality != "poor"
            )

            # Add general recommendations
            if is_valid and len(warnings) == 0:
                recommendations.append(
                    "Video quality is excellent for climbing analysis"
                )
            elif is_valid:
                recommendations.append(
                    "Video is acceptable but improvements would enhance analysis quality"
                )

            return VideoQualityReport(
                is_valid=is_valid,
                file_path=str(path.absolute()),
                file_size_mb=round(file_size_mb, 2),
                width=width,
                height=height,
                fps=round(fps, 2),
                frame_count=frame_count,
                duration_seconds=round(duration, 2),
                codec=codec,
                format_compatible=format_compatible,
                resolution_quality=resolution_quality,
                fps_quality=fps_quality,
                duration_quality=duration_quality,
                lighting_quality=lighting_quality,
                stability_quality=stability_quality,
                issues=issues,
                warnings=warnings,
                recommendations=recommendations,
            )

        finally:
            cap.release()

    def _fourcc_to_string(self, fourcc: int) -> str:
        """Convert fourcc code to readable string."""
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def _check_format_compatibility(self, codec: str) -> bool:
        """Check if video codec is compatible."""
        # Common compatible codecs
        compatible_codecs = [
            "avc1",  # H.264
            "h264",
            "H264",
            "x264",
            "mp4v",  # MPEG-4
            "MP4V",
            "XVID",
            "xvid",
            "DIVX",
            "divx",
        ]
        return any(c in codec for c in compatible_codecs) or codec.strip() == ""

    def _assess_resolution(self, width: int, height: int) -> str:
        """Assess resolution quality."""
        if width >= self.OPTIMAL_RESOLUTION[0] and height >= self.OPTIMAL_RESOLUTION[1]:
            return "excellent"
        elif (
            width >= self.RECOMMENDED_RESOLUTION[0]
            and height >= self.RECOMMENDED_RESOLUTION[1]
        ):
            return "good"
        elif width >= self.MIN_RESOLUTION[0] and height >= self.MIN_RESOLUTION[1]:
            return "acceptable"
        else:
            return "poor"

    def _assess_fps(self, fps: float) -> str:
        """Assess frame rate quality."""
        if fps >= self.OPTIMAL_FPS:
            return "excellent"
        elif fps >= self.RECOMMENDED_FPS:
            return "good"
        elif fps >= self.MIN_FPS:
            return "acceptable"
        else:
            return "poor"

    def _assess_duration(self, duration: float) -> str:
        """Assess video duration."""
        if duration < self.MIN_DURATION:
            return "poor"
        elif duration > self.MAX_DURATION:
            return "poor"
        elif self.RECOMMENDED_DURATION[0] <= duration <= self.RECOMMENDED_DURATION[1]:
            return "excellent"
        else:
            return "acceptable"

    def _analyze_frames(
        self,
        cap: cv2.VideoCapture,
        frame_count: int,
        issues: List[str],
        warnings: List[str],
        recommendations: List[str],
    ) -> Tuple[str, str]:
        """Analyze frame quality through sampling.

        Returns:
            Tuple of (lighting_quality, stability_quality)
        """
        # Sample frames (check every 30th frame or 10 samples, whichever is more)
        sample_interval = max(1, frame_count // 10)
        sample_interval = min(sample_interval, 30)

        brightness_values = []
        blur_values = []

        frame_idx = 0
        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check brightness
            brightness = np.mean(gray)
            brightness_values.append(brightness)

            # Check blur (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_values.append(laplacian_var)

            frame_idx += sample_interval

        # Assess lighting
        avg_brightness = np.mean(brightness_values)
        lighting_quality = self._assess_lighting(
            avg_brightness, issues, warnings, recommendations
        )

        # Assess stability/blur
        avg_blur = np.mean(blur_values)
        stability_quality = self._assess_stability(
            avg_blur, issues, warnings, recommendations
        )

        return lighting_quality, stability_quality

    def _assess_lighting(
        self,
        avg_brightness: float,
        issues: List[str],
        warnings: List[str],
        recommendations: List[str],
    ) -> str:
        """Assess lighting conditions."""
        if avg_brightness < self.MIN_BRIGHTNESS:
            issues.append(f"Video is too dark (brightness: {avg_brightness:.1f}/255)")
            recommendations.append("Improve lighting or adjust camera exposure")
            return "poor"
        elif avg_brightness > self.MAX_BRIGHTNESS:
            warnings.append(
                f"Video is overexposed (brightness: {avg_brightness:.1f}/255)"
            )
            recommendations.append("Reduce exposure or lighting intensity")
            return "acceptable"
        elif self.OPTIMAL_BRIGHTNESS[0] <= avg_brightness <= self.OPTIMAL_BRIGHTNESS[1]:
            return "excellent"
        else:
            return "good"

    def _assess_stability(
        self,
        avg_blur: float,
        issues: List[str],
        warnings: List[str],
        recommendations: List[str],
    ) -> str:
        """Assess camera stability and motion blur."""
        if avg_blur < self.MAX_MOTION_BLUR_THRESHOLD:
            warnings.append(
                f"Possible motion blur detected (sharpness: {avg_blur:.1f})"
            )
            recommendations.append("Use a tripod or stabilization for better results")
            return "acceptable"
        else:
            return "excellent"


def check_video_quality(
    video_path: str, deep_check: bool = False
) -> VideoQualityReport:
    """Convenience function to check video quality.

    Args:
        video_path: Path to video file
        deep_check: Whether to perform deep frame analysis

    Returns:
        VideoQualityReport
    """
    checker = VideoQualityChecker(deep_check=deep_check)
    return checker.check_video(video_path)
