"""Video Quality Service - Standalone video validation.

This service provides video quality assessment independent of climbing analysis.
Can be used by any application that needs to validate video files.
"""

import asyncio
from pathlib import Path
from typing import Optional

from ..video_quality import VideoQualityChecker, VideoQualityReport


class VideoQualityService:
    """Service for video quality assessment.

    This is a standalone service that validates video files for:
    - Format compatibility
    - Resolution quality
    - Frame rate adequacy
    - Duration appropriateness
    - Lighting conditions (optional deep check)
    - Camera stability (optional deep check)

    Usage:
        >>> service = VideoQualityService()
        >>> report = service.analyze_sync("video.mp4")
        >>> if report.is_valid:
        ...     print("Video is ready for processing")

        >>> # Async usage
        >>> report = await service.analyze("video.mp4", deep_check=True)
    """

    def __init__(
        self,
        default_deep_check: bool = False,
    ):
        """Initialize video quality service.

        Args:
            default_deep_check: If True, perform frame-by-frame analysis by default
        """
        self.default_deep_check = default_deep_check

    def analyze_sync(
        self,
        video_path: str,
        deep_check: Optional[bool] = None,
    ) -> VideoQualityReport:
        """Synchronously analyze video quality.

        Args:
            video_path: Path to video file
            deep_check: Override default deep_check setting

        Returns:
            VideoQualityReport with detailed assessment

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        use_deep_check = (
            deep_check if deep_check is not None else self.default_deep_check
        )
        checker = VideoQualityChecker(deep_check=use_deep_check)

        return checker.check_video(video_path)

    async def analyze(
        self,
        video_path: str,
        deep_check: Optional[bool] = None,
    ) -> VideoQualityReport:
        """Asynchronously analyze video quality.

        This runs the video quality check in a thread pool to avoid
        blocking the event loop.

        Args:
            video_path: Path to video file
            deep_check: Override default deep_check setting

        Returns:
            VideoQualityReport with detailed assessment

        Raises:
            FileNotFoundError: If video file doesn't exist
        """
        return await asyncio.to_thread(
            self.analyze_sync,
            video_path,
            deep_check,
        )

    def validate_or_raise(
        self,
        video_path: str,
        deep_check: Optional[bool] = None,
    ) -> VideoQualityReport:
        """Validate video and raise exception if invalid.

        Args:
            video_path: Path to video file
            deep_check: Override default deep_check setting

        Returns:
            VideoQualityReport if valid

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video quality validation fails
        """
        report = self.analyze_sync(video_path, deep_check)

        if not report.is_valid:
            error_msg = "Video quality validation failed:\n"
            for issue in report.issues:
                error_msg += f"  - {issue}\n"
            raise ValueError(error_msg.rstrip())

        return report

    async def validate_or_raise_async(
        self,
        video_path: str,
        deep_check: Optional[bool] = None,
    ) -> VideoQualityReport:
        """Asynchronously validate video and raise exception if invalid.

        Args:
            video_path: Path to video file
            deep_check: Override default deep_check setting

        Returns:
            VideoQualityReport if valid

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video quality validation fails
        """
        return await asyncio.to_thread(
            self.validate_or_raise,
            video_path,
            deep_check,
        )
