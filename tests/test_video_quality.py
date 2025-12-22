"""Tests for video quality checker."""

import pytest
from pathlib import Path
from climb_sensei.video_quality import (
    VideoQualityChecker,
    VideoQualityReport,
    check_video_quality,
)


class TestVideoQualityChecker:
    """Test video quality assessment functionality."""

    def test_checker_initialization(self):
        """Test VideoQualityChecker initialization."""
        # Basic initialization
        checker = VideoQualityChecker(deep_check=False)
        assert checker.deep_check is False

        # With deep check
        checker = VideoQualityChecker(deep_check=True)
        assert checker.deep_check is True

    def test_quality_thresholds(self):
        """Test quality threshold constants."""
        checker = VideoQualityChecker()

        # Resolution thresholds
        assert checker.MIN_RESOLUTION == (640, 480)
        assert checker.RECOMMENDED_RESOLUTION == (1280, 720)
        assert checker.OPTIMAL_RESOLUTION == (1920, 1080)

        # FPS thresholds
        assert checker.MIN_FPS == 15
        assert checker.RECOMMENDED_FPS == 30
        assert checker.OPTIMAL_FPS == 60

        # Duration thresholds
        assert checker.MIN_DURATION == 5.0
        assert checker.MAX_DURATION == 600.0

    def test_fourcc_conversion(self):
        """Test FOURCC code to string conversion."""
        checker = VideoQualityChecker()

        # Common codecs
        # H.264: 0x34363248
        h264_code = 0x34363248
        result = checker._fourcc_to_string(h264_code)
        assert isinstance(result, str)
        assert len(result) == 4

    def test_format_compatibility(self):
        """Test codec format compatibility check."""
        checker = VideoQualityChecker()

        # Compatible codecs
        assert checker._check_format_compatibility("avc1")
        assert checker._check_format_compatibility("h264")
        assert checker._check_format_compatibility("H264")
        assert checker._check_format_compatibility("mp4v")
        assert checker._check_format_compatibility("XVID")

        # Empty codec (sometimes returned)
        assert checker._check_format_compatibility("")

        # Unknown codec
        assert not checker._check_format_compatibility("UNKN")

    def test_resolution_assessment(self):
        """Test resolution quality assessment."""
        checker = VideoQualityChecker()

        # Excellent - Full HD
        assert checker._assess_resolution(1920, 1080) == "excellent"
        assert checker._assess_resolution(2560, 1440) == "excellent"

        # Good - HD
        assert checker._assess_resolution(1280, 720) == "good"
        assert checker._assess_resolution(1600, 900) == "good"

        # Acceptable - SD
        assert checker._assess_resolution(640, 480) == "acceptable"
        assert checker._assess_resolution(800, 600) == "acceptable"

        # Poor - below minimum
        assert checker._assess_resolution(320, 240) == "poor"
        assert checker._assess_resolution(400, 300) == "poor"

    def test_fps_assessment(self):
        """Test frame rate quality assessment."""
        checker = VideoQualityChecker()

        # Excellent - high fps
        assert checker._assess_fps(60.0) == "excellent"
        assert checker._assess_fps(120.0) == "excellent"

        # Good - standard fps
        assert checker._assess_fps(30.0) == "good"
        assert checker._assess_fps(50.0) == "good"

        # Acceptable - minimum fps
        assert checker._assess_fps(15.0) == "acceptable"
        assert checker._assess_fps(24.0) == "acceptable"

        # Poor - below minimum
        assert checker._assess_fps(10.0) == "poor"
        assert checker._assess_fps(5.0) == "poor"

    def test_duration_assessment(self):
        """Test video duration quality assessment."""
        checker = VideoQualityChecker()

        # Excellent - ideal range
        assert checker._assess_duration(30.0) == "excellent"
        assert checker._assess_duration(60.0) == "excellent"
        assert checker._assess_duration(120.0) == "excellent"

        # Acceptable - usable but not ideal
        assert checker._assess_duration(5.0) == "acceptable"
        assert checker._assess_duration(200.0) == "acceptable"

        # Poor - too short
        assert checker._assess_duration(2.0) == "poor"
        assert checker._assess_duration(4.0) == "poor"

        # Poor - too long
        assert checker._assess_duration(700.0) == "poor"

    def test_lighting_assessment(self):
        """Test lighting quality assessment."""
        checker = VideoQualityChecker()
        issues = []
        warnings = []
        recommendations = []

        # Excellent - optimal range
        result = checker._assess_lighting(120.0, issues, warnings, recommendations)
        assert result == "excellent"

        # Good - acceptable range
        issues.clear()
        warnings.clear()
        result = checker._assess_lighting(70.0, issues, warnings, recommendations)
        assert result == "good"

        # Acceptable - overexposed
        issues.clear()
        warnings.clear()
        result = checker._assess_lighting(220.0, issues, warnings, recommendations)
        assert result == "acceptable"
        assert len(warnings) > 0

        # Poor - too dark
        issues.clear()
        warnings.clear()
        result = checker._assess_lighting(30.0, issues, warnings, recommendations)
        assert result == "poor"
        assert len(issues) > 0

    def test_stability_assessment(self):
        """Test camera stability assessment."""
        checker = VideoQualityChecker()
        issues = []
        warnings = []
        recommendations = []

        # Excellent - sharp
        result = checker._assess_stability(200.0, issues, warnings, recommendations)
        assert result == "excellent"
        assert len(warnings) == 0

        # Acceptable - motion blur
        warnings.clear()
        result = checker._assess_stability(50.0, issues, warnings, recommendations)
        assert result == "acceptable"
        assert len(warnings) > 0

    def test_check_video_nonexistent_file(self):
        """Test checking non-existent video file."""
        checker = VideoQualityChecker()

        with pytest.raises(FileNotFoundError):
            checker.check_video("nonexistent_video.mp4")

    def test_video_quality_report_structure(self):
        """Test VideoQualityReport dataclass structure."""
        report = VideoQualityReport(
            is_valid=True,
            file_path="/path/to/video.mp4",
            file_size_mb=10.5,
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=900,
            duration_seconds=30.0,
            codec="avc1",
            format_compatible=True,
            resolution_quality="excellent",
            fps_quality="good",
            duration_quality="excellent",
            lighting_quality="excellent",
            stability_quality="excellent",
            issues=[],
            warnings=[],
            recommendations=[],
        )

        assert report.is_valid is True
        assert report.width == 1920
        assert report.height == 1080
        assert report.fps == 30.0
        assert report.codec == "avc1"
        assert isinstance(report.issues, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.recommendations, list)

    def test_convenience_function(self):
        """Test convenience function check_video_quality."""
        # Should raise FileNotFoundError for non-existent file
        with pytest.raises(FileNotFoundError):
            check_video_quality("nonexistent.mp4")

        # Function signature should accept deep_check parameter
        try:
            check_video_quality("test.mp4", deep_check=True)
        except FileNotFoundError:
            pass  # Expected for non-existent file


class TestVideoQualityIntegration:
    """Integration tests requiring actual video files."""

    @pytest.fixture
    def sample_video_path(self):
        """Get path to sample video if available."""
        # Check common locations for test video
        possible_paths = [
            Path("data/test_video.mp4"),
            Path("tests/data/test_video.mp4"),
            Path("../data/test_video.mp4"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

    @pytest.mark.skipif(not Path("data").exists(), reason="No test video available")
    def test_check_real_video(self, sample_video_path):
        """Test checking a real video file if available."""
        if sample_video_path is None:
            pytest.skip("No test video available")

        checker = VideoQualityChecker(deep_check=False)
        report = checker.check_video(sample_video_path)

        # Basic assertions
        assert isinstance(report, VideoQualityReport)
        assert report.width > 0
        assert report.height > 0
        assert report.fps > 0
        assert report.frame_count > 0
        assert report.duration_seconds > 0
        assert isinstance(report.is_valid, bool)

    @pytest.mark.skipif(not Path("data").exists(), reason="No test video available")
    def test_deep_check_real_video(self, sample_video_path):
        """Test deep analysis on real video if available."""
        if sample_video_path is None:
            pytest.skip("No test video available")

        checker = VideoQualityChecker(deep_check=True)
        report = checker.check_video(sample_video_path)

        # Deep check should populate lighting and stability
        assert report.lighting_quality is not None
        assert report.stability_quality is not None
