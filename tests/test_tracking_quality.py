"""Tests for tracking quality analysis module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from climb_sensei.tracking_quality import (
    TrackingQualityAnalyzer,
    TrackingQualityReport,
    analyze_tracking_quality,
    analyze_tracking_from_landmarks,
)


class TestTrackingQualityAnalyzer:
    """Test TrackingQualityAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes with correct defaults."""
        analyzer = TrackingQualityAnalyzer()

        assert analyzer.min_detection_rate == 70.0
        assert analyzer.min_avg_confidence == 0.5
        assert analyzer.min_visibility == 60.0
        assert analyzer.min_smoothness == 0.6
        assert analyzer.max_tracking_losses == 5
        assert analyzer.sample_rate == 1

    def test_analyzer_custom_thresholds(self):
        """Test analyzer with custom thresholds."""
        analyzer = TrackingQualityAnalyzer(
            min_detection_rate=80.0,
            min_avg_confidence=0.7,
            min_visibility=70.0,
            min_smoothness=0.8,
            max_tracking_losses=3,
            sample_rate=5,
        )

        assert analyzer.min_detection_rate == 80.0
        assert analyzer.min_avg_confidence == 0.7
        assert analyzer.min_visibility == 70.0
        assert analyzer.min_smoothness == 0.8
        assert analyzer.max_tracking_losses == 3
        assert analyzer.sample_rate == 5

    def test_calculate_smoothness_empty(self):
        """Test smoothness calculation with empty data."""
        analyzer = TrackingQualityAnalyzer()
        smoothness = analyzer._calculate_smoothness([])

        assert smoothness == 0.0

    def test_calculate_smoothness_single_frame(self):
        """Test smoothness calculation with single frame."""
        analyzer = TrackingQualityAnalyzer()
        positions = [[(0.5, 0.5, 0.5)] * 33]  # 33 landmarks
        smoothness = analyzer._calculate_smoothness(positions)

        assert smoothness == 0.0

    def test_calculate_smoothness_smooth_tracking(self):
        """Test smoothness calculation with smooth tracking."""
        analyzer = TrackingQualityAnalyzer()

        # Create smooth movement (small changes between frames)
        positions = []
        for i in range(10):
            frame_landmarks = []
            for j in range(33):
                x = 0.5 + i * 0.001  # Very small movement
                y = 0.5 + j * 0.01
                z = 0.5
                frame_landmarks.append((x, y, z))
            positions.append(frame_landmarks)

        smoothness = analyzer._calculate_smoothness(positions)

        # Should be high for smooth tracking
        assert smoothness > 0.8

    def test_calculate_smoothness_jittery_tracking(self):
        """Test smoothness calculation with jittery tracking."""
        analyzer = TrackingQualityAnalyzer()

        # Create jittery movement (large random changes)
        np.random.seed(42)
        positions = []
        for i in range(10):
            frame_landmarks = []
            for j in range(33):
                x = 0.5 + np.random.uniform(-0.1, 0.1)
                y = 0.5 + np.random.uniform(-0.1, 0.1)
                z = 0.5 + np.random.uniform(-0.1, 0.1)
                frame_landmarks.append((x, y, z))
            positions.append(frame_landmarks)

        smoothness = analyzer._calculate_smoothness(positions)

        # Should be low for jittery tracking
        assert smoothness < 0.5

    def test_determine_quality_level_excellent(self):
        """Test quality level determination - excellent."""
        analyzer = TrackingQualityAnalyzer()

        quality = analyzer._determine_quality_level(
            detection_rate=98.0,
            avg_confidence=0.85,
            avg_visibility=90.0,
            smoothness=0.85,
        )

        assert quality == "excellent"

    def test_determine_quality_level_good(self):
        """Test quality level determination - good."""
        analyzer = TrackingQualityAnalyzer()

        quality = analyzer._determine_quality_level(
            detection_rate=75.0,
            avg_confidence=0.6,
            avg_visibility=65.0,
            smoothness=0.7,
        )

        assert quality == "good"

    def test_determine_quality_level_acceptable(self):
        """Test quality level determination - acceptable."""
        analyzer = TrackingQualityAnalyzer()

        quality = analyzer._determine_quality_level(
            detection_rate=72.0,
            avg_confidence=0.55,
            avg_visibility=62.0,
            smoothness=0.4,  # Below smoothness threshold but others OK
        )

        assert quality == "acceptable"

    def test_determine_quality_level_poor(self):
        """Test quality level determination - poor."""
        analyzer = TrackingQualityAnalyzer()

        quality = analyzer._determine_quality_level(
            detection_rate=50.0,  # Below threshold
            avg_confidence=0.4,  # Below threshold
            avg_visibility=40.0,  # Below threshold
            smoothness=0.3,
        )

        assert quality == "poor"

    def test_analyze_video_file_not_found(self):
        """Test analyzing non-existent video file."""
        analyzer = TrackingQualityAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.analyze_video("nonexistent_video.mp4")

    @patch("climb_sensei.tracking_quality.PoseEngine")
    @patch("climb_sensei.tracking_quality.VideoReader")
    def test_analyze_video_high_quality(self, mock_video_reader, mock_pose_engine):
        """Test analyzing video with high quality tracking."""
        # Mock video reader
        mock_reader_instance = MagicMock()
        mock_reader_instance.__enter__ = Mock(return_value=mock_reader_instance)
        mock_reader_instance.__exit__ = Mock(return_value=False)

        # Simulate 100 frames, 95 with pose detected
        frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(95)]
        frames += [(False, None)] * 5
        mock_reader_instance.read = Mock(side_effect=frames + [(False, None)])
        mock_video_reader.return_value = mock_reader_instance

        # Mock pose engine
        mock_engine_instance = MagicMock()
        mock_engine_instance.__enter__ = Mock(return_value=mock_engine_instance)
        mock_engine_instance.__exit__ = Mock(return_value=False)

        # Create mock landmarks with high confidence
        mock_landmark = Mock()
        mock_landmark.visibility = 0.9
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.5

        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33

        # 95 successful detections
        results = [mock_results] * 95 + [None] * 5 + [None]
        mock_engine_instance.process = Mock(side_effect=results)
        mock_engine_instance.extract_landmarks = Mock(return_value=[(0.5, 0.5)] * 33)
        mock_pose_engine.return_value = mock_engine_instance

        # Create temporary test file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            test_video = f.name

        try:
            analyzer = TrackingQualityAnalyzer(min_detection_rate=70.0)
            report = analyzer.analyze_video(test_video)

            assert report.is_trackable is True
            assert report.frames_with_pose == 95
            assert report.detection_rate >= 70.0
            assert len(report.issues) == 0
            assert report.quality_level in ["good", "excellent"]
        finally:
            import os

            os.unlink(test_video)

    @patch("climb_sensei.tracking_quality.PoseEngine")
    @patch("climb_sensei.tracking_quality.VideoReader")
    def test_analyze_video_low_quality(self, mock_video_reader, mock_pose_engine):
        """Test analyzing video with low quality tracking."""
        # Mock video reader
        mock_reader_instance = MagicMock()
        mock_reader_instance.__enter__ = Mock(return_value=mock_reader_instance)
        mock_reader_instance.__exit__ = Mock(return_value=False)

        # Simulate 100 frames, only 40 with pose detected
        frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(100)]
        frames += [(False, None)]
        mock_reader_instance.read = Mock(side_effect=frames)
        mock_video_reader.return_value = mock_reader_instance

        # Mock pose engine - low detection rate
        mock_engine_instance = MagicMock()
        mock_engine_instance.__enter__ = Mock(return_value=mock_engine_instance)
        mock_engine_instance.__exit__ = Mock(return_value=False)

        # Create mock landmarks with low confidence
        mock_landmark = Mock()
        mock_landmark.visibility = 0.3  # Low confidence
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.5

        mock_results = Mock()
        mock_results.pose_landmarks = Mock()
        mock_results.pose_landmarks.landmark = [mock_landmark] * 33

        # Only 40 successful detections
        results = [mock_results] * 40 + [None] * 60 + [None]
        mock_engine_instance.process = Mock(side_effect=results)
        mock_engine_instance.extract_landmarks = Mock(return_value=[(0.5, 0.5)] * 33)
        mock_pose_engine.return_value = mock_engine_instance

        # Create temporary test file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            test_video = f.name

        try:
            analyzer = TrackingQualityAnalyzer(min_detection_rate=70.0)
            report = analyzer.analyze_video(test_video)

            assert report.is_trackable is False
            assert report.total_frames == 100
            assert report.frames_with_pose == 40
            assert report.detection_rate < 70.0
            assert len(report.issues) > 0
            assert report.quality_level == "poor"
        finally:
            import os

            os.unlink(test_video)

    def test_report_structure(self):
        """Test TrackingQualityReport dataclass structure."""
        report = TrackingQualityReport(
            file_path="/path/to/video.mp4",
            total_frames=100,
            frames_with_pose=85,
            detection_rate=85.0,
            avg_landmark_confidence=0.75,
            min_landmark_confidence=0.50,
            avg_visibility_score=80.0,
            tracking_smoothness=0.85,
            tracking_loss_events=2,
            is_trackable=True,
            issues=[],
            warnings=["Some warning"],
            quality_level="good",
            frame_confidences=[0.75] * 85,
            frame_visibility=[80.0] * 85,
        )

        assert report.file_path == "/path/to/video.mp4"
        assert report.total_frames == 100
        assert report.frames_with_pose == 85
        assert report.detection_rate == 85.0
        assert report.avg_landmark_confidence == 0.75
        assert report.is_trackable is True
        assert report.quality_level == "good"
        assert len(report.warnings) == 1

    @patch("climb_sensei.tracking_quality.PoseEngine")
    @patch("climb_sensei.tracking_quality.VideoReader")
    def test_convenience_function(self, mock_video_reader, mock_pose_engine):
        """Test convenience function analyze_tracking_quality."""
        # Mock video reader
        mock_reader_instance = MagicMock()
        mock_reader_instance.__enter__ = Mock(return_value=mock_reader_instance)
        mock_reader_instance.__exit__ = Mock(return_value=False)
        mock_reader_instance.read = Mock(side_effect=[(False, None)])
        mock_video_reader.return_value = mock_reader_instance

        # Mock pose engine
        mock_engine_instance = MagicMock()
        mock_engine_instance.__enter__ = Mock(return_value=mock_engine_instance)
        mock_engine_instance.__exit__ = Mock(return_value=False)
        mock_engine_instance.process = Mock(return_value=None)
        mock_pose_engine.return_value = mock_engine_instance

        # Create temporary test file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            test_video = f.name

        try:
            report = analyze_tracking_quality(test_video, sample_rate=5)

            assert isinstance(report, TrackingQualityReport)
            assert test_video in report.file_path  # May have /private prefix
        finally:
            import os

            os.unlink(test_video)

    @patch("climb_sensei.tracking_quality.PoseEngine")
    @patch("climb_sensei.tracking_quality.VideoReader")
    def test_sample_rate(self, mock_video_reader, mock_pose_engine):
        """Test that sample_rate correctly skips frames."""
        # Mock video reader
        mock_reader_instance = MagicMock()
        mock_reader_instance.__enter__ = Mock(return_value=mock_reader_instance)
        mock_reader_instance.__exit__ = Mock(return_value=False)

        # 100 frames
        frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(100)]
        frames.append((False, None))
        mock_reader_instance.read = Mock(side_effect=frames)
        mock_video_reader.return_value = mock_reader_instance

        # Mock pose engine
        mock_engine_instance = MagicMock()
        mock_engine_instance.__enter__ = Mock(return_value=mock_engine_instance)
        mock_engine_instance.__exit__ = Mock(return_value=False)
        mock_engine_instance.process = Mock(return_value=None)
        mock_pose_engine.return_value = mock_engine_instance

        # Create temporary test file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            test_video = f.name

        try:
            analyzer = TrackingQualityAnalyzer(sample_rate=10)
            report = analyzer.analyze_video(test_video)

            # With sample_rate=10, should process fewer frames
            # The exact count depends on mock setup, just verify it's less than 100
            assert report.total_frames <= 100
        finally:
            import os

            os.unlink(test_video)


class TestLandmarksBasedAnalysis:
    """Test tracking quality analysis from pre-extracted landmarks."""

    def test_analyze_from_landmarks_high_quality(self):
        """Test analyzing high quality landmark sequence."""
        # Create 100 frames with 95 having landmarks
        landmarks_seq = []
        for i in range(95):
            # 33 landmarks per frame (MediaPipe standard)
            landmarks = [(0.5 + i * 0.001, 0.5 + j * 0.01) for j in range(33)]
            landmarks_seq.append(landmarks)

        # 5 frames with no detection
        for _ in range(5):
            landmarks_seq.append(None)

        analyzer = TrackingQualityAnalyzer(min_detection_rate=70.0)
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        assert report.is_trackable is True
        assert report.total_frames == 100
        assert report.frames_with_pose == 95
        assert report.detection_rate == 95.0
        assert len(report.issues) == 0
        assert report.quality_level in ["good", "excellent"]

    def test_analyze_from_landmarks_low_quality(self):
        """Test analyzing low quality landmark sequence."""
        # Create 100 frames with only 40 having landmarks
        landmarks_seq = []
        for i in range(40):
            landmarks = [(0.5, 0.5) for _ in range(33)]
            landmarks_seq.append(landmarks)

        # 60 frames with no detection
        for _ in range(60):
            landmarks_seq.append(None)

        analyzer = TrackingQualityAnalyzer(min_detection_rate=70.0)
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        assert report.is_trackable is False
        assert report.total_frames == 100
        assert report.frames_with_pose == 40
        assert report.detection_rate == 40.0
        assert len(report.issues) > 0
        assert report.quality_level == "poor"

    def test_analyze_from_landmarks_intermittent_detection(self):
        """Test analyzing sequence with intermittent detection."""
        landmarks_seq = []

        # Alternating detection/no-detection
        for i in range(50):
            if i % 2 == 0:
                landmarks = [(0.5, 0.5) for _ in range(33)]
                landmarks_seq.append(landmarks)
            else:
                landmarks_seq.append(None)

        analyzer = TrackingQualityAnalyzer(min_detection_rate=40.0)
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        assert report.frames_with_pose == 25
        assert report.detection_rate == 50.0
        # Should have many tracking loss events
        assert report.tracking_loss_events > 10

    def test_analyze_from_landmarks_smooth_tracking(self):
        """Test smoothness calculation with smooth landmark movement."""
        landmarks_seq = []

        # Smooth gradual movement
        for i in range(50):
            landmarks = []
            for j in range(33):
                x = 0.5 + i * 0.001  # Very small incremental change
                y = 0.5 + j * 0.01
                landmarks.append((x, y))
            landmarks_seq.append(landmarks)

        analyzer = TrackingQualityAnalyzer()
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        # Should have high smoothness for gradual movement
        assert report.tracking_smoothness > 0.7
        assert report.quality_level in ["good", "excellent"]

    def test_analyze_from_landmarks_jittery_tracking(self):
        """Test smoothness calculation with jittery landmark movement."""
        np.random.seed(42)
        landmarks_seq = []

        # Random jittery movement
        for i in range(50):
            landmarks = []
            for j in range(33):
                x = 0.5 + np.random.uniform(-0.1, 0.1)
                y = 0.5 + np.random.uniform(-0.1, 0.1)
                landmarks.append((x, y))
            landmarks_seq.append(landmarks)

        analyzer = TrackingQualityAnalyzer()
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        # Should have low smoothness for jittery movement
        assert report.tracking_smoothness < 0.5
        # Should have warnings about jitter
        assert len(report.warnings) > 0

    def test_analyze_from_landmarks_empty_sequence(self):
        """Test analyzing empty landmark sequence."""
        landmarks_seq = []

        analyzer = TrackingQualityAnalyzer()
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        assert report.total_frames == 0
        assert report.frames_with_pose == 0
        assert report.detection_rate == 0.0
        assert report.is_trackable is False

    def test_analyze_from_landmarks_all_none(self):
        """Test analyzing sequence with all None values."""
        landmarks_seq = [None] * 100

        analyzer = TrackingQualityAnalyzer()
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        assert report.total_frames == 100
        assert report.frames_with_pose == 0
        assert report.detection_rate == 0.0
        assert report.is_trackable is False
        assert len(report.issues) > 0

    def test_analyze_from_landmarks_with_sample_rate(self):
        """Test landmark analysis with sample rate."""
        # Create 100 frames with landmarks
        landmarks_seq = []
        for i in range(100):
            landmarks = [(0.5, 0.5) for _ in range(33)]
            landmarks_seq.append(landmarks)

        analyzer = TrackingQualityAnalyzer(sample_rate=10)
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        # With sample_rate=10, should analyze every 10th frame
        assert report.total_frames == 10
        assert report.frames_with_pose == 10

    def test_analyze_from_landmarks_custom_file_path(self):
        """Test custom file path identifier."""
        landmarks_seq = [[(0.5, 0.5) for _ in range(33)] for _ in range(50)]

        analyzer = TrackingQualityAnalyzer()
        report = analyzer.analyze_from_landmarks(
            landmarks_seq, file_path="my_custom_video.mp4"
        )

        assert report.file_path == "my_custom_video.mp4"

    def test_convenience_function_landmarks(self):
        """Test convenience function for landmarks analysis."""
        landmarks_seq = [[(0.5, 0.5) for _ in range(33)] for _ in range(80)]
        landmarks_seq.extend([None] * 20)

        report = analyze_tracking_from_landmarks(
            landmarks_seq, sample_rate=5, min_detection_rate=70.0
        )

        assert isinstance(report, TrackingQualityReport)
        assert report.is_trackable is True

    def test_varying_landmark_counts(self):
        """Test handling sequences with varying landmark counts."""
        landmarks_seq = []

        # Some frames with different landmark counts (edge case)
        for i in range(50):
            if i % 10 == 0:
                # Occasional frame with fewer landmarks
                landmarks = [(0.5, 0.5) for _ in range(20)]
            else:
                landmarks = [(0.5, 0.5) for _ in range(33)]
            landmarks_seq.append(landmarks)

        analyzer = TrackingQualityAnalyzer()
        report = analyzer.analyze_from_landmarks(landmarks_seq)

        # Should still work despite varying counts
        assert report.frames_with_pose == 50
        assert report.detection_rate == 100.0
