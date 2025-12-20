"""Tests for typed API (models and protocols)."""

import pytest

from climb_sensei import (
    ClimbingAnalyzer,
    Landmark,
    FrameMetrics,
    ClimbingSummary,
    ClimbingAnalysis,
)
from climb_sensei.protocols import MetricsAnalyzer


class TestLandmarkModel:
    """Test Landmark data model."""

    def test_create_landmark(self):
        """Test creating a landmark."""
        landmark = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
        assert landmark.x == 0.5
        assert landmark.y == 0.3
        assert landmark.z == 0.1
        assert landmark.visibility == 0.9

    def test_landmark_immutable(self):
        """Test that landmarks are immutable."""
        landmark = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
        with pytest.raises(AttributeError):
            landmark.x = 0.7  # type: ignore

    def test_from_dict(self):
        """Test creating landmark from dictionary."""
        data = {"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9}
        landmark = Landmark.from_dict(data)
        assert landmark.x == 0.5
        assert landmark.y == 0.3

    def test_to_tuple_2d(self):
        """Test 2D tuple conversion."""
        landmark = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
        assert landmark.to_tuple_2d() == (0.5, 0.3)

    def test_to_tuple_3d(self):
        """Test 3D tuple conversion."""
        landmark = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
        assert landmark.to_tuple_3d() == (0.5, 0.3, 0.1)

    def test_to_dict(self):
        """Test dictionary conversion."""
        landmark = Landmark(x=0.5, y=0.3, z=0.1, visibility=0.9)
        d = landmark.to_dict()
        assert d == {"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9}


class TestFrameMetrics:
    """Test FrameMetrics data model."""

    def test_create_frame_metrics(self):
        """Test creating frame metrics."""
        metrics = FrameMetrics(
            hip_height=0.5,
            com_velocity=0.01,
            com_sway=0.02,
            jerk=5.0,
            vertical_progress=0.1,
            movement_economy=0.8,
            is_lock_off=False,
            left_lock_off=False,
            right_lock_off=False,
            is_rest_position=False,
            body_angle=10.0,
            hand_span=0.4,
            foot_span=0.3,
            left_elbow=145.0,
            right_elbow=148.0,
            left_shoulder=95.0,
            right_shoulder=92.0,
            left_knee=170.0,
            right_knee=168.0,
            left_hip=120.0,
            right_hip=122.0,
        )
        assert metrics.hip_height == 0.5
        assert metrics.com_velocity == 0.01
        assert isinstance(metrics.is_lock_off, bool)

    def test_frame_metrics_immutable(self):
        """Test that frame metrics are immutable."""
        metrics = FrameMetrics(
            hip_height=0.5,
            com_velocity=0.01,
            com_sway=0.02,
            jerk=5.0,
            vertical_progress=0.1,
            movement_economy=0.8,
            is_lock_off=False,
            left_lock_off=False,
            right_lock_off=False,
            is_rest_position=False,
            body_angle=10.0,
            hand_span=0.4,
            foot_span=0.3,
            left_elbow=145.0,
            right_elbow=148.0,
            left_shoulder=95.0,
            right_shoulder=92.0,
            left_knee=170.0,
            right_knee=168.0,
            left_hip=120.0,
            right_hip=122.0,
        )
        with pytest.raises(AttributeError):
            metrics.hip_height = 0.6  # type: ignore

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = FrameMetrics(
            hip_height=0.5,
            com_velocity=0.01,
            com_sway=0.02,
            jerk=5.0,
            vertical_progress=0.1,
            movement_economy=0.8,
            is_lock_off=False,
            left_lock_off=False,
            right_lock_off=False,
            is_rest_position=False,
            body_angle=10.0,
            hand_span=0.4,
            foot_span=0.3,
            left_elbow=145.0,
            right_elbow=148.0,
            left_shoulder=95.0,
            right_shoulder=92.0,
            left_knee=170.0,
            right_knee=168.0,
            left_hip=120.0,
            right_hip=122.0,
        )
        d = metrics.to_dict()
        assert d["hip_height"] == 0.5
        assert d["com_velocity"] == 0.01
        assert not d["is_lock_off"]


class TestTypedAnalyzerAPI:
    """Test typed API methods on ClimbingAnalyzer."""

    def create_test_landmarks(self):
        """Create test landmarks."""
        landmarks = []
        for i in range(33):
            landmarks.append(
                {
                    "x": 0.5 + (i * 0.01),
                    "y": 0.5 - (i * 0.01),
                    "z": 0.0,
                    "visibility": 0.9,
                }
            )
        return landmarks

    def test_analyze_frame_typed(self):
        """Test analyze_frame_typed returns FrameMetrics."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30)
        landmarks = self.create_test_landmarks()

        metrics = analyzer.analyze_frame_typed(landmarks)

        assert isinstance(metrics, FrameMetrics)
        assert isinstance(metrics.hip_height, float)
        assert isinstance(metrics.is_lock_off, bool)

        # Test immutability
        with pytest.raises(AttributeError):
            metrics.hip_height = 0.9  # type: ignore

    def test_get_summary_typed(self):
        """Test get_summary_typed returns ClimbingSummary."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30)
        landmarks = self.create_test_landmarks()

        # Analyze a few frames
        for _ in range(5):
            analyzer.analyze_frame(landmarks)

        summary = analyzer.get_summary_typed()

        assert isinstance(summary, ClimbingSummary)
        assert isinstance(summary.total_frames, int)
        assert isinstance(summary.avg_velocity, float)
        assert summary.total_frames == 5

        # Test immutability
        with pytest.raises(AttributeError):
            summary.total_frames = 10  # type: ignore

    def test_backward_compatibility(self):
        """Test that old dict API still works."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30)
        landmarks = self.create_test_landmarks()

        # Old API should still work
        metrics_dict = analyzer.analyze_frame(landmarks)
        assert isinstance(metrics_dict, dict)
        assert "hip_height" in metrics_dict

        summary_dict = analyzer.get_summary()
        assert isinstance(summary_dict, dict)
        assert "total_frames" in summary_dict

    def test_dict_and_typed_equivalent(self):
        """Test that dict and typed APIs return same values."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30)
        landmarks = self.create_test_landmarks()

        # Get both versions
        metrics_dict = analyzer.analyze_frame(landmarks)
        analyzer.reset()
        metrics_typed = analyzer.analyze_frame_typed(landmarks)

        # Values should be the same
        assert metrics_dict["hip_height"] == metrics_typed.hip_height
        assert metrics_dict["com_velocity"] == metrics_typed.com_velocity
        assert metrics_dict["is_lock_off"] == metrics_typed.is_lock_off


class TestProtocolConformance:
    """Test that classes conform to protocols."""

    def test_climbing_analyzer_conforms_to_protocol(self):
        """Test ClimbingAnalyzer conforms to MetricsAnalyzer protocol."""
        analyzer = ClimbingAnalyzer()
        assert isinstance(analyzer, MetricsAnalyzer)


class TestClimbingAnalysis:
    """Test ClimbingAnalysis model."""

    def test_create_analysis(self):
        """Test creating complete analysis."""
        summary = ClimbingSummary(
            total_frames=100,
            total_vertical_progress=2.5,
            max_height=2.8,
            avg_velocity=0.05,
            max_velocity=0.15,
            avg_sway=0.01,
            max_sway=0.03,
            avg_jerk=10.0,
            max_jerk=30.0,
            avg_body_angle=15.0,
            avg_hand_span=0.4,
            avg_foot_span=0.3,
            total_distance_traveled=3.2,
            avg_movement_economy=0.78,
            lock_off_count=25,
            lock_off_percentage=25.0,
            rest_count=10,
            rest_percentage=10.0,
            fatigue_score=0.15,
            avg_left_elbow=145.0,
            avg_right_elbow=143.0,
            avg_left_shoulder=95.0,
            avg_right_shoulder=93.0,
            avg_left_knee=168.0,
            avg_right_knee=170.0,
            avg_left_hip=120.0,
            avg_right_hip=122.0,
        )

        history = {
            "hip_heights": [0.5, 0.4, 0.3],
            "velocities": [0.01, 0.02, 0.03],
        }

        analysis = ClimbingAnalysis(
            summary=summary, history=history, video_path="test.mp4"
        )

        assert analysis.summary.total_frames == 100
        assert analysis.video_path == "test.mp4"
        assert len(analysis.history["velocities"]) == 3

    def test_analysis_to_dict(self):
        """Test converting analysis to dict."""
        summary = ClimbingSummary(
            total_frames=100,
            total_vertical_progress=2.5,
            max_height=2.8,
            avg_velocity=0.05,
            max_velocity=0.15,
            avg_sway=0.01,
            max_sway=0.03,
            avg_jerk=10.0,
            max_jerk=30.0,
            avg_body_angle=15.0,
            avg_hand_span=0.4,
            avg_foot_span=0.3,
            total_distance_traveled=3.2,
            avg_movement_economy=0.78,
            lock_off_count=25,
            lock_off_percentage=25.0,
            rest_count=10,
            rest_percentage=10.0,
            fatigue_score=0.15,
            avg_left_elbow=145.0,
            avg_right_elbow=143.0,
            avg_left_shoulder=95.0,
            avg_right_shoulder=93.0,
            avg_left_knee=168.0,
            avg_right_knee=170.0,
            avg_left_hip=120.0,
            avg_right_hip=122.0,
        )

        analysis = ClimbingAnalysis(summary=summary, history={}, video_path="test.mp4")

        d = analysis.to_dict()
        assert d["video_path"] == "test.mp4"
        assert d["summary"]["total_frames"] == 100
