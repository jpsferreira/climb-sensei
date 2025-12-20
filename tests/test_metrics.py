"""Tests for the metrics module."""

import pytest
from climb_sensei.metrics import ClimbingAnalyzer, AdvancedClimbingMetrics


@pytest.fixture
def sample_landmarks():
    """Create sample landmarks for testing."""
    landmarks = []
    for i in range(33):
        landmarks.append(
            {
                "x": 0.5,
                "y": 0.5,
                "z": 0.0,
            }
        )

    # Set specific landmarks for testing
    landmarks[11] = {"x": 0.4, "y": 0.4, "z": 0.0}  # LEFT_SHOULDER
    landmarks[12] = {"x": 0.6, "y": 0.4, "z": 0.0}  # RIGHT_SHOULDER
    landmarks[23] = {"x": 0.4, "y": 0.6, "z": 0.0}  # LEFT_HIP
    landmarks[24] = {"x": 0.6, "y": 0.6, "z": 0.0}  # RIGHT_HIP
    landmarks[15] = {"x": 0.3, "y": 0.3, "z": 0.0}  # LEFT_WRIST
    landmarks[16] = {"x": 0.7, "y": 0.3, "z": 0.0}  # RIGHT_WRIST
    landmarks[27] = {"x": 0.4, "y": 0.8, "z": 0.0}  # LEFT_ANKLE
    landmarks[28] = {"x": 0.6, "y": 0.8, "z": 0.0}  # RIGHT_ANKLE

    return landmarks


class TestClimbingAnalyzer:
    """Tests for ClimbingAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30.0)

        assert analyzer.window_size == 30
        assert analyzer.fps == 30.0
        assert analyzer.dt == 1.0 / 30.0
        assert analyzer.total_frames == 0
        assert analyzer.initial_hip_height is None

    def test_analyze_frame_basic(self, sample_landmarks):
        """Test basic frame analysis."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30.0)

        metrics = analyzer.analyze_frame(sample_landmarks)

        assert "hip_height" in metrics
        assert "com_x" in metrics
        assert "com_y" in metrics
        assert "com_velocity" in metrics
        assert "com_sway" in metrics
        assert "vertical_progress" in metrics
        assert "jerk" in metrics
        assert "body_angle" in metrics
        assert "hand_span" in metrics
        assert "foot_span" in metrics

        assert analyzer.total_frames == 1
        assert analyzer.initial_hip_height is not None

    def test_analyze_frame_empty_landmarks(self):
        """Test analysis with empty landmarks."""
        analyzer = ClimbingAnalyzer()

        metrics = analyzer.analyze_frame([])

        assert metrics == {}
        assert analyzer.total_frames == 0

    def test_analyze_frame_insufficient_landmarks(self):
        """Test analysis with insufficient landmarks."""
        analyzer = ClimbingAnalyzer()
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(10)]

        metrics = analyzer.analyze_frame(landmarks)

        assert metrics == {}

    def test_velocity_calculation(self, sample_landmarks):
        """Test velocity calculation over multiple frames."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30.0)

        # First frame - velocity should be 0
        metrics1 = analyzer.analyze_frame(sample_landmarks)
        assert metrics1["com_velocity"] == 0.0

        # Second frame - move landmarks
        landmarks2 = sample_landmarks.copy()
        for lm in landmarks2:
            lm["x"] += 0.1
            lm["y"] += 0.1

        metrics2 = analyzer.analyze_frame(landmarks2)
        assert metrics2["com_velocity"] > 0.0

    def test_sway_calculation(self, sample_landmarks):
        """Test sway calculation over multiple frames."""
        analyzer = ClimbingAnalyzer(window_size=30, fps=30.0)

        # First two frames - sway should be 0
        analyzer.analyze_frame(sample_landmarks)
        metrics2 = analyzer.analyze_frame(sample_landmarks)
        assert metrics2["com_sway"] == 0.0

        # Third frame with lateral movement
        landmarks3 = sample_landmarks.copy()
        for lm in landmarks3:
            lm["x"] += 0.2

        metrics3 = analyzer.analyze_frame(landmarks3)
        assert metrics3["com_sway"] > 0.0

    def test_vertical_progress_tracking(self, sample_landmarks):
        """Test vertical progress tracking."""
        analyzer = ClimbingAnalyzer()

        # First frame
        metrics1 = analyzer.analyze_frame(sample_landmarks)
        assert metrics1["vertical_progress"] == 0.0

        # Second frame - move up (lower y value)
        landmarks2 = sample_landmarks.copy()
        landmarks2[23]["y"] -= 0.1  # LEFT_HIP
        landmarks2[24]["y"] -= 0.1  # RIGHT_HIP

        metrics2 = analyzer.analyze_frame(landmarks2)
        assert metrics2["vertical_progress"] > 0.0

    def test_get_summary(self, sample_landmarks):
        """Test summary statistics calculation."""
        analyzer = ClimbingAnalyzer()

        # Analyze multiple frames
        for i in range(10):
            landmarks = sample_landmarks.copy()
            # Add some variation
            for lm in landmarks:
                lm["y"] -= i * 0.01
                lm["x"] += (i % 3) * 0.01
            analyzer.analyze_frame(landmarks)

        summary = analyzer.get_summary()

        assert "total_frames" in summary
        assert summary["total_frames"] == 10
        assert "avg_velocity" in summary
        assert "max_velocity" in summary
        assert "avg_sway" in summary
        assert "avg_jerk" in summary
        assert "avg_body_angle" in summary
        assert "total_vertical_progress" in summary
        assert "max_height" in summary

    def test_get_summary_empty(self):
        """Test summary with no frames analyzed."""
        analyzer = ClimbingAnalyzer()

        summary = analyzer.get_summary()

        assert summary == {}

    def test_get_history(self, sample_landmarks):
        """Test history retrieval."""
        analyzer = ClimbingAnalyzer()

        # Analyze frames
        for _ in range(5):
            analyzer.analyze_frame(sample_landmarks)

        history = analyzer.get_history()

        assert "hip_heights" in history
        assert "velocities" in history
        assert "sways" in history
        assert "jerks" in history
        assert "body_angles" in history
        assert "hand_spans" in history
        assert "foot_spans" in history

        assert len(history["hip_heights"]) == 5
        assert len(history["velocities"]) == 5

    def test_reset(self, sample_landmarks):
        """Test analyzer reset."""
        analyzer = ClimbingAnalyzer()

        # Analyze some frames
        analyzer.analyze_frame(sample_landmarks)
        analyzer.analyze_frame(sample_landmarks)
        assert analyzer.total_frames == 2

        # Reset
        analyzer.reset()

        assert analyzer.total_frames == 0
        assert analyzer.initial_hip_height is None
        assert len(analyzer._history_hip_heights) == 0
        assert len(analyzer._history_velocities) == 0


class TestAdvancedClimbingMetrics:
    """Tests for AdvancedClimbingMetrics class."""

    def test_calculate_jerk_insufficient_data(self):
        """Test jerk calculation with insufficient data."""
        positions = [(0, 0), (1, 1)]

        jerk = AdvancedClimbingMetrics.calculate_jerk(positions, 0.033)

        assert jerk == 0.0

    def test_calculate_jerk_constant_velocity(self):
        """Test jerk with constant velocity (should be near zero)."""
        positions = [(i * 0.1, i * 0.1) for i in range(10)]

        jerk = AdvancedClimbingMetrics.calculate_jerk(positions, 0.033)

        # Constant velocity means zero acceleration and zero jerk
        assert jerk < 0.1  # Allow small numerical errors

    def test_calculate_jerk_accelerating(self):
        """Test jerk with acceleration (should be positive)."""
        positions = [(i * i * 0.01, i * i * 0.01) for i in range(10)]

        jerk = AdvancedClimbingMetrics.calculate_jerk(positions, 0.033)

        assert jerk > 0.0

    def test_calculate_base_of_support(self, sample_landmarks):
        """Test base of support calculation."""
        result = AdvancedClimbingMetrics.calculate_base_of_support(sample_landmarks)

        assert "hand_span" in result
        assert "foot_span" in result
        assert "hand_foot_span" in result

        assert result["hand_span"] > 0.0
        assert result["foot_span"] > 0.0
        assert result["hand_foot_span"] > 0.0

    def test_calculate_base_of_support_empty(self):
        """Test base of support with empty landmarks."""
        result = AdvancedClimbingMetrics.calculate_base_of_support([])

        assert result == {}

    def test_calculate_base_of_support_insufficient(self):
        """Test base of support with insufficient landmarks."""
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(10)]

        result = AdvancedClimbingMetrics.calculate_base_of_support(landmarks)

        assert result == {}

    def test_calculate_body_angle(self, sample_landmarks):
        """Test body angle calculation."""
        angle = AdvancedClimbingMetrics.calculate_body_angle(sample_landmarks)

        assert isinstance(angle, float)
        # Angle should be between 0 and 90 degrees (lean from vertical)
        assert 0 <= angle <= 90

    def test_calculate_body_angle_vertical(self):
        """Test body angle when perfectly vertical."""
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(33)]

        # Shoulders and hips aligned vertically (shoulders above hips in image coords)
        landmarks[11] = {"x": 0.5, "y": 0.4, "z": 0.0}  # LEFT_SHOULDER
        landmarks[12] = {"x": 0.5, "y": 0.4, "z": 0.0}  # RIGHT_SHOULDER
        landmarks[23] = {"x": 0.5, "y": 0.6, "z": 0.0}  # LEFT_HIP
        landmarks[24] = {"x": 0.5, "y": 0.6, "z": 0.0}  # RIGHT_HIP

        angle = AdvancedClimbingMetrics.calculate_body_angle(landmarks)

        # When perfectly vertical, angle should be 0
        assert abs(angle) < 1.0

    def test_calculate_body_angle_leaning(self):
        """Test body angle when leaning."""
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(33)]

        # Shoulders offset from hips horizontally (leaning)
        landmarks[11] = {"x": 0.3, "y": 0.4, "z": 0.0}  # LEFT_SHOULDER
        landmarks[12] = {"x": 0.3, "y": 0.4, "z": 0.0}  # RIGHT_SHOULDER
        landmarks[23] = {"x": 0.5, "y": 0.6, "z": 0.0}  # LEFT_HIP
        landmarks[24] = {"x": 0.5, "y": 0.6, "z": 0.0}  # RIGHT_HIP

        angle = AdvancedClimbingMetrics.calculate_body_angle(landmarks)

        # With dx=0.2 and dy=0.2, angle should be arctan(0.2/0.2) = 45 degrees
        assert 40 < angle < 50
        """Test body angle with empty landmarks."""
        angle = AdvancedClimbingMetrics.calculate_body_angle([])

        assert angle == 0.0
