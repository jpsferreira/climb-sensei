"""Tests for easy wins improvements in climbing analysis."""

import pytest
from climb_sensei.metrics import ClimbingAnalyzer
from climb_sensei.biomechanics import (
    calculate_limb_angles,
    calculate_total_distance_traveled,
)
from climb_sensei.config import LandmarkIndex


class TestLimbAngles:
    """Tests for limb angle calculations."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_calculate_limb_angles_basic(self, sample_landmarks):
        """Test basic limb angle calculation."""
        angles = calculate_limb_angles(sample_landmarks, LandmarkIndex)

        assert "left_elbow" in angles
        assert "right_elbow" in angles
        assert "left_shoulder" in angles
        assert "right_shoulder" in angles
        assert "left_knee" in angles
        assert "right_knee" in angles
        assert "left_hip" in angles
        assert "right_hip" in angles

    def test_calculate_limb_angles_insufficient_landmarks(self):
        """Test with insufficient landmarks."""
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(10)]
        angles = calculate_limb_angles(landmarks, LandmarkIndex)

        assert angles == {}

    def test_calculate_limb_angles_values_in_range(self, sample_landmarks):
        """Test that angles are in valid range (0-180 degrees)."""
        angles = calculate_limb_angles(sample_landmarks, LandmarkIndex)

        for joint_name, angle in angles.items():
            assert 0.0 <= angle <= 180.0, f"{joint_name} angle out of range: {angle}"


class TestTotalDistance:
    """Tests for total distance calculation."""

    def test_calculate_total_distance_empty(self):
        """Test with empty positions list."""
        distance = calculate_total_distance_traveled([])
        assert distance == 0.0

    def test_calculate_total_distance_single_point(self):
        """Test with single point."""
        distance = calculate_total_distance_traveled([(0.5, 0.5)])
        assert distance == 0.0

    def test_calculate_total_distance_straight_line(self):
        """Test with straight line movement."""
        positions = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0)]
        distance = calculate_total_distance_traveled(positions)
        assert abs(distance - 0.6) < 0.01

    def test_calculate_total_distance_diagonal(self):
        """Test with diagonal movement."""
        positions = [(0.0, 0.0), (0.3, 0.4)]  # 3-4-5 triangle
        distance = calculate_total_distance_traveled(positions)
        assert abs(distance - 0.5) < 0.01


class TestMovementEconomy:
    """Tests for movement economy metric."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_movement_economy_initialization(self, sample_landmarks):
        """Test initial movement economy is zero."""
        analyzer = ClimbingAnalyzer()
        metrics = analyzer.analyze_frame(sample_landmarks)

        assert "movement_economy" in metrics
        assert metrics["movement_economy"] == 0.0

    def test_movement_economy_efficient_climb(self, sample_landmarks):
        """Test movement economy with efficient (mostly vertical) movement."""
        analyzer = ClimbingAnalyzer()

        # Move mostly vertically (efficient)
        for i in range(10):
            landmarks = sample_landmarks.copy()
            for lm in landmarks:
                lm["y"] -= i * 0.01  # Move up
            analyzer.analyze_frame(landmarks)

        summary = analyzer.get_summary()
        # Efficient movement should have economy close to 1.0
        assert summary["avg_movement_economy"] > 0.5

    def test_movement_economy_inefficient_climb(self, sample_landmarks):
        """Test movement economy with inefficient (lots of lateral) movement."""
        analyzer = ClimbingAnalyzer()

        # Move with lots of lateral movement (inefficient)
        for i in range(10):
            landmarks = sample_landmarks.copy()
            for lm in landmarks:
                lm["x"] += i * 0.02  # Lateral movement
                lm["y"] -= i * 0.005  # Small vertical progress
            analyzer.analyze_frame(landmarks)

        summary = analyzer.get_summary()
        # Inefficient movement should have lower economy
        assert summary["avg_movement_economy"] < 0.5


class TestLockOffDetection:
    """Tests for lock-off detection."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_lock_off_not_detected_initially(self, sample_landmarks):
        """Test lock-off not detected on first frame."""
        analyzer = ClimbingAnalyzer()
        metrics = analyzer.analyze_frame(sample_landmarks)

        assert "is_lock_off" in metrics
        assert not metrics["is_lock_off"]

    def test_lock_off_detected_bent_arm_static(self, sample_landmarks):
        """Test lock-off detected with bent arm and low velocity."""
        analyzer = ClimbingAnalyzer()

        # Create landmarks with bent left elbow
        landmarks = sample_landmarks.copy()
        # Make left elbow bent (< 90 degrees)
        landmarks[LandmarkIndex.LEFT_WRIST]["x"] = landmarks[
            LandmarkIndex.LEFT_SHOULDER
        ]["x"]
        landmarks[LandmarkIndex.LEFT_WRIST]["y"] = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["y"] + 0.05
        )
        landmarks[LandmarkIndex.LEFT_ELBOW]["x"] = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["x"] + 0.03
        )
        landmarks[LandmarkIndex.LEFT_ELBOW]["y"] = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["y"] + 0.03
        )

        # First frame
        analyzer.analyze_frame(landmarks)
        # Second frame (same position = low velocity)
        metrics = analyzer.analyze_frame(landmarks)

        # Should detect lock-off
        assert metrics["left_lock_off"] or metrics["is_lock_off"]

    def test_lock_off_count_in_summary(self, sample_landmarks):
        """Test lock-off count in summary."""
        analyzer = ClimbingAnalyzer()

        for i in range(10):
            analyzer.analyze_frame(sample_landmarks)

        summary = analyzer.get_summary()
        assert "lock_off_count" in summary
        assert "lock_off_percentage" in summary
        assert 0 <= summary["lock_off_percentage"] <= 100


class TestRestPositionDetection:
    """Tests for rest position detection."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_rest_position_detected_vertical_static(self, sample_landmarks):
        """Test rest position detected when vertical and static."""
        analyzer = ClimbingAnalyzer()

        # Create landmarks with body in vertical position (shoulders directly above hips)
        landmarks = sample_landmarks.copy()
        # Make shoulders and hips aligned vertically for low body angle

        avg_hip_x = (
            landmarks[LandmarkIndex.LEFT_HIP]["x"]
            + landmarks[LandmarkIndex.RIGHT_HIP]["x"]
        ) / 2

        # Align them (same x coordinate = vertical = 0 degrees)
        for idx in [LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER]:
            landmarks[idx]["x"] = avg_hip_x

        # First frame
        analyzer.analyze_frame(landmarks)
        # Second frame (same position = low velocity, vertical = low body angle)
        metrics = analyzer.analyze_frame(landmarks)

        assert "is_rest_position" in metrics
        # With low body angle and no movement, should be rest
        assert metrics["is_rest_position"]

    def test_rest_position_not_detected_when_moving(self, sample_landmarks):
        """Test rest position not detected when moving."""
        analyzer = ClimbingAnalyzer()

        # First frame
        analyzer.analyze_frame(sample_landmarks)

        # Second frame with movement
        landmarks2 = sample_landmarks.copy()
        for lm in landmarks2:
            lm["x"] += 0.1
            lm["y"] += 0.1
        metrics = analyzer.analyze_frame(landmarks2)

        # Should not be rest position when moving
        assert not metrics["is_rest_position"]

    def test_rest_count_in_summary(self, sample_landmarks):
        """Test rest count in summary."""
        analyzer = ClimbingAnalyzer()

        for i in range(10):
            analyzer.analyze_frame(sample_landmarks)

        summary = analyzer.get_summary()
        assert "rest_count" in summary
        assert "rest_percentage" in summary
        assert 0 <= summary["rest_percentage"] <= 100


class TestFatigueScore:
    """Tests for fatigue score calculation."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_fatigue_score_insufficient_data(self, sample_landmarks):
        """Test fatigue score with insufficient data."""
        analyzer = ClimbingAnalyzer()

        for i in range(20):
            analyzer.analyze_frame(sample_landmarks)

        summary = analyzer.get_summary()
        assert "fatigue_score" in summary
        assert summary["fatigue_score"] == 0.0

    def test_fatigue_score_no_degradation(self, sample_landmarks):
        """Test fatigue score with consistent movement quality."""
        analyzer = ClimbingAnalyzer()

        # Consistent movement
        for i in range(100):
            landmarks = sample_landmarks.copy()
            for lm in landmarks:
                lm["y"] -= i * 0.001
            analyzer.analyze_frame(landmarks)

        summary = analyzer.get_summary()
        assert summary["fatigue_score"] >= 0.0

    def test_fatigue_score_with_degradation(self, sample_landmarks):
        """Test fatigue score with degrading movement quality."""
        analyzer = ClimbingAnalyzer()

        # Good quality early
        for i in range(50):
            landmarks = sample_landmarks.copy()
            for lm in landmarks:
                lm["y"] -= i * 0.001
            analyzer.analyze_frame(landmarks)

        # Poor quality later (more lateral movement = higher jerk/sway)
        for i in range(50, 100):
            landmarks = sample_landmarks.copy()
            for lm in landmarks:
                lm["x"] += (i - 50) * 0.003  # Increased lateral movement
                lm["y"] -= i * 0.001
            analyzer.analyze_frame(landmarks)

        summary = analyzer.get_summary()
        # Should show some fatigue
        assert summary["fatigue_score"] > 0.0


class TestJointAngleHistory:
    """Tests for joint angle history tracking."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_joint_angles_in_metrics(self, sample_landmarks):
        """Test joint angles included in frame metrics."""
        analyzer = ClimbingAnalyzer()
        metrics = analyzer.analyze_frame(sample_landmarks)

        joint_names = [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]

        for joint_name in joint_names:
            assert joint_name in metrics

    def test_joint_angles_in_history(self, sample_landmarks):
        """Test joint angles tracked in history."""
        analyzer = ClimbingAnalyzer()

        for i in range(10):
            analyzer.analyze_frame(sample_landmarks)

        history = analyzer.get_history()

        joint_names = [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]

        for joint_name in joint_names:
            assert joint_name in history
            assert len(history[joint_name]) == 10

    def test_joint_angles_in_summary(self, sample_landmarks):
        """Test average joint angles in summary."""
        analyzer = ClimbingAnalyzer()

        for i in range(10):
            analyzer.analyze_frame(sample_landmarks)

        summary = analyzer.get_summary()

        joint_names = [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]

        for joint_name in joint_names:
            assert f"avg_{joint_name}" in summary


class TestResetWithNewMetrics:
    """Tests for reset functionality with new metrics."""

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_reset_clears_all_histories(self, sample_landmarks):
        """Test reset clears all history buffers."""
        analyzer = ClimbingAnalyzer()

        # Analyze some frames
        for i in range(10):
            analyzer.analyze_frame(sample_landmarks)

        # Reset
        analyzer.reset()

        # Check all histories are cleared
        history = analyzer.get_history()
        assert len(history["movement_economy"]) == 0
        assert len(history["lock_offs"]) == 0
        assert len(history["rest_positions"]) == 0

        for joint_name in [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
        ]:
            assert len(history[joint_name]) == 0

        assert analyzer._total_distance_traveled == 0.0
