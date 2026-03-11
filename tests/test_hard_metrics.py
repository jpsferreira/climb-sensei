"""Tests for climbing analysis via calculator-based service.

Covers movement economy, lock-off detection, rest position detection,
fatigue scoring, joint angle tracking, and reset behavior.
"""

import pytest
from climb_sensei.biomechanics import (
    calculate_limb_angles,
    calculate_total_distance_traveled,
)
from climb_sensei.config import LandmarkIndex
from climb_sensei.services import ClimbingAnalysisService
from climb_sensei.domain.calculators import (
    EfficiencyCalculator,
    TechniqueCalculator,
    FatigueCalculator,
    JointAngleCalculator,
    StabilityCalculator,
)


class TestLimbAngles:
    """Tests for limb angle calculations (biomechanics module)."""

    @pytest.fixture
    def sample_landmarks(self):
        landmarks = []
        for i in range(33):
            landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0})
        return landmarks

    def test_calculate_limb_angles_basic(self, sample_landmarks):
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
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(10)]
        angles = calculate_limb_angles(landmarks, LandmarkIndex)
        assert angles == {}

    def test_calculate_limb_angles_values_in_range(self, sample_landmarks):
        angles = calculate_limb_angles(sample_landmarks, LandmarkIndex)
        for joint_name, angle in angles.items():
            assert 0.0 <= angle <= 180.0, f"{joint_name} angle out of range: {angle}"


class TestTotalDistance:
    """Tests for total distance calculation (biomechanics module)."""

    def test_calculate_total_distance_empty(self):
        assert calculate_total_distance_traveled([]) == 0.0

    def test_calculate_total_distance_single_point(self):
        assert calculate_total_distance_traveled([(0.5, 0.5)]) == 0.0

    def test_calculate_total_distance_straight_line(self):
        positions = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0)]
        assert abs(calculate_total_distance_traveled(positions) - 0.6) < 0.01

    def test_calculate_total_distance_diagonal(self):
        positions = [(0.0, 0.0), (0.3, 0.4)]
        assert abs(calculate_total_distance_traveled(positions) - 0.5) < 0.01


@pytest.fixture
def service_landmarks():
    """Landmarks suitable for service-level tests (with visibility)."""
    landmarks = []
    for i in range(33):
        landmarks.append({"x": 0.5, "y": 0.3 + i * 0.01, "z": 0.0, "visibility": 1.0})
    return landmarks


class TestMovementEconomy:
    """Tests for movement economy via EfficiencyCalculator."""

    def test_initial_economy_zero(self, service_landmarks):
        calc = EfficiencyCalculator()
        metrics = calc.calculate(service_landmarks)
        assert metrics["movement_economy"] == 0.0

    def test_efficient_climb(self, service_landmarks):
        service = ClimbingAnalysisService()
        sequence = []
        for i in range(10):
            lm = [dict(d) for d in service_landmarks]
            for d in lm:
                d["y"] -= i * 0.01
            sequence.append(lm)
        analysis = service.analyze(sequence, fps=30.0)
        # Efficient vertical movement
        assert analysis.summary.avg_movement_economy > 0.0

    def test_inefficient_climb(self, service_landmarks):
        service = ClimbingAnalysisService()
        sequence = []
        for i in range(10):
            lm = [dict(d) for d in service_landmarks]
            for d in lm:
                d["x"] += i * 0.02
                d["y"] -= i * 0.005
            sequence.append(lm)
        analysis = service.analyze(sequence, fps=30.0)
        # Less efficient due to lateral movement
        assert analysis.summary.avg_movement_economy < 0.5


class TestLockOffDetection:
    """Tests for lock-off detection via TechniqueCalculator."""

    def test_no_lock_off_generic_pose(self, service_landmarks):
        calc = TechniqueCalculator()
        metrics = calc.calculate(service_landmarks)
        assert "is_lock_off" in metrics

    def test_lock_off_with_bent_elbow(self, service_landmarks):
        from climb_sensei.domain.calculators import FrameContext

        ctx = FrameContext(
            com=(0.5, 0.5),
            hip_height=0.5,
            joint_angles={"left_elbow": 60.0, "right_elbow": 160.0},
        )
        calc = TechniqueCalculator()
        metrics = calc.calculate(service_landmarks, context=ctx)
        assert metrics["is_lock_off"] is True
        assert metrics["left_lock_off"] is True

    def test_lock_off_count_in_summary(self, service_landmarks):
        calc = TechniqueCalculator()
        for _ in range(10):
            calc.calculate(service_landmarks)
        summary = calc.get_summary()
        assert "total_lock_offs" in summary


class TestRestPositionDetection:
    """Tests for rest position detection via TechniqueCalculator."""

    def test_rest_with_straight_arms(self, service_landmarks):
        from climb_sensei.domain.calculators import FrameContext

        ctx = FrameContext(
            com=(0.5, 0.5),
            hip_height=0.5,
            joint_angles={"left_elbow": 170.0, "right_elbow": 165.0},
        )
        calc = TechniqueCalculator()
        metrics = calc.calculate(service_landmarks, context=ctx)
        assert metrics["is_rest_position"] is True

    def test_rest_count_in_summary(self, service_landmarks):
        calc = TechniqueCalculator()
        for _ in range(10):
            calc.calculate(service_landmarks)
        summary = calc.get_summary()
        assert "total_rest_positions" in summary


class TestFatigueScore:
    """Tests for fatigue score via FatigueCalculator."""

    def test_insufficient_data_returns_zero(self):
        calc = FatigueCalculator(min_frames=90)
        for _ in range(20):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        assert calc.get_summary()["fatigue_score"] == 0.0

    def test_consistent_quality_low_fatigue(self):
        calc = FatigueCalculator(min_frames=90)
        for _ in range(100):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        assert calc.get_summary()["fatigue_score"] == pytest.approx(0.0, abs=0.01)

    def test_degrading_quality_positive_fatigue(self):
        calc = FatigueCalculator(min_frames=90)
        for _ in range(60):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        for _ in range(60):
            calc.record_stability_metrics(jerk=3.0, sway=0.3)
        assert calc.get_summary()["fatigue_score"] > 0.0


class TestJointAngleHistory:
    """Tests for joint angle tracking via JointAngleCalculator."""

    def test_joint_angles_in_frame_metrics(self, service_landmarks):
        calc = JointAngleCalculator()
        metrics = calc.calculate(service_landmarks)
        for joint in [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]:
            assert joint in metrics

    def test_joint_angles_in_history(self, service_landmarks):
        calc = JointAngleCalculator()
        for _ in range(10):
            calc.calculate(service_landmarks)
        history = calc.get_history()
        for joint in ["left_elbow", "right_elbow"]:
            assert joint in history
            assert len(history[joint]) == 10

    def test_joint_angles_in_summary(self, service_landmarks):
        calc = JointAngleCalculator()
        for _ in range(10):
            calc.calculate(service_landmarks)
        summary = calc.get_summary()
        for joint in ["left_elbow", "right_elbow"]:
            assert f"avg_{joint}" in summary


class TestResetBehavior:
    """Tests for calculator reset functionality."""

    def test_reset_clears_stability_history(self, service_landmarks):
        calc = StabilityCalculator()
        for _ in range(5):
            calc.calculate(service_landmarks)
        calc.reset()
        assert calc.total_frames == 0
        assert calc.get_history() == {}

    def test_reset_clears_efficiency_state(self, service_landmarks):
        calc = EfficiencyCalculator()
        for _ in range(5):
            calc.calculate(service_landmarks)
        calc.reset()
        assert calc._total_distance == 0.0
        assert calc._initial_hip_height is None

    def test_reset_clears_technique_counts(self, service_landmarks):
        calc = TechniqueCalculator()
        for _ in range(5):
            calc.calculate(service_landmarks)
        calc.reset()
        assert calc._total_lock_offs == 0
        assert calc._total_rest_positions == 0

    def test_reset_clears_fatigue_history(self):
        calc = FatigueCalculator()
        for _ in range(50):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        calc.reset()
        assert calc._jerk_history == []
        assert calc._sway_history == []
