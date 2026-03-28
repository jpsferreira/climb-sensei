"""Tests for the calculator-based metrics system.

Tests the individual calculators and the ClimbingAnalysisService
that replaced the legacy ClimbingAnalyzer.
"""

import pytest

from climb_sensei.services import ClimbingAnalysisService
from climb_sensei.domain.calculators import (
    StabilityCalculator,
    ProgressCalculator,
    EfficiencyCalculator,
    TechniqueCalculator,
    JointAngleCalculator,
    FatigueCalculator,
    FrameContext,
)


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
                "visibility": 1.0,
            }
        )

    # Set specific landmarks for testing
    landmarks[11] = {"x": 0.4, "y": 0.4, "z": 0.0, "visibility": 1.0}  # LEFT_SHOULDER
    landmarks[12] = {"x": 0.6, "y": 0.4, "z": 0.0, "visibility": 1.0}  # RIGHT_SHOULDER
    landmarks[13] = {"x": 0.35, "y": 0.45, "z": 0.0, "visibility": 1.0}  # LEFT_ELBOW
    landmarks[14] = {"x": 0.65, "y": 0.45, "z": 0.0, "visibility": 1.0}  # RIGHT_ELBOW
    landmarks[15] = {"x": 0.3, "y": 0.3, "z": 0.0, "visibility": 1.0}  # LEFT_WRIST
    landmarks[16] = {"x": 0.7, "y": 0.3, "z": 0.0, "visibility": 1.0}  # RIGHT_WRIST
    landmarks[23] = {"x": 0.4, "y": 0.6, "z": 0.0, "visibility": 1.0}  # LEFT_HIP
    landmarks[24] = {"x": 0.6, "y": 0.6, "z": 0.0, "visibility": 1.0}  # RIGHT_HIP
    landmarks[25] = {"x": 0.4, "y": 0.7, "z": 0.0, "visibility": 1.0}  # LEFT_KNEE
    landmarks[26] = {"x": 0.6, "y": 0.7, "z": 0.0, "visibility": 1.0}  # RIGHT_KNEE
    landmarks[27] = {"x": 0.4, "y": 0.8, "z": 0.0, "visibility": 1.0}  # LEFT_ANKLE
    landmarks[28] = {"x": 0.6, "y": 0.8, "z": 0.0, "visibility": 1.0}  # RIGHT_ANKLE

    return landmarks


class TestStabilityCalculator:
    """Tests for StabilityCalculator."""

    def test_first_frame_zero_velocity(self, sample_landmarks):
        calc = StabilityCalculator(window_size=30, fps=30.0)
        metrics = calc.calculate(sample_landmarks)
        assert metrics["com_velocity"] == 0.0

    def test_velocity_with_movement(self, sample_landmarks):
        calc = StabilityCalculator(window_size=30, fps=30.0)
        calc.calculate(sample_landmarks)

        moved = [dict(lm) for lm in sample_landmarks]
        for lm in moved:
            lm["x"] += 0.1
            lm["y"] += 0.1
        metrics = calc.calculate(moved)
        assert metrics["com_velocity"] > 0.0

    def test_sway_requires_three_frames(self, sample_landmarks):
        calc = StabilityCalculator(window_size=30, fps=30.0)
        calc.calculate(sample_landmarks)
        m2 = calc.calculate(sample_landmarks)
        assert m2["com_sway"] == 0.0

        moved = [dict(lm) for lm in sample_landmarks]
        for lm in moved:
            lm["x"] += 0.2
        m3 = calc.calculate(moved)
        assert m3["com_sway"] > 0.0

    def test_empty_landmarks(self):
        calc = StabilityCalculator()
        assert calc.calculate([]) == {}

    def test_uses_context_com(self, sample_landmarks):
        calc = StabilityCalculator()
        ctx = FrameContext(com=(0.5, 0.5), hip_height=0.6, joint_angles={})
        metrics = calc.calculate(sample_landmarks, context=ctx)
        assert metrics["com_x"] == 0.5
        assert metrics["com_y"] == 0.5

    def test_summary_after_frames(self, sample_landmarks):
        calc = StabilityCalculator(window_size=30, fps=30.0)
        for _ in range(5):
            calc.calculate(sample_landmarks)
        summary = calc.get_summary()
        assert "avg_com_velocity" in summary
        assert "avg_com_sway" in summary


class TestProgressCalculator:
    """Tests for ProgressCalculator."""

    def test_initial_progress_zero(self, sample_landmarks):
        calc = ProgressCalculator()
        metrics = calc.calculate(sample_landmarks)
        assert metrics["vertical_progress"] == 0.0

    def test_upward_movement(self, sample_landmarks):
        calc = ProgressCalculator()
        calc.calculate(sample_landmarks)

        moved = [dict(lm) for lm in sample_landmarks]
        moved[23]["y"] -= 0.1  # LEFT_HIP up
        moved[24]["y"] -= 0.1  # RIGHT_HIP up
        metrics = calc.calculate(moved)
        assert metrics["vertical_progress"] > 0.0

    def test_uses_context_hip_height(self, sample_landmarks):
        calc = ProgressCalculator()
        ctx = FrameContext(com=(0.5, 0.5), hip_height=0.6, joint_angles={})
        metrics = calc.calculate(sample_landmarks, context=ctx)
        assert metrics["hip_height"] == 0.6

    def test_summary_tracks_heights(self, sample_landmarks):
        calc = ProgressCalculator()
        for _ in range(5):
            calc.calculate(sample_landmarks)
        summary = calc.get_summary()
        assert "total_vertical_progress" in summary
        assert "max_height" in summary


class TestEfficiencyCalculator:
    """Tests for EfficiencyCalculator."""

    def test_initial_economy_zero(self, sample_landmarks):
        calc = EfficiencyCalculator()
        metrics = calc.calculate(sample_landmarks)
        assert metrics["movement_economy"] == 0.0

    def test_economy_with_vertical_movement(self, sample_landmarks):
        calc = EfficiencyCalculator()
        calc.calculate(sample_landmarks)

        # Move straight up (lower y = up in image coords)
        moved = [dict(lm) for lm in sample_landmarks]
        for lm in moved:
            lm["y"] -= 0.1
        metrics = calc.calculate(moved)
        # Economy should be positive when moving upward
        assert metrics["total_distance"] > 0.0

    def test_uses_context(self, sample_landmarks):
        calc = EfficiencyCalculator()
        ctx = FrameContext(com=(0.5, 0.5), hip_height=0.6, joint_angles={})
        metrics = calc.calculate(sample_landmarks, context=ctx)
        assert "movement_economy" in metrics


class TestTechniqueCalculator:
    """Tests for TechniqueCalculator."""

    def test_body_angle(self, sample_landmarks):
        calc = TechniqueCalculator()
        metrics = calc.calculate(sample_landmarks)
        assert "body_angle" in metrics
        assert -90 <= metrics["body_angle"] <= 90

    def test_hand_foot_spans(self, sample_landmarks):
        calc = TechniqueCalculator()
        metrics = calc.calculate(sample_landmarks)
        assert metrics["hand_span"] > 0.0
        assert metrics["foot_span"] > 0.0

    def test_lock_off_with_bent_elbow(self):
        """Verify lock-off detected when elbow < 90 degrees."""
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}] * 33

        # Create bent left elbow (angle < 90)
        ctx = FrameContext(
            com=(0.5, 0.5),
            hip_height=0.6,
            joint_angles={
                "left_elbow": 60.0,  # < 90 threshold
                "right_elbow": 160.0,
            },
        )
        calc = TechniqueCalculator()
        metrics = calc.calculate(landmarks, context=ctx)
        assert metrics["is_lock_off"] is True
        assert metrics["left_lock_off"] is True

    def test_rest_position_with_straight_arms(self):
        """Verify rest position detected when both elbows > 150 degrees."""
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}] * 33

        ctx = FrameContext(
            com=(0.5, 0.5),
            hip_height=0.6,
            joint_angles={
                "left_elbow": 165.0,  # > 150 threshold
                "right_elbow": 170.0,
            },
        )
        calc = TechniqueCalculator()
        metrics = calc.calculate(landmarks, context=ctx)
        assert metrics["is_rest_position"] is True

    def test_summary_counts(self, sample_landmarks):
        calc = TechniqueCalculator()
        for _ in range(5):
            calc.calculate(sample_landmarks)
        summary = calc.get_summary()
        assert "total_lock_offs" in summary
        assert "total_rest_positions" in summary


class TestJointAngleCalculator:
    """Tests for JointAngleCalculator."""

    def test_all_angles_present(self, sample_landmarks):
        calc = JointAngleCalculator()
        metrics = calc.calculate(sample_landmarks)
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

    def test_uses_context_angles(self, sample_landmarks):
        angles = {
            "left_elbow": 90.0,
            "right_elbow": 90.0,
            "left_shoulder": 120.0,
            "right_shoulder": 120.0,
            "left_knee": 170.0,
            "right_knee": 170.0,
            "left_hip": 160.0,
            "right_hip": 160.0,
        }
        ctx = FrameContext(com=(0.5, 0.5), hip_height=0.6, joint_angles=angles)
        calc = JointAngleCalculator()
        metrics = calc.calculate(sample_landmarks, context=ctx)
        assert metrics["left_elbow"] == 90.0

    def test_summary_has_min_max_avg(self, sample_landmarks):
        calc = JointAngleCalculator()
        for _ in range(5):
            calc.calculate(sample_landmarks)
        summary = calc.get_summary()
        assert "min_left_elbow" in summary
        assert "max_left_elbow" in summary
        assert "avg_left_elbow" in summary


class TestFatigueCalculator:
    """Tests for FatigueCalculator."""

    def test_insufficient_frames_returns_zero(self):
        calc = FatigueCalculator(min_frames=90)
        for _ in range(50):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        summary = calc.get_summary()
        assert summary["fatigue_score"] == 0.0

    def test_constant_quality_low_fatigue(self):
        calc = FatigueCalculator(min_frames=90)
        for _ in range(120):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        summary = calc.get_summary()
        assert summary["fatigue_score"] == pytest.approx(0.0, abs=0.01)

    def test_degrading_quality_high_fatigue(self):
        calc = FatigueCalculator(min_frames=90)
        # First 60 frames: low jerk/sway
        for _ in range(60):
            calc.record_stability_metrics(jerk=1.0, sway=0.1)
        # Last 60 frames: doubled jerk/sway
        for _ in range(60):
            calc.record_stability_metrics(jerk=2.0, sway=0.2)
        summary = calc.get_summary()
        assert summary["fatigue_score"] > 0.0

    def test_zero_early_values_no_division_error(self):
        calc = FatigueCalculator(min_frames=90)
        # First 60 frames: zero values
        for _ in range(60):
            calc.record_stability_metrics(jerk=0.0, sway=0.0)
        # Last 60 frames: nonzero
        for _ in range(60):
            calc.record_stability_metrics(jerk=2.0, sway=0.2)
        summary = calc.get_summary()
        assert summary["fatigue_score"] == 0.0  # Can't compute ratio from 0


class TestClimbingAnalysisService:
    """Tests for the end-to-end analysis service."""

    def test_analyze_basic(self, sample_landmarks):
        service = ClimbingAnalysisService()
        sequence = [sample_landmarks] * 10
        analysis = service.analyze(sequence, fps=30.0)

        assert analysis.summary.total_frames == 10
        assert "com_velocity" in analysis.history
        assert "hip_height" in analysis.history

    def test_analyze_with_none_frames(self, sample_landmarks):
        service = ClimbingAnalysisService()
        sequence = [sample_landmarks, None, sample_landmarks, None]
        analysis = service.analyze(sequence, fps=30.0)
        assert analysis.summary.total_frames == 2

    def test_analyze_empty_sequence(self):
        service = ClimbingAnalysisService()
        analysis = service.analyze([], fps=30.0)
        assert analysis.summary.total_frames == 0

    def test_summary_has_fatigue_score(self, sample_landmarks):
        service = ClimbingAnalysisService()
        sequence = [sample_landmarks] * 120
        analysis = service.analyze(sequence, fps=30.0)
        assert hasattr(analysis.summary, "fatigue_score")

    def test_lock_off_percentage_computed(self, sample_landmarks):
        service = ClimbingAnalysisService()
        sequence = [sample_landmarks] * 10
        analysis = service.analyze(sequence, fps=30.0)
        # Percentage should be a float (may be 0 if no lock-offs detected)
        assert isinstance(analysis.summary.lock_off_percentage, float)

    def test_custom_calculators(self, sample_landmarks):
        custom = [StabilityCalculator(window_size=10, fps=30.0)]
        service = ClimbingAnalysisService(calculators=custom)
        analysis = service.analyze([sample_landmarks] * 5, fps=30.0)
        assert "com_velocity" in analysis.history
        # No progress metrics since ProgressCalculator not included
        assert "hip_height" not in analysis.history
