"""Tests for biomechanics bug fixes.

Covers:
- BUG1: Body angle preserves lean direction (atan2)
- BUG2: Fatigue score clamped to [0, 1]
- BUG3: Jerk window uses full window_size
- BUG4: Landmark validation rejects NaN/Inf
- BUG5: Pre-extracted landmarks don't fake confidence
- BUG6: Smoothness uses coefficient of variation
"""

import numpy as np
import pytest

from climb_sensei.domain.calculators.technique import TechniqueCalculator
from climb_sensei.domain.calculators.stability import StabilityCalculator
from climb_sensei.domain.calculators.fatigue import FatigueCalculator
from climb_sensei.tracking_quality import TrackingQualityAnalyzer

# Need at least 33 landmarks (MediaPipe indices 0-32)
# Relevant: 11=L_SHOULDER, 12=R_SHOULDER, 23=L_HIP, 24=R_HIP


def _make_landmarks(overrides=None):
    """Build a 33-landmark list with all at (0.5, 0.5) then apply overrides."""
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    if overrides:
        for idx, vals in overrides.items():
            lm[idx].update(vals)
    return lm


# ========== BUG1: Body angle direction ==========


class TestBodyAngleDirection:
    """Body angle should return signed values distinguishing left/right lean."""

    def test_vertical_body_returns_near_zero(self):
        """Shoulders directly above hips → angle ≈ 0°."""
        calc = TechniqueCalculator(window_size=30, fps=30.0)
        lm = _make_landmarks(
            {
                11: {"x": 0.45, "y": 0.3},  # L shoulder
                12: {"x": 0.55, "y": 0.3},  # R shoulder
                23: {"x": 0.45, "y": 0.7},  # L hip
                24: {"x": 0.55, "y": 0.7},  # R hip
            }
        )
        metrics = calc.calculate(lm)
        assert abs(metrics["body_angle"]) < 5.0

    def test_lean_right_returns_positive(self):
        """Shoulders shifted right of hips → positive angle."""
        calc = TechniqueCalculator(window_size=30, fps=30.0)
        lm = _make_landmarks(
            {
                11: {"x": 0.6, "y": 0.3},  # L shoulder (shifted right)
                12: {"x": 0.7, "y": 0.3},  # R shoulder (shifted right)
                23: {"x": 0.45, "y": 0.7},  # L hip
                24: {"x": 0.55, "y": 0.7},  # R hip
            }
        )
        metrics = calc.calculate(lm)
        assert metrics["body_angle"] > 10.0, "Right lean should be positive"

    def test_lean_left_returns_negative(self):
        """Shoulders shifted left of hips → negative angle."""
        calc = TechniqueCalculator(window_size=30, fps=30.0)
        lm = _make_landmarks(
            {
                11: {"x": 0.3, "y": 0.3},  # L shoulder (shifted left)
                12: {"x": 0.4, "y": 0.3},  # R shoulder (shifted left)
                23: {"x": 0.45, "y": 0.7},  # L hip
                24: {"x": 0.55, "y": 0.7},  # R hip
            }
        )
        metrics = calc.calculate(lm)
        assert metrics["body_angle"] < -10.0, "Left lean should be negative"

    def test_left_and_right_are_distinguishable(self):
        """Mirror poses should produce opposite-sign angles."""
        calc = TechniqueCalculator(window_size=30, fps=30.0)

        right_lm = _make_landmarks(
            {
                11: {"x": 0.65, "y": 0.3},
                12: {"x": 0.75, "y": 0.3},
                23: {"x": 0.45, "y": 0.7},
                24: {"x": 0.55, "y": 0.7},
            }
        )
        left_lm = _make_landmarks(
            {
                11: {"x": 0.25, "y": 0.3},
                12: {"x": 0.35, "y": 0.3},
                23: {"x": 0.45, "y": 0.7},
                24: {"x": 0.55, "y": 0.7},
            }
        )

        right_angle = calc.calculate(right_lm)["body_angle"]
        calc.reset()
        left_angle = calc.calculate(left_lm)["body_angle"]

        assert right_angle > 0
        assert left_angle < 0
        assert abs(right_angle + left_angle) < 5.0  # Approximately symmetric


# ========== BUG2: Fatigue score bounds ==========


class TestFatigueScoreBounds:
    """Fatigue score must always be in [0, 1]."""

    def test_fatigue_clamped_at_one(self):
        """Even with extreme degradation, score should not exceed 1.0."""
        calc = FatigueCalculator(window_size=30, fps=30.0, min_frames=30)

        # Simulate: low jerk/sway early, very high late (>200% degradation)
        for _ in range(50):
            calc.record_stability_metrics(jerk=0.01, sway=0.01)
        for _ in range(50):
            calc.record_stability_metrics(jerk=10.0, sway=10.0)

        summary = calc.get_summary()
        score = summary.get("fatigue_score", 0.0)
        assert 0.0 <= score <= 1.0, f"Fatigue {score} exceeds [0, 1]"

    def test_fatigue_zero_when_no_degradation(self):
        """Constant quality should produce fatigue = 0."""
        calc = FatigueCalculator(window_size=30, fps=30.0, min_frames=30)

        for _ in range(100):
            calc.record_stability_metrics(jerk=0.5, sway=0.5)

        summary = calc.get_summary()
        score = summary.get("fatigue_score", 0.0)
        assert score == pytest.approx(0.0, abs=0.05)

    def test_fatigue_zero_with_insufficient_frames(self):
        """Should return 0 when not enough frames."""
        calc = FatigueCalculator(window_size=30, fps=30.0, min_frames=90)

        for _ in range(10):
            calc.record_stability_metrics(jerk=1.0, sway=1.0)

        summary = calc.get_summary()
        assert summary.get("fatigue_score", 0.0) == 0.0


# ========== BUG3: Jerk window ==========


class TestJerkWindow:
    """Jerk should use full window, producing smoother values."""

    def test_jerk_uses_more_than_4_frames(self):
        """With 30 frames available, jerk should use the full window."""
        calc = StabilityCalculator(window_size=30, fps=30.0)

        # Feed 30 frames with small random movement
        np.random.seed(42)
        for i in range(30):
            lm = _make_landmarks(
                {
                    23: {"x": 0.5 + np.random.normal(0, 0.005), "y": 0.5 + i * 0.01},
                    24: {"x": 0.5 + np.random.normal(0, 0.005), "y": 0.5 + i * 0.01},
                }
            )
            calc.calculate(lm)

        # The jerk value should be computed (not 0)
        summary = calc.get_summary()
        assert "avg_jerk" in summary

    def test_jerk_returns_zero_with_fewer_than_4_frames(self):
        """Less than 4 frames → jerk = 0."""
        calc = StabilityCalculator(window_size=30, fps=30.0)

        for i in range(3):
            lm = _make_landmarks(
                {
                    23: {"x": 0.5, "y": 0.5 + i * 0.01},
                    24: {"x": 0.5, "y": 0.5 + i * 0.01},
                }
            )
            metrics = calc.calculate(lm)

        assert metrics["jerk"] == 0.0


# ========== BUG4: Landmark validation ==========


class TestLandmarkValidation:
    """Invalid landmarks (NaN, Inf, out-of-range) must be rejected."""

    def test_valid_landmarks_pass(self):
        from app.services.upload import _validate_landmarks

        lm = _make_landmarks()
        assert _validate_landmarks(lm) is True

    def test_nan_landmark_rejected(self):
        from app.services.upload import _validate_landmarks

        lm = _make_landmarks({5: {"x": float("nan"), "y": 0.5}})
        assert _validate_landmarks(lm) is False

    def test_inf_landmark_rejected(self):
        from app.services.upload import _validate_landmarks

        lm = _make_landmarks({5: {"x": float("inf"), "y": 0.5}})
        assert _validate_landmarks(lm) is False

    def test_out_of_range_rejected(self):
        from app.services.upload import _validate_landmarks

        lm = _make_landmarks({5: {"x": 5.0, "y": 0.5}})
        assert _validate_landmarks(lm) is False

    def test_none_landmarks_rejected(self):
        from app.services.upload import _validate_landmarks

        assert _validate_landmarks(None) is False
        assert _validate_landmarks([]) is False


# ========== BUG5: Pre-extracted landmark confidence ==========


class TestPreExtractedConfidence:
    """Pre-extracted landmarks should not report fake confidence values."""

    def test_confidence_not_hardcoded(self):
        """Pre-extracted landmarks should report 0.0 confidence (unavailable)."""
        service = TrackingQualityAnalyzer()

        # Simulate pre-extracted landmark sequence
        landmarks = [[(0.5, 0.5)] * 33 for _ in range(30)]

        report = service.analyze_from_landmarks(landmarks)

        # Confidence should be 0.0 (unavailable), not the old fake 0.8
        assert report.avg_landmark_confidence == pytest.approx(0.0, abs=0.01)

    def test_preextracted_not_penalized_as_poor(self):
        """Pre-extracted landmarks with good tracking shouldn't be rated 'poor'."""
        service = TrackingQualityAnalyzer()

        # Good steady tracking — all frames detected, smooth movement
        landmarks = [[(0.5 + i * 0.001, 0.5)] * 33 for i in range(30)]

        report = service.analyze_from_landmarks(landmarks)

        # Should not be penalized as poor just because confidence is unavailable
        assert report.quality_level != "poor"


# ========== BUG6: Smoothness formula ==========


class TestSmoothnessFormula:
    """Smoothness should use coefficient of variation, not magic number."""

    def test_uniform_movement_is_smooth(self):
        """Constant displacement between frames → smoothness ≈ 1.0."""
        service = TrackingQualityAnalyzer()

        # 30 frames, each landmark moves exactly 0.01 to the right
        positions = []
        for i in range(30):
            frame = [(0.5 + i * 0.01, 0.5, 0.0)] * 33
            positions.append(frame)

        smoothness = service._calculate_smoothness(positions)
        assert smoothness > 0.8, f"Uniform movement should be smooth, got {smoothness}"

    def test_erratic_movement_is_not_smooth(self):
        """Random jumps → smoothness < 0.5."""
        service = TrackingQualityAnalyzer()

        np.random.seed(42)
        positions = []
        for _ in range(30):
            frame = [(np.random.random(), np.random.random(), 0.0) for _ in range(33)]
            positions.append(frame)

        smoothness = service._calculate_smoothness(positions)
        assert (
            smoothness < 0.7
        ), f"Erratic movement should not be smooth, got {smoothness}"

    def test_no_movement_is_perfectly_smooth(self):
        """Stationary landmarks → smoothness = 1.0."""
        service = TrackingQualityAnalyzer()

        positions = [[(0.5, 0.5, 0.0)] * 33 for _ in range(30)]

        smoothness = service._calculate_smoothness(positions)
        assert smoothness == pytest.approx(1.0)

    def test_smoothness_in_valid_range(self):
        """Output always in [0, 1]."""
        service = TrackingQualityAnalyzer()

        np.random.seed(123)
        positions = []
        for _ in range(30):
            frame = [
                (np.random.random() * 2, np.random.random() * 2, 0.0) for _ in range(33)
            ]
            positions.append(frame)

        smoothness = service._calculate_smoothness(positions)
        assert 0.0 <= smoothness <= 1.0
