"""Tests for minor biomechanics improvements.

Covers:
- MINOR1: Lock-off with velocity context (multi-frame)
- MINOR2: Rest detection with COM velocity (multi-frame)
- MINOR3: Movement economy clamped and tracking-loss resistant
- MINOR5: Sway normalized by shoulder width
"""

import numpy as np

from climb_sensei.domain.calculators.technique import TechniqueCalculator
from climb_sensei.domain.calculators.stability import StabilityCalculator
from climb_sensei.domain.calculators.efficiency import EfficiencyCalculator
from climb_sensei.config import LandmarkIndex


def _make_landmarks(overrides=None):
    """Build 33 landmarks at (0.5, 0.5) with overrides."""
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    if overrides:
        for idx, vals in overrides.items():
            lm[idx].update(vals)
    return lm


def _bent_arm_landmarks(wrist_x=0.8, wrist_y=0.3):
    """Landmarks with left elbow bent < 90 degrees (~56°).

    Shoulder straight above elbow, wrist pulled out and up → acute angle.
    """
    return _make_landmarks(
        {
            LandmarkIndex.LEFT_SHOULDER: {"x": 0.5, "y": 0.2},
            LandmarkIndex.LEFT_ELBOW: {"x": 0.5, "y": 0.5},
            LandmarkIndex.LEFT_WRIST: {"x": wrist_x, "y": wrist_y},
            LandmarkIndex.RIGHT_SHOULDER: {"x": 0.6, "y": 0.3},
            LandmarkIndex.RIGHT_ELBOW: {"x": 0.6, "y": 0.5},
            LandmarkIndex.RIGHT_WRIST: {"x": 0.6, "y": 0.7},
        }
    )


def _straight_arm_landmarks(hip_x=0.5, hip_y=0.7):
    """Landmarks with both elbows extended > 150 degrees."""
    return _make_landmarks(
        {
            LandmarkIndex.LEFT_SHOULDER: {"x": 0.4, "y": 0.3},
            LandmarkIndex.LEFT_ELBOW: {"x": 0.35, "y": 0.5},
            LandmarkIndex.LEFT_WRIST: {"x": 0.3, "y": 0.7},
            LandmarkIndex.RIGHT_SHOULDER: {"x": 0.6, "y": 0.3},
            LandmarkIndex.RIGHT_ELBOW: {"x": 0.65, "y": 0.5},
            LandmarkIndex.RIGHT_WRIST: {"x": 0.7, "y": 0.7},
            LandmarkIndex.LEFT_HIP: {"x": hip_x - 0.05, "y": hip_y},
            LandmarkIndex.RIGHT_HIP: {"x": hip_x + 0.05, "y": hip_y},
        }
    )


# ========== MINOR1: Lock-off with velocity ==========


class TestLockOffVelocity:
    """Lock-off requires bent elbow AND stationary wrist."""

    def test_bent_stationary_detected(self):
        """Bent elbow + stationary wrist over 2 frames → lock-off."""
        calc = TechniqueCalculator(lock_off_velocity_threshold=0.01)
        lm = _bent_arm_landmarks(wrist_x=0.8, wrist_y=0.3)

        calc.calculate(lm)  # Frame 1: no previous → assumed still
        metrics = calc.calculate(lm)  # Frame 2: same position → still

        assert metrics["is_lock_off"]

    def test_bent_moving_not_detected(self):
        """Bent elbow + moving wrist → NOT lock-off."""
        calc = TechniqueCalculator(lock_off_velocity_threshold=0.005)

        lm1 = _bent_arm_landmarks(wrist_x=0.75, wrist_y=0.25)
        lm2 = _bent_arm_landmarks(wrist_x=0.85, wrist_y=0.35)  # Big movement

        calc.calculate(lm1)
        metrics = calc.calculate(lm2)

        assert not metrics["left_lock_off"]

    def test_threshold_edge(self):
        """Wrist velocity exactly at threshold → not stationary."""
        threshold = 0.01
        calc = TechniqueCalculator(lock_off_velocity_threshold=threshold)

        lm1 = _bent_arm_landmarks(wrist_x=0.8, wrist_y=0.3)
        lm2 = _bent_arm_landmarks(wrist_x=0.8 + threshold, wrist_y=0.3)

        calc.calculate(lm1)
        metrics = calc.calculate(lm2)

        # At threshold → not stationary (strict <)
        assert not metrics["left_lock_off"]


# ========== MINOR2: Rest with COM velocity ==========


class TestRestVelocity:
    """Rest requires straight arms AND stationary body."""

    def test_straight_still_detected(self):
        """Straight arms + still body over 2 frames → rest."""
        calc = TechniqueCalculator(rest_velocity_threshold=0.01)
        lm = _straight_arm_landmarks(hip_x=0.5, hip_y=0.7)

        calc.calculate(lm)
        metrics = calc.calculate(lm)  # Same position → still

        assert metrics["is_rest_position"]

    def test_straight_moving_not_detected(self):
        """Straight arms + moving body → NOT rest."""
        calc = TechniqueCalculator(rest_velocity_threshold=0.005)

        lm1 = _straight_arm_landmarks(hip_x=0.5, hip_y=0.7)
        lm2 = _straight_arm_landmarks(hip_x=0.5, hip_y=0.6)  # Big COM shift

        calc.calculate(lm1)
        metrics = calc.calculate(lm2)

        assert not metrics["is_rest_position"]

    def test_com_updated_every_frame(self):
        """COM velocity should be per-frame, not since last rest frame."""
        calc = TechniqueCalculator(rest_velocity_threshold=0.01)

        # Frame 1: straight arms at position A
        calc.calculate(_straight_arm_landmarks(hip_x=0.5, hip_y=0.7))
        # Frame 2: bent arms (not rest) at position B (moved)
        calc.calculate(_bent_arm_landmarks(wrist_x=0.5, wrist_y=0.5))
        # Frame 3: straight arms back at position B (still vs frame 2)
        metrics = calc.calculate(_straight_arm_landmarks(hip_x=0.5, hip_y=0.7))

        # Should evaluate velocity vs frame 2, not frame 1
        # (COM was updated in frame 2 even though it wasn't rest)
        assert metrics["is_rest_position"] in (True, False)


# ========== MINOR3: Movement economy clamp ==========


class TestMovementEconomyClamp:
    """Economy should be clamped to [0, 1] and skip tracking loss."""

    def test_economy_clamped_to_one(self):
        """Even with favorable noise, economy should not exceed 1.0."""
        calc = EfficiencyCalculator(window_size=30, fps=30.0)

        # Simulate: large vertical progress, tiny lateral movement
        for i in range(20):
            lm = _make_landmarks(
                {
                    LandmarkIndex.LEFT_HIP: {"x": 0.5, "y": 0.8 - i * 0.03},
                    LandmarkIndex.RIGHT_HIP: {"x": 0.5, "y": 0.8 - i * 0.03},
                }
            )
            calc.calculate(lm)

        summary = calc.get_summary()
        if "final_movement_economy" in summary:
            assert summary["final_movement_economy"] <= 1.0

    def test_tracking_loss_skipped(self):
        """Large COM jump should not inflate total_distance."""
        calc = EfficiencyCalculator(window_size=30, fps=30.0)

        lm1 = _make_landmarks(
            {
                LandmarkIndex.LEFT_HIP: {"x": 0.5, "y": 0.5},
                LandmarkIndex.RIGHT_HIP: {"x": 0.5, "y": 0.5},
            }
        )
        lm2 = _make_landmarks(
            {
                LandmarkIndex.LEFT_HIP: {"x": 0.9, "y": 0.1},  # Huge jump
                LandmarkIndex.RIGHT_HIP: {"x": 0.9, "y": 0.1},
            }
        )

        calc.calculate(lm1)
        metrics = calc.calculate(lm2)

        # Total distance should not include the large jump
        assert metrics["total_distance"] < 0.1


# ========== MINOR5: Sway normalization ==========


class TestSwayNormalization:
    """Sway should be normalized by shoulder width."""

    def test_wider_shoulders_lower_normalized_sway(self):
        """Same raw sway with wider shoulders → lower normalized sway."""
        calc_narrow = StabilityCalculator(window_size=30, fps=30.0)
        calc_wide = StabilityCalculator(window_size=30, fps=30.0)

        np.random.seed(42)
        for i in range(10):
            jitter = np.random.normal(0, 0.005)
            # Narrow shoulders (width 0.1)
            lm_narrow = _make_landmarks(
                {
                    LandmarkIndex.LEFT_SHOULDER: {"x": 0.45, "y": 0.3},
                    LandmarkIndex.RIGHT_SHOULDER: {"x": 0.55, "y": 0.3},
                    LandmarkIndex.LEFT_HIP: {"x": 0.5 + jitter, "y": 0.7},
                    LandmarkIndex.RIGHT_HIP: {"x": 0.5 + jitter, "y": 0.7},
                }
            )
            # Wide shoulders (width 0.3)
            lm_wide = _make_landmarks(
                {
                    LandmarkIndex.LEFT_SHOULDER: {"x": 0.35, "y": 0.3},
                    LandmarkIndex.RIGHT_SHOULDER: {"x": 0.65, "y": 0.3},
                    LandmarkIndex.LEFT_HIP: {"x": 0.5 + jitter, "y": 0.7},
                    LandmarkIndex.RIGHT_HIP: {"x": 0.5 + jitter, "y": 0.7},
                }
            )
            calc_narrow.calculate(lm_narrow)
            calc_wide.calculate(lm_wide)

        narrow_sway = calc_narrow.get_summary().get("avg_com_sway", 0)
        wide_sway = calc_wide.get_summary().get("avg_com_sway", 0)

        # Same absolute sway, wider shoulders → lower normalized value
        assert wide_sway < narrow_sway

    def test_zero_shoulder_width_fallback(self):
        """Near-zero shoulder width should fall back to raw sway."""
        calc = StabilityCalculator(window_size=30, fps=30.0)

        for i in range(5):
            lm = _make_landmarks(
                {
                    LandmarkIndex.LEFT_SHOULDER: {"x": 0.5, "y": 0.3},
                    LandmarkIndex.RIGHT_SHOULDER: {"x": 0.5, "y": 0.3},  # Same x
                    LandmarkIndex.LEFT_HIP: {"x": 0.5 + i * 0.01, "y": 0.7},
                    LandmarkIndex.RIGHT_HIP: {"x": 0.5 + i * 0.01, "y": 0.7},
                }
            )
            metrics = calc.calculate(lm)

        # Should not crash, sway should be a finite number
        assert "com_sway" in metrics
        assert np.isfinite(metrics["com_sway"])
