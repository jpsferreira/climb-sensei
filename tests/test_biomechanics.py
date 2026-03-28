"""Unit tests for climb-sensei biomechanics module."""

import math
import pytest
from climb_sensei.biomechanics import (
    calculate_joint_angle,
    calculate_joint_angles_batch,
    calculate_reach_distance,
    calculate_center_of_mass,
)


class TestCalculateJointAngle:
    """Test suite for calculate_joint_angle function."""

    def test_right_angle(self) -> None:
        """Test calculation of a 90-degree angle."""
        point_a = (0.0, 1.0)
        point_b = (0.0, 0.0)
        point_c = (1.0, 0.0)

        angle = calculate_joint_angle(point_a, point_b, point_c)

        assert math.isclose(angle, 90.0, abs_tol=0.1)

    def test_straight_angle(self) -> None:
        """Test calculation of a 180-degree angle (straight line)."""
        point_a = (0.0, 0.0)
        point_b = (0.5, 0.0)
        point_c = (1.0, 0.0)

        angle = calculate_joint_angle(point_a, point_b, point_c)

        assert math.isclose(angle, 180.0, abs_tol=0.1)

    def test_acute_angle(self) -> None:
        """Test calculation of an acute angle (less than 90 degrees)."""
        point_a = (0.0, 1.0)
        point_b = (0.0, 0.0)
        point_c = (1.0, 1.0)

        angle = calculate_joint_angle(point_a, point_b, point_c)

        # This should be 45 degrees
        assert 44.0 < angle < 46.0

    def test_zero_length_vector(self) -> None:
        """Test with coincident points (zero-length vector)."""
        point_a = (0.5, 0.5)
        point_b = (0.5, 0.5)  # Same as point_a
        point_c = (1.0, 1.0)

        angle = calculate_joint_angle(point_a, point_b, point_c)

        assert angle == 0.0


class TestCalculateReachDistance:
    """Test suite for calculate_reach_distance function."""

    def test_horizontal_distance(self) -> None:
        """Test distance calculation for horizontal points."""
        point_a = (0.0, 0.5)
        point_b = (1.0, 0.5)

        distance = calculate_reach_distance(point_a, point_b)

        assert math.isclose(distance, 1.0, abs_tol=0.001)

    def test_vertical_distance(self) -> None:
        """Test distance calculation for vertical points."""
        point_a = (0.5, 0.0)
        point_b = (0.5, 1.0)

        distance = calculate_reach_distance(point_a, point_b)

        assert math.isclose(distance, 1.0, abs_tol=0.001)

    def test_diagonal_distance(self) -> None:
        """Test distance calculation for diagonal points."""
        point_a = (0.0, 0.0)
        point_b = (1.0, 1.0)

        distance = calculate_reach_distance(point_a, point_b)

        # Distance should be sqrt(2) ≈ 1.414
        assert math.isclose(distance, math.sqrt(2), abs_tol=0.001)

    def test_same_point(self) -> None:
        """Test distance calculation for identical points."""
        point_a = (0.5, 0.5)
        point_b = (0.5, 0.5)

        distance = calculate_reach_distance(point_a, point_b)

        assert math.isclose(distance, 0.0, abs_tol=0.001)


class TestCalculateCenterOfMass:
    """Test suite for calculate_center_of_mass function."""

    def test_equal_weights(self) -> None:
        """Test center of mass with equal weights."""
        points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        center = calculate_center_of_mass(points)

        # Center should be at (0.5, 0.333...)
        assert math.isclose(center[0], 0.5, abs_tol=0.01)
        assert math.isclose(center[1], 1.0 / 3.0, abs_tol=0.01)

    def test_weighted_center(self) -> None:
        """Test center of mass with custom weights."""
        points = [(0.0, 0.0), (1.0, 0.0)]
        weights = [1.0, 3.0]  # Second point has 3x weight

        center = calculate_center_of_mass(points, weights)

        # Center should be closer to second point: (0.75, 0.0)
        assert math.isclose(center[0], 0.75, abs_tol=0.01)
        assert math.isclose(center[1], 0.0, abs_tol=0.01)

    def test_single_point(self) -> None:
        """Test center of mass with a single point."""
        points = [(0.5, 0.5)]

        center = calculate_center_of_mass(points)

        assert math.isclose(center[0], 0.5, abs_tol=0.001)
        assert math.isclose(center[1], 0.5, abs_tol=0.001)

    def test_empty_points_raises_error(self) -> None:
        """Test that empty points list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_center_of_mass([])

    def test_mismatched_weights_raises_error(self) -> None:
        """Test that mismatched weights length raises ValueError."""
        points = [(0.0, 0.0), (1.0, 1.0)]
        weights = [1.0]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            calculate_center_of_mass(points, weights)

    def test_zero_total_weight_raises_error(self) -> None:
        """Test that zero total weight raises ValueError."""
        points = [(0.0, 0.0), (1.0, 1.0)]
        weights = [0.0, 0.0]

        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_center_of_mass(points, weights)


class TestBatchJointAngles:
    """Tests for vectorized batch joint angle calculation."""

    def _lm(self, coords):
        """Build landmark list from (x, y) tuples."""
        return [{"x": x, "y": y} for x, y in coords]

    def test_batch_matches_scalar(self):
        """Batch results should match individual calculate_joint_angle calls."""
        # 5 landmarks: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1), 4=(0.5,0.5)
        landmarks = self._lm([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
        triplets = [(0, 1, 2), (1, 2, 3), (0, 4, 2)]

        batch = calculate_joint_angles_batch(landmarks, triplets)

        for i, (a, b, c) in enumerate(triplets):
            scalar = calculate_joint_angle(
                (landmarks[a]["x"], landmarks[a]["y"]),
                (landmarks[b]["x"], landmarks[b]["y"]),
                (landmarks[c]["x"], landmarks[c]["y"]),
            )
            assert batch[i] == pytest.approx(scalar, abs=1e-6)

    def test_batch_empty_triplets(self):
        """Empty triplet list should return empty list."""
        landmarks = self._lm([(0, 0), (1, 1)])
        assert calculate_joint_angles_batch(landmarks, []) == []

    def test_batch_zero_length_vector(self):
        """Zero-length vector should return 0.0 (matching scalar behavior)."""
        # Points A and B are the same → BA has zero length
        landmarks = self._lm([(0.5, 0.5), (0.5, 0.5), (1.0, 1.0)])
        batch = calculate_joint_angles_batch(landmarks, [(0, 1, 2)])
        assert batch[0] == pytest.approx(0.0)

    def test_batch_right_angle(self):
        """90-degree angle should return ~90."""
        landmarks = self._lm([(0, 0), (0, 1), (1, 1)])
        batch = calculate_joint_angles_batch(landmarks, [(0, 1, 2)])
        assert batch[0] == pytest.approx(90.0, abs=0.1)

    def test_batch_straight_angle(self):
        """180-degree (collinear) should return ~180."""
        landmarks = self._lm([(0, 0), (0.5, 0), (1, 0)])
        batch = calculate_joint_angles_batch(landmarks, [(0, 1, 2)])
        assert batch[0] == pytest.approx(180.0, abs=0.1)
