"""Integration tests for the new service architecture.

These tests demonstrate:
1. Service independence
2. Composability
3. Isolation and testability
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.services import (
    VideoQualityService,
    TrackingQualityService,
    ClimbingAnalysisService,
)
from climb_sensei.domain.calculators import (
    StabilityCalculator,
    ProgressCalculator,
)


class TestVideoQualityService:
    """Test video quality service independence."""

    def test_service_creation_no_dependencies(self):
        """Video quality service should have no external dependencies."""
        service = VideoQualityService()
        assert service is not None
        assert not service.default_deep_check

    def test_service_configuration(self):
        """Service should be configurable."""
        service = VideoQualityService(default_deep_check=True)
        assert service.default_deep_check


class TestTrackingQualityService:
    """Test tracking quality service independence."""

    def test_analyze_from_landmarks_no_video_needed(self):
        """Tracking quality should work without video file."""
        service = TrackingQualityService()

        # Create dummy landmarks sequence
        dummy_landmarks = [
            [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33 for _ in range(100)
        ]

        # Should work without video file
        report = service.analyze_from_landmarks(dummy_landmarks)

        assert report.total_frames == 100
        assert report.frames_with_pose == 100
        assert report.detection_rate == 100.0

    def test_service_configuration(self):
        """Service should accept configuration."""
        service = TrackingQualityService(
            min_detection_rate=80.0,
            min_smoothness=0.7,
        )
        assert service.min_detection_rate == 80.0
        assert service.min_smoothness == 0.7


class TestClimbingAnalysisService:
    """Test climbing analysis service with calculator composition."""

    def test_default_calculators(self):
        """Service should have default calculators."""
        service = ClimbingAnalysisService()
        assert len(service.calculators) == 6  # 6 default calculators

    def test_custom_calculators(self):
        """Service should accept custom calculators."""
        custom_calcs = [
            StabilityCalculator(),
            ProgressCalculator(),
        ]
        service = ClimbingAnalysisService(calculators=custom_calcs)

        assert len(service.calculators) == 2
        metrics = service.get_available_metrics()

        # Should only have metrics from these two calculators
        assert "com_velocity" in metrics  # From stability
        assert "hip_height" in metrics  # From progress

    def test_add_calculator_dynamically(self):
        """Should be able to add calculators after creation."""
        service = ClimbingAnalysisService(calculators=[])
        assert len(service.calculators) == 0

        service.add_calculator(StabilityCalculator())
        assert len(service.calculators) == 1

    def test_remove_calculator(self):
        """Should be able to remove calculators."""
        service = ClimbingAnalysisService()
        initial_count = len(service.calculators)

        service.remove_calculator(StabilityCalculator)
        assert len(service.calculators) == initial_count - 1

    def test_analyze_with_landmarks(self):
        """Service should analyze landmarks without video."""
        service = ClimbingAnalysisService()

        # Create dummy landmarks sequence - decrease y to simulate climbing UP
        # (in image coords, y increases downward, so climbing up means y decreases)
        dummy_landmarks = [
            [
                {"x": 0.5, "y": 0.5 - i * 0.01, "z": 0.0, "visibility": 0.9}
                for _ in range(33)
            ]
            for i in range(50)
        ]

        analysis = service.analyze(dummy_landmarks, fps=30.0)

        assert analysis.summary.total_frames == 50
        assert analysis.summary.total_vertical_progress > 0
        assert len(analysis.history) > 0


class TestServiceComposition:
    """Test that services can be composed without coupling."""

    def test_services_are_independent(self):
        """Each service should be instantiable independently."""
        vq = VideoQualityService()
        tq = TrackingQualityService()
        ca = ClimbingAnalysisService()

        # All services should exist independently
        assert vq is not None
        assert tq is not None
        assert ca is not None

    def test_climbing_analysis_without_quality_checks(self):
        """Climbing analysis should work without quality services."""
        service = ClimbingAnalysisService()

        # Should work with just landmarks, no quality checks
        dummy_landmarks = [
            [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33 for _ in range(30)
        ]

        analysis = service.analyze(dummy_landmarks)
        assert analysis.summary.total_frames == 30

    def test_tracking_quality_without_climbing_analysis(self):
        """Tracking quality should work independently."""
        service = TrackingQualityService()

        dummy_landmarks = [
            [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33 for _ in range(30)
        ]

        report = service.analyze_from_landmarks(dummy_landmarks)
        assert report.detection_rate == 100.0


class TestCalculatorPlugin:
    """Test the calculator plugin system."""

    def test_stability_calculator_standalone(self):
        """Stability calculator should work independently."""
        calc = StabilityCalculator(window_size=30, fps=30.0)

        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33

        metrics = calc.calculate(landmarks)
        assert "com_x" in metrics
        assert "com_y" in metrics
        assert "com_velocity" in metrics

    def test_progress_calculator_standalone(self):
        """Progress calculator should work independently."""
        calc = ProgressCalculator(fps=30.0)

        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33

        metrics = calc.calculate(landmarks)
        assert "hip_height" in metrics
        assert "vertical_progress" in metrics

    def test_calculator_state_management(self):
        """Calculators should manage their own state."""
        calc = StabilityCalculator()

        # First frame
        landmarks1 = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33
        calc.calculate(landmarks1)

        # Second frame
        landmarks2 = [{"x": 0.5, "y": 0.6, "z": 0.0, "visibility": 0.9}] * 33
        metrics = calc.calculate(landmarks2)

        # Velocity should be calculated (requires 2 frames)
        assert metrics["com_velocity"] > 0

    def test_calculator_reset(self):
        """Calculators should reset cleanly."""
        calc = StabilityCalculator()

        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9}] * 33
        calc.calculate(landmarks)
        calc.calculate(landmarks)

        assert calc.total_frames == 2

        calc.reset()
        assert calc.total_frames == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
