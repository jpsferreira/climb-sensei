"""Tests for two-phase API (extract_landmarks + analyze_from_landmarks)."""

import pytest
from pathlib import Path

from climb_sensei import ClimbingSensei
from climb_sensei.models import ClimbingAnalysis


@pytest.fixture
def test_video():
    """Path to test video."""
    return Path(__file__).parent / "data" / "1.mp4"


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_extract_landmarks(test_video):
    """Test extract_landmarks returns expected structure."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        extracted = sensei.extract_landmarks(
            verbose=False, validate_video_quality=False
        )

    # Check returned structure
    assert "landmarks" in extracted
    assert "pose_results" in extracted
    assert "fps" in extracted
    assert "frame_count" in extracted

    # Check types
    assert isinstance(extracted["landmarks"], list)
    assert isinstance(extracted["pose_results"], list)
    assert isinstance(extracted["fps"], (int, float))
    assert isinstance(extracted["frame_count"], int)

    # Check lengths match
    assert len(extracted["landmarks"]) == len(extracted["pose_results"])
    assert extracted["frame_count"] <= len(extracted["landmarks"])


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_analyze_from_landmarks(test_video):
    """Test analyze_from_landmarks produces valid results."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        # Phase 1: Extract
        extracted = sensei.extract_landmarks(
            verbose=False, validate_video_quality=False
        )

        # Phase 2: Analyze
        analysis = sensei.analyze_from_landmarks(
            landmarks_sequence=extracted["landmarks"],
            fps=extracted["fps"],
            validate_tracking_quality=False,
            verbose=False,
        )

    # Check analysis type
    assert isinstance(analysis, ClimbingAnalysis)

    # Check has summary
    assert analysis.summary is not None
    assert hasattr(analysis.summary, "total_frames")
    assert hasattr(analysis.summary, "max_height")

    # Check has history
    assert analysis.history is not None
    assert isinstance(analysis.history, dict)


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_two_phase_matches_single_phase(test_video):
    """Test two-phase API produces same results as single-phase analyze()."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    # Single-phase approach
    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        analysis_single = sensei.analyze(verbose=False)

    # Two-phase approach
    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        extracted = sensei.extract_landmarks(
            verbose=False, validate_video_quality=False
        )
        analysis_two = sensei.analyze_from_landmarks(
            landmarks_sequence=extracted["landmarks"],
            fps=extracted["fps"],
            validate_tracking_quality=False,
            verbose=False,
        )

    # Compare results (should be identical)
    assert analysis_single.summary.total_frames == analysis_two.summary.total_frames
    assert (
        abs(analysis_single.summary.max_height - analysis_two.summary.max_height)
        < 0.001
    )
    assert (
        abs(analysis_single.summary.avg_velocity - analysis_two.summary.avg_velocity)
        < 0.001
    )

    # History should have same keys
    assert set(analysis_single.history.keys()) == set(analysis_two.history.keys())


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_extract_landmarks_with_quality_validation(test_video):
    """Test extract_landmarks with quality validation enabled."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    with ClimbingSensei(str(test_video), validate_quality=True) as sensei:
        extracted = sensei.extract_landmarks(verbose=False, validate_video_quality=True)

    # Should include video quality report
    assert "video_quality" in extracted
    if extracted["video_quality"] is not None:
        assert hasattr(extracted["video_quality"], "is_valid")
        assert hasattr(extracted["video_quality"], "resolution_quality")


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_analyze_from_landmarks_with_tracking_validation(test_video):
    """Test analyze_from_landmarks with tracking quality validation."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        extracted = sensei.extract_landmarks(
            verbose=False, validate_video_quality=False
        )
        analysis = sensei.analyze_from_landmarks(
            landmarks_sequence=extracted["landmarks"],
            fps=extracted["fps"],
            validate_tracking_quality=True,
            verbose=False,
        )

    # Should include tracking quality report
    assert analysis.tracking_quality is not None
    assert hasattr(analysis.tracking_quality, "is_trackable")
    assert hasattr(analysis.tracking_quality, "detection_rate")


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_reuse_extracted_landmarks_multiple_times(test_video):
    """Test that extracted landmarks can be analyzed multiple times."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        # Extract once
        extracted = sensei.extract_landmarks(
            verbose=False, validate_video_quality=False
        )

        # Analyze multiple times with different settings
        analysis1 = sensei.analyze_from_landmarks(
            landmarks_sequence=extracted["landmarks"],
            fps=extracted["fps"],
            validate_tracking_quality=False,
            verbose=False,
        )

        analysis2 = sensei.analyze_from_landmarks(
            landmarks_sequence=extracted["landmarks"],
            fps=extracted["fps"],
            validate_tracking_quality=True,
            verbose=False,
        )

    # Both should produce valid results
    assert analysis1.summary.total_frames == analysis2.summary.total_frames
    assert analysis1.tracking_quality is None
    assert analysis2.tracking_quality is not None


def test_extract_with_invalid_video():
    """Test extract_landmarks with non-existent video raises error."""
    sensei = ClimbingSensei("nonexistent.mp4", validate_quality=False)

    with pytest.raises(FileNotFoundError):
        sensei.extract_landmarks(verbose=False, validate_video_quality=False)


@pytest.mark.skip(reason="End-to-end video visualization tests are handled elsewhere")
def test_cached_pose_results_for_visualization(test_video):
    """Test that pose_results are cached for video visualization."""
    if not test_video.exists():
        pytest.skip("Test video not available")

    with ClimbingSensei(str(test_video), validate_quality=False) as sensei:
        extracted = sensei.extract_landmarks(
            verbose=False, validate_video_quality=False
        )

    # pose_results should be available for drawing
    assert "pose_results" in extracted
    assert len(extracted["pose_results"]) > 0

    # At least some frames should have valid pose results
    valid_results = [r for r in extracted["pose_results"] if r is not None]
    assert len(valid_results) > 0
