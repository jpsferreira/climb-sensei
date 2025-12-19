"""Tests for the viz module."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock
from climb_sensei.viz import (
    draw_pose_landmarks,
    draw_angle_annotation,
    draw_distance_line,
    draw_metrics_overlay,
)
from climb_sensei.config import CLIMBING_CONNECTIONS, CLIMBING_LANDMARKS


@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_pose_results():
    """Create mock MediaPipe pose results."""
    results = Mock()
    
    # Create mock landmarks
    landmarks = []
    for i in range(33):
        landmark = Mock()
        landmark.x = 0.5 + (i % 5) * 0.05
        landmark.y = 0.3 + (i // 11) * 0.2
        landmarks.append(landmark)
    
    results.pose_landmarks = [landmarks]
    return results


class TestDrawPoseLandmarks:
    """Tests for draw_pose_landmarks function."""
    
    def test_draw_landmarks_basic(self, sample_frame, mock_pose_results):
        """Test basic landmark drawing."""
        result = draw_pose_landmarks(sample_frame, mock_pose_results)
        
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8
        # Should have drawn something (not all zeros)
        assert result.max() > 0
    
    def test_draw_landmarks_with_connections(self, sample_frame, mock_pose_results):
        """Test drawing with specific connections."""
        result = draw_pose_landmarks(
            sample_frame,
            mock_pose_results,
            connections=CLIMBING_CONNECTIONS,
        )
        
        assert result.shape == sample_frame.shape
        assert result.max() > 0
    
    def test_draw_landmarks_subset(self, sample_frame, mock_pose_results):
        """Test drawing only climbing landmarks."""
        result = draw_pose_landmarks(
            sample_frame,
            mock_pose_results,
            connections=CLIMBING_CONNECTIONS,
            landmarks_to_draw=CLIMBING_LANDMARKS,
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_landmarks_custom_colors(self, sample_frame, mock_pose_results):
        """Test drawing with custom colors."""
        result = draw_pose_landmarks(
            sample_frame,
            mock_pose_results,
            landmark_color=(255, 0, 0),
            connection_color=(0, 255, 0),
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_landmarks_custom_thickness(self, sample_frame, mock_pose_results):
        """Test drawing with custom thickness."""
        result = draw_pose_landmarks(
            sample_frame,
            mock_pose_results,
            thickness=5,
            circle_radius=10,
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_landmarks_no_color_coding(self, sample_frame, mock_pose_results):
        """Test drawing without color coding."""
        result = draw_pose_landmarks(
            sample_frame,
            mock_pose_results,
            use_color_coding=False,
            landmark_color=(0, 255, 0),
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_landmarks_no_results(self, sample_frame):
        """Test drawing with no pose detected."""
        result = draw_pose_landmarks(sample_frame, None)
        
        # Should return copy of original frame
        assert result.shape == sample_frame.shape
        np.testing.assert_array_equal(result, sample_frame)
    
    def test_draw_landmarks_empty_results(self, sample_frame):
        """Test drawing with empty results."""
        results = Mock()
        results.pose_landmarks = None
        
        result = draw_pose_landmarks(sample_frame, results)
        
        assert result.shape == sample_frame.shape


class TestDrawAngleAnnotation:
    """Tests for draw_angle_annotation function."""
    
    def test_draw_angle_basic(self, sample_frame):
        """Test basic angle annotation."""
        result = draw_angle_annotation(sample_frame, (100, 100), 45.5)
        
        assert result.shape == sample_frame.shape
        assert result.max() > 0
    
    def test_draw_angle_custom_color(self, sample_frame):
        """Test angle annotation with custom color."""
        result = draw_angle_annotation(
            sample_frame, (100, 100), 90.0,
            color=(0, 255, 0)
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_angle_custom_style(self, sample_frame):
        """Test angle annotation with custom styling."""
        result = draw_angle_annotation(
            sample_frame, (100, 100), 120.0,
            font_scale=1.0,
            thickness=3
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_angle_zero(self, sample_frame):
        """Test angle annotation with zero angle."""
        result = draw_angle_annotation(sample_frame, (100, 100), 0.0)
        
        assert result.shape == sample_frame.shape
    
    def test_draw_angle_large_value(self, sample_frame):
        """Test angle annotation with large value."""
        result = draw_angle_annotation(sample_frame, (100, 100), 179.9)
        
        assert result.shape == sample_frame.shape


class TestDrawDistanceLine:
    """Tests for draw_distance_line function."""
    
    def test_draw_distance_basic(self, sample_frame):
        """Test basic distance line drawing."""
        result = draw_distance_line(
            sample_frame,
            (100, 100),
            (200, 200)
        )
        
        assert result.shape == sample_frame.shape
        assert result.max() > 0
    
    def test_draw_distance_with_value(self, sample_frame):
        """Test distance line with distance value."""
        result = draw_distance_line(
            sample_frame,
            (100, 100),
            (200, 200),
            distance=1.5
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_distance_custom_color(self, sample_frame):
        """Test distance line with custom color."""
        result = draw_distance_line(
            sample_frame,
            (100, 100),
            (200, 200),
            color=(255, 0, 0)
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_distance_custom_thickness(self, sample_frame):
        """Test distance line with custom thickness."""
        result = draw_distance_line(
            sample_frame,
            (100, 100),
            (200, 200),
            thickness=5
        )
        
        assert result.shape == sample_frame.shape


class TestDrawMetricsOverlay:
    """Tests for draw_metrics_overlay function."""
    
    def test_draw_overlay_current_only(self, sample_frame):
        """Test overlay with current metrics only."""
        metrics = {
            "left_elbow_angle": 90.5,
            "right_elbow_angle": 95.2,
            "max_reach": 0.5,
            "body_extension": 0.8,
        }
        
        result = draw_metrics_overlay(sample_frame, current_metrics=metrics)
        
        assert result.shape == sample_frame.shape
        assert result.max() > 0
    
    def test_draw_overlay_cumulative_only(self, sample_frame):
        """Test overlay with cumulative metrics only."""
        cumulative = {
            "avg_left_elbow": 92.0,
            "avg_right_elbow": 93.5,
            "avg_max_reach": 0.45,
            "avg_extension": 0.75,
        }
        
        result = draw_metrics_overlay(sample_frame, cumulative_metrics=cumulative)
        
        assert result.shape == sample_frame.shape
        assert result.max() > 0
    
    def test_draw_overlay_both(self, sample_frame):
        """Test overlay with both current and cumulative metrics."""
        current = {
            "left_elbow_angle": 90.5,
            "right_elbow_angle": 95.2,
            "max_reach": 0.5,
            "body_extension": 0.8,
        }
        cumulative = {
            "avg_left_elbow": 92.0,
            "avg_right_elbow": 93.5,
            "avg_max_reach": 0.45,
            "avg_extension": 0.75,
        }
        
        result = draw_metrics_overlay(
            sample_frame,
            current_metrics=current,
            cumulative_metrics=cumulative
        )
        
        assert result.shape == sample_frame.shape
        assert result.max() > 0
    
    def test_draw_overlay_custom_style(self, sample_frame):
        """Test overlay with custom styling."""
        metrics = {"left_elbow_angle": 90.5}
        
        result = draw_metrics_overlay(
            sample_frame,
            current_metrics=metrics,
            font_scale=0.8,
            thickness=3,
            bg_alpha=0.5
        )
        
        assert result.shape == sample_frame.shape
    
    def test_draw_overlay_no_metrics(self, sample_frame):
        """Test overlay with no metrics."""
        result = draw_metrics_overlay(sample_frame)
        
        # Should still work, just show headers
        assert result.shape == sample_frame.shape


class TestDrawPoseLandmarksEdgeCases:
    """Edge case tests for draw_pose_landmarks."""
    
    def test_draw_on_small_frame(self):
        """Test drawing on very small frame."""
        small_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        
        results = Mock()
        landmarks = []
        for i in range(33):
            landmark = Mock()
            landmark.x = 0.5
            landmark.y = 0.5
            landmarks.append(landmark)
        results.pose_landmarks = [landmarks]
        
        result = draw_pose_landmarks(small_frame, results)
        
        assert result.shape == small_frame.shape
    
    def test_draw_on_large_frame(self):
        """Test drawing on very large frame."""
        large_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        
        results = Mock()
        landmarks = []
        for i in range(33):
            landmark = Mock()
            landmark.x = 0.5
            landmark.y = 0.5
            landmarks.append(landmark)
        results.pose_landmarks = [landmarks]
        
        result = draw_pose_landmarks(large_frame, results)
        
        assert result.shape == large_frame.shape
    
    def test_draw_with_out_of_bounds_landmarks(self):
        """Test drawing with landmarks outside frame bounds."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = Mock()
        landmarks = []
        for i in range(33):
            landmark = Mock()
            landmark.x = 1.5  # Out of bounds
            landmark.y = -0.5  # Out of bounds
            landmarks.append(landmark)
        results.pose_landmarks = [landmarks]
        
        # Should handle gracefully without errors
        result = draw_pose_landmarks(frame, results)
        
        assert result.shape == frame.shape
