"""Tests for compose_frame_with_dashboard function."""

import numpy as np
from climb_sensei.metrics_viz import compose_frame_with_dashboard


class TestComposeFrameWithDashboard:
    """Test the compose_frame_with_dashboard function."""

    def test_compose_right_position(self):
        """Test composing frame with dashboard on right."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.zeros((480, 200, 3), dtype=np.uint8)

        result = compose_frame_with_dashboard(frame, dashboard, position="right")

        # Total width should be frame + dashboard
        assert result.shape == (480, 840, 3)
        # Frame should be on left
        assert np.array_equal(result[:, :640], frame)

    def test_compose_left_position(self):
        """Test composing frame with dashboard on left."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.zeros((480, 200, 3), dtype=np.uint8)

        result = compose_frame_with_dashboard(frame, dashboard, position="left")

        # Total width should be frame + dashboard
        assert result.shape == (480, 840, 3)
        # Frame should be on right
        assert np.array_equal(result[:, 200:], frame)

    def test_compose_with_spacing(self):
        """Test composing with spacing between frame and dashboard."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.zeros((480, 200, 3), dtype=np.uint8)

        result = compose_frame_with_dashboard(
            frame, dashboard, position="right", spacing=20
        )

        # Total width should be frame + dashboard + spacing
        assert result.shape == (480, 860, 3)

    def test_compose_scales_dashboard_height(self):
        """Test that dashboard is scaled to match frame height."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Dashboard with different height
        dashboard = np.zeros((500, 300, 3), dtype=np.uint8)

        result = compose_frame_with_dashboard(frame, dashboard, position="right")

        # Result height should match frame
        assert result.shape[0] == 1080
        # Dashboard should be scaled proportionally
        expected_dash_width = int(300 * (1080 / 500))
        expected_total_width = 1920 + expected_dash_width
        assert result.shape[1] == expected_total_width

    def test_compose_preserves_frame_content(self):
        """Test that frame content is preserved."""
        # Create frame with distinct pattern
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dashboard = np.zeros((480, 200, 3), dtype=np.uint8)

        result = compose_frame_with_dashboard(frame, dashboard, position="right")

        # Original frame should be unchanged in result
        assert np.array_equal(result[:, :640], frame)

    def test_compose_same_height_no_scaling(self):
        """Test that dashboard with same height is not scaled."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dashboard = np.zeros((720, 400, 3), dtype=np.uint8)

        result = compose_frame_with_dashboard(frame, dashboard, position="left")

        # Should be exact concatenation without scaling
        assert result.shape == (720, 1680, 3)
        assert np.array_equal(result[:, :400], dashboard)
        assert np.array_equal(result[:, 400:], frame)
