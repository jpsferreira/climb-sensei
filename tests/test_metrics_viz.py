"""Tests for the metrics_viz module."""

import numpy as np
from climb_sensei.metrics_viz import (
    create_metric_plot,
    create_metrics_dashboard,
    overlay_metrics_on_frame,
    draw_metric_text_overlay,
)


class TestCreateMetricPlot:
    """Tests for create_metric_plot function."""

    def test_create_plot_basic(self):
        """Test basic plot creation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        current_frame = 2

        plot = create_metric_plot(values, current_frame)

        assert plot.shape == (100, 300, 3)
        assert plot.dtype == np.uint8

    def test_create_plot_custom_size(self):
        """Test plot with custom dimensions."""
        values = [1.0, 2.0, 3.0]

        plot = create_metric_plot(values, 1, width=400, height=150)

        assert plot.shape == (150, 400, 3)

    def test_create_plot_empty_values(self):
        """Test plot with empty values."""
        plot = create_metric_plot([], 0)

        assert plot.shape == (100, 300, 3)
        # Should return background without errors

    def test_create_plot_single_value(self):
        """Test plot with single value."""
        plot = create_metric_plot([5.0], 0)

        assert plot.shape == (100, 300, 3)

    def test_create_plot_with_title(self):
        """Test plot with title."""
        values = [1.0, 2.0, 3.0]

        plot = create_metric_plot(values, 1, title="Test Metric")

        assert plot.shape == (100, 300, 3)

    def test_create_plot_custom_color(self):
        """Test plot with custom color."""
        values = [1.0, 2.0, 3.0]

        plot = create_metric_plot(values, 1, color=(255, 0, 0))

        assert plot.shape == (100, 300, 3)

    def test_create_plot_custom_range(self):
        """Test plot with custom value range."""
        values = [5.0, 6.0, 7.0]

        plot = create_metric_plot(values, 1, min_val=0.0, max_val=10.0)

        assert plot.shape == (100, 300, 3)

    def test_create_plot_zero_range(self):
        """Test plot when all values are the same."""
        values = [5.0, 5.0, 5.0]

        plot = create_metric_plot(values, 1)

        # Should handle zero range without errors
        assert plot.shape == (100, 300, 3)

    def test_create_plot_with_y_label(self):
        """Test plot with y-axis label."""
        values = [1.0, 2.0, 3.0]

        plot = create_metric_plot(values, 1, y_label="units")

        assert plot.shape == (100, 300, 3)

    def test_create_plot_show_current(self):
        """Test plot with current frame indicator."""
        values = [1.0, 2.0, 3.0, 4.0]

        plot = create_metric_plot(values, 2, show_current=True)

        assert plot.shape == (100, 300, 3)


class TestCreateMetricsDashboard:
    """Tests for create_metrics_dashboard function."""

    def test_create_dashboard_basic(self):
        """Test basic dashboard creation."""
        history = {
            "hip_heights": [0.5, 0.4, 0.3, 0.2],
            "velocities": [0.1, 0.2, 0.15, 0.1],
            "sways": [0.01, 0.02, 0.015, 0.01],
            "jerks": [1.0, 2.0, 1.5, 1.0],
            "body_angles": [10.0, 15.0, 12.0, 10.0],
            "hand_spans": [0.3, 0.4, 0.35, 0.3],
            "foot_spans": [0.2, 0.25, 0.22, 0.2],
        }

        dashboard = create_metrics_dashboard(history, current_frame=2)

        assert dashboard.shape[1] == 350  # Default plot width
        assert dashboard.shape[2] == 3  # BGR channels
        assert dashboard.shape[0] > 100  # Multiple plots stacked

    def test_create_dashboard_custom_size(self):
        """Test dashboard with custom plot dimensions."""
        history = {
            "hip_heights": [0.5, 0.4, 0.3],
            "velocities": [0.1, 0.2, 0.15],
        }

        dashboard = create_metrics_dashboard(
            history, current_frame=1, plot_width=400, plot_height=120
        )

        assert dashboard.shape[1] == 400

    def test_create_dashboard_empty_history(self):
        """Test dashboard with empty history."""
        history = {}

        dashboard = create_metrics_dashboard(history, current_frame=0)

        assert dashboard.shape == (100, 350, 3)

    def test_create_dashboard_partial_history(self):
        """Test dashboard with only some metrics."""
        history = {
            "velocities": [0.1, 0.2, 0.15],
        }

        dashboard = create_metrics_dashboard(history, current_frame=1)

        assert dashboard.shape[2] == 3
        assert dashboard.shape[0] >= 100


class TestOverlayMetricsOnFrame:
    """Tests for overlay_metrics_on_frame function."""

    def test_overlay_right_position(self):
        """Test overlay with dashboard on right."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.ones((300, 200, 3), dtype=np.uint8) * 255

        result = overlay_metrics_on_frame(frame, dashboard, position="right")

        assert result.shape == frame.shape
        # Check that overlay was applied (some pixels should be brighter)
        assert result.max() > frame.max()

    def test_overlay_left_position(self):
        """Test overlay with dashboard on left."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.ones((300, 200, 3), dtype=np.uint8) * 255

        result = overlay_metrics_on_frame(frame, dashboard, position="left")

        assert result.shape == frame.shape

    def test_overlay_bottom_position(self):
        """Test overlay with dashboard on bottom."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.ones((100, 300, 3), dtype=np.uint8) * 255

        result = overlay_metrics_on_frame(frame, dashboard, position="bottom")

        assert result.shape == frame.shape

    def test_overlay_alpha_blending(self):
        """Test overlay with different alpha values."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dashboard = np.ones((200, 150, 3), dtype=np.uint8) * 255

        # Test different alpha values
        result_opaque = overlay_metrics_on_frame(frame, dashboard, alpha=1.0)
        result_transparent = overlay_metrics_on_frame(frame, dashboard, alpha=0.5)

        # More transparent should have lower max value
        assert result_transparent.max() < result_opaque.max()

    def test_overlay_dashboard_too_large(self):
        """Test overlay when dashboard is larger than frame."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        dashboard = np.ones((400, 400, 3), dtype=np.uint8) * 255

        # Should clip dashboard to fit
        result = overlay_metrics_on_frame(frame, dashboard)

        assert result.shape == frame.shape


class TestDrawMetricTextOverlay:
    """Tests for draw_metric_text_overlay function."""

    def test_draw_text_basic(self):
        """Test basic text overlay."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = {
            "vertical_progress": 0.5,
            "com_velocity": 0.123,
            "com_sway": 0.01,
        }

        result = draw_metric_text_overlay(frame, metrics)

        assert result.shape == frame.shape
        # Check that text was drawn (some pixels should be white)
        assert result.max() > 0

    def test_draw_text_all_metrics(self):
        """Test text overlay with all metrics."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = {
            "vertical_progress": 0.5,
            "com_velocity": 0.123,
            "com_sway": 0.01,
            "jerk": 10.5,
            "body_angle": 15.2,
        }

        result = draw_metric_text_overlay(frame, metrics)

        assert result.shape == frame.shape

    def test_draw_text_custom_position(self):
        """Test text overlay with custom position."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = {"vertical_progress": 0.5}

        result = draw_metric_text_overlay(frame, metrics, position=(100, 100))

        assert result.shape == frame.shape

    def test_draw_text_custom_style(self):
        """Test text overlay with custom styling."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = {"vertical_progress": 0.5}

        result = draw_metric_text_overlay(
            frame,
            metrics,
            font_scale=0.8,
            thickness=3,
            color=(0, 255, 0),
        )

        assert result.shape == frame.shape

    def test_draw_text_no_background(self):
        """Test text overlay without background."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = {"vertical_progress": 0.5}

        result = draw_metric_text_overlay(frame, metrics, background=False)

        assert result.shape == frame.shape

    def test_draw_text_empty_metrics(self):
        """Test text overlay with no metrics."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        metrics = {}

        result = draw_metric_text_overlay(frame, metrics)

        # Should return frame unchanged
        assert result.shape == frame.shape
