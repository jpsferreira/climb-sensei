"""Metrics visualization module.

This module provides functions to visualize climbing metrics as plots
and overlay them on video frames.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2


def create_metric_plot(
    values: List[float],
    current_frame: int,
    width: int = 300,
    height: int = 100,
    title: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Tuple[int, int, int] = (40, 40, 40),
    show_current: bool = True,
    y_label: str = "",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Create a line plot of a metric over time.

    Args:
        values: List of metric values (one per frame)
        current_frame: Current frame index (0-based)
        width: Plot width in pixels
        height: Plot height in pixels
        title: Plot title
        color: Line color (BGR)
        background_color: Background color (BGR)
        show_current: Whether to highlight current value
        y_label: Y-axis label
        min_val: Minimum y value (auto if None)
        max_val: Maximum y value (auto if None)

    Returns:
        Plot image as numpy array (BGR format)
    """
    # Create blank plot
    plot = np.full((height, width, 3), background_color, dtype=np.uint8)

    if not values or len(values) == 0:
        return plot

    # Determine value range
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)

    # Avoid division by zero
    value_range = max_val - min_val
    if value_range == 0:
        value_range = 1.0

    # Plot margins
    margin_left = 50
    margin_right = 10
    margin_top = 30
    margin_bottom = 20

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Draw title
    if title:
        cv2.putText(
            plot,
            title,
            (margin_left, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    # Draw y-axis labels
    if y_label:
        # Max value
        cv2.putText(
            plot,
            f"{max_val:.2f}",
            (5, margin_top + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        # Min value
        cv2.putText(
            plot,
            f"{min_val:.2f}",
            (5, height - margin_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    # Normalize values to plot coordinates
    def value_to_y(val: float) -> int:
        normalized = (val - min_val) / value_range
        y = int(margin_top + plot_height * (1 - normalized))
        return max(margin_top, min(height - margin_bottom, y))

    # Draw grid lines
    for i in range(5):
        y = margin_top + (plot_height * i // 4)
        cv2.line(plot, (margin_left, y), (width - margin_right, y), (60, 60, 60), 1)

    # Draw data line
    num_values = len(values)
    if num_values > 1:
        points = []
        for i, val in enumerate(values):
            x = int(margin_left + (plot_width * i / (num_values - 1)))
            y = value_to_y(val)
            points.append((x, y))

        # Draw line segments
        for i in range(len(points) - 1):
            cv2.line(plot, points[i], points[i + 1], color, 2, cv2.LINE_AA)

        # Highlight current position
        if show_current and current_frame < len(points):
            curr_point = points[current_frame]
            cv2.circle(plot, curr_point, 4, (255, 255, 255), -1)
            cv2.circle(plot, curr_point, 6, color, 2)

            # Draw vertical line at current position
            cv2.line(
                plot,
                (curr_point[0], margin_top),
                (curr_point[0], height - margin_bottom),
                (100, 100, 100),
                1,
            )

    return plot


def create_metrics_dashboard(
    history: Dict[str, List[float]],
    current_frame: int,
    fps: float = 30.0,
    plot_width: int = 350,
    plot_height: int = 100,
) -> np.ndarray:
    """Create a dashboard with multiple metric plots.

    Args:
        history: Dictionary with metric histories from ClimbingAnalyzer.get_history()
        current_frame: Current frame index
        fps: Frames per second (for time axis)
        plot_width: Width of each plot
        plot_height: Height of each plot

    Returns:
        Dashboard image as numpy array (BGR format)
    """
    plots = []

    # Vertical progress (inverted - climbing up)
    if history.get("hip_heights"):
        hip_heights = history["hip_heights"]
        initial = hip_heights[0] if hip_heights else 0
        progress = [initial - h for h in hip_heights]
        plot = create_metric_plot(
            progress,
            current_frame,
            plot_width,
            plot_height,
            title="Vertical Progress",
            color=(0, 255, 255),
            y_label="height",
            min_val=0,
        )
        plots.append(plot)

    # Velocity
    if history.get("velocities"):
        plot = create_metric_plot(
            history["velocities"],
            current_frame,
            plot_width,
            plot_height,
            title="Movement Speed",
            color=(0, 255, 0),
            y_label="vel",
            min_val=0,
        )
        plots.append(plot)

    # Stability (sway)
    if history.get("sways"):
        plot = create_metric_plot(
            history["sways"],
            current_frame,
            plot_width,
            plot_height,
            title="Lateral Sway (stability)",
            color=(255, 128, 0),
            y_label="sway",
            min_val=0,
        )
        plots.append(plot)

    # Smoothness (jerk - lower is better)
    if history.get("jerks"):
        plot = create_metric_plot(
            history["jerks"],
            current_frame,
            plot_width,
            plot_height,
            title="Jerk (smoothness)",
            color=(255, 0, 255),
            y_label="jerk",
            min_val=0,
        )
        plots.append(plot)

    # Body angle
    if history.get("body_angles"):
        plot = create_metric_plot(
            history["body_angles"],
            current_frame,
            plot_width,
            plot_height,
            title="Body Angle (lean)",
            color=(128, 128, 255),
            y_label="deg",
        )
        plots.append(plot)

    # Hand span
    if history.get("hand_spans"):
        plot = create_metric_plot(
            history["hand_spans"],
            current_frame,
            plot_width,
            plot_height,
            title="Hand Span",
            color=(0, 128, 255),
            y_label="span",
            min_val=0,
        )
        plots.append(plot)

    # Stack plots vertically
    if plots:
        dashboard = np.vstack(plots)
    else:
        dashboard = np.zeros((100, plot_width, 3), dtype=np.uint8)

    return dashboard


def overlay_metrics_on_frame(
    frame: np.ndarray,
    dashboard: np.ndarray,
    position: str = "right",
    alpha: float = 0.9,
) -> np.ndarray:
    """Overlay metrics dashboard on video frame.

    Args:
        frame: Input video frame (BGR)
        dashboard: Dashboard image from create_metrics_dashboard
        position: Where to place dashboard ("right", "left", "bottom")
        alpha: Opacity of dashboard (0.0 = transparent, 1.0 = opaque)

    Returns:
        Frame with dashboard overlaid
    """
    frame_h, frame_w = frame.shape[:2]
    dash_h, dash_w = dashboard.shape[:2]

    # Create output frame
    output = frame.copy()

    if position == "right":
        # Place on right side
        x_offset = frame_w - dash_w - 10
        y_offset = 10
    elif position == "left":
        # Place on left side
        x_offset = 10
        y_offset = 10
    elif position == "bottom":
        # Place at bottom center
        x_offset = (frame_w - dash_w) // 2
        y_offset = frame_h - dash_h - 10
    else:
        x_offset = 10
        y_offset = 10

    # Ensure dashboard fits
    if x_offset + dash_w > frame_w:
        x_offset = frame_w - dash_w
    if y_offset + dash_h > frame_h:
        y_offset = frame_h - dash_h
    if x_offset < 0:
        x_offset = 0
    if y_offset < 0:
        y_offset = 0

    # Clip dashboard if needed
    dash_w = min(dash_w, frame_w - x_offset)
    dash_h = min(dash_h, frame_h - y_offset)
    dashboard = dashboard[:dash_h, :dash_w]

    # Blend dashboard onto frame
    roi = output[y_offset : y_offset + dash_h, x_offset : x_offset + dash_w]
    blended = cv2.addWeighted(dashboard, alpha, roi, 1 - alpha, 0)
    output[y_offset : y_offset + dash_h, x_offset : x_offset + dash_w] = blended

    return output


def draw_metric_text_overlay(
    frame: np.ndarray,
    metrics: Dict[str, float],
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    thickness: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
    background: bool = True,
) -> np.ndarray:
    """Draw current metric values as text on frame.

    Args:
        frame: Input video frame
        metrics: Dictionary of current metric values
        position: (x, y) position for text
        font_scale: Font size
        thickness: Text thickness
        color: Text color (BGR)
        background: Whether to draw background for readability

    Returns:
        Frame with text overlaid
    """
    output = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    line_height = int(25 * font_scale)

    # Format metrics for display
    display_metrics = []

    if "vertical_progress" in metrics:
        display_metrics.append(f"Height: {metrics['vertical_progress']:.3f}")
    if "com_velocity" in metrics:
        display_metrics.append(f"Speed: {metrics['com_velocity']:.4f}")
    if "com_sway" in metrics:
        display_metrics.append(f"Sway: {metrics['com_sway']:.4f}")
    if "jerk" in metrics:
        display_metrics.append(f"Jerk: {metrics['jerk']:.4f}")
    if "body_angle" in metrics:
        display_metrics.append(f"Angle: {metrics['body_angle']:.1f}°")

    # Draw background if requested
    if background and display_metrics:
        max_width = max(
            cv2.getTextSize(text, font, font_scale, thickness)[0][0]
            for text in display_metrics
        )
        bg_height = len(display_metrics) * line_height + 10
        cv2.rectangle(
            output,
            (x - 5, y - 20),
            (x + max_width + 5, y + bg_height - 20),
            (0, 0, 0),
            -1,
        )

    # Draw each metric
    for i, text in enumerate(display_metrics):
        text_y = y + i * line_height
        cv2.putText(
            output, text, (x, text_y), font, font_scale, color, thickness, cv2.LINE_AA
        )

    return output
