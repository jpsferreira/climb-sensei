"""Metrics visualization module.

This module provides functions to visualize climbing metrics as plots
and overlay them on video frames.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from .config import VisualizationConfig


def create_metric_plot(
    values: List[float],
    current_frame: int,
    width: int = VisualizationConfig.PLOT_WIDTH,
    height: int = VisualizationConfig.PLOT_HEIGHT,
    title: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Tuple[int, int, int] = VisualizationConfig.PLOT_BACKGROUND_COLOR,
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

    # Plot margins - from config
    margin_left = VisualizationConfig.PLOT_MARGIN_LEFT
    margin_right = VisualizationConfig.PLOT_MARGIN_RIGHT
    margin_top = VisualizationConfig.PLOT_MARGIN_TOP
    margin_bottom = VisualizationConfig.PLOT_MARGIN_BOTTOM

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Draw title
    if title:
        cv2.putText(
            plot,
            title,
            (margin_left, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            VisualizationConfig.PLOT_TITLE_FONT_SCALE,
            VisualizationConfig.PLOT_TITLE_COLOR,
            VisualizationConfig.PLOT_TITLE_THICKNESS,
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
            VisualizationConfig.PLOT_LABEL_FONT_SCALE,
            VisualizationConfig.PLOT_LABEL_COLOR,
            VisualizationConfig.PLOT_LABEL_THICKNESS,
            cv2.LINE_AA,
        )
        # Min value
        cv2.putText(
            plot,
            f"{min_val:.2f}",
            (5, height - margin_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            VisualizationConfig.PLOT_LABEL_FONT_SCALE,
            VisualizationConfig.PLOT_LABEL_COLOR,
            VisualizationConfig.PLOT_LABEL_THICKNESS,
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
        cv2.line(
            plot,
            (margin_left, y),
            (width - margin_right, y),
            VisualizationConfig.PLOT_GRID_COLOR,
            1,
        )

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
            cv2.line(
                plot,
                points[i],
                points[i + 1],
                color,
                VisualizationConfig.PLOT_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        # Highlight current position
        if show_current and current_frame < len(points):
            curr_point = points[current_frame]
            cv2.circle(
                plot,
                curr_point,
                VisualizationConfig.PLOT_CURRENT_MARKER_INNER_RADIUS,
                VisualizationConfig.PLOT_CURRENT_MARKER_INNER_COLOR,
                -1,
            )
            cv2.circle(
                plot,
                curr_point,
                VisualizationConfig.PLOT_CURRENT_MARKER_OUTER_RADIUS,
                color,
                VisualizationConfig.PLOT_CURRENT_MARKER_OUTER_THICKNESS,
            )

            # Draw vertical line at current position
            cv2.line(
                plot,
                (curr_point[0], margin_top),
                (curr_point[0], height - margin_bottom),
                VisualizationConfig.PLOT_CURRENT_LINE_COLOR,
                1,
            )

    return plot


def create_metrics_dashboard(
    history: Dict[str, List[float]],
    current_frame: int,
    fps: float = 30.0,
    plot_width: int = VisualizationConfig.PLOT_WIDTH,
    plot_height: int = VisualizationConfig.PLOT_HEIGHT,
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

    # Movement Economy
    if history.get("movement_economy"):
        plot = create_metric_plot(
            history["movement_economy"],
            current_frame,
            plot_width,
            plot_height,
            title="Movement Economy (efficiency)",
            color=(0, 200, 100),
            y_label="ratio",
            min_val=0,
        )
        plots.append(plot)

    # Lock-offs (boolean visualized as 0/1)
    if history.get("lock_offs"):
        plot = create_metric_plot(
            history["lock_offs"],
            current_frame,
            plot_width,
            plot_height,
            title="Lock-off Positions",
            color=(255, 100, 0),
            y_label="active",
            min_val=0,
            max_val=1,
        )
        plots.append(plot)

    # Fatigue indicator (show in early plots for visibility)
    # We'll calculate a simple rolling average of jerk to show fatigue trend

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


def compose_frame_with_dashboard(
    frame: np.ndarray,
    dashboard: np.ndarray,
    position: str = "right",
    spacing: int = 0,
) -> np.ndarray:
    """Compose video frame side-by-side with metrics dashboard (no overlay).

    Args:
        frame: Input video frame
        dashboard: Metrics dashboard image
        position: Where to place dashboard ("right" or "left")
        spacing: Pixels of spacing between frame and dashboard

    Returns:
        Composite frame with video and dashboard side-by-side
    """
    frame_h, frame_w = frame.shape[:2]
    dash_h, dash_w = dashboard.shape[:2]

    # Match dashboard height to frame height
    if dash_h != frame_h:
        # Scale dashboard to match frame height while maintaining aspect ratio
        scale = frame_h / dash_h
        new_w = int(dash_w * scale)
        dashboard = cv2.resize(
            dashboard, (new_w, frame_h), interpolation=cv2.INTER_LINEAR
        )
        dash_h, dash_w = dashboard.shape[:2]

    # Create composite frame
    total_width = frame_w + dash_w + spacing
    composite = np.zeros((frame_h, total_width, 3), dtype=np.uint8)

    if position == "right":
        # Video on left, dashboard on right
        composite[:, :frame_w] = frame
        composite[:, frame_w + spacing :] = dashboard
    else:  # left
        # Dashboard on left, video on right
        composite[:, :dash_w] = dashboard
        composite[:, dash_w + spacing :] = frame

    return composite


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
    if "movement_economy" in metrics:
        display_metrics.append(f"Economy: {metrics['movement_economy']:.3f}")
    if "is_lock_off" in metrics:
        lock_status = "YES" if metrics["is_lock_off"] else "no"
        display_metrics.append(f"Lock-off: {lock_status}")
    if "is_rest_position" in metrics:
        rest_status = "YES" if metrics["is_rest_position"] else "no"
        display_metrics.append(f"Rest: {rest_status}")

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
