"""Visualization utilities module.

This module provides functions for drawing pose landmarks and
biomechanical data on images for visualization purposes.
"""

from typing import Any, Tuple, Optional, FrozenSet
import cv2
import numpy as np

from .config import FULL_POSE_CONNECTIONS


# Default MediaPipe pose landmark connections (including head)
_POSE_CONNECTIONS = FULL_POSE_CONNECTIONS


def draw_pose_landmarks(
    image: np.ndarray,
    results: Any,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    circle_radius: int = 4,
    connections: Optional[FrozenSet[Tuple[int, int]]] = None,
    landmarks_to_draw: Optional[FrozenSet[int]] = None
) -> np.ndarray:
    """Draw pose landmarks and connections on an image.
    
    This function draws the detected pose landmarks as circles and
    connects them with lines according to MediaPipe's pose model.
    
    Args:
        image: Input image in BGR format (will not be modified).
        results: MediaPipe pose detection results object.
        landmark_color: BGR color tuple for landmark circles (default: green).
        connection_color: BGR color tuple for connection lines (default: red).
        thickness: Line thickness in pixels (default: 2).
        circle_radius: Landmark circle radius in pixels (default: 4).
        connections: Optional set of (start_idx, end_idx) tuples defining which
                    landmarks to connect. If None, uses default full pose connections.
        landmarks_to_draw: Optional set of landmark indices to draw. If None, draws all.
    
    Returns:
        New image with landmarks drawn (BGR format).
    
    Example:
        >>> from climb_sensei.config import CLIMBING_CONNECTIONS, CLIMBING_LANDMARKS
        >>> with PoseEngine() as engine:
        ...     results = engine.process(frame)
        ...     if results:
        ...         # Draw only climbing-relevant landmarks (no head)
        ...         annotated_frame = draw_pose_landmarks(
        ...             frame, results,
        ...             connections=CLIMBING_CONNECTIONS,
        ...             landmarks_to_draw=CLIMBING_LANDMARKS
        ...         )
    """
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    
    if not results or not results.pose_landmarks:
        return annotated_image
    
    # Use default connections if none provided
    if connections is None:
        connections = _POSE_CONNECTIONS
    
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Get first pose (PoseLandmarker can detect multiple poses)
    pose_landmarks = results.pose_landmarks[0]
    
    # Convert landmarks to pixel coordinates
    landmark_points = []
    for landmark in pose_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        landmark_points.append((x, y))
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmark_points) and end_idx < len(landmark_points):
            start_point = landmark_points[start_idx]
            end_point = landmark_points[end_idx]
            cv2.line(annotated_image, start_point, end_point, connection_color, thickness)
    
    # Draw landmarks
    for idx, point in enumerate(landmark_points):
        # Only draw if landmark is in the set to draw (or if no filter is specified)
        if landmarks_to_draw is None or idx in landmarks_to_draw:
            cv2.circle(annotated_image, point, circle_radius, landmark_color, -1)
    
    return annotated_image


def draw_angle_annotation(
    image: np.ndarray,
    point: Tuple[int, int],
    angle: float,
    color: Tuple[int, int, int] = (255, 255, 0),
    font_scale: float = 0.7,
    thickness: int = 2
) -> np.ndarray:
    """Draw angle value annotation at a specific point.
    
    Args:
        image: Input image in BGR format (will not be modified).
        point: (x, y) pixel coordinates where to draw the annotation.
        angle: Angle value in degrees to display.
        color: BGR color tuple for the text (default: cyan).
        font_scale: Font scale factor (default: 0.7).
        thickness: Text thickness in pixels (default: 2).
    
    Returns:
        New image with angle annotation drawn (BGR format).
    """
    annotated_image = image.copy()
    
    # Format angle text
    text = f"{angle:.1f}°"
    
    # Draw text with background for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw background rectangle
    padding = 5
    cv2.rectangle(
        annotated_image,
        (point[0] - padding, point[1] - text_size[1] - padding),
        (point[0] + text_size[0] + padding, point[1] + padding),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        annotated_image,
        text,
        point,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return annotated_image


def draw_distance_line(
    image: np.ndarray,
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    distance: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """Draw a line between two points with optional distance label.
    
    Args:
        image: Input image in BGR format (will not be modified).
        point_a: (x, y) pixel coordinates of the first point.
        point_b: (x, y) pixel coordinates of the second point.
        distance: Optional distance value to display at line midpoint.
        color: BGR color tuple for the line (default: yellow).
        thickness: Line thickness in pixels (default: 2).
    
    Returns:
        New image with distance line drawn (BGR format).
    """
    annotated_image = image.copy()
    
    # Draw line
    cv2.line(annotated_image, point_a, point_b, color, thickness)
    
    # Draw distance label if provided
    if distance is not None:
        midpoint = (
            (point_a[0] + point_b[0]) // 2,
            (point_a[1] + point_b[1]) // 2
        )
        text = f"{distance:.3f}"
        
        cv2.putText(
            annotated_image,
            text,
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
    
    return annotated_image


def draw_metrics_overlay(
    image: np.ndarray,
    current_metrics: Optional[dict] = None,
    cumulative_metrics: Optional[dict] = None,
    font_scale: float = 0.6,
    thickness: int = 2,
    bg_alpha: float = 0.7
) -> np.ndarray:
    """Draw climbing metrics overlay on image.
    
    Args:
        image: Input image in BGR format (will not be modified).
        current_metrics: Dictionary of current frame metrics.
        cumulative_metrics: Dictionary of cumulative/average metrics.
        font_scale: Font scale factor (default: 0.6).
        thickness: Text thickness in pixels (default: 2).
        bg_alpha: Background transparency (0.0-1.0, default: 0.7).
    
    Returns:
        New image with metrics overlay drawn (BGR format).
    """
    annotated_image = image.copy()
    h, w = image.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(25 * font_scale)
    padding = 10
    
    # Prepare text lines
    lines = []
    
    if current_metrics:
        lines.append(("CURRENT FRAME", (0, 255, 255)))  # Yellow
        lines.append((f"Frame: {current_metrics.get('frame', 0)}", (255, 255, 255)))
        lines.append((f"L Elbow: {current_metrics.get('left_elbow_angle', 0):.1f}°", (100, 255, 100)))
        lines.append((f"R Elbow: {current_metrics.get('right_elbow_angle', 0):.1f}°", (100, 255, 100)))
        lines.append((f"L Knee: {current_metrics.get('left_knee_angle', 0):.1f}°", (100, 200, 255)))
        lines.append((f"R Knee: {current_metrics.get('right_knee_angle', 0):.1f}°", (100, 200, 255)))
        lines.append((f"Max Reach: {current_metrics.get('max_reach', 0):.3f}", (255, 150, 100)))
        
        # Lock-off indicators
        if current_metrics.get('left_lock_off'):
            lines.append(("L LOCK-OFF", (0, 0, 255)))
        if current_metrics.get('right_lock_off'):
            lines.append(("R LOCK-OFF", (0, 0, 255)))
    
    if cumulative_metrics and lines:
        lines.append(("", (255, 255, 255)))  # Spacer
    
    if cumulative_metrics:
        lines.append(("AVERAGES", (0, 255, 255)))  # Yellow
        lines.append((f"L Elbow: {cumulative_metrics.get('avg_left_elbow', 0):.1f}°", (150, 255, 150)))
        lines.append((f"R Elbow: {cumulative_metrics.get('avg_right_elbow', 0):.1f}°", (150, 255, 150)))
        lines.append((f"Max Reach: {cumulative_metrics.get('avg_max_reach', 0):.3f}", (255, 180, 150)))
        lines.append((f"Extension: {cumulative_metrics.get('avg_extension', 0):.3f}", (200, 200, 255)))
    
    if not lines:
        return annotated_image
    
    # Calculate overlay dimensions
    max_text_width = 0
    for text, _ in lines:
        if text:
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_text_width = max(max_text_width, text_w)
    
    overlay_width = max_text_width + 2 * padding
    overlay_height = len(lines) * line_height + 2 * padding
    
    # Create semi-transparent background
    overlay = annotated_image.copy()
    x1, y1 = padding, padding
    x2, y2 = x1 + overlay_width, y1 + overlay_height
    
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, annotated_image, 1 - bg_alpha, 0, annotated_image)
    
    # Draw text
    y_offset = y1 + padding + line_height
    for text, color in lines:
        if text:  # Skip empty lines for spacing
            cv2.putText(
                annotated_image,
                text,
                (x1 + padding, y_offset),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )
        y_offset += line_height
    
    return annotated_image

