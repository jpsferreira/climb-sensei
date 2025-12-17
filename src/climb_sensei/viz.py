"""Visualization utilities module.

This module provides functions for drawing pose landmarks and
biomechanical data on images for visualization purposes.
"""

from typing import Any, Tuple, Optional
import cv2
import numpy as np
from mediapipe.tasks.python import vision


# MediaPipe pose landmark connections
_POSE_CONNECTIONS = frozenset([
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (11, 23), (12, 14), (12, 24), (13, 15), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20),
    (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (27, 31),
    (28, 30), (28, 32), (29, 31), (30, 32)
])


def draw_pose_landmarks(
    image: np.ndarray,
    results: Any,
    landmark_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    circle_radius: int = 4
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
    
    Returns:
        New image with landmarks drawn (BGR format).
    
    Example:
        >>> with PoseEngine() as engine:
        ...     results = engine.process(frame)
        ...     if results:
        ...         annotated_frame = draw_pose_landmarks(frame, results)
    """
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    
    if not results or not results.pose_landmarks:
        return annotated_image
    
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
    for connection in _POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmark_points) and end_idx < len(landmark_points):
            start_point = landmark_points[start_idx]
            end_point = landmark_points[end_idx]
            cv2.line(annotated_image, start_point, end_point, connection_color, thickness)
    
    # Draw landmarks
    for point in landmark_points:
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
