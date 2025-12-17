"""Biomechanics calculation module.

This module provides pure mathematical functions for calculating
biomechanical metrics such as joint angles and distances.
All functions are stateless and work with normalized coordinates.
"""

import math
from typing import Tuple


def calculate_joint_angle(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    point_c: Tuple[float, float]
) -> float:
    """Calculate the angle at point B formed by three points A-B-C.
    
    This function calculates the interior angle at the middle point (B)
    using the law of cosines. The angle is measured in degrees.
    
    Args:
        point_a: Coordinates (x, y) of the first point.
        point_b: Coordinates (x, y) of the vertex point (angle point).
        point_c: Coordinates (x, y) of the third point.
    
    Returns:
        Angle in degrees at point B (range: 0-180).
    
    Example:
        >>> # Calculate elbow angle
        >>> shoulder = (0.5, 0.3)
        >>> elbow = (0.6, 0.5)
        >>> wrist = (0.7, 0.6)
        >>> angle = calculate_joint_angle(shoulder, elbow, wrist)
    """
    # Calculate vectors BA and BC
    ba_x = point_a[0] - point_b[0]
    ba_y = point_a[1] - point_b[1]
    bc_x = point_c[0] - point_b[0]
    bc_y = point_c[1] - point_b[1]
    
    # Calculate dot product
    dot_product = ba_x * bc_x + ba_y * bc_y
    
    # Calculate magnitudes
    magnitude_ba = math.sqrt(ba_x ** 2 + ba_y ** 2)
    magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)
    
    # Avoid division by zero
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0
    
    # Calculate angle using dot product formula
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    # Clamp to valid range for arccos
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # Convert to degrees
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees


def calculate_reach_distance(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two points.
    
    This function calculates the straight-line distance between two
    points in 2D space. Useful for measuring reach distances between
    body landmarks.
    
    Args:
        point_a: Coordinates (x, y) of the first point.
        point_b: Coordinates (x, y) of the second point.
    
    Returns:
        Euclidean distance between the two points.
    
    Example:
        >>> # Calculate reach from hip to hand
        >>> hip = (0.5, 0.5)
        >>> hand = (0.7, 0.3)
        >>> distance = calculate_reach_distance(hip, hand)
    """
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    
    distance = math.sqrt(dx ** 2 + dy ** 2)
    
    return distance


def calculate_center_of_mass(
    points: list[Tuple[float, float]],
    weights: list[float] | None = None
) -> Tuple[float, float]:
    """Calculate the weighted center of mass for a set of points.
    
    Args:
        points: List of (x, y) coordinate tuples.
        weights: Optional list of weights for each point. If None, all
            points are weighted equally.
    
    Returns:
        Coordinates (x, y) of the center of mass.
    
    Raises:
        ValueError: If points list is empty or weights length doesn't match.
    
    Example:
        >>> # Calculate body center of mass
        >>> landmarks = [(0.5, 0.3), (0.5, 0.5), (0.5, 0.7)]
        >>> center = calculate_center_of_mass(landmarks)
    """
    if not points:
        raise ValueError("Points list cannot be empty")
    
    if weights is None:
        weights = [1.0] * len(points)
    
    if len(points) != len(weights):
        raise ValueError("Points and weights must have the same length")
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    
    weighted_x = sum(p[0] * w for p, w in zip(points, weights))
    weighted_y = sum(p[1] * w for p, w in zip(points, weights))
    
    center_x = weighted_x / total_weight
    center_y = weighted_y / total_weight
    
    return (center_x, center_y)
