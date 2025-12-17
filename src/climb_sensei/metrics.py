"""Climbing-specific metrics and movement analysis.

This module provides functions to calculate climbing performance metrics
from pose landmarks, including joint angles, reach distances, body positioning,
and movement efficiency indicators.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from .biomechanics import calculate_joint_angle, calculate_reach_distance


# MediaPipe landmark indices for climbing analysis
class LandmarkIndex:
    """MediaPipe Pose landmark indices."""
    # Shoulders
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    
    # Elbows
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    
    # Wrists
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    # Hips
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Knees
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    
    # Ankles
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Hands
    LEFT_INDEX = 19
    RIGHT_INDEX = 20


class ClimbingMetrics:
    """Calculate climbing-specific performance metrics from pose landmarks."""
    
    @staticmethod
    def calculate_arm_angles(landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate elbow angles for both arms.
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Dictionary with left and right elbow angles in degrees.
        """
        if len(landmarks) < 33:
            return {"left_elbow": 0.0, "right_elbow": 0.0}
        
        # Left arm angle
        left_shoulder = (landmarks[LandmarkIndex.LEFT_SHOULDER]["x"], 
                        landmarks[LandmarkIndex.LEFT_SHOULDER]["y"])
        left_elbow = (landmarks[LandmarkIndex.LEFT_ELBOW]["x"], 
                     landmarks[LandmarkIndex.LEFT_ELBOW]["y"])
        left_wrist = (landmarks[LandmarkIndex.LEFT_WRIST]["x"], 
                     landmarks[LandmarkIndex.LEFT_WRIST]["y"])
        
        # Right arm angle
        right_shoulder = (landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"], 
                         landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"])
        right_elbow = (landmarks[LandmarkIndex.RIGHT_ELBOW]["x"], 
                      landmarks[LandmarkIndex.RIGHT_ELBOW]["y"])
        right_wrist = (landmarks[LandmarkIndex.RIGHT_WRIST]["x"], 
                      landmarks[LandmarkIndex.RIGHT_WRIST]["y"])
        
        return {
            "left_elbow": calculate_joint_angle(left_shoulder, left_elbow, left_wrist),
            "right_elbow": calculate_joint_angle(right_shoulder, right_elbow, right_wrist)
        }
    
    @staticmethod
    def calculate_leg_angles(landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate knee angles for both legs.
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Dictionary with left and right knee angles in degrees.
        """
        if len(landmarks) < 33:
            return {"left_knee": 0.0, "right_knee": 0.0}
        
        # Left leg angle
        left_hip = (landmarks[LandmarkIndex.LEFT_HIP]["x"], 
                   landmarks[LandmarkIndex.LEFT_HIP]["y"])
        left_knee = (landmarks[LandmarkIndex.LEFT_KNEE]["x"], 
                    landmarks[LandmarkIndex.LEFT_KNEE]["y"])
        left_ankle = (landmarks[LandmarkIndex.LEFT_ANKLE]["x"], 
                     landmarks[LandmarkIndex.LEFT_ANKLE]["y"])
        
        # Right leg angle
        right_hip = (landmarks[LandmarkIndex.RIGHT_HIP]["x"], 
                    landmarks[LandmarkIndex.RIGHT_HIP]["y"])
        right_knee = (landmarks[LandmarkIndex.RIGHT_KNEE]["x"], 
                     landmarks[LandmarkIndex.RIGHT_KNEE]["y"])
        right_ankle = (landmarks[LandmarkIndex.RIGHT_ANKLE]["x"], 
                      landmarks[LandmarkIndex.RIGHT_ANKLE]["y"])
        
        return {
            "left_knee": calculate_joint_angle(left_hip, left_knee, left_ankle),
            "right_knee": calculate_joint_angle(right_hip, right_knee, right_ankle)
        }
    
    @staticmethod
    def calculate_reach_span(landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate reach distances from hips to hands.
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Dictionary with reach distances for both arms.
        """
        if len(landmarks) < 33:
            return {"left_reach": 0.0, "right_reach": 0.0, "max_reach": 0.0}
        
        # Use center of hips as reference point
        left_hip = (landmarks[LandmarkIndex.LEFT_HIP]["x"], 
                   landmarks[LandmarkIndex.LEFT_HIP]["y"])
        right_hip = (landmarks[LandmarkIndex.RIGHT_HIP]["x"], 
                    landmarks[LandmarkIndex.RIGHT_HIP]["y"])
        hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                     (left_hip[1] + right_hip[1]) / 2)
        
        # Calculate reach to each hand
        left_wrist = (landmarks[LandmarkIndex.LEFT_WRIST]["x"], 
                     landmarks[LandmarkIndex.LEFT_WRIST]["y"])
        right_wrist = (landmarks[LandmarkIndex.RIGHT_WRIST]["x"], 
                      landmarks[LandmarkIndex.RIGHT_WRIST]["y"])
        
        left_reach = calculate_reach_distance(hip_center, left_wrist)
        right_reach = calculate_reach_distance(hip_center, right_wrist)
        
        return {
            "left_reach": left_reach,
            "right_reach": right_reach,
            "max_reach": max(left_reach, right_reach)
        }
    
    @staticmethod
    def calculate_body_extension(landmarks: List[Dict[str, float]]) -> float:
        """Calculate body extension (vertical span from hips to highest hand).
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Body extension distance (normalized).
        """
        if len(landmarks) < 33:
            return 0.0
        
        # Get hip center (lowest body point for climbing)
        left_hip_y = landmarks[LandmarkIndex.LEFT_HIP]["y"]
        right_hip_y = landmarks[LandmarkIndex.RIGHT_HIP]["y"]
        hip_y = (left_hip_y + right_hip_y) / 2
        
        # Get highest hand
        left_wrist_y = landmarks[LandmarkIndex.LEFT_WRIST]["y"]
        right_wrist_y = landmarks[LandmarkIndex.RIGHT_WRIST]["y"]
        highest_hand_y = min(left_wrist_y, right_wrist_y)  # Lower y = higher on screen
        
        return abs(hip_y - highest_hand_y)
    
    @staticmethod
    def calculate_center_of_mass(landmarks: List[Dict[str, float]]) -> Tuple[float, float]:
        """Estimate center of mass from key body landmarks.
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Tuple of (x, y) coordinates for estimated center of mass.
        """
        if len(landmarks) < 33:
            return (0.0, 0.0)
        
        # Approximate body segment weights (simplified)
        # Torso (shoulders + hips): 50%
        # Each arm: 5%
        # Each leg: 20%
        
        points = []
        weights = []
        
        # Torso center
        shoulder_x = (landmarks[LandmarkIndex.LEFT_SHOULDER]["x"] + 
                     landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"]) / 2
        shoulder_y = (landmarks[LandmarkIndex.LEFT_SHOULDER]["y"] + 
                     landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"]) / 2
        hip_x = (landmarks[LandmarkIndex.LEFT_HIP]["x"] + 
                landmarks[LandmarkIndex.RIGHT_HIP]["x"]) / 2
        hip_y = (landmarks[LandmarkIndex.LEFT_HIP]["y"] + 
                landmarks[LandmarkIndex.RIGHT_HIP]["y"]) / 2
        torso_x = (shoulder_x + hip_x) / 2
        torso_y = (shoulder_y + hip_y) / 2
        points.append((torso_x, torso_y))
        weights.append(0.5)
        
        # Left arm (elbow approximation)
        points.append((landmarks[LandmarkIndex.LEFT_ELBOW]["x"], 
                      landmarks[LandmarkIndex.LEFT_ELBOW]["y"]))
        weights.append(0.05)
        
        # Right arm
        points.append((landmarks[LandmarkIndex.RIGHT_ELBOW]["x"], 
                      landmarks[LandmarkIndex.RIGHT_ELBOW]["y"]))
        weights.append(0.05)
        
        # Left leg (knee approximation)
        points.append((landmarks[LandmarkIndex.LEFT_KNEE]["x"], 
                      landmarks[LandmarkIndex.LEFT_KNEE]["y"]))
        weights.append(0.2)
        
        # Right leg
        points.append((landmarks[LandmarkIndex.RIGHT_KNEE]["x"], 
                      landmarks[LandmarkIndex.RIGHT_KNEE]["y"]))
        weights.append(0.2)
        
        # Weighted average
        total_weight = sum(weights)
        com_x = sum(p[0] * w for p, w in zip(points, weights)) / total_weight
        com_y = sum(p[1] * w for p, w in zip(points, weights)) / total_weight
        
        return (com_x, com_y)
    
    @staticmethod
    def detect_lock_off(landmarks: List[Dict[str, float]], threshold: float = 100.0) -> Dict[str, bool]:
        """Detect if climber is in a lock-off position (bent arm hold).
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
            threshold: Maximum elbow angle to consider as lock-off (degrees).
        
        Returns:
            Dictionary indicating lock-off state for each arm.
        """
        angles = ClimbingMetrics.calculate_arm_angles(landmarks)
        
        return {
            "left_lock_off": angles["left_elbow"] < threshold,
            "right_lock_off": angles["right_elbow"] < threshold,
            "any_lock_off": angles["left_elbow"] < threshold or angles["right_elbow"] < threshold
        }
    
    @staticmethod
    def calculate_hip_angle(landmarks: List[Dict[str, float]]) -> float:
        """Calculate hip angle (torso to thigh angle).
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Average hip angle in degrees.
        """
        if len(landmarks) < 33:
            return 0.0
        
        # Left side
        left_shoulder = (landmarks[LandmarkIndex.LEFT_SHOULDER]["x"], 
                        landmarks[LandmarkIndex.LEFT_SHOULDER]["y"])
        left_hip = (landmarks[LandmarkIndex.LEFT_HIP]["x"], 
                   landmarks[LandmarkIndex.LEFT_HIP]["y"])
        left_knee = (landmarks[LandmarkIndex.LEFT_KNEE]["x"], 
                    landmarks[LandmarkIndex.LEFT_KNEE]["y"])
        
        left_hip_angle = calculate_joint_angle(left_shoulder, left_hip, left_knee)
        
        # Right side
        right_shoulder = (landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"], 
                         landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"])
        right_hip = (landmarks[LandmarkIndex.RIGHT_HIP]["x"], 
                    landmarks[LandmarkIndex.RIGHT_HIP]["y"])
        right_knee = (landmarks[LandmarkIndex.RIGHT_KNEE]["x"], 
                     landmarks[LandmarkIndex.RIGHT_KNEE]["y"])
        
        right_hip_angle = calculate_joint_angle(right_shoulder, right_hip, right_knee)
        
        return (left_hip_angle + right_hip_angle) / 2
    
    @staticmethod
    def calculate_all_metrics(landmarks: List[Dict[str, float]]) -> Dict[str, any]:
        """Calculate all climbing metrics at once.
        
        Args:
            landmarks: List of landmark dictionaries with x, y coordinates.
        
        Returns:
            Dictionary containing all calculated metrics.
        """
        if not landmarks or len(landmarks) < 33:
            return {}
        
        arm_angles = ClimbingMetrics.calculate_arm_angles(landmarks)
        leg_angles = ClimbingMetrics.calculate_leg_angles(landmarks)
        reach_span = ClimbingMetrics.calculate_reach_span(landmarks)
        lock_off = ClimbingMetrics.detect_lock_off(landmarks)
        com = ClimbingMetrics.calculate_center_of_mass(landmarks)
        
        return {
            # Joint angles
            "left_elbow_angle": arm_angles["left_elbow"],
            "right_elbow_angle": arm_angles["right_elbow"],
            "left_knee_angle": leg_angles["left_knee"],
            "right_knee_angle": leg_angles["right_knee"],
            "hip_angle": ClimbingMetrics.calculate_hip_angle(landmarks),
            
            # Reach and extension
            "left_reach": reach_span["left_reach"],
            "right_reach": reach_span["right_reach"],
            "max_reach": reach_span["max_reach"],
            "body_extension": ClimbingMetrics.calculate_body_extension(landmarks),
            
            # Lock-off detection
            "left_lock_off": lock_off["left_lock_off"],
            "right_lock_off": lock_off["right_lock_off"],
            
            # Center of mass
            "com_x": com[0],
            "com_y": com[1],
        }
