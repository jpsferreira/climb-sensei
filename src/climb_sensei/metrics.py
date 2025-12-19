"""Climbing analysis metrics.

This module provides climbing performance analysis through temporal tracking
of body position, movement, and stability metrics.

Core Metrics (Active):
- Vertical Progression: Hip height tracking over time
- Movement Stability: Center of mass variance and sway
- Movement Speed: Velocity of center of mass

Additional Metrics (Defined but inactive):
- Movement smoothness (jerk)
- Body positioning (hip-to-wall, body angle)
- Base of support
- Arm reach efficiency
"""

from typing import List, Dict, Optional, Deque
from collections import deque
import numpy as np
from .config import LandmarkIndex
from .biomechanics import calculate_center_of_mass


class ClimbingAnalyzer:
    """Analyzes climbing performance through temporal pose tracking.
    
    Tracks body position over time to calculate movement metrics like
    stability, speed, and vertical progression.
    
    Active Metrics:
    - Hip height (vertical progression)
    - COM velocity (movement speed)
    - COM sway (stability)
    
    Attributes:
        window_size: Number of frames to use for temporal calculations
        fps: Frames per second of video (for time-based metrics)
    """
    
    def __init__(self, window_size: int = 30, fps: float = 30.0):
        """Initialize the climbing analyzer.
        
        Args:
            window_size: Number of frames for temporal window (default: 30 frames = 1 sec @ 30fps)
            fps: Frames per second for velocity calculations
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps  # Time between frames
        
        # Temporal buffers for tracking
        self._hip_heights: Deque[float] = deque(maxlen=window_size)
        self._com_positions: Deque[tuple] = deque(maxlen=window_size)
        
        # Full history for plotting (no max length)
        self._history_hip_heights: List[float] = []
        self._history_com_positions: List[tuple] = []
        self._history_velocities: List[float] = []
        self._history_sways: List[float] = []
        self._history_jerks: List[float] = []
        self._history_body_angles: List[float] = []
        self._history_hand_spans: List[float] = []
        self._history_foot_spans: List[float] = []
        
        # Summary statistics
        self.total_frames = 0
        self.initial_hip_height: Optional[float] = None
    
    def analyze_frame(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze a single frame and return current metrics.
        
        Args:
            landmarks: List of landmark dictionaries with x, y, z coordinates
            
        Returns:
            Dictionary with current frame metrics:
            - hip_height: Current hip height (normalized, 0-1)
            - com_x, com_y: Center of mass position
            - com_velocity: Speed of movement (units/second)
            - com_sway: Lateral stability (std dev of x position)
            - vertical_progress: Height gained from start (normalized)
            - jerk: Movement smoothness
            - body_angle: Lean angle from vertical
            - hand_span: Distance between hands
            - foot_span: Distance between feet
        """
        if len(landmarks) < 33:
            return {}
        
        self.total_frames += 1
        
        # Calculate hip height (average of left and right hip y-coordinates)
        left_hip_y = landmarks[LandmarkIndex.LEFT_HIP]["y"]
        right_hip_y = landmarks[LandmarkIndex.RIGHT_HIP]["y"]
        hip_height = (left_hip_y + right_hip_y) / 2.0
        
        # Store initial height for progress tracking
        if self.initial_hip_height is None:
            self.initial_hip_height = hip_height
        
        # Calculate center of mass (using core body points)
        core_points = [
            (landmarks[LandmarkIndex.LEFT_SHOULDER]["x"], 
             landmarks[LandmarkIndex.LEFT_SHOULDER]["y"]),
            (landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"],
             landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"]),
            (landmarks[LandmarkIndex.LEFT_HIP]["x"],
             landmarks[LandmarkIndex.LEFT_HIP]["y"]),
            (landmarks[LandmarkIndex.RIGHT_HIP]["x"],
             landmarks[LandmarkIndex.RIGHT_HIP]["y"]),
        ]
        weights = np.ones(len(core_points))
        com = calculate_center_of_mass(core_points, weights)
        
        # Update temporal buffers
        self._hip_heights.append(hip_height)
        self._com_positions.append(com)
        
        # Calculate metrics
        metrics = {
            "hip_height": hip_height,
            "com_x": com[0],
            "com_y": com[1],
        }
        
        # Velocity (requires at least 2 frames)
        if len(self._com_positions) >= 2:
            prev_com = self._com_positions[-2]
            curr_com = self._com_positions[-1]
            dx = curr_com[0] - prev_com[0]
            dy = curr_com[1] - prev_com[1]
            distance = np.sqrt(dx**2 + dy**2)
            velocity = distance / self.dt
            metrics["com_velocity"] = velocity
        else:
            velocity = 0.0
            metrics["com_velocity"] = 0.0
        
        # Stability - lateral sway (std dev of x position over window)
        if len(self._com_positions) >= 3:
            com_x_values = [pos[0] for pos in self._com_positions]
            sway = float(np.std(com_x_values))
            metrics["com_sway"] = sway
        else:
            sway = 0.0
            metrics["com_sway"] = 0.0
        
        # Vertical progress from start
        metrics["vertical_progress"] = self.initial_hip_height - hip_height
        
        # Additional metrics
        # Jerk (smoothness)
        if len(self._com_positions) >= 4:
            jerk = AdvancedClimbingMetrics.calculate_jerk(
                list(self._com_positions), self.dt
            )
            metrics["jerk"] = jerk
        else:
            jerk = 0.0
            metrics["jerk"] = 0.0
        
        # Body angle
        body_angle = AdvancedClimbingMetrics.calculate_body_angle(landmarks)
        metrics["body_angle"] = body_angle
        
        # Base of support
        base_support = AdvancedClimbingMetrics.calculate_base_of_support(landmarks)
        metrics["hand_span"] = base_support.get("hand_span", 0.0)
        metrics["foot_span"] = base_support.get("foot_span", 0.0)
        metrics["hand_foot_span"] = base_support.get("hand_foot_span", 0.0)
        
        # Store in history
        self._history_hip_heights.append(hip_height)
        self._history_com_positions.append(com)
        self._history_velocities.append(velocity)
        self._history_sways.append(sway)
        self._history_jerks.append(jerk)
        self._history_body_angles.append(body_angle)
        self._history_hand_spans.append(metrics["hand_span"])
        self._history_foot_spans.append(metrics["foot_span"])
        
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for the entire climb.
        
        Returns:
            Dictionary with summary metrics:
            - total_frames: Number of frames analyzed
            - avg_velocity: Average movement speed
            - max_velocity: Maximum movement speed
            - avg_sway: Average lateral instability
            - avg_jerk: Average movement jerkiness
            - avg_body_angle: Average lean angle
            - avg_hand_span: Average distance between hands
            - avg_foot_span: Average distance between feet
            - total_vertical_progress: Total height gained
            - max_height: Highest hip position reached
            - min_height: Lowest hip position reached
        """
        if self.total_frames == 0:
            return {}
        
        summary = {
            "total_frames": self.total_frames,
            "avg_velocity": float(np.mean(self._history_velocities)) if self._history_velocities else 0.0,
            "max_velocity": float(np.max(self._history_velocities)) if self._history_velocities else 0.0,
            "avg_sway": float(np.mean(self._history_sways)) if self._history_sways else 0.0,
            "max_sway": float(np.max(self._history_sways)) if self._history_sways else 0.0,
            "avg_jerk": float(np.mean(self._history_jerks)) if self._history_jerks else 0.0,
            "max_jerk": float(np.max(self._history_jerks)) if self._history_jerks else 0.0,
            "avg_body_angle": float(np.mean(self._history_body_angles)) if self._history_body_angles else 0.0,
            "avg_hand_span": float(np.mean(self._history_hand_spans)) if self._history_hand_spans else 0.0,
            "avg_foot_span": float(np.mean(self._history_foot_spans)) if self._history_foot_spans else 0.0,
            "total_vertical_progress": self.initial_hip_height - self._history_hip_heights[-1] if self._history_hip_heights else 0.0,
            "max_height": self.initial_hip_height - min(self._history_hip_heights) if self._history_hip_heights else 0.0,
            "min_height": self.initial_hip_height - max(self._history_hip_heights) if self._history_hip_heights else 0.0,
        }
        
        return summary
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get complete time-series history of all metrics.
        
        Returns:
            Dictionary with lists of metric values over time:
            - hip_heights: Hip height at each frame
            - velocities: Velocity at each frame
            - sways: Lateral sway at each frame
            - jerks: Jerk (smoothness) at each frame
            - body_angles: Body angle at each frame
            - hand_spans: Hand span at each frame
            - foot_spans: Foot span at each frame
        """
        return {
            "hip_heights": self._history_hip_heights.copy(),
            "velocities": self._history_velocities.copy(),
            "sways": self._history_sways.copy(),
            "jerks": self._history_jerks.copy(),
            "body_angles": self._history_body_angles.copy(),
            "hand_spans": self._history_hand_spans.copy(),
            "foot_spans": self._history_foot_spans.copy(),
        }
    
    def reset(self):
        """Reset the analyzer for a new climb."""
        self._hip_heights.clear()
        self._com_positions.clear()
        self._history_hip_heights.clear()
        self._history_com_positions.clear()
        self._history_velocities.clear()
        self._history_sways.clear()
        self._history_jerks.clear()
        self._history_body_angles.clear()
        self._history_hand_spans.clear()
        self._history_foot_spans.clear()
        self.total_frames = 0
        self.initial_hip_height = None


# ============================================================================
# ADDITIONAL METRICS (Defined but not active - for future use)
# ============================================================================

class AdvancedClimbingMetrics:
    """Additional climbing metrics (currently inactive).
    
    These metrics are defined but not integrated into the main analyzer.
    They can be activated in future versions for more detailed analysis.
    """
    
    @staticmethod
    def calculate_jerk(positions: List[tuple], dt: float) -> float:
        """Calculate movement smoothness (jerk - rate of change of acceleration).
        
        Lower jerk = smoother, more controlled movement
        
        Args:
            positions: List of (x, y) positions over time
            dt: Time step between positions
            
        Returns:
            Average jerk magnitude
        """
        if len(positions) < 4:
            return 0.0
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append((dx/dt, dy/dt))
        
        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            ax = (velocities[i][0] - velocities[i-1][0]) / dt
            ay = (velocities[i][1] - velocities[i-1][1]) / dt
            accelerations.append((ax, ay))
        
        # Calculate jerk (derivative of acceleration)
        jerks = []
        for i in range(1, len(accelerations)):
            jx = (accelerations[i][0] - accelerations[i-1][0]) / dt
            jy = (accelerations[i][1] - accelerations[i-1][1]) / dt
            jerk_magnitude = np.sqrt(jx**2 + jy**2)
            jerks.append(jerk_magnitude)
        
        return float(np.mean(jerks)) if jerks else 0.0
    
    @staticmethod
    def calculate_base_of_support(landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate base of support metrics.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Dictionary with:
            - hand_span: Distance between hands
            - foot_span: Distance between feet
            - hand_foot_span: Distance between average hand and foot positions
        """
        if len(landmarks) < 33:
            return {}
        
        # Hand positions
        left_hand = (landmarks[LandmarkIndex.LEFT_WRIST]["x"],
                    landmarks[LandmarkIndex.LEFT_WRIST]["y"])
        right_hand = (landmarks[LandmarkIndex.RIGHT_WRIST]["x"],
                     landmarks[LandmarkIndex.RIGHT_WRIST]["y"])
        hand_span = np.sqrt((right_hand[0] - left_hand[0])**2 + 
                           (right_hand[1] - left_hand[1])**2)
        
        # Foot positions
        left_foot = (landmarks[LandmarkIndex.LEFT_ANKLE]["x"],
                    landmarks[LandmarkIndex.LEFT_ANKLE]["y"])
        right_foot = (landmarks[LandmarkIndex.RIGHT_ANKLE]["x"],
                     landmarks[LandmarkIndex.RIGHT_ANKLE]["y"])
        foot_span = np.sqrt((right_foot[0] - left_foot[0])**2 + 
                           (right_foot[1] - left_foot[1])**2)
        
        # Hand-foot distance (vertical component of base)
        avg_hand_y = (left_hand[1] + right_hand[1]) / 2
        avg_foot_y = (left_foot[1] + right_foot[1]) / 2
        hand_foot_span = abs(avg_hand_y - avg_foot_y)
        
        return {
            "hand_span": float(hand_span),
            "foot_span": float(foot_span),
            "hand_foot_span": float(hand_foot_span),
        }
    
    @staticmethod
    def calculate_body_angle(landmarks: List[Dict[str, float]]) -> float:
        """Calculate body lean angle from vertical.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Angle in degrees (0 = vertical, positive = leaning back)
        """
        if len(landmarks) < 33:
            return 0.0
        
        # Use shoulder and hip midpoints
        shoulder_x = (landmarks[LandmarkIndex.LEFT_SHOULDER]["x"] +
                     landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"]) / 2
        shoulder_y = (landmarks[LandmarkIndex.LEFT_SHOULDER]["y"] +
                     landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"]) / 2
        
        hip_x = (landmarks[LandmarkIndex.LEFT_HIP]["x"] +
                landmarks[LandmarkIndex.RIGHT_HIP]["x"]) / 2
        hip_y = (landmarks[LandmarkIndex.LEFT_HIP]["y"] +
                landmarks[LandmarkIndex.RIGHT_HIP]["y"]) / 2
        
        # Calculate angle from vertical
        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        
        angle = np.degrees(np.arctan2(dx, dy))
        return float(angle)
