"""Climbing analysis metrics.

This module provides climbing performance analysis through temporal tracking
of body position, movement, and stability metrics.

Core Metrics (Active):
- Vertical Progression: Hip height tracking over time
- Movement Stability: Center of mass variance and sway
- Movement Speed: Velocity of center of mass
- Movement Economy: Efficiency of movement
- Lock-off Detection: Identification of static strength positions
- Rest Position Detection: Low-stress recovery positions
- Fatigue Indicators: Movement quality degradation over time

Additional Metrics (Defined but inactive):
- Movement smoothness (jerk)
- Body positioning (hip-to-wall, body angle)
- Base of support
- Arm reach efficiency
"""

from typing import List, Dict, Optional, Deque
from collections import deque
import numpy as np
from .config import LandmarkIndex, MetricsConfig
from .biomechanics import calculate_center_of_mass, calculate_limb_angles
from .models import FrameMetrics, ClimbingSummary


class ClimbingAnalyzer:
    """Analyzes climbing performance through temporal pose tracking.

    Tracks body position over time to calculate movement metrics like
    stability, speed, vertical progression, efficiency, and technique quality.

    Active Metrics:
    - Hip height (vertical progression)
    - COM velocity (movement speed)
    - COM sway (stability)
    - Movement economy (efficiency)
    - Lock-off detection (static strength)
    - Rest position detection (recovery periods)
    - Fatigue indicators (quality degradation)
    - Joint angles (elbows, shoulders, knees, hips)

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
        self._history_movement_economy: List[float] = []
        self._history_lock_offs: List[bool] = []
        self._history_rest_positions: List[bool] = []

        # Joint angle tracking
        self._history_joint_angles: Dict[str, List[float]] = {
            "left_elbow": [],
            "right_elbow": [],
            "left_shoulder": [],
            "right_shoulder": [],
            "left_knee": [],
            "right_knee": [],
            "left_hip": [],
            "right_hip": [],
        }

        # Summary statistics
        self.total_frames = 0
        self.initial_hip_height: Optional[float] = None
        self._total_distance_traveled = 0.0

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
            - movement_economy: Vertical progress / total distance (efficiency)
            - is_lock_off: Boolean indicating lock-off position
            - is_rest_position: Boolean indicating rest position
            - left_elbow, right_elbow: Elbow angles
            - left_shoulder, right_shoulder: Shoulder angles
            - left_knee, right_knee: Knee angles
            - left_hip, right_hip: Hip angles
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
            (
                landmarks[LandmarkIndex.LEFT_SHOULDER]["x"],
                landmarks[LandmarkIndex.LEFT_SHOULDER]["y"],
            ),
            (
                landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"],
                landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"],
            ),
            (
                landmarks[LandmarkIndex.LEFT_HIP]["x"],
                landmarks[LandmarkIndex.LEFT_HIP]["y"],
            ),
            (
                landmarks[LandmarkIndex.RIGHT_HIP]["x"],
                landmarks[LandmarkIndex.RIGHT_HIP]["y"],
            ),
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

        # Calculate joint angles
        joint_angles = calculate_limb_angles(landmarks, LandmarkIndex)
        metrics.update(joint_angles)

        # Lock-off detection (elbow < threshold AND low velocity)
        left_lock_off = (
            joint_angles.get("left_elbow", 180)
            < MetricsConfig.LOCK_OFF_THRESHOLD_DEGREES
            and velocity < MetricsConfig.REST_VELOCITY_THRESHOLD
        )
        right_lock_off = (
            joint_angles.get("right_elbow", 180)
            < MetricsConfig.LOCK_OFF_THRESHOLD_DEGREES
            and velocity < MetricsConfig.REST_VELOCITY_THRESHOLD
        )
        is_lock_off = left_lock_off or right_lock_off
        metrics["is_lock_off"] = is_lock_off
        metrics["left_lock_off"] = left_lock_off
        metrics["right_lock_off"] = right_lock_off

        # Rest position detection (low body angle AND low velocity)
        # Body angle close to 0° indicates vertical position
        is_rest_position = (
            body_angle < MetricsConfig.REST_BODY_ANGLE_THRESHOLD
            and velocity < MetricsConfig.REST_VELOCITY_THRESHOLD
        )
        metrics["is_rest_position"] = is_rest_position

        # Movement economy (vertical progress / total distance traveled)
        # Update total distance
        if len(self._com_positions) >= 2:
            prev_pos = self._com_positions[-2]
            curr_pos = self._com_positions[-1]
            segment_distance = np.sqrt(
                (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
            )
            self._total_distance_traveled += segment_distance

        # Calculate economy ratio
        if self._total_distance_traveled > 0:
            vertical_progress = self.initial_hip_height - hip_height
            movement_economy = vertical_progress / self._total_distance_traveled
        else:
            movement_economy = 0.0
        metrics["movement_economy"] = movement_economy

        # Store in history
        self._history_hip_heights.append(hip_height)
        self._history_com_positions.append(com)
        self._history_velocities.append(velocity)
        self._history_sways.append(sway)
        self._history_jerks.append(jerk)
        self._history_body_angles.append(body_angle)
        self._history_hand_spans.append(metrics["hand_span"])
        self._history_foot_spans.append(metrics["foot_span"])
        self._history_movement_economy.append(movement_economy)
        self._history_lock_offs.append(is_lock_off)
        self._history_rest_positions.append(is_rest_position)

        # Store joint angles
        for joint_name in self._history_joint_angles:
            angle_value = joint_angles.get(joint_name, 0.0)
            self._history_joint_angles[joint_name].append(angle_value)

        # Return dictionary for backward compatibility
        # TODO: In next major version, return FrameMetrics directly
        return metrics

    def analyze_frame_typed(self, landmarks: List[Dict[str, float]]) -> FrameMetrics:
        """Analyze frame and return typed metrics object.

        This is the preferred method for type-safe code. Returns an immutable
        FrameMetrics object instead of a dictionary.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Immutable FrameMetrics object with type safety
        """
        metrics_dict = self.analyze_frame(landmarks)
        return FrameMetrics(
            hip_height=metrics_dict["hip_height"],
            com_velocity=metrics_dict["com_velocity"],
            com_sway=metrics_dict["com_sway"],
            jerk=metrics_dict["jerk"],
            vertical_progress=metrics_dict["vertical_progress"],
            movement_economy=metrics_dict["movement_economy"],
            is_lock_off=metrics_dict["is_lock_off"],
            left_lock_off=metrics_dict["left_lock_off"],
            right_lock_off=metrics_dict["right_lock_off"],
            is_rest_position=metrics_dict["is_rest_position"],
            body_angle=metrics_dict["body_angle"],
            hand_span=metrics_dict["hand_span"],
            foot_span=metrics_dict["foot_span"],
            left_elbow=metrics_dict["left_elbow"],
            right_elbow=metrics_dict["right_elbow"],
            left_shoulder=metrics_dict["left_shoulder"],
            right_shoulder=metrics_dict["right_shoulder"],
            left_knee=metrics_dict["left_knee"],
            right_knee=metrics_dict["right_knee"],
            left_hip=metrics_dict["left_hip"],
            right_hip=metrics_dict["right_hip"],
        )

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
            - total_distance_traveled: Total COM movement distance
            - avg_movement_economy: Average efficiency ratio
            - lock_off_count: Number of frames in lock-off position
            - lock_off_percentage: Percentage of time in lock-off
            - rest_count: Number of frames in rest position
            - rest_percentage: Percentage of time in rest
            - fatigue_score: Quality degradation indicator (0-1, higher = more fatigued)
            - avg_left_elbow, avg_right_elbow: Average elbow angles
            - avg_left_shoulder, avg_right_shoulder: Average shoulder angles
            - avg_left_knee, avg_right_knee: Average knee angles
        """
        if self.total_frames == 0:
            return {}

        summary = {
            "total_frames": self.total_frames,
            "avg_velocity": (
                float(np.mean(self._history_velocities))
                if self._history_velocities
                else 0.0
            ),
            "max_velocity": (
                float(np.max(self._history_velocities))
                if self._history_velocities
                else 0.0
            ),
            "avg_sway": (
                float(np.mean(self._history_sways)) if self._history_sways else 0.0
            ),
            "max_sway": (
                float(np.max(self._history_sways)) if self._history_sways else 0.0
            ),
            "avg_jerk": (
                float(np.mean(self._history_jerks)) if self._history_jerks else 0.0
            ),
            "max_jerk": (
                float(np.max(self._history_jerks)) if self._history_jerks else 0.0
            ),
            "avg_body_angle": (
                float(np.mean(self._history_body_angles))
                if self._history_body_angles
                else 0.0
            ),
            "avg_hand_span": (
                float(np.mean(self._history_hand_spans))
                if self._history_hand_spans
                else 0.0
            ),
            "avg_foot_span": (
                float(np.mean(self._history_foot_spans))
                if self._history_foot_spans
                else 0.0
            ),
            "total_vertical_progress": (
                self.initial_hip_height - self._history_hip_heights[-1]
                if self._history_hip_heights
                else 0.0
            ),
            "max_height": (
                self.initial_hip_height - min(self._history_hip_heights)
                if self._history_hip_heights
                else 0.0
            ),
            "min_height": (
                self.initial_hip_height - max(self._history_hip_heights)
                if self._history_hip_heights
                else 0.0
            ),
            "total_distance_traveled": self._total_distance_traveled,
            "avg_movement_economy": (
                float(np.mean(self._history_movement_economy))
                if self._history_movement_economy
                else 0.0
            ),
            "lock_off_count": sum(self._history_lock_offs),
            "lock_off_percentage": (
                100.0 * sum(self._history_lock_offs) / len(self._history_lock_offs)
                if self._history_lock_offs
                else 0.0
            ),
            "rest_count": sum(self._history_rest_positions),
            "rest_percentage": (
                100.0
                * sum(self._history_rest_positions)
                / len(self._history_rest_positions)
                if self._history_rest_positions
                else 0.0
            ),
            "fatigue_score": self._calculate_fatigue_score(),
        }

        # Add average joint angles
        for joint_name, angles in self._history_joint_angles.items():
            if angles:
                summary[f"avg_{joint_name}"] = float(np.mean(angles))

        # Return dictionary for backward compatibility
        # TODO: In next major version, return ClimbingSummary directly
        return summary

    def get_summary_typed(self) -> ClimbingSummary:
        """Get typed summary statistics.

        Returns immutable ClimbingSummary object for type-safe code.

        Returns:
            ClimbingSummary object with all statistics
        """
        summary_dict = self.get_summary()
        return ClimbingSummary(
            total_frames=summary_dict["total_frames"],
            total_vertical_progress=summary_dict["total_vertical_progress"],
            max_height=summary_dict["max_height"],
            avg_velocity=summary_dict["avg_velocity"],
            max_velocity=summary_dict["max_velocity"],
            avg_sway=summary_dict["avg_sway"],
            max_sway=summary_dict["max_sway"],
            avg_jerk=summary_dict["avg_jerk"],
            max_jerk=summary_dict["max_jerk"],
            avg_body_angle=summary_dict["avg_body_angle"],
            avg_hand_span=summary_dict["avg_hand_span"],
            avg_foot_span=summary_dict["avg_foot_span"],
            total_distance_traveled=summary_dict["total_distance_traveled"],
            avg_movement_economy=summary_dict["avg_movement_economy"],
            lock_off_count=summary_dict["lock_off_count"],
            lock_off_percentage=summary_dict["lock_off_percentage"],
            rest_count=summary_dict["rest_count"],
            rest_percentage=summary_dict["rest_percentage"],
            fatigue_score=summary_dict["fatigue_score"],
            avg_left_elbow=summary_dict["avg_left_elbow"],
            avg_right_elbow=summary_dict["avg_right_elbow"],
            avg_left_shoulder=summary_dict["avg_left_shoulder"],
            avg_right_shoulder=summary_dict["avg_right_shoulder"],
            avg_left_knee=summary_dict["avg_left_knee"],
            avg_right_knee=summary_dict["avg_right_knee"],
            avg_left_hip=summary_dict["avg_left_hip"],
            avg_right_hip=summary_dict["avg_right_hip"],
        )

    def _calculate_fatigue_score(self) -> float:
        """Calculate fatigue score based on quality degradation.

        Compares movement quality (jerk and sway) in first third vs last third.
        Higher score = more fatigued (0.0 = no change, 1.0 = significant degradation)

        Returns:
            Fatigue score (0.0-1.0+)
        """
        if len(self._history_jerks) < MetricsConfig.FATIGUE_WINDOW_SIZE:
            return 0.0

        # Split into first third and last third
        third = len(self._history_jerks) // 3
        if third < 10:  # Need enough data
            return 0.0

        early_jerks = self._history_jerks[:third]
        late_jerks = self._history_jerks[-third:]

        early_sways = self._history_sways[:third]
        late_sways = self._history_sways[-third:]

        # Calculate average values
        early_jerk_avg = np.mean(early_jerks) if early_jerks else 0.0
        late_jerk_avg = np.mean(late_jerks) if late_jerks else 0.0

        early_sway_avg = np.mean(early_sways) if early_sways else 0.0
        late_sway_avg = np.mean(late_sways) if late_sways else 0.0

        # Calculate degradation (normalized to 0-1 range)
        jerk_degradation = 0.0
        if early_jerk_avg > 0:
            jerk_degradation = (late_jerk_avg - early_jerk_avg) / early_jerk_avg

        sway_degradation = 0.0
        if early_sway_avg > 0:
            sway_degradation = (late_sway_avg - early_sway_avg) / early_sway_avg

        # Combined score (average of both indicators)
        fatigue_score = max(0.0, (jerk_degradation + sway_degradation) / 2.0)

        return float(fatigue_score)

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
            - movement_economy: Movement economy at each frame
            - lock_offs: Lock-off detection at each frame (boolean)
            - rest_positions: Rest position detection at each frame (boolean)
            - joint_angles: Dictionary of joint angle histories
        """
        history = {
            "hip_heights": self._history_hip_heights.copy(),
            "velocities": self._history_velocities.copy(),
            "sways": self._history_sways.copy(),
            "jerks": self._history_jerks.copy(),
            "body_angles": self._history_body_angles.copy(),
            "hand_spans": self._history_hand_spans.copy(),
            "foot_spans": self._history_foot_spans.copy(),
            "movement_economy": self._history_movement_economy.copy(),
            "lock_offs": self._history_lock_offs.copy(),
            "rest_positions": self._history_rest_positions.copy(),
        }

        # Add joint angle histories
        for joint_name, angles in self._history_joint_angles.items():
            history[joint_name] = angles.copy()

        return history

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
        self._history_movement_economy.clear()
        self._history_lock_offs.clear()
        self._history_rest_positions.clear()

        # Reset joint angle histories
        for joint_name in self._history_joint_angles:
            self._history_joint_angles[joint_name].clear()

        self.total_frames = 0
        self.initial_hip_height = None
        self._total_distance_traveled = 0.0


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
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            velocities.append((dx / dt, dy / dt))

        # Calculate accelerations
        accelerations = []
        for i in range(1, len(velocities)):
            ax = (velocities[i][0] - velocities[i - 1][0]) / dt
            ay = (velocities[i][1] - velocities[i - 1][1]) / dt
            accelerations.append((ax, ay))

        # Calculate jerk (derivative of acceleration)
        jerks = []
        for i in range(1, len(accelerations)):
            jx = (accelerations[i][0] - accelerations[i - 1][0]) / dt
            jy = (accelerations[i][1] - accelerations[i - 1][1]) / dt
            jerk_magnitude = np.sqrt(jx**2 + jy**2)
            jerks.append(jerk_magnitude)

        return float(np.mean(jerks)) if jerks else 0.0

    @staticmethod
    def calculate_base_of_support(
        landmarks: List[Dict[str, float]],
    ) -> Dict[str, float]:
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
        left_hand = (
            landmarks[LandmarkIndex.LEFT_WRIST]["x"],
            landmarks[LandmarkIndex.LEFT_WRIST]["y"],
        )
        right_hand = (
            landmarks[LandmarkIndex.RIGHT_WRIST]["x"],
            landmarks[LandmarkIndex.RIGHT_WRIST]["y"],
        )
        hand_span = np.sqrt(
            (right_hand[0] - left_hand[0]) ** 2 + (right_hand[1] - left_hand[1]) ** 2
        )

        # Foot positions
        left_foot = (
            landmarks[LandmarkIndex.LEFT_ANKLE]["x"],
            landmarks[LandmarkIndex.LEFT_ANKLE]["y"],
        )
        right_foot = (
            landmarks[LandmarkIndex.RIGHT_ANKLE]["x"],
            landmarks[LandmarkIndex.RIGHT_ANKLE]["y"],
        )
        foot_span = np.sqrt(
            (right_foot[0] - left_foot[0]) ** 2 + (right_foot[1] - left_foot[1]) ** 2
        )

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

        Measures the deviation from a perfectly vertical torso alignment.
        In climbing, 0° = vertical (shoulders directly above hips),
        positive values indicate lean.

        Args:
            landmarks: List of landmark dictionaries

        Returns:
            Angle in degrees (0 = vertical, up to 90 = horizontal lean)
        """
        if len(landmarks) < 33:
            return 0.0

        # Use shoulder and hip midpoints
        shoulder_x = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["x"]
            + landmarks[LandmarkIndex.RIGHT_SHOULDER]["x"]
        ) / 2
        shoulder_y = (
            landmarks[LandmarkIndex.LEFT_SHOULDER]["y"]
            + landmarks[LandmarkIndex.RIGHT_SHOULDER]["y"]
        ) / 2

        hip_x = (
            landmarks[LandmarkIndex.LEFT_HIP]["x"]
            + landmarks[LandmarkIndex.RIGHT_HIP]["x"]
        ) / 2
        hip_y = (
            landmarks[LandmarkIndex.LEFT_HIP]["y"]
            + landmarks[LandmarkIndex.RIGHT_HIP]["y"]
        ) / 2

        # Calculate horizontal and vertical distances
        dx = abs(shoulder_x - hip_x)  # Horizontal distance (lean)
        dy = abs(
            hip_y - shoulder_y
        )  # Vertical distance (in image coords, y increases down)

        # Handle edge case where shoulders and hips are at same position
        if dy < 1e-6:
            return 90.0  # Horizontal body

        # Calculate angle from vertical using arctan
        # arctan(dx/dy) gives angle from vertical axis
        angle = np.degrees(np.arctan(dx / dy))
        return float(angle)
