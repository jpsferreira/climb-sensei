# Usage Guide

## Analyzing Climbing Performance

### ClimbingAnalyzer

The `ClimbingAnalyzer` class is the main interface for analyzing climbing metrics:

```python
from climb_sensei import ClimbingAnalyzer

# Initialize with custom parameters
analyzer = ClimbingAnalyzer(
    window_size=30,  # Frames for moving average (1 second at 30fps)
    fps=30           # Video frame rate
)

# Analyze each frame
metrics = analyzer.analyze_frame(landmarks)
```

### Available Metrics

Each call to `analyze_frame()` returns a dictionary with:

#### Core Movement Metrics

- **`hip_height`**: Current hip position (vertical coordinate)
- **`com_velocity`**: Movement speed (units/second)
- **`com_sway`**: Lateral stability (lower = more stable)
- **`jerk`**: Movement smoothness (lower = smoother)
- **`body_angle`**: Lean from vertical (0° = vertical, 90° = horizontal)
- **`hand_span`**: Distance between hands
- **`foot_span`**: Distance between feet
- **`vertical_progress`**: Height gained from start

#### Efficiency & Technique

- **`movement_economy`**: Vertical progress / total distance (higher = more efficient)
- **`is_lock_off`**: Static bent-arm position detected (boolean)
- **`left_lock_off`**: Left arm lock-off detection (boolean)
- **`right_lock_off`**: Right arm lock-off detection (boolean)
- **`is_rest_position`**: Low-stress vertical position detected (boolean)

#### Joint Angles (8 joints)

- **`left_elbow`**, **`right_elbow`**: Elbow flexion angles
- **`left_shoulder`**, **`right_shoulder`**: Shoulder angles
- **`left_knee`**, **`right_knee`**: Knee flexion angles
- **`left_hip`**, **`right_hip`**: Hip angles

### Getting Historical Data

```python
# Get complete time-series history
history = analyzer.get_history()

# Access specific metric histories
velocities = history['velocities']
sways = history['sways']
body_angles = history['body_angles']
joint_angles = history['joint_angles']  # Dict of all 8 joints
```

### Summary Statistics

```python
# Get summary with all aggregated metrics
summary = analyzer.get_summary()

# Available summary stats:
# - avg_velocity: Average movement speed
# - total_vertical_progress: Total height gained
# - avg_movement_economy: Average efficiency
# - lock_off_count: Number of lock-off moments
# - lock_off_percentage: Percentage of time in lock-off
# - rest_count: Number of rest positions
# - rest_percentage: Percentage of time resting
# - fatigue_score: Quality degradation indicator
# - avg_joint_angles: Average angle for each joint
```

## Video Processing

### Reading Videos

```python
from climb_sensei import VideoReader

with VideoReader('input.mp4') as video:
    # Get video properties
    print(f"FPS: {video.fps}")
    print(f"Size: {video.width}x{video.height}")
    print(f"Total frames: {video.frame_count}")

    # Read frames
    while True:
        success, frame = video.read()
        if not success:
            break
        # Process frame...
```

### Writing Videos

```python
from climb_sensei import VideoWriter

with VideoWriter('output.mp4', fps=30, width=1920, height=1080) as writer:
    for frame in frames:
        writer.write(frame)
```

## Pose Detection

### PoseEngine

```python
from climb_sensei import PoseEngine

# Initialize with custom confidence thresholds
engine = PoseEngine(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process frames
results = engine.process(frame)

# Extract landmarks (33 pose landmarks)
if results:
    landmarks = engine.extract_landmarks(results)
    # landmarks is a list of (x, y) tuples

# Close when done (or use context manager)
engine.close()
```

### Context Manager

```python
with PoseEngine() as engine:
    results = engine.process(frame)
    # Engine automatically closes
```

## Visualization

### Drawing Pose Landmarks

```python
from climb_sensei.viz import draw_pose_landmarks

# Draw pose on frame
annotated_frame = draw_pose_landmarks(frame, landmarks)
```

### Creating Dashboards

```python
from climb_sensei.metrics_viz import (
    create_metrics_dashboard,
    compose_frame_with_dashboard
)

# Create dashboard with plots
dashboard = create_metrics_dashboard(history, width=800, height=1920)

# Compose side-by-side (default)
output = compose_frame_with_dashboard(frame, dashboard, position='right')

# Or use overlay mode
from climb_sensei.metrics_viz import overlay_metrics_on_frame
output = overlay_metrics_on_frame(frame, dashboard, alpha=0.7)
```

## Biomechanics Functions

Low-level mathematical functions for custom calculations:

```python
from climb_sensei.biomechanics import (
    calculate_joint_angle,
    calculate_reach_distance,
    calculate_center_of_mass,
    calculate_limb_angles,
    calculate_total_distance_traveled
)

# Joint angle at point B formed by A-B-C
angle = calculate_joint_angle(point_a, point_b, point_c)

# Euclidean distance
distance = calculate_reach_distance(point_a, point_b)

# Weighted center of mass
center = calculate_center_of_mass(points, weights)

# Calculate all 8 joint angles at once
angles = calculate_limb_angles(landmarks)
# Returns dict: {
#   'left_elbow': 145.2,
#   'right_elbow': 148.3,
#   'left_shoulder': 92.1,
#   ...
# }

# Total distance traveled
distance = calculate_total_distance_traveled(com_positions)
```

## Complete Pipeline Example

```python
from climb_sensei import (
    PoseEngine, VideoReader, VideoWriter,
    ClimbingAnalyzer, draw_pose_landmarks
)
from climb_sensei.metrics_viz import compose_frame_with_dashboard

# Initialize
analyzer = ClimbingAnalyzer(window_size=30, fps=30)

with PoseEngine() as engine:
    with VideoReader('input.mp4') as reader:
        with VideoWriter('output.mp4', fps=reader.fps,
                        width=reader.width + 800,
                        height=reader.height) as writer:

            while True:
                success, frame = reader.read()
                if not success:
                    break

                # Detect and analyze
                results = engine.process(frame)
                if results:
                    landmarks = engine.extract_landmarks(results)
                    metrics = analyzer.analyze_frame(landmarks)

                    # Visualize
                    annotated = draw_pose_landmarks(frame, landmarks)
                    history = analyzer.get_history()
                    dashboard = create_metrics_dashboard(history, 800, frame.shape[0])
                    output = compose_frame_with_dashboard(annotated, dashboard)

                    writer.write(output)

# Export data
import json
summary = analyzer.get_summary()
with open('analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)
```
