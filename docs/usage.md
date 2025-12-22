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

## Video Quality Validation

Before processing videos, especially in a backend API service, validate video quality to ensure successful analysis.

### Quick Validation

```python
from climb_sensei import check_video_quality

# Basic quality check
report = check_video_quality('uploaded_video.mp4')

if report.is_valid:
    print("✓ Video is ready for processing")
else:
    print("✗ Video has issues:")
    for issue in report.issues:
        print(f"  - {issue}")
    for warning in report.warnings:
        print(f"  ⚠ {warning}")
```

### Deep Quality Analysis

```python
# Include lighting and stability checks
report = check_video_quality('video.mp4', deep_check=True)

# Check specific quality aspects
print(f"Format: {report.format_compatible}")
print(f"Resolution: {report.resolution_quality}")  # 'poor', 'acceptable', 'good', 'excellent'
print(f"FPS: {report.fps_quality}")
print(f"Duration: {report.duration_quality}")

# Deep analysis results (if enabled)
if report.lighting_quality:
    print(f"Lighting: {report.lighting_quality}")  # 'dark', 'acceptable', 'good'
if report.stability_quality:
    print(f"Stability: {report.stability_quality}")  # 'shaky', 'acceptable', 'stable'
```

### Using VideoQualityChecker

```python
from climb_sensei import VideoQualityChecker

# Initialize with custom thresholds
checker = VideoQualityChecker(
    min_resolution=(640, 480),       # Minimum width, height
    recommended_resolution=(1280, 720),
    optimal_resolution=(1920, 1080),
    min_fps=15,
    recommended_fps=30,
    optimal_fps=60,
    min_duration=5.0,                # Seconds
    max_duration=600.0,
    optimal_min_duration=10.0,
    optimal_max_duration=180.0
)

# Check video
report = checker.check_video('video.mp4', deep_check=True)

# Access detailed properties
print(f"Width: {report.width}px, Height: {report.height}px")
print(f"FPS: {report.fps}")
print(f"Duration: {report.duration:.2f}s")
print(f"Codec: {report.codec}")
```

### Backend API Integration

```python
from climb_sensei import check_video_quality
import json

def validate_uploaded_video(filepath):
    """
    Validate video before processing in backend service.
    Returns (valid, response_data).
    """
    report = check_video_quality(filepath, deep_check=True)

    if report.is_valid:
        return True, {
            'status': 'valid',
            'properties': {
                'width': report.width,
                'height': report.height,
                'fps': report.fps,
                'duration': report.duration,
                'codec': report.codec
            }
        }
    else:
        return False, {
            'status': 'invalid',
            'errors': report.issues,
            'warnings': report.warnings,
            'quality': {
                'resolution': report.resolution_quality,
                'fps': report.fps_quality,
                'duration': report.duration_quality,
                'lighting': report.lighting_quality,
                'stability': report.stability_quality
            }
        }

# Usage in API endpoint
is_valid, response = validate_uploaded_video('/tmp/upload.mp4')
if not is_valid:
    return json.dumps(response), 400  # Bad Request
```

### CLI Tool

Use the included script for command-line validation:

```bash
# Basic check
python scripts/check_video_quality.py video.mp4

# Deep check with lighting and stability analysis
python scripts/check_video_quality.py video.mp4 --deep

# Export to JSON
python scripts/check_video_quality.py video.mp4 --json report.json

# Quiet mode (only return exit code)
python scripts/check_video_quality.py video.mp4 --quiet
# Exit code: 0 = valid, 1 = invalid, 2 = error
```

## Tracking Quality Analysis

Analyze the quality of pose/skeleton tracking to ensure reliable results. Can be used with videos or with already-extracted landmarks.

### From Video File

```python
from climb_sensei import analyze_tracking_quality

# Analyze tracking quality from video
report = analyze_tracking_quality('climbing.mp4', sample_rate=5)

if report.is_trackable:
    print(f"✓ Detection rate: {report.detection_rate}%")
    print(f"  Smoothness: {report.tracking_smoothness:.3f}")
    print(f"  Quality: {report.quality_level}")
else:
    print("✗ Poor tracking quality:")
    for issue in report.issues:
        print(f"  - {issue}")
```

### From Landmarks (More Efficient)

When landmarks are already extracted (e.g., during `analyze_climb`), analyze tracking quality without re-running pose detection:

```python
from climb_sensei import (
    PoseEngine,
    VideoReader,
    ClimbingAnalyzer,
    analyze_tracking_from_landmarks,
)

analyzer = ClimbingAnalyzer(window_size=30, fps=30)
landmarks_history = []

with PoseEngine() as engine:
    with VideoReader('climbing.mp4') as reader:
        while True:
            success, frame = reader.read()
            if not success:
                break

            results = engine.process(frame)
            if results:
                landmarks = engine.extract_landmarks(results)
                if landmarks:
                    analyzer.analyze_frame(landmarks)
                    landmarks_history.append(landmarks)
                else:
                    landmarks_history.append(None)
            else:
                landmarks_history.append(None)

# Get climbing metrics
climbing_summary = analyzer.get_summary()

# Analyze tracking quality from landmarks we already have
tracking_report = analyze_tracking_from_landmarks(landmarks_history)

print(f"Detection rate: {tracking_report.detection_rate}%")
print(f"Smoothness: {tracking_report.tracking_smoothness:.3f}")
print(f"Tracking losses: {tracking_report.tracking_loss_events}")
```

### Using TrackingQualityAnalyzer

```python
from climb_sensei import TrackingQualityAnalyzer

# Initialize with custom thresholds
analyzer = TrackingQualityAnalyzer(
    min_detection_rate=80.0,      # Minimum % of frames with detection
    min_avg_confidence=0.6,        # Minimum average landmark confidence
    min_visibility=70.0,           # Minimum % of visible landmarks
    min_smoothness=0.7,            # Minimum smoothness score
    max_tracking_losses=3,         # Maximum acceptable tracking loss events
    sample_rate=5,                 # Analyze every 5th frame
)

# Analyze from video
report = analyzer.analyze_video('video.mp4')

# Or analyze from landmarks
report = analyzer.analyze_from_landmarks(landmarks_sequence)

# Check quality
print(f"Quality level: {report.quality_level}")  # poor/acceptable/good/excellent
print(f"Is trackable: {report.is_trackable}")
```

### Tracking Quality Metrics

The report includes:

- **`detection_rate`**: Percentage of frames with pose detected
- **`avg_landmark_confidence`**: Average confidence across all landmarks (0-1)
- **`avg_visibility_score`**: Average percentage of visible landmarks
- **`tracking_smoothness`**: Smoothness based on landmark jitter (0-1, higher = smoother)
- **`tracking_loss_events`**: Number of times tracking was lost then regained
- **`quality_level`**: Overall quality ('poor', 'acceptable', 'good', 'excellent')
- **`is_trackable`**: Whether video has sufficient tracking quality
- **`issues`**: List of critical tracking problems
- **`warnings`**: List of quality concerns

### CLI Tool with Tracking

```bash
# Video quality + tracking quality analysis
python scripts/check_video_quality.py video.mp4 --tracking

# With custom sample rate
python scripts/check_video_quality.py video.mp4 --tracking --sample-rate 10

# Complete analysis with JSON export
python scripts/check_video_quality.py video.mp4 --deep --tracking --json quality.json
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
