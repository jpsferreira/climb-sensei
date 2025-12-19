# climb-sensei

A Python pose estimation tool for analyzing climbing footage. Extract vertical movement metrics, calculate biomechanics, visualize technique with animated dashboards, and analyze climbing performance using computer vision.

## Features

- 🎯 **Pose Detection**: Real-time human pose estimation using MediaPipe with temporal smoothing
- 📊 **Performance Analysis**: Comprehensive climbing metrics including speed, stability, smoothness, and body positioning
- 🎯 **Efficiency Metrics**: Movement economy, lock-off detection, rest position identification, and fatigue scoring
- 📐 **Biomechanics**: Calculate joint angles (8 joints), reach distances, and center of mass
- 📹 **Video Processing**: Easy video I/O with pose overlay and animated metrics dashboards
- 🎨 **Visualization**: Draw pose landmarks, annotate metrics, and create real-time performance graphs
- 📈 **Temporal Analysis**: Track metrics over time with jerk calculation, sway detection, and progression tracking
- ✅ **Well-Tested**: 77% code coverage with 107 unit tests

## Installation

This project uses `uv` for fast, reliable package management:

```bash
# Install uv (if not already installed)
pip install uv

# Install the package
uv sync

# Or install with development dependencies
uv sync --extra dev
```

## Quick Start

### Analyze a Climbing Video

```bash
# Quick analysis with summary statistics
python scripts/analyze_climb.py climbing_video.mp4

# Save detailed analysis to JSON
python scripts/analyze_climb.py climbing_video.mp4 --output analysis.json
```

### Process Video with Animated Metrics Dashboard

```bash
# Add animated metrics dashboard to video (6 real-time plots)
python scripts/process_video_with_metrics.py input.mp4 output.mp4

# Position dashboard on left or bottom
python scripts/process_video_with_metrics.py input.mp4 output.mp4 --position left
```

### Running the Demo

```bash
# Run the interactive demo
uv run python -m climb_sensei
```

### Basic Usage

```python
from climb_sensei import PoseEngine, VideoReader, ClimbingAnalyzer

# Analyze climbing performance
analyzer = ClimbingAnalyzer(window_size=30, fps=30)

with PoseEngine() as engine:
    with VideoReader('climbing_video.mp4') as video:
        while True:
            success, frame = video.read()
            if not success:
                break
                
            # Detect pose
            results = engine.process(frame)
            if results:
                # Extract landmarks
                landmarks = engine.extract_landmarks(results)
                
                # Analyze frame - get all metrics
                metrics = analyzer.analyze_frame(landmarks)
                print(f"Velocity: {metrics['com_velocity']:.4f}")
                print(f"Stability: {metrics['com_sway']:.4f}")
                print(f"Progress: {metrics['vertical_progress']:.3f}")

# Get summary statistics
summary = analyzer.get_summary()
print(f"Average speed: {summary['avg_velocity']:.4f}")
print(f"Total vertical progress: {summary['total_vertical_progress']:.3f}")
```

### Climbing Metrics

```python
from climb_sensei import ClimbingAnalyzer

analyzer = ClimbingAnalyzer(window_size=30, fps=30)

# Analyze each frame
metrics = analyzer.analyze_frame(landmarks)

# Core Movement Metrics:
# - hip_height: Current hip position
# - com_velocity: Movement speed
# - com_sway: Lateral stability (lower = more stable)
# - jerk: Movement smoothness (lower = smoother)
# - body_angle: Lean from vertical
# - hand_span: Distance between hands
# - foot_span: Distance between feet
# - vertical_progress: Height gained from start

# Efficiency & Technique:
# - movement_economy: Vertical progress / total distance (higher = more efficient)
# - is_lock_off: Static bent-arm position detected (boolean)
# - left_lock_off, right_lock_off: Per-arm lock-off detection
# - is_rest_position: Low-stress vertical position (boolean)

# Joint Angles (8 joints):
# - left_elbow, right_elbow: Elbow flexion angles
# - left_shoulder, right_shoulder: Shoulder angles
# - left_knee, right_knee: Knee flexion angles
# - left_hip, right_hip: Hip angles

# Get complete time-series history
history = analyzer.get_history()
# Returns: hip_heights, velocities, sways, jerks, body_angles, hand_spans, foot_spans,
#          movement_economies, lock_offs, rest_positions, joint_angles (8 joints)
```

## Project Structure

```
climb-sensei/
├── src/climb_sensei/
│   ├── __init__.py           # Package exports
│   ├── __main__.py           # Demo application
│   ├── video_io.py           # Video input/output handling
│   ├── pose_engine.py        # MediaPipe pose estimation
│   ├── biomechanics.py       # Pure mathematical calculations
│   └── viz.py                # Visualization utilities
├── tests/
│   └── test_biomechanics.py  # Unit tests
└── pyproject.toml            # Project configuration
```

## Architecture

The package follows strict **Separation of Concerns**:

- **video_io**: Handles all video I/O operations (no business logic)
- **pose_engine**: Wraps MediaPipe pose detection (single responsibility)
- **biomechanics**: Pure mathematical functions (stateless, testable)
- **viz**: Rendering and annotation utilities (presentation layer)

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=climb_sensei
```

## API Reference

### PoseEngine

```python
from climb_sensei import PoseEngine

engine = PoseEngine(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process an image
results = engine.process(image)

# Extract landmarks
landmarks = engine.extract_landmarks(results)

# Close when done
engine.close()
```

### VideoReader & VideoWriter

```python
from climb_sensei import VideoReader, VideoWriter

# Read video
with VideoReader('input.mp4') as video:
    success, frame = video.read()
    print(f"FPS: {video.fps}, Size: {video.width}x{video.height}")

# Write video
with VideoWriter('output.mp4', fps=30, width=640, height=480) as writer:
    writer.write(frame)
```

### Biomechanics Functions

```python
from climb_sensei.biomechanics import (
    calculate_joint_angle,
    calculate_reach_distance,
    calculate_center_of_mass
)

# Joint angle at point B formed by A-B-C
angle = calculate_joint_angle(point_a, point_b, point_c)

# Euclidean distance between two points
distance = calculate_reach_distance(point_a, point_b)

# Weighted center of mass
center = calculate_center_of_mass(points, weights)
```

## Requirements

- Python 3.12+
- mediapipe >= 0.10.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0

## Development

Install development dependencies:

```bash
uv sync --extra dev
```

Run tests:

```bash
uv run pytest tests/ -v
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:
- All functions have type hints
- Comprehensive docstrings
- Unit tests for new functionality
- Code follows existing style conventions

## About

climb-sensei provides a maintainable toolkit for analyzing climbing movement using computer vision. The modular architecture makes it easy to extend and integrate into larger applications.
