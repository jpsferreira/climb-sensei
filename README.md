# climb-sensei

A Python pose estimation tool for analyzing climbing footage. Extract vertical movement metrics, calculate biomechanics, visualize technique with animated dashboards, and analyze climbing performance using computer vision.

```markdown
[![CI](https://github.com/jpsferreira/climb-sensei/actions/workflows/ci.yaml/badge.svg)](https://github.com/jpsferreira/climb-sensei/actions/workflows/ci.yaml)
[![Pre-commit](https://github.com/jpsferreira/climb-sensei/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/jpsferreira/climb-sensei/actions/workflows/pre-commit.yaml)
[![codecov](https://codecov.io/gh/jpsferreira/climb-sensei/branch/main/graph/badge.svg)](https://codecov.io/gh/jpsferreira/climb-sensei)
```

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
# Quick analysis with terminal summary (fast)
python scripts/analyze_climb.py climbing_video.mp4

# Export detailed JSON data
python scripts/analyze_climb.py climbing_video.mp4 --json analysis.json

# Create annotated video with metrics dashboard on the side (default: no overlay)
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4

# Customize dashboard position (left or right)
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4 --position left

# Use overlay mode instead of side-by-side
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4 --overlay

# Add text overlay with current metric values
python scripts/analyze_climb.py climbing_video.mp4 --video output.mp4 --show-text

# Export both JSON and video in one pass
python scripts/analyze_climb.py climbing_video.mp4 --json data.json --video output.mp4
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
#          movement_economy, lock_offs, rest_positions, joint_angles (8 joints)

# Get summary statistics with all metrics
summary = analyzer.get_summary()
# Includes: avg_velocity, total_vertical_progress, avg_movement_economy,
#           lock_off_count, lock_off_percentage, rest_count, rest_percentage,
#           fatigue_score, and average joint angles
```

## Project Structure

```
climb-sensei/
├── src/climb_sensei/
│   ├── __init__.py           # Package exports
│   ├── __main__.py           # Demo application
│   ├── config.py             # Configuration and constants
│   ├── video_io.py           # Video input/output handling
│   ├── pose_engine.py        # MediaPipe pose estimation
│   ├── biomechanics.py       # Pure mathematical calculations
│   ├── metrics.py            # ClimbingAnalyzer with temporal tracking
│   ├── metrics_viz.py        # Metrics dashboard visualization
│   └── viz.py                # Pose visualization utilities
├── scripts/
│   └── analyze_climb.py      # Unified CLI (analysis + video generation)
├── tests/
│   ├── test_*.py             # 107 comprehensive unit tests
│   └── __init__.py
├── METRICS_REFERENCE.md      # Complete metrics documentation
└── pyproject.toml            # Project configuration
```

## Architecture

The package follows strict **Separation of Concerns**:

- **config**: Application-wide configuration and constants
- **video_io**: Handles all video I/O operations (no business logic)
- **pose_engine**: Wraps MediaPipe pose detection (single responsibility)
- **biomechanics**: Pure mathematical functions (stateless, testable)
- **metrics**: Temporal analysis with ClimbingAnalyzer (OOP, stateful tracking)
- **metrics_viz**: Dashboard visualization (plots and overlays)
- **viz**: Pose rendering and annotation utilities (presentation layer)

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
    calculate_center_of_mass,
    calculate_limb_angles,
    calculate_total_distance_traveled
)

# Joint angle at point B formed by A-B-C
angle = calculate_joint_angle(point_a, point_b, point_c)

# Euclidean distance between two points
distance = calculate_reach_distance(point_a, point_b)

# Weighted center of mass
center = calculate_center_of_mass(points, weights)

# Calculate all 8 joint angles at once
angles = calculate_limb_angles(landmarks)
# Returns: left_elbow, right_elbow, left_shoulder, right_shoulder,
#          left_knee, right_knee, left_hip, right_hip

# Total distance traveled by center of mass
distance = calculate_total_distance_traveled(com_positions)
```

## Metrics Documentation

For complete documentation of all 25+ available metrics, see [METRICS_REFERENCE.md](METRICS_REFERENCE.md).

Key metric categories:

- **Core Movement**: Velocity, sway, jerk, body angle, spans, vertical progress
- **Efficiency & Technique**: Movement economy, lock-off detection, rest positions
- **Joint Angles**: All 8 major joints (elbows, shoulders, knees, hips)
- **Fatigue & Endurance**: Quality degradation scoring

## Requirements

- Python 3.13+ (tested on 3.13.5)
- mediapipe >= 0.10.30
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pytest >= 9.0.0 (for testing)
- pytest-cov >= 7.0.0 (for coverage)

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
