# climb-sensei

A Python pose estimation tool for analyzing climbing footage. Extract vertical movement metrics, calculate joint angles, and visualize biomechanical data using computer vision.

## Features

- 🎯 **Pose Detection**: Real-time human pose estimation using MediaPipe
- 📐 **Biomechanics**: Calculate joint angles, reach distances, and center of mass
- 📹 **Video I/O**: Easy video processing with OpenCV
- 🎨 **Visualization**: Draw pose landmarks and annotate metrics

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

### Running the Demo

```bash
# Run the interactive demo
uv run python -m climb_sensei
```

### Basic Usage

```python
from climb_sensei import PoseEngine, VideoReader, calculate_joint_angle

# Process a video file
with PoseEngine() as engine:
    with VideoReader('climbing_video.mp4') as video:
        success, frame = video.read()
        if success:
            # Detect pose
            results = engine.process(frame)
            if results:
                # Extract landmarks
                landmarks = engine.extract_landmarks(results)
                
                # Calculate elbow angle (example)
                shoulder = (landmarks[12]["x"], landmarks[12]["y"])
                elbow = (landmarks[14]["x"], landmarks[14]["y"])
                wrist = (landmarks[16]["x"], landmarks[16]["y"])
                
                angle = calculate_joint_angle(shoulder, elbow, wrist)
                print(f"Elbow angle: {angle:.1f}°")
```

### Biomechanics Calculations

```python
from climb_sensei import calculate_joint_angle, calculate_reach_distance

# Calculate joint angle (in degrees)
shoulder = (0.4, 0.3)
elbow = (0.5, 0.5)
wrist = (0.6, 0.6)
angle = calculate_joint_angle(shoulder, elbow, wrist)

# Calculate reach distance
hip = (0.5, 0.6)
hand = (0.7, 0.2)
reach = calculate_reach_distance(hip, hand)
```

### Visualization

```python
from climb_sensei import draw_pose_landmarks, draw_angle_annotation

# Draw pose skeleton
annotated_frame = draw_pose_landmarks(frame, results)

# Add angle annotation
annotated_frame = draw_angle_annotation(
    annotated_frame, 
    point=(x, y), 
    angle=elbow_angle
)
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
