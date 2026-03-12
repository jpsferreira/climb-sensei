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
- ✅ **Video Quality Validation**: Pre-processing quality checks for resolution, FPS, duration, lighting, and stability
- 🧪 **Well-Tested**: Comprehensive test suite with high code coverage

## Quick Example

```python
from climb_sensei import PoseEngine, VideoReader, ClimbingAnalysis

# Analyze climbing performance
analyzer = ClimbingAnalysis(window_size=30, fps=30)

with PoseEngine() as engine:
    with VideoReader('climbing_video.mp4') as video:
        while True:
            success, frame = video.read()
            if not success:
                break

            # Detect pose and analyze
            results = engine.process(frame)
            if results:
                landmarks = engine.extract_landmarks(results)
                metrics = analyzer.analyze_frame(landmarks)

                print(f"Velocity: {metrics['com_velocity']:.4f}")
                print(f"Stability: {metrics['com_sway']:.4f}")

# Get summary statistics
summary = analyzer.get_summary()
print(f"Average speed: {summary['avg_velocity']:.4f}")
print(f"Total vertical progress: {summary['total_vertical_progress']:.3f}")
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
│   ├── metrics.py            # ClimbingAnalysis with temporal tracking
│   ├── metrics_viz.py        # Metrics dashboard visualization
│   └── viz.py                # Pose visualization utilities
├── scripts/
│   └── analyze_climb.py      # Unified CLI (analysis + video generation)
├── tests/
│   ├── test_*.py             # Comprehensive unit tests
│   └── __init__.py
└── docs/                     # Documentation
```

## Requirements

- Python 3.12+
- mediapipe >= 0.10.30
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- tqdm >= 4.66.0

## License

See [LICENSE](https://github.com/jpsferreira/climb-sensei/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:

- All functions have type hints
- Comprehensive docstrings
- Unit tests for new functionality
- Code follows existing style conventions
